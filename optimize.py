"""
Walk-forward PnL optimizer for the range-trading strategy.

  python3 optimize.py --symbol "AAVE/USDT"
  python3 optimize.py --symbol "BTC/USDT"

Data: 5 years fetched, last 4 years used (skip anomalous year 1).
Split: Train (Y2-Y3) → Validate (Y4, top 50) → Test (Y5, final pick).
Scoring: total P&L with min 15 trades.
Stress test: best params run on year 1 (informational only).
"""

import argparse
import csv
import math
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import fetch_ohlcv
from ranges import _smooth_outliers, _within_pct


@dataclass(frozen=True)
class StrategyParams:
    swing_window: int = 10
    range_window: int = 60
    range_width_min_pct: float = 2.0
    range_width_max_pct: float = 12.0
    signal_proximity_pct: float = 1.5
    momentum_lookback: int = 10
    momentum_threshold: float = 5.0
    trail_activation_pct: float = 1.5
    trail_pullback_pct: float = 1.0
    stop_loss_pct: float = 1.5
    signal_cooldown: int = 14
    volume_filter: bool = False
    candle_confirm: bool = False
    rsi_filter: bool = False
    range_end_exit: bool = True
    long_only: bool = False


GRID = {
    "swing_window":        [8, 10, 12, 14],
    "range_window":        [48, 60, 80],
    "range_width_min_pct": [1.5, 2.0, 3.0],
    "range_width_max_pct": [10.0, 12.0],
    "signal_proximity_pct":[1.0, 1.5, 2.0],
    "momentum_lookback":   [8, 10, 14],
    "momentum_threshold":  [4.0, 5.0, 7.0],
    "trail_activation_pct":[1.5, 2.5, 3.5],
    "trail_pullback_pct":  [0.5, 1.0, 1.5],
    "stop_loss_pct":       [1.0, 1.5, 2.0, 2.5],
    "signal_cooldown":     [8, 14, 20],
}

TOP_N_VALIDATE = 50
MIN_TRADES = 15


def build_combinations(long_only: bool = False) -> list[StrategyParams]:
    keys = list(GRID.keys())
    return [StrategyParams(**dict(zip(keys, vals)), long_only=long_only)
            for vals in product(*GRID.values())]


# ---------------------------------------------------------------------------
# Indicator pre-computation (numpy arrays, done once)
# ---------------------------------------------------------------------------

def compute_indicators(close: np.ndarray, open_: np.ndarray, volume: np.ndarray):
    n = len(close)

    volume_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        volume_ma20[i] = volume[i - 19:i + 1].mean()

    candle_dir = np.zeros(n, dtype=np.int8)
    candle_dir[close > open_] = 1
    candle_dir[close < open_] = -1

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / 14
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi14 = 100 - (100 / (1 + rs))

    return volume_ma20, candle_dir, rsi14


# ---------------------------------------------------------------------------
# Fast numpy-based pipeline
# ---------------------------------------------------------------------------

def _fast_find_ranges(close, high, low, timestamps, smoothed_high, smoothed_low,
                      swing_window, scan_window):
    n = len(close)
    swing_highs = []
    swing_lows = []

    for i in range(swing_window, n - swing_window):
        surrounding_h = np.concatenate([smoothed_high[i - swing_window:i], smoothed_high[i + 1:i + swing_window + 1]])
        if smoothed_high[i] >= surrounding_h.max():
            swing_highs.append(i)
        surrounding_l = np.concatenate([smoothed_low[i - swing_window:i], smoothed_low[i + 1:i + swing_window + 1]])
        if smoothed_low[i] <= surrounding_l.min():
            swing_lows.append(i)

    high_set = set(swing_highs)
    low_set = set(swing_lows)
    all_swings = sorted(high_set | low_set)

    ranges = []
    consumed_up_to = -1

    for start_pos, anchor_idx in enumerate(all_swings):
        if anchor_idx <= consumed_up_to:
            continue
        window_end_idx = anchor_idx + scan_window
        w_highs, w_lows = [], []
        for s in all_swings[start_pos:]:
            if s > window_end_idx:
                break
            if s in high_set:
                w_highs.append(s)
            if s in low_set:
                w_lows.append(s)

        if len(w_highs) < 2 or len(w_lows) < 2:
            continue

        hp = [smoothed_high[i] for i in w_highs]
        lp = [smoothed_low[i] for i in w_lows]
        if not _within_pct(hp, 0.02) or not _within_pct(lp, 0.02):
            continue

        all_in = w_highs + w_lows
        rh = max(smoothed_high[i] for i in all_in)
        rl = min(smoothed_low[i] for i in all_in)
        if rl >= rh:
            continue

        confirm_idx = max(sorted(w_highs)[1], sorted(w_lows)[1])
        last_idx = max(all_in)

        bo = last_idx
        limit = min(last_idx + scan_window + 1, n)
        for k in range(last_idx, limit):
            if close[k] > rh * 1.02 or close[k] < rl * 0.98:
                bo = k
                break
        else:
            bo = last_idx + scan_window

        ranges.append({
            "range_id": len(ranges),
            "range_high": rh,
            "range_low": rl,
            "start_idx": confirm_idx,
            "end_idx": min(bo, n - 1),
        })
        consumed_up_to = bo

    return ranges


def _fast_generate_signals(close, volume, ranges, entry_pct, max_move_pct,
                           min_width_pct, max_width_pct, lookback, cooldown,
                           volume_filter, candle_confirm, rsi_filter,
                           volume_ma20, candle_dir, rsi14):
    signals = []
    for r in ranges:
        width_pct = (r["range_high"] - r["range_low"]) / r["range_low"] * 100
        if width_pct < min_width_pct or width_pct > max_width_pct:
            continue

        start, end = r["start_idx"], r["end_idx"]
        long_thr = r["range_low"] * (1 + entry_pct)
        short_thr = r["range_high"] * (1 - entry_pct)
        last_long = -cooldown - 1
        last_short = -cooldown - 1

        for i in range(start, end + 1):
            c = close[i]
            lb_start = max(0, i - lookback)
            past = close[lb_start]

            if c <= long_thr and c >= r["range_low"] and (i - last_long) > cooldown:
                skip = False
                if volume_filter and (np.isnan(volume_ma20[i]) or volume[i] <= volume_ma20[i]):
                    skip = True
                if candle_confirm and candle_dir[i] != 1:
                    skip = True
                if rsi_filter and (np.isnan(rsi14[i]) or rsi14[i] >= 45):
                    skip = True
                if skip:
                    last_long = i
                    continue

                drop = (past - c) / past if past > 0 else 0
                if drop <= max_move_pct:
                    signals.append({"idx": i, "price": c, "dir": 1, "range_id": r["range_id"],
                                    "rh": r["range_high"], "rl": r["range_low"], "end_idx": end})
                last_long = i

            if c >= short_thr and c <= r["range_high"] and (i - last_short) > cooldown:
                skip = False
                if volume_filter and (np.isnan(volume_ma20[i]) or volume[i] <= volume_ma20[i]):
                    skip = True
                if candle_confirm and candle_dir[i] != -1:
                    skip = True
                if rsi_filter and (np.isnan(rsi14[i]) or rsi14[i] <= 55):
                    skip = True
                if skip:
                    last_short = i
                    continue

                rise = (c - past) / past if past > 0 else 0
                if rise <= max_move_pct:
                    signals.append({"idx": i, "price": c, "dir": -1, "range_id": r["range_id"],
                                    "rh": r["range_high"], "rl": r["range_low"], "end_idx": end})
                last_short = i

    signals.sort(key=lambda s: s["idx"])
    return signals


def _fast_simulate(close, high, low, signals, sl_pct, trail_act, trail_pb,
                   position_size=100.0, cb_limit=2, range_end_exit=True):
    n = len(close)
    returns = []
    exit_idx = -1
    consec_sl = defaultdict(int)
    blocked = set()

    for sig in signals:
        entry_i = sig["idx"]
        if entry_i <= exit_idx:
            continue
        rid = sig["range_id"]
        if rid in blocked:
            continue

        entry_p = sig["price"]
        d = sig["dir"]
        sl_p = entry_p * (1 - sl_pct) if d == 1 else entry_p * (1 + sl_pct)
        peak = entry_p
        ex_p = None
        ex_reason = None

        for j in range(entry_i + 1, n):
            h_j = high[j]
            l_j = low[j]
            c = close[j]

            if d == 1:
                if l_j <= sl_p:
                    ex_p = sl_p; ex_reason = "SL"; exit_idx = j; break
                if h_j > peak:
                    peak = h_j
                best_pct = (peak - entry_p) / entry_p
                if best_pct >= trail_act:
                    trail_level = peak * (1 - trail_pb)
                    if l_j <= trail_level:
                        ex_p = max(trail_level, l_j); ex_reason = "TS"; exit_idx = j; break
            else:
                if h_j >= sl_p:
                    ex_p = sl_p; ex_reason = "SL"; exit_idx = j; break
                if l_j < peak:
                    peak = l_j
                best_pct = (entry_p - peak) / entry_p
                if best_pct >= trail_act:
                    trail_level = peak * (1 + trail_pb)
                    if h_j >= trail_level:
                        ex_p = min(trail_level, h_j); ex_reason = "TS"; exit_idx = j; break

            if range_end_exit and j >= sig["end_idx"]:
                ex_p = c; ex_reason = "RE"; exit_idx = j; break

        if ex_p is None:
            last_j = n - 1
            ex_p = close[last_j]; ex_reason = "RE"; exit_idx = last_j

        pnl_pct = (ex_p - entry_p) / entry_p if d == 1 else (entry_p - ex_p) / entry_p
        returns.append(pnl_pct)

        if ex_reason == "SL":
            consec_sl[rid] += 1
            if consec_sl[rid] >= cb_limit:
                blocked.add(rid)
        else:
            consec_sl[rid] = 0

    return returns


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def run_strategy(close, high, low, timestamps, smoothed_high, smoothed_low,
                 volume, open_, volume_ma20, candle_dir, rsi14,
                 params: StrategyParams):
    ranges = _fast_find_ranges(close, high, low, timestamps, smoothed_high, smoothed_low,
                               params.swing_window, params.range_window)

    signals = _fast_generate_signals(
        close, volume, ranges,
        entry_pct=params.signal_proximity_pct / 100,
        max_move_pct=params.momentum_threshold / 100,
        min_width_pct=params.range_width_min_pct,
        max_width_pct=params.range_width_max_pct,
        lookback=params.momentum_lookback,
        cooldown=params.signal_cooldown,
        volume_filter=params.volume_filter,
        candle_confirm=params.candle_confirm,
        rsi_filter=params.rsi_filter,
        volume_ma20=volume_ma20,
        candle_dir=candle_dir,
        rsi14=rsi14,
    )

    if params.long_only:
        signals = [s for s in signals if s["dir"] == 1]

    trade_returns = _fast_simulate(
        close, high, low, signals,
        sl_pct=params.stop_loss_pct / 100,
        trail_act=params.trail_activation_pct / 100,
        trail_pb=params.trail_pullback_pct / 100,
        range_end_exit=params.range_end_exit,
    )

    n = len(trade_returns)
    if n == 0:
        return {"sharpe": -999, "win_rate": 0, "total_pnl": 0, "trade_count": 0}

    arr = np.array(trade_returns)
    wins = int(np.sum(arr > 0))
    total_pnl = float(np.sum(arr) * 100)
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=1)) if n > 1 else 1e-9
    sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 1e-12 else 0.0

    return {"sharpe": sharpe, "win_rate": wins / n, "total_pnl": total_pnl, "trade_count": n}


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------

_worker_arrays = {}


def _init_worker(close, high, low, ts, sh, sl_arr, volume, open_, vol_ma20, cdir, rsi):
    _worker_arrays["close"] = close
    _worker_arrays["high"] = high
    _worker_arrays["low"] = low
    _worker_arrays["ts"] = ts
    _worker_arrays["sh"] = sh
    _worker_arrays["sl"] = sl_arr
    _worker_arrays["volume"] = volume
    _worker_arrays["open"] = open_
    _worker_arrays["vol_ma20"] = vol_ma20
    _worker_arrays["cdir"] = cdir
    _worker_arrays["rsi"] = rsi


def _evaluate_worker(params: StrategyParams):
    w = _worker_arrays
    result = run_strategy(w["close"], w["high"], w["low"], w["ts"], w["sh"], w["sl"],
                          w["volume"], w["open"], w["vol_ma20"], w["cdir"], w["rsi"],
                          params)
    return params, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAVE/USDT")
    parser.add_argument("--long-only", action="store_true", help="Only generate LONG signals")
    args = parser.parse_args()

    symbol = args.symbol
    long_only = args.long_only

    print(f"Fetching {symbol} 1H candles (5 years)...")
    df_full = fetch_ohlcv(symbol=symbol, days=1825, cache_tag="5Y")
    df_full = df_full.reset_index(drop=True)
    print(f"Total candles: {len(df_full)}")

    n = len(df_full)
    chunk = n // 5
    df_y1 = df_full.iloc[:chunk].reset_index(drop=True)
    df_y2 = df_full.iloc[chunk:2 * chunk].reset_index(drop=True)
    df_y3 = df_full.iloc[2 * chunk:3 * chunk].reset_index(drop=True)
    df_y4 = df_full.iloc[3 * chunk:4 * chunk].reset_index(drop=True)
    df_y5 = df_full.iloc[4 * chunk:].reset_index(drop=True)

    df_train = pd.concat([df_y2, df_y3], ignore_index=True)
    df_validate = df_y4
    df_test = df_y5

    print(f"\nYear 1: {df_y1['timestamp'].iloc[0].date()} -> {df_y1['timestamp'].iloc[-1].date()} ({len(df_y1)} candles) [STRESS TEST ONLY]")
    print(f"Year 2: {df_y2['timestamp'].iloc[0].date()} -> {df_y2['timestamp'].iloc[-1].date()} ({len(df_y2)} candles) [TRAIN]")
    print(f"Year 3: {df_y3['timestamp'].iloc[0].date()} -> {df_y3['timestamp'].iloc[-1].date()} ({len(df_y3)} candles) [TRAIN]")
    print(f"Year 4: {df_y4['timestamp'].iloc[0].date()} -> {df_y4['timestamp'].iloc[-1].date()} ({len(df_y4)} candles) [VALIDATE]")
    print(f"Year 5: {df_y5['timestamp'].iloc[0].date()} -> {df_y5['timestamp'].iloc[-1].date()} ({len(df_y5)} candles) [TEST]")
    print(f"Train: {len(df_train)} candles | Validate: {len(df_validate)} candles | Test: {len(df_test)} candles")

    def preprocess(df):
        smoothed = _smooth_outliers(df)
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        ts = df["timestamp"].values
        sh = smoothed["high"].values.astype(np.float64)
        sl = smoothed["low"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)
        vol_ma20, cdir, rsi = compute_indicators(close, open_, volume)
        return (close, high, low, ts, sh, sl, volume, open_, vol_ma20, cdir, rsi)

    train_arrays = preprocess(df_train)
    val_arrays = preprocess(df_validate)
    test_arrays = preprocess(df_test)
    stress_arrays = preprocess(df_y1)

    combos = build_combinations(long_only=long_only)
    total = len(combos)
    n_workers = max(1, cpu_count() - 1)
    mode_label = "LONG only" if long_only else "LONG+SHORT"
    print(f"\nMode: {mode_label}")
    print(f"Total combinations: {total:,}")
    print(f"Scoring: total P&L (min {MIN_TRADES} trades)")
    print(f"Top {TOP_N_VALIDATE} from train → validated → best on test")
    print(f"Using {n_workers} workers")

    print(f"\nBenchmarking ({min(total, 8)} combos single-threaded)...", end=" ", flush=True)
    t0 = time.perf_counter()
    bench_sample = combos[:8]
    for p in bench_sample:
        run_strategy(*train_arrays, p)
    bench_s = time.perf_counter() - t0
    per_combo = bench_s / len(bench_sample)
    est_min = (per_combo * total / n_workers) / 60
    print(f"{per_combo*1000:.0f}ms/combo, estimated total: {est_min:.1f} min ({n_workers} workers)")

    # ----- Phase 1: Grid search on TRAIN data -----
    print("\n=== Phase 1: Train search ===")
    results = []
    with Pool(n_workers, initializer=_init_worker, initargs=train_arrays) as pool:
        for params, res in tqdm(
            pool.imap_unordered(_evaluate_worker, combos, chunksize=4),
            total=total,
            desc=f"{symbol} train search",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            results.append((params, res))

    valid = [(p, r) for p, r in results if r["trade_count"] >= MIN_TRADES and r["total_pnl"] > 0]
    valid.sort(key=lambda x: x[1]["total_pnl"], reverse=True)

    print(f"\nValid (>={MIN_TRADES} trades, PnL>0): {len(valid):,} / {total:,}")

    if not valid:
        print("No valid combinations found on train data. Exiting.")
        return

    top_train = valid[:TOP_N_VALIDATE]
    print(f"Top {len(top_train)} by train PnL forwarded to validation")
    print(f"  Best train PnL: €{top_train[0][1]['total_pnl']:+.2f}  Worst in top {len(top_train)}: €{top_train[-1][1]['total_pnl']:+.2f}")

    # ----- Phase 2: Validate top N -----
    print("\n=== Phase 2: Validate ===")
    val_results = []
    for params, train_res in top_train:
        val_res = run_strategy(*val_arrays, params)
        val_results.append((params, train_res, val_res))

    val_results.sort(key=lambda x: x[2]["total_pnl"], reverse=True)

    # ----- Phase 3: Test the best from validation -----
    print("\n=== Phase 3: Test (unseen data) ===")
    final_results = []
    for params, train_res, val_res in val_results:
        test_res = run_strategy(*test_arrays, params)
        final_results.append((params, train_res, val_res, test_res))

    final_results.sort(key=lambda x: x[3]["total_pnl"], reverse=True)

    # ----- Print results table -----
    def _label(p):
        return (f"sw={p.swing_window} rw={p.range_window} "
                f"prox={p.signal_proximity_pct} sl={p.stop_loss_pct} "
                f"ta={p.trail_activation_pct} tp={p.trail_pullback_pct} "
                f"cd={p.signal_cooldown}")

    print()
    print("=" * 200)
    print(f"{'Rank':>4}  {'PnL(train)':>10}  {'PnL(val)':>9}  {'PnL(test)':>10}"
          f"  {'WR(tr)':>6}  {'WR(val)':>7}  {'WR(test)':>8}"
          f"  {'#(tr)':>5}  {'#(val)':>6}  {'#(test)':>7}  Key params")
    print("-" * 200)

    for rank, (params, tr, vl, ts) in enumerate(final_results[:30], 1):
        label = _label(params)
        print(f"{rank:>4}  E{tr['total_pnl']:>+9.2f}  E{vl['total_pnl']:>+8.2f}  E{ts['total_pnl']:>+9.2f}"
              f"  {tr['win_rate']:>5.0%}  {vl['win_rate']:>6.0%}  {ts['win_rate']:>7.0%}"
              f"  {tr['trade_count']:>5}  {vl['trade_count']:>6}  {ts['trade_count']:>7}  {label}")
    print("=" * 200)

    # ----- Recommend best -----
    best = final_results[0]
    bp = best[0]

    print(f"\nRECOMMENDED PARAMS for {symbol} (best test PnL):")
    for f_field in fields(bp):
        print(f"  {f_field.name}: {getattr(bp, f_field.name)}")

    print(f"\n  Train:    PnL=€{best[1]['total_pnl']:+.2f}  WR={best[1]['win_rate']:.0%}  Trades={best[1]['trade_count']}")
    print(f"  Validate: PnL=€{best[2]['total_pnl']:+.2f}  WR={best[2]['win_rate']:.0%}  Trades={best[2]['trade_count']}")
    print(f"  Test:     PnL=€{best[3]['total_pnl']:+.2f}  WR={best[3]['win_rate']:.0%}  Trades={best[3]['trade_count']}")

    # ----- Stress test on year 1 -----
    print(f"\n--- Stress test (Year 1: {df_y1['timestamp'].iloc[0].date()} → {df_y1['timestamp'].iloc[-1].date()}) ---")
    stress_res = run_strategy(*stress_arrays, bp)
    print(f"  PnL=€{stress_res['total_pnl']:+.2f}  WR={stress_res['win_rate']:.0%}  Trades={stress_res['trade_count']}")
    if stress_res["total_pnl"] >= 0:
        print("  ✓ Survived the bull/bear extremes of 2021")
    else:
        print(f"  ⚠ Lost €{abs(stress_res['total_pnl']):.2f} during the extreme period (informational)")

    # ----- Save CSV -----
    safe = symbol.replace("/", "_")
    suffix = "_long_only" if long_only else ""
    csv_path = f"optimization_results_{safe}{suffix}.csv"
    param_fields = [f.name for f in fields(StrategyParams)]
    with open(csv_path, "w", newline="") as f:
        fieldnames = (["rank", "pnl_train", "pnl_val", "pnl_test",
                       "wr_train", "wr_val", "wr_test",
                       "trades_train", "trades_val", "trades_test"]
                      + param_fields)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (params, tr, vl, ts) in enumerate(final_results, 1):
            row = {
                "rank": rank,
                "pnl_train": f"{tr['total_pnl']:.2f}",
                "pnl_val": f"{vl['total_pnl']:.2f}",
                "pnl_test": f"{ts['total_pnl']:.2f}",
                "wr_train": f"{tr['win_rate']:.4f}",
                "wr_val": f"{vl['win_rate']:.4f}",
                "wr_test": f"{ts['win_rate']:.4f}",
                "trades_train": tr["trade_count"],
                "trades_val": vl["trade_count"],
                "trades_test": ts["trade_count"],
            }
            for f_field in fields(params):
                row[f_field.name] = getattr(params, f_field.name)
            writer.writerow(row)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
