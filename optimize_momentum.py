"""
Walk-forward PnL optimizer for the momentum strategy.

  python3 optimize_momentum.py --symbol "AAVE/USDT"

Data: 5 years fetched, last 4 years used (skip anomalous year 1).
Split: Train (Y2-Y3) -> Validate (Y4, top 50) -> Test (Y5, final pick).
Scoring: total P&L with min 10 trades.
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


# ---------------------------------------------------------------------------
# Params dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MomentumParams:
    entry_mode: str = "engulfing"        # engulfing | strong_candle | macd_only
    strong_candle_pct: float = 0.6
    rsi_long_max: float = 65.0
    rsi_short_min: float = 35.0
    cooldown: int = 10
    adx_min: float = 0.0
    trail_activation_pct: float = 2.0
    trail_pullback_pct: float = 1.0
    stop_loss_pct: float = 2.0
    long_only: bool = False


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

GRID_ENGULFING = {
    "rsi_long_max":         [60, 65, 70, 75],
    "rsi_short_min":        [25, 30, 35, 40],
    "cooldown":             [6, 8, 10, 14],
    "adx_min":              [0.0, 15.0, 20.0, 25.0],
    "trail_activation_pct": [2.0, 3.0, 4.0, 5.0],
    "trail_pullback_pct":   [0.5, 1.0, 1.5],
    "stop_loss_pct":        [1.5, 2.0, 3.0, 4.0],
}

GRID_STRONG = {
    "strong_candle_pct": [0.5, 0.6, 0.7],
    **GRID_ENGULFING,
}

# Deeper/finer grid specifically for long-only optimization.
# Note: rsi_short_min has no effect in long-only mode, so it is fixed.
GRID_ENGULFING_LONG_ONLY = {
    "rsi_long_max":         [55, 58, 60, 62, 65, 68, 70],
    "rsi_short_min":        [25],
    "cooldown":             [4, 5, 6, 7, 8, 10, 12],
    "adx_min":              [0.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0],
    "trail_activation_pct": [3.5, 4.0, 4.5, 5.0, 5.5],
    "trail_pullback_pct":   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "stop_loss_pct":        [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
}

GRID_STRONG_LONG_ONLY = {
    "strong_candle_pct": [0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
    **GRID_ENGULFING_LONG_ONLY,
}

TOP_N_VALIDATE = 50
MIN_TRADES = 10


def build_combinations(long_only: bool = False) -> list[MomentumParams]:
    combos = []
    if long_only:
        mode_grids = [
            ("engulfing", GRID_ENGULFING_LONG_ONLY),
            ("strong_candle", GRID_STRONG_LONG_ONLY),
            ("macd_only", GRID_ENGULFING_LONG_ONLY),
        ]
    else:
        mode_grids = [
            ("engulfing", GRID_ENGULFING),
            ("strong_candle", GRID_STRONG),
            ("macd_only", GRID_ENGULFING),
        ]

    for mode, grid in mode_grids:
        keys = list(grid.keys())
        for vals in product(*grid.values()):
            kw = dict(zip(keys, vals))
            kw["entry_mode"] = mode
            if mode != "strong_candle":
                kw["strong_candle_pct"] = 0.6
            combos.append(MomentumParams(**kw, long_only=long_only))
    return combos


# ---------------------------------------------------------------------------
# Numpy indicator computation
# ---------------------------------------------------------------------------

def compute_indicators(close, open_, high, low, volume):
    n = len(close)

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

    ema12 = _ewm(close, 12)
    ema26 = _ewm(close, 26)
    macd = ema12 - ema26
    macd_signal = _ewm(macd, 9)
    macd_hist = macd - macd_signal

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]
    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr14 = _ewm_alpha(tr, alpha)
    plus_di = 100 * _ewm_alpha(plus_dm, alpha) / np.where(atr14 == 0, 1e-10, atr14)
    minus_di = 100 * _ewm_alpha(minus_dm, alpha) / np.where(atr14 == 0, 1e-10, atr14)
    di_sum = plus_di + minus_di
    dx = 100 * np.abs(plus_di - minus_di) / np.where(di_sum == 0, 1e-10, di_sum)
    adx14 = _ewm_alpha(dx, alpha)

    return rsi14, macd_hist, adx14


def _ewm(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _ewm_alpha(arr, alpha):
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Fast numpy signal generation
# ---------------------------------------------------------------------------

def _fast_momentum_signals(close, open_, high, low, rsi14, macd_hist, adx14, params: MomentumParams):
    n = len(close)
    signals = []
    last_sig = -params.cooldown - 1

    for i in range(1, n):
        if i - last_sig < params.cooldown:
            continue
        if np.isnan(macd_hist[i]) or np.isnan(macd_hist[i - 1]) or np.isnan(rsi14[i]):
            continue
        if params.adx_min > 0 and (np.isnan(adx14[i]) or adx14[i] < params.adx_min):
            continue

        macd_bull = macd_hist[i - 1] <= 0 and macd_hist[i] > 0
        macd_bear = macd_hist[i - 1] >= 0 and macd_hist[i] < 0
        rsi = rsi14[i]

        if params.entry_mode == "engulfing":
            prev_body_top = max(open_[i - 1], close[i - 1])
            prev_body_bot = min(open_[i - 1], close[i - 1])
            curr_body_top = max(open_[i], close[i])
            curr_body_bot = min(open_[i], close[i])
            c_bull = (close[i - 1] < open_[i - 1] and close[i] > open_[i]
                      and curr_body_bot <= prev_body_bot and curr_body_top >= prev_body_top)
            c_bear = (close[i - 1] > open_[i - 1] and close[i] < open_[i]
                      and curr_body_bot <= prev_body_bot and curr_body_top >= prev_body_top)
        elif params.entry_mode == "strong_candle":
            total_range = high[i] - low[i]
            body = abs(close[i] - open_[i])
            is_strong = (body / total_range >= params.strong_candle_pct) if total_range > 0 else False
            c_bull = is_strong and close[i] > open_[i]
            c_bear = is_strong and close[i] < open_[i]
        else:  # macd_only
            c_bull = True
            c_bear = True

        if c_bull and macd_bull and 30 <= rsi <= params.rsi_long_max:
            signals.append({"idx": i, "price": close[i], "dir": 1})
            last_sig = i
        elif not params.long_only and c_bear and macd_bear and params.rsi_short_min <= rsi <= 70:
            signals.append({"idx": i, "price": close[i], "dir": -1})
            last_sig = i

    return signals


# ---------------------------------------------------------------------------
# Fast simulator (no ranges, no circuit breaker)
# ---------------------------------------------------------------------------

def _fast_simulate(close, high, low, signals, sl_pct, trail_act, trail_pb):
    n = len(close)
    returns = []
    exit_idx = -1

    for sig in signals:
        entry_i = sig["idx"]
        if entry_i <= exit_idx:
            continue

        entry_p = sig["price"]
        d = sig["dir"]
        sl_p = entry_p * (1 - sl_pct) if d == 1 else entry_p * (1 + sl_pct)
        peak = entry_p
        ex_p = None

        for j in range(entry_i + 1, n):
            h_j = high[j]
            l_j = low[j]

            if d == 1:
                if l_j <= sl_p:
                    ex_p = sl_p; exit_idx = j; break
                if h_j > peak:
                    peak = h_j
                best_pct = (peak - entry_p) / entry_p
                if best_pct >= trail_act:
                    trail_level = peak * (1 - trail_pb)
                    if l_j <= trail_level:
                        ex_p = max(trail_level, l_j); exit_idx = j; break
            else:
                if h_j >= sl_p:
                    ex_p = sl_p; exit_idx = j; break
                if l_j < peak:
                    peak = l_j
                best_pct = (entry_p - peak) / entry_p
                if best_pct >= trail_act:
                    trail_level = peak * (1 + trail_pb)
                    if h_j >= trail_level:
                        ex_p = min(trail_level, h_j); exit_idx = j; break

        if ex_p is None:
            ex_p = close[-1]; exit_idx = n - 1

        pnl_pct = (ex_p - entry_p) / entry_p if d == 1 else (entry_p - ex_p) / entry_p
        returns.append(pnl_pct)

    return returns


# ---------------------------------------------------------------------------
# Single-run evaluation
# ---------------------------------------------------------------------------

def run_strategy(close, open_, high, low, rsi14, macd_hist, adx14, params: MomentumParams):
    signals = _fast_momentum_signals(close, open_, high, low, rsi14, macd_hist, adx14, params)

    trade_returns = _fast_simulate(
        close, high, low, signals,
        sl_pct=params.stop_loss_pct / 100,
        trail_act=params.trail_activation_pct / 100,
        trail_pb=params.trail_pullback_pct / 100,
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


def _init_worker(close, open_, high, low, rsi14, macd_hist, adx14):
    _worker_arrays["close"] = close
    _worker_arrays["open"] = open_
    _worker_arrays["high"] = high
    _worker_arrays["low"] = low
    _worker_arrays["rsi14"] = rsi14
    _worker_arrays["macd_hist"] = macd_hist
    _worker_arrays["adx14"] = adx14


def _evaluate_worker(params: MomentumParams):
    w = _worker_arrays
    result = run_strategy(w["close"], w["open"], w["high"], w["low"],
                          w["rsi14"], w["macd_hist"], w["adx14"], params)
    return params, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAVE/USDT")
    parser.add_argument("--long-only", action="store_true")
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

    print(f"\nYear 1: {df_y1['timestamp'].iloc[0].date()} -> {df_y1['timestamp'].iloc[-1].date()} ({len(df_y1)} candles) [STRESS TEST]")
    print(f"Year 2-3: {df_train['timestamp'].iloc[0].date()} -> {df_train['timestamp'].iloc[-1].date()} ({len(df_train)} candles) [TRAIN]")
    print(f"Year 4: {df_validate['timestamp'].iloc[0].date()} -> {df_validate['timestamp'].iloc[-1].date()} ({len(df_validate)} candles) [VALIDATE]")
    print(f"Year 5: {df_test['timestamp'].iloc[0].date()} -> {df_test['timestamp'].iloc[-1].date()} ({len(df_test)} candles) [TEST]")

    def preprocess(df):
        close = df["close"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)
        rsi14, macd_hist, adx14 = compute_indicators(close, open_, high, low, volume)
        return (close, open_, high, low, rsi14, macd_hist, adx14)

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
    print(f"Top {TOP_N_VALIDATE} from train -> validated -> best on test")
    print(f"Using {n_workers} workers")

    print(f"\nBenchmarking ({min(total, 16)} combos single-threaded)...", end=" ", flush=True)
    t0 = time.perf_counter()
    bench = combos[:16]
    for p in bench:
        run_strategy(*train_arrays, p)
    bench_s = time.perf_counter() - t0
    per_combo = bench_s / len(bench)
    est_min = (per_combo * total / n_workers) / 60
    print(f"{per_combo*1000:.1f}ms/combo, estimated total: {est_min:.1f} min ({n_workers} workers)")

    # ----- Phase 1: Train -----
    print("\n=== Phase 1: Train search ===")
    results = []
    with Pool(n_workers, initializer=_init_worker, initargs=train_arrays) as pool:
        for params, res in tqdm(
            pool.imap_unordered(_evaluate_worker, combos, chunksize=8),
            total=total,
            desc=f"{symbol} momentum train",
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

    # ----- Phase 2: Validate -----
    print("\n=== Phase 2: Validate ===")
    val_results = []
    for params, train_res in top_train:
        val_res = run_strategy(*val_arrays, params)
        val_results.append((params, train_res, val_res))

    val_results.sort(key=lambda x: x[2]["total_pnl"], reverse=True)

    # ----- Phase 3: Test -----
    print("\n=== Phase 3: Test (unseen data) ===")
    final_results = []
    for params, train_res, val_res in val_results:
        test_res = run_strategy(*test_arrays, params)
        final_results.append((params, train_res, val_res, test_res))

    final_results.sort(key=lambda x: x[3]["total_pnl"], reverse=True)

    # ----- Results table -----
    def _label(p):
        return (f"mode={p.entry_mode} sc={p.strong_candle_pct} "
                f"rsi=[{p.rsi_short_min},{p.rsi_long_max}] cd={p.cooldown} "
                f"adx={p.adx_min} sl={p.stop_loss_pct} "
                f"ta={p.trail_activation_pct} tp={p.trail_pullback_pct}")

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

    # ----- Stress test -----
    print(f"\n--- Stress test (Year 1: {df_y1['timestamp'].iloc[0].date()} -> {df_y1['timestamp'].iloc[-1].date()}) ---")
    stress_res = run_strategy(*stress_arrays, bp)
    print(f"  PnL=€{stress_res['total_pnl']:+.2f}  WR={stress_res['win_rate']:.0%}  Trades={stress_res['trade_count']}")
    if stress_res["total_pnl"] >= 0:
        print("  Survived the extreme period")
    else:
        print(f"  Lost €{abs(stress_res['total_pnl']):.2f} during extreme period (informational)")

    # ----- Save CSV -----
    safe = symbol.replace("/", "_")
    suffix = "_long_only" if long_only else ""
    csv_path = f"optimization_momentum_{safe}{suffix}.csv"
    param_fields = [f.name for f in fields(MomentumParams)]
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
