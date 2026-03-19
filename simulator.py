import pandas as pd
from collections import defaultdict


def simulate(
    df: pd.DataFrame,
    signals: list[dict],
    ranges: list[dict],
    capital: float = 1000.0,
    position_size: float = 1000.0,
    sl_pct: float = 0.02,
    trail_activation: float = 0.01,
    trail_pullback: float = 0.01,
    max_hold: int = 48,
    circuit_breaker_limit: int = 2,
    range_end_exit: bool = True,
) -> tuple[list[dict], dict]:
    """Returns (trades, blocked_ranges_info)."""
    sorted_signals = sorted(signals, key=lambda s: s["timestamp"])

    trades: list[dict] = []
    in_trade = False
    exit_time = None

    consecutive_sl: dict[int, int] = defaultdict(int)
    blocked_ranges: set[int] = set()
    skipped_by_breaker: dict[int, int] = defaultdict(int)

    for sig in sorted_signals:
        if in_trade and sig["timestamp"] <= exit_time:
            continue

        range_id = sig["range_id"]

        if range_id in blocked_ranges:
            skipped_by_breaker[range_id] += 1
            continue

        entry_price = sig["close_price"]
        direction = sig["direction"]
        entry_time = sig["timestamp"]

        range_end = None
        for r in ranges:
            if r["range_id"] == range_id:
                range_end = r["end_time"]
                break
        if range_end is None:
            range_end = df["timestamp"].iloc[-1]

        if direction == "LONG":
            sl_price = entry_price * (1 - sl_pct)
        else:
            sl_price = entry_price * (1 + sl_pct)

        future = df[df["timestamp"] > entry_time]
        entry_pos = df[df["timestamp"] == entry_time].index[0]

        exit_price = None
        exit_reason = None
        exit_ts = None
        exit_pos = None
        peak = entry_price
        peak_pct = 0.0

        for row_idx, c in future.iterrows():
            candles_since_entry = int(row_idx - entry_pos)
            close = c["close"]
            candle_high = c["high"]
            candle_low = c["low"]

            if direction == "LONG":
                if candle_low <= sl_price:
                    exit_price, exit_reason, exit_ts, exit_pos = sl_price, "SL", c["timestamp"], row_idx
                    peak_pct = max(peak_pct, (peak - entry_price) / entry_price)
                    break

                if candle_high > peak:
                    peak = candle_high
                best_pct = (peak - entry_price) / entry_price

                if best_pct >= trail_activation:
                    peak_pct = best_pct
                    trail_level = peak * (1 - trail_pullback)
                    if candle_low <= trail_level:
                        exit_price = max(trail_level, candle_low)
                        exit_reason, exit_ts, exit_pos = "TRAILING_STOP", c["timestamp"], row_idx
                        break
                else:
                    peak_pct = max(peak_pct, (close - entry_price) / entry_price)
            else:
                if candle_high >= sl_price:
                    exit_price, exit_reason, exit_ts, exit_pos = sl_price, "SL", c["timestamp"], row_idx
                    peak_pct = max(peak_pct, (entry_price - peak) / entry_price)
                    break

                if candle_low < peak:
                    peak = candle_low
                best_pct = (entry_price - peak) / entry_price

                if best_pct >= trail_activation:
                    peak_pct = best_pct
                    trail_level = peak * (1 + trail_pullback)
                    if candle_high >= trail_level:
                        exit_price = min(trail_level, candle_high)
                        exit_reason, exit_ts, exit_pos = "TRAILING_STOP", c["timestamp"], row_idx
                        break
                else:
                    peak_pct = max(peak_pct, (entry_price - close) / entry_price)

            if range_end_exit and c["timestamp"] >= range_end:
                exit_price, exit_reason, exit_ts, exit_pos = close, "RANGE_END", c["timestamp"], row_idx
                break

        if exit_price is None:
            last_idx = future.index[-1] if not future.empty else df.index[-1]
            last = df.loc[last_idx]
            exit_price, exit_reason, exit_ts, exit_pos = last["close"], "RANGE_END", last["timestamp"], last_idx

        if direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        candles_held = int(exit_pos - entry_pos)

        trades.append({
            "entry_time": entry_time,
            "exit_time": exit_ts,
            "direction": direction,
            "range_id": range_id,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_eur": position_size * pnl_pct,
            "pnl_pct": pnl_pct * 100,
            "peak_pct": peak_pct * 100,
            "exit_reason": exit_reason,
            "candles_held": candles_held,
        })

        if exit_reason == "SL":
            consecutive_sl[range_id] += 1
            if consecutive_sl[range_id] >= circuit_breaker_limit:
                blocked_ranges.add(range_id)
        else:
            consecutive_sl[range_id] = 0

        in_trade = True
        exit_time = exit_ts

    blocked_info = {rid: skipped_by_breaker[rid] for rid in blocked_ranges}
    return trades, blocked_info


def print_diary(trades: list[dict], blocked_info: dict[int, int] | None = None) -> None:
    if not trades:
        print("\nNo trades executed.")
        return

    header = (
        f"{'#':>3}  {'Dir':5s}  {'RID':>3s}  {'Entry Time':25s}  {'Exit Time':25s}"
        f"  {'Entry':>8s}  {'Exit':>8s}  {'P&L €':>8s}  {'P&L %':>7s}  {'Peak%':>6s}  {'Held':>4s}  {'Reason'}"
    )
    print(f"\n{'=' * len(header)}")
    print("TRADE DIARY")
    print(f"{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")

    for i, t in enumerate(trades, 1):
        print(
            f"{i:>3}  {t['direction']:5s}  {t['range_id']:>3d}  {str(t['entry_time']):25s}  {str(t['exit_time']):25s}"
            f"  {t['entry_price']:>8.2f}  {t['exit_price']:>8.2f}"
            f"  {t['pnl_eur']:>+8.2f}  {t['pnl_pct']:>+6.2f}%  {t['peak_pct']:>+5.1f}%  {t['candles_held']:>4d}  {t['exit_reason']}"
        )

    fast_exits = sum(1 for t in trades if t["candles_held"] <= 3)
    if len(trades) > 0 and fast_exits / len(trades) > 0.30:
        print(f"\n⚠ WARNING: Possible lookahead bias detected — "
              f"{fast_exits}/{len(trades)} trades ({fast_exits/len(trades)*100:.0f}%) exit within 3 candles.")

    wins = [t for t in trades if t["pnl_eur"] > 0]
    losses = [t for t in trades if t["pnl_eur"] <= 0]
    total_pnl = sum(t["pnl_eur"] for t in trades)
    avg_win = sum(t["pnl_eur"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_eur"] for t in losses) / len(losses) if losses else 0
    best = max(trades, key=lambda t: t["pnl_eur"])
    worst = min(trades, key=lambda t: t["pnl_eur"])

    print(f"\n{'— SUMMARY —':^{len(header)}}")
    print(f"  Total trades:  {len(trades)}")
    print(f"  Win rate:      {len(wins)}/{len(trades)} ({len(wins)/len(trades)*100:.0f}%)")
    print(f"  Total P&L:     €{total_pnl:+.2f}")
    print(f"  Avg win:       €{avg_win:+.2f}")
    print(f"  Avg loss:      €{avg_loss:+.2f}")
    print(f"  Best trade:    €{best['pnl_eur']:+.2f} ({best['direction']} @ {best['entry_time']})")
    print(f"  Worst trade:   €{worst['pnl_eur']:+.2f} ({worst['direction']} @ {worst['entry_time']})")

    if blocked_info:
        total_skipped = sum(blocked_info.values())
        print(f"\n  Circuit breaker blocked {len(blocked_info)} range(s), skipped {total_skipped} signal(s):")
        for rid, count in sorted(blocked_info.items()):
            print(f"    Range {rid}: {count} signal(s) skipped")
