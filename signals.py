import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    ranges: list[dict],
    entry_pct: float = 0.02,
    max_move_pct: float = 0.05,
    min_width_pct: float = 1.5,
    max_width_pct: float = 10.0,
    lookback: int = 8,
    cooldown: int = 12,
    volume_filter: bool = False,
    candle_confirm: bool = False,
    rsi_filter: bool = False,
) -> tuple[list[dict], int, int]:
    """Returns (accepted_signals, momentum_rejected, volatility_rejected)."""
    signals = []
    momentum_rejected = 0
    volatility_rejected = 0

    has_vol = volume_filter and "volume_ma20" in df.columns
    has_dir = candle_confirm and "candle_direction" in df.columns
    has_rsi = rsi_filter and "rsi14" in df.columns

    for r in ranges:
        width_pct = (r["range_high"] - r["range_low"]) / r["range_low"] * 100
        if width_pct < min_width_pct or width_pct > max_width_pct:
            volatility_rejected += 1
            continue

        mask = (df["timestamp"] >= r["start_time"]) & (df["timestamp"] <= r["end_time"])
        candles = df.loc[mask]
        long_threshold = r["range_low"] * (1 + entry_pct)
        short_threshold = r["range_high"] * (1 - entry_pct)
        last_long_pos = -cooldown - 1
        last_short_pos = -cooldown - 1

        for idx, c in candles.iterrows():
            pos = df.index.get_loc(idx)
            start = max(0, pos - lookback)
            past_close = df["close"].iloc[start]

            if c["close"] <= long_threshold and c["close"] >= r["range_low"] and (pos - last_long_pos) > cooldown:
                if has_vol and c["volume"] <= c["volume_ma20"]:
                    last_long_pos = pos
                    continue
                if has_dir and c["candle_direction"] != 1:
                    last_long_pos = pos
                    continue
                if has_rsi and c["rsi14"] >= 45:
                    last_long_pos = pos
                    continue
                drop = (past_close - c["close"]) / past_close
                if drop > max_move_pct:
                    momentum_rejected += 1
                    last_long_pos = pos
                else:
                    signals.append({
                        "timestamp": c["timestamp"],
                        "close_price": c["close"],
                        "range_id": r["range_id"],
                        "range_low": r["range_low"],
                        "range_high": r["range_high"],
                        "range_width_pct": width_pct,
                        "direction": "LONG",
                    })
                    last_long_pos = pos

            if c["close"] >= short_threshold and c["close"] <= r["range_high"] and (pos - last_short_pos) > cooldown:
                if has_vol and c["volume"] <= c["volume_ma20"]:
                    last_short_pos = pos
                    continue
                if has_dir and c["candle_direction"] != -1:
                    last_short_pos = pos
                    continue
                if has_rsi and c["rsi14"] <= 55:
                    last_short_pos = pos
                    continue
                rise = (c["close"] - past_close) / past_close
                if rise > max_move_pct:
                    momentum_rejected += 1
                    last_short_pos = pos
                else:
                    signals.append({
                        "timestamp": c["timestamp"],
                        "close_price": c["close"],
                        "range_id": r["range_id"],
                        "range_low": r["range_low"],
                        "range_high": r["range_high"],
                        "range_width_pct": width_pct,
                        "direction": "SHORT",
                    })
                    last_short_pos = pos

    return signals, momentum_rejected, volatility_rejected
