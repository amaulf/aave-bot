import pandas as pd


def generate_momentum_signals(
    df: pd.DataFrame,
    rsi_long_max: float = 65.0,
    rsi_short_min: float = 35.0,
    cooldown: int = 10,
    long_only: bool = False,
    entry_mode: str = "engulfing",
    strong_candle_pct: float = 0.6,
    adx_min: float = 0.0,
) -> list[dict]:
    """
    Momentum strategy with configurable entry modes.

    entry_mode:
        "engulfing"     – classic bullish/bearish engulfing (original)
        "strong_candle" – body > strong_candle_pct of total range (relaxed)
        "macd_only"     – MACD flip + RSI only, no candle pattern required
    adx_min:
        Minimum ADX value to confirm trending market (0 = disabled).
    """
    signals: list[dict] = []
    last_signal_idx = -cooldown - 1
    has_adx = "adx14" in df.columns and adx_min > 0

    for i in range(1, len(df)):
        if i - last_signal_idx < cooldown:
            continue

        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        if pd.isna(curr.get("macd_hist")) or pd.isna(prev.get("macd_hist")):
            continue
        if pd.isna(curr.get("rsi14")):
            continue
        if has_adx and (pd.isna(curr.get("adx14")) or curr["adx14"] < adx_min):
            continue

        macd_bullish = prev["macd_hist"] <= 0 and curr["macd_hist"] > 0
        macd_bearish = prev["macd_hist"] >= 0 and curr["macd_hist"] < 0
        rsi = curr["rsi14"]

        if entry_mode == "engulfing":
            prev_body_top = max(prev["open"], prev["close"])
            prev_body_bot = min(prev["open"], prev["close"])
            curr_body_top = max(curr["open"], curr["close"])
            curr_body_bot = min(curr["open"], curr["close"])

            candle_bull = (
                prev["close"] < prev["open"]
                and curr["close"] > curr["open"]
                and curr_body_bot <= prev_body_bot
                and curr_body_top >= prev_body_top
            )
            candle_bear = (
                prev["close"] > prev["open"]
                and curr["close"] < curr["open"]
                and curr_body_bot <= prev_body_bot
                and curr_body_top >= prev_body_top
            )

        elif entry_mode == "strong_candle":
            total_range = curr["high"] - curr["low"]
            body = abs(curr["close"] - curr["open"])
            is_strong = (body / total_range >= strong_candle_pct) if total_range > 0 else False
            candle_bull = is_strong and curr["close"] > curr["open"]
            candle_bear = is_strong and curr["close"] < curr["open"]

        else:  # macd_only
            candle_bull = True
            candle_bear = True

        if candle_bull and macd_bullish and 30 <= rsi <= rsi_long_max:
            signals.append({
                "timestamp": curr["timestamp"],
                "close_price": curr["close"],
                "direction": "LONG",
                "range_id": -1,
            })
            last_signal_idx = i

        elif not long_only and candle_bear and macd_bearish and rsi_short_min <= rsi <= 70:
            signals.append({
                "timestamp": curr["timestamp"],
                "close_price": curr["close"],
                "direction": "SHORT",
                "range_id": -1,
            })
            last_signal_idx = i

    return signals
