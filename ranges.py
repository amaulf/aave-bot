import pandas as pd


def _smooth_outliers(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Return a copy with wick outliers neutralized for range detection."""
    smoothed = df[["high", "low"]].copy()
    n = len(df)
    for i in range(2, n - 2):
        prev_low, next_low = df["low"].iloc[i - 1], df["low"].iloc[i + 1]
        if df["low"].iloc[i] < prev_low * (1 - threshold) and df["low"].iloc[i] < next_low * (1 - threshold):
            smoothed.iloc[i, smoothed.columns.get_loc("low")] = (
                df["low"].iloc[i - 2] + df["low"].iloc[i - 1] + df["low"].iloc[i + 1] + df["low"].iloc[i + 2]
            ) / 4

        prev_high, next_high = df["high"].iloc[i - 1], df["high"].iloc[i + 1]
        if df["high"].iloc[i] > prev_high * (1 + threshold) and df["high"].iloc[i] > next_high * (1 + threshold):
            smoothed.iloc[i, smoothed.columns.get_loc("high")] = (
                df["high"].iloc[i - 2] + df["high"].iloc[i - 1] + df["high"].iloc[i + 1] + df["high"].iloc[i + 2]
            ) / 4

    return smoothed


def detect_swings(df: pd.DataFrame, smoothed: pd.DataFrame, window: int = 10) -> tuple[list[int], list[int]]:
    highs, lows = [], []
    for i in range(window, len(df) - window):
        surrounding_highs = pd.concat([smoothed["high"].iloc[i - window : i], smoothed["high"].iloc[i + 1 : i + window + 1]])
        if smoothed["high"].iloc[i] >= surrounding_highs.max():
            highs.append(i)

        surrounding_lows = pd.concat([smoothed["low"].iloc[i - window : i], smoothed["low"].iloc[i + 1 : i + window + 1]])
        if smoothed["low"].iloc[i] <= surrounding_lows.min():
            lows.append(i)

    return highs, lows


def _within_pct(values: list[float], tolerance: float) -> bool:
    if len(values) < 2:
        return False
    mid = (max(values) + min(values)) / 2
    return (max(values) - min(values)) / mid <= tolerance


def _find_breakout(df: pd.DataFrame, range_high: float, range_low: float, start_idx: int, max_end: int) -> int:
    """Return the index where price closes >2% outside the range, or max_end if none."""
    threshold = 0.02
    for i in range(start_idx, min(max_end + 1, len(df))):
        close = df["close"].iloc[i]
        if close > range_high * (1 + threshold) or close < range_low * (1 - threshold):
            return i
    return max_end


def group_into_ranges(
    df: pd.DataFrame,
    smoothed: pd.DataFrame,
    swing_highs: list[int],
    swing_lows: list[int],
    scan_window: int = 60,
    price_tol: float = 0.02,
) -> list[dict]:
    high_set = set(swing_highs)
    low_set = set(swing_lows)
    all_swings = sorted(high_set | low_set)

    ranges: list[dict] = []
    consumed_up_to = -1

    for start_pos, anchor_idx in enumerate(all_swings):
        if anchor_idx <= consumed_up_to:
            continue

        window_highs = []
        window_lows = []
        window_end_idx = anchor_idx + scan_window

        for s in all_swings[start_pos:]:
            if s > window_end_idx:
                break
            if s in high_set:
                window_highs.append(s)
            if s in low_set:
                window_lows.append(s)

        if len(window_highs) < 2 or len(window_lows) < 2:
            continue

        high_prices = [smoothed["high"].iloc[i] for i in window_highs]
        low_prices = [smoothed["low"].iloc[i] for i in window_lows]

        if not _within_pct(high_prices, price_tol):
            continue
        if not _within_pct(low_prices, price_tol):
            continue

        all_in_window = window_highs + window_lows
        all_highs = [smoothed["high"].iloc[i] for i in all_in_window]
        all_lows = [smoothed["low"].iloc[i] for i in all_in_window]
        range_high = max(all_highs)
        range_low = min(all_lows)
        if range_low >= range_high:
            continue

        second_high_idx = sorted(window_highs)[1]
        second_low_idx = sorted(window_lows)[1]
        confirmation_idx = max(second_high_idx, second_low_idx)

        last_swing_idx = max(all_in_window)

        breakout_idx = _find_breakout(df, range_high, range_low, last_swing_idx, last_swing_idx + scan_window)

        ranges.append({
            "range_id": len(ranges),
            "range_high": range_high,
            "range_low": range_low,
            "start_time": df["timestamp"].iloc[confirmation_idx],
            "end_time": df["timestamp"].iloc[min(breakout_idx, len(df) - 1)],
        })

        consumed_up_to = breakout_idx

    return ranges


def find_ranges(df: pd.DataFrame, window: int = 10, scan_window: int = 60, price_tol: float = 0.02) -> tuple[list[int], list[int], list[dict]]:
    smoothed = _smooth_outliers(df)
    swing_highs, swing_lows = detect_swings(df, smoothed, window)
    ranges = group_into_ranges(df, smoothed, swing_highs, swing_lows, scan_window=scan_window, price_tol=price_tol)
    return swing_highs, swing_lows, ranges
