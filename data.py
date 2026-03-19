import os
import time
import ccxt
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_MAX_AGE = 3600


def fetch_ohlcv(symbol: str = "AAVE/USDT", timeframe: str = "1h", days: int = 365, cache_tag: str = "") -> pd.DataFrame:
    safe_symbol = symbol.replace("/", "_")
    filename = f"{safe_symbol}_{timeframe.upper()}{'_' + cache_tag if cache_tag else ''}.csv"
    cache_file = CACHE_DIR / filename

    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < CACHE_MAX_AGE:
        print(f"Loading from cache ({filename})...")
        df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        _add_indicators(df)
        return df

    print(f"Fetching {days}d from Binance...")
    exchange = ccxt.binance()
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_candles = []

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        if len(candles) < 1000:
            break

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    CACHE_DIR.mkdir(exist_ok=True)
    df.to_csv(cache_file, index=False)

    _add_indicators(df)
    return df


def _add_indicators(df: pd.DataFrame) -> None:
    df["volume_ma20"] = df["volume"].rolling(20).mean()

    df["candle_direction"] = 0
    df.loc[df["close"] > df["open"], "candle_direction"] = 1
    df.loc[df["close"] < df["open"], "candle_direction"] = -1

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

    atr14 = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr14.replace(0, 1e-10)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr14.replace(0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    df["adx14"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
