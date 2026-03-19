import pandas as pd

from strategies.base import Strategy, ParamDef
from momentum_signals import generate_momentum_signals


class MomentumStrategy(Strategy):

    @property
    def name(self) -> str:
        return "Momentum"

    @property
    def param_schema(self) -> dict[str, ParamDef]:
        return {
            "entry_mode":        ParamDef("select", "engulfing", options=["engulfing", "strong_candle", "macd_only"], label="Entry mode"),
            "strong_candle_pct": ParamDef("float", 0.6, 0.3, 0.9, 0.05, label="Strong candle body %"),
            "rsi_long_max":      ParamDef("float", 65.0, 50.0, 80.0, 1.0, label="RSI max for LONG"),
            "rsi_short_min":     ParamDef("float", 35.0, 20.0, 50.0, 1.0, label="RSI min for SHORT"),
            "adx_min":           ParamDef("float", 0.0, 0.0, 40.0, 1.0, label="ADX min (0 = off)"),
        }

    def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[list[dict], dict]:
        signals = generate_momentum_signals(
            df,
            rsi_long_max=params.get("rsi_long_max", 65.0),
            rsi_short_min=params.get("rsi_short_min", 35.0),
            cooldown=params.get("signal_cooldown", 10),
            long_only=params.get("long_only", False),
            entry_mode=params.get("entry_mode", "engulfing"),
            strong_candle_pct=params.get("strong_candle_pct", 0.6),
            adx_min=params.get("adx_min", 0.0),
        )
        metadata = {
            "ranges": [],
            "momentum_rejected": 0,
            "volatility_rejected": 0,
            "simulator_overrides": {},
        }
        return signals, metadata
