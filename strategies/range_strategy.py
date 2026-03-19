import pandas as pd

from strategies.base import Strategy, ParamDef
from ranges import find_ranges
from signals import generate_signals


class RangeStrategy(Strategy):

    @property
    def name(self) -> str:
        return "Range"

    @property
    def param_schema(self) -> dict[str, ParamDef]:
        return {
            "swing_window":         ParamDef("int",   10,   5,   20,  1,    label="Swing window"),
            "range_window":         ParamDef("int",   60,  30,  120,  1,    label="Range window"),
            "signal_proximity_pct": ParamDef("float", 1.5, 0.5,  4.0, 0.1, label="Signal proximity %"),
            "momentum_lookback":    ParamDef("int",   10,   4,   20,  1,    label="Momentum lookback"),
            "momentum_threshold":   ParamDef("float", 5.0, 2.0, 10.0, 0.5, label="Momentum threshold %"),
            "range_width_min_pct":  ParamDef("float", 2.0, 0.5,  5.0, 0.5, label="Range width min %"),
            "range_width_max_pct":  ParamDef("float",12.0, 5.0, 20.0, 1.0, label="Range width max %"),
            "volume_filter":        ParamDef("bool", False, label="Volume filter"),
            "candle_confirm":       ParamDef("bool", False, label="Candle direction confirm"),
            "rsi_filter":           ParamDef("bool", False, label="RSI filter"),
            "range_end_exit":       ParamDef("bool", True,  label="Close on range breakout"),
        }

    @property
    def uses_ranges(self) -> bool:
        return True

    @property
    def simulator_overrides(self) -> dict:
        return {}

    def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[list[dict], dict]:
        _, _, detected_ranges = find_ranges(
            df,
            window=params["swing_window"],
            scan_window=params["range_window"],
        )

        signals, momentum_rejected, volatility_rejected = generate_signals(
            df, detected_ranges,
            entry_pct=params["signal_proximity_pct"] / 100,
            max_move_pct=params["momentum_threshold"] / 100,
            min_width_pct=params["range_width_min_pct"],
            max_width_pct=params["range_width_max_pct"],
            lookback=params["momentum_lookback"],
            cooldown=params["signal_cooldown"],
            volume_filter=params.get("volume_filter", False),
            candle_confirm=params.get("candle_confirm", False),
            rsi_filter=params.get("rsi_filter", False),
        )

        if params.get("long_only"):
            signals = [s for s in signals if s["direction"] == "LONG"]

        sim_kw = {
            "range_end_exit": params.get("range_end_exit", True),
        }

        metadata = {
            "ranges": detected_ranges,
            "momentum_rejected": momentum_rejected,
            "volatility_rejected": volatility_rejected,
            "simulator_overrides": sim_kw,
        }
        return signals, metadata
