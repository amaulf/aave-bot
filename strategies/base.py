from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ParamDef:
    """Describes a single tunable parameter for the UI."""
    type: str          # "float", "int", "bool", "select"
    default: Any
    min: Any = None
    max: Any = None
    step: Any = None
    options: list | None = None
    label: str = ""


SHARED_PARAMS: dict[str, ParamDef] = {
    "trail_activation_pct": ParamDef("float", 2.0, 0.5, 5.0, 0.1, label="Trail activation %"),
    "trail_pullback_pct":   ParamDef("float", 1.0, 0.5, 3.0, 0.1, label="Trail pullback %"),
    "stop_loss_pct":        ParamDef("float", 2.0, 0.5, 5.0, 0.1, label="Stop loss %"),
    "signal_cooldown":      ParamDef("int",  10,   4,  24,  1,    label="Signal cooldown (candles)"),
    "long_only":            ParamDef("bool", False, label="LONG only (no shorts)"),
}


class Strategy(ABC):
    """Common interface for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @property
    @abstractmethod
    def param_schema(self) -> dict[str, ParamDef]:
        """Strategy-specific parameters (excluding shared ones)."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict) -> tuple[list[dict], dict]:
        """
        Returns (signals, metadata).
        - signals: list of dicts with at least {timestamp, close_price, direction, range_id}
        - metadata: strategy-specific info for the UI (e.g. ranges, rejection counts)
        """

    @property
    def uses_ranges(self) -> bool:
        return False

    @property
    def simulator_overrides(self) -> dict:
        """Extra kwargs to pass to simulate() for this strategy type."""
        return {"range_end_exit": False, "circuit_breaker_limit": 9999}
