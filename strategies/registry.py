"""
Single source of truth for all strategies, assets, and experiments.
"""

from dataclasses import dataclass
from strategies.range_strategy import RangeStrategy
from strategies.momentum_strategy import MomentumStrategy

# ---------------------------------------------------------------------------
# Strategy instances
# ---------------------------------------------------------------------------

STRATEGIES = {
    "range": RangeStrategy(),
    "momentum": MomentumStrategy(),
}

# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Experiment:
    name: str
    strategy: str                    # key into STRATEGIES
    params: dict                     # values for strategy + shared params
    primary: bool = False            # highlighted in the UI
    description: tuple[str, str] = ("", "")  # (headline, body)


# ---------------------------------------------------------------------------
# AAVE/USDT experiments
# ---------------------------------------------------------------------------

_AAVE = [
    Experiment(
        name="Range · PnL Max",
        strategy="range",
        primary=True,
        params=dict(
            swing_window=12, range_window=60, range_width_min_pct=1.5,
            range_width_max_pct=10.0, signal_proximity_pct=2.0, momentum_lookback=10,
            momentum_threshold=7.0, trail_activation_pct=3.5, trail_pullback_pct=0.5,
            stop_loss_pct=2.0, signal_cooldown=8,
            volume_filter=False, candle_confirm=False, rsi_filter=False,
            range_end_exit=True, long_only=False,
        ),
        description=(
            "Maximise total profit",
            "Grid search scored on **total P&L** instead of Sharpe. The optimizer is "
            "free to accept higher variance if it means more euros at the end. This "
            "results in wider trailing activation, looser momentum thresholds, and "
            "a shorter cooldown — more trades, bigger swings.",
        ),
    ),
    Experiment(
        name="Range · Sharpe",
        strategy="range",
        params=dict(
            swing_window=8, range_window=80, range_width_min_pct=1.5,
            range_width_max_pct=12.0, signal_proximity_pct=1.5, momentum_lookback=14,
            momentum_threshold=5.0, trail_activation_pct=1.5, trail_pullback_pct=1.0,
            stop_loss_pct=1.5, signal_cooldown=14,
            volume_filter=False, candle_confirm=False, rsi_filter=False,
            range_end_exit=True, long_only=False,
        ),
        description=(
            "Maximise risk-adjusted returns",
            "Parameters were selected by a grid search scored on **Sharpe ratio** "
            "(return divided by volatility). This favours consistency over raw profit — "
            "the strategy takes fewer but steadier trades, with tighter trailing stops "
            "and a longer cooldown between signals.",
        ),
    ),
    Experiment(
        name="Range · PnL Max · Long Only",
        strategy="range",
        params=dict(
            swing_window=10, range_window=48, range_width_min_pct=3.0,
            range_width_max_pct=10.0, signal_proximity_pct=2.0, momentum_lookback=10,
            momentum_threshold=5.0, trail_activation_pct=3.5, trail_pullback_pct=0.5,
            stop_loss_pct=1.5, signal_cooldown=8,
            volume_filter=False, candle_confirm=False, rsi_filter=False,
            range_end_exit=True, long_only=True,
        ),
        description=(
            "Maximise profit without shorting",
            "Same P&L-based grid search, but restricted to **LONG-only** signals. "
            "Designed for exchanges or jurisdictions where shorting is unavailable "
            "(e.g. Binance in France). The optimizer re-tunes all parameters knowing "
            "it can only buy the dips, not sell the tops.",
        ),
    ),
    Experiment(
        name="Momentum · PnL Max",
        strategy="momentum",
        primary=True,
        params=dict(
            entry_mode="macd_only", strong_candle_pct=0.6, adx_min=0.0,
            rsi_long_max=65.0, rsi_short_min=40.0,
            trail_activation_pct=5.0, trail_pullback_pct=0.5,
            stop_loss_pct=4.0, signal_cooldown=6,
            long_only=False,
        ),
        description=(
            "Optimized momentum: MACD-only with wide stops",
            "Walk-forward grid search over 5 years scored on **total P&L**. The optimizer "
            "found that dropping the engulfing candle requirement (`macd_only` mode) and "
            "using a **wide 4% stop loss** with a **5% trail activation / 0.5% pullback** "
            "captures the big 6-15% momentum moves while surviving initial noise. "
            "Test year: €+155 (+15.5%), stress test (2021): €+296.",
        ),
    ),
    Experiment(
        name="Momentum · PnL Max · Long Only",
        strategy="momentum",
        params=dict(
            entry_mode="macd_only", strong_candle_pct=0.6, adx_min=10.0,
            rsi_long_max=65.0, rsi_short_min=25.0,
            trail_activation_pct=5.0, trail_pullback_pct=0.3,
            stop_loss_pct=1.2, signal_cooldown=6,
            long_only=True,
        ),
        description=(
            "Optimized momentum restricted to LONG entries",
            "Walk-forward optimization with `long_only=True` to force buy-only behavior. "
            "A deeper refined grid selected `macd_only` with ADX filter, tighter pullback, and "
            "a lower stop loss. Optimizer test-year result (at optimizer sizing): €+53.8; "
            "validation stayed positive.",
        ),
    ),
    Experiment(
        name="Momentum · Balanced · Long Only",
        strategy="momentum",
        params=dict(
            entry_mode="macd_only", strong_candle_pct=0.6, adx_min=0.0,
            rsi_long_max=65.0, rsi_short_min=25.0,
            trail_activation_pct=4.5, trail_pullback_pct=0.3,
            stop_loss_pct=2.5, signal_cooldown=6,
            long_only=True,
        ),
        description=(
            "Long-only momentum tuned for smoother psychology",
            "Selected from the deep long-only search using a balanced score that rewards "
            "higher win rate and penalizes drawdown/loss streaks (not just raw P&L). "
            "Compared to pure PnL long-only, this profile trades with a tighter behavioral "
            "footprint while keeping strong absolute return.",
        ),
    ),
    Experiment(
        name="Momentum · Baseline (Engulfing)",
        strategy="momentum",
        params=dict(
            entry_mode="engulfing", strong_candle_pct=0.6, adx_min=0.0,
            rsi_long_max=65.0, rsi_short_min=35.0,
            trail_activation_pct=2.0, trail_pullback_pct=1.0,
            stop_loss_pct=2.0, signal_cooldown=10,
            long_only=False,
        ),
        description=(
            "Baseline momentum with strict engulfing filter",
            "Original momentum approach requiring a **bullish/bearish engulfing candle** + "
            "**MACD histogram flip** + RSI filter. More conservative but generates fewer "
            "signals — kept as a comparison baseline against the optimized version.",
        ),
    ),
]

# ---------------------------------------------------------------------------
# BTC/USDT experiments
# ---------------------------------------------------------------------------

_BTC = [
    Experiment(
        name="Range · Sharpe",
        strategy="range",
        primary=True,
        params=dict(
            swing_window=10, range_window=80, range_width_min_pct=3.0,
            range_width_max_pct=10.0, signal_proximity_pct=2.0, momentum_lookback=8,
            momentum_threshold=5.0, trail_activation_pct=1.5, trail_pullback_pct=1.0,
            stop_loss_pct=1.5, signal_cooldown=14,
            volume_filter=False, candle_confirm=False, rsi_filter=False,
            range_end_exit=True, long_only=False,
        ),
        description=(
            "Maximise risk-adjusted returns",
            "Parameters were selected by a grid search scored on **Sharpe ratio**.",
        ),
    ),
    Experiment(
        name="Momentum · Baseline (Engulfing)",
        strategy="momentum",
        params=dict(
            entry_mode="engulfing", strong_candle_pct=0.6, adx_min=0.0,
            rsi_long_max=65.0, rsi_short_min=35.0,
            trail_activation_pct=2.0, trail_pullback_pct=1.0,
            stop_loss_pct=2.0, signal_cooldown=10,
            long_only=False,
        ),
        description=(
            "Trend-following with candlestick confirmation",
            "MACD histogram flip + bullish/bearish engulfing candle + RSI filter.",
        ),
    ),
]

# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

ASSETS = ["AAVE/USDT", "BTC/USDT"]

EXPERIMENTS: dict[str, list[Experiment]] = {
    "AAVE/USDT": _AAVE,
    "BTC/USDT": _BTC,
}
