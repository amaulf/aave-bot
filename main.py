"""
CLI entrypoint — runs the primary experiment for a given asset and prints results.
Usage: python main.py [AAVE/USDT|BTC/USDT]
"""

import sys

from data import fetch_ohlcv
from simulator import simulate, print_diary
from strategies.registry import EXPERIMENTS, STRATEGIES


def main(symbol: str = "AAVE/USDT") -> None:
    experiments = EXPERIMENTS.get(symbol)
    if not experiments:
        print(f"Unknown asset: {symbol}")
        return

    exp = next((e for e in experiments if e.primary), experiments[0])
    strategy_obj = STRATEGIES[exp.strategy]

    print(f"Running '{exp.name}' ({strategy_obj.name}) on {symbol}...")
    df = fetch_ohlcv(symbol=symbol)
    print(f"Total candles fetched: {len(df)}")

    signals, metadata = strategy_obj.generate_signals(df, exp.params)

    longs = [s for s in signals if s["direction"] == "LONG"]
    shorts = [s for s in signals if s["direction"] == "SHORT"]
    print(f"Signals: {len(longs)} LONG, {len(shorts)} SHORT")

    if metadata.get("ranges"):
        print(f"Ranges detected: {len(metadata['ranges'])}")
    if metadata.get("momentum_rejected"):
        print(f"Rejected — momentum: {metadata['momentum_rejected']}, volatility: {metadata.get('volatility_rejected', 0)}")

    sim_kw = {**strategy_obj.simulator_overrides, **metadata.get("simulator_overrides", {})}
    trades, blocked_info = simulate(
        df, signals, metadata.get("ranges", []),
        capital=1000.0,
        position_size=1000.0,
        sl_pct=exp.params["stop_loss_pct"] / 100,
        trail_activation=exp.params["trail_activation_pct"] / 100,
        trail_pullback=exp.params["trail_pullback_pct"] / 100,
        **sim_kw,
    )
    print_diary(trades, blocked_info)


if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv) > 1 else "AAVE/USDT"
    main(asset)
