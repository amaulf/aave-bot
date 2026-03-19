import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_chart(df: pd.DataFrame, swing_highs: list[int], swing_lows: list[int], ranges: list[dict], signals: list[dict] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(df["timestamp"], df["close"], linewidth=0.8, color="#1f77b4", label="AAVE/USDT Close")

    ax.scatter(
        df["timestamp"].iloc[swing_highs],
        df["high"].iloc[swing_highs],
        color="red", s=30, zorder=5, label=f"Swing Highs ({len(swing_highs)})",
    )
    ax.scatter(
        df["timestamp"].iloc[swing_lows],
        df["low"].iloc[swing_lows],
        color="green", s=30, zorder=5, label=f"Swing Lows ({len(swing_lows)})",
    )

    for r in ranges:
        mask = (df["timestamp"] >= r["start_time"]) & (df["timestamp"] <= r["end_time"])
        ts = df.loc[mask, "timestamp"]
        if ts.empty:
            continue
        ax.fill_between(
            ts, r["range_low"], r["range_high"],
            alpha=0.25, color="green", edgecolor="green", linewidth=0.8,
        )

    if signals:
        longs = [s for s in signals if s.get("direction") == "LONG"]
        shorts = [s for s in signals if s.get("direction") == "SHORT"]
        if longs:
            ax.scatter([s["timestamp"] for s in longs], [s["close_price"] for s in longs],
                       marker="^", s=120, color="blue", zorder=6, label=f"BUY Signals ({len(longs)})")
        if shorts:
            ax.scatter([s["timestamp"] for s in shorts], [s["close_price"] for s in shorts],
                       marker="v", s=120, color="red", zorder=6, label=f"SELL Signals ({len(shorts)})")

    ax.set_title("AAVE/USDT – 1H (90 days) with Detected Ranges", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
