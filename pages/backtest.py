import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data import fetch_ohlcv, data_source
from simulator import simulate
from strategies.base import SHARED_PARAMS
from strategies.registry import EXPERIMENTS, STRATEGIES

CAPITAL_EUR = 1000.0
POSITION_SIZE_EUR = 1000.0
UI_ASSETS = ["AAVE/USDT"]

EXIT_LABELS = {
    "SL": "Stop Loss",
    "TRAILING_STOP": "Trailing Stop",
    "RANGE_END": "Range Ended",
}


def _fmt_ts(ts):
    return str(ts)[:16]


# ---------------------------------------------------------------------------
# Sidebar — asset & experiment
# ---------------------------------------------------------------------------

query_asset = st.query_params.get("asset")
default_asset_idx = UI_ASSETS.index(query_asset) if query_asset in UI_ASSETS else 0
selected_coin = st.sidebar.selectbox("Asset", UI_ASSETS, index=default_asset_idx)

experiments = EXPERIMENTS[selected_coin]
exp_names = [e.name for e in experiments]
query_experiment = st.query_params.get("experiment")
default_exp_idx = exp_names.index(query_experiment) if query_experiment in exp_names else 0
selected_exp_name = st.sidebar.selectbox("Experiment", exp_names, index=default_exp_idx)
exp = next(e for e in experiments if e.name == selected_exp_name)

# Keep URL in sync so deep links to a specific backtest stay stable.
st.query_params["asset"] = selected_coin
st.query_params["experiment"] = selected_exp_name

strategy_obj = STRATEGIES[exp.strategy]

if st.session_state.get("_active_exp") != (selected_coin, selected_exp_name):
    st.session_state["_active_exp"] = (selected_coin, selected_exp_name)
    st.session_state["_exp_v"] = st.session_state.get("_exp_v", 0) + 1
v = st.session_state["_exp_v"]

# ---------------------------------------------------------------------------
# Sidebar — dynamic parameter sliders from param_schema
# ---------------------------------------------------------------------------

param_values: dict = {}


def _render_params(schema: dict, header: str, defaults: dict) -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader(header)
    for key, pdef in schema.items():
        label = pdef.label or key
        default = defaults.get(key, pdef.default)
        uid = f"{key}_{v}"

        if pdef.type == "float":
            param_values[key] = st.sidebar.slider(
                label, float(pdef.min), float(pdef.max), float(default), float(pdef.step), key=uid,
            )
        elif pdef.type == "int":
            param_values[key] = st.sidebar.slider(
                label, int(pdef.min), int(pdef.max), int(default), int(pdef.step), key=uid,
            )
        elif pdef.type == "bool":
            param_values[key] = st.sidebar.checkbox(label, bool(default), key=uid)
        elif pdef.type == "select":
            idx = pdef.options.index(default) if default in pdef.options else 0
            param_values[key] = st.sidebar.selectbox(label, pdef.options, index=idx, key=uid)


_render_params(strategy_obj.param_schema, f"{strategy_obj.name} Parameters", exp.params)
_render_params(SHARED_PARAMS, "Trade Management", exp.params)

# ---------------------------------------------------------------------------
# Main content — tabs
# ---------------------------------------------------------------------------

run_clicked = st.sidebar.button("Recompute Backtest", type="primary", use_container_width=True)

st.markdown("""
<style>
.hero-box {
  border: 1px solid rgba(120,120,120,0.25);
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

status_txt = "Validated" if exp.primary else "Alternative"
goal_txt = exp.description[0] if exp.description and exp.description[0] else "Strategy preset"
st.title(f"{selected_coin} Backtest")
st.markdown(
    f"""
    <div class="hero-box">
      <strong>{exp.name}</strong> · {status_txt}<br/>
      Goal: {goal_txt}. Uses 1H candles with €{POSITION_SIZE_EUR:.0f} per trade on €{CAPITAL_EUR:.0f} capital.
    </div>
    """,
    unsafe_allow_html=True,
)
tab_dashboard, tab_stats = st.tabs(["Dashboard", "Statistics"])


@st.cache_data(ttl=3600)
def load_data(symbol: str):
    return fetch_ohlcv(symbol=symbol)


# ---------------------------------------------------------------------------
# Run strategy
# ---------------------------------------------------------------------------

current_params = (selected_coin, exp.strategy, tuple(sorted(param_values.items())))
params_changed = st.session_state.get("last_params") != current_params
st.session_state["last_params"] = current_params

if run_clicked or params_changed or "results" not in st.session_state:
    with st.spinner("Running strategy..."):
        df = load_data(selected_coin)
        if data_source(selected_coin) == "seed":
            st.info("Live data unavailable from this server. Showing pre-loaded data (last refreshed Mar 19 2026).", icon="ℹ️")
        signals, metadata = strategy_obj.generate_signals(df, param_values)

        sim_kw = {**strategy_obj.simulator_overrides, **metadata.get("simulator_overrides", {})}
        trades, blocked_info = simulate(
            df, signals, metadata.get("ranges", []),
            capital=CAPITAL_EUR,
            position_size=POSITION_SIZE_EUR,
            sl_pct=param_values["stop_loss_pct"] / 100,
            trail_activation=param_values["trail_activation_pct"] / 100,
            trail_pullback=param_values["trail_pullback_pct"] / 100,
            **sim_kw,
        )
        st.session_state["results"] = dict(
            df=df, strategy=exp.strategy,
            detected_ranges=metadata.get("ranges", []),
            entry_signals=signals,
            trades=trades, blocked_info=blocked_info,
            momentum_rejected=metadata.get("momentum_rejected", 0),
            volatility_rejected=metadata.get("volatility_rejected", 0),
            exp_label=selected_exp_name,
        )

res = st.session_state["results"]
df = res["df"]
detected_ranges = res["detected_ranges"]
entry_signals = res["entry_signals"]
trades = res["trades"]
momentum_rejected = res["momentum_rejected"]
volatility_rejected = res["volatility_rejected"]

trade_df = pd.DataFrame(trades) if trades else pd.DataFrame()
total_trades = len(trades)
wins = [t for t in trades if t["pnl_eur"] > 0]
losses = [t for t in trades if t["pnl_eur"] <= 0]
win_rate = len(wins) / total_trades * 100 if total_trades else 0
total_pnl = sum(t["pnl_eur"] for t in trades)

# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------

with tab_dashboard:
    capital = CAPITAL_EUR
    return_pct = (total_pnl / capital) * 100 if capital else 0

    st.markdown(f"### Results — {res['exp_label']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", total_trades)
    c2.metric("Win Rate", f"{win_rate:.0f}%")
    c3.metric("Total P&L", f"€{total_pnl:+.2f}")
    c4.metric("Return on €1,000", f"{return_pct:+.1f}%")
    st.caption(f"€{POSITION_SIZE_EUR:.0f} per trade, €{CAPITAL_EUR:.0f} capital.")

    n_long = len([s for s in entry_signals if s["direction"] == "LONG"])
    n_short = len([s for s in entry_signals if s["direction"] == "SHORT"])
    if strategy_obj.uses_ranges:
        st.markdown(
            f"*{len(detected_ranges)} ranges detected · "
            f"{n_long} LONG / {n_short} SHORT signals · "
            f"Rejected: {momentum_rejected} momentum, {volatility_rejected} volatility*"
        )
    else:
        st.markdown(f"*{n_long} LONG / {n_short} SHORT signals ({strategy_obj.name})*")

    st.markdown("### Price Chart")

    import pandas as _pd

    date_min = df["timestamp"].min().date()
    date_max = df["timestamp"].max().date()

    _RANGES = ["All", "Last 7 days", "Last 30 days", "Last 90 days", "Last 6 months", "Custom"]
    range_choice = st.selectbox("Date range", _RANGES, index=0, key="date_range")

    if range_choice == "Custom":
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            chart_start = st.date_input("From", value=date_min, min_value=date_min, max_value=date_max, key="chart_from")
        with dcol2:
            chart_end = st.date_input("To", value=date_max, min_value=date_min, max_value=date_max, key="chart_to")
        ts_start = _pd.Timestamp(chart_start, tz="UTC")
        ts_end = _pd.Timestamp(chart_end, tz="UTC") + _pd.Timedelta(days=1)
    elif range_choice == "All":
        ts_start = _pd.Timestamp(date_min, tz="UTC")
        ts_end = _pd.Timestamp(date_max, tz="UTC") + _pd.Timedelta(days=1)
    else:
        days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "Last 6 months": 182}
        ts_end = _pd.Timestamp(date_max, tz="UTC") + _pd.Timedelta(days=1)
        ts_start = ts_end - _pd.Timedelta(days=days_map[range_choice])

    chart_df = df[(df["timestamp"] >= ts_start) & (df["timestamp"] < ts_end)]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_df["timestamp"], open=chart_df["open"], high=chart_df["high"],
            low=chart_df["low"], close=chart_df["close"], name=selected_coin,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        )
    )

    for r in detected_ranges:
        r_start = _pd.Timestamp(r["start_time"]).tz_localize("UTC") if _pd.Timestamp(r["start_time"]).tzinfo is None else _pd.Timestamp(r["start_time"])
        r_end = _pd.Timestamp(r["end_time"]).tz_localize("UTC") if _pd.Timestamp(r["end_time"]).tzinfo is None else _pd.Timestamp(r["end_time"])
        if r_end < ts_start or r_start >= ts_end:
            continue
        fig.add_shape(
            type="rect",
            x0=r["start_time"], x1=r["end_time"],
            y0=r["range_low"], y1=r["range_high"],
            fillcolor="rgba(76,175,80,0.15)",
            line=dict(color="rgba(76,175,80,0.5)", width=1),
        )

    def _in_range(ts):
        t = _pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return ts_start <= t < ts_end

    longs = [s for s in entry_signals if s["direction"] == "LONG" and _in_range(s["timestamp"])]
    shorts = [s for s in entry_signals if s["direction"] == "SHORT" and _in_range(s["timestamp"])]

    if longs:
        fig.add_trace(go.Scatter(
            x=[s["timestamp"] for s in longs],
            y=[s["close_price"] for s in longs],
            mode="markers", name=f"BUY ({len(longs)})",
            marker=dict(symbol="triangle-up", size=12, color="#2196F3"),
        ))

    if shorts:
        fig.add_trace(go.Scatter(
            x=[s["timestamp"] for s in shorts],
            y=[s["close_price"] for s in shorts],
            mode="markers", name=f"SELL ({len(shorts)})",
            marker=dict(symbol="triangle-down", size=12, color="#F44336"),
        ))

    for t in trades:
        t_entry = _pd.Timestamp(t["entry_time"]).tz_localize("UTC") if _pd.Timestamp(t["entry_time"]).tzinfo is None else _pd.Timestamp(t["entry_time"])
        t_exit = _pd.Timestamp(t["exit_time"]).tz_localize("UTC") if _pd.Timestamp(t["exit_time"]).tzinfo is None else _pd.Timestamp(t["exit_time"])
        if t_exit < ts_start or t_entry >= ts_end:
            continue
        color = "#4CAF50" if t["pnl_eur"] > 0 else "#F44336"
        fig.add_trace(go.Scatter(
            x=[t["entry_time"], t["exit_time"]],
            y=[t["entry_price"], t["exit_price"]],
            mode="lines", line=dict(color=color, width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        height=600, xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
        yaxis_title="Price (USDT)",
    )
    st.plotly_chart(fig, use_container_width=True)

    if trades:
        st.markdown("### Equity Curve")

        eq_data = []
        cumul = 0.0
        for t in trades:
            cumul += t["pnl_eur"]
            eq_data.append({"time": t["exit_time"], "cumulative_pnl": cumul})
        eq_df = pd.DataFrame(eq_data)

        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(
            x=eq_df["time"], y=eq_df["cumulative_pnl"],
            mode="lines+markers", name="Cumulative P&L",
            line=dict(color="#2196F3", width=2),
            marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
        ))
        eq_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        eq_fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Cumulative P&L (€)", hovermode="x unified",
        )
        st.plotly_chart(eq_fig, use_container_width=True)

    if not trade_df.empty:
        st.markdown("### Trade Diary")
        display_df = trade_df.copy()
        display_df["entry_time"] = display_df["entry_time"].apply(_fmt_ts)
        display_df["exit_time"] = display_df["exit_time"].apply(_fmt_ts)
        display_df["exit_reason"] = display_df["exit_reason"].map(EXIT_LABELS).fillna(display_df["exit_reason"])

        compact_df = display_df[[
            "direction", "entry_time", "exit_time", "pnl_eur", "pnl_pct", "exit_reason",
        ]].copy()
        compact_df.columns = ["Dir", "Entry Time", "Exit Time", "P&L €", "P&L %", "Exit Reason"]
        st.dataframe(
            compact_df.style.format({"P&L €": "{:+.2f}", "P&L %": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Show advanced trade columns"):
            advanced_df = display_df[[
                "direction", "entry_time", "exit_time",
                "entry_price", "exit_price", "pnl_eur", "pnl_pct",
                "peak_pct", "candles_held", "exit_reason",
            ]].copy()
            advanced_df.columns = [
                "Dir", "Entry Time", "Exit Time", "Entry $", "Exit $",
                "P&L €", "P&L %", "Best Profit %", "Hours Held", "Exit Reason",
            ]
            st.dataframe(
                advanced_df.style.format({
                    "Entry $": "{:.2f}", "Exit $": "{:.2f}",
                    "P&L €": "{:+.2f}", "P&L %": "{:+.2f}", "Best Profit %": "{:+.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    if not trades:
        st.info("No trades generated with current parameters.")

# ---------------------------------------------------------------------------
# Statistics tab
# ---------------------------------------------------------------------------

with tab_stats:
    capital = CAPITAL_EUR
    return_pct = (total_pnl / capital) * 100 if capital else 0

    st.markdown(f"### Results — {res['exp_label']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", total_trades)
    c2.metric("Win Rate", f"{win_rate:.0f}%")
    c3.metric("Total P&L", f"€{total_pnl:+.2f}")
    c4.metric("Return on €1,000", f"{return_pct:+.1f}%")

    if trades:
        st.markdown("### P&L Breakdown")
        left, right = st.columns(2)

        with left:
            colors = ["#4CAF50" if t["pnl_eur"] > 0 else "#F44336" for t in trades]
            bar_fig = go.Figure(go.Bar(
                x=list(range(1, len(trades) + 1)),
                y=[t["pnl_eur"] for t in trades],
                marker_color=colors,
                hovertext=[
                    f"{t['direction']} | {EXIT_LABELS.get(t['exit_reason'], t['exit_reason'])}<br>"
                    f"€{t['pnl_eur']:+.2f} ({t['pnl_pct']:+.2f}%)"
                    for t in trades
                ],
                hoverinfo="text",
            ))
            bar_fig.add_hline(y=0, line_color="gray", opacity=0.3)
            bar_fig.update_layout(
                title="P&L per Trade",
                height=350, margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Trade #", yaxis_title="P&L (€)",
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        with right:
            reasons = {}
            for t in trades:
                label = EXIT_LABELS.get(t["exit_reason"], t["exit_reason"])
                reasons[label] = reasons.get(label, 0) + 1
            reason_colors = {
                "Stop Loss": "#F44336", "Trailing Stop": "#4CAF50",
                "Range Ended": "#2196F3",
            }
            pie_fig = go.Figure(go.Pie(
                labels=list(reasons.keys()),
                values=list(reasons.values()),
                marker_colors=[reason_colors.get(r, "#9E9E9E") for r in reasons],
                hole=0.4, textinfo="label+value",
            ))
            pie_fig.update_layout(
                title="Exit Reasons",
                height=350, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        st.markdown("### Direction & Outcome Stats")

        long_trades = [t for t in trades if t["direction"] == "LONG"]
        short_trades = [t for t in trades if t["direction"] == "SHORT"]
        win_trades = [t for t in trades if t["pnl_pct"] > 0]
        loss_trades = [t for t in trades if t["pnl_pct"] <= 0]

        def _avg_pct(trade_list):
            if not trade_list:
                return 0.0
            return sum(t["pnl_pct"] for t in trade_list) / len(trade_list)

        avg_long = _avg_pct(long_trades)
        avg_short = _avg_pct(short_trades)
        avg_win = _avg_pct(win_trades)
        avg_loss = _avg_pct(loss_trades)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Return LONG", f"{avg_long:+.2f}%")
        m2.metric("Avg Return SHORT", f"{avg_short:+.2f}%")
        m3.metric("Avg Return WIN", f"{avg_win:+.2f}%")
        m4.metric("Avg Return LOSS", f"{avg_loss:+.2f}%")

        def _row(label, trade_list):
            n = len(trade_list)
            wins_n = len([t for t in trade_list if t["pnl_eur"] > 0])
            pnl = sum(t["pnl_eur"] for t in trade_list) if n else 0.0
            return {
                "Group": label,
                "Trades": n,
                "Win Rate": f"{(wins_n / n * 100):.0f}%" if n else "0%",
                "Avg Return %": f"{_avg_pct(trade_list):+.2f}%",
                "Total P&L €": f"{pnl:+.2f}",
            }

        stats_df = pd.DataFrame([
            _row("LONG", long_trades),
            _row("SHORT", short_trades),
            _row("WIN", win_trades),
            _row("LOSS", loss_trades),
        ])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades generated with current parameters.")
