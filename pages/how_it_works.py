import streamlit as st
import plotly.graph_objects as go

st.title("How It Works")

st.markdown("""
### Using this app

This app has **three pages** — you can switch between them in the sidebar on the left:

| Page | What it does |
|---|---|
| **How It Works** *(this page)* | Explains the strategy step by step with illustrations. |
| **Experiments** | Compare all predefined strategies side-by-side for a given asset. Pick an experiment to understand the trade-offs. |
| **Backtest** | Dive into a single experiment. See the price chart, equity curve, trade diary, and detailed statistics. |

**On the Backtest page**, the sidebar gives you full control:

1. **Asset** — choose which coin to analyse (AAVE, BTC).
2. **Experiment** — select a preset (e.g. *Optimized PnL*). All sliders reset to that experiment's values.
3. **Sliders** — tweak any parameter (swing window, stop loss, trailing stop, etc.) to see the impact in real time.
4. **Checkboxes** — toggle optional filters (volume, RSI, candle direction), LONG-only mode, and range breakout exit.
5. **Recompute Backtest** — click this or simply move a slider to re-run the backtest with your changes.

The **Dashboard** tab shows KPIs, the interactive price chart, equity curve, and trade diary.
The **Statistics** tab shows P&L breakdown per trade and Long vs Short analysis.

---
""")

tab_range, tab_momentum = st.tabs(["Range Strategy", "Momentum Strategy"])

with tab_range:
    st.markdown("""
    ### The core idea: buy low, sell high — inside a box.

    Crypto prices don't always trend up or down. Sometimes they bounce sideways between
    a ceiling and a floor for days or weeks. This is called a **price range**
    (or "consolidation zone"). Our bot detects these zones automatically, then trades
    the bounces inside them.
    """)

    st.markdown("---")
    st.markdown("""
    ### Step 1 — Detect swing points

    A **swing high** is a candle whose price is the highest in the surrounding hours
    (controlled by the *swing window* parameter). Think of it as a local peak — the moment
    where price stopped going up and reversed. A **swing low** is the opposite — a local
    valley where price stopped falling and bounced back up.
    """)

    _prices1 = [100, 102, 105, 108, 106, 103, 100, 98, 96, 99, 102, 106, 109, 107, 104, 101, 98, 97, 99, 103]
    _highs1 = [3, 12]
    _lows1 = [8, 17]
    _fig1 = go.Figure()
    _fig1.add_trace(go.Scatter(y=_prices1, mode="lines", line=dict(color="#888", width=2), name="Price"))
    _fig1.add_trace(go.Scatter(
        x=_highs1, y=[_prices1[i] for i in _highs1], mode="markers",
        marker=dict(symbol="triangle-down", size=14, color="#F44336"), name="Swing High",
    ))
    _fig1.add_trace(go.Scatter(
        x=_lows1, y=[_prices1[i] for i in _lows1], mode="markers",
        marker=dict(symbol="triangle-up", size=14, color="#4CAF50"), name="Swing Low",
    ))
    _fig1.update_layout(
        height=250, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Time",
        yaxis_title="Price", legend=dict(orientation="h", y=1.12),
        xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(_fig1, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### Step 2 — Group swings into ranges

    When we find at least **2 swing highs** and **2 swing lows** close together (within
    the *range window*), and they all stay within ~2% of each other, we know price is
    bouncing between two levels. We draw a box:

    - **Top of box** = highest swing high (**resistance** — a ceiling).
    - **Bottom of box** = lowest swing low (**support** — a floor).

    A range **ends** when price breaks out — closing more than 2% past the edges.
    """)

    _prices2 = [100, 102, 104, 106, 108, 107, 105, 103, 101, 100, 102, 104, 107, 108, 106, 103, 101, 100, 102, 105, 108, 107, 105, 103, 100, 98, 95, 92]
    _fig2 = go.Figure()
    _fig2.add_trace(go.Scatter(y=_prices2, mode="lines", line=dict(color="#888", width=2), name="Price"))
    _fig2.add_shape(type="rect", x0=0, x1=23, y0=99.5, y1=108.5,
                    fillcolor="rgba(76,175,80,0.18)", line=dict(color="rgba(76,175,80,0.6)", width=2))
    _fig2.add_annotation(x=11, y=109.5, text="Range (support → resistance)", showarrow=False,
                         font=dict(size=12, color="#4CAF50"))
    _fig2.add_annotation(x=26, y=93, text="Breakout ↓", showarrow=False,
                         font=dict(size=11, color="#F44336"))
    _fig2.update_layout(
        height=250, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Time",
        yaxis_title="Price", showlegend=False, xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(_fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### Step 3 — Generate trade signals

    Once a range is confirmed, we watch for price to approach the edges:

    - **BUY (LONG)**: price within *signal proximity %* above the floor.
    - **SELL (SHORT)**: price within *signal proximity %* below the ceiling.
    """)

    _prices3 = [104, 106, 108, 106, 103, 101, 100, 102, 105, 107, 108, 106, 103, 100, 101, 104, 107, 108, 105, 102]
    _fig3 = go.Figure()
    _fig3.add_trace(go.Scatter(y=_prices3, mode="lines", line=dict(color="#888", width=2), name="Price"))
    _fig3.add_shape(type="rect", x0=0, x1=19, y0=99.5, y1=108.5,
                    fillcolor="rgba(76,175,80,0.12)", line=dict(color="rgba(76,175,80,0.4)", width=1))
    _fig3.add_shape(type="rect", x0=0, x1=19, y0=99.5, y1=101.5,
                    fillcolor="rgba(33,150,243,0.15)", line=dict(width=0))
    _fig3.add_shape(type="rect", x0=0, x1=19, y0=106.5, y1=108.5,
                    fillcolor="rgba(244,67,54,0.15)", line=dict(width=0))
    _fig3.add_annotation(x=10, y=100.2, text="BUY zone", showarrow=False,
                         font=dict(size=11, color="#2196F3"))
    _fig3.add_annotation(x=10, y=107.8, text="SELL zone", showarrow=False,
                         font=dict(size=11, color="#F44336"))
    _buy_idx3 = [6, 13]
    _sell_idx3 = [2, 10]
    _fig3.add_trace(go.Scatter(
        x=_buy_idx3, y=[_prices3[i] for i in _buy_idx3], mode="markers",
        marker=dict(symbol="triangle-up", size=14, color="#2196F3"), name="BUY signal",
    ))
    _fig3.add_trace(go.Scatter(
        x=_sell_idx3, y=[_prices3[i] for i in _sell_idx3], mode="markers",
        marker=dict(symbol="triangle-down", size=14, color="#F44336"), name="SELL signal",
    ))
    _fig3.update_layout(
        height=250, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Time",
        yaxis_title="Price", legend=dict(orientation="h", y=1.12),
        xaxis=dict(showticklabels=False),
    )
    st.plotly_chart(_fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### Step 4 — Filter bad signals

    **Momentum filter** — Skip if price moved more than *momentum threshold %* in the
    last *momentum lookback* candles against the trade direction.

    **Volatility filter** — Range must be between *range width min/max %* wide.

    **Optional** — Volume confirmation, candle direction, and RSI filters.

    ---

    ### Step 5 — Manage the trade

    **Stop loss** — Exit if price moves *stop loss %* against the position.

    **Trailing stop** — Once *trail activation %* profit is reached, track the peak.
    Exit if price pulls back *trail pullback %* from peak.

    **Range end** — Close at market if price breaks out of the range.
    """)

    _prices5 = [100, 101, 102.5, 103.5, 104.5, 105, 105.5, 104.8, 105.2, 106, 105.3, 104.5, 103.8]
    _fig5 = go.Figure()
    _fig5.add_trace(go.Scatter(y=_prices5, mode="lines+markers", line=dict(color="#888", width=2),
                               marker=dict(size=5), name="Price"))
    _fig5.add_hline(y=98, line_dash="dash", line_color="#F44336", opacity=0.7,
                    annotation_text="Stop Loss (−2%)", annotation_position="bottom left")
    _fig5.add_hline(y=103.5, line_dash="dot", line_color="#FF9800", opacity=0.7,
                    annotation_text="Trail activates (+3.5%)", annotation_position="top left")
    _fig5.add_trace(go.Scatter(x=[9], y=[106], mode="markers",
                               marker=dict(symbol="star", size=14, color="#FF9800"), name="Peak"))
    _trail_exit = 106 * (1 - 0.005)
    _fig5.add_hline(y=_trail_exit, line_dash="dashdot", line_color="#4CAF50", opacity=0.7,
                    annotation_text="Trailing stop exit", annotation_position="bottom right")
    _fig5.add_trace(go.Scatter(x=[0], y=[100], mode="markers",
                               marker=dict(symbol="triangle-up", size=14, color="#2196F3"), name="Entry"))
    _fig5.add_trace(go.Scatter(x=[12], y=[103.8], mode="markers",
                               marker=dict(symbol="x", size=14, color="#4CAF50"), name="Exit (trailing)"))
    _fig5.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Candles after entry",
        yaxis_title="Price", legend=dict(orientation="h", y=1.15),
        xaxis=dict(showticklabels=True),
    )
    st.plotly_chart(_fig5, use_container_width=True)

    st.markdown("""
    ---

    ### Step 6 — Circuit breaker

    Two consecutive stop losses on the same range → stop trading that range.

    ---

    ### How the parameters were chosen

    All the numbers above (swing window, trailing stop thresholds, stop loss, etc.) were
    **not chosen by hand**. We ran an automated **grid search optimization**:

    1. **Fetched 5 years** of historical price data
    2. **Skipped year 1** (2021) — too anomalous (DeFi bull run / crash), not representative
       of current market conditions
    3. **Split the remaining 4 years** into three periods:
       - **Train (years 2-3)** — the optimizer tests all parameter combinations here
       - **Validate (year 4)** — the top 50 from training are evaluated here
       - **Test (year 5)** — the best from validation is checked on this truly unseen data
    4. **Tested 209,952 combinations** of 11 parameters across the training data
    5. **Ranked by total P&L** with a minimum of 15 trades to avoid flukes
    6. **Forwarded the top 50** to the validation period, then picked the best performer
       on the final test period

    This three-way split prevents **overfitting** — the test data was never seen by any
    step of the selection process. If a parameter set performs well across train,
    validation, *and* test, it's likely capturing a real market pattern rather than
    fitting noise.

    As a final sanity check, we run the winning parameters on the skipped year 1 (2021)
    as a **stress test** — if the strategy survives that extreme period without
    catastrophic loss, it adds confidence in its robustness.
    """)

with tab_momentum:
    st.markdown("""
    ### The core idea: follow momentum shifts.

    Instead of waiting for a sideway range, this strategy tries to catch directional moves.
    It looks for momentum turning points and then rides the move with trailing exits.
    """)

    st.markdown("---")
    st.markdown("""
    ### Step 1 — Detect momentum flips (MACD histogram)

    We use the MACD histogram sign change:
    - Histogram goes **negative -> positive**: bullish momentum flip (LONG candidate)
    - Histogram goes **positive -> negative**: bearish momentum flip (SHORT candidate)
    """)

    _price_m1 = [100, 99, 98, 97, 98, 99, 100, 102, 104, 106, 107, 106, 104, 102]
    _macd_h =   [-1.2, -1.0, -0.8, -0.5, -0.2, 0.1, 0.4, 0.8, 1.1, 1.3, 1.0, 0.4, -0.1, -0.6]
    _figm1 = go.Figure()
    _figm1.add_trace(go.Scatter(y=_price_m1, mode="lines", line=dict(color="#888", width=2), name="Price", yaxis="y1"))
    _figm1.add_trace(go.Bar(y=_macd_h, marker_color=["#4CAF50" if v >= 0 else "#F44336" for v in _macd_h], name="MACD hist", yaxis="y2", opacity=0.5))
    _figm1.add_trace(go.Scatter(x=[5], y=[_price_m1[5]], mode="markers", marker=dict(symbol="triangle-up", size=12, color="#2196F3"), name="Bullish flip"))
    _figm1.add_trace(go.Scatter(x=[12], y=[_price_m1[12]], mode="markers", marker=dict(symbol="triangle-down", size=12, color="#F44336"), name="Bearish flip"))
    _figm1.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showticklabels=False),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="MACD hist", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(_figm1, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### Step 2 — Validate entry conditions

    Depending on the experiment preset, the strategy can require:
    - **Baseline mode**: engulfing candle + MACD flip + RSI filter
    - **Optimized mode**: MACD flip + RSI only (`macd_only`)

    RSI filter avoids entering when price is too stretched.
    """)

    _cand = [100, 99.2, 98.9, 98.7, 99.1, 99.8, 100.6, 101.1]
    _figm2 = go.Figure()
    _figm2.add_trace(go.Scatter(y=_cand, mode="lines+markers", line=dict(color="#888", width=2), marker=dict(size=5), name="Price"))
    _figm2.add_shape(type="rect", x0=4.5, x1=6.5, y0=98.6, y1=101.0,
                     fillcolor="rgba(33,150,243,0.15)", line=dict(color="#2196F3", width=1))
    _figm2.add_annotation(x=5.5, y=100.9, text="Entry window after flip", showarrow=False, font=dict(size=11, color="#2196F3"))
    _figm2.add_annotation(x=2, y=99.7, text="RSI in allowed band", showarrow=False, font=dict(size=11, color="#4CAF50"))
    _figm2.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(showticklabels=False), yaxis_title="Price")
    st.plotly_chart(_figm2, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### Step 3 — Manage the trend trade

    The momentum strategy uses:
    - **Stop loss** to cap downside
    - **Trailing stop** to lock profit after trend extension
    - No range-end logic (because this strategy is not range-based)
    """)

    _pm = [100, 101, 102, 103, 104, 106, 109, 111, 110, 109, 107.5, 106.8]
    _figm3 = go.Figure()
    _figm3.add_trace(go.Scatter(y=_pm, mode="lines+markers", line=dict(color="#888", width=2), marker=dict(size=5), name="Price"))
    _figm3.add_trace(go.Scatter(x=[0], y=[100], mode="markers", marker=dict(symbol="triangle-up", size=14, color="#2196F3"), name="Entry"))
    _figm3.add_trace(go.Scatter(x=[7], y=[111], mode="markers", marker=dict(symbol="star", size=14, color="#FF9800"), name="Peak"))
    _figm3.add_hline(y=96, line_dash="dash", line_color="#F44336", annotation_text="Stop loss", annotation_position="bottom left")
    _figm3.add_hline(y=106.5, line_dash="dashdot", line_color="#4CAF50", annotation_text="Trailing exit", annotation_position="bottom right")
    _figm3.add_trace(go.Scatter(x=[11], y=[106.8], mode="markers", marker=dict(symbol="x", size=14, color="#4CAF50"), name="Exit"))
    _figm3.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Candles after entry", yaxis_title="Price", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(_figm3, use_container_width=True)

    st.markdown("""
    ---

    ### How momentum parameters are tuned

    We optimize momentum experiments with walk-forward validation too:
    - train on past years
    - validate on next year
    - test on unseen final year

    For momentum, key levers are usually:
    - entry mode (`engulfing`, `strong_candle`, `macd_only`)
    - stop loss
    - trail activation / pullback
    - signal cooldown
    - RSI and ADX thresholds
    """)
