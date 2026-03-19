import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from data import fetch_ohlcv
from simulator import simulate
from strategies.base import SHARED_PARAMS
from strategies.registry import EXPERIMENTS, STRATEGIES, Experiment

CAPITAL_EUR = 1000.0
POSITION_SIZE_EUR = 1000.0
UI_ASSETS = ["AAVE/USDT"]
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "cache" / "experiment_summaries.csv"
SNAPSHOT_COLUMNS = [
    "key",
    "asset",
    "experiment_name",
    "strategy",
    "params_hash",
    "pnl",
    "return_pct",
    "win_rate",
    "trades",
    "updated_at",
]


def _params_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _exp_key(asset: str, exp: Experiment) -> str:
    return f"{asset}|{exp.name}|{exp.strategy}|{_params_hash(exp.params)}"


def _load_snapshots() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    df = pd.read_csv(SNAPSHOT_PATH)
    for col in SNAPSHOT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SNAPSHOT_COLUMNS]


def _save_snapshots(df: pd.DataFrame) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SNAPSHOT_PATH, index=False)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Experiments Overview")

selected_coin = st.selectbox("Asset", UI_ASSETS)
experiments: list[Experiment] = EXPERIMENTS[selected_coin]

st.caption("Table uses snapshot metrics for fast loading. Backtest page remains fully dynamic.")


@st.cache_data(ttl=3600)
def load_data(symbol: str):
    return fetch_ohlcv(symbol=symbol)


@st.cache_data(ttl=3600)
def _run_summary(symbol: str, strategy_key: str, params_tuple: tuple) -> dict:
    df = load_data(symbol)
    params = dict(params_tuple)
    strategy_obj = STRATEGIES[strategy_key]

    signals, metadata = strategy_obj.generate_signals(df, params)
    sim_kw = {**strategy_obj.simulator_overrides, **metadata.get("simulator_overrides", {})}
    tds, _ = simulate(
        df, signals, metadata.get("ranges", []),
        capital=CAPITAL_EUR,
        position_size=POSITION_SIZE_EUR,
        sl_pct=params["stop_loss_pct"] / 100,
        trail_activation=params["trail_activation_pct"] / 100,
        trail_pullback=params["trail_pullback_pct"] / 100,
        **sim_kw,
    )

    n = len(tds)
    pnl = sum(t["pnl_eur"] for t in tds)
    w = len([t for t in tds if t["pnl_eur"] > 0])
    return {
        "trades": n,
        "win_rate": round(w / n * 100, 1) if n else 0,
        "pnl": round(pnl, 2),
        "return_pct": round(pnl / CAPITAL_EUR * 100, 1),
    }


def _refresh_asset_snapshots(asset: str, exps: list[Experiment]) -> pd.DataFrame:
    store = _load_snapshots()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for exp in exps:
        summary = _run_summary(asset, exp.strategy, tuple(sorted(exp.params.items())))
        rows.append({
            "key": _exp_key(asset, exp),
            "asset": asset,
            "experiment_name": exp.name,
            "strategy": exp.strategy,
            "params_hash": _params_hash(exp.params),
            "pnl": summary["pnl"],
            "return_pct": summary["return_pct"],
            "win_rate": summary["win_rate"],
            "trades": summary["trades"],
            "updated_at": now,
        })

    fresh = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    keep_other_assets = store[store["asset"] != asset] if not store.empty else pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    merged = pd.concat([keep_other_assets, fresh], ignore_index=True)
    _save_snapshots(merged)
    return merged


def _badge(exp: Experiment) -> str:
    return "🟢" if exp.primary else "🟠"


def _backtest_href(asset: str, exp_name: str) -> str:
    return f"/backtest?asset={quote_plus(asset)}&experiment={quote_plus(exp_name)}"


left, _right = st.columns([3, 1])
with left:
    refresh_clicked = st.button("Refresh table results", type="secondary")

if refresh_clicked:
    with st.spinner(f"Refreshing {len(experiments)} experiments for {selected_coin}..."):
        snapshot_store = _refresh_asset_snapshots(selected_coin, experiments)
    st.success("Experiment table refreshed.")
else:
    snapshot_store = _load_snapshots()

asset_snapshots = snapshot_store[snapshot_store["asset"] == selected_coin] if not snapshot_store.empty else pd.DataFrame(columns=SNAPSHOT_COLUMNS)
snap_by_key = {row["key"]: row for _, row in asset_snapshots.iterrows()}

# ---------------------------------------------------------------------------
# Helpers for strategy sections
# ---------------------------------------------------------------------------

def _summary_table(exps: list[Experiment], title: str) -> None:
    st.markdown(f"### {title}")
    rows = []
    adv_rows = []
    for i, exp in enumerate(exps, start=1):
        snap = snap_by_key.get(_exp_key(selected_coin, exp))
        if snap is None:
            pnl_txt = "—"
            ret_txt = "—"
            win_txt = "—"
            trades_txt = "—"
            updated_txt = "Not computed"
        else:
            pnl_txt = f"{float(snap['pnl']):+.2f}"
            ret_txt = f"{float(snap['return_pct']):+.1f}%"
            win_txt = f"{float(snap['win_rate']):.0f}%"
            trades_txt = str(int(snap["trades"]))
            updated_txt = str(snap["updated_at"])

        rows.append({
            "#": i,
            "Experiment": f'{_badge(exp)} <a href="{_backtest_href(selected_coin, exp.name)}">{exp.name}</a>',
            "P&L (€)": pnl_txt,
            "Return on €1,000": ret_txt,
            "Win Rate": win_txt,
            "Trades": trades_txt,
        })
        adv_rows.append({
            "#": i,
            "Experiment": f'{_badge(exp)} <a href="{_backtest_href(selected_coin, exp.name)}">{exp.name}</a>',
            "Updated": updated_txt,
        })

    if rows:
        st.markdown(pd.DataFrame(rows).to_html(escape=False, index=False), unsafe_allow_html=True)
        with st.expander("Show update timestamps"):
            st.markdown(pd.DataFrame(adv_rows).to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No experiments in this group.")


def _param_table(strategy_key: str, exps: list[Experiment], header: str) -> None:
    strat_obj = STRATEGIES[strategy_key]
    if not exps:
        return

    schema = {**strat_obj.param_schema, **SHARED_PARAMS}
    st.markdown(f"#### {header}")
    param_rows = []
    for exp in exps:
        row = {"Experiment": f"{_badge(exp)} {exp.name}"}
        for key, pdef in schema.items():
            label = pdef.label or key
            val = exp.params.get(key, pdef.default)
            if pdef.type == "bool":
                row[label] = "On" if val else "Off"
            else:
                row[label] = val
        param_rows.append(row)

    pdf = pd.DataFrame(param_rows)
    pdf.index = range(1, len(pdf) + 1)
    st.dataframe(pdf, use_container_width=True, hide_index=True)


def _details(exps: list[Experiment]) -> None:
    st.markdown("---")
    st.markdown("### Experiment Details")
    if not exps:
        st.info("No experiments in this group.")
        return
    for exp in exps:
        headline, body = exp.description
        if not headline:
            headline = "Custom experiment"
        if not body:
            body = "No description available."
        with st.expander(f"{_badge(exp)} **{exp.name}** — {headline}"):
            st.markdown(body)


range_exps = [e for e in experiments if e.strategy == "range"]
momentum_exps = [e for e in experiments if e.strategy == "momentum"]

tab_range, tab_momentum = st.tabs(["Range", "Momentum"])

with tab_range:
    _summary_table(range_exps, "Range Results")
    st.caption("🟢 Validated  ·  🟠 Alternative")
    _param_table("range", range_exps, "Range strategy parameters")
    _details(range_exps)

with tab_momentum:
    _summary_table(momentum_exps, "Momentum Results")
    st.caption("🟢 Validated  ·  🟠 Alternative")
    _param_table("momentum", momentum_exps, "Momentum strategy parameters")
    _details(momentum_exps)

st.caption("Head to Backtest to explore each experiment in detail.")
