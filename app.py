"""
app.py – Streamlit Dashboard for Smart Energy Management Platform
──────────────────────────────────────────────────────────────────
Visual gauges and charts for:
  • Live Power Flow    (Solar Generation vs Consumer Demand vs Supply MW)
  • Battery SoC %      (Gauge + trend)
  • Grid Status banner (SURPLUS / BALANCED / DEFICIT / CRITICAL)
  • Outage Risk Map    (Risk score gauge + signal list)
  • 24-hour Solar Forecast Chart
  • Agent Reasoning Logs (expandable)

Run:
  streamlit run app.py

The dashboard polls the FastAPI /api/v1/simulate endpoint every N seconds
(configurable via the sidebar). When the backend is offline it renders with
locally generated demo data so the UI is always usable.
"""

from __future__ import annotations

import math
import random
import time
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gujarat Smart Grid – Agentic AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS injection
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Compact metric cards */
    [data-testid="metric-container"] {
        background: #f7f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 18px;
    }
    /* Status badges */
    .badge-surplus  { background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-balanced { background:#dbeafe; color:#1e40af; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-deficit  { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:20px; font-weight:600; }
    .badge-critical { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:20px; font-weight:600; }
    /* Risk badge */
    .risk-normal   { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:15px; font-size:0.85em; }
    .risk-watch    { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:15px; font-size:0.85em; }
    .risk-alert    { background:#fed7aa; color:#9a3412; padding:3px 10px; border-radius:15px; font-size:0.85em; }
    .risk-critical { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:15px; font-size:0.85em; }
    /* Panel divider */
    .section-title { font-size:1.05rem; font-weight:700; color:#1e293b; margin-bottom:6px; }
    hr.thin { border:none; border-top:1px solid #e2e8f0; margin:14px 0; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Configuration
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQbuqfcqosqrl5lh_32asS_xzSVQ7cp1YBbsfq3hnlhlw&s=10",
        width=100,
    )
    st.markdown("## ⚡ Gujarat Smart Grid")
    st.markdown("*Agentic AI – IBM Granite LLM*")
    st.divider()

    api_base = st.text_input(
        "FastAPI Base URL",
        value="http://localhost:8000",
        help="URL of the running FastAPI backend",
    )
    refresh_interval = st.slider(
        "Auto-refresh interval (s)", min_value=5, max_value=120, value=30, step=5
    )
    manual_hour = st.slider(
        "Simulate IST Hour", min_value=0, max_value=23, value=datetime.now().hour
    )

    col_a, col_b = st.columns(2)
    with col_a:
        run_now = st.button("🔄 Refresh Now", use_container_width=True)
    with col_b:
        demo_mode = st.toggle("Demo Mode", value=True,
                              help="Use local data when backend is unavailable")

    st.divider()
    st.markdown("**Agents Active**")
    st.markdown("🌞 Solar Forecast Agent")
    st.markdown("⚖️ Demand Balance Agent")
    st.markdown("🔋 Battery Optimizer Agent")
    st.markdown("🚨 Outage Predictor Agent")
    st.markdown("📊 Dashboard Aggregator")
    st.divider()
    st.caption("IBM Watsonx · Granite-3-8B-Instruct")


# ──────────────────────────────────────────────────────────────────────────────
# Data fetching helpers
# ──────────────────────────────────────────────────────────────────────────────

def _local_demo(hour: int) -> dict:
    """Generate plausible demo data without any backend call."""
    solar_output  = round(max(0.0, 3800 * math.sin(math.pi * max(0, hour - 6) / 12) + random.gauss(0, 50)), 1)
    demand        = round(1200 + 700 * abs(math.sin(math.pi * (hour - 5) / 13)) + random.gauss(0, 30), 1)
    battery_soc   = round(random.uniform(30, 85), 1)
    gap           = round(solar_output - demand, 1)
    risk_score    = round(random.uniform(10, 75), 1)
    risk_level    = (
        "CRITICAL" if risk_score > 80 else
        "ALERT"    if risk_score > 60 else
        "WATCH"    if risk_score > 30 else "NORMAL"
    )
    grid_status = (
        "SURPLUS"  if gap > demand * 0.05 else
        "DEFICIT"  if gap < -demand * 0.05 else "BALANCED"
    )
    forecast_24h = [
        round(max(0.0, 3800 * math.sin(math.pi * max(0, (hour + i) % 24 - 6) / 12) + random.gauss(0, 80)), 1)
        for i in range(1, 25)
    ]
    return {
        "kpis": {
            "solar_output_mw":   solar_output,
            "demand_mw":         demand,
            "supply_mw":         solar_output + max(0, battery_soc / 100 * 200),
            "gap_mw":            gap,
            "grid_status":       grid_status,
            "battery_soc_pct":   battery_soc,
            "battery_action":    random.choice(["CHARGE", "DISCHARGE", "HOLD"]),
            "battery_power_mw":  round(random.uniform(10, 80), 1),
            "outage_risk_score": risk_score,
            "outage_risk_level": risk_level,
            "solar_anomaly":     random.random() < 0.15,
            "confidence_pct":    round(random.uniform(70, 95), 1),
        },
        "solar_forecast": {
            "current_output_mw": solar_output,
            "forecast_24h_mw":   forecast_24h,
            "anomaly_detected":  random.random() < 0.15,
            "anomaly_description": "Sudden cloud burst detected during peak window." if random.random() < 0.15 else "None",
            "agent_reasoning": (
                "Gujarat solar output is tracking expected insolation curves. "
                "Cloud cover is moderate and no significant anomalies are present. "
                "Battery reserves are sufficient for evening ramp support."
            ),
        },
        "demand_balance": {
            "realtime_demand_mw":   demand,
            "total_supply_mw":      solar_output + 150,
            "supply_demand_gap_mw": gap,
            "grid_status":          grid_status,
            "dispatch_commands": [
                {
                    "priority": 1,
                    "source":   "Battery",
                    "action":   f"Discharge 80 MW from battery storage",
                    "mw_adjustment": 80,
                    "reason":   "Cover evening demand peak using pre-charged battery.",
                }
            ],
            "agent_reasoning": (
                "Current demand exceeds solar by a moderate margin. "
                "Battery discharge covers the deficit cleanly. "
                "Monitor feeder loading for the next 30 minutes."
            ),
        },
        "battery_status": {
            "action":                  "DISCHARGE" if gap < 0 else "CHARGE",
            "power_setpoint_mw":       round(abs(gap) * 0.6, 1),
            "projected_soc_after_pct": round(battery_soc - 5 if gap < 0 else battery_soc + 3, 1),
            "degradation_risk":        "LOW",
            "estimated_roi_rs":        round(random.uniform(5000, 50000), 0),
            "safety_override":         False,
            "agent_reasoning": (
                "Battery operation is within optimal SoC range. "
                "Arbitrage conditions are favourable at current tariff levels. "
                "No degradation risk identified in this cycle."
            ),
        },
        "outage_risk": {
            "asset_id":       "GRID-TX-01",
            "risk_score":     risk_score,
            "risk_level":     risk_level,
            "predicted_failure_window_h": 3.5 if risk_score > 60 else None,
            "contributing_factors": [
                f"Transformer temp 88.4°C approaching caution limit",
                f"Line loading 84.2% – thermal overload risk",
            ] if risk_score > 50 else [],
            "preventive_signals": [
                {
                    "signal_type": "ALERT",
                    "target":      "Grid Operations Centre",
                    "message":     "Asset GRID-TX-01 at WATCH status. Increase monitoring.",
                    "urgency":     "WITHIN_1H",
                }
            ] if risk_score > 30 else [],
            "agent_reasoning": (
                "Transformer temperature is elevated but within caution range. "
                "Line loading is approaching the 85% threshold; proactive deloading recommended. "
                "Weather conditions are clear and not compounding risk at this time."
            ),
        },
    }


@st.cache_data(ttl=25)
def _fetch_data(api_base: str, hour: int) -> tuple[dict, bool]:
    """
    Try to fetch from FastAPI backend.
    Returns (data_dict, is_live).
    """
    try:
        resp = requests.get(f"{api_base}/api/v1/simulate", params={"hour": hour}, timeout=8)
        resp.raise_for_status()
        return resp.json(), True
    except Exception:
        return _local_demo(hour), False


# ──────────────────────────────────────────────────────────────────────────────
# Plotly gauge factory
# ──────────────────────────────────────────────────────────────────────────────

def _gauge(value: float, title: str, max_val: float, color: str,
           suffix: str = "", steps=None) -> go.Figure:
    steps = steps or [
        {"range": [0, max_val * 0.4], "color": "#d1fae5"},
        {"range": [max_val * 0.4, max_val * 0.75], "color": "#fef3c7"},
        {"range": [max_val * 0.75, max_val], "color": "#fee2e2"},
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"suffix": suffix, "font": {"size": 22}},
        gauge={
            "axis":       {"range": [0, max_val], "tickwidth": 1},
            "bar":        {"color": color},
            "bgcolor":    "white",
            "borderwidth": 2,
            "bordercolor":"#e2e8f0",
            "steps":       steps,
            "threshold":  {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": max_val * 0.85,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, sans-serif"},
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Status badge helper
# ──────────────────────────────────────────────────────────────────────────────

_STATUS_BADGE = {
    "SURPLUS":  "badge-surplus",
    "BALANCED": "badge-balanced",
    "DEFICIT":  "badge-deficit",
    "CRITICAL": "badge-critical",
}
_RISK_BADGE = {
    "NORMAL":   "risk-normal",
    "WATCH":    "risk-watch",
    "ALERT":    "risk-alert",
    "CRITICAL": "risk-critical",
}


# ──────────────────────────────────────────────────────────────────────────────
# Main render function
# ──────────────────────────────────────────────────────────────────────────────

def render_dashboard(data: dict, is_live: bool) -> None:
    kpis     = data.get("kpis", {})
    solar    = data.get("solar_forecast", {})
    demand_d = data.get("demand_balance", {})
    battery  = data.get("battery_status", {})
    outage   = data.get("outage_risk", {})

    # ── Header ──────────────────────────────────────────────────────────────
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown("## ⚡ Gujarat Smart Grid – Live Operations Dashboard")
        st.caption(
            f"{'🟢 Live (FastAPI)' if is_live else '🟡 Demo Mode (offline)'}"
            f"  ·  IST Hour: {manual_hour:02d}:00"
            f"  ·  Updated: {datetime.now().strftime('%H:%M:%S')}"
        )
    with col_status:
        gs = kpis.get("grid_status", "BALANCED")
        css = _STATUS_BADGE.get(gs, "badge-balanced")
        st.markdown(f"<br><span class='{css}'>GRID: {gs}</span>", unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Top KPI Strip ────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("☀️ Solar Output",     f"{kpis.get('solar_output_mw', 0):,.0f} MW",
              delta=f"CF {kpis.get('confidence_pct', 0):.0f}%")
    k2.metric("📈 Consumer Demand",  f"{kpis.get('demand_mw', 0):,.0f} MW")
    k3.metric("⚖️ Supply-Demand Gap", f"{kpis.get('gap_mw', 0):+,.0f} MW",
              delta_color="inverse" if kpis.get("gap_mw", 0) < 0 else "normal")
    k4.metric("🔋 Battery SoC",      f"{kpis.get('battery_soc_pct', 0):.1f}%",
              delta=f"{battery.get('action','HOLD')}")
    k5.metric("🚨 Outage Risk",      f"{kpis.get('outage_risk_score', 0):.0f}/100",
              delta=kpis.get("outage_risk_level", "NORMAL"),
              delta_color="inverse" if kpis.get("outage_risk_score", 0) > 60 else "off")

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Row 1: Gauges ────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("<div class='section-title'>☀️ Solar Generation</div>", unsafe_allow_html=True)
        fig = _gauge(
            kpis.get("solar_output_mw", 0), "Solar Output (MW)", 10000,
            "#f59e0b", " MW",
            steps=[
                {"range": [0, 3000],  "color": "#fef3c7"},
                {"range": [3000, 7000], "color": "#d1fae5"},
                {"range": [7000, 10000], "color": "#a7f3d0"},
            ],
        )
        st.plotly_chart(fig, use_container_width=True)
        if solar.get("anomaly_detected"):
            st.warning(f"⚠️ Anomaly: {solar.get('anomaly_description', '')}")

    with g2:
        st.markdown("<div class='section-title'>🔋 Battery State of Charge</div>", unsafe_allow_html=True)
        soc = kpis.get("battery_soc_pct", 60)
        soc_color = "#10b981" if 25 <= soc <= 85 else "#ef4444"
        fig = _gauge(
            soc, "Battery SoC (%)", 100, soc_color, "%",
            steps=[
                {"range": [0, 15],   "color": "#fee2e2"},   # danger – too low
                {"range": [15, 25],  "color": "#fed7aa"},
                {"range": [25, 85],  "color": "#d1fae5"},   # optimal
                {"range": [85, 90],  "color": "#fed7aa"},
                {"range": [90, 100], "color": "#fee2e2"},   # danger – too high
            ],
        )
        st.plotly_chart(fig, use_container_width=True)
        batt_action = kpis.get("battery_action", "HOLD")
        action_icon = {"CHARGE": "🔼 CHARGING", "DISCHARGE": "🔽 DISCHARGING", "HOLD": "⏸ HOLD"}
        st.info(f"{action_icon.get(batt_action, batt_action)} · "
                f"{battery.get('power_setpoint_mw', 0):.1f} MW · "
                f"Proj SoC: {battery.get('projected_soc_after_pct', soc):.1f}%")

    with g3:
        st.markdown("<div class='section-title'>🚨 Outage Risk Score</div>", unsafe_allow_html=True)
        risk_score = kpis.get("outage_risk_score", 0)
        risk_color = (
            "#ef4444" if risk_score > 80 else
            "#f97316" if risk_score > 60 else
            "#f59e0b" if risk_score > 30 else "#10b981"
        )
        fig = _gauge(
            risk_score, "Risk Score / 100", 100, risk_color, "",
            steps=[
                {"range": [0,  30], "color": "#d1fae5"},
                {"range": [30, 60], "color": "#fef3c7"},
                {"range": [60, 80], "color": "#fed7aa"},
                {"range": [80, 100],"color": "#fee2e2"},
            ],
        )
        st.plotly_chart(fig, use_container_width=True)
        rl = kpis.get("outage_risk_level", "NORMAL")
        css = _RISK_BADGE.get(rl, "risk-normal")
        st.markdown(
            f"Asset: `{outage.get('asset_id','—')}`  "
            f"<span class='{css}'>{rl}</span>",
            unsafe_allow_html=True,
        )
        if outage.get("predicted_failure_window_h"):
            st.error(f"⏱ Predicted failure within {outage['predicted_failure_window_h']:.1f}h")

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Row 2: 24h Forecast Chart ───────────────────────────────────────────
    st.markdown("<div class='section-title'>📈 24-Hour Solar Generation Forecast (MW)</div>",
                unsafe_allow_html=True)

    forecast_24h = solar.get("forecast_24h_mw", [0] * 24)
    hours_labels = [f"{(manual_hour + i) % 24:02d}:00" for i in range(1, 25)]

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=hours_labels, y=forecast_24h,
        mode="lines+markers",
        name="Solar Forecast (MW)",
        line=dict(color="#f59e0b", width=2.5),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.12)",
    ))
    # Demand overlay (flat estimate)
    demand_val = kpis.get("demand_mw", 0)
    if demand_val > 0:
        fig_forecast.add_hline(
            y=demand_val, line_dash="dash", line_color="#3b82f6",
            annotation_text=f"Current Demand: {demand_val:,.0f} MW",
            annotation_position="bottom right",
        )
    fig_forecast.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=30),
        xaxis_title="Hour (IST)",
        yaxis_title="MW",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(247,249,252,1)",
        font={"family": "Segoe UI, sans-serif", "size": 12},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ── Row 3: Power Flow Bar ────────────────────────────────────────────────
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚡ Real-Time Power Flow Breakdown (MW)</div>",
                unsafe_allow_html=True)

    supply_mw  = kpis.get("supply_mw", 0)
    demand_mw  = kpis.get("demand_mw", 0)
    solar_mw   = kpis.get("solar_output_mw", 0)
    battery_mw = battery.get("power_setpoint_mw", 0) if kpis.get("battery_action") == "DISCHARGE" else 0

    fig_flow = go.Figure()
    fig_flow.add_trace(go.Bar(
        name="Solar Generation",  x=["Power Sources"], y=[solar_mw],
        marker_color="#f59e0b", text=[f"{solar_mw:,.0f}"], textposition="auto",
    ))
    fig_flow.add_trace(go.Bar(
        name="Battery Dispatch",  x=["Power Sources"], y=[battery_mw],
        marker_color="#10b981", text=[f"{battery_mw:,.0f}"], textposition="auto",
    ))
    fig_flow.add_trace(go.Bar(
        name="Consumer Demand",   x=["Demand"],        y=[demand_mw],
        marker_color="#3b82f6", text=[f"{demand_mw:,.0f}"], textposition="auto",
    ))
    fig_flow.update_layout(
        barmode="stack", height=220,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(247,249,252,1)",
        font={"family": "Segoe UI, sans-serif", "size": 12},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    # ── Row 4: Dispatch Commands & Preventive Signals ────────────────────────
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    col_disp, col_signals = st.columns(2)

    with col_disp:
        st.markdown("<div class='section-title'>⚙️ Active Dispatch Commands</div>",
                    unsafe_allow_html=True)
        cmds = demand_d.get("dispatch_commands", [])
        if cmds:
            for cmd in cmds:
                icon = {"Solar": "☀️", "Battery": "🔋", "Thermal": "🔥",
                        "Grid": "✅", "Demand Response": "📉"}.get(cmd.get("source", ""), "⚡")
                mw = cmd.get("mw_adjustment", 0)
                color = "success" if mw > 0 else "warning" if mw < 0 else "info"
                getattr(st, color)(
                    f"**[P{cmd.get('priority',1)}] {icon} {cmd.get('action','')}**  \n"
                    f"_{cmd.get('reason','')}_"
                )
        else:
            st.success("✅ No dispatch actions required – grid balanced.")

    with col_signals:
        st.markdown("<div class='section-title'>🛡️ Preventive Signals</div>",
                    unsafe_allow_html=True)
        signals = outage.get("preventive_signals", [])
        if signals:
            for sig in signals:
                urgency_color = {"IMMEDIATE": "error", "WITHIN_1H": "warning",
                                 "SCHEDULED": "info"}.get(sig.get("urgency", "info"), "info")
                getattr(st, urgency_color)(
                    f"**[{sig.get('urgency','')}] {sig.get('signal_type','')} → {sig.get('target','')}**  \n"
                    f"{sig.get('message','')}"
                )
        else:
            st.success("✅ No preventive signals active.")

    # ── Row 5: Agent Reasoning Logs ─────────────────────────────────────────
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🤖 IBM Granite Agent Reasoning Logs</div>",
                unsafe_allow_html=True)

    agents_reasoning = [
        ("☀️ Solar Forecasting Agent",    solar.get("agent_reasoning", "—")),
        ("⚖️ Demand Balancing Agent",     demand_d.get("agent_reasoning", "—")),
        ("🔋 Battery Optimizer Agent",    battery.get("agent_reasoning", "—")),
        ("🚨 Outage Prediction Agent",    outage.get("agent_reasoning", "—")),
    ]
    for agent_name, reasoning in agents_reasoning:
        with st.expander(agent_name, expanded=False):
            st.markdown(f"> {reasoning}")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    st.caption(
        "Smart Energy Management & Grid Balancing Platform · "
        "IBM Watsonx · Granite-3-8B-Instruct · LangGraph · FastAPI  "
        "| Gujarat SLDC Prototype © 2025"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Auto-refresh loop
# ──────────────────────────────────────────────────────────────────────────────

# Session state for last fetch time
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = 0.0
if "cached_data" not in st.session_state:
    st.session_state.cached_data = None
if "cached_live" not in st.session_state:
    st.session_state.cached_live = False

now = time.time()
should_refresh = (
    run_now
    or st.session_state.cached_data is None
    or (now - st.session_state.last_fetch) >= refresh_interval
)

if should_refresh:
    with st.spinner("Running agent pipeline…"):
        if demo_mode:
            data, is_live = _local_demo(manual_hour), False
        else:
            data, is_live = _fetch_data(api_base, manual_hour)
    st.session_state.cached_data = data
    st.session_state.cached_live = is_live
    st.session_state.last_fetch  = now
else:
    data    = st.session_state.cached_data
    is_live = st.session_state.cached_live

render_dashboard(data, is_live)

# Schedule a rerun after the refresh interval
time.sleep(0.1)   # Tiny sleep to yield control; Streamlit handles the loop
