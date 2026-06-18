"""FreightBrain Dashboard — Streamlit app for judge demo."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from parser import parse_all_files
from cost_model import calculate_net_profit, haversine_miles, net_rpm
from market_scorer import (
    compute_market_liquidity_scores,
    get_top_markets,
    get_mls_for_city,
)
from agent import FreightBrainAgent
from city_coords import CITY_COORDS, lookup_coords

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FreightBrain",
    layout="wide",
    page_icon="\U0001f69b",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.winner-card {
    background: linear-gradient(135deg, #1a4731 0%, #0d2c1e 100%);
    border: 2px solid #2ea84a;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}
.winner-card h2 { color: #4ade80; margin: 0 0 8px 0; }
</style>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading trucking loads...")
def _load_data() -> pd.DataFrame:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return parse_all_files(data_dir)


@st.cache_data(show_spinner="Computing Market Liquidity Scores...")
def _load_mls(_df: pd.DataFrame) -> pd.DataFrame:
    return compute_market_liquidity_scores(_df)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = [
    "\U0001f50d Load Finder",
    "\U0001f916 AI Recommendation",
    "\U0001f3c6 Top 10 Markets",
    "\U0001f5fa Lane Heat Map",
]

with st.sidebar:
    st.title("\U0001f69b FreightBrain")
    st.caption("AI Load Intelligence for Small Carriers")
    st.divider()
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("104K real broker loads · 50 US markets")
    st.caption("Powered by Claude AI + Databricks Delta Lake")

df = _load_data()
mls_df = _load_mls(df)

CITIES = sorted(
    {f"{k.split(',')[0]},{k.split(',')[1]}" for k in CITY_COORDS}
)
EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer", "Step Deck", "Power Only", "Any"]


def _attach_financials(
    fdf: pd.DataFrame, driver_lat: float | None, driver_lon: float | None, max_dh: int
) -> pd.DataFrame:
    fdf = fdf.copy()
    if driver_lat is not None:
        fdf["deadhead_miles"] = fdf.apply(
            lambda r: haversine_miles(driver_lat, driver_lon, r["origin_lat"], r["origin_lon"])
            if pd.notna(r["origin_lat"])
            else 9999.0,
            axis=1,
        )
        fdf = fdf[fdf["deadhead_miles"] <= max_dh]
    else:
        fdf["deadhead_miles"] = 0.0

    fdf["dest_mls"] = fdf.apply(
        lambda r: get_mls_for_city(mls_df, str(r["dest_city"]), str(r["dest_state"])), axis=1
    )
    results = fdf.apply(
        lambda r: calculate_net_profit(
            float(r["gross_rate"]),
            float(r["miles"]),
            float(r["deadhead_miles"]),
            str(r["equipment"]),
            float(r["dest_mls"]),
        ),
        axis=1,
    )
    fdf["net_profit"] = results.apply(lambda x: x[0])
    fdf["net_rpm_val"] = fdf.apply(
        lambda r: net_rpm(
            float(r["gross_rate"]),
            float(r["miles"]),
            float(r["deadhead_miles"]),
            str(r["equipment"]),
            float(r["dest_mls"]),
        ),
        axis=1,
    )
    return fdf


# ===========================================================================
# PAGE 1 — Load Finder
# ===========================================================================
if page == PAGES[0]:
    st.title("\U0001f50d Load Finder")
    st.caption("Find the most profitable loads within your deadhead radius")

    c1, c2, c3 = st.columns([2, 1, 1])
    default_idx = CITIES.index("Atlanta,GA") if "Atlanta,GA" in CITIES else 0
    with c1:
        driver_loc = st.selectbox("Driver Location", CITIES, index=default_idx)
    with c2:
        equip_sel = st.selectbox("Equipment", EQUIPMENT_TYPES)
    with c3:
        max_dh = st.slider("Max Deadhead (mi)", 50, 300, 150, step=25)

    st.button("\U0001f680 Find Best Loads", type="primary", use_container_width=True)

    driver_city, driver_state = driver_loc.split(",", 1)
    coords = lookup_coords(driver_city, driver_state)
    dlat, dlon = (coords[0], coords[1]) if coords else (None, None)

    fdf = df[df["equipment"] == equip_sel].copy() if equip_sel != "Any" else df.copy()
    fdf = _attach_financials(fdf, dlat, dlon, max_dh)
    top10 = fdf.nlargest(10, "net_profit")

    if top10.empty:
        st.warning("No loads found. Try expanding deadhead or changing equipment.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Loads Found", f"{len(fdf):,}")
    k2.metric("Best Net Profit", f"${top10.iloc[0]['net_profit']:,.0f}")
    k3.metric("Best Net RPM", f"${top10.iloc[0]['net_rpm_val']:.2f}")
    k4.metric("Avg Deadhead", f"{fdf['deadhead_miles'].mean():.0f} mi")

    st.divider()
    st.subheader("Top 10 Loads by Net Profit")

    disp = top10.copy()
    disp["Lane"] = disp["origin_city"] + ", " + disp["origin_state"] + " \u2192 " + disp["dest_city"] + ", " + disp["dest_state"]
    st.dataframe(
        disp[["load_id","Lane","miles","gross_rate","net_profit","net_rpm_val",
              "dest_mls","deadhead_miles","pickup_date","equipment"]]
        .rename(columns={
            "load_id": "Load ID", "miles": "Miles", "gross_rate": "Gross $",
            "net_profit": "Net $", "net_rpm_val": "Net $/mi",
            "dest_mls": "Dest MLS", "deadhead_miles": "DH mi",
            "pickup_date": "Pickup Date", "equipment": "Equipment",
        })
        .style.format({
            "Gross $": "${:,.0f}", "Net $": "${:,.0f}",
            "Net $/mi": "${:.2f}", "Dest MLS": "{:.0f}",
            "DH mi": "{:.0f}", "Miles": "{:.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Gross vs Net Profit — Top 5")
    top5 = top10.head(5).copy()
    top5["label"] = top5["origin_city"].str[:3] + "\u2192" + top5["dest_city"].str[:3]
    bar_df = pd.melt(
        top5[["label","gross_rate","net_profit"]],
        id_vars="label", var_name="Type", value_name="Amount",
    )
    bar_df["Type"] = bar_df["Type"].map({"gross_rate": "Gross Rate", "net_profit": "Net Profit"})
    fig_bar = px.bar(
        bar_df, x="label", y="Amount", color="Type", barmode="group",
        color_discrete_map={"Gross Rate": "#3b82f6", "Net Profit": "#22c55e"},
        labels={"label": "Load", "Amount": "USD ($)"},
        template="plotly_dark",
    )
    fig_bar.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Load Map")
    map_pts = []
    if coords:
        map_pts.append({"lat": coords[0], "lon": coords[1],
                        "label": f"YOU: {driver_city},{driver_state}",
                        "color": "Driver", "size": 14})
    for _, r in top5.iterrows():
        if pd.notna(r.get("origin_lat")):
            map_pts.append({"lat": r["origin_lat"], "lon": r["origin_lon"],
                            "label": f"Pickup: {r['origin_city']}",
                            "color": "Pickup", "size": 10})
        if pd.notna(r.get("dest_lat")):
            map_pts.append({"lat": r["dest_lat"], "lon": r["dest_lon"],
                            "label": f"Dest: {r['dest_city']} (Net ${r['net_profit']:.0f})",
                            "color": "Destination", "size": 10})
    mdf = pd.DataFrame(map_pts).dropna(subset=["lat","lon"])
    if not mdf.empty:
        fig_map = px.scatter_mapbox(
            mdf, lat="lat", lon="lon", color="color", text="label", size="size",
            color_discrete_map={"Driver":"#3b82f6","Pickup":"#22c55e","Destination":"#ef4444"},
            zoom=4, height=420, mapbox_style="carto-darkmatter", template="plotly_dark",
        )
        fig_map.update_traces(textposition="top center")
        fig_map.update_layout(margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_map, use_container_width=True)


# ===========================================================================
# PAGE 2 — AI Recommendation
# ===========================================================================
elif page == PAGES[1]:
    st.title("\U0001f916 AI Load Recommendation")
    st.caption("Claude analyzes your options and picks the best load with full reasoning")

    c1, c2, c3 = st.columns([2,1,1])
    chi_idx = CITIES.index("Chicago,IL") if "Chicago,IL" in CITIES else 0
    with c1:
        loc2 = st.selectbox("Driver Location", CITIES, index=chi_idx, key="ai_loc")
    with c2:
        eq2 = st.selectbox("Equipment", ["Dry Van","Flatbed","Reefer","Step Deck"], key="ai_eq")
    with c3:
        dh2 = st.slider("Max Deadhead (mi)", 50, 300, 150, key="ai_dh")

    st.button("\U0001f9e0 Get AI Recommendation", type="primary", use_container_width=True)

    city2, state2 = loc2.split(",", 1)
    coords2 = lookup_coords(city2, state2)
    dlat2, dlon2 = (coords2[0], coords2[1]) if coords2 else (None, None)

    fdf2 = df[df["equipment"] == eq2].copy()
    fdf2 = _attach_financials(fdf2, dlat2, dlon2, dh2)
    candidates = fdf2.nlargest(10, "net_profit")

    if candidates.empty:
        st.warning("No loads found. Expand deadhead or change equipment.")
        st.stop()

    agent = FreightBrainAgent()
    with st.spinner("Claude is analyzing your loads..."):
        rec, full_text = agent.recommend(city2, state2, eq2, candidates, mls_df)

    if rec:
        st.markdown(
            f'<div class="winner-card">'
            f"<h2>\U0001f3c6 Winner: Load {rec.load_id}</h2>"
            f"<b>{rec.origin} \u2192 {rec.destination}</b><br>"
            f"{rec.miles:.0f} mi &nbsp;|&nbsp; ${rec.gross_rate:,.0f} gross &nbsp;|&nbsp;"
            f"<span style=\"color:#4ade80;font-size:1.2em\"><b>${rec.net_profit:,.0f} NET</b></span>"
            f"&nbsp;|&nbsp; ${rec.net_rpm:.2f}/mi &nbsp;|&nbsp; MLS {rec.dest_mls:.0f}"
            f"</div>",
            unsafe_allow_html=True,
        )

        col_text, col_chart = st.columns([3, 2])
        with col_text:
            st.subheader("Claude's Analysis")
            st.markdown(full_text)

        with col_chart:
            st.subheader("Cost Breakdown")
            bd = rec.cost_breakdown
            labels = ["Fuel","Driver Pay","Insurance","Maintenance",
                      "Tolls","Deadhead","Repo Penalty","Equip Surcharge"]
            values = [bd["fuel_cost"], bd["driver_pay"], bd["insurance"],
                      bd["maintenance"], bd["tolls"], bd["deadhead_cost"],
                      bd["repo_penalty"], bd["equipment_surcharge"]]
            nz = [(l, v) for l, v in zip(labels, values) if v > 0]
            if nz:
                total_cost = sum(v for _, v in nz)
                fig_d = go.Figure(go.Pie(
                    labels=[x[0] for x in nz],
                    values=[x[1] for x in nz],
                    hole=0.55,
                    marker_colors=px.colors.qualitative.Set3,
                ))
                fig_d.update_layout(
                    template="plotly_dark", height=360,
                    annotations=[dict(
                        text=f"${total_cost:,.0f}<br>Total Cost",
                        x=0.5, y=0.5, font_size=13, showarrow=False, font_color="white",
                    )],
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig_d, use_container_width=True)

            if rec.risk_flags:
                st.subheader("Risk Flags")
                for flag in rec.risk_flags:
                    st.warning(f"\u26a0 {flag}")


# ===========================================================================
# PAGE 3 — Top 10 Markets
# ===========================================================================
elif page == PAGES[2]:
    st.title("\U0001f3c6 Top 10 US Markets to Start a Carrier")
    st.caption("Ranked by Market Liquidity Score — the #1 predictor of carrier profitability")

    if mls_df.empty:
        st.error("MLS data unavailable.")
        st.stop()

    top10m = get_top_markets(mls_df, 10)

    state_mls = (
        mls_df.groupby("state")
        .agg(avg_mls=("mls_score","mean"), total_loads=("outbound_loads","sum"))
        .reset_index()
    )
    fig_choro = px.choropleth(
        state_mls, locations="state", locationmode="USA-states",
        color="avg_mls", scope="usa",
        color_continuous_scale=["#0d2137","#1e6091","#1abc9c","#f1c40f","#e74c3c"],
        title="Average Market Liquidity Score by State",
        labels={"avg_mls": "Avg MLS"},
        template="plotly_dark",
    )
    fig_choro.update_layout(height=420, margin=dict(t=40,b=0,l=0,r=0))
    st.plotly_chart(fig_choro, use_container_width=True)

    st.divider()
    col_tbl, col_bar = st.columns([3, 2])

    with col_tbl:
        st.subheader("Market Rankings")
        rank_df = top10m[["city","state","mls_score","outbound_loads",
                           "lane_balance","avg_rpm","grade"]].copy()
        rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))
        st.dataframe(
            rank_df.rename(columns={
                "city":"City","state":"ST","mls_score":"MLS Score",
                "outbound_loads":"Outbound Loads","lane_balance":"Lane Balance",
                "avg_rpm":"Avg RPM","grade":"Grade",
            }).style.format({
                "MLS Score": "{:.1f}", "Lane Balance": "{:.2f}", "Avg RPM": "${:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

    with col_bar:
        st.subheader("MLS Score Components")
        def _norm_s(s):
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5

        comp = top10m.copy()
        comp["Outbound Density"] = _norm_s(comp["outbound_loads"]) * 35
        comp["Lane Balance Score"] = comp["lane_balance"] * 25
        comp["Rate Level"] = _norm_s(comp["avg_rpm"]) * 25
        comp["Diversity"] = _norm_s(comp["unique_dest_states"]) * 15
        comp["label"] = comp["city"].str[:10]
        stacked = pd.melt(
            comp[["label","Outbound Density","Lane Balance Score","Rate Level","Diversity"]],
            id_vars="label", var_name="Component", value_name="Score",
        )
        fig_s = px.bar(
            stacked, x="label", y="Score", color="Component",
            color_discrete_sequence=["#3b82f6","#22c55e","#f59e0b","#a855f7"],
            template="plotly_dark", labels={"label":"City","Score":"Points"},
        )
        fig_s.update_layout(height=380, margin=dict(t=20,b=60), xaxis_tickangle=-30)
        st.plotly_chart(fig_s, use_container_width=True)


# ===========================================================================
# PAGE 4 — Lane Heat Map
# ===========================================================================
elif page == PAGES[3]:
    st.title("\U0001f5fa Lane Heat Map")
    st.caption("Bubble size = outbound load volume  ·  Color = average net RPM")

    eq_heat = st.selectbox(
        "Filter Equipment", ["All","Dry Van","Flatbed","Reefer","Step Deck"], key="heat_eq"
    )
    hdf = df.copy() if eq_heat == "All" else df[df["equipment"] == eq_heat].copy()

    hdf["dest_mls"] = hdf.apply(
        lambda r: get_mls_for_city(mls_df, str(r["dest_city"]), str(r["dest_state"])), axis=1
    )
    hdf["net_rpm_val"] = hdf.apply(
        lambda r: net_rpm(
            float(r["gross_rate"]), float(r["miles"]), 0.0,
            str(r["equipment"]), float(r["dest_mls"]),
        ),
        axis=1,
    )

    city_agg = (
        hdf.groupby(["origin_city","origin_state"])
        .agg(
            load_count=("load_id","count"),
            avg_net_rpm=("net_rpm_val","mean"),
            lat=("origin_lat","first"),
            lon=("origin_lon","first"),
        )
        .reset_index()
        .dropna(subset=["lat","lon"])
    )
    city_agg["hover"] = (
        city_agg["origin_city"] + ", " + city_agg["origin_state"]
        + "<br>Loads: " + city_agg["load_count"].astype(str)
        + "<br>Avg Net RPM: $" + city_agg["avg_net_rpm"].round(2).astype(str)
    )
    fig_b = px.scatter_mapbox(
        city_agg, lat="lat", lon="lon", size="load_count", color="avg_net_rpm",
        hover_name="hover",
        color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
        size_max=50, zoom=3, height=500, mapbox_style="carto-darkmatter",
        template="plotly_dark",
        labels={"avg_net_rpm":"Avg Net RPM ($)","load_count":"Loads"},
    )
    fig_b.update_layout(margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig_b, use_container_width=True)

    st.divider()
    st.subheader("Top 20 Lanes by Avg Net Profit")
    lane_agg = (
        hdf.groupby(["origin_city","origin_state","dest_city","dest_state"])
        .agg(
            load_count=("load_id","count"),
            avg_gross=("gross_rate","mean"),
            avg_miles=("miles","mean"),
            avg_net_rpm=("net_rpm_val","mean"),
        )
        .reset_index()
    )
    lane_agg["avg_net"] = lane_agg.apply(
        lambda r: calculate_net_profit(r["avg_gross"], r["avg_miles"])[0], axis=1
    )
    lane_agg["Lane"] = (
        lane_agg["origin_city"] + ", " + lane_agg["origin_state"]
        + " \u2192 "
        + lane_agg["dest_city"] + ", " + lane_agg["dest_state"]
    )
    top20 = lane_agg.nlargest(20, "avg_net")
    st.dataframe(
        top20[["Lane","load_count","avg_miles","avg_gross","avg_net","avg_net_rpm"]]
        .rename(columns={
            "load_count":"Loads","avg_miles":"Avg Miles",
            "avg_gross":"Avg Gross $","avg_net":"Avg Net $","avg_net_rpm":"Avg Net RPM",
        })
        .style.format({
            "Avg Miles":"{:.0f}","Avg Gross $":"${:,.0f}",
            "Avg Net $":"${:,.0f}","Avg Net RPM":"${:.3f}",
        }),
        use_container_width=True, hide_index=True,
    )
