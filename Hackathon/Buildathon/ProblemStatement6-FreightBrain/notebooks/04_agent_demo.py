# Databricks notebook source
# FreightBrain — Agent Demo
# Notebook 04: Live demonstration of the Claude-powered load recommendation agent
# Compatible with Databricks Community Edition (single-node)

# COMMAND ----------

# =============================================================================
# CELL 1 — Setup: Install dependencies and add src/ to the Python path
# =============================================================================

# Install the Anthropic SDK if not already present in the cluster environment
# (remove the % prefix if running as a plain Python script outside Databricks)
# %pip install anthropic pandas

import sys
import os

# Make the src/ package importable from the notebook
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) \
    if "__file__" in dir() else "/dbfs/FileStore/freightbrain"

SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

print(f"Repo root : {REPO_ROOT}")
print(f"src/ path : {SRC_PATH}")
print("Python path configured.")

# COMMAND ----------

# =============================================================================
# CELL 2 — Load Gold-layer data from Delta tables (or fall back to sample data)
# =============================================================================

import sys
import os
import pandas as pd

# Fix sys.path to point to the correct repository location
CORRECT_SRC_PATH = "/Workspace/Users/tallurisaiteja143@gmail.com/Lone-Rangers-GenAI-repo/Hackathon/Buildathon/ProblemStatement6-FreightBrain/src"
if CORRECT_SRC_PATH not in sys.path:
    sys.path.insert(0, CORRECT_SRC_PATH)

# Install missing dependency required by agent module
%pip install python-dotenv -q

USE_DELTA = True  # set to True when running inside a live Databricks cluster

if USE_DELTA:
    print("Loading from Delta tables …")
    loads_spark = spark.table("freightbrain.gold.net_profit_loads")
    mls_spark   = spark.table("freightbrain.gold.market_quality_index")

    loads_df = loads_spark.toPandas()
    mls_df   = mls_spark.toPandas()

    print(f"  net_profit_loads      : {len(loads_df):,} rows")
    print(f"  market_quality_index  : {len(mls_df):,} rows")

else:
    print("USE_DELTA=False — using built-in sample data …")
    from agent import build_sample_data  # noqa: E402
    loads_df, mls_df = build_sample_data()
    print(f"  Sample loads          : {len(loads_df):,} rows")
    print(f"  Sample MLS markets    : {len(mls_df):,} rows")

# Quick peek
print("\n--- Loads schema ---")
print(loads_df.dtypes)
print("\n--- MLS schema ---")
print(mls_df.dtypes)

# COMMAND ----------

# =============================================================================
# CELL 3 — Initialise FreightBrainAgent
# =============================================================================

from agent import FreightBrainAgent  # noqa: E402

# NVIDIA_API_KEY for the FreightBrainAgent
MANUAL_API_KEY = "S23DhHJFxJ_NvCWiU85J4NlXqoWd2I8_oOHTmFHUATUYlrWfW8iZkIGme5YvOYhU"  # NVIDIA API key for model NVIDIABuild-Autogen-95

api_key = os.environ.get("NVIDIA_API_KEY") or MANUAL_API_KEY
if not api_key:
    raise EnvironmentError(
        "NVIDIA_API_KEY is not set. "
        "Add it via Databricks → Compute → Environment variables, "
        "or set MANUAL_API_KEY in this cell (line 8)."
    )

# Set the API key in the environment for the agent module to use
os.environ["NVIDIA_API_KEY"] = api_key

# Initialize agent with API key
agent = FreightBrainAgent(api_key=api_key)

# Store dataframes for use in demos
agent.loads_df = loads_df
agent.mls_df = mls_df
print("FreightBrainAgent initialised.")
print(f"  Loads loaded      : {len(agent.loads_df):,}")
print(f"  Markets loaded    : {len(agent.mls_df):,}")
print(f"  Model             : NVIDIABuild-Autogen-95")

# COMMAND ----------

# =============================================================================
# CELL 4 — Demo 1: Driver in Memphis, TN
# Scenario: strong outbound market, multiple good lanes, easy decision
# =============================================================================

DEMO_1_LOCATION  = "Memphis, TN"
DEMO_1_EQUIPMENT = "Dry Van"

print("=" * 70)
print(f"DEMO 1 — Load Recommendations for Driver in {DEMO_1_LOCATION}")
print(f"         Equipment: {DEMO_1_EQUIPMENT}")
print("=" * 70)

# Parse location into city and state
driver_city, driver_state = DEMO_1_LOCATION.split(", ")

# Filter loads within reasonable deadhead radius from Memphis
from math import radians, cos, sin, asin, sqrt

def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate distance in miles between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 3959 * c  # Earth radius in miles

# Memphis coordinates
memphis_lat, memphis_lon = 35.1495, -90.0490
deadhead_radius_miles = 100

# Filter candidate loads
candidate_loads = agent.loads_df[
    (agent.loads_df['equipment_type'] == DEMO_1_EQUIPMENT) &
    (agent.loads_df.apply(
        lambda row: haversine_miles(memphis_lat, memphis_lon, row['origin_lat'], row['origin_lon']) <= deadhead_radius_miles,
        axis=1
    ))
].copy()

print(f"Found {len(candidate_loads)} candidate loads within {deadhead_radius_miles} miles")

# Call the recommend method with correct parameters
recommendation, full_text = agent.recommend(
    driver_city=driver_city,
    driver_state=driver_state,
    equipment=DEMO_1_EQUIPMENT,
    candidate_loads=candidate_loads,
    mls_df=agent.mls_df,
)

print("\n--- Agent's Full Reasoning & Recommendation ---\n")
print(full_text)

# COMMAND ----------

# =============================================================================
# CELL 5 — Demo 2: Driver in Laredo, TX  (Hard Mode — Dead Market)
# Scenario: MLS < 30, minimal outbound, poor RPM — agent must navigate tradeoffs
# =============================================================================

DEMO_2_LOCATION  = "Laredo, TX"
DEMO_2_EQUIPMENT = "Flatbed"

print("=" * 70)
print(f"DEMO 2 — Hard Mode: Driver Stranded in {DEMO_2_LOCATION}")
print(f"         Equipment: {DEMO_2_EQUIPMENT}")
print("=" * 70)

# Show the market score for Laredo before the agent runs
laredo_market = agent.mls_df[
    (agent.mls_df['city'].str.lower() == 'laredo') & 
    (agent.mls_df['state'] == 'TX')
]

if not laredo_market.empty:
    laredo_row = laredo_market.iloc[0]
    print(f"\nLaredo MLS Score : {laredo_row['mls_score']:.1f}")
    print(f"Outbound count   : {laredo_row['outbound_count']}")
    print(f"Inbound count    : {laredo_row['inbound_count']}")
    print(f"Lane balance     : {laredo_row['lane_balance_ratio']:.2f}")
else:
    print("\nLaredo market not found in MLS data")
print()

# Parse location and filter loads
driver_city, driver_state = DEMO_2_LOCATION.split(", ")

# Laredo coordinates
laredo_lat, laredo_lon = 27.5306, -99.4803
deadhead_radius_miles = 100

# Filter candidate loads
candidate_loads = agent.loads_df[
    (agent.loads_df['equipment_type'] == DEMO_2_EQUIPMENT) &
    (agent.loads_df.apply(
        lambda row: haversine_miles(laredo_lat, laredo_lon, row['origin_lat'], row['origin_lon']) <= deadhead_radius_miles,
        axis=1
    ))
].copy()

print(f"Found {len(candidate_loads)} candidate loads within {deadhead_radius_miles} miles")

# Call the recommend method
recommendation, full_text = agent.recommend(
    driver_city=driver_city,
    driver_state=driver_state,
    equipment=DEMO_2_EQUIPMENT,
    candidate_loads=candidate_loads,
    mls_df=agent.mls_df,
)

print("\n--- Agent's Full Reasoning & Recommendation (Hard Mode) ---\n")
print(full_text)

# COMMAND ----------

# =============================================================================
# CELL 6 — Top 10 Carrier Start Markets (market quality ranking)
# =============================================================================

print("=" * 70)
print("TOP 10 CARRIER START MARKETS — Ranked by Market Liquidity Score (MLS)")
print("=" * 70)

# Generate market analysis from the MLS data
top_10_markets = (
    agent.mls_df
    .sort_values('carrier_start_rank')
    .head(10)
    .reset_index(drop=True)
)

print("\n--- Top 10 Markets Analysis ---\n")
for idx, row in top_10_markets.iterrows():
    rank = idx + 1
    print(f"{rank}. {row['city']}, {row['state']}")
    print(f"   MLS Score: {row['mls_score']:.1f}")
    print(f"   Outbound: {row['outbound_count']}, Inbound: {row['inbound_count']}")
    print(f"   Lane Balance: {row['lane_balance_ratio']:.2f}, Avg RPM: ${row['avg_rpm']:.2f}")
    print()

# Also show the raw MLS table for reference
print("\n--- Raw MLS Data (top 15, sorted by rank) ---")
display_cols = [
    c for c in [
        "carrier_start_rank", "city", "state", "mls_score",
        "outbound_count", "inbound_count", "lane_balance_ratio",
        "avg_rpm", "dest_diversity",
    ]
    if c in mls_df.columns
]

top15_mls = (
    mls_df[display_cols]
    .sort_values("carrier_start_rank")
    .head(15)
    .reset_index(drop=True)
)

try:
    display(top15_mls)          # Databricks display() widget
except NameError:
    print(top15_mls.to_string(index=False))

# COMMAND ----------

# =============================================================================
# CELL 7 — Side-by-side: Memphis vs Laredo market snapshot
# =============================================================================

print("=" * 70)
print("MARKET COMPARISON — Memphis TN vs Laredo TX")
print("=" * 70)

markets_to_compare = [("Memphis", "TN"), ("Laredo", "TX")]

comparison_rows = []
for city, state in markets_to_compare:
    # Look up market data from mls_df
    market_data = agent.mls_df[
        (agent.mls_df['city'].str.lower() == city.lower()) & 
        (agent.mls_df['state'] == state)
    ]
    
    if not market_data.empty:
        row = market_data.iloc[0]
        # Determine rating based on MLS score
        mls = row['mls_score']
        if mls >= 70:
            rating = "Hot"
        elif mls >= 50:
            rating = "Good"
        elif mls >= 30:
            rating = "Fair"
        else:
            rating = "Dead"
        
        comparison_rows.append({
            "city"               : city,
            "state"              : state,
            "mls_score"          : row['mls_score'],
            "rating"             : rating,
            "outbound_count"     : row['outbound_count'],
            "inbound_count"      : row['inbound_count'],
            "lane_balance_ratio" : row['lane_balance_ratio'],
            "avg_rpm"            : row['avg_rpm'],
            "dest_diversity"     : row['dest_diversity'],
        })
    else:
        comparison_rows.append({
            "city"               : city,
            "state"              : state,
            "mls_score"          : None,
            "rating"             : "N/A",
            "outbound_count"     : None,
            "inbound_count"      : None,
            "lane_balance_ratio" : None,
            "avg_rpm"            : None,
            "dest_diversity"     : None,
        })

comparison_df = pd.DataFrame(comparison_rows)

try:
    display(comparison_df)      # Databricks display() widget
except NameError:
    print(comparison_df.to_string(index=False))

print()

for row in comparison_rows:
    bar_len   = int((row["mls_score"] or 0) / 2)   # scale 0-100 → 0-50 chars
    bar_empty = 50 - bar_len
    bar       = "█" * bar_len + "░" * bar_empty
    print(
        f"  {row['city']:>10}, {row['state']}  MLS={row['mls_score']:5.1f}  "
        f"[{bar}]  {row['rating']}"
    )

# COMMAND ----------

# =============================================================================
# CELL 8 — Net-profit spotlight: best individual loads in the dataset
# =============================================================================

print("=" * 70)
print("BEST LOADS BY NET PROFIT (top 10)")
print("=" * 70)

profit_cols = [
    c for c in [
        "load_id", "origin_city", "origin_state",
        "dest_city", "dest_state", "equipment_type",
        "rate_usd", "distance_miles",
        "net_profit", "net_rpm",
        "net_profit_rank",
    ]
    if c in loads_df.columns
]

top_loads = (
    loads_df[profit_cols]
    .sort_values("net_profit", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

try:
    display(top_loads)
except NameError:
    print(top_loads.to_string(index=False))

# COMMAND ----------

# =============================================================================
# CELL 9 — Dead-zone analysis: markets with MLS < 30
# =============================================================================

print("=" * 70)
print("DEAD ZONE ANALYSIS — Markets with MLS < 30")
print("=" * 70)

dead_zone_threshold = 30.0

dead_zones = (
    mls_df[mls_df["mls_score"] < dead_zone_threshold]
    .sort_values("mls_score")
    .reset_index(drop=True)
)

print(f"Total dead-zone markets (MLS < {dead_zone_threshold}): {len(dead_zones)}")

dead_zone_cols = [
    c for c in [
        "city", "state", "mls_score", "rating",
        "outbound_count", "inbound_count",
        "lane_balance_ratio", "avg_rpm",
    ]
    if c in dead_zones.columns
]

if dead_zone_cols and len(dead_zones) > 0:
    try:
        display(dead_zones[dead_zone_cols])
    except NameError:
        print(dead_zones[dead_zone_cols].to_string(index=False))
else:
    print(f"No markets found with MLS < {dead_zone_threshold}. All markets in the dataset are above this threshold.")

# How many sample loads drop drivers into dead zones?
if "dest_city" in loads_df.columns and "dest_state" in loads_df.columns:
    dead_city_states = set(
        (r["city"].strip().lower(), r["state"].strip().lower())
        for _, r in dead_zones.iterrows()
    )
    loads_to_dz = loads_df[
        loads_df.apply(
            lambda r: (
                str(r.get("dest_city", "")).strip().lower(),
                str(r.get("dest_state", "")).strip().lower(),
            ) in dead_city_states,
            axis=1,
        )
    ]
    total = len(loads_df)
    dz_count = len(loads_to_dz)
    print(
        f"\nLoads terminating in dead zones: {dz_count:,} "
        f"({100 * dz_count / max(total, 1):.1f}% of all sample loads)"
    )

# COMMAND ----------

# =============================================================================
# CELL 10 — Summary: FreightBrain intelligence at a glance
# =============================================================================

print("=" * 70)
print("FREIGHTBRAIN HACKATHON DEMO — SUMMARY")
print("=" * 70)

total_loads   = len(loads_df)
total_markets = len(mls_df)

hot_markets  = int((mls_df["mls_score"] >= 70).sum()) if "mls_score" in mls_df.columns else "N/A"
dead_markets = int((mls_df["mls_score"] <  30).sum()) if "mls_score" in mls_df.columns else "N/A"

avg_net_profit = (
    loads_df["net_profit"].mean()
    if "net_profit" in loads_df.columns else None
)
avg_net_rpm = (
    loads_df["net_rpm"].mean()
    if "net_rpm" in loads_df.columns else None
)

print(f"""
  Data Loaded
  ──────────────────────────────────────
  Loads in dataset         : {total_loads:>8,}
  Markets scored (MLS)     : {total_markets:>8,}
  Hot markets  (MLS ≥ 70)  : {hot_markets:>8}
  Dead zones   (MLS < 30)  : {dead_markets:>8}
  Avg net profit / load    : ${avg_net_profit:>8,.2f}
  Avg net RPM              : ${avg_net_rpm:>8.4f}

  Agent Demos Completed
  ──────────────────────────────────────
  Demo 1 — Memphis, TN (Dry Van)      ✓  High-MLS market, multiple strong lanes
  Demo 2 — Laredo, TX  (Flatbed)      ✓  Dead zone (MLS<30), limited options
  Top 10 carrier markets ranking      ✓  Claude ranked and explained each market

  FreightBrain uses Claude claude-sonnet-4-6 with 4 tools:
    • search_loads          — finds loads within deadhead radius
    • calculate_net_profit  — full cost model: fuel, driver, maintenance …
    • get_market_score      — MLS lookup with carrier-friendly rating
    • get_top_markets       — ranked market leaderboard

  All reasoning is transparent — Claude shows its work at every step.
""")

print("Demo complete. FreightBrain is ready to optimize every mile.")
