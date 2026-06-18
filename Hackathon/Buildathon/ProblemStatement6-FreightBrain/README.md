# FreightBrain: AI-Powered Load Intelligence Platform

> **Buildathon 2026 — Data Pipeline Track**
> Team: Lone Rangers | Problem Statement 6

---

## What It Does

95% of US trucking companies run fewer than 10 trucks on 3–6% margins. Load boards show **gross rate** — but a driver has no idea what they actually pocket after fuel, tolls, deadhead miles, and getting stranded in a dead market.

**FreightBrain** solves this with:

1. **6-Component Net Profit Engine** — not gross rate, actual pocket money after fuel, driver pay, insurance, maintenance, tolls, and deadhead
2. **Dual Deadhead Pricing** — costs deadhead TO pickup AND repositioning FROM delivery city (most tools only count one direction)
3. **Market Liquidity Score (MLS 0–100)** — quantifies how easy a city is to "escape" with another load
4. **NVIDIA AI Agent** — explains in plain English *why* Load #1 beats Load #2 even if Load #2 pays more gross
5. **Top 10 Carrier Start Markets** — actionable market intelligence from 100,000 real loads

---

## Architecture

```
data/text/*.txt  (100,000 loads, 3 rotating formats)
       │
   [src/parser.py]       regex block parser — all 3 formats
       │
   [src/cost_model.py]   6-component net profit + dual deadhead
       │
   [src/market_scorer.py] Market Liquidity Score (MLS 0–100)
       │
   [src/agent.py]        NVIDIA NIM (meta/llama-3.1-70b-instruct)
       │
   [src/dashboard.py]    Streamlit 4-screen app
       │
   [notebooks/]          Databricks Delta Lake medallion pipeline
                         Bronze → Silver → Gold
```

### Databricks Medallion Tables

| Layer | Table | Contents |
|-------|-------|----------|
| Bronze | `freightbrain.bronze.loads` | Raw parsed loads (100K rows) |
| Silver | `freightbrain.silver.enriched_loads` | + all 6 cost components via PySpark UDFs |
| Gold | `freightbrain.gold.net_profit_loads` | All loads ranked by net profitability |
| Gold | `freightbrain.gold.market_quality_index` | MLS scores per city |

---

## Net Profit Formula

```
NET_PROFIT = GROSS_RATE − (
    fuel_cost          = miles / 6.5 MPG × $3.85/gal
    driver_pay         = miles × $0.55
    insurance          = miles × $0.085
    maintenance        = miles × $0.18
    toll_cost          = corridor lookup by state pair
    deadhead_cost      = (dh_miles / 6.5 × $3.85) + (dh_miles × $0.55)
    repositioning_cost = (1 − dest_MLS/100) × $150   ← dead market penalty
)
```

Variable cost benchmark: ~$1.00/mile fully loaded. Any net RPM above $1.00 is real profit.

---

## Market Liquidity Score

```
MLS(city) = outbound_load_density × 0.40
          + lane_balance_score    × 0.30   (outbound / inbound ratio)
          + avg_rate_per_mile     × 0.20
          + destination_diversity × 0.10

Score 0–100. Cities below 40 incur a $150 repositioning penalty.
```

Top markets from our 100K-load dataset:

| Rank | City | State | MLS |
|------|------|-------|-----|
| 1 | Chicago | IL | 85.0 |
| 2 | Atlanta | GA | 82.2 |
| 3 | Indianapolis | IN | 81.3 |
| 4 | Memphis | TN | 79.1 |
| 5 | Columbus | OH | 78.4 |
| 6 | Nashville | TN | 77.6 |
| 7 | Louisville | KY | 76.7 |
| 8 | Dallas | TX | 75.8 |
| 9 | Charlotte | NC | 75.7 |
| 10 | St. Louis | MO | 73.5 |

---

## Project Structure

```
ProblemStatement6-FreightBrain/
├── data/
│   ├── text/                        # 13 × loads_part_NNN.txt  (100K loads)
│   └── pdf/                         # 12 × broker_load_sheet_NNN.pdf
├── notebooks/                       # Databricks notebooks (.py format)
│   ├── 01_bronze_ingestion.py       # text files → Delta Bronze table
│   ├── 02_silver_enrichment.py      # PySpark cost model UDFs → Silver
│   ├── 03_gold_analytics.py         # net_profit_loads + market_quality_index
│   └── 04_agent_demo.py             # NVIDIA AI agent demo in Databricks
├── src/
│   ├── parser.py                    # Regex parser: handles 3 unstructured formats
│   ├── cost_model.py                # 6-component net profit engine
│   ├── market_scorer.py             # MLS calculation + city rankings
│   ├── agent.py                     # NVIDIA NIM agent (OpenAI-compatible API)
│   ├── city_coords.py               # 110+ US city lat/lon lookup table
│   └── dashboard.py                 # Streamlit 4-screen app
├── tests/
│   ├── test_cost_model.py           # 21 unit tests
│   └── test_market_scorer.py        # 17 unit tests
├── requirements.txt
├── .env.example                     # Copy to .env and fill in your keys
└── README.md
```

---

## Quick Start (Local Machine)

### 1. Clone and install

```bash
git clone https://github.com/saiteja-optimus/Lone-Rangers-GenAI.git
cd Lone-Rangers-GenAI/ProblemStatement6-FreightBrain
pip install -r requirements.txt
pip install openai
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.1-70b-instruct
DATABRICKS_HOST=https://community.cloud.databricks.com
DATABRICKS_TOKEN=dapi-your-token-here
```

### 3. Run tests

```bash
pytest tests/ -v
# 38 passed
```

### 4. Launch the dashboard

```bash
cd src
streamlit run dashboard.py
# http://localhost:8501
```

---

## Databricks Setup — Complete Step-by-Step

### Step 1 — Pull the repo into Databricks

**Recommended: Databricks Repos UI**

1. In your Databricks workspace, click **Workspace** in the left sidebar
2. Go to **Repos → Add Repo**
3. Enter URL: `https://github.com/saiteja-optimus/Lone-Rangers-GenAI.git`
4. Click **Create Repo**
5. The repo lands at `/Repos/<your-email>/Lone-Rangers-GenAI`

**Alternative: Clone via notebook shell cell**

```python
%sh
git clone https://github.com/saiteja-optimus/Lone-Rangers-GenAI.git /databricks/driver/FreightBrain
ls /databricks/driver/FreightBrain/ProblemStatement6-FreightBrain/
```

---

### Step 2 — Install Python dependencies on the cluster

Open any notebook and run:

```python
%pip install openai pandas numpy plotly pydeck python-dotenv geopy pytest
```

The cluster will restart automatically. Wait for it, then continue.

---

### Step 3 — Set your NVIDIA API key

Add this cell at the top of each notebook before any imports:

```python
import os, sys

# --- API credentials (use Databricks Secrets in production) ---
os.environ["NVIDIA_API_KEY"]   = "nvapi-your-key-here"
os.environ["NVIDIA_BASE_URL"]  = "https://integrate.api.nvidia.com/v1"
os.environ["NVIDIA_MODEL"]     = "meta/llama-3.1-70b-instruct"

# --- Point Python at the src folder ---
REPO_ROOT = "/Repos/your-email/Lone-Rangers-GenAI/ProblemStatement6-FreightBrain"
# If you used %sh git clone, use this instead:
# REPO_ROOT = "/databricks/driver/FreightBrain/ProblemStatement6-FreightBrain"

sys.path.insert(0, f"{REPO_ROOT}/src")
DATA_DIR = f"{REPO_ROOT}/data"
```

---

### Step 4 — Run Notebook 01: Bronze Ingestion

**File:** `notebooks/01_bronze_ingestion.py`

Import this file into Databricks (File → Import) or open it from Repos.

**What it does:**
- Reads all 13 `data/text/loads_part_NNN.txt` files
- Parses 100,000 loads using the 3-format regex parser
- Creates Databricks database `freightbrain` if it does not exist
- Writes Delta table `freightbrain.bronze.loads` with full overwrite (idempotent)
- Runtime: ~3–5 minutes on Community Edition single node

**Run and verify:**
```python
spark.sql("SELECT COUNT(*) AS total FROM freightbrain.bronze.loads").show()
# Expected: 100000

spark.sql("""
    SELECT equipment_type, COUNT(*) AS cnt
    FROM freightbrain.bronze.loads
    GROUP BY equipment_type ORDER BY cnt DESC
""").show()
# Expected:
# Dry Van    ~50,000
# Reefer     ~20,000
# Flatbed    ~20,000
# Step Deck  ~10,000
```

---

### Step 5 — Run Notebook 02: Silver Enrichment

**File:** `notebooks/02_silver_enrichment.py`

**What it does:**
- Reads `freightbrain.bronze.loads`
- Applies pandas UDFs in PySpark for each cost component:
  - `fuel_cost` — miles / 6.5 MPG × $3.85/gal
  - `driver_cost` — miles × $0.55/mi
  - `insurance_cost` — miles × $0.085/mi
  - `maintenance_cost` — miles × $0.18/mi
  - `toll_cost` — state-pair corridor lookup
  - `deadhead_cost` — empty miles cost (fuel + driver)
  - `repositioning_cost` — dead market penalty based on dest MLS
  - `equipment_surcharge` — Reefer +12%, Flatbed +8%, Step Deck +6%
- Computes `net_profit` and `net_rpm`
- Writes Delta table `freightbrain.silver.enriched_loads`

**Spot-check (a 500-mile Dry Van at $2.50/mi should net ~$600–700):**
```python
spark.sql("""
    SELECT load_id, distance_miles, rate_usd,
           ROUND(net_profit, 2) AS net_profit,
           ROUND(net_rpm, 3)    AS net_rpm
    FROM freightbrain.silver.enriched_loads
    WHERE distance_miles BETWEEN 490 AND 510
      AND equipment_type = 'Dry Van'
    ORDER BY net_profit DESC
    LIMIT 5
""").show()
# net_profit should be in range $600–$750
```

---

### Step 6 — Run Notebook 03: Gold Analytics

**File:** `notebooks/03_gold_analytics.py`

**What it does:**
- Creates `freightbrain.gold.net_profit_loads`
  - All 100K loads ranked by `net_profit` descending
  - Partitioned by `equipment_type` for fast dashboard queries
- Creates `freightbrain.gold.market_quality_index`
  - MLS score 0–100 for every origin city
  - Includes: `outbound_loads`, `inbound_loads`, `lane_balance`, `avg_rpm`, `grade`

**Verify top markets:**
```python
spark.sql("""
    SELECT city, state,
           ROUND(mls_score, 1) AS mls,
           grade,
           outbound_loads,
           ROUND(avg_rpm, 2) AS avg_rpm
    FROM freightbrain.gold.market_quality_index
    ORDER BY mls_score DESC
    LIMIT 10
""").show()
# Top cities: Chicago (85) → Atlanta (82) → Indianapolis (81) → Memphis (79)
```

---

### Step 7 — Run Notebook 04: Agent Demo

**File:** `notebooks/04_agent_demo.py`

**What it does:**
- Loads top candidate loads from Gold table
- Calls NVIDIA NIM AI (`meta/llama-3.1-70b-instruct`) via OpenAI-compatible API
- Demonstrates the core hackathon scenario:
  - Driver in Memphis, TN with a Dry Van
  - AI compares Atlanta load vs Laredo load
  - Shows why net profit matters more than gross rate
- Prints full AI reasoning in structured format

**Expected AI output:**
```
**RECOMMENDED LOAD:** L-XXXXXXX

**WHY THIS LOAD:** This X-mile run generates $X,XXX net profit —
the strongest opportunity in your deadhead radius. The destination
market has healthy outbound load availability, reducing repositioning risk.

**RISK FLAGS:**
- Deadhead of XXX miles adds to fuel cost
- Confirm rate lock with broker before dispatch

**RUNNER-UP:** L-XXXXXXX — second-best net profit, shorter haul

**MARKET INSIGHT:** [Destination city] market remains strong
due to [reason]. Expect good next-load availability.
```

---

### Step 8 — Run the Dashboard

**Option A — Local machine (best for judge demo)**

```bash
# Pull latest from GitHub on your local machine
git pull origin main

cd ProblemStatement6-FreightBrain/src
streamlit run dashboard.py
# Opens at http://localhost:8501
```

**Option B — Inside Databricks with public URL (ngrok)**

```python
%pip install streamlit pyngrok
import subprocess, threading, time
from pyngrok import ngrok

REPO_ROOT = "/Repos/your-email/Lone-Rangers-GenAI/ProblemStatement6-FreightBrain"

def _run():
    subprocess.run([
        "python", "-m", "streamlit", "run",
        f"{REPO_ROOT}/src/dashboard.py",
        "--server.port", "8501",
        "--server.headless", "true",
    ])

threading.Thread(target=_run, daemon=True).start()
time.sleep(8)
url = ngrok.connect(8501)
print(f"\n Dashboard URL: {url.public_url}\n")
```

Copy the printed URL and share it with judges.

---

## Dashboard Screens

### Screen 1 — Load Finder
Enter your city, state, equipment type, and max deadhead miles.  
Returns the top 10 loads ranked by **net profit** (not gross rate).  
Shows a cost breakdown bar chart and a map with origin/destination pins.

### Screen 2 — AI Explanation
Live NVIDIA AI reasoning for the #1 recommended load:
- Why this load wins (net profit, RPM, market outlook)
- Risk flags (long deadhead, destination MLS, rate volatility)
- Runner-up comparison
- Market insight for the destination city

### Screen 3 — Market Rankings
Top 10 US cities to start a carrier, ranked by MLS.  
Table shows: MLS score, grade (A/B/C), outbound loads, lane balance ratio, avg RPM.  
Includes a bubble map: size = outbound volume, color = MLS score.

### Screen 4 — Lane Heat Map
Outbound load volume for every market.  
Bubble size = load count. Color gradient = avg net RPM.  
Filter by equipment type to see Reefer vs Flatbed market differences.

---

## The Core Demo: Why Dual Deadhead Changes Everything

```
Driver: Memphis, TN | Dry Van | Max Deadhead: 150 miles

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Load A: Little Rock, AR → Portland, OR
  Gross Rate:    $7,250
  Distance:      2,350 miles
  Deadhead:      130 miles (to Little Rock)
  Portland MLS:  38.3  (repositioning cost: $88)
  NET PROFIT:    $3,423  ← AI recommends
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Load B: Memphis, TN → Laredo, TX
  Gross Rate:    $3,100
  Distance:      875 miles
  Deadhead:      0 miles (right here)
  Laredo MLS:    55.7  (repositioning cost: $67)
  NET PROFIT:    $1,205
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Load A wins by $2,218 net despite lower destination MLS,
because the gross rate more than compensates.
Most tools would show Load B as "safer" — FreightBrain
prices the full picture including where you end up.
```

---

## Running Tests

```bash
cd ProblemStatement6-FreightBrain
pytest tests/ -v --tb=short

# Output:
# tests/test_cost_model.py::TestFuelCost::test_500_mile PASSED
# tests/test_cost_model.py::TestNetProfit::test_atl_fw_baseline PASSED
# tests/test_market_scorer.py::TestHubVsRural::test_hub_beats_rural PASSED
# ... 38 passed in 1.5s
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data warehouse | Databricks Community Edition + Delta Lake |
| Pipeline | PySpark with pandas UDFs |
| AI agent | NVIDIA NIM — `meta/llama-3.1-70b-instruct` |
| Dashboard | Streamlit + Plotly + pydeck |
| Parser | Custom regex (3-format unstructured text) |
| Testing | pytest — 38 tests |
| Version control | GitHub |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Yes | Your NVIDIA NIM key (`nvapi-...`) |
| `NVIDIA_BASE_URL` | Yes | `https://integrate.api.nvidia.com/v1` |
| `NVIDIA_MODEL` | No | Default: `meta/llama-3.1-70b-instruct` |
| `DATABRICKS_HOST` | Notebooks | `https://community.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | Notebooks | Personal access token from Databricks settings |

Copy `.env.example` → `.env`. The `.env` file is in `.gitignore` and will never be committed to git.

---

## Team: Lone Rangers
**Buildathon 2026 — Data Pipeline Track — Problem Statement 6**
