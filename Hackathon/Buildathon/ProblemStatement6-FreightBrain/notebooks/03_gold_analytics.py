# Databricks notebook source
# FreightBrain — Gold Analytics Layer
# Notebook 03: Build gold.net_profit_loads and gold.market_quality_index
# Compatible with Databricks Community Edition (single-node)

# COMMAND ----------

# =============================================================================
# CELL 1 — Read from Silver
# =============================================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("Reading enriched loads from Silver layer...")

silver_df = spark.table("freightbrain.silver.enriched_loads")

print(f"Silver record count : {silver_df.count():,}")
print(f"Columns             : {len(silver_df.columns)}")

silver_df.printSchema()

# Quick sanity check
silver_df.select(
    F.count("*").alias("total_loads"),
    F.countDistinct("equipment_type").alias("equipment_types"),
    F.round(F.avg("net_profit"), 2).alias("avg_net_profit"),
    F.round(F.avg("net_rpm"), 4).alias("avg_net_rpm"),
    F.min("pickup_date").alias("earliest_pickup"),
    F.max("pickup_date").alias("latest_pickup"),
).show(truncate=False)

# COMMAND ----------

# =============================================================================
# CELL 2 — Build gold.net_profit_loads
# =============================================================================

print("Building gold.net_profit_loads ...")

# Create gold schema if it doesn't exist
spark.sql("CREATE SCHEMA IF NOT EXISTS freightbrain.gold")

# Window specs for rankings
w_profit_global = Window.orderBy(F.desc("net_profit"))
w_rpm_global    = Window.orderBy(F.desc("net_rpm"))

net_profit_loads = (
    silver_df
    .withColumn("net_profit_rank", F.rank().over(w_profit_global))
    .withColumn("net_rpm_rank",    F.rank().over(w_rpm_global))
)

# Write as Delta, partitioned by equipment_type
(
    net_profit_loads.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("equipment_type")
    .saveAsTable("freightbrain.gold.net_profit_loads")
)

count = spark.table("freightbrain.gold.net_profit_loads").count()
print(f"gold.net_profit_loads written: {count:,} rows")

# Peek at the top 5 by net profit
print("\nTop 5 loads by net_profit:")
(
    spark.table("freightbrain.gold.net_profit_loads")
    .select(
        "load_id", "origin_city", "origin_state",
        "dest_city", "dest_state",
        "equipment_type", "rate_usd",
        "net_profit", "net_profit_rank",
        "net_rpm", "net_rpm_rank",
    )
    .orderBy("net_profit_rank")
    .limit(5)
    .show(truncate=False)
)

# COMMAND ----------

# =============================================================================
# CELL 3 — Build gold.market_quality_index (Market Liquidity Score)
# =============================================================================

print("Building gold.market_quality_index ...")

# ── Outbound stats ──────────────────────────────────────────────────────────
outbound = (
    silver_df
    .groupBy(
        F.col("origin_city").alias("city"),
        F.col("origin_state").alias("state"),
    )
    .agg(
        F.count("*").alias("outbound_count"),
        F.round(F.avg("net_rpm"), 4).alias("avg_outbound_rpm"),
        F.countDistinct("dest_city").alias("dest_diversity"),
    )
)

# ── Inbound stats ──────────────────────────────────────────────────────────
inbound = (
    silver_df
    .groupBy(
        F.col("dest_city").alias("city"),
        F.col("dest_state").alias("state"),
    )
    .agg(F.count("*").alias("inbound_count"))
)

# ── Join outbound + inbound ────────────────────────────────────────────────
market_base = (
    outbound
    .join(inbound, ["city", "state"], "left")
    .fillna({"inbound_count": 0})
)

# ── Lane balance ratio  (>1 = more outbound than inbound = carrier-friendly)
market_base = market_base.withColumn(
    "lane_balance_ratio",
    F.round(
        F.col("outbound_count") / (F.col("inbound_count") + F.lit(1)),
        4,
    ),
)

# ── Compute raw MLS components ─────────────────────────────────────────────
#   Component 1 — outbound volume score  (log-normalised, 0-40 pts)
#   Component 2 — lane balance score     (ratio capped at 2.0,  0-30 pts)
#   Component 3 — dest diversity score   (log-normalised, 0-20 pts)
#   Component 4 — avg RPM score          (percentile,      0-10 pts)

max_outbound = market_base.agg(F.max("outbound_count")).collect()[0][0] or 1
max_diversity = market_base.agg(F.max("dest_diversity")).collect()[0][0] or 1
max_rpm       = market_base.agg(F.max("avg_outbound_rpm")).collect()[0][0] or 1.0

import math as _math

market_scored = (
    market_base
    .withColumn(
        "vol_score",
        F.round(
            F.log1p(F.col("outbound_count")) /
            F.lit(_math.log1p(max_outbound)) * F.lit(40),
            2,
        ),
    )
    .withColumn(
        "balance_score",
        F.round(
            F.least(F.col("lane_balance_ratio"), F.lit(2.0)) /
            F.lit(2.0) * F.lit(30),
            2,
        ),
    )
    .withColumn(
        "diversity_score",
        F.round(
            F.log1p(F.col("dest_diversity")) /
            F.lit(_math.log1p(max_diversity)) * F.lit(20),
            2,
        ),
    )
    .withColumn(
        "rpm_score",
        F.round(
            (F.col("avg_outbound_rpm") / F.lit(max_rpm)) * F.lit(10),
            2,
        ),
    )
    .withColumn(
        "mls_score",
        F.round(
            F.col("vol_score") +
            F.col("balance_score") +
            F.col("diversity_score") +
            F.col("rpm_score"),
            1,
        ),
    )
    # Clamp 0–100
    .withColumn("mls_score", F.least(F.greatest(F.col("mls_score"), F.lit(0.0)), F.lit(100.0)))
    .withColumn("avg_rpm", F.col("avg_outbound_rpm"))
)

# ── Rankings ───────────────────────────────────────────────────────────────
w_mls  = Window.orderBy(F.desc("mls_score"))

market_quality_index = (
    market_scored
    .withColumn("mls_rank",           F.rank().over(w_mls))
    .withColumn("carrier_start_rank", F.rank().over(w_mls))  # same ranking, semantic alias
    .select(
        "city", "state",
        F.col("mls_score").cast("double"),
        "outbound_count", "inbound_count",
        "lane_balance_ratio", "avg_rpm", "dest_diversity",
        "mls_rank", "carrier_start_rank",
    )
)

# ── Persist ────────────────────────────────────────────────────────────────
(
    market_quality_index.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("freightbrain.gold.market_quality_index")
)

mq_count = spark.table("freightbrain.gold.market_quality_index").count()
print(f"gold.market_quality_index written: {mq_count:,} markets")

print("\nTop 10 carrier start markets (by MLS):")
(
    spark.table("freightbrain.gold.market_quality_index")
    .orderBy("mls_rank")
    .limit(10)
    .show(truncate=False)
)

# COMMAND ----------

# =============================================================================
# CELL 4 — Analytics Summary
# =============================================================================

from pyspark.sql import functions as F

gold_loads  = spark.table("freightbrain.gold.net_profit_loads")
gold_market = spark.table("freightbrain.gold.market_quality_index")

# ── 4a: Top 10 markets for carriers ─────────────────────────────────────────
print("=" * 60)
print("TOP 10 CARRIER START MARKETS")
print("=" * 60)
(
    gold_market
    .select(
        "carrier_start_rank", "city", "state",
        F.round("mls_score", 1).alias("mls"),
        "outbound_count",
        F.round("avg_rpm", 4).alias("avg_rpm"),
        F.round("lane_balance_ratio", 2).alias("balance"),
        "dest_diversity",
    )
    .orderBy("carrier_start_rank")
    .limit(10)
    .show(truncate=False)
)

# ── 4b: Average net profit by equipment type ─────────────────────────────────
print("=" * 60)
print("AVERAGE NET PROFIT BY EQUIPMENT TYPE")
print("=" * 60)
(
    gold_loads
    .groupBy("equipment_type")
    .agg(
        F.round(F.avg("net_profit"), 2).alias("avg_net_profit"),
        F.round(F.avg("net_rpm"), 4).alias("avg_net_rpm"),
        F.round(F.avg("rate_usd"), 2).alias("avg_gross"),
        F.count("*").alias("load_count"),
    )
    .orderBy(F.desc("avg_net_profit"))
    .show(truncate=False)
)

# ── 4c: Best lanes by avg net_rpm ───────────────────────────────────────────
print("=" * 60)
print("BEST LANES BY AVG NET RPM (min 10 loads)")
print("=" * 60)
(
    gold_loads
    .groupBy("origin_city", "origin_state", "dest_city", "dest_state", "equipment_type")
    .agg(
        F.round(F.avg("net_rpm"), 4).alias("avg_net_rpm"),
        F.round(F.avg("net_profit"), 2).alias("avg_net_profit"),
        F.count("*").alias("load_count"),
    )
    .filter(F.col("load_count") >= 10)
    .withColumn(
        "lane",
        F.concat(
            F.col("origin_city"), F.lit(", "), F.col("origin_state"),
            F.lit(" → "),
            F.col("dest_city"), F.lit(", "), F.col("dest_state"),
        ),
    )
    .select("lane", "equipment_type", "avg_net_rpm", "avg_net_profit", "load_count")
    .orderBy(F.desc("avg_net_rpm"))
    .limit(15)
    .show(truncate=False)
)

# ── 4d: Deadhead "dead zones" — destinations with MLS < 30 ──────────────────
print("=" * 60)
print("DEADHEAD DANGER ZONES — Destinations with MLS < 30")
print("=" * 60)

dead_zones = gold_market.filter(F.col("mls_score") < 30)
dead_zone_count = dead_zones.count()
print(f"Markets with MLS < 30: {dead_zone_count}")

# How many loads drop drivers into dead zones?
dead_zone_cities = dead_zones.select(
    F.concat(F.col("city"), F.lit(","), F.col("state")).alias("city_state")
)

loads_to_dead_zones = gold_loads.join(
    dead_zone_cities,
    F.concat(F.col("dest_city"), F.lit(","), F.col("dest_state")) == F.col("city_state"),
    "inner",
)

dead_zone_load_count = loads_to_dead_zones.count()
total_loads = gold_loads.count()

print(f"Loads dropping into dead zones: {dead_zone_load_count:,} "
      f"({100*dead_zone_load_count/max(total_loads,1):.1f}% of all loads)")

print("\nWorst dead zones (lowest MLS, highest inbound traffic):")
(
    dead_zones
    .select(
        "city", "state",
        F.round("mls_score", 1).alias("mls"),
        "inbound_count",
        "outbound_count",
        F.round("avg_rpm", 4).alias("avg_rpm"),
    )
    .orderBy(F.desc("inbound_count"), "mls_score")
    .limit(10)
    .show(truncate=False)
)

# COMMAND ----------

# =============================================================================
# CELL 5 — Final stats and confirmation
# =============================================================================

print("=" * 60)
print("FREIGHTBRAIN GOLD LAYER — FINAL SUMMARY")
print("=" * 60)

net_profit_stats = spark.table("freightbrain.gold.net_profit_loads").agg(
    F.count("*").alias("total_loads"),
    F.round(F.min("net_profit"), 2).alias("min_net_profit"),
    F.round(F.avg("net_profit"), 2).alias("avg_net_profit"),
    F.round(F.max("net_profit"), 2).alias("max_net_profit"),
    F.round(F.avg("net_rpm"), 4).alias("avg_net_rpm"),
    F.countDistinct("equipment_type").alias("equipment_types"),
    F.countDistinct("origin_city").alias("origin_cities"),
    F.countDistinct("dest_city").alias("dest_cities"),
).collect()[0]

market_stats = spark.table("freightbrain.gold.market_quality_index").agg(
    F.count("*").alias("total_markets"),
    F.round(F.avg("mls_score"), 1).alias("avg_mls"),
    F.sum(F.when(F.col("mls_score") >= 70, 1).otherwise(0)).alias("hot_markets"),
    F.sum(F.when(F.col("mls_score") < 30, 1).otherwise(0)).alias("dead_zones"),
).collect()[0]

print(f"""
  gold.net_profit_loads
  ─────────────────────────────────────────
  Total loads          : {net_profit_stats['total_loads']:>10,}
  Equipment types      : {net_profit_stats['equipment_types']:>10}
  Origin cities        : {net_profit_stats['origin_cities']:>10,}
  Destination cities   : {net_profit_stats['dest_cities']:>10,}
  Min net profit       : ${net_profit_stats['min_net_profit']:>10,.2f}
  Avg net profit       : ${net_profit_stats['avg_net_profit']:>10,.2f}
  Max net profit       : ${net_profit_stats['max_net_profit']:>10,.2f}
  Avg net RPM          : ${net_profit_stats['avg_net_rpm']:>10.4f}

  gold.market_quality_index
  ─────────────────────────────────────────
  Total markets scored : {market_stats['total_markets']:>10,}
  Avg MLS score        : {market_stats['avg_mls']:>10.1f}
  Hot markets (MLS≥70) : {market_stats['hot_markets']:>10,}
  Dead zones  (MLS<30) : {market_stats['dead_zones']:>10,}
""")

# Verify both tables are registered in the metastore
tables = spark.sql("SHOW TABLES IN freightbrain.gold").collect()
print("Tables in freightbrain.gold:")
for t in tables:
    rows = spark.table(f"freightbrain.gold.{t['tableName']}").count()
    print(f"  {t['tableName']:35s}  {rows:>10,} rows")

print("\nGold layer build complete.")
