# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver Enrichment
# MAGIC
# MAGIC Reads `freightbrain.bronze.loads`, applies all 8 cost-model components via
# MAGIC pandas UDFs, writes `freightbrain.silver.enriched_loads`.

# COMMAND ----------
# Cell 1 — Setup: add src/ to sys.path, imports

import sys
import os

# Allow imports from src/ whether running locally or on Databricks
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) if "__file__" in dir() else "/Workspace/Repos/freightbrain"
_src_path = os.path.join(_repo_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from cost_model import (
    FUEL_CPM,
    DRIVER_CPM,
    INSURANCE_CPM,
    MAINTENANCE_CPM,
    TOLL_CPM,
    DEADHEAD_FUEL_CPM,
    EQUIPMENT_SURCHARGE,
    MAX_REPO_PENALTY,
    calculate_net_profit,
    fuel_cost_for_miles,
    net_rpm,
)

print("Setup complete. cost_model imported.")
print(f"  FUEL_CPM={FUEL_CPM:.4f}  DRIVER_CPM={DRIVER_CPM}  MAX_REPO_PENALTY={MAX_REPO_PENALTY}")

# COMMAND ----------
# Cell 2 — Read freightbrain.bronze.loads Delta table

bronze_df = spark.table("freightbrain.bronze.loads")
print(f"Bronze loads: {bronze_df.count():,} rows")
bronze_df.printSchema()
bronze_df.show(5, truncate=False)

# COMMAND ----------
# Cell 3 — Register pandas UDFs for each cost component

# ------------------------------------------------------------------
# Fuel cost  (loaded miles only)
# ------------------------------------------------------------------
@F.pandas_udf(DoubleType())
def udf_fuel_cost(miles: "pd.Series") -> "pd.Series":
    return (miles * FUEL_CPM).round(2)


# ------------------------------------------------------------------
# Driver pay  (loaded miles)
# ------------------------------------------------------------------
@F.pandas_udf(DoubleType())
def udf_driver_pay(miles: "pd.Series") -> "pd.Series":
    return (miles * DRIVER_CPM).round(2)


# ------------------------------------------------------------------
# Insurance  (loaded miles)
# ------------------------------------------------------------------
@F.pandas_udf(DoubleType())
def udf_insurance(miles: "pd.Series") -> "pd.Series":
    return (miles * INSURANCE_CPM).round(2)


# ------------------------------------------------------------------
# Maintenance  (loaded miles)
# ------------------------------------------------------------------
@F.pandas_udf(DoubleType())
def udf_maintenance(miles: "pd.Series") -> "pd.Series":
    return (miles * MAINTENANCE_CPM).round(2)


# ------------------------------------------------------------------
# Tolls  (loaded miles)
# ------------------------------------------------------------------
@F.pandas_udf(DoubleType())
def udf_tolls(miles: "pd.Series") -> "pd.Series":
    return (miles * TOLL_CPM).round(2)


# ------------------------------------------------------------------
# Deadhead cost  (deadhead_miles = 0 if column absent)
# ------------------------------------------------------------------
@F.pandas_udf(DoubleType())
def udf_deadhead_cost(deadhead_miles: "pd.Series") -> "pd.Series":
    # cost = deadhead_miles * (DEADHEAD_FUEL_CPM + DRIVER_CPM * 0.5)
    rate = DEADHEAD_FUEL_CPM + DRIVER_CPM * 0.5
    return (deadhead_miles.fillna(0.0) * rate).round(2)


# ------------------------------------------------------------------
# Equipment surcharge  (loaded miles * per-equipment CPM)
# ------------------------------------------------------------------
@F.pandas_udf(DoubleType())
def udf_equipment_surcharge(miles: "pd.Series", equipment: "pd.Series") -> "pd.Series":
    surcharge_cpm = equipment.map(lambda e: EQUIPMENT_SURCHARGE.get(e, 0.0))
    return (miles * surcharge_cpm).round(2)


# ------------------------------------------------------------------
# Repositioning penalty  (dest_mls default = 50)
# ------------------------------------------------------------------
@F.pandas_udf(DoubleType())
def udf_repo_penalty(dest_mls: "pd.Series") -> "pd.Series":
    filled = dest_mls.fillna(50.0)
    return (MAX_REPO_PENALTY * (1.0 - filled / 100.0)).round(2)


print("All 8 pandas UDFs registered.")

# COMMAND ----------
# Cell 4 — Apply UDFs: create enriched_loads with all cost columns + net_profit + net_rpm

# Ensure deadhead_miles column exists (default 0 if absent from bronze)
if "deadhead_miles" not in bronze_df.columns:
    working_df = bronze_df.withColumn("deadhead_miles", F.lit(0.0).cast(DoubleType()))
else:
    working_df = bronze_df.withColumn("deadhead_miles", F.col("deadhead_miles").cast(DoubleType()))

# Ensure dest_mls column exists (default 50 if absent)
if "dest_mls" not in working_df.columns:
    working_df = working_df.withColumn("dest_mls", F.lit(50.0).cast(DoubleType()))
else:
    working_df = working_df.withColumn("dest_mls", F.col("dest_mls").cast(DoubleType()))

# Cast core numeric columns for safety
working_df = (
    working_df
    .withColumn("miles", F.col("miles").cast(DoubleType()))
    .withColumn("gross_rate", F.col("gross_rate").cast(DoubleType()))
)

# Apply individual cost UDFs
enriched_loads = (
    working_df
    .withColumn("fuel_cost",           udf_fuel_cost(F.col("miles")))
    .withColumn("driver_pay",          udf_driver_pay(F.col("miles")))
    .withColumn("insurance",           udf_insurance(F.col("miles")))
    .withColumn("maintenance",         udf_maintenance(F.col("miles")))
    .withColumn("tolls",               udf_tolls(F.col("miles")))
    .withColumn("deadhead_cost",       udf_deadhead_cost(F.col("deadhead_miles")))
    .withColumn("equipment_surcharge", udf_equipment_surcharge(F.col("miles"), F.col("equipment_type")))
    .withColumn("repo_penalty",        udf_repo_penalty(F.col("dest_mls")))
)

# Derive total_cost, net_profit, net_rpm from the individual columns
enriched_loads = (
    enriched_loads
    .withColumn(
        "total_cost",
        F.round(
            F.col("fuel_cost") + F.col("driver_pay") + F.col("insurance") +
            F.col("maintenance") + F.col("tolls") + F.col("deadhead_cost") +
            F.col("equipment_surcharge") + F.col("repo_penalty"),
            2,
        ),
    )
    .withColumn(
        "net_profit",
        F.round(F.col("gross_rate") - F.col("total_cost"), 2),
    )
    .withColumn(
        "net_rpm",
        F.when(F.col("miles") > 0, F.round(F.col("net_profit") / F.col("miles"), 4))
         .otherwise(F.lit(0.0)),
    )
)

print(f"Enriched loads schema ({enriched_loads.count():,} rows):")
enriched_loads.printSchema()
enriched_loads.show(5, truncate=False)

# COMMAND ----------
# Cell 5 — Write to freightbrain.silver.enriched_loads (overwrite mode)

(
    enriched_loads
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("freightbrain.silver.enriched_loads")
)

print("Written to freightbrain.silver.enriched_loads (overwrite).")
spark.sql("SELECT COUNT(*) AS row_count FROM freightbrain.silver.enriched_loads").show()

# COMMAND ----------
# Cell 6 — Analytics: avg net_rpm by equipment_type + top 5 most profitable loads

print("=== Average net_rpm by equipment_type ===")
spark.sql("""
    SELECT
        equipment_type,
        COUNT(*) AS load_count,
        ROUND(AVG(net_rpm), 4)   AS avg_net_rpm,
        ROUND(AVG(net_profit), 2) AS avg_net_profit,
        ROUND(AVG(gross_rate), 2) AS avg_gross_rate
    FROM freightbrain.silver.enriched_loads
    GROUP BY equipment_type
    ORDER BY avg_net_rpm DESC
""").show(truncate=False)

print("=== Top 5 most profitable loads ===")
spark.sql("""
    SELECT
        load_id,
        origin_city,
        dest_city,
        equipment_type,
        miles,
        gross_rate,
        total_cost,
        net_profit,
        net_rpm
    FROM freightbrain.silver.enriched_loads
    ORDER BY net_profit DESC
    LIMIT 5
""").show(truncate=False)
