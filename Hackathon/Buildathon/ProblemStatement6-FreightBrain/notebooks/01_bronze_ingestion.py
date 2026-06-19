# Databricks notebook source
# FreightBrain — Bronze Layer Ingestion
# Parses raw text files and writes the `freightbrain.bronze.loads` Delta table.
#
# Compatible with Databricks Community Edition (no streaming, no Jobs API).
# Run cells top-to-bottom in a cluster with Python 3.9+ and Delta Lake enabled.

# COMMAND ----------

# ── 0. Dependencies ──────────────────────────────────────────────────────────
# pandas is pre-installed on Databricks; install nothing extra for CE.
# The src/ directory must be on the path (see cell below).

import sys
import os
from datetime import datetime, timezone

# COMMAND ----------

# ── 1. Add project src/ to Python path ───────────────────────────────────────
# Adjust REPO_ROOT to wherever the repo is mounted in your Databricks workspace.
# On DBFS repos it is typically /Workspace/Repos/<user>/FreightBrain or similar.
REPO_ROOT = "/Workspace/Users/tallurisaiteja143@gmail.com/Lone-Rangers-GenAI-repo/Hackathon/Buildathon/ProblemStatement6-FreightBrain"

SRC_DIR = os.path.join(REPO_ROOT, "src")
DATA_DIR = os.path.join(REPO_ROOT, "data", "text")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print(f"Source dir : {SRC_DIR}")
print(f"Data dir   : {DATA_DIR}")

# COMMAND ----------

# ── 2. Parse raw text files into a pandas DataFrame ───────────────────────────
from parser import parse_all_files

print("Parsing load files …")
pdf = parse_all_files(DATA_DIR)
print(f"Parsed {len(pdf):,} records")
pdf.head(3)

# COMMAND ----------

# ── 3. Add ingestion metadata ─────────────────────────────────────────────────
ingested_at = datetime.now(timezone.utc).isoformat()
pdf["ingested_at"] = ingested_at

print(f"Stamped ingested_at = {ingested_at}")

# COMMAND ----------

# ── 4. Convert to Spark DataFrame ─────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, IntegerType, LongType,
)

spark = SparkSession.builder.getOrCreate()

sdf = spark.createDataFrame(pdf)

# Cast numeric columns to proper Spark types
sdf = (
    sdf
    .withColumn("origin_lat",      F.col("origin_lat").cast("double"))
    .withColumn("origin_lon",      F.col("origin_lon").cast("double"))
    .withColumn("dest_lat",        F.col("dest_lat").cast("double"))
    .withColumn("dest_lon",        F.col("dest_lon").cast("double"))
    .withColumn("weight_lbs",      F.col("weight_lbs").cast("long"))
    .withColumn("distance_miles",  F.col("distance_miles").cast("double"))
    .withColumn("rate_usd",        F.col("rate_usd").cast("double"))
    .withColumn("rate_per_mile",   F.col("rate_per_mile").cast("double"))
    .withColumn("ingested_at",     F.col("ingested_at").cast("timestamp"))
)

print(f"Spark DataFrame: {sdf.count():,} rows  |  {len(sdf.columns)} columns")

# COMMAND ----------

# ── 5. Create catalog / schema if not present ─────────────────────────────────
spark.sql("CREATE CATALOG IF NOT EXISTS freightbrain")
spark.sql("CREATE SCHEMA IF NOT EXISTS freightbrain.bronze")

print("Catalog freightbrain.bronze is ready.")

# COMMAND ----------

# ── 6. Write to Delta — full replace on each run (idempotent) ─────────────────
TABLE = "freightbrain.bronze.loads"

(
    sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE)
)

print(f"Written to Delta table: {TABLE}")

# COMMAND ----------

# ── 7. Verify row count and sample ────────────────────────────────────────────
count = spark.table(TABLE).count()
print(f"Row count in {TABLE}: {count:,}")

spark.table(TABLE).limit(10).display()

# COMMAND ----------

# ── 8. Quality summary ────────────────────────────────────────────────────────
from pyspark.sql import functions as F

tbl = spark.table(TABLE)

quality = (
    tbl.select(
        F.count("*").alias("total_rows"),
        F.count("load_id").alias("non_null_load_id"),
        F.countDistinct("equipment_type").alias("distinct_equip_types"),
        F.avg("rate_usd").alias("avg_rate_usd"),
        F.avg("distance_miles").alias("avg_distance_miles"),
        F.sum(F.when(F.col("origin_lat").isNull(), 1).otherwise(0)).alias("missing_origin_coords"),
        F.sum(F.when(F.col("dest_lat").isNull(), 1).otherwise(0)).alias("missing_dest_coords"),
        F.sum(F.when(F.col("weight_lbs").isNull(), 1).otherwise(0)).alias("missing_weight"),

    )
)

quality.display()

# COMMAND ----------

# ── 9. Equipment-type distribution ───────────────────────────────────────────
(
    tbl
    .groupBy("equipment_type")
    .agg(
        F.count("*").alias("count"),
        F.round(F.avg("rate_usd"), 2).alias("avg_rate_usd"),
        F.round(F.avg("distance_miles"), 1).alias("avg_miles"),
    )
    .orderBy(F.desc("count"))
    .display()
)

# COMMAND ----------

# ── 10. Partition info ────────────────────────────────────────────────────────
spark.sql(f"DESCRIBE DETAIL {TABLE}").select(
    "name", "location", "numFiles", "sizeInBytes", "format"
).display()

print("Bronze ingestion complete.")
