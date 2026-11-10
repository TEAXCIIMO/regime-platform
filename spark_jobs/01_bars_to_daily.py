"""Clean minute bars and compute per-symbol daily and basket co-movement features."""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType, TimestampType

from regime import config as cfg
from regime.features import intraday

DAILY_SCHEMA = StructType([
    StructField("symbol", StringType()), StructField("date", TimestampType()),
    *[StructField(c, DoubleType()) for c in ("open", "high", "low", "close", "volume", "dollar_volume", "vwap")],
    StructField("n_minutes", IntegerType()),
    *[StructField(c, DoubleType()) for c in ("rv", "bv", "jump_frac", "first30_ret", "last30_ret")],
])
BUCKET_SCHEMA = StructType([
    StructField("symbol", StringType()), StructField("date", TimestampType()),
    StructField("bucket", IntegerType()), StructField("ret", DoubleType()),
])
COMOVEMENT_SCHEMA = StructType([
    StructField("date", TimestampType()), StructField("avg_corr", DoubleType()),
    StructField("absorption", DoubleType()), StructField("n_assets", IntegerType()),
])


def daily_udf(bars: pd.DataFrame) -> pd.DataFrame:
    result = intraday.daily_features(bars)
    return result.astype({"n_minutes": "int32"}) if not result.empty else result


def buckets_udf(bars: pd.DataFrame) -> pd.DataFrame:
    session = intraday.regular_session(bars)
    if session.empty:
        return pd.DataFrame(columns=intraday.BUCKET_COLUMNS)
    return intraday.bucket_returns(session).astype({"bucket": "int32"})


def comovement_udf(frame: pd.DataFrame) -> pd.DataFrame:
    return intraday.basket_comovement(frame).astype({"n_assets": "int32"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="GCS bucket or local root, for example gs://name")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Fraction of symbol-years to use")
    args = parser.parse_args()

    started = time.time()
    spark = SparkSession.builder.appName("regime-01-bars-to-daily").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    universe = cfg.load_universe()

    bars = spark.read.option("basePath", f"{args.bucket}/raw/bars").parquet(f"{args.bucket}/raw/bars/symbol=*/year=*/*.parquet")
    if args.sample_fraction < 1.0:
        # Cached so that every later action sees the same sampled symbol-years.
        keys = bars.select("symbol", "year").distinct().sample(False, args.sample_fraction, seed=2026).cache()
        bars = bars.join(keys, ["symbol", "year"])

    clean = (
        bars.withColumn("ts", F.col("ts").cast("timestamp"))
        .dropDuplicates(["symbol", "ts"])
        .filter((F.col("close") > 0) & (F.col("open") > 0) & (F.col("high") >= F.col("low")) & (F.col("volume") >= 0))
        .select("symbol", "year", "ts", "open", "high", "low", "close", "volume", "vwap")
    ).cache()
    raw_rows = bars.count()
    clean_rows = clean.count()

    daily = clean.groupBy("symbol", "year").applyInPandas(daily_udf, DAILY_SCHEMA)
    daily.write.mode("overwrite").partitionBy("symbol").parquet(f"{args.bucket}/curated/daily")

    basket = universe["correlation_basket"]
    buckets = clean.filter(F.col("symbol").isin(basket)).groupBy("symbol", "year").applyInPandas(buckets_udf, BUCKET_SCHEMA)
    buckets.groupBy("date").applyInPandas(comovement_udf, COMOVEMENT_SCHEMA).write.mode("overwrite").parquet(
        f"{args.bucket}/curated/comovement"
    )

    written = spark.read.parquet(f"{args.bucket}/curated/daily").count()
    stats = {
        "job": "01_bars_to_daily", "raw_rows": raw_rows, "clean_rows": clean_rows, "daily_rows": written,
        "seconds": round(time.time() - started, 1),
        "executors": spark.sparkContext.getConf().get("spark.executor.instances", "dynamic"),
        "sample_fraction": args.sample_fraction,
    }
    print(json.dumps(stats))
    spark.createDataFrame([stats]).coalesce(1).write.mode("append").json(f"{args.bucket}/system/job_stats")
    spark.stop()


if __name__ == "__main__":
    main()
