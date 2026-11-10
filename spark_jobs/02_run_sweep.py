"""Run the walk-forward model and backtest sweep in parallel."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import pandas as pd
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from regime import config as cfg  # noqa: E402
from regime import experiment as exp  # noqa: E402
from regime import panel  # noqa: E402


def load_inputs(spark: SparkSession, bucket: str, universe: dict) -> tuple[dict, exp.MarketData, pd.DataFrame]:
    daily = spark.read.parquet(f"{bucket}/curated/daily").toPandas()
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None).dt.normalize()
    comovement = spark.read.parquet(f"{bucket}/curated/comovement").toPandas()
    comovement["date"] = pd.to_datetime(comovement["date"]).dt.tz_localize(None).dt.normalize()
    macro = pd.read_parquet(f"{bucket}/raw/macro.parquet")
    # Workers receive nothing after the validation end, so they cannot touch the holdout.
    split = cfg.load_experiment()["split"]
    tables = {name: cfg.seal(table, split) for name, table in panel.feature_tables(daily, macro, comovement, universe).items()}
    full = panel.market_data(daily, macro, universe["traded_assets"])
    market = exp.MarketData(
        prices=cfg.seal(full.prices, split),
        risk_free=cfg.seal(full.risk_free.to_frame("r"), split)["r"],
        spy_return=cfg.seal(full.spy_return.to_frame("r"), split)["r"],
        spy_rv=cfg.seal(full.spy_rv.to_frame("r"), split)["r"],
    )
    return tables, market, macro


def run_partition(configs, tables_bc, market_bc, experiment_bc):
    for config_dict in configs:
        config = exp.SignalConfig(**config_dict)
        started = time.time()
        try:
            result = exp.evaluate_config(config, tables_bc.value[config.universe], market_bc.value, experiment_bc.value)
        except Exception as error:  # A failed model must be visible, not fatal to the sweep.
            yield Row(kind="error", payload=json.dumps({"config": config.key, "error": repr(error)}))
            continue
        yield Row(kind="stats", payload=json.dumps({
            "config": config.key, "n_fits": result["n_fits"], "seconds": round(time.time() - started, 1),
            "n_metric_rows": len(result["metrics"]), "feature_columns": result["feature_columns"],
        }))
        for row in result["metrics"]:
            yield Row(kind="metric", payload=json.dumps(row))
        for row in result["quality"]:
            yield Row(kind="quality", payload=json.dumps(row))
        for row in result["returns"]:
            yield Row(kind="returns", payload=json.dumps(row))
        for name, frame in result["signal_table"].items():
            yield Row(kind="signal", payload=json.dumps({
                "config": config.key, "signal": name, "columns": list(frame.columns),
                "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
                "values": frame.to_numpy().round(6).tolist(),
            }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--limit-configs", type=int, default=0, help="Run only the first N configurations")
    parser.add_argument("--run-name", default="full")
    args = parser.parse_args()

    started = time.time()
    spark = SparkSession.builder.appName(f"regime-02-sweep-{args.run_name}").getOrCreate()
    context = spark.sparkContext
    universe = cfg.load_universe()
    experiment = cfg.load_experiment()

    tables, market, _ = load_inputs(spark, args.bucket, universe)
    configs = [asdict(c) for c in exp.signal_configs(experiment)]
    if args.limit_configs:
        configs = configs[: args.limit_configs]

    tables_bc = context.broadcast(tables)
    market_bc = context.broadcast(market)
    experiment_bc = context.broadcast(experiment)
    outputs = f"{args.bucket}/results/{args.run_name}"

    rdd = context.parallelize(configs, numSlices=len(configs)).mapPartitions(
        lambda items: run_partition(items, tables_bc, market_bc, experiment_bc)
    )
    frame = spark.createDataFrame(rdd).cache()
    frame.write.mode("overwrite").partitionBy("kind").json(f"{outputs}/raw")

    counts = {row["kind"]: row["count"] for row in frame.groupBy("kind").count().collect()}
    n_fits = sum(json.loads(r["payload"])["n_fits"] for r in frame.filter(F.col("kind") == "stats").collect())
    stats = {
        "job": "02_run_sweep", "run_name": args.run_name, "n_configs": len(configs), "n_hmm_fits": n_fits,
        "counts": counts, "seconds": round(time.time() - started, 1),
        "default_parallelism": context.defaultParallelism,
    }
    print(json.dumps(stats))
    spark.createDataFrame([Row(payload=json.dumps(stats))]).coalesce(1).write.mode("append").json(f"{args.bucket}/system/job_stats")
    spark.stop()


if __name__ == "__main__":
    main()
