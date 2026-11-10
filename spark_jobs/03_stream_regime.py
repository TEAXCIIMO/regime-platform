"""Structured Streaming job: pull replayed bars from Pub/Sub and update regimes.

Spark has no built-in Pub/Sub source, so a ``rate`` stream drives micro-batches
and each batch pulls available messages with the Pub/Sub client.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from google.api_core.exceptions import DeadlineExceeded
from google.cloud import pubsub_v1
from pyspark.sql import SparkSession

from regime import config as cfg
from regime import panel
from regime.experiment import select_features
from regime.online import OnlineRegime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--subscription", default="regime-bars-sub")
    parser.add_argument("--replay-date", required=True)
    parser.add_argument("--universe", default="core")
    parser.add_argument("--feature-set", default="all")
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--covariance", default="diag")
    parser.add_argument("--trigger-seconds", type=int, default=5)
    parser.add_argument("--max-idle-batches", type=int, default=24, help="Stop after this many empty batches")
    parser.add_argument("--max-wait-batches", type=int, default=240, help="Empty batches allowed before the first message")
    args = parser.parse_args()

    replay_date = pd.Timestamp(args.replay_date)
    experiment = cfg.load_experiment()
    if replay_date > pd.Timestamp(experiment["split"]["val_end"]) and not cfg.holdout_open():
        raise cfg.HoldoutSealedError("Replay dates in the holdout are not allowed while it is sealed.")

    spark = SparkSession.builder.appName("regime-03-stream").getOrCreate()
    universe = cfg.load_universe()
    daily = spark.read.parquet(f"{args.bucket}/curated/daily").toPandas()
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None).dt.normalize()
    comovement = spark.read.parquet(f"{args.bucket}/curated/comovement").toPandas()
    comovement["date"] = pd.to_datetime(comovement["date"]).dt.tz_localize(None).dt.normalize()
    macro = pd.read_parquet(f"{args.bucket}/raw/macro.parquet")

    symbols = panel.universe_symbols(universe, args.universe)
    groups = {g: [s for s in members if s in symbols] for g, members in universe["groups"].items()}
    tables = panel.feature_tables(daily, macro, comovement, universe)
    features = select_features(tables[args.universe], args.feature_set)
    target = panel.market_data(daily, macro, universe["traded_assets"]).spy_return
    daily_subset = daily[daily["symbol"].isin(symbols)]
    model = OnlineRegime.fit(
        features, target, args.n_states, args.covariance, experiment["hmm"]["seeds"], daily_subset, comovement,
        macro, groups, universe["correlation_basket"], replay_date,
    )

    subscriber = pubsub_v1.SubscriberClient()
    path = subscriber.subscription_path(args.project, args.subscription)
    state = {"bars": [], "idle": 0, "seen": False}
    output = f"{args.bucket}/results/streaming/{replay_date:%Y-%m-%d}"

    def process(_batch, batch_id: int) -> None:
        try:
            response = subscriber.pull(request={"subscription": path, "max_messages": 1000}, timeout=args.trigger_seconds)
        except DeadlineExceeded:  # An empty subscription can time out instead of returning nothing.
            response = None
        received = time.time()
        if response is None or not response.received_messages:
            state["idle"] += 1
            return
        state["idle"] = 0
        state["seen"] = True
        published = []
        for item in response.received_messages:
            payload = json.loads(item.message.data.decode("utf-8"))
            published.append(float(payload.pop("published_at")))
            state["bars"].append(payload)
        subscriber.acknowledge(request={"subscription": path, "ack_ids": [m.ack_id for m in response.received_messages]})

        bars = pd.DataFrame(state["bars"])
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
        probabilities = model.update(bars)
        finished = time.time()
        record = {
            "batch_id": batch_id, "event_time": bars["ts"].max().isoformat(), "n_bars": len(bars),
            "new_messages": len(published), "mean_queue_seconds": received - sum(published) / len(published),
            "end_to_end_seconds": finished - min(published), "processing_seconds": finished - received,
            **(probabilities or {}),
        }
        print(json.dumps(record), flush=True)
        pd.DataFrame([record]).to_parquet(f"{output}/batch={batch_id:06d}.parquet", index=False)

    stream = spark.readStream.format("rate").option("rowsPerSecond", 1).load()
    query = stream.writeStream.foreachBatch(process).trigger(processingTime=f"{args.trigger_seconds} seconds").start()
    while query.isActive and state["idle"] < (args.max_idle_batches if state["seen"] else args.max_wait_batches):
        time.sleep(args.trigger_seconds)
    query.stop()
    spark.stop()


if __name__ == "__main__":
    main()
