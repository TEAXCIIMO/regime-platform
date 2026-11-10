"""Publish one historical day of minute bars to Pub/Sub, faster than real time.

Usage:
    python scripts/replay_bars.py --date 2023-10-30 --project PROJECT --speed 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from google.cloud import pubsub_v1  # noqa: E402

from regime import config as cfg  # noqa: E402
from regime import panel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--topic", default="regime-bars")
    parser.add_argument("--universe", default="core")
    parser.add_argument("--speed", type=float, default=60.0, help="Simulated minutes per real minute")
    parser.add_argument("--bars-dir", default=str(ROOT / "data" / "raw" / "bars"))
    args = parser.parse_args()

    day = pd.Timestamp(args.date)
    if day > pd.Timestamp(cfg.load_experiment()["split"]["val_end"]) and not cfg.holdout_open():
        raise cfg.HoldoutSealedError("Replay dates in the holdout are not allowed while it is sealed.")

    frames = []
    for symbol in panel.universe_symbols(cfg.load_universe(), args.universe):
        frame = pd.read_parquet(f"{args.bars_dir}/symbol={symbol}/year={day.year}/bars.parquet")
        if not frame.empty:
            local = frame["ts"].dt.tz_convert("America/New_York")
            frames.append(frame[local.dt.strftime("%Y-%m-%d") == args.date])
    bars = pd.concat(frames, ignore_index=True).sort_values("ts")
    print(f"Replaying {len(bars):,} bars for {args.date} at {args.speed:g}x", flush=True)

    publisher = pubsub_v1.PublisherClient()
    topic = publisher.topic_path(args.project, args.topic)
    for ts, minute in bars.groupby("ts", sort=True):
        started = time.time()
        futures = []
        for row in minute.to_dict("records"):
            row["ts"] = ts.isoformat()
            row["published_at"] = time.time()
            futures.append(publisher.publish(topic, json.dumps(row).encode("utf-8")))
        for future in futures:
            future.result(timeout=60)
        time.sleep(max(0.0, 60.0 / args.speed - (time.time() - started)))
    print("Replay finished.")


if __name__ == "__main__":
    main()
