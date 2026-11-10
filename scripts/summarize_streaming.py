"""Summarize latency and probabilities from a streaming replay.

Usage:
    python scripts/summarize_streaming.py --date 2023-10-30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    folder = ROOT / "results" / "runs" / "streaming" / args.date
    batches = pd.concat([pd.read_parquet(p) for p in sorted(folder.glob("batch=*.parquet"))], ignore_index=True)
    batches = batches.sort_values("batch_id").reset_index(drop=True)
    probability_columns = [c for c in batches.columns if c.startswith("p") and c[1:].isdigit()]
    with_output = batches.dropna(subset=probability_columns) if probability_columns else batches.iloc[0:0]

    def quantiles(column: str) -> dict:
        values = batches[column]
        return {"p50": float(values.median()), "p95": float(values.quantile(0.95)), "max": float(values.max())}

    summary = {
        "date": args.date,
        "batches": int(len(batches)),
        "messages": int(batches["new_messages"].sum()),
        "batches_with_probabilities": int(len(with_output)),
        "first_probability_event_time": with_output["event_time"].iloc[0] if len(with_output) else None,
        "end_to_end_seconds": quantiles("end_to_end_seconds"),
        "queue_seconds": quantiles("mean_queue_seconds"),
        "processing_seconds": quantiles("processing_seconds"),
        "final_probabilities": {c: float(with_output[c].iloc[-1]) for c in probability_columns} if len(with_output) else None,
    }
    out = ROOT / "results" / "tables" / "streaming"
    out.mkdir(parents=True, exist_ok=True)
    batches.to_csv(out / f"{args.date}_batches.csv", index=False)
    (out / f"{args.date}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
