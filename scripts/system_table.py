"""Build the scaling table from the Dataproc job logs kept in data/.

The job statistics do not record the worker count, so it is taken from the
log file names written by the scaling runs (job01_w4.log, job02_w7.log, ...).
The 7-worker job01 run is the full run in job01_full.log.

Usage:
    python scripts/system_table.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data"
LOG_NAMES = {
    ("01 bars to daily (103M rows)", 7): "job01_full.log",
    ("01 bars to daily (103M rows)", 4): "job01_w4.log",
    ("01 bars to daily (103M rows)", 2): "job01_w2.log",
    ("02 sweep (16 configs)", 7): "job02_w7.log",
    ("02 sweep (16 configs)", 4): "job02_w4.log",
    ("02 sweep (16 configs)", 2): "job02_w2.log",
    ("02 sweep (84 configs)", 7): "job02_full.log",
}


def job_stats(path: Path) -> dict:
    # gcloud wraps long lines in the log, so join them before parsing.
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    joined = re.sub(r"\n(?=[^\n{]*[\",:}\]])", "", text)
    for line in joined.splitlines():
        if line.startswith('{"job"'):
            return json.loads(line[: line.rindex("}") + 1])
    raise ValueError(f"No job statistics in {path}")


def main() -> None:
    rows = []
    for (job, workers), name in LOG_NAMES.items():
        stats = job_stats(LOGS / name)
        rows.append({"job": job, "workers": workers, "worker_vcpus": 2 * workers, "seconds": stats["seconds"]})
    table = pd.DataFrame(rows)
    base = table[table["workers"] == 2].set_index("job")["seconds"]
    table["speedup_vs_2"] = table.apply(lambda r: base.get(r["job"], float("nan")) / r["seconds"], axis=1).round(2)
    out = ROOT / "results" / "tables" / "system"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "scaling.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
