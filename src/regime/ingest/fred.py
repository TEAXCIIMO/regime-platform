"""Download daily macro series from FRED's public CSV endpoint."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_series(series_id: str, start: str) -> pd.DataFrame:
    response = requests.get(CSV_URL, params={"id": series_id, "cosd": start}, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame.insert(1, "series", series_id)
    return frame.dropna(subset=["value"]).reset_index(drop=True)


def download_macro(series: list[str], start: str, out_path: str | Path) -> pd.DataFrame:
    table = pd.concat([fetch_series(s, start) for s in series], ignore_index=True)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(target, index=False)
    return table
