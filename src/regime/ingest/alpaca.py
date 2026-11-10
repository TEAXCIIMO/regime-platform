"""Resumable download of 1-minute bars from the Alpaca market data API."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
PAGE_LIMIT = 10_000
COLUMNS = {"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "trades"}


class RateLimiter:
    def __init__(self, calls_per_minute: int) -> None:
        self.interval = 60.0 / calls_per_minute
        self.next_time = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self.next_time:
            time.sleep(self.next_time - now)
        self.next_time = max(now, self.next_time) + self.interval


def auth_headers() -> dict[str, str]:
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def request_page(session: requests.Session, params: dict, limiter: RateLimiter, retries: int = 6) -> dict:
    for attempt in range(retries):
        limiter.wait()
        response = session.get(BARS_URL, params=params, timeout=60)
        if response.status_code == 200:
            return response.json()
        if response.status_code in {429, 500, 502, 503, 504}:
            time.sleep(min(60, 2**attempt))
            continue
        response.raise_for_status()
    raise RuntimeError(f"Request failed after {retries} attempts: {params.get('symbols')}")


def bars_to_frame(symbol: str, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).rename(columns=COLUMNS)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame.insert(0, "symbol", symbol)
    return frame[["symbol", "ts", "open", "high", "low", "close", "volume", "vwap", "trades"]]


def fetch_symbol(
    session: requests.Session,
    symbol: str,
    start: str,
    end: str,
    limiter: RateLimiter,
    feed: str = "sip",
) -> pd.DataFrame:
    params = {
        "symbols": symbol,
        "timeframe": "1Min",
        "start": start,
        "end": end,
        "limit": PAGE_LIMIT,
        "adjustment": "all",
        "feed": feed,
        "sort": "asc",
    }
    rows: list[dict] = []
    while True:
        payload = request_page(session, params, limiter)
        rows.extend((payload.get("bars") or {}).get(symbol, []))
        token = payload.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not rows:
        return pd.DataFrame()
    return bars_to_frame(symbol, rows)


def download_universe(
    symbols: list[str],
    start_year: int,
    end_date: str,
    out_dir: str | Path,
    calls_per_minute: int = 180,
    feed: str = "sip",
) -> list[dict]:
    """Download each symbol-year to Parquet, skipping files already present."""
    out = Path(out_dir)
    end = pd.Timestamp(end_date)
    limiter = RateLimiter(calls_per_minute)
    session = requests.Session()
    session.headers.update(auth_headers())
    report: list[dict] = []

    for symbol in symbols:
        for year in range(start_year, end.year + 1):
            target = out / f"symbol={symbol}" / f"year={year}" / "bars.parquet"
            if target.exists():
                continue
            year_end = min(pd.Timestamp(year=year, month=12, day=31), end)
            frame = fetch_symbol(
                session,
                symbol,
                f"{year}-01-01T00:00:00Z",
                year_end.strftime("%Y-%m-%dT23:59:59Z"),
                limiter,
                feed,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            # An empty file marks a symbol-year with no listing, so it is not retried.
            frame.to_parquet(target, index=False)
            report.append({"symbol": symbol, "year": year, "rows": len(frame)})
            print(f"{symbol} {year}: {len(frame):,} rows", flush=True)
    return report
