"""Download FRED series and Alpaca minute bars to data/raw.

Usage:
    python scripts/download_data.py macro
    python scripts/download_data.py check
    python scripts/download_data.py bars [--symbols SPY,TLT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from regime import config as cfg  # noqa: E402
from regime.ingest import alpaca, fred  # noqa: E402


def check_feed() -> None:
    """Compare the sip and iex feeds for one past day of SPY and one thin ETF."""
    import requests

    session = requests.Session()
    session.headers.update(alpaca.auth_headers())
    limiter = alpaca.RateLimiter(100)
    for symbol in ("SPY", "INDA"):
        for feed in ("sip", "iex"):
            try:
                frame = alpaca.fetch_symbol(session, symbol, "2023-06-14T13:30:00Z", "2023-06-14T20:00:00Z", limiter, feed)
            except requests.HTTPError as error:
                print(f"{symbol} {feed}: HTTP {error.response.status_code}")
                continue
            volume = frame["volume"].sum() if not frame.empty else 0
            print(f"{symbol} {feed}: {len(frame)} bars, volume {volume:,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["macro", "check", "bars"])
    parser.add_argument("--symbols", default="")
    parser.add_argument("--out", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--feed", default="sip")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    universe = cfg.load_universe()
    experiment = cfg.load_experiment()
    out = Path(args.out)

    if args.what == "macro":
        table = fred.download_macro(universe["macro_series"], experiment["data"]["start"], out / "macro.parquet")
        summary = table.groupby("series")["date"].agg(["count", "min", "max"])
        print(summary.to_string())
    elif args.what == "check":
        check_feed()
    else:
        symbols = args.symbols.split(",") if args.symbols else cfg.full_universe(universe)
        start_year = pd.Timestamp(experiment["data"]["start"]).year
        report = alpaca.download_universe(symbols, start_year, experiment["data"]["end"], out / "bars", feed=args.feed)
        print(f"Downloaded {sum(r['rows'] for r in report):,} rows in {len(report)} new files.")


if __name__ == "__main__":
    main()
