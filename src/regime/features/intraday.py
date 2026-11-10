"""Per-symbol intraday features.

These are plain pandas functions so that Spark can call them through
``applyInPandas`` and unit tests can call them directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_MINUTES = 390
BUCKET_MINUTES = 5
MIN_SESSION_MINUTES = 180
MU1_SQUARED = 2.0 / np.pi

DAILY_COLUMNS = [
    "symbol", "date", "open", "high", "low", "close", "volume", "dollar_volume", "vwap",
    "n_minutes", "rv", "bv", "jump_frac", "first30_ret", "last30_ret",
]
BUCKET_COLUMNS = ["symbol", "date", "bucket", "ret"]


def regular_session(bars: pd.DataFrame) -> pd.DataFrame:
    """Keep 09:30-16:00 New York bars and add ``date`` and session ``minute``."""
    local = pd.to_datetime(bars["ts"], utc=True).dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute - 570
    keep = (minute >= 0) & (minute < SESSION_MINUTES)
    out = bars.loc[keep].copy()
    # Build a timezone-free midnight so the column matches across engines.
    out["date"] = pd.to_datetime(local[keep].dt.strftime("%Y-%m-%d"))
    out["minute"] = minute[keep].astype(int)
    return out.sort_values(["date", "minute"]).reset_index(drop=True)


def bucket_returns(session: pd.DataFrame) -> pd.DataFrame:
    """Log returns between the last closes of consecutive 5-minute buckets."""
    data = session.assign(bucket=session["minute"] // BUCKET_MINUTES)
    last = data.groupby(["symbol", "date", "bucket"], sort=True)["close"].last().reset_index()
    last["ret"] = np.log(last["close"]).groupby([last["symbol"], last["date"]]).diff()
    return last.dropna(subset=["ret"])[BUCKET_COLUMNS].reset_index(drop=True)


def _variation(returns: pd.DataFrame) -> pd.DataFrame:
    def per_day(values: np.ndarray) -> pd.Series:
        rv = float(np.sum(values**2))
        if len(values) > 1:
            bv = float(np.sum(np.abs(values[1:]) * np.abs(values[:-1])) / MU1_SQUARED)
        else:
            bv = rv
        return pd.Series({"rv": rv, "bv": bv})

    rows = [
        {"date": date, **per_day(group["ret"].to_numpy()).to_dict()}
        for date, group in returns.groupby("date", sort=True)
    ]
    frame = pd.DataFrame(rows, columns=["date", "rv", "bv"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(frame["rv"] > 0, (frame["rv"] - frame["bv"]) / frame["rv"], 0.0)
    frame["jump_frac"] = np.clip(ratio, 0.0, 1.0)
    return frame


def daily_features(bars: pd.DataFrame, min_minutes: int = MIN_SESSION_MINUTES) -> pd.DataFrame:
    """Aggregate one symbol's minute bars into one row per trading day."""
    session = regular_session(bars)
    if session.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    session["notional"] = session["vwap"].fillna(session["close"]) * session["volume"]
    grouped = session.groupby(["symbol", "date"], sort=True)
    daily = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        dollar_volume=("notional", "sum"),
        n_minutes=("minute", "count"),
    ).reset_index()
    daily = daily[daily["n_minutes"] >= min_minutes].copy()
    if daily.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    with np.errstate(divide="ignore", invalid="ignore"):
        daily["vwap"] = daily["dollar_volume"] / daily["volume"].replace(0, np.nan)
    daily["vwap"] = daily["vwap"].fillna(daily["close"])

    variation = _variation(bucket_returns(session))
    daily = daily.merge(variation, on="date", how="left")

    first30 = session[session["minute"] < 30].groupby("date")["close"].last()
    last30_open = session[session["minute"] < 360].groupby("date")["close"].last()
    daily = daily.set_index("date")
    daily["first30_ret"] = np.log(first30.reindex(daily.index) / daily["open"])
    daily["last30_ret"] = np.log(daily["close"] / last30_open.reindex(daily.index))
    return daily.reset_index()[DAILY_COLUMNS]


def basket_comovement(bucket_frame: pd.DataFrame) -> pd.DataFrame:
    """One trading day of average pairwise correlation and absorption ratio.

    The absorption ratio is the share of variance explained by the first
    principal component of the basket's 5-minute returns.
    """
    date = bucket_frame["date"].iloc[0]
    wide = bucket_frame.pivot_table(index="bucket", columns="symbol", values="ret").dropna(axis=1, thresh=30).dropna()
    empty = pd.DataFrame({"date": [date], "avg_corr": [np.nan], "absorption": [np.nan], "n_assets": [0]})
    if wide.shape[1] < 3 or wide.shape[0] < 30:
        return empty
    wide = wide.loc[:, wide.std() > 0]
    if wide.shape[1] < 3:
        return empty
    corr = np.corrcoef(wide.to_numpy(), rowvar=False)
    n = corr.shape[0]
    average = float((corr.sum() - n) / (n * (n - 1)))
    eigenvalues = np.linalg.eigvalsh(corr)
    return pd.DataFrame(
        {"date": [date], "avg_corr": [average], "absorption": [float(eigenvalues[-1] / n)], "n_assets": [n]}
    )
