"""Daily market-level feature table built from per-symbol daily rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION = 252
GROUP_FEATURES = ["equity_broad", "equity_sector", "international", "rates", "credit", "commodity", "fx"]
DAILY_FEATURES = [
    "spy_ret_1d", "tlt_ret_1d", "gld_ret_1d", "spy_vol_20d", "tlt_vol_20d", "gld_vol_20d",
    "vix_log", "vix_chg_5d", "yc_slope", "credit_ret_5d", "eq_bond_ret_5d",
]
INTRADAY_FEATURES = [f"rv_{g}" for g in GROUP_FEATURES] + [
    "jump_equity", "spy_overnight", "spy_last30", "avg_corr", "absorption", "dispersion", "illiq",
]
ALL_FEATURES = DAILY_FEATURES + INTRADAY_FEATURES
FEATURE_SETS = {"daily": DAILY_FEATURES, "intraday": INTRADAY_FEATURES, "all": ALL_FEATURES}
MIN_GROUP_MEMBERS = 3


def macro_as_known(macro: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Align FRED series to the trading calendar, one trading day late.

    A value dated ``d`` is first usable on the trading day after ``d``.
    """
    wide = macro.pivot_table(index="date", columns="series", values="value").sort_index()
    union = wide.index.union(trading_dates)
    aligned = wide.reindex(union).ffill().reindex(trading_dates)
    return aligned.shift(1)


def _wide(daily: pd.DataFrame, column: str) -> pd.DataFrame:
    return daily.pivot_table(index="date", columns="symbol", values=column).sort_index()


def build_market_features(
    daily: pd.DataFrame,
    macro: pd.DataFrame,
    comovement: pd.DataFrame,
    groups: dict[str, list[str]],
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a date-indexed table; each row uses data through that date only."""
    panel = daily if symbols is None else daily[daily["symbol"].isin(symbols)]
    close = _wide(panel, "close")
    opens = _wide(panel, "open")
    dates = pd.DatetimeIndex(close.index)
    returns = np.log(close).diff()
    rv = _wide(panel, "rv").reindex(dates)
    known = macro_as_known(macro, dates)

    table = pd.DataFrame(index=dates)
    for symbol, name in (("SPY", "spy"), ("TLT", "tlt"), ("GLD", "gld")):
        table[f"{name}_ret_1d"] = returns[symbol]
        table[f"{name}_vol_20d"] = np.log(returns[symbol].rolling(20).std() * np.sqrt(ANNUALIZATION))

    table["vix_log"] = np.log(known["VIXCLS"])
    table["vix_chg_5d"] = np.log(known["VIXCLS"]).diff(5)
    table["yc_slope"] = known["DGS10"] - known["DGS2"]
    table["credit_ret_5d"] = np.log(close["HYG"]).diff(5) - np.log(close["LQD"]).diff(5)
    table["eq_bond_ret_5d"] = np.log(close["SPY"]).diff(5) - np.log(close["TLT"]).diff(5)

    log_rv = np.log((rv * ANNUALIZATION).where(rv > 0))
    for group, members in groups.items():
        present = [s for s in members if s in log_rv.columns]
        if len(present) >= MIN_GROUP_MEMBERS or (present and symbols is not None):
            table[f"rv_{group}"] = log_rv[present].mean(axis=1)
        else:
            table[f"rv_{group}"] = np.nan

    equities = [s for g in ("equity_broad", "equity_sector") for s in groups[g] if s in close.columns]
    jumps = _wide(panel, "jump_frac").reindex(dates)
    table["jump_equity"] = jumps[equities].mean(axis=1)
    table["spy_overnight"] = np.log(opens["SPY"] / close["SPY"].shift(1))
    table["spy_last30"] = _wide(panel, "last30_ret").reindex(dates)["SPY"]

    co = comovement.copy()
    co["date"] = pd.to_datetime(co["date"])
    co = co.set_index("date").sort_index().reindex(dates)
    table["avg_corr"] = co["avg_corr"]
    table["absorption"] = co["absorption"]

    table["dispersion"] = np.log(returns[equities].std(axis=1).where(lambda s: s > 0))
    dollar_volume = _wide(panel, "dollar_volume").reindex(dates)
    illiq = (returns[equities].abs() / dollar_volume[equities].where(lambda f: f > 0)) * 1e9
    table["illiq"] = np.log(illiq.median(axis=1).where(lambda s: s > 0))

    return table[ALL_FEATURES].replace([np.inf, -np.inf], np.nan)
