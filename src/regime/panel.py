"""Build feature tables and market data from per-symbol daily rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime import config as cfg
from regime.experiment import MarketData
from regime.features.market import build_market_features
from regime.portfolio.backtest import daily_risk_free


def universe_symbols(universe: dict, name: str) -> list[str]:
    if name == "full":
        return cfg.full_universe(universe)
    return universe[f"{name}_universe"]


def feature_tables(daily: pd.DataFrame, macro: pd.DataFrame, comovement: pd.DataFrame, universe: dict) -> dict[str, pd.DataFrame]:
    """One market-feature table for each universe in the experiment."""
    tables = {}
    for name in ("minimal", "core", "full"):
        symbols = universe_symbols(universe, name)
        groups = {g: [s for s in members if s in symbols] for g, members in universe["groups"].items()}
        tables[name] = build_market_features(daily, macro, comovement, groups, symbols)
    return tables


def market_data(daily: pd.DataFrame, macro: pd.DataFrame, assets: list[str]) -> MarketData:
    traded = daily[daily["symbol"].isin(assets)]
    prices = {
        field: traded.pivot_table(index="date", columns="symbol", values=field).sort_index()
        for field in ("open", "vwap", "close")
    }
    panel = pd.concat(prices, axis=1).dropna()
    spy = daily[daily["symbol"] == "SPY"].set_index("date").sort_index()
    bills = macro[macro["series"] == "DTB3"].set_index("date")["value"].sort_index()
    known = bills.reindex(bills.index.union(panel.index)).ffill().reindex(panel.index).shift(1)
    return MarketData(
        prices=panel,
        risk_free=daily_risk_free(known),
        spy_return=np.log(spy["close"]).diff().fillna(0.0),
        spy_rv=spy["rv"],
    )
