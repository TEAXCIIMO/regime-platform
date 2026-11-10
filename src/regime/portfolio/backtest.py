"""Daily backtest with next-day execution, drift rebalancing and costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANNUALIZATION = 252
EXECUTION_PRICE = {"next_open": "open", "next_vwap": "vwap", "next_close": "close"}


@dataclass(frozen=True)
class BacktestResult:
    gross: pd.Series
    traded: pd.Series
    weights: pd.DataFrame

    def net(self, cost_bps: float) -> pd.Series:
        return self.gross - self.traded * cost_bps / 10_000.0


def execution_returns(prices: pd.DataFrame, execution: str) -> pd.DataFrame:
    """Return earned from execution on date ``d`` to execution on the next date.

    ``prices`` has a (field, symbol) column index. The returned row labelled
    ``d`` is the holding-period return of a position entered at ``d``.
    """
    field = EXECUTION_PRICE[execution]
    series = prices[field]
    return (series.shift(-1) / series - 1.0).iloc[:-1]


def run_backtest(signal_weights: pd.DataFrame, prices: pd.DataFrame, execution: str) -> BacktestResult:
    """Simulate weights dated at signal time ``t`` and executed on ``t+1``.

    The result is indexed by the execution date.
    """
    returns = execution_returns(prices, execution)[list(signal_weights.columns)].dropna()
    calendar = pd.DatetimeIndex(prices.index)
    position = calendar.get_indexer(signal_weights.index)
    if (position < 0).any():
        raise ValueError("Signal dates must be on the price calendar.")
    next_position = position + 1
    keep = next_position < len(calendar)
    execution_dates = calendar[next_position[keep]]
    targets = pd.DataFrame(
        signal_weights.to_numpy()[keep], index=execution_dates, columns=signal_weights.columns
    )
    targets = targets.loc[targets.index.intersection(returns.index)]
    period = returns.loc[targets.index].to_numpy()
    weights = targets.to_numpy()

    gross = (weights * period).sum(axis=1)
    drifted = np.vstack([weights[0], weights[:-1] * (1.0 + period[:-1])])
    drifted[1:] /= (1.0 + gross[:-1])[:, None]
    traded = np.abs(weights - drifted).sum(axis=1)
    # The opening trade is common to every strategy and is not a turnover signal.
    traded[0] = 0.0

    return BacktestResult(
        pd.Series(gross, index=targets.index, name="gross"),
        pd.Series(traded, index=targets.index, name="traded"),
        targets,
    )


def constant_weights(index: pd.DatetimeIndex, weights: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({k: float(v) for k, v in weights.items()}, index=index)


def performance(returns: pd.Series, risk_free: pd.Series) -> dict[str, float]:
    """Annualized metrics of a daily return series in excess of T-bills."""
    excess = returns - risk_free.reindex(returns.index).ffill().fillna(0.0)
    volatility = float(excess.std(ddof=1))
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1.0
    return {
        "observations": int(len(returns)),
        "annual_return": float(nav.iloc[-1] ** (ANNUALIZATION / len(returns)) - 1.0),
        "annual_volatility": float(returns.std(ddof=1) * np.sqrt(ANNUALIZATION)),
        "sharpe": float(excess.mean() / volatility * np.sqrt(ANNUALIZATION)) if volatility > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
    }


def information_ratio(returns: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    active = (returns - benchmark.reindex(returns.index)).dropna()
    tracking = float(active.std(ddof=1))
    return {
        "active_return": float(active.mean() * ANNUALIZATION),
        "tracking_error": float(tracking * np.sqrt(ANNUALIZATION)),
        "information_ratio": float(active.mean() / tracking * np.sqrt(ANNUALIZATION)) if tracking > 0 else np.nan,
    }


def daily_risk_free(annual_percent: pd.Series) -> pd.Series:
    """Convert a T-bill yield in percent per year to a daily rate."""
    return annual_percent / 100.0 / ANNUALIZATION
