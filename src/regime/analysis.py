"""Turn sweep output into the tables that answer the research questions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime.validation import stats

HEADLINE = {"execution": "next_open", "cost_bps": 2.0}
FACTORS = ["universe", "feature_set", "n_states", "covariance", "refit"]


def returns_matrix(rows: list[dict]) -> pd.DataFrame:
    """Wide daily return table with one column per strategy."""
    return pd.DataFrame(
        {row["strategy"]: pd.Series(row["returns"], index=pd.to_datetime(row["dates"])) for row in rows}
    ).sort_index()


def headline(metrics: pd.DataFrame, period: str = "dev") -> pd.DataFrame:
    mask = (metrics["period"] == period)
    for key, value in HEADLINE.items():
        mask &= metrics[key] == value
    return metrics[mask]


def rq1_regime_quality(quality: pd.DataFrame) -> pd.DataFrame:
    """Does the feature set change how well HMM signals anticipate volatility?"""
    hmm = quality[quality["signal"] == "hmm"]
    return (
        hmm.groupby(["period", "feature_set"])[["rank_corr_forward_rv", "brier_high_vol", "tercile_return_gap", "avg_confidence"]]
        .mean()
        .reset_index()
    )


def factorial_table(metrics: pd.DataFrame, period: str = "dev") -> pd.DataFrame:
    """Mean net Sharpe for each signal and portfolio rule."""
    data = headline(metrics, period)
    return data.pivot_table(index="signal", columns="rule", values="sharpe", aggfunc="mean")


def factor_effects(metrics: pd.DataFrame, period: str = "dev") -> pd.DataFrame:
    """Mean net Sharpe and information ratio of HMM strategies by model factor."""
    data = headline(metrics, period)
    data = data[data["signal"] == "hmm"]
    rows = []
    for factor in FACTORS:
        grouped = data.groupby(factor)[["sharpe", "information_ratio"]].mean()
        for level, values in grouped.iterrows():
            rows.append({"factor": factor, "level": str(level), **values.to_dict()})
    table = pd.DataFrame(rows)
    table["sharpe_range_within_factor"] = table.groupby("factor")["sharpe"].transform(lambda s: s.max() - s.min())
    return table


def sensitivity(metrics: pd.DataFrame, period: str = "dev") -> pd.DataFrame:
    data = metrics[(metrics["period"] == period) & (metrics["signal"] == "hmm")]
    return data.groupby(["execution", "cost_bps"])[["sharpe", "information_ratio", "annual_turnover"]].mean().reset_index()


def _sharpe(frame: pd.DataFrame) -> pd.Series:
    deviation = frame.std(ddof=1)
    return (frame.mean() / deviation.where(deviation > 0)) * np.sqrt(252)


def nested_walk_forward(returns: pd.DataFrame, first_year: int) -> tuple[pd.Series, pd.DataFrame]:
    """Each year, use the strategy with the best Sharpe over all earlier dates."""
    years = sorted({d.year for d in returns.index if d.year >= first_year})
    pieces, selections = [], []
    for year in years:
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        past = returns.loc[returns.index < cutoff]
        ranked = _sharpe(past).dropna()
        if ranked.empty:
            continue
        choice = str(ranked.idxmax())
        current = returns.loc[returns.index.year == year, choice]
        pieces.append(current)
        selections.append({"year": year, "strategy": choice, "past_sharpe": float(ranked.max())})
    return pd.concat(pieces).rename("selected"), pd.DataFrame(selections)


def overfitting_report(returns: pd.DataFrame, selected: pd.Series, partitions: int = 16) -> dict:
    clean = returns.dropna(axis=1).dropna(axis=0)
    sharpes = _sharpe(clean).to_numpy()
    return {
        **stats.probability_of_backtest_overfitting(clean.to_numpy(), partitions),
        **stats.deflated_sharpe(selected.dropna().to_numpy(), sharpes),
    }


def strategies_for(returns: pd.DataFrame, signal: str) -> pd.DataFrame:
    keep = [c for c in returns.columns if c.split("#")[1] == signal]
    return returns[keep]


def compare_selected(returns: pd.DataFrame, benchmark: pd.Series, first_year: int, resamples: int = 5000) -> dict:
    """Run nested selection for HMM and placebos and compare them with 60/40."""
    selected, log = nested_walk_forward(strategies_for(returns, "hmm"), first_year)
    placebos = {}
    for name in ("placebo_shuffle", "placebo_markov"):
        placebo_returns = strategies_for(returns, name)
        if placebo_returns.shape[1]:
            placebos[name] = nested_walk_forward(placebo_returns, first_year)[0]
    results = {"vs_benchmark": stats.sharpe_difference_ci(selected, benchmark, resamples)}
    for name, series in placebos.items():
        results[f"vs_{name}"] = stats.sharpe_difference_ci(selected, series, resamples)
    return {"comparisons": results, "selection_log": log, "selected": selected, "placebos": placebos}
