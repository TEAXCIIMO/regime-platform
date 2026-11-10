"""Stationary bootstrap, CSCV overfitting probability and Deflated Sharpe."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ANNUALIZATION = 252
EULER_GAMMA = 0.5772156649015329


def annualized_sharpe(returns: np.ndarray) -> float:
    deviation = np.std(returns, ddof=1)
    return float(np.mean(returns) / deviation * np.sqrt(ANNUALIZATION)) if deviation > 0 else np.nan


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One Politis-Romano stationary bootstrap index path of length ``n``."""
    restart = rng.random(n) < 1.0 / mean_block
    restart[0] = True
    start_positions = rng.integers(0, n, size=int(restart.sum()))
    block_start = np.flatnonzero(restart)
    offsets = np.arange(n) - block_start[np.searchsorted(block_start, np.arange(n), side="right") - 1]
    origin = np.zeros(n, dtype=int)
    origin[restart] = start_positions
    origin = origin[block_start[np.searchsorted(block_start, np.arange(n), side="right") - 1]]
    return (origin + offsets) % n


def sharpe_difference_ci(
    strategy: pd.Series,
    benchmark: pd.Series,
    resamples: int = 5000,
    mean_block: float = 21.0,
    alpha: float = 0.05,
    seed: int = 2026,
) -> dict[str, float]:
    """Stationary block bootstrap interval for Sharpe(strategy) - Sharpe(benchmark)."""
    joined = pd.concat([strategy, benchmark], axis=1, join="inner").dropna().to_numpy()
    a, b = joined[:, 0], joined[:, 1]
    n = len(a)
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for i in range(resamples):
        index = stationary_indices(n, mean_block, rng)
        differences[i] = annualized_sharpe(a[index]) - annualized_sharpe(b[index])
    return {
        "difference": annualized_sharpe(a) - annualized_sharpe(b),
        "ci_low": float(np.quantile(differences, alpha / 2)),
        "ci_high": float(np.quantile(differences, 1 - alpha / 2)),
        "share_above_zero": float(np.mean(differences > 0)),
    }


def probability_of_backtest_overfitting(matrix: np.ndarray, partitions: int = 16) -> dict[str, float]:
    """Combinatorially symmetric cross-validation (Bailey et al.).

    ``matrix`` holds one row per day and one column per strategy. The result
    is the share of train/test splits whose in-sample best strategy ranks
    below the out-of-sample median.
    """
    if partitions % 2:
        raise ValueError("Partitions must be even.")
    t_rows, n_strategies = matrix.shape
    groups = np.array_split(np.arange(t_rows - t_rows % partitions), partitions)
    counts = np.array([len(g) for g in groups], dtype=float)
    sums = np.vstack([matrix[g].sum(axis=0) for g in groups])
    squares = np.vstack([(matrix[g] ** 2).sum(axis=0) for g in groups])

    def sharpe(selected: list[int]) -> np.ndarray:
        n = counts[selected].sum()
        mean = sums[selected].sum(axis=0) / n
        variance = squares[selected].sum(axis=0) / n - mean**2
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(mean / np.sqrt(np.clip(variance, 0.0, None)))

    logits = []
    universe = set(range(partitions))
    for train in combinations(range(partitions), partitions // 2):
        test = sorted(universe - set(train))
        best = int(np.argmax(sharpe(list(train))))
        oos = sharpe(test)
        rank = scipy_stats.rankdata(oos)[best] / (n_strategies + 1)
        logits.append(np.log(rank / (1.0 - rank)))
    values = np.asarray(logits)
    return {
        "pbo": float(np.mean(values <= 0)),
        "median_logit": float(np.median(values)),
        "n_splits": int(len(values)),
        "n_strategies": int(n_strategies),
    }


def deflated_sharpe(returns: np.ndarray, trial_sharpes: np.ndarray) -> dict[str, float]:
    """Deflated Sharpe ratio, using per-period Sharpe ratios internally.

    ``trial_sharpes`` are the annualized Sharpe ratios of all strategies tried.
    """
    n = len(returns)
    observed = np.mean(returns) / np.std(returns, ddof=1)
    trials = np.asarray(trial_sharpes)[np.isfinite(trial_sharpes)] / np.sqrt(ANNUALIZATION)
    n_trials = len(trials)
    deviation = float(np.std(trials, ddof=1))
    normal = scipy_stats.norm
    benchmark = deviation * (
        (1 - EULER_GAMMA) * normal.ppf(1 - 1 / n_trials) + EULER_GAMMA * normal.ppf(1 - 1 / (n_trials * np.e))
    )
    skew = float(scipy_stats.skew(returns))
    kurt = float(scipy_stats.kurtosis(returns, fisher=False))
    denominator = np.sqrt(1 - skew * observed + (kurt - 1) / 4 * observed**2)
    z_score = (observed - benchmark) * np.sqrt(n - 1) / denominator
    return {
        "dsr": float(normal.cdf(z_score)),
        "expected_max_sharpe": float(benchmark * np.sqrt(ANNUALIZATION)),
        "observed_sharpe": float(observed * np.sqrt(ANNUALIZATION)),
        "n_trials": int(n_trials),
    }
