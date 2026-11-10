"""Walk-forward regime signals and placebo controls.

Every signal is a frame indexed by date ``t`` with columns ``p0 .. p{K-1}``.
A row is the probability of each state on ``t+1``, using data through ``t``.
State 0 is the lowest-return state in the training window.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CLIP = 10.0
EPSILON = 1e-300


@dataclass(frozen=True)
class WalkForwardResult:
    probabilities: pd.DataFrame
    log_likelihood: pd.Series
    n_fits: int


def refit_positions(dates: pd.DatetimeIndex, min_train: int, refit: str) -> list[int]:
    """Row positions that start a refit block: first trading day of each period."""
    if refit == "M":
        keys = dates.year * 12 + dates.month
    elif refit == "Q":
        keys = dates.year * 4 + dates.quarter
    else:
        raise ValueError(f"Unknown refit frequency: {refit}")
    keys = np.asarray(keys)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return [int(i) for i in starts if i >= min_train]


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train)
    return (
        np.clip(scaler.transform(train), -CLIP, CLIP),
        np.clip(scaler.transform(test), -CLIP, CLIP),
    )


def gaussian_log_density(x: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Log density of each row of ``x`` under each state's full covariance."""
    n_states = means.shape[0]
    n_features = x.shape[1]
    log_density = np.zeros((x.shape[0], n_states))
    for k in range(n_states):
        factor = np.linalg.cholesky(covariances[k])
        solved = np.linalg.solve(factor, (x - means[k]).T).T
        log_det = 2.0 * np.sum(np.log(np.diag(factor)))
        log_density[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + np.sum(solved**2, axis=1))
    return log_density


def forward_filter(
    log_density: np.ndarray, transition: np.ndarray, prior: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the forward recursion.

    Returns filtered posteriors P(z_t | x_1:t), one-step-ahead predictions
    P(z_t+1 | x_1:t), and the predictive log-likelihood of each observation.
    """
    n_rows, n_states = log_density.shape
    filtered = np.zeros((n_rows, n_states))
    predicted = np.zeros((n_rows, n_states))
    log_likelihood = np.zeros(n_rows)
    current = prior
    for t in range(n_rows):
        joint = np.log(np.clip(current, EPSILON, None)) + log_density[t]
        log_likelihood[t] = logsumexp(joint)
        posterior = np.exp(joint - log_likelihood[t])
        posterior /= posterior.sum()
        filtered[t] = posterior
        current = posterior @ transition
        current /= current.sum()
        predicted[t] = current
    return filtered, predicted, log_likelihood


def state_order(weights: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Raw state indices sorted from lowest to highest weighted target mean."""
    totals = np.clip(weights.sum(axis=0), EPSILON, None)
    return np.argsort((weights * target[:, None]).sum(axis=0) / totals, kind="stable")


def fit_hmm(train: np.ndarray, n_states: int, covariance: str, seeds: list[int], n_iter: int, tol: float) -> GaussianHMM:
    best: GaussianHMM | None = None
    best_score = -np.inf
    for seed in seeds:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance,
            n_iter=n_iter,
            tol=tol,
            min_covar=1e-3,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(train)
                score = float(model.score(train))
            except (ValueError, np.linalg.LinAlgError):
                continue
        if np.isfinite(score) and score > best_score:
            best, best_score = model, score
    if best is None:
        raise RuntimeError("All HMM restarts failed.")
    return best


def _prepare(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    clean = features.dropna()
    return clean, target.reindex(clean.index).fillna(0.0)


def walk_forward_hmm(
    features: pd.DataFrame,
    target: pd.Series,
    n_states: int,
    covariance: str,
    refit: str,
    min_train: int,
    seeds: list[int],
    n_iter: int = 100,
    tol: float = 1e-4,
) -> WalkForwardResult:
    """Fit on an expanding window and filter forward through each refit block."""
    data, y = _prepare(features, target)
    dates = pd.DatetimeIndex(data.index)
    values = data.to_numpy(dtype=float)
    returns = y.to_numpy(dtype=float)
    starts = refit_positions(dates, min_train, refit)
    if not starts:
        raise ValueError("Not enough history for one refit block.")

    probabilities = np.full((len(dates), n_states), np.nan)
    log_likelihood = np.full(len(dates), np.nan)
    for block, start in enumerate(starts):
        end = starts[block + 1] if block + 1 < len(starts) else len(dates)
        train, test = _standardize(values[:start], values[start:end])
        model = fit_hmm(train, n_states, covariance, seeds, n_iter, tol)
        covariances = np.asarray(model.covars_)
        means = np.asarray(model.means_)
        transition = np.asarray(model.transmat_)

        train_density = gaussian_log_density(train, means, covariances)
        filtered, predicted, _ = forward_filter(train_density, transition, np.asarray(model.startprob_))
        order = state_order(filtered, returns[:start])

        test_density = gaussian_log_density(test, means, covariances)
        _, block_predicted, block_ll = forward_filter(test_density, transition, predicted[-1])
        probabilities[start:end] = block_predicted[:, order]
        log_likelihood[start:end] = block_ll

    columns = [f"p{k}" for k in range(n_states)]
    frame = pd.DataFrame(probabilities, index=dates, columns=columns).dropna()
    likelihood = pd.Series(log_likelihood, index=dates, name="log_likelihood").dropna()
    return WalkForwardResult(frame, likelihood, len(starts) * len(seeds))


def walk_forward_kmeans(
    features: pd.DataFrame,
    target: pd.Series,
    n_states: int,
    refit: str,
    min_train: int,
    seed: int = 7,
    variance_kept: float = 0.9,
) -> pd.DataFrame:
    """PCA + K-means on an expanding window, softmax of distances as probabilities."""
    data, y = _prepare(features, target)
    dates = pd.DatetimeIndex(data.index)
    values = data.to_numpy(dtype=float)
    returns = y.to_numpy(dtype=float)
    starts = refit_positions(dates, min_train, refit)
    probabilities = np.full((len(dates), n_states), np.nan)

    for block, start in enumerate(starts):
        end = starts[block + 1] if block + 1 < len(starts) else len(dates)
        train, test = _standardize(values[:start], values[start:end])
        pca = PCA(n_components=variance_kept, random_state=seed).fit(train)
        train_scores = pca.transform(train)
        kmeans = KMeans(n_clusters=n_states, n_init=10, random_state=seed).fit(train_scores)
        one_hot = np.eye(n_states)[kmeans.labels_]
        order = state_order(one_hot, returns[:start])
        distances = kmeans.transform(pca.transform(test))
        logits = -(distances**2)
        logits -= logits.max(axis=1, keepdims=True)
        soft = np.exp(logits)
        soft /= soft.sum(axis=1, keepdims=True)
        probabilities[start:end] = soft[:, order]

    return pd.DataFrame(probabilities, index=dates, columns=[f"p{k}" for k in range(n_states)]).dropna()


def shuffled_placebo(signal: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute whole calendar months, keeping within-month persistence."""
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(signal.index)
    months = np.asarray(dates.year * 12 + dates.month)
    unique = np.unique(months)
    blocks = [signal.to_numpy()[months == m] for m in rng.permutation(unique)]
    shuffled = np.concatenate(blocks, axis=0)
    return pd.DataFrame(shuffled, index=dates, columns=signal.columns)


def markov_placebo(index: pd.DatetimeIndex, n_states: int, persistence: float, confidence: float, seed: int) -> pd.DataFrame:
    """Random Markov chain with no relationship to the data."""
    rng = np.random.default_rng(seed)
    states = np.zeros(len(index), dtype=int)
    states[0] = rng.integers(n_states)
    for t in range(1, len(index)):
        if rng.random() < persistence:
            states[t] = states[t - 1]
        else:
            states[t] = rng.choice([k for k in range(n_states) if k != states[t - 1]])
    other = (1.0 - confidence) / (n_states - 1)
    probabilities = np.full((len(index), n_states), other)
    probabilities[np.arange(len(index)), states] = confidence
    return pd.DataFrame(probabilities, index=index, columns=[f"p{k}" for k in range(n_states)])
