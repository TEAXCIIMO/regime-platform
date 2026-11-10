"""Experiment grid. One function evaluates one signal configuration end to end."""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from regime import config as cfg
from regime.features.market import FEATURE_SETS
from regime.models import signals
from regime.portfolio import backtest as bt
from regime.portfolio import rules as rule_lib

STORED_EXECUTION = "next_open"
STORED_COST_BPS = 2.0
PERIODS = ("dev", "val")
MIN_COVERAGE = 0.95
FORWARD_DAYS = 5


@dataclass(frozen=True)
class SignalConfig:
    universe: str
    n_states: int
    covariance: str
    refit: str
    feature_set: str

    @property
    def key(self) -> str:
        return f"{self.universe}-{self.n_states}s-{self.covariance}-{self.refit}-{self.feature_set}"


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    risk_free: pd.Series
    spy_return: pd.Series
    spy_rv: pd.Series


def signal_configs(experiment: dict) -> list[SignalConfig]:
    """Grid of model configurations.

    The daily feature set does not depend on the universe, so it runs once.
    """
    grid = experiment["signal_grid"]
    configs = []
    for universe, n_states, covariance, refit, feature_set in itertools.product(
        grid["universe"], grid["n_states"], grid["covariance"], grid["refit"], grid["feature_set"]
    ):
        if feature_set == "daily" and universe != "minimal":
            continue
        configs.append(SignalConfig(universe, n_states, covariance, refit, feature_set))
    return configs


def select_features(table: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Columns of a feature set that have usable data in this universe."""
    columns = [c for c in FEATURE_SETS[feature_set] if c in table.columns]
    subset = table[columns]
    usable = [c for c in columns if subset[c].notna().mean() >= MIN_COVERAGE and subset[c].std() > 0]
    return subset[usable]


def rule_key(rule: dict) -> str:
    return "|".join(str(rule[k]) for k in ("rule", "amplitude", "threshold", "scale") if k in rule)


def regime_quality(signal: pd.DataFrame, market: MarketData) -> dict[str, float]:
    """Compare a signal with what happens over the following five trading days."""
    scores = rule_lib.state_scores(signal.shape[1])
    score = pd.Series(signal.to_numpy() @ scores, index=signal.index)
    stress = signal["p0"]

    rv = market.spy_rv.sort_index()
    forward_rv = np.log(rv.rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS).where(lambda s: s > 0))
    forward_return = market.spy_return.rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS)
    # The event threshold uses only volatility known at signal time.
    threshold = rv.rolling(FORWARD_DAYS).sum().expanding(min_periods=252).quantile(2 / 3)
    event = (rv.rolling(FORWARD_DAYS).sum().shift(-FORWARD_DAYS) > threshold).astype(float)

    frame = pd.DataFrame(
        {"score": score, "stress": stress, "forward_rv": forward_rv, "forward_return": forward_return,
         "event": event, "threshold": threshold}
    ).dropna()
    if len(frame) < 30:
        return {"rank_corr_forward_rv": np.nan, "brier_high_vol": np.nan, "tercile_return_gap": np.nan,
                "avg_confidence": float(signal.max(axis=1).mean())}
    low, high = frame["score"].quantile([1 / 3, 2 / 3])
    return {
        "rank_corr_forward_rv": float(frame["score"].corr(frame["forward_rv"], method="spearman")),
        "brier_high_vol": float(((frame["stress"] - frame["event"]) ** 2).mean()),
        "tercile_return_gap": float(
            frame.loc[frame["score"] >= high, "forward_return"].mean()
            - frame.loc[frame["score"] <= low, "forward_return"].mean()
        ),
        "avg_confidence": float(signal.max(axis=1).mean()),
    }


def build_signals(config: SignalConfig, features: pd.DataFrame, market: MarketData, experiment: dict) -> tuple[dict, pd.Series, int]:
    hmm_settings = experiment["hmm"]
    placebo = experiment["placebo"]
    min_train = experiment["walk_forward"]["min_train_days"]
    hmm = signals.walk_forward_hmm(
        features, market.spy_return, config.n_states, config.covariance, config.refit, min_train,
        hmm_settings["seeds"], hmm_settings["n_iter"], hmm_settings["tol"],
    )
    seed = placebo["seed"] + abs(hash_config(config)) % 100_000
    built = {
        "hmm": hmm.probabilities,
        "placebo_shuffle": signals.shuffled_placebo(hmm.probabilities, seed),
        "placebo_markov": signals.markov_placebo(
            pd.DatetimeIndex(hmm.probabilities.index), config.n_states,
            placebo["markov_persistence"], placebo["chain_confidence"], seed,
        ),
    }
    # K-means has no covariance setting, so it runs for one covariance value only.
    if config.covariance == "diag":
        built["kmeans"] = signals.walk_forward_kmeans(
            features, market.spy_return, config.n_states, config.refit, min_train
        )
    return built, hmm.log_likelihood, hmm.n_fits


def hash_config(config: SignalConfig) -> int:
    """A stable integer hash, unlike Python's per-process ``hash``."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(config.key))


def evaluate_config(config: SignalConfig, table: pd.DataFrame, market: MarketData, experiment: dict) -> dict:
    """Fit all signals for one configuration and backtest every rule.

    Returns metric rows, regime-quality rows, stored daily return rows, the
    probabilities, and the number of HMM fits.
    """
    split = experiment["split"]
    spec = rule_lib.PortfolioSpec.from_config(experiment["portfolio"])
    grid = rule_lib.rule_grid(experiment["portfolio"])
    features = select_features(table, config.feature_set)
    built, log_likelihood, n_fits = build_signals(config, features, market, experiment)
    fields = asdict(config)
    fields["config"] = config.key
    fields["n_features"] = features.shape[1]

    benchmarks = {}
    index = pd.DatetimeIndex(built["hmm"].index)
    for execution in experiment["execution"]:
        weights = bt.constant_weights(index, dict(zip(spec.assets, spec.base)))
        benchmarks[execution] = bt.run_backtest(weights, market.prices, execution)

    metric_rows: list[dict] = []
    quality_rows: list[dict] = []
    return_rows: list[dict] = []

    for signal_name, signal in built.items():
        sealed_signal = cfg.seal(signal, split)
        for period in PERIODS:
            part = cfg.select_period(sealed_signal, period, split)
            quality_rows.append({**fields, "signal": signal_name, "period": period, **regime_quality(part, market)})
        for rule in grid:
            weights = rule_lib.apply_rule(sealed_signal, spec, rule)
            for execution in experiment["execution"]:
                result = bt.run_backtest(weights, market.prices, execution)
                base = benchmarks[execution]
                for cost in experiment["costs_bps"]:
                    net = result.net(cost)
                    benchmark_net = base.net(cost)
                    for period in PERIODS:
                        series = cfg.select_period(net.to_frame("r"), period, split)["r"]
                        if len(series) < 20:
                            continue
                        metric_rows.append({
                            **fields, "signal": signal_name, "rule": rule["rule"], "rule_key": rule_key(rule),
                            "execution": execution, "cost_bps": float(cost), "period": period,
                            **bt.performance(series, market.risk_free),
                            **bt.information_ratio(series, benchmark_net),
                            "annual_turnover": float(
                                result.traded.reindex(series.index).mean() * bt.ANNUALIZATION / 2.0
                            ),
                        })
                if execution == STORED_EXECUTION:
                    stored = cfg.select_period(result.net(STORED_COST_BPS).to_frame("r"), "dev_val", split)["r"]
                    return_rows.append({
                        "strategy": f"{config.key}#{signal_name}#{rule_key(rule)}",
                        "config": config.key, "signal": signal_name, "rule_key": rule_key(rule),
                        "start": stored.index[0].strftime("%Y-%m-%d"),
                        "dates": [d.strftime("%Y-%m-%d") for d in stored.index],
                        "returns": [float(x) for x in stored.to_numpy()],
                    })

    likelihood = cfg.seal(log_likelihood.to_frame("ll"), split)["ll"]
    for period in PERIODS:
        quality_rows.append({
            **fields, "signal": "hmm_log_likelihood", "period": period,
            "avg_log_likelihood_per_feature": float(
                cfg.select_period(likelihood.to_frame("ll"), period, split)["ll"].mean() / features.shape[1]
            ),
        })

    return {
        "metrics": metric_rows,
        "quality": quality_rows,
        "returns": return_rows,
        "signal_table": {name: cfg.seal(frame, split) for name, frame in built.items()},
        "n_fits": n_fits,
        "feature_columns": list(features.columns),
    }


def benchmark_table(market: MarketData, experiment: dict, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Benchmark net returns for every execution and cost, sealed at validation end."""
    spec = rule_lib.PortfolioSpec.from_config(experiment["portfolio"])
    weights = bt.constant_weights(index, dict(zip(spec.assets, spec.base)))
    rows = []
    for execution in experiment["execution"]:
        result = bt.run_backtest(weights, market.prices, execution)
        for cost in experiment["costs_bps"]:
            series = cfg.seal(result.net(cost).to_frame("r"), experiment["split"])["r"]
            rows.append(pd.DataFrame({"date": series.index, "execution": execution, "cost_bps": float(cost), "net": series.values}))
    return pd.concat(rows, ignore_index=True)


def to_json(value: object) -> str:
    return json.dumps(value, sort_keys=True)
