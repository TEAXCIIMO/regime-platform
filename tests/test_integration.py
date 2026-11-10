"""End-to-end test of the batch logic. All data here is synthetic test data."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from regime import analysis
from regime import config as cfg
from regime import experiment as exp
from regime import panel
from regime.features.market import ALL_FEATURES, DAILY_FEATURES
from regime.online import OnlineRegime


@pytest.fixture(scope="module")
def universe():
    return cfg.load_universe()


@pytest.fixture(scope="module")
def synthetic(universe):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2016-01-04", "2023-12-29")
    states = np.zeros(len(dates), dtype=int)
    for t in range(1, len(dates)):
        states[t] = states[t - 1] if rng.random() < 0.985 else 1 - states[t - 1]
    market_factor = rng.normal(np.where(states == 0, 0.0008, -0.001), np.where(states == 0, 0.006, 0.02))

    rows = []
    for symbol in cfg.full_universe(universe):
        beta = rng.uniform(0.2, 1.2)
        idiosyncratic = rng.normal(0, 0.004, len(dates))
        returns = beta * market_factor + idiosyncratic
        close = 100 * np.exp(np.cumsum(returns))
        opens = close * np.exp(rng.normal(0, 0.002, len(dates)))
        rows.append(pd.DataFrame({
            "symbol": symbol, "date": dates, "open": opens, "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": 1e6, "dollar_volume": close * 1e6, "vwap": (opens + close) / 2, "n_minutes": 390,
            "rv": (0.5 * returns**2 + 1e-5 * (1 + states * 4)) * rng.uniform(0.8, 1.2, len(dates)),
            "bv": 1e-5, "jump_frac": rng.uniform(0, 0.3, len(dates)), "first30_ret": 0.0,
            "last30_ret": rng.normal(0, 0.001, len(dates)),
        }))
    daily = pd.concat(rows, ignore_index=True)
    macro = pd.concat([
        pd.DataFrame({"date": dates, "series": name, "value": value})
        for name, value in (("DGS2", 2 + np.sin(np.arange(len(dates)) / 200)),
                            ("DGS10", 3 + np.cos(np.arange(len(dates)) / 300)),
                            ("DTB3", np.full(len(dates), 1.5)),
                            ("VIXCLS", 15 + 15 * states + rng.normal(0, 1, len(dates))))
    ], ignore_index=True)
    comovement = pd.DataFrame({
        "date": dates, "avg_corr": 0.3 + 0.3 * states + rng.normal(0, 0.05, len(dates)),
        "absorption": 0.4 + 0.2 * states + rng.normal(0, 0.05, len(dates)), "n_assets": 13,
    })
    return daily, macro, comovement


@pytest.fixture(scope="module")
def small_experiment():
    experiment = copy.deepcopy(cfg.load_experiment())
    experiment["hmm"]["seeds"] = [7]
    experiment["hmm"]["n_iter"] = 20
    experiment["portfolio"].update(tilt_amplitudes=[0.1, 0.2], gate_thresholds=[0.6], gate_scales=[0.2])
    experiment["execution"] = ["next_open", "next_close"]
    experiment["costs_bps"] = [0, 2]
    return experiment


def test_grid_skips_duplicate_daily_universes():
    configs = exp.signal_configs(cfg.load_experiment())
    assert len(configs) == 84
    assert all(c.universe == "minimal" for c in configs if c.feature_set == "daily")


def test_feature_tables_have_expected_shape(synthetic, universe):
    daily, macro, comovement = synthetic
    tables = panel.feature_tables(daily, macro, comovement, universe)
    assert list(tables["full"].columns) == ALL_FEATURES
    assert len(ALL_FEATURES) <= 30
    full = exp.select_features(tables["full"], "all")
    assert full.shape[1] == len(ALL_FEATURES)
    minimal = exp.select_features(tables["minimal"], "daily")
    assert list(minimal.columns) == DAILY_FEATURES


def test_feature_at_t_does_not_use_later_rows(synthetic, universe):
    daily, macro, comovement = synthetic
    cutoff = pd.Timestamp("2020-06-30")
    full = panel.feature_tables(daily, macro, comovement, universe)["core"]
    keep = daily["date"] <= cutoff
    partial = panel.feature_tables(daily[keep], macro[macro["date"] <= cutoff], comovement[comovement["date"] <= cutoff], universe)["core"]
    pd.testing.assert_frame_equal(full.loc[:cutoff], partial, check_freq=False)


def test_full_config_evaluation_and_analysis(synthetic, universe, small_experiment):
    daily, macro, comovement = synthetic
    tables = panel.feature_tables(daily, macro, comovement, universe)
    market = panel.market_data(daily, macro, universe["traded_assets"])
    all_returns, metrics, quality = [], [], []
    for config in [exp.SignalConfig("core", 2, "diag", "Q", "all"), exp.SignalConfig("minimal", 2, "diag", "Q", "daily")]:
        result = exp.evaluate_config(config, tables[config.universe], market, small_experiment)
        assert set(result["signal_table"]) == {"hmm", "placebo_shuffle", "placebo_markov", "kmeans"}
        for frame in result["signal_table"].values():
            assert frame.index.max() <= pd.Timestamp(small_experiment["split"]["val_end"])
        all_returns += result["returns"]
        metrics += result["metrics"]
        quality += result["quality"]

    metrics_frame = pd.DataFrame(metrics)
    assert set(metrics_frame["period"]) == {"dev", "val"}
    net = metrics_frame[(metrics_frame["signal"] == "hmm") & (metrics_frame["rule"] == "tilt_gate")]
    assert net[net["cost_bps"] == 2.0]["annual_return"].mean() < net[net["cost_bps"] == 0.0]["annual_return"].mean()

    # This synthetic market has real regimes, so stress probability should track volatility.
    quality_frame = pd.DataFrame(quality)
    rq1 = analysis.rq1_regime_quality(quality_frame)
    assert rq1["rank_corr_forward_rv"].mean() < 0
    assert not analysis.factorial_table(metrics_frame).empty
    assert not analysis.factor_effects(metrics_frame).empty

    returns = analysis.returns_matrix(all_returns)
    index = pd.DatetimeIndex(returns.index)
    bench = exp.benchmark_table(market, small_experiment, index).query("execution == 'next_open' and cost_bps == 2.0")
    compared = analysis.compare_selected(returns, bench.set_index("date")["net"], 2019, resamples=100)
    assert "vs_benchmark" in compared["comparisons"] and "vs_placebo_shuffle" in compared["comparisons"]
    report = analysis.overfitting_report(analysis.strategies_for(returns, "hmm"), compared["selected"], partitions=8)
    assert 0 <= report["pbo"] <= 1 and 0 <= report["dsr"] <= 1


def test_online_update_matches_batch_interface(synthetic, universe, small_experiment):
    daily, macro, comovement = synthetic
    table = panel.feature_tables(daily, macro, comovement, universe)["core"]
    features = exp.select_features(table, "daily")
    market = panel.market_data(daily, macro, universe["traded_assets"])
    before = pd.Timestamp("2023-10-30")
    symbols = panel.universe_symbols(universe, "core")
    groups = {g: [s for s in m if s in symbols] for g, m in universe["groups"].items()}
    model = OnlineRegime.fit(features, market.spy_return, 2, "diag", [7], daily[daily["symbol"].isin(symbols)],
                             comovement, macro, groups, universe["correlation_basket"], before)

    rng = np.random.default_rng(5)
    rows = []
    start = pd.Timestamp("2023-10-30 09:30", tz="America/New_York")
    ts = pd.date_range(start, periods=120, freq="min").tz_convert("UTC")
    previous = daily[daily["date"] < before].sort_values("date").groupby("symbol")["close"].last()
    for symbol in symbols:
        closes = float(previous[symbol]) * np.exp(np.cumsum(rng.normal(0, 0.0004, len(ts))))
        rows.append(pd.DataFrame({"symbol": symbol, "ts": ts, "open": closes, "high": closes, "low": closes,
                                  "close": closes, "volume": 100.0, "vwap": closes}))
    output = model.update(pd.concat(rows, ignore_index=True))
    assert output is not None
    assert sum(output.values()) == pytest.approx(1.0)
