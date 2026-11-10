"""Unit tests. All data here is synthetic and exists only to test the code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime import config as cfg
from regime.features import intraday, market
from regime.models import signals
from regime.portfolio import backtest as bt
from regime.portfolio import rules
from regime.validation import stats


def minute_bars(symbol: str, day: str, closes: np.ndarray) -> pd.DataFrame:
    start = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
    ts = pd.date_range(start, periods=len(closes), freq="min").tz_convert("UTC")
    return pd.DataFrame(
        {"symbol": symbol, "ts": ts, "open": closes, "high": closes * 1.001, "low": closes * 0.999,
         "close": closes, "volume": 100.0, "vwap": closes, "trades": 5}
    )


SPEC = rules.PortfolioSpec.from_config(cfg.load_experiment()["portfolio"])


def test_session_filter_handles_daylight_saving():
    winter = minute_bars("SPY", "2024-01-16", np.full(390, 100.0))
    summer = minute_bars("SPY", "2024-07-16", np.full(390, 100.0))
    before_open = winter.assign(ts=winter["ts"] - pd.Timedelta(minutes=1)).iloc[:1]
    session = intraday.regular_session(pd.concat([before_open, winter, summer]))
    assert len(session) == 780
    assert session["minute"].min() == 0 and session["minute"].max() == 389
    assert set(session["date"].dt.strftime("%Y-%m-%d")) == {"2024-01-16", "2024-07-16"}


def test_realized_variance_matches_five_minute_returns():
    closes = np.full(390, 100.0)
    closes[10:] = 101.0
    daily = intraday.daily_features(minute_bars("SPY", "2024-01-16", closes)).iloc[0]
    assert daily["rv"] == pytest.approx(np.log(1.01) ** 2)
    assert daily["bv"] == pytest.approx(0.0, abs=1e-12)
    assert daily["jump_frac"] == pytest.approx(1.0)
    assert daily["open"] == 100.0 and daily["close"] == 101.0


def test_short_sessions_are_dropped():
    assert intraday.daily_features(minute_bars("SPY", "2024-01-16", np.full(60, 100.0))).empty


def test_comovement_of_identical_assets_is_one():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.001, 77)
    rows = [pd.DataFrame({"symbol": s, "date": pd.Timestamp("2024-01-16"), "bucket": range(77), "ret": r})
            for s in "ABC"]
    result = intraday.basket_comovement(pd.concat(rows)).iloc[0]
    assert result["avg_corr"] == pytest.approx(1.0)
    assert result["absorption"] == pytest.approx(1.0)


def test_macro_is_one_trading_day_late():
    dates = pd.bdate_range("2024-01-01", periods=5)
    macro = pd.DataFrame({"date": dates, "series": "DGS10", "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    known = market.macro_as_known(macro, dates)["DGS10"]
    assert np.isnan(known.iloc[0])
    assert known.iloc[3] == 3.0


def make_hmm_data(seed: int = 3, n: int = 900):
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < 0.98 else 1 - states[t - 1]
    dates = pd.bdate_range("2018-01-01", periods=n)
    x = rng.normal(0, 1, (n, 3)) + np.array([[3.0, -3.0, 2.0]]) * states[:, None]
    target = pd.Series(np.where(states == 1, 0.01, -0.01), index=dates)
    return pd.DataFrame(x, index=dates, columns=list("abc")), target, states


def test_hmm_recovers_states_and_orders_by_return():
    features, target, states = make_hmm_data()
    result = signals.walk_forward_hmm(features, target, 2, "diag", "Q", 400, [7, 11])
    predicted = result.probabilities.to_numpy().argmax(axis=1)
    index = features.index.get_indexer(result.probabilities.index)
    # The row at t predicts t+1.
    usable = index + 1 < len(states)
    assert np.mean(predicted[usable] == states[index[usable] + 1]) > 0.9
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.log_likelihood.notna().all()


def test_hmm_has_no_look_ahead():
    features, target, _ = make_hmm_data()
    base = signals.walk_forward_hmm(features, target, 2, "diag", "Q", 400, [7])
    changed_features = features.copy()
    changed_target = target.copy()
    cutoff = features.index[700]
    changed_features.loc[changed_features.index > cutoff] += 50.0
    changed_target.loc[changed_target.index > cutoff] *= -1
    changed = signals.walk_forward_hmm(changed_features, changed_target, 2, "diag", "Q", 400, [7])
    pd.testing.assert_frame_equal(base.probabilities.loc[:cutoff], changed.probabilities.loc[:cutoff])


def test_forward_filter_prediction_step():
    log_density = np.log(np.array([[0.9, 0.1]]))
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    filtered, predicted, ll = signals.forward_filter(log_density, transition, np.array([0.5, 0.5]))
    assert filtered[0] == pytest.approx([0.9, 0.1])
    assert predicted[0] == pytest.approx([0.75, 0.25])
    assert ll[0] == pytest.approx(np.log(0.5))


def test_kmeans_shape_and_order():
    features, target, _ = make_hmm_data()
    result = signals.walk_forward_kmeans(features, target, 2, "Q", 400)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert list(result.columns) == ["p0", "p1"]


def test_placebos_preserve_shape():
    index = pd.bdate_range("2020-01-01", periods=300)
    chain = signals.markov_placebo(index, 3, 0.95, 0.9, 1)
    assert np.allclose(chain.sum(axis=1), 1.0)
    assert (chain.to_numpy().argmax(axis=1)[1:] == chain.to_numpy().argmax(axis=1)[:-1]).mean() > 0.9
    shuffled = signals.shuffled_placebo(chain, 2)
    assert shuffled.shape == chain.shape
    assert not shuffled.equals(chain)
    assert np.isclose(np.sort(shuffled.to_numpy(), axis=None), np.sort(chain.to_numpy(), axis=None)).all()


@pytest.mark.parametrize("n_states", [2, 3, 4])
@pytest.mark.parametrize("amplitude", [0.05, 0.3, 0.4])
def test_state_weights_are_valid(n_states, amplitude):
    weights = rules.state_weights(SPEC, n_states, amplitude)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0).all()
    assert weights[0, 0] < weights[-1, 0]


def test_gate_returns_base_below_threshold():
    index = pd.bdate_range("2024-01-01", periods=2)
    signal = pd.DataFrame([[0.5, 0.3, 0.2], [0.95, 0.03, 0.02]], index=index, columns=["p0", "p1", "p2"])
    weights = rules.tilt_with_gate(signal, SPEC, 0.2, 0.6, 0.2)
    assert weights.iloc[0].to_numpy() == pytest.approx(SPEC.base)
    assert weights.iloc[1]["SPY"] < SPEC.base[0]


def test_rule_grid_size():
    assert len(rules.rule_grid(cfg.load_experiment()["portfolio"])) == 2 + 5 + 60


def price_panel(n: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes = pd.DataFrame({"A": 100.0 * 1.01 ** np.arange(n), "B": 100.0}, index=dates)
    opens = closes * 0.5
    return pd.concat({"close": closes, "open": opens, "vwap": closes}, axis=1)


def test_signal_executes_on_next_day():
    prices = price_panel()
    weights = pd.DataFrame({"A": 1.0, "B": 0.0}, index=prices.index[:2])
    result = bt.run_backtest(weights, prices, "next_close")
    assert list(result.gross.index) == list(prices.index[1:3])
    assert result.gross.iloc[0] == pytest.approx(0.01)


def test_costs_include_drift_rebalancing():
    prices = price_panel(5)
    weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=prices.index[:4])
    result = bt.run_backtest(weights, prices, "next_close")
    assert result.traded.iloc[0] == 0.0
    assert result.traded.iloc[1] > 0.0
    assert result.net(10).iloc[1] < result.gross.iloc[1]
    assert result.net(0).equals(result.gross.rename("gross"))


def test_performance_uses_excess_returns():
    index = pd.bdate_range("2024-01-01", periods=252)
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 252), index=index)
    zero = bt.performance(returns, pd.Series(0.0, index=index))
    positive = bt.performance(returns, pd.Series(0.0005, index=index))
    assert positive["sharpe"] < zero["sharpe"]


def test_bootstrap_ci_covers_zero_for_identical_series():
    index = pd.bdate_range("2020-01-01", periods=500)
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 500), index=index)
    result = stats.sharpe_difference_ci(returns, returns, resamples=50)
    assert result["difference"] == 0.0 and result["ci_low"] == 0.0 and result["ci_high"] == 0.0


def test_stationary_indices_are_block_like():
    index = stats.stationary_indices(2000, 21.0, np.random.default_rng(0))
    assert index.min() >= 0 and index.max() < 2000
    continues = np.mean(np.diff(index) == 1)
    assert 0.9 < continues < 0.99


def test_pbo_is_high_for_noise_and_low_for_real_skill():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, (1600, 30))
    assert stats.probability_of_backtest_overfitting(noise, 8)["pbo"] > 0.25
    skilled = noise.copy()
    skilled[:, 0] += 0.01
    assert stats.probability_of_backtest_overfitting(skilled, 8)["pbo"] < 0.05


def test_deflated_sharpe_penalizes_many_trials():
    returns = np.random.default_rng(0).normal(0.0007, 0.01, 1500)
    few = stats.deflated_sharpe(returns, np.array([1.0, 0.5, 0.8]))
    many = stats.deflated_sharpe(returns, np.random.default_rng(1).normal(0.3, 0.5, 2000))
    assert 0 <= many["dsr"] < few["dsr"] <= 1


def test_holdout_is_sealed(monkeypatch):
    split = cfg.load_experiment()["split"]
    index = pd.bdate_range("2022-12-01", "2024-02-01")
    frame = pd.DataFrame({"x": 1.0}, index=index)
    monkeypatch.delenv(cfg.HOLDOUT_FLAG, raising=False)
    with pytest.raises(cfg.HoldoutSealedError):
        cfg.select_period(frame, "holdout", split)
    assert cfg.seal(frame, split).index.max() <= pd.Timestamp(split["val_end"])
    monkeypatch.setenv(cfg.HOLDOUT_FLAG, "1")
    assert not cfg.select_period(frame, "holdout", split).empty


def test_universe_size():
    assert len(cfg.full_universe(cfg.load_universe())) == 100
