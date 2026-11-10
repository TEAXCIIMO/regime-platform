"""Evaluate the frozen design on the sealed holdout. Run this only once.

The design is the configuration and rule chosen by the last nested
walk-forward selection on data before the holdout.

Usage:
    python scripts/open_holdout.py --run full --confirm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from regime import config as cfg  # noqa: E402
from regime import experiment as exp  # noqa: E402
from regime import panel  # noqa: E402
from regime.models import signals  # noqa: E402
from regime.portfolio import backtest as bt  # noqa: E402
from regime.portfolio import rules as rule_lib  # noqa: E402
from regime.validation import stats  # noqa: E402


def parse_rule(key: str) -> dict:
    parts = key.split("|")
    rule = {"rule": parts[0]}
    for name, value in zip(("amplitude", "threshold", "scale"), parts[1:]):
        rule[name] = float(value)
    return rule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="full")
    parser.add_argument("--confirm", action="store_true", help="Required: this opens the holdout")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to open the holdout without --confirm.")

    out = ROOT / "results" / "tables" / args.run
    target = out / "holdout.json"
    if target.exists():
        raise SystemExit(f"The holdout was already opened: {target}")
    os.environ[cfg.HOLDOUT_FLAG] = "1"

    selection = pd.read_csv(out / "selection_log.csv").iloc[-1]
    config_key, _, rule_text = selection["strategy"].split("#")
    universe_name, states, covariance, refit, feature_set = config_key.split("-")
    config = exp.SignalConfig(universe_name, int(states[:-1]), covariance, refit, feature_set)

    experiment = cfg.load_experiment()
    universe = cfg.load_universe()
    daily = pd.read_parquet(ROOT / "data" / "curated" / "daily")
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None).dt.normalize()
    comovement = pd.read_parquet(ROOT / "data" / "curated" / "comovement")
    comovement["date"] = pd.to_datetime(comovement["date"]).dt.tz_localize(None).dt.normalize()
    macro = pd.read_parquet(ROOT / "data" / "raw" / "macro.parquet")
    table = panel.feature_tables(daily, macro, comovement, universe)[config.universe]
    market = panel.market_data(daily, macro, universe["traded_assets"])
    features = exp.select_features(table, config.feature_set)

    spec = rule_lib.PortfolioSpec.from_config(experiment["portfolio"])
    rule = parse_rule(rule_text)
    built, _, _ = exp.build_signals(config, features, market, experiment)
    index = pd.DatetimeIndex(built["hmm"].index)

    baseline = bt.run_backtest(bt.constant_weights(index, dict(zip(spec.assets, spec.base))), market.prices, "next_open")
    benchmark = cfg.select_period(baseline.net(2.0).to_frame("r"), "holdout", experiment["split"])["r"]

    series = {}
    for name, signal in built.items():
        result = bt.run_backtest(rule_lib.apply_rule(signal, spec, rule), market.prices, "next_open")
        series[name] = cfg.select_period(result.net(2.0).to_frame("r"), "holdout", experiment["split"])["r"]

    report = {
        "design": {"config": config_key, "rule": rule},
        "period": [str(benchmark.index.min().date()), str(benchmark.index.max().date())],
        "hmm": bt.performance(series["hmm"], market.risk_free),
        "benchmark_60_40": bt.performance(benchmark, market.risk_free),
        "information_ratio": bt.information_ratio(series["hmm"], benchmark),
        "vs_benchmark": stats.sharpe_difference_ci(series["hmm"], benchmark),
    }
    for name in ("placebo_shuffle", "placebo_markov", "kmeans"):
        if name in series:
            report[f"vs_{name}"] = stats.sharpe_difference_ci(series["hmm"], series[name])
    placebo_diffs = {k: v for k, v in report.items() if k.startswith("vs_placebo")}
    report["rq2_supported"] = bool(
        report["vs_benchmark"]["ci_low"] > 0 and placebo_diffs and min(v["ci_low"] for v in placebo_diffs.values()) > 0
    )
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
