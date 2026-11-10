"""Build analysis tables from a synced sweep run into results/tables/<run>.

Usage:
    python scripts/analyze.py --run full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from regime import analysis  # noqa: E402
from regime import config as cfg  # noqa: E402
from regime import experiment as exp  # noqa: E402
from regime import panel  # noqa: E402


def read_kind(run_dir: Path, kind: str) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "raw" / f"kind={kind}").glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            rows += [json.loads(json.loads(line)["payload"]) for line in handle if line.strip()]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="full")
    args = parser.parse_args()

    run_dir = ROOT / "results" / "runs" / args.run
    out = ROOT / "results" / "tables" / args.run
    out.mkdir(parents=True, exist_ok=True)

    errors = read_kind(run_dir, "error")
    if errors:
        print(f"WARNING: {len(errors)} configurations failed, listed in errors.json")
        (out / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")

    metrics = pd.DataFrame(read_kind(run_dir, "metric"))
    quality = pd.DataFrame(read_kind(run_dir, "quality"))
    returns = analysis.returns_matrix(read_kind(run_dir, "returns"))

    tables = {
        "rq1_quality": analysis.rq1_regime_quality(quality),
        "rq2_factorial_dev": analysis.factorial_table(metrics, "dev").reset_index(),
        "rq2_factorial_val": analysis.factorial_table(metrics, "val").reset_index(),
        "rq3_factors_dev": analysis.factor_effects(metrics, "dev"),
        "sensitivity_dev": analysis.sensitivity(metrics, "dev"),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name} ==\n{table.to_string(index=False)}")

    experiment = cfg.load_experiment()
    daily = pd.read_parquet(ROOT / "data" / "curated" / "daily")
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None).dt.normalize()
    macro = pd.read_parquet(ROOT / "data" / "raw" / "macro.parquet")
    market = panel.market_data(daily, macro, cfg.load_universe()["traded_assets"])
    benchmark = exp.benchmark_table(market, experiment, pd.DatetimeIndex(returns.index))
    series = benchmark.query("execution == 'next_open' and cost_bps == 2.0").set_index("date")["net"]

    first_year = pd.Timestamp(experiment["split"]["dev_start"]).year + 1
    compared = analysis.compare_selected(returns, series, first_year)
    hmm_returns = analysis.strategies_for(returns, "hmm")
    summary = {
        "comparisons": compared["comparisons"],
        "overfitting": analysis.overfitting_report(hmm_returns, compared["selected"]),
        "n_strategies": int(returns.shape[1]),
        "n_metric_rows": int(len(metrics)),
    }
    compared["selection_log"].to_csv(out / "selection_log.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
