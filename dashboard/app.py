"""Interactive results dashboard.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"

st.set_page_config(page_title="Cross-Asset Regime Platform", layout="wide")
st.title("Cross-Asset Regime Platform")


def available_runs() -> list[str]:
    # A sweep run is a folder that analyze.py finished; streaming and system tables live elsewhere.
    return sorted(p.name for p in TABLES.iterdir() if (p / "summary.json").exists()) if TABLES.exists() else []


runs = available_runs()
if not runs:
    st.info("No results yet. Run the sweep, then scripts/analyze.py.")
    st.stop()

run = st.sidebar.selectbox("Run", runs, index=runs.index("full") if "full" in runs else len(runs) - 1)
folder = TABLES / run


def table(name: str) -> pd.DataFrame | None:
    path = folder / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_signals(run_name: str) -> list[dict]:
    rows = []
    for path in sorted((RUNS / run_name / "raw" / "kind=signal").glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            rows += [json.loads(json.loads(line)["payload"]) for line in handle if line.strip()]
    return rows


regimes, factorial, models, costs, system = st.tabs(["Regimes", "Signal × rule", "Model factors", "Costs", "System"])

with regimes:
    rows = load_signals(run)
    if rows:
        configs = sorted({r["config"] for r in rows})
        names = sorted({r["signal"] for r in rows})
        left, right = st.columns(2)
        chosen = left.selectbox("Configuration", configs)
        signal = right.selectbox("Signal", names, index=names.index("hmm") if "hmm" in names else 0)
        match = next(r for r in rows if r["config"] == chosen and r["signal"] == signal)
        frame = pd.DataFrame(match["values"], columns=match["columns"], index=pd.to_datetime(match["dates"]))
        st.plotly_chart(
            px.area(frame, labels={"index": "Signal date", "value": "Probability", "variable": "State"},
                    title="Next-day state probabilities (p0 is the lowest-return state)"),
            use_container_width=True,
        )
    quality = table("rq1_quality")
    if quality is not None:
        st.subheader("Regime quality by feature set")
        st.dataframe(quality, use_container_width=True)

with factorial:
    period = st.radio("Period", ["dev", "val"], horizontal=True)
    grid = table(f"rq2_factorial_{period}")
    if grid is not None:
        matrix = grid.set_index("signal")
        st.plotly_chart(
            px.imshow(matrix, text_auto=".2f", color_continuous_scale="RdBu", origin="upper",
                      title="Mean net Sharpe, next-open execution, 2 bps"),
            use_container_width=True,
        )
    summary_path = folder / "summary.json"
    if summary_path.exists():
        st.subheader("Nested walk-forward selection and overfitting checks")
        st.json(json.loads(summary_path.read_text(encoding="utf-8")))
    holdout_path = folder / "holdout.json"
    if holdout_path.exists():
        st.subheader("Sealed holdout")
        st.json(json.loads(holdout_path.read_text(encoding="utf-8")))

with models:
    factors = table("rq3_factors_dev")
    if factors is not None:
        st.plotly_chart(
            px.bar(factors, x="level", y="sharpe", facet_col="factor", facet_col_wrap=3,
                   title="Mean net Sharpe of HMM strategies by factor level"),
            use_container_width=True,
        )
        st.dataframe(factors, use_container_width=True)

with costs:
    sensitivity = table("sensitivity_dev")
    if sensitivity is not None:
        st.plotly_chart(
            px.line(sensitivity, x="cost_bps", y="sharpe", color="execution", markers=True,
                    labels={"cost_bps": "Cost per dollar traded (bps)", "sharpe": "Mean net Sharpe"}),
            use_container_width=True,
        )

with system:
    scaling_path = TABLES / "system" / "scaling.csv"
    if scaling_path.exists():
        scaling = pd.read_csv(scaling_path)
        st.subheader("Runtime by number of workers")
        st.plotly_chart(
            px.line(scaling, x="workers", y="seconds", color="job", markers=True,
                    labels={"workers": "n2-standard-2 workers", "seconds": "Runtime (s)"}),
            use_container_width=True,
        )
        st.dataframe(scaling, use_container_width=True)
    records = []
    for path in sorted((RUNS / "system" / "job_stats").glob("**/*.json")) if (RUNS / "system").exists() else []:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    records.append(json.loads(row["payload"]) if "payload" in row else row)
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True)
    streams = list((RUNS / "streaming").glob("*")) if (RUNS / "streaming").exists() else []
    if streams:
        day = st.selectbox("Replay day", sorted(p.name for p in streams))
        batches = pd.concat(pd.read_parquet(p) for p in sorted((RUNS / "streaming" / day).glob("*.parquet")))
        probability_columns = [c for c in batches.columns if c.startswith("p") and c[1:].isdigit()]
        batches["event_time"] = pd.to_datetime(batches["event_time"])
        st.plotly_chart(px.line(batches, x="event_time", y=probability_columns, title="Online regime probabilities"),
                        use_container_width=True)
        st.metric("Median end-to-end latency (s)", round(float(batches["end_to_end_seconds"].median()), 2))
