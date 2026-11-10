# Cross-Asset Regime Platform

Intraday big data for cross-asset regime detection and portfolio allocation.
The evaluation rules are fixed in [docs/protocol.md](docs/protocol.md).

Parts of the HMM filtering and backtest logic are adapted from the team's earlier
open-source regime research code. All data, features, experiments and results
here are new.

Results and discussion are in [docs/final_report.md](docs/final_report.md).

## Layout

| Path | Purpose |
|---|---|
| `src/regime/` | Features, models, portfolio rules, backtest, validation |
| `src/regime/resources/` | Asset universe and frozen experiment grid |
| `spark_jobs/` | Dataproc jobs: minute bars to daily rows, sweep, streaming |
| `scripts/` | Download, GCP operations, analysis, holdout, stream replay |
| `dashboard/` | Streamlit results dashboard |
| `tests/` | Unit and integration tests, on synthetic data only |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
Copy-Item .env.example .env   # then fill in your Alpaca keys
```

## Run

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py macro
.\.venv\Scripts\python.exe scripts\download_data.py check
.\.venv\Scripts\python.exe scripts\download_data.py bars

.\scripts\gcp.ps1 setup
.\scripts\gcp.ps1 upload
.\scripts\gcp.ps1 cluster -Workers 7
.\scripts\gcp.ps1 job01 -Sample 0.05
.\scripts\gcp.ps1 job01
.\scripts\gcp.ps1 job02 -Limit 2 -RunName test
.\scripts\gcp.ps1 job02
.\scripts\gcp.ps1 job03 -ReplayDate 2023-10-30   # run together with the next line
.\scripts\gcp.ps1 replay -ReplayDate 2023-10-30
.\scripts\gcp.ps1 bigquery
.\scripts\gcp.ps1 sync
.\.venv\Scripts\python.exe scripts\analyze.py --run full
.\.venv\Scripts\python.exe scripts\summarize_streaming.py --date 2023-10-30
.\.venv\Scripts\python.exe scripts\system_table.py
streamlit run dashboard\app.py
```

Clusters delete themselves after 20 idle minutes or 8 hours. Run
`.\scripts\gcp.ps1 delete` when finished.

The sealed holdout is opened once, as the last research step:

```powershell
.\.venv\Scripts\python.exe scripts\open_holdout.py --run full --confirm
```
