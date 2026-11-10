# Intraday Big Data for Cross-Asset Regime Detection and Portfolio Allocation

EECS E6893 Big Data Analytics, final project report.
Team: [Team member names]

## Abstract

We built a cloud pipeline that turns 103 million one-minute bars for 100 ETFs
(January 2016 to October 2026) into daily cross-asset features. A walk-forward
hidden Markov model (HMM) reads these features as regime probabilities, and
these drive a 60/40-style allocation. Spark on Dataproc ran 84 model
configurations, 19,740 HMM fits and about 59,000 net-of-cost backtests in
20 minutes on 7 workers. A Pub/Sub and Structured Streaming path updates the
regime during a replayed trading day with a median end-to-end latency of 5.2
seconds. Intraday features separated future volatility better than daily
features (RQ1, partly supported). On the sealed 2024–2026 holdout the selected
design had a Sharpe ratio of 0.88 against 0.72 for 60/40. The 95% intervals
for its Sharpe advantage over 60/40 and over both placebo signals all include
zero, so RQ2 is not supported. The number of states and the refit frequency
mattered more than asset breadth or data frequency (RQ3).

## 1. Research questions

- **RQ1.** Do intraday-derived features detect market regimes better than daily
  features alone?
- **RQ2.** Do regime signals add portfolio value over a static 60/40 benchmark
  and placebo signals, after costs?
- **RQ3.** Which matters most: asset breadth, data frequency, or model
  complexity?

The evaluation rules were written in [protocol.md](protocol.md) before any
market data was loaded. The sealed holdout was opened once, as the last
research step, by [open_holdout.py](../scripts/open_holdout.py).

## 2. Data

| Source | Content | Size |
|---|---|---|
| Alpaca Market Data API, SIP feed | 1-minute OHLCV bars, split and dividend adjusted, 100 ETFs | 103,385,195 rows, 1,100 symbol-year Parquet files, 1.9 GB |
| FRED (keyless CSV) | DGS2, DGS10, DTB3, VIXCLS | daily |

The universe covers US broad equity, sectors, international equity, rates,
credit, commodities and currencies. Three nested universes were tested:
`minimal` (8 ETFs), `core` (16) and `full` (100).

We used the consolidated SIP feed rather than the single-exchange IEX feed.
On 2023-06-14 SIP had 370 bars for INDA where IEX had 86, and 88.5 million
SPY shares where IEX had 1.15 million.

**Data quality.** All 1,100 files share one schema with no duplicate
timestamps. A day is kept for a symbol only if it has at least 180 regular
session minutes. Five currency ETFs (FXA, FXB, FXC, FXE, FXY) trade too
thinly to pass this test on most days (FXA passes on 1 day). The currency
volatility feature therefore covers only 82–85% of days and is dropped by the
95% coverage rule. No other feature is affected.

## 3. System

```
Alpaca SIP ──download──> GCS raw/bars (Parquet, symbol=/year=)
                              │
                 Dataproc job 01 (Spark, applyInPandas)
                              │  clean, regular session, 5-min RV / BV / jumps,
                              │  basket co-movement per day
                              v
                  GCS curated/daily, curated/comovement
                              │
                 Dataproc job 02 (Spark RDD over 84 configurations)
                              │  broadcast sealed feature tables, walk-forward
                              │  HMM + placebos + K-means, rule grid, backtests
                              v
            GCS results/<run>/raw (JSON lines) ──> BigQuery dataset regime
                              │
          local analysis (bootstrap, CSCV PBO, Deflated Sharpe) ──> Streamlit
Pub/Sub regime-bars ──> Dataproc job 03 (Structured Streaming) ──> GCS results/streaming
```

- **Job 01** reads all bars, removes duplicates and invalid prices, and
  computes one row per symbol and day with `groupBy(symbol, year).applyInPandas`.
  A second pass computes the average pairwise correlation and absorption ratio
  of a 13-ETF basket from 5-minute returns.
- **Job 02** parallelizes the 84 signal configurations as Spark tasks. Before
  broadcasting, every input is cut at the end of the validation year, so the
  workers cannot read holdout rows. Each task writes metric, quality, return
  and probability rows as JSON lines.
- **Job 03** has no native Pub/Sub source in Spark, so a `rate` stream drives
  micro-batches every 5 seconds. Each batch pulls the waiting messages,
  rebuilds the day's partial features and runs one forward-filter step of an
  HMM fitted strictly before the replay day. A separate Dataproc job publishes
  one historical day to Pub/Sub at 60 times real speed.
- **BigQuery** holds the sweep metrics (472,752 rows), regime quality, job
  statistics and streaming batches for SQL queries.
- **Dashboard.** [dashboard/app.py](../dashboard/app.py) shows regime
  probabilities, the signal-by-rule matrix, model factors, cost sensitivity,
  scaling and the streaming replay.

## 4. Methods

**Features (24 at most).** Eleven daily features: 1-day returns of SPY, TLT
and GLD; their 20-day volatility; log VIX and its 5-day change; the 10y–2y
slope; 5-day credit return (HYG minus LQD); and 5-day equity minus bond
return. Thirteen intraday features: mean log realized variance by asset group;
equity jump share (RV minus bipower variation); SPY overnight and last-30-
minute returns; basket correlation and absorption ratio; cross-sectional
dispersion; and Amihud illiquidity. A feature dated `t` uses data up to the
close of `t`. FRED values are used one trading day late.

**Signals.** (1) Walk-forward Gaussian HMM, 5 random starts, refit monthly
(M) or quarterly (Q) after 504 training days. The forward filter gives
P(z_{t+1} | x_1..x_t), and states are ordered by training-window SPY return.
(2) PCA plus K-means with softmax distances. (3) A month-block shuffle of the
HMM probabilities. (4) A random Markov chain with fixed persistence. Signals
3 and 4 are placebos.

**Configuration grid.** Universe {minimal, core, full} × states {2, 3, 4} ×
covariance {diag, tied} × refit {M, Q} × feature set {daily, intraday, all}.
The daily feature set is the same in every universe, so it is run on
`minimal` only. That leaves 84 configurations.

**Portfolio rules (67 settings).** Hard template, probability template,
probability tilt (5 amplitudes) and tilt with a confidence gate (60 settings),
all around a 60% SPY / 20% TLT / 20% IEF base. Each is executed at the next
open, VWAP or close, with costs of 0, 2, 5 or 10 bps per dollar traded,
including drift rebalancing. Sharpe ratios use daily returns in excess of
T-bills. Annual returns are compound returns after costs.

**Validation.** Warm-up 2016–2017, development 2018–2022, validation 2023,
sealed holdout 2024-01 to 2026-10. Each year, nested walk-forward selection
picks the strategy with the best past net Sharpe. Then we compute:

- a stationary block bootstrap of the Sharpe difference (mean block 21 days,
  5,000 resamples);
- CSCV probability of backtest overfitting (16 partitions);
- the Deflated Sharpe ratio.

## 5. Results

Headline numbers use next-open execution and 2 bps costs.

### RQ1: regime quality

Spearman correlation between the regime score and SPY realized volatility
over the following 5 days. More negative means risk-off states come before
higher volatility.

| Feature set | Development | Validation | Brier (dev) |
|---|---:|---:|---:|
| Daily only | −0.24 | −0.04 | 0.354 |
| Intraday only | **−0.43** | **−0.24** | 0.355 |
| All | −0.33 | −0.15 | 0.349 |

Intraday features separate future volatility more strongly in both periods.
Brier scores for the high-volatility event are about the same in all three.
RQ1 is supported on volatility separation, the criterion in the protocol. The
evidence on calibration is neutral.

### RQ2: portfolio value

Mean net Sharpe in development, signal × rule:

| Signal | Hard template | Probability template | Tilt | Tilt + gate |
|---|---:|---:|---:|---:|
| HMM | 0.32 | 0.35 | 0.43 | 0.43 |
| K-means | 0.19 | 0.21 | 0.37 | 0.38 |
| Markov placebo | 0.36 | 0.39 | 0.44 | 0.44 |
| Shuffle placebo | 0.34 | 0.37 | 0.43 | 0.44 |

The 60/40 benchmark had a Sharpe of 0.42 in development and 1.03 in 2023.
On average the HMM signal is no better than the placebos, and smaller tilts
do better than full swings.

Nested walk-forward selection before the holdout (2019–2023) trailed 60/40 by
0.31 Sharpe (95% CI −0.66 to 0.05). The probability of backtest overfitting
was 0.71 across 2,814 HMM strategies. Selection mostly picked strategies that
then ranked below the median out of sample.

**Sealed holdout, 2024-01-02 to 2026-10-29.** The last selection chose
`minimal-4s-tied-M-daily` with the probability template.

| | Annual return (net) | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| HMM design | 13.1% | 9.7% | 0.88 | −12.1% |
| 60/40 | 11.9% | 10.3% | 0.72 | −12.3% |

| HMM Sharpe minus | Difference | 95% CI |
|---|---:|---|
| 60/40 | +0.18 | −0.20 to 0.54 |
| Shuffle placebo | +0.42 | −0.15 to 0.97 |
| Markov placebo | +0.22 | −0.30 to 0.73 |

The information ratio against 60/40 is 0.24. All three intervals include
zero, so RQ2 is not supported. We report this as a null result.

### RQ3: breadth, frequency, complexity

Spread of mean development Sharpe across the levels of each factor (HMM
strategies):

| Factor | Levels | Sharpe range |
|---|---|---:|
| Number of states | 2: 0.36, 3: 0.46, 4: 0.46 | 0.096 |
| Refit frequency | M: 0.38, Q: 0.47 | 0.086 |
| Feature frequency | daily 0.44, intraday 0.41, all 0.44 | 0.029 |
| Asset breadth | minimal 0.43, core 0.42, full 0.43 | 0.015 |
| Covariance | diag 0.42, tied 0.43 | 0.004 |

Model complexity, meaning the number of states and the refit frequency,
matters most. Intraday features improve volatility separation (RQ1), but this
does not become higher portfolio Sharpe. Going from 8 to 100 ETFs changes
little.

**Cost sensitivity.** Mean Sharpe falls from 0.44 to 0.36 as costs rise from
0 to 10 bps. Annual turnover is about 5 times the portfolio. The execution
price (open, VWAP, close) changes Sharpe by less than 0.03.

## 6. System performance

Runtime on `n2-standard-2` workers (2 vCPUs each):

| Job | 2 workers | 4 workers | 7 workers | Speedup 2→7 |
|---|---:|---:|---:|---:|
| 01 bars to daily, 103M rows | 1,876 s | 977 s | 609 s | 3.1× |
| 02 sweep, 16 configurations | 686 s | 460 s | 343 s | 2.0× |
| 02 sweep, all 84 configurations | | | 1,219 s | |

Job 01 scales almost linearly: 3.1× for 3.5× the workers. Job 02 scales less
well because each configuration is one task and run times differ by up to
4×. With 16 tasks on 14 cores, the last slow configurations set the finish
time.

**Streaming replay of 2023-10-30.**

| | Value |
|---|---|
| Messages | 8,054 bars in 174 micro-batches, all processed |
| End-to-end latency (publish to probability) | median 5.2 s, 95th percentile 11.2 s |
| Queue time | median 2.6 s |
| Processing per batch | median 0.67 s |
| Agreement with batch | closing feature vector matches the batch table to 4×10⁻¹³ |

Probabilities first appear at 12:00 New York time. The co-movement features
need 30 five-minute intervals, so earlier batches have incomplete features.
Most of the latency is the 5-second trigger interval, not computation.

**Cost.** About 18 node-hours of `n2-standard-2` over two clusters, plus
storage, BigQuery and download traffic. At list prices this is about $3,
against a $50 budget. The billing console has the exact figure.

## 7. Deviations from the proposal and limitations

- **Workers.** The project has a 16 vCPU limit, so scaling was measured on
  2, 4 and 7 two-core workers instead of 1, 4 and 16 four-core workers. This
  is recorded in the protocol change log. Research settings did not change.
- **Grid size.** The final grid has 84 configurations, 19,740 HMM fits and
  about 59,000 backtests (472,752 metric rows). The proposal estimated about
  86,000 fits and 78,000 backtests. The difference comes from skipping
  duplicate daily-feature universes and running K-means for only one
  covariance setting.
- **Regime-quality metrics.** The protocol also lists out-of-sample log-
  likelihood. HMMs with different feature sets have densities over different
  spaces, so their likelihoods cannot be compared. RQ1 therefore rests on
  volatility separation and Brier score.
- **Currency volatility.** The currency group is missing because its ETFs
  are illiquid at minute frequency (see Section 2).
- **Statistical power.** The holdout has 710 trading days. A Sharpe
  difference of about 0.2 cannot be told apart from zero in that window.
- **Engineering issues found during the cloud run:**
  - Dataproc 2.2 creates internal-IP clusters by default, so pip installs
    failed until public IPs were enabled.
  - PowerShell dropped the `--` argument separator when submitting jobs.
  - A pandas 2.x concatenation error and a missing symbol scope in the
    online feature path stopped the first streaming run. Both are fixed, and
    the fix was checked against the batch features.

## 8. Reproduction

See [README.md](../README.md). The main steps are:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py bars
.\scripts\gcp.ps1 upload
.\scripts\gcp.ps1 cluster -Workers 7
.\scripts\gcp.ps1 job01
.\scripts\gcp.ps1 job02
.\scripts\gcp.ps1 job03 -ReplayDate 2023-10-30   # with: .\scripts\gcp.ps1 replay
.\scripts\gcp.ps1 bigquery
.\scripts\gcp.ps1 sync
.\.venv\Scripts\python.exe scripts\analyze.py --run full
.\.venv\Scripts\python.exe scripts\open_holdout.py --run full --confirm
streamlit run dashboard\app.py
```
