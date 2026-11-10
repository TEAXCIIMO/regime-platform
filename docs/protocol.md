# Evaluation protocol

This protocol was fixed before any market data was loaded. Changes after that
point must be recorded in the change log at the bottom, with a reason.

## Research questions

- **RQ1.** Do intraday-derived features detect market regimes better than daily
  features alone?
- **RQ2.** Do regime signals add portfolio value over a static benchmark and
  placebo signals, after costs?
- **RQ3.** Which matters most: asset breadth, data frequency, or model
  complexity?

## Data split

| Period | Dates | Use |
|---|---|---|
| Warm-up | 2016-01-01 to 2017-12-31 | Initial model training only |
| Development | 2018-01-01 to 2022-12-31 | Design, parameter selection, PBO |
| Validation | 2023-01-01 to 2023-12-31 | One check of the selected design |
| Sealed holdout | 2024-01-01 onward | Opened once, in the final week |

The code refuses to return holdout rows unless `REGIME_OPEN_HOLDOUT=1` is set.

## Information timing

- Features dated `t` use only data available after the close of `t`.
- FRED series dated `d` are treated as available one trading day after `d`.
- A signal dated `t` is executed on `t+1` at the open, VWAP, or close. The
  default for headline results is the next open.

## Signals (4)

1. Walk-forward Gaussian HMM.
2. Walk-forward PCA + K-means, using softmax distances as probabilities.
3. Placebo: month-block shuffle of the baseline HMM probabilities.
4. Placebo: random Markov chain with the persistence fixed below.

## Portfolio rules (4)

Assets are SPY, TLT, IEF and GLD. The base portfolio is 60% SPY, 20% TLT and
20% IEF. Regime states are ordered by training-window SPY return and mapped to
scores from -1 to +1.

Direction vectors, fixed in advance:

| Asset | Risk-on unit | Risk-off unit |
|---|---:|---:|
| SPY | +1.0 | -1.0 |
| TLT | -0.5 | +0.4 |
| IEF | -0.5 | +0.2 |
| GLD | 0.0 | +0.4 |

1. **Hard template:** full swing (amplitude 0.40) using the most likely state.
2. **Probability template:** full swing, averaged over state probabilities.
3. **Probability tilt:** amplitude in {0.05, 0.10, 0.15, 0.20, 0.30}.
4. **Tilt + confidence gate:** adds threshold in {0.5, 0.6, 0.7, 0.8} and scale
   in {0.1, 0.2, 0.3}.

Parameters are selected only by nested walk-forward on past net Sharpe.

## Metrics

- **Regime quality:** out-of-sample predictive log-likelihood, Brier score and
  calibration against forward realized volatility states, and separation of
  forward 5-day realized volatility and drawdown between states.
- **Portfolio (net of costs, in excess of T-bills):** annual return, Sharpe,
  information ratio against 60/40, maximum drawdown, annual turnover.
- **Cost model:** 0, 2, 5 and 10 bps per dollar traded, on the strategy and the
  benchmark, including drift rebalancing.
- **Statistical validity:** stationary block bootstrap (mean block 21 days, 5,000
  resamples) confidence interval of the Sharpe difference, probability of
  backtest overfitting (CSCV, 16 partitions), Deflated Sharpe.
- **System:** rows processed, runtime on 1, 4 and 16 workers, cloud cost, and
  streaming end-to-end latency.

## Decision rules

- RQ1 is supported if intraday or combined features beat daily features on
  out-of-sample log-likelihood per feature-standardized observation or forward
  volatility separation, in the development period.
- RQ2 is supported only if the 95% interval for the net Sharpe difference
  against both 60/40 and the best placebo excludes zero on the sealed holdout.
- Null results are reported as findings.

## Change log

- 2026-11-09: The system metric "runtime on 1, 4 and 16 workers" is now runtime
  on 2, 4 and 7 `n2-standard-2` workers. The GCP project allows 16 vCPUs in
  total and 8 external IPs, and a standard Dataproc cluster needs at least 2
  workers. This changes only the system measurement. Research settings are
  unchanged.
