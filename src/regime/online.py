"""Update regime probabilities during the trading day from partial minute bars."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from regime.features import intraday
from regime.features.market import build_market_features
from regime.models import signals

HISTORY_DAYS = 80
MIN_LIVE_MINUTES = 30
COMOVEMENT_COLUMNS = ["date", "avg_corr", "absorption", "n_assets"]


@dataclass
class OnlineRegime:
    """A model fitted strictly before the replay day."""

    columns: list[str]
    scaler: StandardScaler
    means: np.ndarray
    covariances: np.ndarray
    transition: np.ndarray
    order: np.ndarray
    prior: np.ndarray
    daily_history: pd.DataFrame
    comovement_history: pd.DataFrame
    macro: pd.DataFrame
    groups: dict[str, list[str]]
    basket: list[str]

    @classmethod
    def fit(cls, features: pd.DataFrame, target: pd.Series, n_states: int, covariance: str, seeds: list[int],
            daily: pd.DataFrame, comovement: pd.DataFrame, macro: pd.DataFrame,
            groups: dict[str, list[str]], basket: list[str], before: pd.Timestamp) -> "OnlineRegime":
        history = features.loc[features.index < before].dropna()
        scaler = StandardScaler().fit(history.to_numpy())
        train = np.clip(scaler.transform(history.to_numpy()), -signals.CLIP, signals.CLIP)
        model = signals.fit_hmm(train, n_states, covariance, seeds, 100, 1e-4)
        covariances, means, transition = np.asarray(model.covars_), np.asarray(model.means_), np.asarray(model.transmat_)
        density = signals.gaussian_log_density(train, means, covariances)
        filtered, predicted, _ = signals.forward_filter(density, transition, np.asarray(model.startprob_))
        order = signals.state_order(filtered, target.reindex(history.index).fillna(0.0).to_numpy())
        dates = sorted(daily.loc[daily["date"] < before, "date"].unique())[-HISTORY_DAYS:]
        daily_history = daily.loc[daily["date"].isin(dates), intraday.DAILY_COLUMNS].reset_index(drop=True)
        comovement_history = comovement.loc[comovement["date"].isin(dates), COMOVEMENT_COLUMNS].reset_index(drop=True)
        return cls(
            list(history.columns), scaler, means, covariances, transition, order, predicted[-1],
            daily_history, comovement_history, macro, groups, basket,
        )

    @staticmethod
    def _append(history: pd.DataFrame, today: pd.DataFrame) -> pd.DataFrame:
        # Same columns and dtypes on both sides; pandas 2 fails on mixed datetime layouts.
        today = today[list(history.columns)].astype(history.dtypes.to_dict())
        return pd.concat([history, today], ignore_index=True)

    def feature_vector(self, bars: pd.DataFrame) -> pd.Series | None:
        rows = []
        for _, group in bars.groupby("symbol"):
            row = intraday.daily_features(group, min_minutes=MIN_LIVE_MINUTES)
            if not row.empty:
                rows.append(row)
        if not rows or "SPY" not in {r["symbol"].iloc[0] for r in rows}:
            return None
        today = pd.concat(rows, ignore_index=True)
        basket = bars[bars["symbol"].isin(self.basket)]
        session = intraday.regular_session(basket)
        comove = intraday.basket_comovement(intraday.bucket_returns(session))
        table = build_market_features(
            self._append(self.daily_history, today),
            self.macro,
            self._append(self.comovement_history, comove),
            self.groups,
            # Same symbol scope as the batch tables, so small groups are kept the same way.
            symbols=sorted(self.daily_history["symbol"].unique()),
        )
        return table.iloc[-1][self.columns]

    def update(self, bars: pd.DataFrame) -> dict | None:
        """Return provisional state probabilities given the day's bars so far."""
        vector = self.feature_vector(bars)
        if vector is None or vector.isna().any():
            return None
        scaled = np.clip(self.scaler.transform(vector.to_frame().T.to_numpy()), -signals.CLIP, signals.CLIP)
        density = signals.gaussian_log_density(scaled, self.means, self.covariances)
        filtered, _, _ = signals.forward_filter(density, self.transition, self.prior)
        ordered = filtered[0][self.order]
        return {f"p{k}": float(v) for k, v in enumerate(ordered)}
