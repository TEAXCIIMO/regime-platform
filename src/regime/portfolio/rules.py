"""Map regime probabilities to portfolio weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioSpec:
    assets: tuple[str, ...]
    base: np.ndarray
    unit_on: np.ndarray
    unit_off: np.ndarray
    full_amplitude: float

    @classmethod
    def from_config(cls, config: dict) -> "PortfolioSpec":
        assets = tuple(config["base"])
        return cls(
            assets=assets,
            base=np.array([config["base"][a] for a in assets], dtype=float),
            unit_on=np.array([config["unit_on"][a] for a in assets], dtype=float),
            unit_off=np.array([config["unit_off"][a] for a in assets], dtype=float),
            full_amplitude=float(config["full_amplitude"]),
        )


def state_scores(n_states: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n_states)


def state_weights(spec: PortfolioSpec, n_states: int, amplitude: float) -> np.ndarray:
    """Target weights for each regime state, one row per state."""
    scores = state_scores(n_states)
    rows = []
    for score in scores:
        unit = spec.unit_on if score >= 0 else spec.unit_off
        rows.append(spec.base + abs(score) * amplitude * unit)
    templates = np.vstack(rows)
    if (templates < -1e-9).any():
        raise ValueError("Amplitude produces a negative weight.")
    return np.clip(templates, 0.0, None)


def _frame(weights: np.ndarray, signal: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    return pd.DataFrame(weights, index=signal.index, columns=list(spec.assets))


def hard_template(signal: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    probabilities = signal.to_numpy(dtype=float)
    templates = state_weights(spec, probabilities.shape[1], spec.full_amplitude)
    return _frame(templates[probabilities.argmax(axis=1)], signal, spec)


def probability_template(signal: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    probabilities = signal.to_numpy(dtype=float)
    templates = state_weights(spec, probabilities.shape[1], spec.full_amplitude)
    return _frame(probabilities @ templates, signal, spec)


def probability_tilt(signal: pd.DataFrame, spec: PortfolioSpec, amplitude: float) -> pd.DataFrame:
    probabilities = signal.to_numpy(dtype=float)
    templates = state_weights(spec, probabilities.shape[1], amplitude)
    return _frame(probabilities @ templates, signal, spec)


def tilt_with_gate(
    signal: pd.DataFrame, spec: PortfolioSpec, amplitude: float, threshold: float, scale: float
) -> pd.DataFrame:
    probabilities = signal.to_numpy(dtype=float)
    templates = state_weights(spec, probabilities.shape[1], amplitude)
    blended = probabilities @ templates
    gate = np.clip((probabilities.max(axis=1) - threshold) / scale, 0.0, 1.0)[:, None]
    return _frame(spec.base + gate * (blended - spec.base), signal, spec)


def rule_grid(config: dict) -> list[dict]:
    """All rule configurations from the frozen portfolio grid."""
    grid: list[dict] = [{"rule": "hard_template"}, {"rule": "probability_template"}]
    grid += [{"rule": "probability_tilt", "amplitude": a} for a in config["tilt_amplitudes"]]
    grid += [
        {"rule": "tilt_gate", "amplitude": a, "threshold": t, "scale": s}
        for a in config["tilt_amplitudes"]
        for t in config["gate_thresholds"]
        for s in config["gate_scales"]
    ]
    return grid


def apply_rule(signal: pd.DataFrame, spec: PortfolioSpec, rule: dict) -> pd.DataFrame:
    name = rule["rule"]
    if name == "hard_template":
        return hard_template(signal, spec)
    if name == "probability_template":
        return probability_template(signal, spec)
    if name == "probability_tilt":
        return probability_tilt(signal, spec, rule["amplitude"])
    if name == "tilt_gate":
        return tilt_with_gate(signal, spec, rule["amplitude"], rule["threshold"], rule["scale"])
    raise ValueError(f"Unknown rule: {name}")
