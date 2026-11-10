"""Project configuration and the sealed-holdout guard."""

from __future__ import annotations

import json
import os
from importlib import resources

import pandas as pd

HOLDOUT_FLAG = "REGIME_OPEN_HOLDOUT"


class HoldoutSealedError(RuntimeError):
    """Raised when holdout rows are requested while the holdout is sealed."""


def _load_resource(name: str) -> dict:
    # importlib.resources also works when the package ships as a zip on Spark.
    return json.loads(resources.files("regime.resources").joinpath(name).read_text(encoding="utf-8"))


def load_universe() -> dict:
    return _load_resource("universe.json")


def load_experiment() -> dict:
    return _load_resource("experiment.json")


def full_universe(universe: dict) -> list[str]:
    return sorted({s for members in universe["groups"].values() for s in members})


def holdout_open() -> bool:
    return os.environ.get(HOLDOUT_FLAG) == "1"


def select_period(frame: pd.DataFrame, period: str, split: dict) -> pd.DataFrame:
    """Return rows of a date-indexed frame belonging to a named period.

    Periods are ``dev``, ``val``, ``dev_val``, ``holdout`` and ``all``. Asking
    for ``holdout`` or ``all`` while the holdout is sealed raises an error.
    """
    dates = pd.DatetimeIndex(frame.index)
    dev_start = pd.Timestamp(split["dev_start"])
    dev_end = pd.Timestamp(split["dev_end"])
    val_end = pd.Timestamp(split["val_end"])

    if period in {"holdout", "all"} and not holdout_open():
        raise HoldoutSealedError(f"Set {HOLDOUT_FLAG}=1 to read period '{period}'.")

    masks = {
        "dev": (dates >= dev_start) & (dates <= dev_end),
        "val": (dates > dev_end) & (dates <= val_end),
        "dev_val": (dates >= dev_start) & (dates <= val_end),
        "holdout": dates > val_end,
        "all": dates >= dev_start,
    }
    if period not in masks:
        raise ValueError(f"Unknown period: {period}")
    return frame.loc[masks[period]]


def seal(frame: pd.DataFrame, split: dict) -> pd.DataFrame:
    """Drop rows after the validation end unless the holdout has been opened."""
    if holdout_open():
        return frame
    return frame.loc[pd.DatetimeIndex(frame.index) <= pd.Timestamp(split["val_end"])]
