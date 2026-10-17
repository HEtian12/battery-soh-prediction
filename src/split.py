"""Chronological split utilities."""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def chronological_split(data: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered cycles into an initial training period and final test period."""
    ordered = data.sort_values("cycle_index").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_fraction))
    if cut < 2 or cut >= len(ordered):
        raise ValueError("Need at least three cycles for a chronological split")
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Return expanding-window cross-validation."""
    return TimeSeriesSplit(n_splits=n_splits)
