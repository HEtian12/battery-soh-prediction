"""Remaining useful life target and metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def add_rul_target(data: pd.DataFrame, eol_soh: float = 70.0) -> pd.DataFrame:
    """Add cycles remaining until the first observed SOH at or below the threshold."""
    result = data.sort_values("cycle_index").copy()
    crossing = result.loc[result.soh <= eol_soh, "cycle_index"]
    if crossing.empty:
        raise ValueError(f"No observed EOL crossing at {eol_soh:g}% SOH")
    result["rul_cycles"] = (int(crossing.iloc[0]) - result.cycle_index).clip(lower=0)
    return result


def rul_rmse(actual: pd.Series, predicted: np.ndarray) -> float:
    """Return RUL root mean squared error measured in cycles."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))
