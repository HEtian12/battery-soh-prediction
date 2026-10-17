"""Load NASA PCoE MATLAB files into discharge, series, and impedance tables."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.io import loadmat


def _array(value: Any) -> np.ndarray:
    """Convert a MATLAB value to a flat numeric array."""
    if value is None:
        return np.array([], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def _text(value: Any) -> str:
    """Convert a MATLAB scalar or character array to text."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
        return "".join(str(item) for item in value).strip()
    return str(value).strip()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dict or scipy MATLAB struct."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def load_battery_mat(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse one NASA file into discharge metadata, long raw series, and impedance rows."""
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    root = next(value for key, value in loaded.items() if not key.startswith("__"))
    cycle = _get(root, "cycle")
    if cycle is None and isinstance(root, dict):
        cycle = root.get("cycle")
    cycles = np.atleast_1d(cycle)
    discharge, samples, impedance = [], [], []
    for raw_index, item in enumerate(cycles, start=1):
        kind = _text(_get(item, "type", "")).lower()
        data = _get(item, "data", {})
        timestamp = _text(_get(item, "time", ""))
        if kind == "impedance":
            re, rct = _array(_get(data, "Re")), _array(_get(data, "Rct"))
            impedance.append({"raw_cycle_index": raw_index, "cycle_timestamp": timestamp,
                             "Re": re[0] if len(re) else np.nan, "Rct": rct[0] if len(rct) else np.nan})
            continue
        if kind != "discharge":
            continue
        arrays = {key: _array(_get(data, key)) for key in ("Voltage_measured", "Current_measured", "Temperature_measured", "Time")}
        capacity = _array(_get(data, "Capacity"))
        if not len(capacity) or not len(arrays["Voltage_measured"]):
            continue
        cycle_index = len(discharge) + 1
        ambient = _array(_get(item, "ambient_temperature"))
        discharge.append({"cycle_index": cycle_index, "raw_cycle_index": raw_index,
                          "cycle_timestamp": timestamp, "ambient_temperature": ambient[0] if len(ambient) else np.nan,
                          "capacity": capacity[0]})
        count = min(len(values) for values in arrays.values() if len(values))
        for index in range(count):
            samples.append({"cycle_index": cycle_index, "time": arrays["Time"][index],
                            "voltage": arrays["Voltage_measured"][index],
                            "current": arrays["Current_measured"][index],
                            "temperature": arrays["Temperature_measured"][index] if len(arrays["Temperature_measured"]) > index else np.nan})
    return pd.DataFrame(discharge), pd.DataFrame(samples), pd.DataFrame(impedance)


def parse_all_cells(raw_dir: str | Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Parse every available B0005/B0006/B0007/B0018 file."""
    return {cell: load_battery_mat(Path(raw_dir) / f"{cell}.mat") for cell in ("B0005", "B0006", "B0007", "B0018") if (Path(raw_dir) / f"{cell}.mat").exists()}
