"""Extract scalar degradation and electrochemical features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

FEATURE_COLUMNS = ["capacity_prev", "capacity_delta", "rolling_mean_capacity_3", "time_to_cutoff", "voltage_at_t_25pct", "voltage_at_t_50pct", "voltage_at_t_75pct", "mean_voltage", "min_voltage", "voltage_curve_auc", "voltage_plateau_slope", "max_temperature", "temperature_rise", "mean_temperature", "Re", "Rct"]


def extract_cycle_features(cycle_data: dict) -> dict:
    """Return flat voltage, duration, temperature, and curve-shape features."""
    time = np.asarray(cycle_data.get("Time", cycle_data.get("time", [])), dtype=float).reshape(-1)
    voltage = np.asarray(cycle_data.get("Voltage_measured", cycle_data.get("voltage", [])), dtype=float).reshape(-1)
    temperature = np.asarray(cycle_data.get("Temperature_measured", cycle_data.get("temperature", [])), dtype=float).reshape(-1)
    valid = np.isfinite(time) & np.isfinite(voltage)
    time, voltage = time[valid], voltage[valid]
    if len(temperature) == len(valid):
        temperature = temperature[valid]
    if len(time) < 4:
        raise ValueError("fewer than four finite voltage/time samples")
    order = np.argsort(time); time, voltage = time[order], voltage[order]
    duration = max(float(time[-1] - time[0]), 1e-9)
    sample = lambda fraction: float(np.interp(time[0] + fraction * duration, time, voltage))
    middle = (time >= time[0] + .25 * duration) & (time <= time[0] + .75 * duration)
    slope = float(np.polyfit(time[middle], voltage[middle], 1)[0]) if middle.sum() >= 2 else np.nan
    temp = temperature[np.isfinite(temperature)] if len(temperature) else np.array([np.nan])
    cutoff = time[voltage <= 2.5]
    return {"time_to_cutoff": float(cutoff[0] - time[0]) if len(cutoff) else duration, "voltage_at_t_25pct": sample(.25), "voltage_at_t_50pct": sample(.5), "voltage_at_t_75pct": sample(.75), "mean_voltage": float(voltage.mean()), "min_voltage": float(voltage.min()), "voltage_curve_auc": float(trapezoid(voltage, time)), "voltage_plateau_slope": slope, "max_temperature": float(np.nanmax(temp)), "temperature_rise": float(np.nanmax(temp) - temp[0]), "mean_temperature": float(np.nanmean(temp))}


def build_features(cycles: pd.DataFrame, series: pd.DataFrame, impedance: pd.DataFrame, rated_capacity: float = 2.0) -> pd.DataFrame:
    """Join cycle features and nearest-prior impedance readings without future leakage."""
    rows = []
    ordered = cycles.sort_values("cycle_index").reset_index(drop=True)
    for position, cycle in ordered.iterrows():
        samples = series[series.cycle_index == cycle.cycle_index]
        try:
            values = extract_cycle_features(samples.to_dict("list"))
        except ValueError:
            continue
        previous = ordered.iloc[position - 1].capacity if position else np.nan
        prior = impedance[impedance.raw_cycle_index < cycle.raw_cycle_index].tail(1) if not impedance.empty else pd.DataFrame()
        values.update(cycle.to_dict()); values["capacity_prev"] = previous
        values["capacity_delta"] = previous - cycle.capacity if pd.notna(previous) else np.nan
        values["rolling_mean_capacity_3"] = ordered.iloc[max(0, position - 2):position + 1].capacity.mean()
        values["Re"] = prior.Re.iloc[0] if len(prior) else np.nan; values["Rct"] = prior.Rct.iloc[0] if len(prior) else np.nan
        values["soh"] = cycle.capacity / rated_capacity * 100
        rows.append(values)
    return pd.DataFrame(rows).sort_values("cycle_index").reset_index(drop=True)
