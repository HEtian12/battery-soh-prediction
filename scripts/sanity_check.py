import sys
from pathlib import Path

sys.path.insert(0, r"F:\AIagent\projects\battery_soh_project\battery-soh-prediction")
import numpy as np

from src.parse_mat import load_battery_mat
from src.features import build_features
from scripts.add_dqdv_features import add_dqdv_features

DATA = Path(r"F:\AIagent\projects\battery_soh_project\battery-soh-prediction\data\raw")

for cell in ["B0005", "B0029", "B0053"]:
    d, s, i = load_battery_mat(DATA / f"{cell}.mat")
    c1 = s[s.cycle_index == 1]
    I = float(abs(c1.current.mean())) if len(c1) else 0.0
    f = build_features(d, s, i)
    f2 = add_dqdv_features(f, s)
    ph = f2["dqdv_peak_height"].values
    pv = f2["dqdv_peak_voltage"].values
    print(f"{cell}: 放电电流 ~{I:.2f}A | 特征行 {len(f2)} | 主峰高(前3) {np.round(ph[:3], 2)} | 主峰位(前3) {np.round(pv[:3], 3)}")
