"""SOC 估计：库仑计数算真实 SOC，用电压/温度估计 SOC。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.parse_mat import load_battery_mat

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
d, s, i = load_battery_mat(DATA / "B0018.mat")

# 逐放电循环：库仑计数算 SOC，收集 (V, T, 时间比例) -> SOC 样本
rows = []
for cyc, grp in s.groupby("cycle_index"):
    t = grp.time.values.astype(float)
    V = grp.voltage.values.astype(float)
    I = grp.current.values.astype(float)
    T = grp.temperature.values.astype(float)
    if len(t) < 5:
        continue
    cap = float(d[d.cycle_index == cyc].capacity.iloc[0])  # Ah
    # 库仑计数：已放电量 Q_disch(t) = ∫|I| dt
    dt = np.diff(t)
    I_mid = (np.abs(I[:-1]) + np.abs(I[1:])) / 2.0
    Q = np.concatenate([[0.0], np.cumsum(I_mid * dt) / 3600.0])  # Ah
    soc = (1 - Q / cap) * 100.0  # 100% -> 0%
    frac = (t - t[0]) / max(t[-1] - t[0], 1e-9)
    for k in range(len(t)):
        rows.append({"cycle": cyc, "voltage": V[k], "temperature": T[k],
                     "time_frac": frac[k], "soc": soc[k]})

df = pd.DataFrame(rows)
print(f"样本数 {len(df)}，SOC 范围 {df.soc.min():.1f}~{df.soc.max():.1f}%")

# 按时序切分：前 80% 循环训练，后 20% 循环测试
cycles = sorted(df.cycle.unique())
cut = int(len(cycles) * 0.8)
train_c, test_c = cycles[:cut], cycles[cut:]
train_df = df[df.cycle.isin(train_c)]
test_df = df[df.cycle.isin(test_c)]
print(f"训练循环 {len(train_c)} 个，测试循环 {len(test_c)} 个")

FEATS = ["voltage", "temperature"]


def run(model):
    pipe = Pipeline([("scale", StandardScaler()), ("m", model)])
    pipe.fit(train_df[FEATS], train_df.soc)
    pred = pipe.predict(test_df[FEATS])
    rmse = float(np.sqrt(mean_squared_error(test_df.soc, pred)))
    mae = float(mean_absolute_error(test_df.soc, pred))
    r2 = float(r2_score(test_df.soc, pred))
    return rmse, mae, r2


from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

for name, model in [
    ("Ridge", Ridge(alpha=1.0)),
    ("RandomForest", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1)),
    ("XGBoost", XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1)),
]:
    rmse, mae, r2 = run(model)
    print(f"{name:14s} SOC RMSE={rmse:.3f}%  MAE={mae:.3f}%  R2={r2:.4f}")
