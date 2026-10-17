"""电池异常循环检测：GP 不确定性 + SOH 突变（容量再生）检测。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.parse_mat import load_battery_mat
from src.features import build_features, FEATURE_COLUMNS
from src.split import chronological_split

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
d, s, i = load_battery_mat(DATA / "B0018.mat")
f = build_features(d, s, i)

soh = f.soh.values
cycles = f.cycle_index.values

# ===== 1) SOH 突变检测：相邻循环 SOH 差值 dSOH =====
dSOH = np.diff(soh)
# 用稳健阈值：|dSOH| 超过 3 倍中位数绝对偏差(MAD)视为突变
med_dsoh = np.median(dSOH)
mad = np.median(np.abs(dSOH - med_dsoh))
thresh = 3 * 1.4826 * mad  # MAD 到标准差的稳健换算
print(f"SOH 相邻差值 dSOH：中位数 {med_dsoh:.3f}%，MAD {mad:.3f}%，阈值 {thresh:.3f}%")

jump_up = dSOH > thresh    # 容量再生（突然上升）
jump_down = dSOH < -thresh  # 突然下降（加速老化/异常）
print(f"\n容量再生（突升）循环 {jump_up.sum()} 个：")
for idx in np.where(jump_up)[0]:
    print(f"  cycle {int(cycles[idx])} -> {int(cycles[idx+1])}: SOH {soh[idx]:.2f}% -> {soh[idx+1]:.2f}%  (+{dSOH[idx]:.2f}%)")
print(f"突降循环 {jump_down.sum()} 个：")
for idx in np.where(jump_down)[0]:
    print(f"  cycle {int(cycles[idx])} -> {int(cycles[idx+1])}: SOH {soh[idx]:.2f}% -> {soh[idx+1]:.2f}%  ({dSOH[idx]:.2f}%)")

# ===== 2) GP 不确定性（预警信号）=====
train, test = chronological_split(f)
cols = [c for c in FEATURE_COLUMNS if c in train]
X = f[cols].replace([np.inf, -np.inf], np.nan)
Xtr = train[cols].replace([np.inf, -np.inf], np.nan)
med = Xtr.median()
X = X.fillna(med).fillna(0)
Xtr = Xtr.fillna(med).fillna(0)

kernel = ConstantKernel(1) * Matern(nu=1.5) + WhiteKernel(1e-3)
gp = Pipeline([("scale", StandardScaler()),
               ("m", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42))])
gp.fit(Xtr, train.soh)
_, std = gp.named_steps["m"].predict(gp.named_steps["scale"].transform(X), return_std=True)
print(f"\nGP 不确定性 std 范围 {std.min():.3f}%~{std.max():.3f}%，"
      f"末期（外推段）平均 std {std[len(train):].mean():.3f}% > 训练段 {std[:len(train)].mean():.3f}%")
