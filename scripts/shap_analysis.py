"""SHAP 可解释性：特征重要性 + 单样本拆解。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xgboost as xgb
import shap

from src.parse_mat import load_battery_mat
from src.features import build_features, FEATURE_COLUMNS
from src.split import chronological_split

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "B0018.mat"
d, s, i = load_battery_mat(DATA)
f = build_features(d, s, i)
train, test = chronological_split(f)

cols = [c for c in FEATURE_COLUMNS if c in train]
xt = train[cols].replace([np.inf, -np.inf], np.nan)
xe = test[cols].replace([np.inf, -np.inf], np.nan)
med = xt.median()
xt = xt.fillna(med).fillna(0)
xe = xe.fillna(med).fillna(0)

m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
m.fit(xt, train.soh)

print("shap version:", shap.__version__)
explainer = shap.TreeExplainer(m)
sv = explainer.shap_values(xe)
if isinstance(sv, list):
    sv = sv[0]

# ① 特征重要性 = mean |SHAP|
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
print("\n=== SHAP 特征重要性（mean|SHAP|）===")
for idx in order[:8]:
    print(f"{cols[idx]:28s} {mean_abs[idx]:.4f}")

# ② 单样本拆解（第 0 个测试样本）
row = 0
shap_row = sv[row]
idx_sorted = np.argsort(np.abs(shap_row))[::-1]
base = float(explainer.expected_value)
pred = base + shap_row.sum()
print(f"\n=== 测试样本 cycle {int(test.cycle_index.values[row])} 的 SHAP 拆解 ===")
print(f"真实 SOH = {test.soh.values[row]:.2f}%")
print(f"基准值 base_value = {base:.2f}%")
print(f"预测值 = base + sum(SHAP) = {pred:.2f}%")
print("各特征贡献（按|贡献|排序）:")
for idx in idx_sorted[:8]:
    print(f"  {cols[idx]:28s} SHAP={shap_row[idx]:+.4f}  特征值={xe.iloc[row][cols[idx]]:.3f}")
