"""用 XGBoost 的 feature_importances 看哪些特征在驱动 SOH 预测。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import xgboost as xgb

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

imp = m.feature_importances_
order = np.argsort(imp)[::-1]
print("=== XGBoost 特征重要性（从高到低）===")
for idx in order:
    print(f"{cols[idx]:28s} {imp[idx]:.4f}")
