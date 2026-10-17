"""全量 NASA 电芯的留一电芯交叉验证（Ridge + 16 基础特征）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.parse_mat import parse_all_cells
from src.features import build_features, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"

parsed = parse_all_cells(DATA)
print(f"解析到 {len(parsed)} 块电芯")

features = {}
skipped = []
for cell, (cycles, series, impedance) in parsed.items():
    try:
        f = build_features(cycles, series, impedance)
        # 数据质量过滤：
        # 1) 整块电芯剔除：容量 > 2.2 Ah（超过 2Ah 额定，明显坏数据）
        if f["capacity"].max() > 2.2:
            skipped.append((cell, f"容量异常 max={f['capacity'].max():.2f}"))
            continue
        # 2) 逐循环剔除：容量 < 0.05 Ah 的异常循环（保留该电芯其余循环）
        f = f[f["capacity"] >= 0.05]
        if len(f) < 10:
            skipped.append((cell, f"过滤后仅 {len(f)} 个有效放电循环"))
            continue
        f = f.copy()
        f["battery_id"] = cell
        f["cohort"] = "cohort1" if int(cell[1:]) <= 18 else ("cohort2" if int(cell[1:]) <= 44 else "cohort3")
        features[cell] = f
    except Exception as e:
        skipped.append((cell, repr(e)))

print(f"成功建特征 {len(features)} 块，跳过 {len(skipped)} 块")
for cell, why in skipped:
    print(f"  跳过 {cell}: {why}")

if not features:
    raise SystemExit("没有可用电芯")

all_df = pd.concat(features.values(), ignore_index=True)
cols = [c for c in FEATURE_COLUMNS if c in all_df.columns]
print(f"合并后总行数 {len(all_df)}，特征列 {len(cols)}")

results = []
for heldout, test_df in features.items():
    train_df = all_df[all_df["battery_id"] != heldout]
    xt = train_df[cols].replace([np.inf, -np.inf], np.nan)
    xe = test_df[cols].replace([np.inf, -np.inf], np.nan)
    med = xt.median()
    xt = xt.fillna(med).fillna(0)
    xe = xe.fillna(med).fillna(0)
    m = Pipeline([("scale", StandardScaler()), ("m", Ridge(alpha=1.0))])
    m.fit(xt, train_df.soh)
    pred = m.predict(xe)
    results.append({
        "heldout": heldout,
        "cohort": test_df["cohort"].iloc[0],
        "n": len(test_df),
        "RMSE": float(np.sqrt(mean_squared_error(test_df.soh, pred))),
        "MAE": float(mean_absolute_error(test_df.soh, pred)),
        "R2": float(r2_score(test_df.soh, pred)),
    })

res = pd.DataFrame(results)
print("\n=== 每块电芯留一结果 ===")
print(res.to_string(index=False))
print("\n=== 按队列平均 ===")
print(res.groupby("cohort").agg(
    n_cells=("heldout", "count"),
    RMSE=("RMSE", "mean"),
    MAE=("MAE", "mean"),
    R2=("R2", "mean"),
).round(4).to_string())
print("\n=== 全体平均 ===")
print(f"RMSE = {res.RMSE.mean():.4f}%")
print(f"MAE  = {res.MAE.mean():.4f}%")
print(f"R2   = {res.R2.mean():.4f}")

out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)
res.to_csv(out_dir / "leave_one_cell_out_results.csv", index=False, encoding="utf-8-sig")
print(f"\n结果已存: {out_dir / 'leave_one_cell_out_results.csv'}")
