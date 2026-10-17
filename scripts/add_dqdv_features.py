"""把 dQ/dV 主峰抽成特征，对比加/不加对 SOH 预测的影响。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.parse_mat import load_battery_mat
from src.features import build_features, FEATURE_COLUMNS
from src.split import chronological_split
from src.models import metrics


def add_dqdv_features(features: pd.DataFrame, samples: pd.DataFrame, window: int = 31) -> pd.DataFrame:
    """对每个放电循环，算 dQ/dV，抽主峰高度和主峰位置，合并进特征表。"""
    rows = []
    for cycle_index, grp in samples.groupby("cycle_index"):
        t = grp.time.values
        V = grp.voltage.values
        # 电流恒定 2A，容量 Q = I * t（转成 Ah）
        Q = 2.0 * (t - t[0]) / 3600.0
        # 滤波：滑动平均，去掉测量毛刺（否则微分会爆炸）
        Vs = np.convolve(V, np.ones(window) / window, mode="same")
        dV = np.diff(Vs)
        dQ = np.diff(Q)
        with np.errstate(divide="ignore", invalid="ignore"):
            dqdv = -dQ / dV  # 放电取负，峰朝上
        vmid = (Vs[:-1] + Vs[1:]) / 2.0  # diff 后取电压中点
        # 只看 3.3~3.6V 区间（石墨主峰在这里；3.0V 附近是放电末端假峰，要避开）
        mask = (vmid >= 3.3) & (vmid <= 3.6) & np.isfinite(dqdv)
        vals = np.clip(dqdv[mask], 0, 50)
        vm = vmid[mask]
        if len(vals) == 0:
            rows.append({"cycle_index": cycle_index, "dqdv_peak_height": np.nan, "dqdv_peak_voltage": np.nan})
            continue
        imax = int(np.argmax(vals))
        rows.append({"cycle_index": cycle_index, "dqdv_peak_height": float(vals[imax]),
                     "dqdv_peak_voltage": float(vm[imax])})
    extra = pd.DataFrame(rows)
    return features.merge(extra, on="cycle_index")


def run_ridge(train, test, cols):
    xt = train[cols].replace([np.inf, -np.inf], np.nan)
    xe = test[cols].replace([np.inf, -np.inf], np.nan)
    med = xt.median()
    xt = xt.fillna(med).fillna(0)
    xe = xe.fillna(med).fillna(0)
    m = Pipeline([("scale", StandardScaler()), ("m", Ridge(alpha=1.0))])
    m.fit(xt, train.soh)
    return metrics(test.soh, m.predict(xe))


if __name__ == "__main__":
    DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "B0018.mat"
    d, s, i = load_battery_mat(DATA)
    f = build_features(d, s, i)
    f = add_dqdv_features(f, s)

    print("新特征两列示例（前5行）:")
    print(f[["cycle_index", "dqdv_peak_height", "dqdv_peak_voltage"]].head(5).round(3).to_string(index=False))
    print("主峰高度: 首循环 vs 末循环 =", round(f.dqdv_peak_height.iloc[0], 2), "vs",
          round(f.dqdv_peak_height.iloc[-1], 2))

    train, test = chronological_split(f)
    cols_old = [c for c in FEATURE_COLUMNS if c in train]
    cols_new = cols_old + ["dqdv_peak_height", "dqdv_peak_voltage"]

    print("\nRidge 不加 dQ/dV:", {k: round(v, 4) for k, v in run_ridge(train, test, cols_old).items()})
    print("Ridge 加了 dQ/dV:", {k: round(v, 4) for k, v in run_ridge(train, test, cols_new).items()})
