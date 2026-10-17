"""LSTM 时序预测 SOH：过去 K 个循环 SOH -> 下一个（含目标归一化 + 基线对照）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from src.parse_mat import parse_all_cells
from src.features import build_features

SEED = 42
K = 20
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
parsed = parse_all_cells(DATA)

xs_tr, ys_tr, xs_te, ys_te = [], [], [], []
for cell, (cycles, series, impedance) in parsed.items():
    try:
        f = build_features(cycles, series, impedance)
        if f["capacity"].max() > 2.2 or len(f) < 10:
            continue
        soh = f.sort_values("cycle_index").soh.values.astype(float)
        n = len(soh)
        cut = int(n * 0.8)
        for i in range(K, n):
            x, y = soh[i - K:i], soh[i]
            (xs_tr.append(x), ys_tr.append(y)) if i < cut else (xs_te.append(x), ys_te.append(y))
    except Exception:
        continue

Xtr, ytr = np.array(xs_tr), np.array(ys_tr)
Xte, yte = np.array(xs_te), np.array(ys_te)

# 基线：持久性预测（下一个 = 最后一个）
persist_pred = Xte[:, -1]
persist_rmse = float(np.sqrt(np.mean((persist_pred - yte) ** 2)))
persist_r2 = 1 - float(np.sum((yte - persist_pred) ** 2) / np.sum((yte - yte.mean()) ** 2))
print(f"持久性基线(下个=上个): RMSE={persist_rmse:.3f}%  R2={persist_r2:.4f}")

# 归一化输入 + 目标（用训练集统计量）
x_mu, x_sd = Xtr.mean(), Xtr.std() + 1e-8
y_mu, y_sd = ytr.mean(), ytr.std() + 1e-8
Xtr_t = torch.tensor((Xtr - x_mu) / x_sd, dtype=torch.float32).unsqueeze(-1)
ytr_t = torch.tensor((ytr - y_mu) / y_sd, dtype=torch.float32)
Xte_t = torch.tensor((Xte - x_mu) / x_sd, dtype=torch.float32).unsqueeze(-1)


class SOHLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, 1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


model = SOHLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

for epoch in range(300):
    opt.zero_grad()
    loss = loss_fn(model(Xtr_t), ytr_t)
    loss.backward()
    opt.step()
    if (epoch + 1) % 50 == 0:
        print(f"epoch {epoch+1}: loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    pred_n = model(Xte_t).numpy()
pred = pred_n * y_sd + y_mu
rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
mae = float(np.mean(np.abs(pred - yte)))
r2 = 1 - float(np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))
print(f"LSTM SOH: RMSE={rmse:.4f}%  MAE={mae:.4f}%  R2={r2:.4f}")
