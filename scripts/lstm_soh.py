"""LSTM 时序模型预测 SOH（需 torch）。放电 V/I/T 序列 -> SOH。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.parse_mat import load_battery_mat
from src.features import build_features
from src.split import chronological_split

SEED = 42
SEQ_LEN = 128
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
d, s, i = load_battery_mat(DATA / "B0018.mat")
f = build_features(d, s, i)
train, test = chronological_split(f)


def cycle_to_seq(grp, seq_len=SEQ_LEN):
    V = grp.voltage.values.astype(float)
    I = grp.current.values.astype(float)
    T = grp.temperature.values.astype(float)
    idx = np.linspace(0, len(V) - 1, seq_len).astype(int)
    return np.stack([V[idx], I[idx], T[idx]], axis=1)  # (seq_len, 3)


def build(samples, features):
    xs, ys = [], []
    for cyc, soh in zip(features.cycle_index, features.soh):
        grp = samples[samples.cycle_index == cyc]
        if len(grp) < 4:
            continue
        xs.append(cycle_to_seq(grp))
        ys.append(soh)
    return np.stack(xs), np.array(ys)


Xtr, ytr = build(s, train)
Xte, yte = build(s, test)
print(f"训练序列 {Xtr.shape}，测试序列 {Xte.shape}")


class SOHLSTM(nn.Module):
    def __init__(self, in_dim=3, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


model = SOHLSTM()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
ytr_t = torch.tensor(ytr, dtype=torch.float32)
Xte_t = torch.tensor(Xte, dtype=torch.float32)

model.train()
for epoch in range(80):
    opt.zero_grad()
    pred = model(Xtr_t)
    loss = loss_fn(pred, ytr_t)
    loss.backward()
    opt.step()

model.eval()
with torch.no_grad():
    pred = model(Xte_t).numpy()
rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
mae = float(np.mean(np.abs(pred - yte)))
ss = 1 - float(np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))
print(f"LSTM SOH: RMSE={rmse:.4f}%  MAE={mae:.4f}%  R2={ss:.4f}")
