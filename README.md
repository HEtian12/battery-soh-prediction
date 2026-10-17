# 锂离子电池 SOH / RUL 预测

基于 NASA PCoE 电池退化数据集（B0005、B0006、B0007、B0018），构建端到端的锂离子电池健康状态（SOH）与剩余使用寿命（RUL）预测流水线，并引入 **dQ/dV 增量容量特征**与 **SHAP 可解释性分析**。

## 结果

### B0018 时序留出集（80/20，防泄漏切分）

| 模型 | RMSE(%) | MAE(%) | R² |
|---|---:|---:|---:|
| 多项式基线（deg=2，仅循环号） | 5.07 | 4.95 | -9.48 |
| Ridge | 0.31 | 0.28 | 0.961 |
| 随机森林 | 0.72 | 0.63 | 0.790 |
| SVR | 2.48 | 1.91 | -1.50 |
| XGBoost | 0.73 | 0.44 | 0.784 |
| **高斯过程（GPR）** | **0.13** | **0.11** | **0.993** |

跨电芯泛化（B0005/6/7 训练 → B0018 测试）：Ridge R² 0.996。

### dQ/dV 特征增强

- 引入 dQ/dV 增量容量特征（滑动平均滤波 + 主峰追踪），将电化学相变峰量化为模型特征；
- 主峰高度随老化由 5.15 单调下降至 2.77，对应活性物质损失与内阻增大；
- 加入该特征后，Ridge 模型 **R² 由 0.961 提升至 0.977，RMSE 下降 23%**。

### 可解释性

SHAP 分析表明，模型主要依赖容量相关特征（`time_to_cutoff`、`capacity_prev`），建立了数据特征与电化学机理之间的可解释关联。

## 项目结构

```
├── src/                       # 数据解析、特征、切分、模型、RUL
│   ├── parse_mat.py           # MATLAB 解析
│   ├── features.py            # 特征工程
│   ├── split.py               # 时序切分
│   ├── models.py              # 模型训练与评估
│   └── rul.py                 # RUL 目标构建
├── scripts/
│   ├── run_pipeline.py        # 端到端流水线
│   ├── add_dqdv_features.py   # dQ/dV 特征增强
│   ├── shap_analysis.py       # SHAP 可解释性
│   └── feature_importance.py  # 特征重要性
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt

# 将 NASA 数据集的 .mat 文件放入 data/raw/（B0005/B0006/B0007/B0018.mat）
python scripts/run_pipeline.py
```

## 数据

NASA Prognostics Center of Excellence Battery Data Set（B0005/B0006/B0007/B0018）。
