# 锂离子电池 SOH / RUL 预测（NASA 全量数据集）

基于 NASA PCoE 电池退化数据集（B0005–B0056，**3 个队列、34 块电芯**），构建端到端的锂离子电池健康状态（SOH）与剩余使用寿命（RUL）预测流水线，并引入 **dQ/dV 增量容量特征**与 **SHAP 可解释性分析**。

## 结果

### 留一电芯交叉验证（Ridge，全量数据）

对每块电芯，用其余全部电芯训练、预测该电芯，轮流进行：

| 队列 | 测试条件 | 电芯数 | RMSE(%) | R² |
|---|---|---:|---:|---:|
| 队列 1 | 室温、同工况 | 4 | 0.89 | 0.99 |
| 队列 2 | 变温 / 变流 | 17 | 2.31 | 0.31 |
| 队列 3 | 混合工况 | 8 | 2.04 | 0.75 |

### 关键发现

- **同队列（同条件）跨电芯泛化可达 R² 0.99**；
- **跨队列（不同温度 / 电流）泛化显著下降**（R² 0.31–0.75）。

结论：**跨工况泛化是电池 SOH 预测的核心难点**，单纯增加电芯数量并不能直接提升跨工况预测能力。

### dQ/dV 特征增强

- 引入 dQ/dV 增量容量特征（滑动平均滤波 + 主峰追踪），**电流自适应地从放电数据读取**（不再写死 2A）；
- 主峰高度随老化单调下降，对应活性物质损失与内阻增大。

### 单电芯基准（B0018 时序留出）

| 模型 | RMSE(%) | R² |
|---|---:|---:|
| 高斯过程（GPR） | 0.13 | 0.993 |
| Ridge | 0.31 | 0.961 |

## 项目结构

```
├── src/                          # 数据解析、特征、切分、模型、RUL
│   ├── parse_mat.py              # MATLAB 解析（自动发现所有 B*.mat）
│   ├── features.py               # 特征工程（自适应截止电压）
│   ├── split.py                  # 时序切分
│   ├── models.py                 # 模型训练与评估
│   └── rul.py                    # RUL 目标构建
├── scripts/
│   ├── run_pipeline.py           # 单电芯端到端流水线
│   ├── evaluate_all_cells.py     # 全量留一电芯交叉验证
│   ├── add_dqdv_features.py      # dQ/dV 特征增强
│   ├── shap_analysis.py          # SHAP 可解释性
│   └── feature_importance.py     # 特征重要性
├── requirements.txt
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt

# 将 NASA 数据集的 .mat 文件放入 data/raw/（见下方数据说明）
python scripts/run_pipeline.py          # 单电芯（B0018）流水线
python scripts/evaluate_all_cells.py    # 全量 34 块电芯留一交叉验证
```

## 数据

NASA Prognostics Center of Excellence Battery Data Set（B0005–B0056）。

- 数据来源：NASA PCoE 数据仓库（https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository ）
- 将 `.mat` 文件放入 `data/raw/` 后即可运行（数据不入库，`.gitignore` 已排除）。
