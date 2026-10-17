"""Run the NASA battery SOH pipeline from raw MATLAB files."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parse_mat import parse_all_cells
from src.features import FEATURE_COLUMNS, build_features
from src.models import metrics, polynomial_baseline, train_models
from src.rul import add_rul_target, rul_rmse
from src.split import chronological_split


def append_quality(message: str, path: Path) -> None:
    """Append an auditable data-quality event."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n- {message}\n")


def run(raw_dir: Path, processed_dir: Path, figure_dir: Path, report_path: Path) -> pd.DataFrame:
    """Parse available cells, generate artifacts, and return the B0018 metrics table."""
    processed_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    quality = raw_dir.parent.parent / "data_quality_log.md"
    parsed = parse_all_cells(raw_dir)
    if "B0018" not in parsed:
        raise FileNotFoundError(f"B0018.mat is required in {raw_dir}")
    feature_sets = {}
    for cell, (cycles, series, impedance) in parsed.items():
        features = build_features(cycles, series, impedance)
        feature_sets[cell] = features; features.to_csv(processed_dir / f"features_{cell.lower()}.csv", index=False)
        if features.empty:
            append_quality(f"{cell}: no valid discharge cycles after parsing", quality)
        plt.figure(figsize=(9, 4)); plt.plot(features.cycle_index, features.soh, marker="."); plt.axhline(70, color="crimson", linestyle="--", label="70% EOL"); plt.xlabel("Discharge cycle"); plt.ylabel("SOH (%)"); plt.title(f"{cell} capacity health"); plt.legend(); plt.tight_layout(); plt.savefig(figure_dir / f"soh_{cell.lower()}.png", dpi=160); plt.close()
    data = feature_sets["B0018"]; train, test = chronological_split(data)
    eol = data.loc[data.soh <= 70, "cycle_index"]
    eol_cycle = int(eol.iloc[0]) if len(eol) else None
    baseline_metrics, baseline_pred = polynomial_baseline(train, test)
    results, fitted = train_models(train, test, include_cycle_index=False)
    results = pd.concat([pd.DataFrame([{"model": "Polynomial degree 2", **baseline_metrics, "notes": "cycle_index-only baseline"}]), results], ignore_index=True)
    results.to_csv(processed_dir / "model_results_b0018.csv", index=False)
    plt.figure(figsize=(10, 5)); plt.plot(data.cycle_index, data.soh, label="actual", color="black"); plt.plot(test.cycle_index, baseline_pred, label="polynomial baseline")
    best_name = results.sort_values("RMSE").iloc[0].model
    if best_name in fitted: plt.plot(test.cycle_index, fitted[best_name][1], label=f"best: {best_name}")
    plt.axvline(train.cycle_index.iloc[-1], color="grey", linestyle=":", label="train/test boundary"); plt.xlabel("Cycle"); plt.ylabel("SOH (%)"); plt.legend(); plt.tight_layout(); plt.savefig(figure_dir / "predictions_b0018.png", dpi=160); plt.close()
    rul_note = "No 70% crossing observed; RUL was not scored."
    if eol_cycle:
        rul_data = add_rul_target(data)
        rul_train, rul_test = chronological_split(rul_data)
        rul_columns = [column for column in FEATURE_COLUMNS if column in rul_train]
        rul_x_train = rul_train[rul_columns].replace([np.inf, -np.inf], np.nan)
        rul_x_test = rul_test[rul_columns].replace([np.inf, -np.inf], np.nan)
        medians = rul_x_train.median(); rul_x_train = rul_x_train.fillna(medians).fillna(0); rul_x_test = rul_x_test.fillna(medians).fillna(0)
        rul_model = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
        rul_model.fit(rul_x_train, rul_train.rul_cycles); rul_prediction = rul_model.predict(rul_x_test)
        rul_score = float(np.sqrt(mean_squared_error(rul_test.rul_cycles, rul_prediction)))
        pd.DataFrame({"cycle_index": rul_test.cycle_index, "actual_rul": rul_test.rul_cycles, "predicted_rul": rul_prediction}).to_csv(processed_dir / "rul_predictions_b0018.csv", index=False)
        rul_note = f"70% EOL crossing: cycle {eol_cycle}. Chronological RUL Ridge RMSE: {rul_score:.3f} cycles."
    if all(cell in feature_sets for cell in ("B0005", "B0006", "B0007")):
        cross_train = pd.concat([feature_sets[cell] for cell in ("B0005", "B0006", "B0007")], ignore_index=True)
        cross_results, _ = train_models(cross_train, data, include_cycle_index=False)
        cross_results.to_csv(processed_dir / "cross_cell_results_b0018.csv", index=False)
        cross_note = "Cross-cell metrics are in `data/processed/cross_cell_results_b0018.csv`, trained only on B0005/B0006/B0007."
    else:
        cross_note = "Cross-cell metrics were skipped because all three source cells were not available."
    report_path.write_text("# Results summary\n\n```text\n" + results.to_string(index=False) + f"\n```\n\n{rul_note}\n\n{cross_note}\n\nMetrics are from a chronological 80/20 B0018 holdout.\n", encoding="utf-8")
    return results


def main() -> None:
    """Parse command-line arguments and execute the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--raw-dir", type=Path, default=Path("data/raw")); args = parser.parse_args()
    results = run(args.raw_dir, Path("data/processed"), Path("report/figures"), Path("report/results_summary.md")); print(results.to_string(index=False))


if __name__ == "__main__":
    main()
