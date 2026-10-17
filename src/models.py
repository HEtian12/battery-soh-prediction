"""Chronological regressors and evaluation metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from .features import FEATURE_COLUMNS


def metrics(actual, predicted) -> dict[str, float]:
    """Compute RMSE, MAE, and R2."""
    return {"RMSE": float(np.sqrt(mean_squared_error(actual, predicted))), "MAE": float(mean_absolute_error(actual, predicted)), "R2": float(r2_score(actual, predicted))}


def polynomial_baseline(train, test, degree=2):
    """Fit the cycle-index-only polynomial baseline."""
    model = Pipeline([("poly", PolynomialFeatures(degree)), ("scale", StandardScaler()), ("ridge", Ridge(alpha=1e-6))])
    model.fit(train[["cycle_index"]], train.soh); pred = model.predict(test[["cycle_index"]])
    return metrics(test.soh, pred), pred


def train_models(train: pd.DataFrame, test: pd.DataFrame, include_cycle_index=False):
    """Train Ridge, RF, SVR, XGBoost, and GPR on a chronological holdout."""
    columns = (["cycle_index"] if include_cycle_index else []) + FEATURE_COLUMNS
    columns = [column for column in columns if column in train]
    x_train, x_test = train[columns].replace([np.inf, -np.inf], np.nan), test[columns].replace([np.inf, -np.inf], np.nan)
    median = x_train.median(); x_train = x_train.fillna(median).fillna(0); x_test = x_test.fillna(median).fillna(0)
    y_train, y_test = train.soh, test.soh
    cv = TimeSeriesSplit(n_splits=min(5, max(2, len(train) // 25)))
    models = {"Ridge": (Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1))]), {}), "RandomForest": (RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1), {"max_depth": [None, 8], "min_samples_leaf": [1, 2]}), "SVR": (Pipeline([("scale", StandardScaler()), ("model", SVR(C=10))]), {})}
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = (XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=1), {"max_depth": [2, 4], "learning_rate": [.05, .1], "n_estimators": [100]})
    except ImportError:
        pass
    results, fitted = [], {}
    for name, (estimator, grid) in models.items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1) if grid else estimator
        search.fit(x_train, y_train); pred = search.predict(x_test)
        results.append({"model": name, **metrics(y_test, pred), "notes": "cycle_index included" if include_cycle_index else "cycle_index excluded"}); fitted[name] = (search, pred, columns)
    kernel = ConstantKernel(1) * Matern(nu=1.5) + WhiteKernel(1e-3)
    gp = Pipeline([("scale", StandardScaler()), ("model", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42))])
    gp.fit(x_train, y_train); pred, std = gp.named_steps["model"].predict(gp.named_steps["scale"].transform(x_test), return_std=True)
    results.append({"model": "GaussianProcess", **metrics(y_test, pred), "notes": "95% interval available"}); fitted["GaussianProcess"] = (gp, pred, columns, std)
    return pd.DataFrame(results), fitted
