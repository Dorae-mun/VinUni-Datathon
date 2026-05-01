from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src import DATA_DIR, MODEL_DIR, OUTPUT_DIR
from src.feature_engineering import (
    ALL_FEATURES,
    STATIC_FEATURES,
    build_feature_frame,
    build_recursive_feature_row,
    build_supervised_training_frame,
)
from src.metrics import evaluate_forecast, evaluate_targets
from src.seasonal_models import (
    CalendarProfileModel,
    DayOfYearProfileModel,
    LGBMCalendarModel,
    LastYearTrendModel,
    RecentYearWeightedModel,
)


MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"
SHAP_EXPLAINER_PATH = MODEL_DIR / "shap_explainer.pkl"
SUBMISSION_PATH = MODEL_DIR / "submission.csv"
VALID_START = "2021-01-01"
VALID_END = "2022-12-31"
COGS_CAP = 0.95
SHAP_REFERENCE_SAMPLE_SIZE = 200


def default_model_params(random_state: int = 42) -> dict[str, Any]:
    return {
        "n_estimators": 1400,
        "learning_rate": 0.03,
        "max_depth": 8,
        "num_leaves": 31,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.2,
        "min_child_samples": 20,
        "random_state": random_state,
        "objective": "regression",
        "n_jobs": -1,
        "verbosity": -1,
    }


def default_xgb_params(random_state: int = 42) -> dict[str, Any]:
    return {
        "n_estimators": 2500,
        "learning_rate": 0.02,
        "max_depth": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def default_elasticnet_params(random_state: int = 42) -> dict[str, Any]:
    return {
        "alpha": 0.0008,
        "l1_ratio": 0.15,
        "fit_intercept": True,
        "max_iter": 8000,
        "random_state": random_state,
    }


@dataclass
class ForecastBundle:
    model_type: str
    revenue_model: Any | None
    cogs_model: Any | None
    features: list[str]
    static_features: list[str]
    params: dict[str, Any]
    use_log_target: bool = True
    metadata: dict[str, Any] | None = None


def load_forecast_inputs(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(data_dir / "sales.csv", parse_dates=["Date"])
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv", parse_dates=["Date"])
    promotions = pd.read_csv(data_dir / "promotions.csv", parse_dates=["start_date", "end_date"])
    web_traffic = pd.read_csv(data_dir / "web_traffic.csv", parse_dates=["date"])
    inventory = pd.read_csv(data_dir / "inventory.csv", parse_dates=["snapshot_date"])
    return sales, sample_submission, promotions, web_traffic, inventory


def _transform_target(values: pd.Series, use_log_target: bool) -> pd.Series:
    return np.log1p(values) if use_log_target else values.copy()


def _inverse_transform(values: np.ndarray, use_log_target: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if use_log_target:
        # Avoid overflow in expm1 for extreme model outputs.
        arr = np.clip(arr, a_min=-50.0, a_max=50.0)
        out = np.expm1(arr)
    else:
        out = arr
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, a_min=0.0, a_max=None)


def train_lgbm_models(
    sales_df: pd.DataFrame,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    use_log_target: bool = True,
) -> tuple[ForecastBundle, pd.DataFrame]:
    train_df = build_supervised_training_frame(
        sales_df=sales_df,
        promotions_df=promotions_df,
        web_traffic_df=web_traffic_df,
        inventory_df=inventory_df,
    )

    params = params or default_model_params()
    X = train_df[list(ALL_FEATURES)]
    y_rev = _transform_target(train_df["Revenue"], use_log_target)
    y_cogs = _transform_target(train_df["COGS"], use_log_target)
    revenue_model = LGBMRegressor(**params)
    cogs_model = LGBMRegressor(**params)
    revenue_model.fit(X, y_rev)
    cogs_model.fit(X, y_cogs)

    bundle = ForecastBundle(
        model_type="lightgbm_recursive",
        revenue_model=revenue_model,
        cogs_model=cogs_model,
        features=list(ALL_FEATURES),
        static_features=list(STATIC_FEATURES),
        params=params,
        use_log_target=use_log_target,
        metadata={"candidate_name": "lightgbm_recursive"},
    )
    return bundle, train_df


def train_xgb_models(
    sales_df: pd.DataFrame,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    use_log_target: bool = True,
) -> tuple[ForecastBundle, pd.DataFrame]:
    train_df = build_supervised_training_frame(
        sales_df=sales_df,
        promotions_df=promotions_df,
        web_traffic_df=web_traffic_df,
        inventory_df=inventory_df,
    )
    params = params or default_xgb_params()
    X = train_df[list(ALL_FEATURES)]
    y_rev = _transform_target(train_df["Revenue"], use_log_target)
    y_cogs = _transform_target(train_df["COGS"], use_log_target)

    revenue_model = XGBRegressor(**params)
    cogs_model = XGBRegressor(**params)
    revenue_model.fit(X, y_rev)
    cogs_model.fit(X, y_cogs)

    bundle = ForecastBundle(
        model_type="xgb_recursive",
        revenue_model=revenue_model,
        cogs_model=cogs_model,
        features=list(ALL_FEATURES),
        static_features=list(STATIC_FEATURES),
        params=params,
        use_log_target=use_log_target,
        metadata={"candidate_name": "xgb_recursive"},
    )
    return bundle, train_df


def train_elasticnet_models(
    sales_df: pd.DataFrame,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    use_log_target: bool = True,
) -> tuple[ForecastBundle, pd.DataFrame]:
    train_df = build_supervised_training_frame(
        sales_df=sales_df,
        promotions_df=promotions_df,
        web_traffic_df=web_traffic_df,
        inventory_df=inventory_df,
    )
    params = params or default_elasticnet_params()
    X = train_df[list(ALL_FEATURES)]
    y_rev = _transform_target(train_df["Revenue"], use_log_target)
    y_cogs = _transform_target(train_df["COGS"], use_log_target)

    revenue_model = ElasticNet(**params)
    cogs_model = ElasticNet(**params)
    revenue_model.fit(X, y_rev)
    cogs_model.fit(X, y_cogs)

    bundle = ForecastBundle(
        model_type="elasticnet_recursive",
        revenue_model=revenue_model,
        cogs_model=cogs_model,
        features=list(ALL_FEATURES),
        static_features=list(STATIC_FEATURES),
        params=params,
        use_log_target=use_log_target,
        metadata={"candidate_name": "elasticnet_recursive"},
    )
    return bundle, train_df


def train_candidate_models(sales_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {
        "last_year_mapping": {
            "revenue": LastYearTrendModel().fit(sales_df),
            "cogs": LastYearTrendModel().fit(sales_df),
        },
        "day_of_year_profile": {
            "revenue": DayOfYearProfileModel(smooth_window=3).fit(sales_df),
            "cogs": DayOfYearProfileModel(smooth_window=3).fit(sales_df),
        },
        "calendar_profile": {
            "revenue": CalendarProfileModel().fit(sales_df),
            "cogs": CalendarProfileModel().fit(sales_df),
        },
        "recent_year_weighted": {
            "revenue": RecentYearWeightedModel().fit(sales_df),
            "cogs": RecentYearWeightedModel().fit(sales_df),
        },
        "lgbm_calendar_only": {
            "revenue": LGBMCalendarModel().fit(sales_df, "Revenue", valid_start=VALID_START, valid_end=VALID_END),
            "cogs": LGBMCalendarModel().fit(sales_df, "COGS", valid_start=VALID_START, valid_end=VALID_END),
        },
    }
    return models


def _predict_candidate(model_obj: Any, dates: pd.Series, target: str) -> np.ndarray:
    pred_df = model_obj.predict(pd.to_datetime(dates))
    return pred_df[target].to_numpy(dtype=float)


def _project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto simplex: w >= 0 and sum(w) = 1.
    """
    v = np.asarray(weights, dtype=float)
    if v.ndim != 1:
        raise ValueError("weights must be 1-D")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u - (cssv - 1.0) / (np.arange(len(u)) + 1) > 0)[0]
    if len(rho) == 0:
        return np.ones_like(v) / len(v)
    rho_idx = rho[-1]
    theta = (cssv[rho_idx] - 1.0) / (rho_idx + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        return np.ones_like(v) / len(v)
    return w / s


def _optimize_weights_simplex(
    preds_dict: dict[str, np.ndarray],
    y_true: np.ndarray,
    max_iter: int = 600,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> tuple[dict[str, float], float]:
    keys = list(preds_dict.keys())
    if not keys:
        return {}, float("inf")

    pred_matrix = np.column_stack([np.asarray(preds_dict[k], dtype=float) for k in keys])
    y = np.asarray(y_true, dtype=float)
    n_models = pred_matrix.shape[1]
    rng = np.random.default_rng(seed=random_state)

    w = np.ones(n_models, dtype=float) / n_models
    best_w = w.copy()
    best_rmse = float(np.sqrt(np.mean((y - np.maximum(pred_matrix @ w, 0.0)) ** 2)))

    for _ in range(max_iter):
        pred = pred_matrix @ w
        grad = (2.0 / len(y)) * (pred_matrix.T @ (pred - y))
        noise = rng.normal(loc=0.0, scale=1e-6, size=n_models)
        w = _project_to_simplex(w - learning_rate * grad + noise)
        pred_clipped = np.maximum(pred_matrix @ w, 0.0)
        rmse_val = float(np.sqrt(np.mean((y - pred_clipped) ** 2)))
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_w = w.copy()

    return dict(zip(keys, best_w.tolist())), best_rmse


def apply_constraints(revenue: np.ndarray, cogs: np.ndarray, cogs_cap: float = COGS_CAP) -> tuple[np.ndarray, np.ndarray]:
    revenue = np.maximum(np.asarray(revenue, dtype=float), 0.0)
    cogs = np.maximum(np.asarray(cogs, dtype=float), 0.0)
    cogs = np.where(cogs > revenue * cogs_cap, revenue * cogs_cap, cogs)
    return revenue, cogs


def sanity_check_predictions(sales_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict[str, float]:
    recent_max = float(
        sales_df.loc[sales_df["Date"] >= sales_df["Date"].max() - pd.DateOffset(years=2), "Revenue"].quantile(0.95)
    )
    return {
        "forecast_revenue_min": float(forecast_df["Revenue"].min()),
        "forecast_revenue_max": float(forecast_df["Revenue"].max()),
        "forecast_revenue_mean": float(forecast_df["Revenue"].mean()),
        "forecast_cogs_min": float(forecast_df["COGS"].min()),
        "forecast_cogs_max": float(forecast_df["COGS"].max()),
        "forecast_cogs_mean": float(forecast_df["COGS"].mean()),
        "recent_revenue_p95_last_2y": recent_max,
    }


def _recursive_forecast_supervised(
    bundle: ForecastBundle,
    history_sales_df: pd.DataFrame,
    future_dates: pd.Series,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    def _prepare_model_features(feature_row: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        # Ensure the model input is strictly numeric and NaN-free for sklearn estimators.
        model_features = feature_row.reindex(columns=feature_names, fill_value=0.0).copy()
        for col in model_features.columns:
            model_features[col] = pd.to_numeric(model_features[col], errors="coerce")
        model_features = model_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return model_features.astype(float)

    history_sales = history_sales_df[["Date", "Revenue", "COGS"]].copy()
    history_sales["Date"] = pd.to_datetime(history_sales["Date"])
    future_dates = pd.to_datetime(pd.Series(future_dates)).sort_values().reset_index(drop=True)
    static_frame = build_feature_frame(
        sales_df=history_sales,
        future_dates=future_dates.tolist(),
        promotions_df=promotions_df,
        web_traffic_df=web_traffic_df,
        inventory_df=inventory_df,
    )
    history = static_frame[["Date", "Revenue", "COGS", "day_of_week", "month"]].copy()
    history["Revenue"] = pd.to_numeric(history["Revenue"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    history["COGS"] = pd.to_numeric(history["COGS"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rev_cap = float(history_sales["Revenue"].quantile(0.995)) if not history_sales.empty else 0.0
    cogs_cap = float(history_sales["COGS"].quantile(0.995)) if not history_sales.empty else 0.0
    if not np.isfinite(rev_cap) or rev_cap <= 0:
        rev_cap = 1e7
    if not np.isfinite(cogs_cap) or cogs_cap <= 0:
        cogs_cap = rev_cap * COGS_CAP
    predictions = []
    for target_date in future_dates:
        feature_row = build_recursive_feature_row(history, target_date, static_frame)
        model_features = _prepare_model_features(feature_row, bundle.features)
        pred_cogs = _inverse_transform(bundle.cogs_model.predict(model_features), bundle.use_log_target)[0]
        pred_cogs = float(np.clip(np.nan_to_num(pred_cogs, nan=0.0, posinf=0.0, neginf=0.0), 0.0, cogs_cap * 5.0))
        history.loc[history["Date"] == target_date, "COGS"] = pred_cogs
        feature_row = build_recursive_feature_row(history, target_date, static_frame)
        model_features = _prepare_model_features(feature_row, bundle.features)
        pred_rev = _inverse_transform(bundle.revenue_model.predict(model_features), bundle.use_log_target)[0]
        pred_rev = float(np.clip(np.nan_to_num(pred_rev, nan=0.0, posinf=0.0, neginf=0.0), 0.0, rev_cap * 5.0))
        pred_rev, pred_cogs = apply_constraints(np.array([pred_rev]), np.array([pred_cogs]))
        pred_rev = float(pred_rev[0])
        pred_cogs = float(pred_cogs[0])
        history.loc[history["Date"] == target_date, "COGS"] = pred_cogs
        history.loc[history["Date"] == target_date, "Revenue"] = pred_rev
        predictions.append({"Date": target_date, "Revenue": pred_rev, "COGS": pred_cogs})
    return pd.DataFrame(predictions)

def validate_candidate_ensemble(
    sales_df: pd.DataFrame,
    candidate_models: dict[str, dict[str, Any]],
    valid_start: str = VALID_START,
    valid_end: str = VALID_END,
    weight_tune_ratio: float = 0.5,
) -> dict[str, Any]:
    train_internal = sales_df[sales_df["Date"] < valid_start].copy()
    valid_df = sales_df[(sales_df["Date"] >= valid_start) & (sales_df["Date"] <= valid_end)].copy()
    valid_df = valid_df.sort_values("Date").reset_index(drop=True)
    tune_rows = max(1, int(len(valid_df) * float(weight_tune_ratio)))
    weight_tune_df = valid_df.iloc[:tune_rows].copy()
    report_df = valid_df.iloc[tune_rows:].copy()
    if report_df.empty:
        report_df = valid_df.copy()
        weight_tune_df = valid_df.copy()

    tune_dates = weight_tune_df["Date"]
    revenue_preds_tune = {
        name: _predict_candidate(models["revenue"], tune_dates, "Revenue")
        for name, models in candidate_models.items()
    }
    cogs_preds_tune = {
        name: _predict_candidate(models["cogs"], tune_dates, "COGS")
        for name, models in candidate_models.items()
    }
    revenue_weights, _ = _optimize_weights_simplex(revenue_preds_tune, weight_tune_df["Revenue"].to_numpy())
    cogs_weights, _ = _optimize_weights_simplex(cogs_preds_tune, weight_tune_df["COGS"].to_numpy())

    val_dates = valid_df["Date"]
    revenue_preds = {
        name: _predict_candidate(models["revenue"], val_dates, "Revenue")
        for name, models in candidate_models.items()
    }
    cogs_preds = {
        name: _predict_candidate(models["cogs"], val_dates, "COGS")
        for name, models in candidate_models.items()
    }
    report_dates = report_df["Date"]
    revenue_preds_report = {
        name: _predict_candidate(models["revenue"], report_dates, "Revenue")
        for name, models in candidate_models.items()
    }
    cogs_preds_report = {
        name: _predict_candidate(models["cogs"], report_dates, "COGS")
        for name, models in candidate_models.items()
    }
    ensemble_rev = sum(revenue_weights[name] * revenue_preds_report[name] for name in revenue_weights)
    ensemble_cogs = sum(cogs_weights[name] * cogs_preds_report[name] for name in cogs_weights)
    ensemble_rev, ensemble_cogs = apply_constraints(ensemble_rev, ensemble_cogs)
    ensemble_df = pd.DataFrame({"Date": report_dates.to_numpy(), "Revenue": ensemble_rev, "COGS": ensemble_cogs})
    metrics = evaluate_targets(report_df[["Date", "Revenue", "COGS"]].reset_index(drop=True), ensemble_df.reset_index(drop=True))

    candidate_rows = []
    for name in candidate_models:
        rev_metrics = evaluate_forecast(valid_df["Revenue"], revenue_preds[name])
        cogs_metrics = evaluate_forecast(valid_df["COGS"], cogs_preds[name])
        candidate_rows.append({"candidate": name, "target": "Revenue", **rev_metrics})
        candidate_rows.append({"candidate": name, "target": "COGS", **cogs_metrics})
    candidate_metrics = pd.DataFrame(candidate_rows)

    return {
        "train_internal": train_internal,
        "validation_df": report_df,
        "weight_tune_df": weight_tune_df,
        "validation_forecast": ensemble_df,
        "validation_metrics": metrics,
        "candidate_metrics": candidate_metrics,
        "revenue_weights": revenue_weights,
        "cogs_weights": cogs_weights,
    }


def build_weighted_ensemble_bundle(
    sales_df: pd.DataFrame,
    candidate_models: dict[str, dict[str, Any]],
    revenue_weights: dict[str, float],
    cogs_weights: dict[str, float],
) -> ForecastBundle:
    return ForecastBundle(
        model_type="weighted_ensemble",
        revenue_model=candidate_models,
        cogs_model=None,
        features=[],
        static_features=[],
        params={},
        use_log_target=False,
        metadata={
            "revenue_weights": revenue_weights,
            "cogs_weights": cogs_weights,
            "candidate_names": list(candidate_models.keys()),
        },
    )


def recursive_forecast(
    bundle: ForecastBundle,
    history_sales_df: pd.DataFrame,
    future_dates: pd.Series | list[pd.Timestamp],
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if bundle.model_type in {"lightgbm_recursive", "xgb_recursive", "elasticnet_recursive"}:
        return _recursive_forecast_supervised(
            bundle,
            history_sales_df,
            pd.Series(future_dates),
            promotions_df=promotions_df,
            web_traffic_df=web_traffic_df,
            inventory_df=inventory_df,
        )

    if bundle.model_type != "weighted_ensemble":
        raise ValueError(f"Unsupported model_type: {bundle.model_type}")

    dates = pd.to_datetime(pd.Series(future_dates))
    candidate_models = bundle.revenue_model
    revenue_weights = bundle.metadata["revenue_weights"]
    cogs_weights = bundle.metadata["cogs_weights"]
    revenue_pred = np.zeros(len(dates), dtype=float)
    cogs_pred = np.zeros(len(dates), dtype=float)
    for name, models in candidate_models.items():
        if revenue_weights.get(name, 0.0) > 0:
            revenue_pred += revenue_weights[name] * _predict_candidate(models["revenue"], dates, "Revenue")
        if cogs_weights.get(name, 0.0) > 0:
            cogs_pred += cogs_weights[name] * _predict_candidate(models["cogs"], dates, "COGS")
    revenue_pred, cogs_pred = apply_constraints(revenue_pred, cogs_pred)
    return pd.DataFrame({"Date": dates.to_numpy(), "Revenue": revenue_pred, "COGS": cogs_pred})


def cross_validate_forecast(bundle: ForecastBundle, sales_df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    base_df = sales_df[["Date", "Revenue", "COGS"]].copy().sort_values("Date").reset_index(drop=True)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(base_df), start=1):
        fold_train = base_df.iloc[train_idx].copy()
        fold_val = base_df.iloc[val_idx].copy()
        fold_pred = recursive_forecast(bundle, fold_train, fold_val["Date"])
        rev_metrics = evaluate_forecast(fold_val["Revenue"], fold_pred["Revenue"])
        cogs_metrics = evaluate_forecast(fold_val["COGS"], fold_pred["COGS"])
        rows.append({"fold": fold, "target": "Revenue", **rev_metrics})
        rows.append({"fold": fold, "target": "COGS", **cogs_metrics})
    scores = pd.DataFrame(rows)
    mean_rows = scores.groupby("target", as_index=False)[["mae", "rmse", "r2"]].mean().assign(fold="mean")
    return pd.concat([scores, mean_rows], ignore_index=True)


def walk_forward_cv_lightgbm(
    sales_df: pd.DataFrame,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    n_splits: int = 5,
    params: dict[str, Any] | None = None,
    use_log_target: bool = True,
) -> pd.DataFrame:
    """
    Leakage-safe CV for the LightGBM recursive forecaster.

    For each fold:
    - fit models on fold_train only
    - recursively forecast fold_val dates using fold_train history only
    """
    base_df = sales_df[["Date", "Revenue", "COGS"]].copy().sort_values("Date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(base_df), start=1):
        fold_train = base_df.iloc[train_idx].copy()
        fold_val = base_df.iloc[val_idx].copy()

        bundle, _ = train_lgbm_models(
            fold_train,
            promotions_df=promotions_df,
            web_traffic_df=web_traffic_df,
            inventory_df=inventory_df,
            params=params,
            use_log_target=use_log_target,
        )
        fold_pred = recursive_forecast(
            bundle,
            history_sales_df=fold_train,
            future_dates=fold_val["Date"],
            promotions_df=promotions_df,
            web_traffic_df=web_traffic_df,
            inventory_df=inventory_df,
        )
        rev_metrics = evaluate_forecast(fold_val["Revenue"], fold_pred["Revenue"])
        cogs_metrics = evaluate_forecast(fold_val["COGS"], fold_pred["COGS"])
        rows.append({"fold": fold, "target": "Revenue", **rev_metrics})
        rows.append({"fold": fold, "target": "COGS", **cogs_metrics})

    scores = pd.DataFrame(rows)
    mean_rows = scores.groupby("target", as_index=False)[["mae", "rmse", "r2"]].mean().assign(fold="mean")
    return pd.concat([scores, mean_rows], ignore_index=True)


def _mean_rmse(cv_df: pd.DataFrame) -> float:
    mean = cv_df[cv_df["fold"] == "mean"].copy()
    if mean.empty:
        mean = cv_df.groupby("target", as_index=False)[["rmse"]].mean()
    return float(mean["rmse"].mean())


def tune_lightgbm_params(
    sales_df: pd.DataFrame,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    cv_splits: int = 3,
    n_trials: int = 12,
    random_state: int = 42,
    tuning_window_years: int = 5,
    tracking_uri: str | None = None,
    experiment_name: str = "vinuni-datathon-forecast",
) -> dict[str, Any]:
    """
    Lightweight tuning (no Optuna dependency). Logs every trial to MLflow (nested runs).
    Objective: minimize mean RMSE over Revenue & COGS on walk-forward CV.
    """
    from src.mlflow_tracking import configure_mlflow

    rng = np.random.default_rng(seed=random_state)
    base = default_model_params(random_state=random_state)

    # Speed: tune on the most recent window (keeps seasonality, reduces runtime a lot).
    sales_df = sales_df.copy()
    sales_df["Date"] = pd.to_datetime(sales_df["Date"])
    cutoff = sales_df["Date"].max() - pd.DateOffset(years=int(tuning_window_years))
    sales_tune = sales_df[sales_df["Date"] >= cutoff].copy().reset_index(drop=True)

    search_space = {
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        # Keep tuning reasonably fast; use bigger values only after you find a good region.
        "n_estimators": [300, 600, 900],
        "max_depth": [6, 7, 8, 10],
        "num_leaves": [31, 63, 127],
        "subsample": [0.75, 0.85, 0.95],
        "colsample_bytree": [0.75, 0.85, 0.95],
        "reg_alpha": [0.0, 0.1, 0.2],
        "reg_lambda": [0.8, 1.2, 2.0],
        "min_child_samples": [15, 20, 30],
    }

    def sample_params() -> dict[str, Any]:
        params = dict(base)
        for k, choices in search_space.items():
            params[k] = choices[int(rng.integers(0, len(choices)))]
        return params

    configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    import mlflow

    best_score = np.inf
    best_params: dict[str, Any] = dict(base)
    leaderboard: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="lgbm-tuning") as parent:
        mlflow.log_param("tuning_window_years", int(tuning_window_years))
        for trial in range(1, n_trials + 1):
            params = sample_params()
            print(f"[tuning] trial {trial}/{n_trials} ...", flush=True)
            with mlflow.start_run(run_name=f"trial-{trial:03d}", nested=True):
                for k, v in params.items():
                    mlflow.log_param(k, v)
                cv_df = walk_forward_cv_lightgbm(
                    sales_tune,
                    promotions_df=promotions_df,
                    web_traffic_df=web_traffic_df,
                    inventory_df=inventory_df,
                    n_splits=cv_splits,
                    params=params,
                )
                score = _mean_rmse(cv_df)
                mlflow.log_metric("mean_rmse", score)
                print(f"[tuning] trial {trial} mean_rmse={score:.4f}", flush=True)
                leaderboard.append({"trial": trial, "mean_rmse": score, "params": params})
                if score < best_score:
                    best_score = score
                    best_params = params
                    mlflow.set_tag("best_so_far", "true")

        mlflow.log_metric("best_mean_rmse", float(best_score))
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

    leaderboard_sorted = sorted(leaderboard, key=lambda r: r["mean_rmse"])
    return {
        "best_score": float(best_score),
        "best_params": best_params,
        "leaderboard": leaderboard_sorted,
        "n_trials": int(n_trials),
        "cv_splits": int(cv_splits),
        "mlflow_parent_run_id": parent.info.run_id,
    }


def fit_full_forecaster(
    sales_df: pd.DataFrame,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    use_log_target: bool = True,
    save_artifacts: bool = True,
) -> ForecastBundle:
    candidate_models = train_candidate_models(sales_df)
    candidate_eval = validate_candidate_ensemble(sales_df, candidate_models, valid_start=VALID_START, valid_end=VALID_END)
    ensemble_bundle = build_weighted_ensemble_bundle(
        sales_df,
        candidate_models=candidate_models,
        revenue_weights=candidate_eval["revenue_weights"],
        cogs_weights=candidate_eval["cogs_weights"],
    )
    if save_artifacts:
        save_model(ensemble_bundle)
        lgbm_bundle, lgbm_train_df = train_lgbm_models(
            sales_df,
            promotions_df=promotions_df,
            web_traffic_df=web_traffic_df,
            inventory_df=inventory_df,
            params=params,
            use_log_target=use_log_target,
        )
        save_shap_explainer(lgbm_bundle, lgbm_train_df[lgbm_bundle.features])
    return ensemble_bundle


def predict_submission(
    bundle: ForecastBundle,
    sales_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    sample = sample_submission_df.copy()
    sample["Date"] = pd.to_datetime(sample["Date"])
    original_order = sample["Date"].copy()
    forecast = recursive_forecast(bundle, sales_df, sample["Date"], promotions_df, web_traffic_df, inventory_df)
    submission = pd.DataFrame({"Date": original_order.values}).merge(forecast, on="Date", how="left")
    submission["Date"] = pd.to_datetime(submission["Date"]).dt.strftime("%Y-%m-%d")
    if output_path is not None:
        submission.to_csv(output_path, index=False)
    return submission


def save_model(bundle: ForecastBundle, path: Path = MODEL_PATH) -> Path:
    payload = {
        "model_type": bundle.model_type,
        "revenue_model": bundle.revenue_model,
        "cogs_model": bundle.cogs_model,
        "features": bundle.features,
        "static_features": bundle.static_features,
        "params": bundle.params,
        "use_log_target": bundle.use_log_target,
        "metadata": bundle.metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def load_model(path: Path = MODEL_PATH) -> ForecastBundle:
    payload = joblib.load(path)
    return ForecastBundle(
        model_type=payload["model_type"],
        revenue_model=payload["revenue_model"],
        cogs_model=payload.get("cogs_model"),
        features=list(payload.get("features", [])),
        static_features=list(payload.get("static_features", [])),
        params=dict(payload.get("params", {})),
        use_log_target=bool(payload.get("use_log_target", True)),
        metadata=payload.get("metadata"),
    )


def save_shap_explainer(bundle: ForecastBundle, reference_frame: pd.DataFrame, path: Path = SHAP_EXPLAINER_PATH) -> Path:
    import shap

    reference = reference_frame[bundle.features].copy()
    explainer_payload = {
        "revenue": shap.TreeExplainer(bundle.revenue_model),
        "cogs": shap.TreeExplainer(bundle.cogs_model),
        "reference_columns": bundle.features,
        "reference_sample": reference.head(200),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer_payload, path)
    return path


def export_explainability_figures(
    lgbm_bundle: ForecastBundle,
    reference_frame: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR / "figures",
) -> dict[str, Path]:
    """
    Export report-ready explainability figures:
    - LightGBM feature importance (Revenue / COGS)
    - SHAP summary bar plots (Revenue / COGS) on a small reference sample
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    X_ref = reference_frame[lgbm_bundle.features].copy()
    X_small = X_ref.head(SHAP_REFERENCE_SAMPLE_SIZE)

    paths: dict[str, Path] = {}

    # LightGBM feature importances
    def _save_importance(model: Any, title: str, filename: str) -> Path:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise ValueError("Model does not expose feature_importances_.")
        df_imp = (
            pd.DataFrame({"feature": lgbm_bundle.features, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(30)
        )
        fig = plt.figure(figsize=(10, 7))
        plt.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color="#2563eb")
        plt.title(title)
        plt.xlabel("Importance")
        plt.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    paths["lgbm_feature_importance_revenue_png"] = _save_importance(
        lgbm_bundle.revenue_model,
        "LightGBM Feature Importance â€” Revenue",
        "lgbm_feature_importance_revenue.png",
    )
    paths["lgbm_feature_importance_cogs_png"] = _save_importance(
        lgbm_bundle.cogs_model,
        "LightGBM Feature Importance â€” COGS",
        "lgbm_feature_importance_cogs.png",
    )

    # SHAP summaries
    import shap

    rev_explainer = shap.TreeExplainer(lgbm_bundle.revenue_model)
    cogs_explainer = shap.TreeExplainer(lgbm_bundle.cogs_model)
    shap_rev = rev_explainer.shap_values(X_small)
    shap_cogs = cogs_explainer.shap_values(X_small)

    def _save_shap_bar(shap_values: Any, X: pd.DataFrame, title: str, filename: str) -> Path:
        fig = plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=25)
        plt.title(title)
        plt.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _save_shap_beeswarm(shap_values: Any, X: pd.DataFrame, title: str, filename: str) -> Path:
        fig = plt.figure(figsize=(8, 10))
        shap.summary_plot(shap_values, X, show=False, max_display=20)
        plt.title(title)
        plt.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return out_path

    paths["shap_summary_bar_revenue_png"] = _save_shap_bar(
        shap_rev,
        X_small,
        "SHAP Summary (bar) - Revenue",
        "shap_summary_bar_revenue.png",
    )
    paths["shap_summary_bar_cogs_png"] = _save_shap_bar(
        shap_cogs,
        X_small,
        "SHAP Summary (bar) - COGS",
        "shap_summary_bar_cogs.png",
    )
    paths["shap_beeswarm_revenue_png"] = _save_shap_beeswarm(
        shap_rev,
        X_small,
        "Revenue Fast SHAP on Validation-like Sample",
        "shap_beeswarm_revenue.png",
    )
    paths["shap_beeswarm_cogs_png"] = _save_shap_beeswarm(
        shap_cogs,
        X_small,
        "COGS Fast SHAP on Validation-like Sample",
        "shap_beeswarm_cogs.png",
    )
    return paths


def load_shap_explainer(path: Path = SHAP_EXPLAINER_PATH) -> dict[str, Any]:
    return joblib.load(path)


def train_and_export(
    data_dir: Path = DATA_DIR,
    model_path: Path = MODEL_PATH,
    shap_path: Path = SHAP_EXPLAINER_PATH,
    submission_path: Path = SUBMISSION_PATH,
    cv_splits: int = 3,
    tune_lgbm: bool = True,
    tuning_trials: int = 10,
    enable_mlflow: bool = True,
    tracking_uri: str | None = None,
    experiment_name: str = "vinuni-datathon-forecast",
    tuning_window_years: int = 5,
) -> dict[str, Any]:
    sales, sample_submission, promotions, web_traffic, inventory = load_forecast_inputs(data_dir)
    candidate_models = train_candidate_models(sales)
    candidate_eval = validate_candidate_ensemble(sales, candidate_models, valid_start=VALID_START, valid_end=VALID_END)
    ensemble_bundle = build_weighted_ensemble_bundle(
        sales,
        candidate_models=candidate_models,
        revenue_weights=candidate_eval["revenue_weights"],
        cogs_weights=candidate_eval["cogs_weights"],
    )
    tuning_result = None
    tuned_params = None
    if tune_lgbm:
        tuning_result = tune_lightgbm_params(
            sales,
            promotions_df=promotions,
            web_traffic_df=web_traffic,
            inventory_df=inventory,
            cv_splits=int(cv_splits),
            n_trials=int(tuning_trials),
            tuning_window_years=int(tuning_window_years),
            tracking_uri=tracking_uri if enable_mlflow else None,
            experiment_name=experiment_name,
        )
        tuned_params = tuning_result["best_params"]

    # Leakage-safe CV for tuned LGBM (technical report)
    lgbm_cv_scores = walk_forward_cv_lightgbm(
        sales,
        promotions_df=promotions,
        web_traffic_df=web_traffic,
        inventory_df=inventory,
        n_splits=int(cv_splits),
        params=tuned_params,
    )
    submission = predict_submission(
        ensemble_bundle,
        sales_df=sales,
        sample_submission_df=sample_submission,
        promotions_df=promotions,
        web_traffic_df=web_traffic,
        inventory_df=inventory,
        output_path=submission_path,
    )
    sanity = sanity_check_predictions(sales, pd.read_csv(submission_path))
    save_model(ensemble_bundle, model_path)
    lgbm_bundle, lgbm_train_df = train_lgbm_models(
        sales,
        promotions_df=promotions,
        web_traffic_df=web_traffic,
        inventory_df=inventory,
        params=tuned_params,
    )
    save_shap_explainer(lgbm_bundle, lgbm_train_df[lgbm_bundle.features], shap_path)
    explain_paths = export_explainability_figures(lgbm_bundle, lgbm_train_df, output_dir=OUTPUT_DIR / "figures")

    # Baselines for comparison
    xgb_bundle, _ = train_xgb_models(
        sales,
        promotions_df=promotions,
        web_traffic_df=web_traffic,
        inventory_df=inventory,
    )
    en_bundle, _ = train_elasticnet_models(
        sales,
        promotions_df=promotions,
        web_traffic_df=web_traffic,
        inventory_df=inventory,
    )

    candidate_metrics_path = OUTPUT_DIR / "forecast_candidate_metrics.csv"
    lgbm_cv_scores_path = OUTPUT_DIR / "forecast_lgbm_walkforward_cv.csv"
    val_metrics_path = OUTPUT_DIR / "forecast_validation_metrics.csv"
    candidate_eval["candidate_metrics"].to_csv(candidate_metrics_path, index=False)
    lgbm_cv_scores.to_csv(lgbm_cv_scores_path, index=False)
    candidate_eval["validation_metrics"].to_csv(val_metrics_path, index=False)

    # Simple comparison table on the official validation window
    valid_df = sales[(sales["Date"] >= VALID_START) & (sales["Date"] <= VALID_END)].copy()
    train_internal = sales[sales["Date"] < VALID_START].copy()
    comp_rows = []
    for name, bundle in [
        ("weighted_ensemble", ensemble_bundle),
        ("lgbm_tuned_recursive", lgbm_bundle),
        ("xgb_recursive", xgb_bundle),
        ("elasticnet_recursive", en_bundle),
    ]:
        pred = recursive_forecast(
            bundle,
            history_sales_df=train_internal,
            future_dates=valid_df["Date"],
            promotions_df=promotions,
            web_traffic_df=web_traffic,
            inventory_df=inventory,
        )
        metrics = evaluate_targets(
            valid_df[["Date", "Revenue", "COGS"]].reset_index(drop=True),
            pred.reset_index(drop=True),
        )
        for _, row in metrics.iterrows():
            comp_rows.append({"model": name, "target": row["target"], "mae": row["mae"], "rmse": row["rmse"], "r2": row["r2"]})
    comparison_df = pd.DataFrame(comp_rows)
    comparison_path = OUTPUT_DIR / "forecast_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    result = {
        "selected_model": "weighted_ensemble",
        "bundle": ensemble_bundle,
        "lgbm_walkforward_cv": lgbm_cv_scores,
        "validation_forecast": candidate_eval["validation_forecast"],
        "validation_metrics": candidate_eval["validation_metrics"],
        "submission": submission,
        "model_path": model_path,
        "shap_path": shap_path,
        "submission_path": submission_path,
        "candidate_metrics": candidate_eval["candidate_metrics"],
        "candidate_weights": {
            "revenue": candidate_eval["revenue_weights"],
            "cogs": candidate_eval["cogs_weights"],
        },
        "sanity": sanity,
        "tuning": tuning_result,
        "model_comparison": comparison_df,
        "artifacts": {
            "candidate_metrics_csv": candidate_metrics_path,
            "lgbm_walkforward_cv_csv": lgbm_cv_scores_path,
            "validation_metrics_csv": val_metrics_path,
            "model_comparison_csv": comparison_path,
            **{k: v for k, v in explain_paths.items()},
        },
    }
    if enable_mlflow:
        from src.mlflow_tracking import log_forecast_run

        run_id = log_forecast_run(
            result,
            run_name="forecast-train-export",
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
        )
        result["mlflow_run_id"] = run_id
    return result

