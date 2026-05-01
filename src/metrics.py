from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(r2_score(y_true, y_pred))


def evaluate_forecast(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    finite_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not finite_mask.any():
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    y_true_arr = y_true_arr[finite_mask]
    y_pred_arr = y_pred_arr[finite_mask]
    return {
        "mae": mae(y_true_arr, y_pred_arr),
        "rmse": rmse(y_true_arr, y_pred_arr),
        "r2": r2(y_true_arr, y_pred_arr),
    }


def evaluate_targets(
    actual_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    actual_cols: tuple[str, str] = ("Revenue", "COGS"),
    pred_cols: tuple[str, str] = ("Revenue", "COGS"),
) -> pd.DataFrame:
    rev_metrics = evaluate_forecast(actual_df[actual_cols[0]], pred_df[pred_cols[0]])
    cogs_metrics = evaluate_forecast(actual_df[actual_cols[1]], pred_df[pred_cols[1]])
    return pd.DataFrame(
        [
            {"target": "Revenue", **rev_metrics},
            {"target": "COGS", **cogs_metrics},
        ]
    )
