from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src import OUTPUT_DIR, ROOT_DIR


DEFAULT_TRACKING_DIR = ROOT_DIR / "mlruns"


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = "vinuni-datathon-forecast",
) -> None:
    uri = tracking_uri or DEFAULT_TRACKING_DIR.as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def _log_metrics_table(prefix: str, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        target = str(row.get("target", "all")).lower()
        if "mae" in row:
            mlflow.log_metric(f"{prefix}_{target}_mae", float(row["mae"]))
        if "rmse" in row:
            mlflow.log_metric(f"{prefix}_{target}_rmse", float(row["rmse"]))
        if "r2" in row:
            mlflow.log_metric(f"{prefix}_{target}_r2", float(row["r2"]))


def log_forecast_run(
    result: dict[str, Any],
    run_name: str = "forecast-train",
    tracking_uri: str | None = None,
    experiment_name: str = "vinuni-datathon-forecast",
) -> str:
    configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("selected_model", result["selected_model"])
        bundle = result["bundle"]
        mlflow.log_param("bundle_model_type", bundle.model_type)
        if bundle.metadata and "candidate_names" in bundle.metadata:
            mlflow.log_param("candidate_count", len(bundle.metadata["candidate_names"]))

        _log_metrics_table("validation", result["validation_metrics"])
        cv_df = None
        if "cv_scores" in result:
            cv_df = result.get("cv_scores")
        elif "lgbm_walkforward_cv" in result:
            cv_df = result.get("lgbm_walkforward_cv")
        if isinstance(cv_df, pd.DataFrame) and not cv_df.empty:
            mean_rows = cv_df[cv_df["fold"] == "mean"] if "fold" in cv_df.columns else cv_df
            _log_metrics_table("cv_mean", mean_rows)

        for target, weights in result.get("candidate_weights", {}).items():
            for candidate_name, weight in weights.items():
                mlflow.log_param(f"{target}_weight_{candidate_name}", weight)

        tuning = result.get("tuning")
        if isinstance(tuning, dict):
            if "best_score" in tuning:
                mlflow.log_metric("tuning_best_score", float(tuning["best_score"]))
            if "n_trials" in tuning:
                mlflow.log_param("tuning_n_trials", int(tuning["n_trials"]))
            if "best_params" in tuning and isinstance(tuning["best_params"], dict):
                for k, v in tuning["best_params"].items():
                    mlflow.log_param(f"tuning_best_{k}", v)

        for key, value in result.get("sanity", {}).items():
            mlflow.log_metric(key, float(value))

        payload_path = OUTPUT_DIR / "forecast_bundle_metadata.json"
        payload = {
            "selected_model": result["selected_model"],
            "candidate_weights": result.get("candidate_weights", {}),
            "sanity": result.get("sanity", {}),
        }
        payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        artifact_paths = [
            Path(result["model_path"]),
            Path(result["shap_path"]),
            Path(result["submission_path"]),
            payload_path,
        ]
        for _, path in result.get("artifacts", {}).items():
            artifact_paths.append(Path(path))

        for artifact in artifact_paths:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

        candidate_metrics = result.get("candidate_metrics")
        if isinstance(candidate_metrics, pd.DataFrame) and not candidate_metrics.empty:
            candidate_artifact = OUTPUT_DIR / "forecast_candidate_metrics_logged.csv"
            candidate_metrics.to_csv(candidate_artifact, index=False)
            mlflow.log_artifact(str(candidate_artifact))

        return run.info.run_id
