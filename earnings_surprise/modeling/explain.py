from __future__ import annotations

import json

import numpy as np
import pandas as pd

from earnings_surprise.config import ARTIFACT_DIR, MODEL_FEATURES


def export_shap_values(model, data: pd.DataFrame, output_path: str | None = None) -> pd.DataFrame:
    """Create SHAP values when available, otherwise use importances as a fallback."""

    features = data[MODEL_FEATURES]
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(features)
        if isinstance(values, list):
            values = values[1]
        shap_frame = pd.DataFrame(values, columns=MODEL_FEATURES)
    except Exception:
        importances = getattr(model, "feature_importances_", np.ones(len(MODEL_FEATURES)))
        centered = features - features.mean()
        shap_frame = centered * importances

    target = output_path or ARTIFACT_DIR / "shap_values.csv"
    shap_frame.to_csv(target, index=False)
    return shap_frame


def summarize_shap(shap_values: pd.DataFrame, output_path: str | None = None) -> pd.DataFrame:
    summary = (
        shap_values.abs()
        .mean()
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("mean_abs_shap", ascending=False)
    )
    target = output_path or ARTIFACT_DIR / "shap_summary.csv"
    summary.to_csv(target, index=False)
    return summary


def write_metrics(metrics: dict[str, float], path: str | None = None) -> None:
    target = path or ARTIFACT_DIR / "metrics.json"
    with open(target, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
