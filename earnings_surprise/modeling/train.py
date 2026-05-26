from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score

from earnings_surprise.config import ARTIFACT_DIR, MODEL_FEATURES, OPERATING_THRESHOLD, RANDOM_STATE, TARGET_COLUMN


@dataclass(frozen=True)
class TrainingResult:
    model_path: str
    metrics: dict[str, float]
    classification_report: dict
    feature_importance: pd.DataFrame


def build_model():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=260,
            max_depth=3,
            learning_rate=0.045,
            subsample=0.88,
            colsample_bytree=0.86,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, random_state=RANDOM_STATE)


def train_classifier(train: pd.DataFrame, test: pd.DataFrame) -> TrainingResult:
    model = build_model()
    x_train = train[MODEL_FEATURES]
    y_train = train[TARGET_COLUMN]
    x_test = test[MODEL_FEATURES]
    y_test = test[TARGET_COLUMN]

    model.fit(x_train, y_train)
    probabilities = predict_proba(model, x_test)
    predictions = (probabilities >= OPERATING_THRESHOLD).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if len(set(y_test)) > 1 else 0.0,
    }
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    importance = feature_importance_frame(model)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "earnings_surprise_model.joblib"
    joblib.dump({"model": model, "features": MODEL_FEATURES, "metrics": metrics, "threshold": OPERATING_THRESHOLD}, model_path)
    return TrainingResult(str(model_path), metrics, report, importance)


def predict_proba(model, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    scores = model.decision_function(features)
    return 1 / (1 + np.exp(-scores))


def feature_importance_frame(model) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.zeros(len(MODEL_FEATURES))
    return (
        pd.DataFrame({"feature": MODEL_FEATURES, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
