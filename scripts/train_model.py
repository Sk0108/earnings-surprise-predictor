from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib

from earnings_surprise.config import ARTIFACT_DIR
from earnings_surprise.data.dataset import load_training_frame, time_split
from earnings_surprise.modeling.backtest import rolling_year_backtest
from earnings_surprise.modeling.explain import export_shap_values, summarize_shap, write_metrics
from earnings_surprise.modeling.train import train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train earnings surprise classifier.")
    parser.add_argument("--dataset", default=None, help="Path to training CSV.")
    parser.add_argument("--test-year", type=int, default=2026, help="Holdout year for evaluation.")
    parser.add_argument("--demo", action="store_true", help="Use generated demo dataset defaults.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_training_frame(args.dataset)
    train, test = time_split(data, test_year=args.test_year)
    result = train_classifier(train, test)

    bundle = joblib.load(result.model_path)
    shap_values = export_shap_values(bundle["model"], test)
    shap_summary = summarize_shap(shap_values)
    backtest = rolling_year_backtest(data)

    result.feature_importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    test.assign(predicted_probability=bundle["model"].predict_proba(test[bundle["features"]])[:, 1]).to_csv(
        ARTIFACT_DIR / "holdout_predictions.csv", index=False
    )
    backtest.to_csv(ARTIFACT_DIR / "backtest_metrics.csv", index=False)
    write_metrics(result.metrics)
    with open(ARTIFACT_DIR / "classification_report.json", "w", encoding="utf-8") as file:
        json.dump(result.classification_report, file, indent=2)

    print("Training complete")
    print(json.dumps(result.metrics, indent=2))
    print("Top SHAP drivers:")
    print(shap_summary.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
