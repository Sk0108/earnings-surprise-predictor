from __future__ import annotations

import pandas as pd
from sklearn.metrics import precision_score, recall_score

from earnings_surprise.config import MODEL_FEATURES, OPERATING_THRESHOLD, TARGET_COLUMN
from earnings_surprise.modeling.train import build_model, predict_proba


def rolling_year_backtest(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    frame = data.sort_values("earnings_date").copy()
    years = sorted(frame["earnings_date"].dt.year.unique())

    for year in years[1:]:
        train = frame[frame["earnings_date"].dt.year < year]
        test = frame[frame["earnings_date"].dt.year == year]
        if len(train) < 20 or test.empty:
            continue
        model = build_model()
        model.fit(train[MODEL_FEATURES], train[TARGET_COLUMN])
        probabilities = predict_proba(model, test[MODEL_FEATURES])
        predictions = (probabilities >= OPERATING_THRESHOLD).astype(int)
        rows.append(
            {
                "year": int(year),
                "events": int(len(test)),
                "precision": float(precision_score(test[TARGET_COLUMN], predictions, zero_division=0)),
                "recall": float(recall_score(test[TARGET_COLUMN], predictions, zero_division=0)),
                "avg_predicted_beat_probability": float(probabilities.mean()),
                "actual_beat_rate": float(test[TARGET_COLUMN].mean()),
            }
        )
    return pd.DataFrame(rows)
