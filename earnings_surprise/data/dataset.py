from __future__ import annotations

import pandas as pd

from earnings_surprise.config import MODEL_FEATURES, PROCESSED_DIR
from earnings_surprise.features.tabular import add_tabular_features


def load_training_frame(path: str | None = None) -> pd.DataFrame:
    source = path or PROCESSED_DIR / "earnings_surprise_dataset.csv"
    data = pd.read_csv(source, parse_dates=["earnings_date"])
    data = add_tabular_features(data)
    missing = [column for column in MODEL_FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {missing}")
    return data


def time_split(data: pd.DataFrame, test_year: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = data.sort_values("earnings_date").copy()
    if test_year is not None:
        test_mask = ordered["earnings_date"].dt.year == test_year
    else:
        cutoff = ordered["earnings_date"].quantile(0.8)
        test_mask = ordered["earnings_date"] >= cutoff
    train = ordered.loc[~test_mask].copy()
    test = ordered.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set.")
    return train, test
