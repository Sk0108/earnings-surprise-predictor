from __future__ import annotations

import pandas as pd


def add_tabular_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["eps_trend_4q"] = data.groupby("ticker")["reported_eps"].transform(lambda series: series.diff(4))
    data["eps_trend_4q"] = data["eps_trend_4q"].fillna(data.groupby("ticker")["reported_eps"].transform("mean"))
    data["surprise_pct"] = (data["reported_eps"] - data["estimated_eps"]) / data["estimated_eps"].abs().clip(lower=0.05)
    data["beat"] = (data["reported_eps"] > data["estimated_eps"]).astype(int)
    numeric_columns = data.select_dtypes(include=["number"]).columns
    data[numeric_columns] = data[numeric_columns].replace([float("inf"), float("-inf")], 0).fillna(0)
    return data
