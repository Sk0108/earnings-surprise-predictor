from __future__ import annotations

import html

import pandas as pd
import streamlit as st


def metric_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-caption">{html.escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_badge(probability: float, threshold: float = 0.60) -> str:
    label = "Beat likely" if probability >= threshold else "Miss risk"
    class_name = "beat" if probability >= threshold else "miss"
    return f'<span class="status-pill {class_name}">{label} · {probability:.0%}</span>'


def format_feature_name(name: str) -> str:
    return name.replace("_", " ").title().replace("Eps", "EPS").replace("Yoy", "YoY").replace("Pe", "P/E")


def top_events_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "company",
        "sector",
        "fiscal_quarter",
        "earnings_date",
        "reported_eps",
        "estimated_eps",
        "surprise_pct",
        "predicted_probability",
    ]
    visible = frame[columns].copy()
    visible["earnings_date"] = pd.to_datetime(visible["earnings_date"]).dt.date
    visible["surprise_pct"] = visible["surprise_pct"].map(lambda value: f"{value:.1%}")
    visible["predicted_probability"] = visible["predicted_probability"].map(lambda value: f"{value:.1%}")
    return visible.rename(
        columns={
            "ticker": "Ticker",
            "company": "Company",
            "sector": "Sector",
            "fiscal_quarter": "Quarter",
            "earnings_date": "Date",
            "reported_eps": "Reported EPS",
            "estimated_eps": "Estimate",
            "surprise_pct": "Surprise",
            "predicted_probability": "Beat Probability",
        }
    )
