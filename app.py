from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from earnings_surprise.config import ARTIFACT_DIR, MODEL_FEATURES, OPERATING_THRESHOLD, PROCESSED_DIR
from earnings_surprise.data.dataset import load_training_frame
from earnings_surprise.modeling.backtest import rolling_year_backtest
from earnings_surprise.modeling.explain import export_shap_values, summarize_shap, write_metrics
from earnings_surprise.modeling.train import train_classifier
from earnings_surprise.ui.components import format_feature_name, metric_card, probability_badge, top_events_table
from earnings_surprise.ui.theme import apply_theme
from scripts.generate_demo_data import main as generate_demo_data


def load_artifacts() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_path = PROCESSED_DIR / "earnings_surprise_dataset.csv"
    model_path = ARTIFACT_DIR / "earnings_surprise_model.joblib"
    if not dataset_path.exists() or not model_path.exists():
        with st.spinner("Preparing demo dataset and model artifacts for this deployment..."):
            create_demo_artifacts()

    data = load_training_frame(str(dataset_path))
    bundle = joblib.load(model_path)
    data["predicted_probability"] = bundle["model"].predict_proba(data[MODEL_FEATURES])[:, 1]
    metrics = read_json(ARTIFACT_DIR / "metrics.json")
    shap_summary = read_csv_or_empty(ARTIFACT_DIR / "shap_summary.csv")
    backtest = read_csv_or_empty(ARTIFACT_DIR / "backtest_metrics.csv")
    predictions = read_csv_or_empty(ARTIFACT_DIR / "holdout_predictions.csv")
    return data, metrics, shap_summary, backtest, predictions


def create_demo_artifacts() -> None:
    generate_demo_data()
    data = load_training_frame()
    train, test = time_split_for_app(data)
    result = train_classifier(train, test)
    bundle = joblib.load(result.model_path)
    probabilities = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]

    shap_values = export_shap_values(bundle["model"], test)
    summarize_shap(shap_values)
    result.feature_importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    test.assign(predicted_probability=probabilities).to_csv(ARTIFACT_DIR / "holdout_predictions.csv", index=False)
    rolling_year_backtest(data).to_csv(ARTIFACT_DIR / "backtest_metrics.csv", index=False)
    write_metrics(result.metrics)


def time_split_for_app(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from earnings_surprise.data.dataset import time_split

    try:
        return time_split(data, test_year=2026)
    except ValueError:
        return time_split(data)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def sidebar_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Controls")
    sectors = st.sidebar.multiselect("Sector", sorted(data["sector"].unique()), default=sorted(data["sector"].unique()))
    tickers = st.sidebar.multiselect("Ticker", sorted(data["ticker"].unique()), default=sorted(data["ticker"].unique())[:8])
    min_probability = st.sidebar.slider("Minimum beat probability", 0.0, 1.0, 0.0, 0.05)
    date_range = st.sidebar.date_input(
        "Earnings window",
        value=(data["earnings_date"].min().date(), data["earnings_date"].max().date()),
    )

    filtered = data[data["sector"].isin(sectors) & data["ticker"].isin(tickers)]
    filtered = filtered[filtered["predicted_probability"] >= min_probability]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered["earnings_date"] >= start) & (filtered["earnings_date"] <= end)]
    return filtered


def main() -> None:
    apply_theme()
    data, metrics, shap_summary, backtest, predictions = load_artifacts()
    filtered = sidebar_filters(data)

    st.title("Earnings Surprise Predictor")
    st.markdown(
        '<div class="subtle">NLP sentiment from management commentary combined with point-in-time market and fundamental signals.</div>',
        unsafe_allow_html=True,
    )

    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)
    roc_auc = metrics.get("roc_auc", 0)
    coverage = len(filtered)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Holdout Precision", f"{precision:.0%}", f"At {OPERATING_THRESHOLD:.0%} operating threshold")
    with col2:
        metric_card("Holdout Recall", f"{recall:.0%}", "Actual beats captured by the classifier")
    with col3:
        metric_card("ROC AUC", f"{roc_auc:.2f}", "Ranking quality across thresholds")
    with col4:
        metric_card("Filtered Events", f"{coverage:,}", "Earnings observations in current view")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Probability Distribution")
        fig = px.histogram(
            filtered,
            x="predicted_probability",
            color="beat",
            nbins=24,
            color_discrete_map={0: "#b42318", 1: "#12805c"},
            labels={"predicted_probability": "Predicted beat probability", "beat": "Actual beat"},
            template="plotly_white",
        )
        fig.update_layout(bargap=0.08, legend_title_text="Outcome", height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Backtest Quality")
        if not backtest.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=backtest["year"], y=backtest["precision"], mode="lines+markers", name="Precision", line=dict(color="#2764d8")))
            fig.add_trace(go.Scatter(x=backtest["year"], y=backtest["recall"], mode="lines+markers", name="Recall", line=dict(color="#12805c")))
            fig.update_layout(template="plotly_white", yaxis_tickformat=".0%", height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Backtest metrics will appear after training.")

    st.subheader("Explainability")
    exp_left, exp_right = st.columns([0.95, 1.05])
    with exp_left:
        if not shap_summary.empty:
            plot_data = shap_summary.head(10).copy()
            plot_data["Feature"] = plot_data["feature"].map(format_feature_name)
            fig = px.bar(
                plot_data.sort_values("mean_abs_shap"),
                x="mean_abs_shap",
                y="Feature",
                orientation="h",
                labels={"mean_abs_shap": "Mean absolute SHAP"},
                template="plotly_white",
                color="mean_abs_shap",
                color_continuous_scale=["#d6e4ff", "#2764d8"],
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with exp_right:
        selected_ticker = st.selectbox("Inspect company", sorted(filtered["ticker"].unique()) if not filtered.empty else sorted(data["ticker"].unique()))
        company_events = data[data["ticker"] == selected_ticker].sort_values("earnings_date")
        latest = company_events.iloc[-1]
        st.markdown(probability_badge(float(latest["predicted_probability"]), OPERATING_THRESHOLD), unsafe_allow_html=True)
        st.write("")
        feature_snapshot = latest[MODEL_FEATURES].rename(index=format_feature_name).reset_index()
        feature_snapshot.columns = ["Feature", "Value"]
        st.dataframe(feature_snapshot, hide_index=True, use_container_width=True, height=330)

    st.subheader("Earnings Event Monitor")
    table_data = filtered.sort_values("predicted_probability", ascending=False)
    st.dataframe(top_events_table(table_data), hide_index=True, use_container_width=True, height=420)

    with st.expander("Model And Data Card", expanded=False):
        st.markdown(
            """
            **Target:** binary EPS beat where reported EPS exceeds consensus estimate.

            **Feature families:** FinBERT sentiment, uncertainty/litigious tone, EPS trend, revenue growth, margins, valuation, leverage, price momentum, volatility, and estimate revision.

            **Backtest:** rolling year evaluation to reduce look-ahead bias. For production usage, replace demo fundamentals with point-in-time vendor data.

            **Operating threshold:** model probabilities are converted into beat calls at 60% to favor precision.
            """
        )
        if not predictions.empty:
            st.download_button(
                "Download holdout predictions",
                data=predictions.to_csv(index=False),
                file_name="holdout_predictions.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
