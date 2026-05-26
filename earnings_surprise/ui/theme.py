from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.set_page_config(
        page_title="Earnings Surprise Predictor",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f8fa;
            --panel: #ffffff;
            --ink: #17202a;
            --muted: #647080;
            --line: #d9dee7;
            --green: #12805c;
            --red: #b42318;
            --blue: #2764d8;
            --amber: #b7791f;
        }
        .stApp {
            background: var(--bg);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
            color: var(--ink);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--ink);
        }
        [data-baseweb="select"] > div,
        [data-testid="stDateInput"] input {
            background: #ffffff;
            border-color: var(--line);
            color: var(--ink);
        }
        [data-baseweb="tag"] {
            background: #eef4ff;
        }
        [data-baseweb="tag"] span {
            color: #174ea6 !important;
        }
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin-bottom: 0.2rem;
        }
        .subtle {
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            min-height: 118px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.95rem;
            font-weight: 760;
            line-height: 1.1;
        }
        .metric-caption {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 8px;
        }
        .section-shell {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.76rem;
            font-weight: 700;
            border: 1px solid var(--line);
        }
        .beat {
            color: var(--green);
            background: #ecfdf3;
        }
        .miss {
            color: var(--red);
            background: #fff1f0;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
