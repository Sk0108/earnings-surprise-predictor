from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, ARTIFACT_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)


SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "EarningsSurprisePredictor/0.1 contact@example.com")
RANDOM_STATE = 42
TARGET_COLUMN = "beat"
OPERATING_THRESHOLD = 0.60
MODEL_FEATURES = [
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
    "tone_uncertainty",
    "tone_litigious",
    "eps_trend_4q",
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "pe_ratio",
    "debt_to_equity",
    "price_momentum_63d",
    "volatility_63d",
    "estimate_revision_30d",
]
