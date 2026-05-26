# Project: Earnings Surprise Predictor

Predict whether a public company will beat or miss analyst EPS estimates by combining NLP signals from filings/transcripts with market and fundamental features.

## What This Project Includes

- SEC EDGAR filing ingestion scaffolding for 10-Q and 10-K reports
- Earnings call transcript ingestion interface with a local/demo provider
- FinBERT sentiment feature extraction for management commentary
- yfinance-based fundamentals and price features
- XGBoost/LightGBM-compatible binary classifier pipeline
- SHAP explainability exports
- S&P 500-style three-year backtest workflow
- Professional Streamlit dashboard for model monitoring and explainability

The repository is designed to run in two modes:

- **Demo mode:** Uses generated sample data so the dashboard and model flow work immediately.
- **Live mode:** Uses SEC EDGAR, yfinance, and optional HuggingFace FinBERT dependencies to collect real data.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/generate_demo_data.py
python scripts/train_model.py --demo
streamlit run app.py
```

For VS Code setup, see:

```text
docs/VS_CODE_RUN_GUIDE.md
```

For the full technical explanation, metrics, terminology, and architecture, see:

```text
docs/TECHNICAL_OVERVIEW.md
```

On Windows, use the virtual environment Python explicitly if PowerShell is not activated:

```powershell
.\.venv\Scripts\python.exe scripts\generate_demo_data.py
.\.venv\Scripts\python.exe scripts\train_model.py --demo
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Project Layout

```text
earnings_surprise/
  config.py              Project paths and settings
  data/
    edgar.py             SEC EDGAR client and filing parsing helpers
    fundamentals.py      yfinance feature loader
    transcripts.py       Transcript provider interface and demo provider
    dataset.py           Dataset assembly and target creation
  features/
    nlp.py               FinBERT sentiment and fallback NLP features
    tabular.py           Tabular feature engineering
  modeling/
    train.py             Model training and evaluation
    explain.py           SHAP explainability utilities
    backtest.py          Time-based backtest
  ui/
    theme.py             Streamlit styling
    components.py        Reusable dashboard components
app.py                   Streamlit dashboard
scripts/
  generate_demo_data.py  Creates realistic demo backtest data
  train_model.py         Trains classifier and exports artifacts
```

## Real Data Notes

SEC EDGAR requires a responsible `User-Agent` that includes contact information. Set:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
```

For FinBERT features, install the HuggingFace dependencies from `requirements-ml.txt` if they are not already installed. Without these packages, the project uses the included lexical NLP fallback so the full pipeline still runs.

```powershell
python -m pip install -r requirements-ml.txt
```
