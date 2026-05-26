# Technical Overview

## Project Goal

This project predicts whether a company will **beat** or **miss** analyst EPS estimates.

In simple terms:

- Analysts estimate a company's earnings per share before results are announced.
- The company reports actual EPS.
- If actual EPS is higher than estimated EPS, the company **beats**.
- If actual EPS is lower than or equal to estimated EPS, the company **misses**.

The project combines:

- Text signals from management commentary
- Financial fundamentals
- Market momentum and volatility
- Machine learning classification
- SHAP explainability
- A Streamlit analytics dashboard

## Technology Stack

### Python

Python is the main programming language. It is used for:

- Data generation and loading
- Feature engineering
- Machine learning
- Model evaluation
- Dashboard backend logic

Main files:

- `scripts/generate_demo_data.py`
- `scripts/train_model.py`
- `app.py`
- `earnings_surprise/`

### pandas

pandas is used for tabular data handling.

It helps with:

- Reading CSV files
- Cleaning data
- Creating new columns
- Sorting by date
- Grouping by ticker
- Exporting model artifacts

Example project use:

- Loading `data/processed/earnings_surprise_dataset.csv`
- Creating `beat`
- Creating `surprise_pct`
- Creating EPS trend features

### NumPy

NumPy is used for numerical operations.

It helps with:

- Random demo data generation
- Mathematical transformations
- Probability-style feature generation
- Vectorized numeric calculations

### scikit-learn

scikit-learn is used for model evaluation metrics and fallback modeling support.

Project use:

- `precision_score`
- `recall_score`
- `accuracy_score`
- `roc_auc_score`
- `classification_report`

### XGBoost

XGBoost is the main model used by the training pipeline.

XGBoost is a gradient boosting library. It builds many decision trees in sequence. Each new tree tries to correct errors made by the previous trees.

Why it is useful here:

- Handles tabular financial data well
- Works with nonlinear relationships
- Performs strongly on structured prediction tasks
- Provides feature importance
- Works with SHAP explainability

Project file:

```text
earnings_surprise/modeling/train.py
```

### LightGBM

LightGBM is included in the stack as an alternative gradient boosting model.

It is useful for:

- Large datasets
- Fast tree training
- Tabular ML problems

The current default implementation uses XGBoost, but LightGBM is included in `requirements.txt` so the project can be extended easily.

### FinBERT / HuggingFace

FinBERT is a BERT-style language model trained for financial sentiment.

It classifies financial text into:

- Positive
- Negative
- Neutral

HuggingFace provides the model loading pipeline.

Current project behavior:

- If `transformers` and `torch` are installed, the project can use FinBERT.
- If they are not installed, the project uses a lexical fallback sentiment extractor.

This makes the project easy to run locally while still being architected for real FinBERT use.

Project file:

```text
earnings_surprise/features/nlp.py
```

### SEC EDGAR API

SEC EDGAR is the official database for public company filings.

The project includes an EDGAR client for:

- Fetching 10-Q filings
- Fetching 10-K filings
- Cleaning filing HTML
- Extracting management discussion sections

Important filings:

- **10-Q:** Quarterly report filed by public companies.
- **10-K:** Annual report filed by public companies.

Project file:

```text
earnings_surprise/data/edgar.py
```

For real SEC calls, set:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
```

### yfinance

yfinance is used to fetch public Yahoo Finance data.

It can provide:

- P/E ratio
- Debt-to-equity
- Margins
- Revenue growth
- Price history

Project file:

```text
earnings_surprise/data/fundamentals.py
```

Important note:

For a real institutional-quality backtest, point-in-time fundamentals are preferred. yfinance is useful for prototyping, but historical data may not always be point-in-time clean.

### SHAP

SHAP explains model predictions.

SHAP stands for **SHapley Additive exPlanations**. It estimates how much each feature contributed to a prediction.

Example:

If the model predicts that a company is likely to beat earnings, SHAP can show whether that came from:

- Positive FinBERT tone
- Strong revenue growth
- Positive estimate revisions
- Low volatility
- Improving EPS trend

Project files:

- `earnings_surprise/modeling/explain.py`
- `artifacts/shap_summary.csv`
- `artifacts/shap_values.csv`

### Streamlit

Streamlit is used for the dashboard.

The dashboard includes:

- Sidebar filters
- KPI cards
- Probability distribution chart
- Backtest chart
- SHAP feature importance chart
- Company inspection table
- Earnings event monitor
- Model and data card

Main file:

```text
app.py
```

Theme files:

- `.streamlit/config.toml`
- `earnings_surprise/ui/theme.py`
- `earnings_surprise/ui/components.py`

### Plotly

Plotly is used for interactive charts in the Streamlit dashboard.

Charts include:

- Predicted probability distribution
- Rolling backtest precision and recall
- SHAP feature driver bar chart

## How The System Works

## Step 1: Collect Text Data

For real data, the system is designed to collect:

- 10-Q filings
- 10-K filings
- Earnings call transcripts

The important text is management commentary because management language often contains signals about:

- Demand strength
- Margin pressure
- Customer budgets
- Execution risk
- Guidance confidence
- Supply chain conditions

In demo mode, synthetic transcript-like text and synthetic financial data are generated so the project runs immediately.

## Step 2: Extract NLP Features

The NLP feature extractor converts text into numeric model features.

Important NLP features:

- `finbert_positive`: how positive the financial text sounds
- `finbert_negative`: how negative the financial text sounds
- `finbert_neutral`: how neutral the text sounds
- `tone_uncertainty`: share of uncertainty-related words
- `tone_litigious`: share of legal/regulatory words

Why this matters:

Management tone can reveal business strength or weakness before the final EPS result.

## Step 3: Add Fundamental Features

Fundamental features describe the company financially.

Included features:

- `eps_trend_4q`: EPS change over the last four quarters
- `revenue_growth_yoy`: year-over-year revenue growth
- `gross_margin`: gross profit divided by revenue
- `operating_margin`: operating income divided by revenue
- `pe_ratio`: price-to-earnings ratio
- `debt_to_equity`: leverage ratio
- `estimate_revision_30d`: change in analyst estimates over the last 30 days

Why this matters:

Companies with improving fundamentals and upward analyst revisions are often more likely to beat expectations.

## Step 4: Add Market Features

Market features describe recent stock behavior.

Included features:

- `price_momentum_63d`: stock return over roughly one trading quarter
- `volatility_63d`: annualized volatility over roughly one trading quarter

Why this matters:

Price momentum can reflect market expectations. Volatility can reflect uncertainty or risk.

## Step 5: Create The Target

The model predicts:

```text
beat = 1 if reported_eps > estimated_eps
beat = 0 otherwise
```

Related term:

```text
surprise_pct = (reported_eps - estimated_eps) / abs(estimated_eps)
```

If `surprise_pct` is positive, the company beat estimates. If it is negative, the company missed.

## Step 6: Train A Binary Classifier

The model is a binary classifier.

Binary classifier means:

- It predicts one of two outcomes.
- Here, the outcomes are `beat` or `miss`.

The classifier outputs a probability:

```text
Predicted beat probability = 0.71
```

That means the model estimates a 71% chance that the company beats EPS expectations.

## Step 7: Apply Operating Threshold

The project uses a 60% operating threshold.

That means:

```text
If predicted probability >= 60%, classify as beat likely.
If predicted probability < 60%, classify as miss risk.
```

Why not always use 50%?

Because for this use case, precision matters. A higher threshold makes the model more selective about calling something a beat.

## Step 8: Backtest

Backtesting means testing the strategy/model on past data as if predictions were made at that time.

The project uses a rolling-year style backtest:

- Train on earlier years
- Test on a later year
- Repeat across years

This is better than random splitting because earnings prediction is time-based. Random splitting can accidentally leak future patterns into training.

## Step 9: Explain Predictions With SHAP

After training, SHAP values explain what drove the model.

Global explanation:

- Which features matter most overall?

Local explanation:

- Which features influenced one company/event prediction?

The dashboard currently shows a global SHAP summary and a company-level feature snapshot.

## Metrics Used

### Precision

Precision answers:

```text
Of all companies the model predicted as beats, how many actually beat?
```

Formula:

```text
Precision = True Positives / (True Positives + False Positives)
```

In this project:

```text
Precision = 71%
```

Interpretation:

When the model called a beat on the holdout set, it was correct about 71% of the time.

### Recall

Recall answers:

```text
Of all companies that actually beat, how many did the model catch?
```

Formula:

```text
Recall = True Positives / (True Positives + False Negatives)
```

In this project:

```text
Recall = 62%
```

Interpretation:

The model captured about 62% of actual beats in the holdout set.

### Accuracy

Accuracy answers:

```text
How many total predictions were correct?
```

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

In this project:

```text
Accuracy = 75%
```

Accuracy is useful, but it can be misleading if the dataset is imbalanced.

### ROC AUC

ROC AUC measures how well the model ranks beats above misses across different thresholds.

ROC means **Receiver Operating Characteristic**.

AUC means **Area Under the Curve**.

Interpretation:

- `0.50`: no better than random ranking
- `0.70`: useful signal
- `0.80+`: strong signal
- `1.00`: perfect ranking

In this project:

```text
ROC AUC = 0.76
```

### Predicted Beat Probability

This is the model's probability estimate that a company will beat EPS estimates.

Example:

```text
NVDA predicted beat probability = 82%
```

This does not mean certainty. It means the model sees features similar to historical beat events.

### SHAP Mean Absolute Value

This measures average feature impact.

Higher value means the feature tends to move predictions more strongly.

Example:

If `finbert_positive` has a high mean absolute SHAP value, management sentiment is a major driver of model predictions.

## Important Terms

### EPS

EPS means **Earnings Per Share**.

It is company profit divided by the number of shares outstanding.

### Analyst EPS Estimate

This is the expected EPS number from Wall Street analysts before earnings are reported.

### Earnings Surprise

The difference between reported EPS and estimated EPS.

Positive surprise:

```text
reported_eps > estimated_eps
```

Negative surprise:

```text
reported_eps <= estimated_eps
```

### Beat

A company beats when reported EPS is higher than analyst estimated EPS.

### Miss

A company misses when reported EPS is lower than or equal to analyst estimated EPS.

### 10-Q

A quarterly SEC filing with financial statements and management discussion.

### 10-K

An annual SEC filing with full-year financials, risks, business overview, and management discussion.

### Management Commentary

The text where company leadership explains business performance, risks, demand, margins, and outlook.

### Sentiment

Sentiment measures whether text is positive, negative, or neutral.

### Tone

Tone is broader than sentiment. It can include uncertainty, legal risk, confidence, caution, or optimism.

### Feature

A feature is an input column used by the model.

Examples:

- `finbert_positive`
- `revenue_growth_yoy`
- `volatility_63d`

### Target

The target is what the model tries to predict.

Here:

```text
beat
```

### Holdout Set

The holdout set is data kept separate from training.

It is used to evaluate how well the model performs on unseen data.

### Point-In-Time Data

Point-in-time data means data exactly as it was known at the prediction date.

This matters in finance because using future-revised data can create look-ahead bias.

### Look-Ahead Bias

Look-ahead bias happens when a model accidentally uses information that would not have been available at prediction time.

Example:

Using financial ratios updated after earnings to predict that same earnings event.

### Backtest

A backtest evaluates a model on historical periods to estimate how it might have performed in the past.

### Classifier

A classifier is a model that predicts categories.

This model predicts:

- Beat
- Miss

### Probability Threshold

The probability threshold converts probability into a final decision.

This project uses:

```text
60%
```

### False Positive

The model predicted a beat, but the company missed.

### False Negative

The model predicted a miss, but the company beat.

### True Positive

The model predicted a beat, and the company beat.

### True Negative

The model predicted a miss, and the company missed.

## Current Demo Results

The generated demo artifacts currently show:

```text
Precision: 71%
Recall:    62%
Accuracy:  75%
ROC AUC:   0.76
```

These are from the included demo dataset and are intended for portfolio demonstration. For real production claims, retrain and evaluate on a verified point-in-time S&P 500 dataset.

## Portfolio Talking Points

- Built an end-to-end ML system for earnings surprise prediction.
- Combined NLP sentiment with financial and market features.
- Used a time-based holdout and rolling backtest instead of a random split.
- Tuned the operating threshold toward precision.
- Added SHAP explainability for model transparency.
- Built a Streamlit dashboard with professional monitoring views.
- Designed the project so it runs offline in demo mode but can be extended to real SEC/yfinance/FinBERT data.
