# VS Code Run Guide

## 1. Open The Project Folder

Open this folder in VS Code:

```text
C:\Users\Samiyah\Documents\Codex\2026-05-26\project-01-earnings-surprise-predictor-predict
```

In VS Code:

1. Go to **File > Open Folder**.
2. Select `project-01-earnings-surprise-predictor-predict`.
3. Trust the workspace if VS Code asks.

## 2. Select Python Interpreter

The project already has a virtual environment at:

```text
.venv\Scripts\python.exe
```

In VS Code:

1. Press `Ctrl + Shift + P`.
2. Search **Python: Select Interpreter**.
3. Choose:

```text
.\.venv\Scripts\python.exe
```

The `.vscode/settings.json` file is already configured to prefer this interpreter.

## 3. Install Dependencies If Needed

If VS Code says packages are missing, run this in the VS Code terminal:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional FinBERT dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

The app works without the optional FinBERT packages because it has a local lexical sentiment fallback.

## 4. Generate Demo Data

Run:

```powershell
.\.venv\Scripts\python.exe scripts\generate_demo_data.py
```

This creates:

```text
data\processed\earnings_surprise_dataset.csv
```

## 5. Train The Model

Run:

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --demo
```

This creates model artifacts in:

```text
artifacts\
```

Important generated files:

- `earnings_surprise_model.joblib`: trained model bundle
- `metrics.json`: holdout precision, recall, accuracy, ROC AUC
- `shap_summary.csv`: global explanation summary
- `holdout_predictions.csv`: predictions for the holdout set
- `backtest_metrics.csv`: rolling-year backtest results

## 6. Run The Dashboard

Run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL shown in the terminal. Usually it is:

```text
http://localhost:8501
```

## 7. Use VS Code Tasks

You can also run everything from VS Code tasks:

1. Press `Ctrl + Shift + P`.
2. Search **Tasks: Run Task**.
3. Choose one of:
   - `Generate Demo Data`
   - `Train Model`
   - `Run Streamlit Dashboard`

## Recommended Demo Flow

For a presentation or resume walkthrough:

```powershell
.\.venv\Scripts\python.exe scripts\generate_demo_data.py
.\.venv\Scripts\python.exe scripts\train_model.py --demo
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```
