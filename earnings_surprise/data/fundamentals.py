from __future__ import annotations

import pandas as pd


def load_yfinance_features(ticker: str) -> dict[str, float]:
    """Fetch current fundamental fields from yfinance.

    This function is intentionally narrow. Historical point-in-time fundamentals are
    vendor-dependent, so production backtests should replace this with a point-in-time
    source to avoid look-ahead bias.
    """

    import yfinance as yf

    info = yf.Ticker(ticker).get_info()
    return {
        "pe_ratio": _safe_float(info.get("trailingPE")),
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "gross_margin": _safe_float(info.get("grossMargins")),
        "operating_margin": _safe_float(info.get("operatingMargins")),
        "revenue_growth_yoy": _safe_float(info.get("revenueGrowth")),
    }


def price_features(price_history: pd.DataFrame) -> dict[str, float]:
    close = price_history["Close"].dropna()
    returns = close.pct_change().dropna()
    if close.empty or len(returns) < 20:
        return {"price_momentum_63d": 0.0, "volatility_63d": 0.0}
    lookback = min(63, len(close) - 1)
    momentum = close.iloc[-1] / close.iloc[-lookback - 1] - 1
    volatility = returns.tail(lookback).std() * (252**0.5)
    return {"price_momentum_63d": float(momentum), "volatility_63d": float(volatility)}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
