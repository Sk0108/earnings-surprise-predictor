from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from earnings_surprise.config import PROCESSED_DIR, RANDOM_STATE


TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "JPM",
    "UNH",
    "XOM",
    "PG",
    "HD",
    "COST",
    "AVGO",
    "LLY",
    "V",
    "MA",
    "CRM",
    "ADBE",
    "NFLX",
    "AMD",
]


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    dates = pd.date_range("2023-03-15", "2026-03-15", freq="QS")

    for ticker in TICKERS:
        quality = rng.normal(0.0, 0.45)
        base_eps = rng.uniform(0.6, 4.8)
        for index, date in enumerate(dates):
            sentiment_signal = rng.normal(0.0, 0.8) + quality
            finbert_positive = float(1 / (1 + np.exp(-sentiment_signal)))
            finbert_negative = float(1 - finbert_positive + rng.normal(0, 0.04))
            finbert_negative = float(np.clip(finbert_negative, 0.02, 0.98))
            finbert_neutral = float(np.clip(1 - abs(finbert_positive - finbert_negative), 0.05, 0.95))
            revenue_growth = float(np.clip(rng.normal(0.08 + quality * 0.04, 0.09), -0.18, 0.38))
            gross_margin = float(np.clip(rng.normal(0.46 + quality * 0.03, 0.12), 0.12, 0.84))
            operating_margin = float(np.clip(gross_margin - rng.uniform(0.08, 0.24), 0.02, 0.55))
            pe_ratio = float(np.clip(rng.normal(27 - quality * 3, 11), 5, 75))
            debt_to_equity = float(np.clip(rng.normal(82 - quality * 20, 48), 0, 260))
            momentum = float(np.clip(rng.normal(0.05 + quality * 0.04 + sentiment_signal * 0.03, 0.17), -0.42, 0.66))
            volatility = float(np.clip(rng.normal(0.28 - quality * 0.02, 0.08), 0.08, 0.62))
            revision = float(np.clip(rng.normal(0.01 + sentiment_signal * 0.015, 0.045), -0.14, 0.18))
            tone_uncertainty = float(np.clip(rng.normal(0.035 - quality * 0.008, 0.015), 0.001, 0.11))
            tone_litigious = float(np.clip(rng.normal(0.007, 0.005), 0, 0.035))

            latent = (
                1.95 * finbert_positive
                - 1.25 * finbert_negative
                + 2.75 * revision
                + 1.55 * revenue_growth
                + 1.05 * momentum
                + 0.7 * operating_margin
                - 1.35 * tone_uncertainty
                - 0.5 * volatility
                + quality
                + rng.normal(0, 0.28)
            )
            beat = latent > 0.58
            estimated_eps = base_eps + 0.08 * index + rng.normal(0, 0.16)
            surprise_pct = rng.normal(0.075 if beat else -0.055, 0.05)
            reported_eps = estimated_eps * (1 + surprise_pct)

            rows.append(
                {
                    "ticker": ticker,
                    "company": f"{ticker} Corporation",
                    "sector": rng.choice(["Technology", "Financials", "Health Care", "Consumer", "Energy", "Industrials"]),
                    "earnings_date": date + pd.Timedelta(days=int(rng.integers(5, 38))),
                    "fiscal_quarter": f"FY{date.year} Q{((date.month - 1) // 3) + 1}",
                    "reported_eps": round(float(reported_eps), 2),
                    "estimated_eps": round(float(estimated_eps), 2),
                    "finbert_positive": finbert_positive,
                    "finbert_negative": finbert_negative,
                    "finbert_neutral": finbert_neutral,
                    "tone_uncertainty": tone_uncertainty,
                    "tone_litigious": tone_litigious,
                    "eps_trend_4q": float(rng.normal(0.12 + quality * 0.07, 0.21)),
                    "revenue_growth_yoy": revenue_growth,
                    "gross_margin": gross_margin,
                    "operating_margin": operating_margin,
                    "pe_ratio": pe_ratio,
                    "debt_to_equity": debt_to_equity,
                    "price_momentum_63d": momentum,
                    "volatility_63d": volatility,
                    "estimate_revision_30d": revision,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["earnings_date", "ticker"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_DIR / "earnings_surprise_dataset.csv", index=False)
    print(f"Wrote {len(frame)} demo events to {PROCESSED_DIR / 'earnings_surprise_dataset.csv'}")


if __name__ == "__main__":
    main()
