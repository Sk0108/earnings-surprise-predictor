from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Transcript:
    ticker: str
    quarter: str
    speaker_section: str
    text: str


class TranscriptProvider:
    def fetch(self, ticker: str, quarter: str) -> Transcript:
        raise NotImplementedError


class DemoTranscriptProvider(TranscriptProvider):
    """Deterministic transcript-like content for demos and offline tests."""

    POSITIVE = (
        "Management highlighted resilient demand, disciplined execution, margin expansion, "
        "improving customer retention, and strong visibility into enterprise renewals. "
        "Leadership expects operating leverage to continue as supply constraints normalize."
    )
    NEGATIVE = (
        "Management discussed softer order timing, elevated inventory, cautious customer budgets, "
        "pricing pressure, and uncertainty in near-term demand. Leadership emphasized cost controls "
        "while acknowledging execution risk."
    )

    def fetch(self, ticker: str, quarter: str) -> Transcript:
        checksum = sum(ord(char) for char in f"{ticker}-{quarter}")
        text = self.POSITIVE if checksum % 3 else self.NEGATIVE
        return Transcript(ticker=ticker, quarter=quarter, speaker_section="prepared_remarks", text=text)
