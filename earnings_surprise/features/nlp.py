from __future__ import annotations

import re
from dataclasses import dataclass


POSITIVE_WORDS = {
    "accelerating",
    "beat",
    "disciplined",
    "expansion",
    "improving",
    "leverage",
    "resilient",
    "strong",
    "visibility",
}
NEGATIVE_WORDS = {
    "cautious",
    "decline",
    "elevated",
    "miss",
    "pressure",
    "risk",
    "soft",
    "uncertainty",
    "weak",
}
UNCERTAINTY_WORDS = {"uncertain", "uncertainty", "risk", "volatile", "cautious", "may", "could"}
LITIGIOUS_WORDS = {"legal", "claim", "litigation", "regulatory", "settlement", "investigation"}


@dataclass(frozen=True)
class SentimentFeatures:
    finbert_positive: float
    finbert_negative: float
    finbert_neutral: float
    tone_uncertainty: float
    tone_litigious: float


class FinBertFeatureExtractor:
    """FinBERT feature extractor with a fast lexical fallback.

    The fallback keeps the project runnable on machines where PyTorch/transformers
    are not installed. Install requirements-ml.txt to use the real model.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", use_transformers: bool = True):
        self.model_name = model_name
        self.pipeline = None
        if use_transformers:
            try:
                from transformers import pipeline

                self.pipeline = pipeline("text-classification", model=model_name, tokenizer=model_name, top_k=None)
            except Exception:
                self.pipeline = None

    def transform_text(self, text: str) -> SentimentFeatures:
        if self.pipeline is not None:
            return self._transform_with_finbert(text)
        return lexical_sentiment(text)

    def _transform_with_finbert(self, text: str) -> SentimentFeatures:
        chunks = chunk_text(text)
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for chunk in chunks:
            result = self.pipeline(chunk[:1500])[0]
            for item in result:
                scores[item["label"].lower()] += float(item["score"])
        divisor = max(len(chunks), 1)
        lexical = lexical_sentiment(text)
        return SentimentFeatures(
            finbert_positive=scores["positive"] / divisor,
            finbert_negative=scores["negative"] / divisor,
            finbert_neutral=scores["neutral"] / divisor,
            tone_uncertainty=lexical.tone_uncertainty,
            tone_litigious=lexical.tone_litigious,
        )


def lexical_sentiment(text: str) -> SentimentFeatures:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    total = max(len(tokens), 1)
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    uncertainty = sum(token in UNCERTAINTY_WORDS for token in tokens)
    litigious = sum(token in LITIGIOUS_WORDS for token in tokens)
    raw_total = positive + negative
    if raw_total == 0:
        pos_score = neg_score = 0.15
    else:
        pos_score = positive / raw_total
        neg_score = negative / raw_total
    neutral = max(0.0, 1.0 - abs(pos_score - neg_score))
    return SentimentFeatures(
        finbert_positive=float(pos_score),
        finbert_negative=float(neg_score),
        finbert_neutral=float(neutral),
        tone_uncertainty=float(uncertainty / total),
        tone_litigious=float(litigious / total),
    )


def chunk_text(text: str, max_words: int = 220) -> list[str]:
    words = text.split()
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)] or [""]
