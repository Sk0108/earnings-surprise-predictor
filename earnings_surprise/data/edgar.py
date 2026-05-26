from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

import requests
from bs4 import BeautifulSoup

from earnings_surprise.config import SEC_USER_AGENT


SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


@dataclass(frozen=True)
class FilingMetadata:
    ticker: str
    cik: str
    accession_number: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str


class EdgarClient:
    """Small SEC EDGAR client for filing metadata and document text."""

    def __init__(self, user_agent: str = SEC_USER_AGENT, sleep_seconds: float = 0.12):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.sleep_seconds = sleep_seconds

    def get_company_filings(self, cik: str, forms: Iterable[str] = ("10-Q", "10-K")) -> list[FilingMetadata]:
        normalized_cik = str(cik).zfill(10)
        response = self.session.get(SEC_SUBMISSIONS_URL.format(cik=normalized_cik), timeout=30)
        response.raise_for_status()
        time.sleep(self.sleep_seconds)

        payload = response.json()
        ticker = payload.get("tickers", ["UNKNOWN"])[0]
        recent = payload["filings"]["recent"]
        forms_set = set(forms)
        filings: list[FilingMetadata] = []

        for index, form in enumerate(recent["form"]):
            if form not in forms_set:
                continue
            filings.append(
                FilingMetadata(
                    ticker=ticker,
                    cik=normalized_cik,
                    accession_number=recent["accessionNumber"][index],
                    form=form,
                    filing_date=recent["filingDate"][index],
                    report_date=recent["reportDate"][index],
                    primary_document=recent["primaryDocument"][index],
                )
            )
        return filings

    def get_filing_text(self, filing: FilingMetadata) -> str:
        accession = filing.accession_number.replace("-", "")
        cik_no_leading_zero = str(int(filing.cik))
        url = f"{SEC_ARCHIVES_URL}/{cik_no_leading_zero}/{accession}/{filing.primary_document}"
        response = self.session.get(url, timeout=45)
        response.raise_for_status()
        time.sleep(self.sleep_seconds)
        return clean_html(response.text)


def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_management_discussion(text: str, max_chars: int = 16000) -> str:
    """Extract a compact MD&A-like section from a 10-Q/10-K filing."""

    patterns = [
        r"management'?s discussion and analysis.*?(?=quantitative and qualitative disclosures|controls and procedures|financial statements)",
        r"item 2\.\s*management'?s discussion.*?(?=item 3\.|item 4\.)",
        r"item 7\.\s*management'?s discussion.*?(?=item 7a\.|item 8\.)",
    ]
    lower_text = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lower_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return text[match.start() : match.end()][:max_chars]
    return text[:max_chars]
