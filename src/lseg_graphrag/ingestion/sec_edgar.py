"""SEC EDGAR 10-K fetcher and section extractor (Item 1, Item 1A, Exhibit 21)."""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import requests
from bs4 import BeautifulSoup
from lseg_graphrag.schema.provenance import DocumentMetadata


class SecEdgarClient:
    """Client for querying SEC EDGAR and extracting key 10-K sections."""

    SEC_SUBMISSION_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{doc_filename}"

    def __init__(self, user_agent: Optional[str] = None, cache_dir: str = "data/raw"):
        self.user_agent = user_agent or os.getenv(
            "SEC_EDGAR_USER_AGENT",
            "LSEG-GraphRAG-Research rishang19dx@users.noreply.github.com"
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov" if "data.sec" in self.SEC_SUBMISSION_URL else "www.sec.gov",
        })

    def get_latest_10k_metadata(self, cik: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Query the SEC submissions endpoint to find the latest 10-K filing."""
        clean_cik = str(cik).lstrip("0")
        padded_cik = clean_cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"

        try:
            headers = {"User-Agent": self.user_agent}
            resp = self.session.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                return None

            data = resp.json()
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            filing_dates = recent.get("filingDate", [])
            primary_docs = recent.get("primaryDocument", [])

            for idx, form in enumerate(forms):
                if form == "10-K":
                    accession = accessions[idx]
                    return {
                        "symbol": symbol,
                        "cik": padded_cik,
                        "company_name": data.get("name", symbol),
                        "form": form,
                        "accession": accession,
                        "accession_clean": accession.replace("-", ""),
                        "filing_date": filing_dates[idx],
                        "primary_doc": primary_docs[idx],
                        "doc_id": f"SEC_10K_{symbol}_{filing_dates[idx][:4]}",
                    }
            return None
        except Exception:
            return None

    def parse_10k_sections(self, raw_html_or_text: str) -> Dict[str, str]:
        """Extract Item 1 (Business), Item 1A (Risk Factors), and Exhibit 21 from 10-K text."""
        # Convert HTML to clean text while preserving structure
        soup = BeautifulSoup(raw_html_or_text, "html.parser")
        text = soup.get_text(separator="\n")
        text = re.sub(r"\xa0", " ", text)
        text = re.sub(r"[ \t]+", " ", text)

        sections: Dict[str, str] = {
            "item_1": "",
            "item_1a": "",
            "exhibit_21": "",
        }

        # Regex patterns for Item 1 (Business)
        item1_pattern = re.compile(
            r"(?:ITEM\s+1\b[\.:\-–\s]*BUSINESS)(.*?)(?:ITEM\s+1A\b[\.:\-–\s]*RISK\s+FACTORS)",
            re.IGNORECASE | re.DOTALL,
        )
        # Regex patterns for Item 1A (Risk Factors)
        item1a_pattern = re.compile(
            r"(?:ITEM\s+1A\b[\.:\-–\s]*RISK\s+FACTORS)(.*?)(?:ITEM\s+1B\b|ITEM\s+2\b)",
            re.IGNORECASE | re.DOTALL,
        )
        # Regex pattern for Exhibit 21 (Subsidiaries)
        exhibit21_pattern = re.compile(
            r"(?:EXHIBIT\s+21\b.*?SUBSIDIARIES)(.*?)(?:EXHIBIT\s+22\b|EXHIBIT\s+23\b|EXHIBIT\s+31\b|$)",
            re.IGNORECASE | re.DOTALL,
        )

        m1 = item1_pattern.search(text)
        if m1:
            sections["item_1"] = self._clean_section(m1.group(1))

        m1a = item1a_pattern.search(text)
        if m1a:
            sections["item_1a"] = self._clean_section(m1a.group(1))

        m21 = exhibit21_pattern.search(text)
        if m21:
            sections["exhibit_21"] = self._clean_section(m21.group(1))

        return sections

    def _clean_section(self, section_text: str) -> str:
        """Strip boilerplate headers and clean section text."""
        lines = [line.strip() for line in section_text.splitlines() if line.strip()]
        # Filter out common table-of-contents page number lines (e.g. "Table of Contents", numbers alone)
        filtered = [
            line for line in lines
            if not re.match(r"^Table of Contents$", line, re.I) and not re.match(r"^\d+$", line)
        ]
        return "\n\n".join(filtered)
