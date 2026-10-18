"""Corporate name normalizer and suffix cleaner."""

import re


LEGAL_SUFFIXES = [
    r"\binc(?:\.|\b)",
    r"\bcorp(?:\.|\b)",
    r"\bcorporation\b",
    r"\bco(?:\.|\b)",
    r"\bcompany\b",
    r"\bltd(?:\.|\b)",
    r"\blimited\b",
    r"\bllc\b",
    r"\bl\.l\.c\b",
    r"\bplc\b",
    r"\bn\.v\b",
    r"\bnv\b",
    r"\bb\.v\b",
    r"\bbv\b",
    r"\bgmbh\b",
    r"\bholdings?\b",
    r"\bholding\b",
    r"\bgroup\b",
    r"\bpte(?:\.|\b)",
    r"\bpvt(?:\.|\b)",
    r"\bprivate\b",
    r"\bkgaa\b",
    r"\bsa\b",
    r"\bag\b",
]


def normalize_company_name(name: str) -> str:
    """Normalize company name by stripping legal suffixes, punctuation, and extra spaces."""
    cleaned = name.strip()

    # Lowercase for suffix stripping
    lower = cleaned.lower()
    for pattern in LEGAL_SUFFIXES:
        lower = re.sub(pattern, "", lower, flags=re.IGNORECASE)

    # Clean punctuation and extra spaces
    lower = re.sub(r"[,;:\(\)\.\-]", " ", lower)
    lower = " ".join(lower.split())

    return lower.strip()
