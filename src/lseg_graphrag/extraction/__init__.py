"""Extraction module for knowledge graph triples."""

from lseg_graphrag.extraction.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT_TEMPLATE
from lseg_graphrag.extraction.llm_backends import (
    BaseLLMBackend,
    OpenAIBackend,
    RuleBasedFinancialExtractor,
)
from lseg_graphrag.extraction.extractor import TripleExtractor

__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_USER_PROMPT_TEMPLATE",
    "BaseLLMBackend",
    "OpenAIBackend",
    "RuleBasedFinancialExtractor",
    "TripleExtractor",
]
