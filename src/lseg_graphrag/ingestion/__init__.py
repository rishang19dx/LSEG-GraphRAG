"""Ingestion module for SEC EDGAR filings, news articles, and section-aware chunking."""

from lseg_graphrag.ingestion.chunker import SectionAwareChunker
from lseg_graphrag.ingestion.sec_edgar import SecEdgarClient
from lseg_graphrag.ingestion.news_fetcher import NewsFetcher, NewsArticle
from lseg_graphrag.ingestion.corpus_builder import (
    get_semiconductor_seed_documents,
    build_and_save_sample_corpus,
)

__all__ = [
    "SectionAwareChunker",
    "SecEdgarClient",
    "NewsFetcher",
    "NewsArticle",
    "get_semiconductor_seed_documents",
    "build_and_save_sample_corpus",
]
