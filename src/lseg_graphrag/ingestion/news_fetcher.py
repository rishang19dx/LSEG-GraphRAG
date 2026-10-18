"""News document loader and parser for supply chain and market events."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from lseg_graphrag.schema.provenance import DocumentMetadata


class NewsArticle(BaseModel):
    """Structured financial news article model."""
    article_id: str
    headline: str
    publisher: str
    published_date: str
    company_symbols: List[str]
    url: Optional[str] = None
    body: str


class NewsFetcher:
    """Manages news article ingestion and metadata generation."""

    def to_document_metadata(self, article: NewsArticle, primary_symbol: str) -> DocumentMetadata:
        """Convert a news article into document metadata for the chunking pipeline."""
        return DocumentMetadata(
            doc_id=article.article_id,
            company_symbol=primary_symbol,
            company_name=f"News: {article.publisher}",
            cik=None,
            form_type="News",
            filing_date=article.published_date,
            source_url=article.url,
        )
