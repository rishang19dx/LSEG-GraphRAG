"""Document, chunk, and citation provenance schemas."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata describing a financial filing or news document."""
    doc_id: str = Field(..., description="Unique document ID (e.g. 'SEC_10K_NVDA_2024')")
    company_symbol: str = Field(..., description="Ticker symbol or primary company tag")
    company_name: str = Field(..., description="Full legal name of the reporting company")
    cik: Optional[str] = Field(default=None, description="SEC CIK number")
    form_type: str = Field(default="10-K", description="Document type, e.g. 10-K, 10-Q, 8-K, News")
    filing_date: str = Field(..., description="Filing date in YYYY-MM-DD format")
    source_url: Optional[str] = Field(default=None, description="EDGAR URL or news source link")
    raw_path: Optional[str] = Field(default=None, description="Local path to raw file")


class ChunkMetadata(BaseModel):
    """Chunk of text extracted from a financial document with full section provenance."""
    chunk_id: str = Field(..., description="Globally unique chunk ID, e.g. 'SEC_10K_NVDA_2024_item1_c001'")
    doc_id: str = Field(..., description="Parent document identifier")
    company_symbol: str = Field(..., description="Primary company symbol")
    section: str = Field(..., description="Section key (e.g. item_1, item_1a, exhibit_21)")
    section_title: str = Field(..., description="Readable section title (e.g. 'Business', 'Risk Factors')")
    chunk_index: int = Field(..., description="0-indexed position within the document section")
    text: str = Field(..., description="The raw textual content of the chunk")
    token_count: Optional[int] = Field(default=None, description="Estimated token count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata attributes")


class Citation(BaseModel):
    """Citation linking an answer fact back to source evidence."""
    doc_id: str
    section: str
    chunk_id: Optional[str] = None
    evidence_span: str
    relation: Optional[str] = None
    confidence: float = 1.0

    def format_citation(self) -> str:
        """Render a readable inline citation."""
        return f"[{self.doc_id} | {self.section} | \"{self.evidence_span[:80]}...\"]"
