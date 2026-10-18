"""Relation types and triple schemas for the Financial Knowledge Graph."""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class RelationType(str, Enum):
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    SUPPLIER_OF = "SUPPLIER_OF"
    CUSTOMER_OF = "CUSTOMER_OF"
    COMPETITOR_OF = "COMPETITOR_OF"
    OPERATES_IN = "OPERATES_IN"
    EXPOSED_TO = "EXPOSED_TO"


class Triple(BaseModel):
    """Core knowledge graph triple connecting head to tail via a relation."""
    head: str = Field(..., description="Subject canonical entity ID or mention")
    relation: RelationType = Field(..., description="Relation type connecting head and tail")
    tail: str = Field(..., description="Object canonical entity ID or mention")

    def __repr__(self) -> str:
        return f"({self.head}) -[:{self.relation.value}]-> ({self.tail})"


class ExtractedTriple(BaseModel):
    """An extracted triple accompanied by grounded textual provenance and confidence."""
    head: str = Field(..., description="Subject entity mention")
    head_type: str = Field(default="Company", description="Entity category of subject")
    relation: RelationType = Field(..., description="Extracted relation type")
    tail: str = Field(..., description="Object entity mention")
    tail_type: str = Field(default="Company", description="Entity category of object")
    evidence_span: str = Field(..., description="Exact substring from source chunk supporting this relation")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction model confidence (0.0 to 1.0)")
    source_doc_id: str = Field(..., description="Identifier of the source document")
    source_section: str = Field(..., description="Section of the document (e.g. Item 1, Item 1A, Exhibit 21)")
    chunk_id: Optional[str] = Field(default=None, description="Identifier of the source chunk")
    filing_date: Optional[str] = Field(default=None, description="Date of document filing or article publication")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata")


class ExtractionResult(BaseModel):
    """Container for all triples extracted from a single chunk or document."""
    chunk_id: str
    doc_id: str
    triples: List[ExtractedTriple] = Field(default_factory=list)
    extraction_tokens: int = Field(default=0)
    latency_ms: float = Field(default=0.0)
