"""Orchestrator for grounded knowledge graph triple extraction."""

import logging
from typing import List, Optional, Tuple
from lseg_graphrag.schema.relations import ExtractedTriple, ExtractionResult
from lseg_graphrag.schema.provenance import ChunkMetadata
from lseg_graphrag.extraction.llm_backends import BaseLLMBackend, OpenAIBackend, RuleBasedFinancialExtractor

logger = logging.getLogger(__name__)


class TripleExtractor:
    """Extracts grounded triples from document chunks using configured backends."""

    def __init__(
        self,
        backend: Optional[BaseLLMBackend] = None,
        confidence_threshold: float = 0.70,
        require_evidence_span: bool = True,
    ):
        self.backend = backend or RuleBasedFinancialExtractor()
        self.confidence_threshold = confidence_threshold
        self.require_evidence_span = require_evidence_span

    def extract_from_chunk(self, chunk: ChunkMetadata) -> ExtractionResult:
        """Extract triples from a single chunk, validating evidence and confidence."""
        raw_triples, tokens, latency_ms = self.backend.extract(chunk)

        validated: List[ExtractedTriple] = []
        for t in raw_triples:
            if t.confidence < self.confidence_threshold:
                continue

            if self.require_evidence_span:
                # Groundedness verification: check if evidence span is actually in the chunk
                # Allow minor punctuation/whitespace normalization
                span_clean = " ".join(t.evidence_span.split()).lower()
                chunk_clean = " ".join(chunk.text.split()).lower()

                if span_clean not in chunk_clean and not any(part in chunk_clean for part in span_clean.split(".") if len(part) > 15):
                    logger.warning(
                        f"Skipping ungrounded triple ({t.head}) -[{t.relation.value}]-> ({t.tail}): "
                        f"evidence span not in chunk {chunk.chunk_id}"
                    )
                    continue

            validated.append(t)

        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            triples=validated,
            extraction_tokens=tokens,
            latency_ms=latency_ms,
        )

    def extract_from_corpus(self, chunks: List[ChunkMetadata]) -> List[ExtractionResult]:
        """Extract triples across all chunks in a corpus."""
        results: List[ExtractionResult] = []
        for chunk in chunks:
            res = self.extract_from_chunk(chunk)
            results.append(res)
        return results
