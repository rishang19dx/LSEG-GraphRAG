"""Vector RAG baseline retriever (B1) over document chunks."""

from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from lseg_graphrag.schema.provenance import ChunkMetadata, Citation
from lseg_graphrag.retrieval.reranker import PassageReranker


class VectorRetriever:
    """Standard Vector RAG retriever using dense/sparse text vectorization and reranking."""

    def __init__(self, chunks: Optional[List[ChunkMetadata]] = None, use_reranker: bool = True):
        self.chunks: List[ChunkMetadata] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            stop_words="english",
        )
        self.chunk_matrix = None
        self.use_reranker = use_reranker
        self.reranker = PassageReranker() if use_reranker else None

        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: List[ChunkMetadata]):
        """Build the vector search index over the provided document chunks."""
        self.chunks = list(chunks)
        texts = [f"{c.company_symbol} {c.section_title}\n{c.text}" for c in self.chunks]
        self.chunk_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[ChunkMetadata], List[Citation]]:
        """Retrieve the top_k most relevant chunks and their citations for a query."""
        if not self.chunks or self.chunk_matrix is None:
            return [], []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_matrix)[0]

        # Get initial top candidates (2x top_k for reranking)
        fetch_k = min(len(self.chunks), top_k * 2) if self.use_reranker else top_k
        top_indices = np.argsort(scores)[::-1][:fetch_k]

        candidates = [self.chunks[i] for i in top_indices if scores[i] > 0.0]
        if not candidates:
            # Fallback to top_indices anyway if all scores zero
            candidates = [self.chunks[i] for i in top_indices[:top_k]]

        # Apply reranking if enabled
        if self.use_reranker and self.reranker and candidates:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            final_chunks = [c for c, _ in reranked]
        else:
            final_chunks = candidates[:top_k]

        # Extract citations
        citations: List[Citation] = []
        for c in final_chunks:
            # First 120 chars as representative evidence span
            span = c.text[:140].replace("\n", " ") + "..."
            citations.append(
                Citation(
                    doc_id=c.doc_id,
                    section=c.section,
                    chunk_id=c.chunk_id,
                    evidence_span=span,
                    confidence=1.0,
                )
            )

        return final_chunks, citations
