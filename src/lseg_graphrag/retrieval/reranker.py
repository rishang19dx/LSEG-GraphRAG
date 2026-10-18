"""Passage reranking utilities combining lexical overlap and semantic relevance."""

import re
from typing import List, Tuple
from difflib import SequenceMatcher
from lseg_graphrag.schema.provenance import ChunkMetadata


class PassageReranker:
    """Reranks candidate chunks based on question token coverage and sequence matching."""

    def __init__(self):
        self.stop_words = {
            "what", "which", "who", "where", "when", "how", "why", "is", "are", "was",
            "were", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "about", "does", "do", "did", "their", "its", "our",
        }

    def rerank(self, query: str, chunks: List[ChunkMetadata], top_k: int = 5) -> List[Tuple[ChunkMetadata, float]]:
        """Rerank chunks by scoring query term overlap and phrase matching."""
        if not chunks:
            return []

        query_tokens = [
            t.lower() for t in re.findall(r"\b\w+\b", query)
            if t.lower() not in self.stop_words and len(t) > 1
        ]

        scored: List[Tuple[ChunkMetadata, float]] = []
        for chunk in chunks:
            text_lower = chunk.text.lower()
            token_matches = sum(1 for tok in query_tokens if tok in text_lower)
            token_score = token_matches / max(1, len(query_tokens))

            # Sequence match ratio for phrase similarity
            seq_score = SequenceMatcher(None, query.lower(), text_lower[:300]).ratio()

            combined_score = 0.7 * token_score + 0.3 * seq_score
            scored.append((chunk, combined_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
