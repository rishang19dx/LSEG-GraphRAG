"""Hybrid RAG retriever (S2) combining structural graph traversal with vector similarity."""

from typing import List, Dict, Any, Tuple, Set, Optional
import networkx as nx

from lseg_graphrag.schema.provenance import ChunkMetadata, Citation
from lseg_graphrag.retrieval.vector_rag import VectorRetriever
from lseg_graphrag.retrieval.graph_rag import GraphRetriever


class HybridRetriever:
    """Combines graph traversal for entity candidate expansion with vector search."""

    def __init__(self, graph_retriever: GraphRetriever, vector_retriever: VectorRetriever):
        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever

    def retrieve(
        self,
        query: str,
        max_hops: int = 2,
        top_k_chunks: int = 3,
    ) -> Tuple[str, List[Citation], nx.MultiDiGraph]:
        """Execute hybrid retrieval returning synthesized context and deduplicated citations."""
        # 1. Graph retrieval
        graph_context, graph_citations, subgraph = self.graph_retriever.retrieve(
            query=query,
            max_hops=max_hops,
        )

        # 2. Extract entity names from subgraph to enrich vector query
        subgraph_entity_names = []
        for n in subgraph.nodes:
            entity_data = self.graph_retriever.kg.get_entity(n)
            if entity_data:
                subgraph_entity_names.append(entity_data.get("name", n))

        # Enriched query combines original question with discovered entity context
        enriched_query = f"{query} " + " ".join(subgraph_entity_names[:5])

        # 3. Vector retrieval over chunks
        vector_chunks, vector_citations = self.vector_retriever.retrieve(
            query=enriched_query,
            top_k=top_k_chunks,
        )

        # 4. Synthesize hybrid context
        context_parts = []
        if graph_context:
            context_parts.append("### Structured Knowledge Graph Facts:")
            context_parts.append(graph_context)

        if vector_chunks:
            context_parts.append("\n### Supporting Document Passages:")
            for c in vector_chunks:
                context_parts.append(
                    f"[{c.doc_id} | {c.section_title}]\n{c.text.strip()}\n"
                )

        combined_context = "\n".join(context_parts)

        # 5. Merge and deduplicate citations
        all_citations: List[Citation] = []
        seen_spans: Set[str] = set()

        # Prioritize graph citations (they have exact edge evidence spans)
        for c in graph_citations:
            span_key = c.evidence_span[:80].strip().lower()
            if span_key and span_key not in seen_spans:
                seen_spans.add(span_key)
                all_citations.append(c)

        for c in vector_citations:
            span_key = c.evidence_span[:80].strip().lower()
            if span_key and span_key not in seen_spans:
                seen_spans.add(span_key)
                all_citations.append(c)

        return combined_context, all_citations, subgraph
