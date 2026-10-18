"""GraphRAG retriever (S1) utilizing entity linking, k-hop traversal, and provenance."""

import re
from typing import List, Dict, Any, Tuple, Set, Optional
import networkx as nx
from lseg_graphrag.graph.store import FinancialKnowledgeGraph
from lseg_graphrag.graph.traverser import GraphTraverser
from lseg_graphrag.resolution.resolver import EntityResolver
from lseg_graphrag.schema.provenance import Citation
from lseg_graphrag.schema.entities import EntityNode


class GraphRetriever:
    """GraphRAG retriever that performs entity linking and multi-hop subgraph extraction."""

    def __init__(self, kg: FinancialKnowledgeGraph, resolver: Optional[EntityResolver] = None):
        self.kg = kg
        self.resolver = resolver or EntityResolver()
        self.traverser = GraphTraverser(kg)

    def link_entities_in_query(self, query: str) -> List[EntityNode]:
        """Extract and link all candidate entities mentioned in the natural language query."""
        linked_entities: List[EntityNode] = []
        seen_ids: Set[str] = set()
        q_lower = query.lower()

        # Check against all registered canonical names and aliases
        for entity in self.resolver.canonical_entities.values():
            # Check canonical name
            if entity.name.lower() in q_lower:
                if entity.id not in seen_ids:
                    seen_ids.add(entity.id)
                    linked_entities.append(entity)
                continue

            # Check ticker (exact word match)
            if entity.ticker and re.search(rf"\b{re.escape(entity.ticker.lower())}\b", q_lower):
                if entity.id not in seen_ids:
                    seen_ids.add(entity.id)
                    linked_entities.append(entity)
                continue

            # Check aliases
            for alias in entity.aliases:
                if len(alias) >= 2 and re.search(rf"\b{re.escape(alias.lower())}\b", q_lower):
                    if entity.id not in seen_ids:
                        seen_ids.add(entity.id)
                        linked_entities.append(entity)
                    break

        return linked_entities

    def retrieve(
        self,
        query: str,
        max_hops: int = 2,
        direction: str = "both",
    ) -> Tuple[str, List[Citation], nx.MultiDiGraph]:
        """Retrieve k-hop subgraph context and provenance citations for a query."""
        linked_entities = self.link_entities_in_query(query)

        if not linked_entities:
            # Fallback: scan for any token matching known entities
            return "", [], nx.MultiDiGraph()

        seed_ids = [e.id for e in linked_entities]
        subgraph = self.kg.get_k_hop_subgraph(seed_ids, max_hops=max_hops, direction=direction)

        # If two or more entities are mentioned, find explicit connection paths
        path_facts = []
        if len(seed_ids) >= 2:
            for i in range(len(seed_ids)):
                for j in range(i + 1, len(seed_ids)):
                    paths = self.traverser.explain_connection_path(seed_ids[i], seed_ids[j], max_hops=max_hops + 1)
                    for p in paths[:2]:
                        path_facts.append(f"[CONNECTION PATH ({p['hop_count']} hops)]: {p['chain_description']}")

        # Extract structured edge facts with evidence spans
        facts: List[str] = list(path_facts)
        citations: List[Citation] = []
        seen_spans: Set[str] = set()

        for u, v, data in subgraph.edges(data=True):
            u_node = self.kg.get_entity(u)
            v_node = self.kg.get_entity(v)
            u_name = u_node.get("name", u) if u_node else u
            v_name = v_node.get("name", v) if v_node else v
            rel = data.get("relation", "CONNECTED_TO")
            span = data.get("evidence_span", "").strip()
            doc_id = data.get("source_doc_id", "Unknown")
            sec = data.get("source_section", "Unknown")

            fact_str = f"- {u_name} -[:{rel}]-> {v_name} | Evidence: \"{span}\" [{doc_id} | {sec}]"
            facts.append(fact_str)

            if span and span not in seen_spans:
                seen_spans.add(span)
                citations.append(
                    Citation(
                        doc_id=doc_id,
                        section=sec,
                        chunk_id=data.get("chunk_id"),
                        evidence_span=span,
                        relation=rel,
                        confidence=data.get("confidence", 1.0),
                    )
                )

        formatted_context = "\n".join(facts)
        return formatted_context, citations, subgraph
