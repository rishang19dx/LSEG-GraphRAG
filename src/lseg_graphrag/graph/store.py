"""Financial Knowledge Graph property store with full provenance preservation."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx
from lseg_graphrag.schema.entities import EntityNode, EntityType
from lseg_graphrag.schema.relations import ExtractedTriple, RelationType
from lseg_graphrag.schema.provenance import Citation


class FinancialKnowledgeGraph:
    """Property graph backed by NetworkX MultiDiGraph preserving edge provenance."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity: EntityNode):
        """Add or update an entity node in the graph."""
        self.graph.add_node(
            entity.id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            aliases=entity.aliases,
            ticker=entity.ticker,
            cik=entity.cik,
            metadata=entity.metadata,
        )

    def add_triple(self, head_node: EntityNode, tail_node: EntityNode, triple: ExtractedTriple):
        """Add a directed edge with full provenance and evidence."""
        # Ensure nodes exist
        if head_node.id not in self.graph:
            self.add_entity(head_node)
        if tail_node.id not in self.graph:
            self.add_entity(tail_node)

        # Unique key for edge to avoid exact duplicate identical citations
        edge_key = f"{triple.relation.value}_{triple.source_doc_id}_{triple.chunk_id}"

        self.graph.add_edge(
            head_node.id,
            tail_node.id,
            key=edge_key,
            relation=triple.relation.value,
            evidence_span=triple.evidence_span,
            confidence=triple.confidence,
            source_doc_id=triple.source_doc_id,
            source_section=triple.source_section,
            chunk_id=triple.chunk_id,
            filing_date=triple.filing_date,
        )

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve node attributes by ID."""
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return None

    def get_neighbors(self, entity_id: str, direction: str = "both") -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get connected entities and edge data."""
        neighbors: List[Tuple[str, str, Dict[str, Any]]] = []
        if entity_id not in self.graph:
            return neighbors

        if direction in ("out", "both"):
            for _, target, edge_data in self.graph.out_edges(entity_id, data=True):
                neighbors.append((entity_id, target, edge_data))

        if direction in ("in", "both"):
            for source, _, edge_data in self.graph.in_edges(entity_id, data=True):
                neighbors.append((source, entity_id, edge_data))

        return neighbors

    def get_k_hop_subgraph(
        self,
        start_entity_ids: List[str],
        max_hops: int = 2,
        direction: str = "both",
    ) -> nx.MultiDiGraph:
        """Extract a k-hop localized subgraph starting from seed entity IDs."""
        visited_nodes: Set[str] = set()
        current_frontier = set(start_entity_ids) & set(self.graph.nodes)
        visited_nodes.update(current_frontier)

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for node in current_frontier:
                if direction in ("out", "both"):
                    for _, target in self.graph.out_edges(node):
                        if target not in visited_nodes:
                            next_frontier.add(target)
                if direction in ("in", "both"):
                    for source, _ in self.graph.in_edges(node):
                        if source not in visited_nodes:
                            next_frontier.add(source)

            visited_nodes.update(next_frontier)
            current_frontier = next_frontier
            if not current_frontier:
                break

        return self.graph.subgraph(visited_nodes).copy()

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        cutoff: int = 3,
        as_undirected: bool = True,
    ) -> List[List[str]]:
        """Find all simple paths between source and target up to cutoff hops."""
        g = self.graph.to_undirected() if as_undirected else self.graph
        if source_id not in g or target_id not in g:
            return []
        try:
            paths = list(nx.all_simple_paths(g, source=source_id, target=target_id, cutoff=cutoff))
            return paths
        except Exception:
            return []

    def get_citations_for_subgraph(self, subgraph: nx.MultiDiGraph) -> List[Citation]:
        """Collect deduplicated citations from all edges in a subgraph."""
        citations: List[Citation] = []
        seen_spans: Set[str] = set()

        for u, v, data in subgraph.edges(data=True):
            span = data.get("evidence_span", "")
            if span and span not in seen_spans:
                seen_spans.add(span)
                citations.append(
                    Citation(
                        doc_id=data.get("source_doc_id", "Unknown"),
                        section=data.get("source_section", "Unknown"),
                        chunk_id=data.get("chunk_id"),
                        evidence_span=span,
                        relation=data.get("relation"),
                        confidence=data.get("confidence", 1.0),
                    )
                )

        return citations

    def get_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the knowledge graph."""
        rel_counts: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "UNKNOWN")
            rel_counts[rel] = rel_counts.get(rel, 0) + 1

        node_types: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("entity_type", "UNKNOWN")
            node_types[nt] = node_types.get(nt, 0) + 1

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "relation_distribution": rel_counts,
            "node_type_distribution": node_types,
            "density": nx.density(self.graph),
        }

    def save_to_json(self, file_path: str = "data/graphs/financial_kg.json"):
        """Serialize knowledge graph nodes and edges to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nodes = []
        for n, data in self.graph.nodes(data=True):
            node_dict = dict(data)
            node_dict["id"] = n
            nodes.append(node_dict)

        edges = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_dict = dict(data)
            edge_dict["source"] = u
            edge_dict["target"] = v
            edge_dict["key"] = key
            edges.append(edge_dict)

        payload = {
            "stats": self.get_stats(),
            "nodes": nodes,
            "edges": edges,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str = "data/graphs/financial_kg.json") -> "FinancialKnowledgeGraph":
        """Load knowledge graph from serialized JSON file."""
        kg = cls()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for n in data.get("nodes", []):
            node_id = n["id"]
            node_attrs = {k: v for k, v in n.items() if k != "id"}
            kg.graph.add_node(node_id, **node_attrs)

        for e in data.get("edges", []):
            u = e["source"]
            v = e["target"]
            key = e.get("key", None)
            edge_attrs = {k: val for k, val in e.items() if k not in ("source", "target", "key")}
            kg.graph.add_edge(u, v, key=key, **edge_attrs)

        return kg
