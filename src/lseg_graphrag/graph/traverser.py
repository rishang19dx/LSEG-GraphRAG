"""Specialized multi-hop graph traversers for financial relationship patterns."""

from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx
from lseg_graphrag.graph.store import FinancialKnowledgeGraph
from lseg_graphrag.schema.provenance import Citation


class GraphTraverser:
    """Executes multi-hop traversal patterns matching the PRD financial use cases."""

    def __init__(self, kg: FinancialKnowledgeGraph):
        self.kg = kg

    def get_suppliers(self, company_id: str) -> List[Tuple[str, List[Citation]]]:
        """1-hop: Find all direct suppliers of a company with citations."""
        results: List[Tuple[str, List[Citation]]] = []
        if company_id not in self.kg.graph:
            return results

        # In MultiDiGraph: (Supplier) -[:SUPPLIER_OF]-> (Company)
        for u, v, data in self.kg.graph.in_edges(company_id, data=True):
            if data.get("relation") == "SUPPLIER_OF":
                citation = Citation(
                    doc_id=data.get("source_doc_id", "Unknown"),
                    section=data.get("source_section", "Unknown"),
                    chunk_id=data.get("chunk_id"),
                    evidence_span=data.get("evidence_span", ""),
                    relation="SUPPLIER_OF",
                )
                results.append((u, [citation]))

        return results

    def get_subsidiaries(self, company_id: str) -> List[Tuple[str, List[Citation]]]:
        """1-hop: Find all direct subsidiaries of a parent company."""
        results: List[Tuple[str, List[Citation]]] = []
        if company_id not in self.kg.graph:
            return results

        # (Subsidiary) -[:SUBSIDIARY_OF]-> (Parent)
        for u, v, data in self.kg.graph.in_edges(company_id, data=True):
            if data.get("relation") == "SUBSIDIARY_OF":
                citation = Citation(
                    doc_id=data.get("source_doc_id", "Unknown"),
                    section=data.get("source_section", "Unknown"),
                    chunk_id=data.get("chunk_id"),
                    evidence_span=data.get("evidence_span", ""),
                    relation="SUBSIDIARY_OF",
                )
                results.append((u, [citation]))

        return results

    def get_suppliers_exposed_to(self, company_id: str, risk_or_region_term: str) -> List[Dict[str, Any]]:
        """2-to-3 hop: Which of company's suppliers are exposed to risk factor or region Y?"""
        exposures: List[Dict[str, Any]] = []
        direct_suppliers = self.get_suppliers(company_id)
        term_lower = risk_or_region_term.lower()

        for supplier_id, sup_citations in direct_suppliers:
            supplier_node = self.kg.get_entity(supplier_id)
            supplier_name = supplier_node.get("name", supplier_id) if supplier_node else supplier_id

            # Check edges from this supplier for OPERATES_IN or EXPOSED_TO
            matched_edges = []
            for _, target, data in self.kg.graph.out_edges(supplier_id, data=True):
                rel = data.get("relation")
                target_node = self.kg.get_entity(target)
                target_name = target_node.get("name", target) if target_node else target

                # Check if target matches query term
                is_match = False
                if term_lower in target.lower() or term_lower in target_name.lower():
                    is_match = True
                elif target_node and any(term_lower in alias.lower() for alias in target_node.get("aliases", [])):
                    is_match = True

                if is_match and rel in ("OPERATES_IN", "EXPOSED_TO"):
                    matched_edges.append((target_name, rel, data))

            if matched_edges:
                all_cits = list(sup_citations)
                for t_name, rel, data in matched_edges:
                    all_cits.append(
                        Citation(
                            doc_id=data.get("source_doc_id", "Unknown"),
                            section=data.get("source_section", "Unknown"),
                            chunk_id=data.get("chunk_id"),
                            evidence_span=data.get("evidence_span", ""),
                            relation=rel,
                        )
                    )

                exposures.append({
                    "supplier_id": supplier_id,
                    "supplier_name": supplier_name,
                    "target_exposures": [m[0] for m in matched_edges],
                    "citations": all_cits,
                })

        return exposures

    def get_shared_suppliers(self, company_a: str, company_b: str) -> List[Dict[str, Any]]:
        """2-to-3 hop aggregation: Which companies share suppliers with Company A and Company B?"""
        suppliers_a = {s[0]: s[1] for s in self.get_suppliers(company_a)}
        suppliers_b = {s[0]: s[1] for s in self.get_suppliers(company_b)}

        common = set(suppliers_a.keys()) & set(suppliers_b.keys())
        results = []
        for s_id in common:
            s_node = self.kg.get_entity(s_id)
            s_name = s_node.get("name", s_id) if s_node else s_id
            citations = suppliers_a[s_id] + suppliers_b[s_id]
            results.append({
                "supplier_id": s_id,
                "supplier_name": s_name,
                "citations": citations,
            })
        return results

    def explain_connection_path(self, source_id: str, target_id: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """Path explanation: How is Company A connected to Company B?"""
        paths = self.kg.find_paths(source_id, target_id, cutoff=max_hops, as_undirected=True)
        explanations = []

        for p in paths:
            hops_info = []
            path_citations = []
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                # Look for edge in either direction
                edge_found = None
                direction = "forward"
                if self.kg.graph.has_edge(u, v):
                    edge_dict = self.kg.graph.get_edge_data(u, v)
                    first_key = list(edge_dict.keys())[0]
                    edge_found = edge_dict[first_key]
                elif self.kg.graph.has_edge(v, u):
                    edge_dict = self.kg.graph.get_edge_data(v, u)
                    first_key = list(edge_dict.keys())[0]
                    edge_found = edge_dict[first_key]
                    direction = "reverse"

                if edge_found:
                    rel = edge_found.get("relation", "CONNECTED_TO")
                    u_name = self.kg.get_entity(u).get("name", u) if self.kg.get_entity(u) else u
                    v_name = self.kg.get_entity(v).get("name", v) if self.kg.get_entity(v) else v
                    hops_info.append(f"{u_name} -[{rel}]-> {v_name}" if direction == "forward" else f"{v_name} -[{rel}]-> {u_name}")
                    path_citations.append(
                        Citation(
                            doc_id=edge_found.get("source_doc_id", "Unknown"),
                            section=edge_found.get("source_section", "Unknown"),
                            chunk_id=edge_found.get("chunk_id"),
                            evidence_span=edge_found.get("evidence_span", ""),
                            relation=rel,
                        )
                    )

            explanations.append({
                "path_nodes": p,
                "hop_count": len(p) - 1,
                "chain_description": " => ".join(hops_info),
                "citations": path_citations,
            })

        return explanations
