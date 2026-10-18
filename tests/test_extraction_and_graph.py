"""Unit tests for triple extraction, entity resolution, graph store, and multi-hop traversal."""

import pytest
from lseg_graphrag.schema.provenance import ChunkMetadata
from lseg_graphrag.schema.relations import RelationType
from lseg_graphrag.extraction.extractor import TripleExtractor
from lseg_graphrag.extraction.llm_backends import RuleBasedFinancialExtractor
from lseg_graphrag.resolution.normalizer import normalize_company_name
from lseg_graphrag.resolution.resolver import EntityResolver
from lseg_graphrag.graph.store import FinancialKnowledgeGraph
from lseg_graphrag.graph.traverser import GraphTraverser


def test_company_normalizer():
    assert normalize_company_name("NVIDIA Corporation") == "nvidia"
    assert normalize_company_name("Apple Inc.") == "apple"
    assert normalize_company_name("ASML Holding N.V.") == "asml"
    assert normalize_company_name("Mellanox Technologies, Ltd.") == "mellanox technologies"


def test_entity_resolution():
    resolver = EntityResolver(similarity_threshold=0.85)

    e1 = resolver.resolve("NVIDIA")
    assert e1.id == "company:nvda"

    e2 = resolver.resolve("NVDA")
    assert e2.id == "company:nvda"

    e3 = resolver.resolve("Taiwan Semiconductor")
    assert e3.id == "company:tsm"

    e4 = resolver.resolve("TSMC Limited")
    assert e4.id == "company:tsm"

    # Region resolution
    reg = resolver.resolve("Taiwan", expected_type="Region")
    assert reg.id == "region:taiwan"


def test_triple_extraction_and_grounding():
    chunk = ChunkMetadata(
        chunk_id="SEC_NVDA_test_01",
        doc_id="SEC_10K_NVDA_2024",
        company_symbol="NVDA",
        section="item_1",
        section_title="Item 1: Business",
        chunk_index=0,
        text="Taiwan Semiconductor Manufacturing Company (TSMC) manufactures our advanced Hopper and Blackwell GPUs.",
    )

    extractor = TripleExtractor(backend=RuleBasedFinancialExtractor())
    res = extractor.extract_from_chunk(chunk)

    assert len(res.triples) >= 1
    t = res.triples[0]
    assert t.relation in (RelationType.SUPPLIER_OF, RelationType.CUSTOMER_OF)
    assert t.confidence >= 0.70
    assert "manufactures" in t.evidence_span


def test_graph_store_provenance_and_subgraph(tmp_path):
    kg = FinancialKnowledgeGraph()
    resolver = EntityResolver()

    # Build small multi-hop graph:
    # (ASML) -[:SUPPLIER_OF]-> (TSMC) -[:SUPPLIER_OF]-> (NVIDIA)
    # (TSMC) -[:OPERATES_IN]-> (Taiwan)
    # (TSMC) -[:EXPOSED_TO]-> (Natural Disaster)

    asml_node = resolver.resolve("ASML")
    tsmc_node = resolver.resolve("TSMC")
    nvda_node = resolver.resolve("NVIDIA")
    taiwan_node = resolver.resolve("Taiwan", expected_type="Region")

    chunk = ChunkMetadata(
        chunk_id="c01", doc_id="SEC_NVDA", company_symbol="NVDA", section="item_1",
        section_title="Business", chunk_index=0, text="TSMC supplies NVIDIA",
    )

    t1 = resolver.resolve_triple(
        type("Obj", (), {
            "head": "TSMC", "head_type": "Company",
            "relation": RelationType.SUPPLIER_OF,
            "tail": "NVIDIA", "tail_type": "Company",
            "evidence_span": "TSMC supplies NVIDIA chips.",
            "confidence": 0.95, "source_doc_id": "SEC_10K_NVDA_2024",
            "source_section": "item_1", "chunk_id": "c01", "filing_date": "2024-02-21",
        })()
    )
    kg.add_triple(t1[0], t1[1], type("Obj", (), {
        "relation": RelationType.SUPPLIER_OF, "evidence_span": "TSMC supplies NVIDIA chips.",
        "confidence": 0.95, "source_doc_id": "SEC_10K_NVDA_2024",
        "source_section": "item_1", "chunk_id": "c01", "filing_date": "2024-02-21",
    })())

    t2 = resolver.resolve_triple(
        type("Obj", (), {
            "head": "ASML", "head_type": "Company",
            "relation": RelationType.SUPPLIER_OF,
            "tail": "TSMC", "tail_type": "Company",
            "evidence_span": "ASML supplies EUV tools to TSMC.",
            "confidence": 0.95, "source_doc_id": "SEC_10K_TSM_2024",
            "source_section": "item_1", "chunk_id": "c02", "filing_date": "2024-04-10",
        })()
    )
    kg.add_triple(t2[0], t2[1], type("Obj", (), {
        "relation": RelationType.SUPPLIER_OF, "evidence_span": "ASML supplies EUV tools to TSMC.",
        "confidence": 0.95, "source_doc_id": "SEC_10K_TSM_2024",
        "source_section": "item_1", "chunk_id": "c02", "filing_date": "2024-04-10",
    })())

    kg.add_triple(tsmc_node, taiwan_node, type("Obj", (), {
        "relation": RelationType.OPERATES_IN, "evidence_span": "TSMC fabs operate in Hsinchu, Taiwan.",
        "confidence": 0.90, "source_doc_id": "SEC_10K_TSM_2024",
        "source_section": "item_1a", "chunk_id": "c03", "filing_date": "2024-04-10",
    })())

    stats = kg.get_stats()
    assert stats["num_nodes"] == 4
    assert stats["num_edges"] == 3

    # Test 2-hop traversal
    subgraph = kg.get_k_hop_subgraph(["company:nvda"], max_hops=2)
    assert "company:asml" in subgraph.nodes
    assert "region:taiwan" in subgraph.nodes

    citations = kg.get_citations_for_subgraph(subgraph)
    assert len(citations) == 3

    # Test serialization round-trip
    save_file = tmp_path / "test_kg.json"
    kg.save_to_json(str(save_file))
    loaded_kg = FinancialKnowledgeGraph.load_from_json(str(save_file))
    assert loaded_kg.graph.number_of_nodes() == 4
    assert loaded_kg.graph.number_of_edges() == 3

    # Test multi-hop traverser
    traverser = GraphTraverser(kg)
    suppliers_exposed = traverser.get_suppliers_exposed_to("company:nvda", "taiwan")
    assert len(suppliers_exposed) == 1
    assert suppliers_exposed[0]["supplier_id"] == "company:tsm"

    paths = traverser.explain_connection_path("company:asml", "company:nvda", max_hops=3)
    assert len(paths) >= 1
    assert paths[0]["hop_count"] == 2
