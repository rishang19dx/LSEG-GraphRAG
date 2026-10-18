"""Unit tests for retrieval systems (Vector RAG, GraphRAG, Hybrid RAG) and answer generator."""

import pytest
from lseg_graphrag.schema.provenance import ChunkMetadata
from lseg_graphrag.graph.store import FinancialKnowledgeGraph
from lseg_graphrag.retrieval.vector_rag import VectorRetriever
from lseg_graphrag.retrieval.graph_rag import GraphRetriever
from lseg_graphrag.retrieval.hybrid_rag import HybridRetriever
from lseg_graphrag.generation.answerer import GroundedAnswerer


@pytest.fixture
def sample_kg_and_chunks():
    kg = FinancialKnowledgeGraph.load_from_json("data/graphs/financial_kg.json")

    chunks = [
        ChunkMetadata(
            chunk_id="c_nvda_1",
            doc_id="SEC_10K_NVDA_2024",
            company_symbol="NVDA",
            section="item_1",
            section_title="Item 1: Business",
            chunk_index=0,
            text="TSMC manufactures our advanced Hopper and Blackwell GPUs. Micron supplies HBM3e memory.",
        ),
        ChunkMetadata(
            chunk_id="c_tsm_1",
            doc_id="SEC_10K_TSM_2024",
            company_symbol="TSM",
            section="item_1",
            section_title="Item 1: Business",
            chunk_index=0,
            text="ASML is our sole supplier of EUV lithography tools. We manufacture chips for Apple and NVIDIA.",
        ),
        ChunkMetadata(
            chunk_id="c_tsm_1a",
            doc_id="SEC_10K_TSM_2024",
            company_symbol="TSM",
            section="item_1a",
            section_title="Item 1A: Risk Factors",
            chunk_index=1,
            text="Our wafer fabrication plants in Taiwan are exposed to earthquakes, typhoons, and geopolitical risk.",
        ),
    ]
    return kg, chunks


def test_vector_retriever(sample_kg_and_chunks):
    _, chunks = sample_kg_and_chunks
    retriever = VectorRetriever(chunks=chunks, use_reranker=True)

    query = "Who supplies EUV lithography tools to TSMC?"
    results, citations = retriever.retrieve(query, top_k=2)

    assert len(results) > 0
    assert any("ASML" in r.text for r in results)
    assert len(citations) == len(results)
    assert citations[0].doc_id in ("SEC_10K_TSM_2024", "SEC_10K_NVDA_2024")


def test_graph_retriever(sample_kg_and_chunks):
    kg, _ = sample_kg_and_chunks
    retriever = GraphRetriever(kg=kg)

    # Multi-hop query: NVIDIA -> TSMC -> Taiwan / Earthquake
    query = "Which suppliers of NVIDIA operate in Taiwan or have earthquake risk?"
    context, citations, subgraph = retriever.retrieve(query, max_hops=2)

    assert "TSMC" in context or "Taiwan Semiconductor" in context
    assert len(citations) >= 1
    assert subgraph.number_of_nodes() >= 2


def test_hybrid_retriever(sample_kg_and_chunks):
    kg, chunks = sample_kg_and_chunks
    v_retriever = VectorRetriever(chunks=chunks)
    g_retriever = GraphRetriever(kg=kg)
    h_retriever = HybridRetriever(graph_retriever=g_retriever, vector_retriever=v_retriever)

    query = "How is ASML connected to NVIDIA in the chip manufacturing chain?"
    context, citations, subgraph = h_retriever.retrieve(query, max_hops=2, top_k_chunks=2)

    assert "Knowledge Graph Facts" in context
    assert "Supporting Document Passages" in context
    assert len(citations) >= 1


def test_answer_generator_and_abstain():
    answerer = GroundedAnswerer()

    # Normal grounded answer
    ctx = "TSMC is the primary foundry supplier manufacturing Hopper chips for NVIDIA. [SEC_10K_NVDA_2024 | item_1]"
    ans = answerer.generate_answer(
        system_id="S1",
        question="Who supplies wafers to NVIDIA?",
        context=ctx,
        citations=[],
    )
    assert not ans.abstained
    assert "TSMC" in ans.predicted_entities

    # FR12: Abstain on empty context
    ans_empty = answerer.generate_answer(
        system_id="S1",
        question="Who supplies quantum computers to Apple?",
        context="",
        citations=[],
    )
    assert ans_empty.abstained
    assert "ABSTAIN" in ans_empty.answer_text
