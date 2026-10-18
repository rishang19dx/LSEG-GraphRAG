"""Unit tests for schema models, validation, and configuration loading."""

import pytest
from lseg_graphrag.schema.entities import EntityNode, EntityType
from lseg_graphrag.schema.relations import RelationType, Triple, ExtractedTriple
from lseg_graphrag.schema.provenance import DocumentMetadata, ChunkMetadata, Citation
from lseg_graphrag.config import AppConfig


def test_entity_node_mention_matching():
    entity = EntityNode(
        id="company:nvda",
        name="NVIDIA Corporation",
        entity_type=EntityType.COMPANY,
        aliases=["NVIDIA", "Nvidia Corp", "NVDA Corp"],
        ticker="NVDA",
        cik="0001045810",
    )
    assert entity.matches_mention("NVIDIA")
    assert entity.matches_mention("nvidia corporation")
    assert entity.matches_mention("nvda")
    assert not entity.matches_mention("Intel")


def test_extracted_triple_validation():
    triple = ExtractedTriple(
        head="TSMC",
        head_type="Company",
        relation=RelationType.SUPPLIER_OF,
        tail="NVIDIA",
        tail_type="Company",
        evidence_span="TSMC is our primary foundry manufacturing our advanced Hopper and Blackwell GPUs.",
        confidence=0.95,
        source_doc_id="SEC_10K_NVDA_2024",
        source_section="item_1",
        chunk_id="SEC_10K_NVDA_2024_item1_c001",
        filing_date="2024-02-21",
    )
    assert triple.head == "TSMC"
    assert triple.relation == RelationType.SUPPLIER_OF
    assert triple.confidence == 0.95
    assert "Hopper and Blackwell" in triple.evidence_span


def test_citation_formatting():
    citation = Citation(
        doc_id="SEC_10K_NVDA_2024",
        section="item_1",
        chunk_id="SEC_10K_NVDA_2024_item1_c001",
        evidence_span="TSMC is our primary foundry manufacturing our advanced Hopper and Blackwell GPUs.",
        confidence=0.95,
    )
    formatted = citation.format_citation()
    assert "[SEC_10K_NVDA_2024 | item_1" in formatted


def test_config_loader():
    config = AppConfig.load_from_yaml("config/default.yaml")
    assert config.project_name == "LSEG-GraphRAG"
    assert len(config.companies) >= 10
    symbols = [c.symbol for c in config.companies]
    assert "NVDA" in symbols
    assert "TSM" in symbols
    assert "AAPL" in symbols
    assert len(config.schema_relations) == 6
