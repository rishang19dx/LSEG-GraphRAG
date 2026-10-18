"""Unit tests for ingestion, SEC parsing, and section chunking."""

import os
import json
import pytest
from lseg_graphrag.schema.provenance import DocumentMetadata
from lseg_graphrag.ingestion.chunker import SectionAwareChunker
from lseg_graphrag.ingestion.sec_edgar import SecEdgarClient
from lseg_graphrag.ingestion.corpus_builder import (
    get_semiconductor_seed_documents,
    build_and_save_sample_corpus,
)


def test_chunker_provenance_and_indexing():
    doc = DocumentMetadata(
        doc_id="SEC_10K_TEST_2024",
        company_symbol="TEST",
        company_name="Test Company",
        cik="0001234567",
        form_type="10-K",
        filing_date="2024-01-15",
    )
    text = (
        "Paragraph 1 discusses supply chain relationships with vendor A and vendor B.\n\n"
        "Paragraph 2 discusses operational locations and risks in regional facilities.\n\n"
        "Paragraph 3 details revenue metrics and sales distribution across markets."
    )
    chunker = SectionAwareChunker(chunk_size=120, chunk_overlap=30, respect_paragraphs=True)
    chunks = chunker.chunk_section(doc, "item_1", "Item 1: Business", text)

    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"SEC_10K_TEST_2024_item_1_c{idx:03d}"
        assert c.doc_id == "SEC_10K_TEST_2024"
        assert c.company_symbol == "TEST"
        assert c.section == "item_1"
        assert c.token_count is not None and c.token_count > 0


def test_sec_edgar_section_parser():
    sample_html = """
    <html>
        <body>
            <h1>ANNUAL REPORT</h1>
            <div>ITEM 1. BUSINESS</div>
            <p>We are a fabless chip company. TSMC manufactures our wafers.</p>
            <div>ITEM 1A. RISK FACTORS</div>
            <p>We are exposed to earthquakes in Taiwan and trade tariffs.</p>
            <div>ITEM 1B. UNRESOLVED STAFF COMMENTS</div>
            <p>None.</p>
            <div>EXHIBIT 21. SUBSIDIARIES OF THE REGISTRANT</div>
            <p>Mellanox Technologies - Israel</p>
            <div>EXHIBIT 23. CONSENT OF EXPERTS</div>
        </body>
    </html>
    """
    client = SecEdgarClient(cache_dir="/tmp/sec_cache")
    sections = client.parse_10k_sections(sample_html)

    assert "fabless chip company" in sections["item_1"]
    assert "earthquakes in Taiwan" in sections["item_1a"]
    assert "Mellanox Technologies" in sections["exhibit_21"]


def test_corpus_builder_generation(tmp_path):
    chunks = build_and_save_sample_corpus(output_dir=str(tmp_path), chunk_size=800, chunk_overlap=150)
    assert len(chunks) > 10

    # Check files exist
    assert (tmp_path / "documents.json").exists()
    assert (tmp_path / "chunks.json").exists()

    # Check that companies in universe are covered
    symbols = {c.company_symbol for c in chunks}
    assert {"NVDA", "TSM", "AAPL", "ASML", "AMD", "INTC"}.issubset(symbols)

    # Check sections present
    sections = {c.section for c in chunks}
    assert {"item_1", "item_1a", "exhibit_21"}.issubset(sections)
