# LSEG-GraphRAG: Financial Knowledge Graph + GraphRAG for Multi-Hop Questions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRD Status](https://img.shields.io/badge/PRD-Draft%20v1-green.svg)]()

> **Project for LSEG AI/ML Intern (Innovation & Intelligence Team)**  
> **Author:** Rishang  
> **Question under investigation:** *When is the extra cost of building and maintaining a knowledge graph worth it compared to standard Vector RAG for financial relationship analysis?*

---

## 1. Executive Summary

Financial analysis frequently demands connecting relationship chains dispersed across filings, exhibits, and news reports. For instance:
* *"Which of NVIDIA's suppliers are exposed to natural disaster risks in Taiwan?"*
* *"Which subsidiaries of ASML operate in regions affected by export controls?"*
* *"Which companies share more than one tier-1 supplier with Apple?"*

While **Standard Vector RAG** excels at semantic similarity, it systematically degrades on multi-hop reasoning, relationship chain traversal, and cross-document entity aggregation. 

**LSEG-GraphRAG** builds an end-to-end pipeline that:
1. Ingests SEC EDGAR 10-K filings (Item 1 Business, Item 1A Risk Factors, Exhibit 21 Subsidiaries) and financial news across a connected semiconductor and hardware supply chain universe.
2. Extracts high-fidelity triples under a strictly typed schema with mandatory textual evidence spans and confidence scores.
3. Performs entity resolution (canonical ticker/CIK mapping, alias normalization, and fuzzy/embedding clustering).
4. Stores the property graph with complete audit provenance on every edge.
5. Evaluates and benchmarks 4 retrieval architectures (**Zero-Shot B0**, **Vector RAG B1**, **GraphRAG S1**, and **Hybrid S2**).
6. Delivers quantifiable empirical evidence: F1 vs hop count, supporting-fact recall, context precision, hallucination rate, token latency, dollar cost, and a **graph build break-even model**.

---

## 2. Systems Under Comparison

| System | Retrieval Mechanism | Primary Purpose |
| :--- | :--- | :--- |
| **B0: Zero-Shot LLM** | None (direct parametric prompting) | Baseline lower bound; measures hallucination & parametric knowledge |
| **B1: Vector RAG** | Dense text embeddings + cross-encoder reranking over document chunks | Industry standard baseline |
| **S1: GraphRAG** | Entity linking on question + k-hop subgraph traversal + edge-attached evidence spans | Structural multi-hop reasoning |
| **S2: Hybrid RAG** | Graph candidate expansion + vector similarity search over scoped chunks | Best of both: structural recall + semantic richness |

All systems operate on the identical underlying chunks and use the exact same LLM prompt template to isolate the retrieval method as the sole independent variable.

---

## 3. Schema Architecture

```
                    ┌─────────────────────────┐
                    │      Company (Head)     │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
   [:SUPPLIER_OF]        [:CUSTOMER_OF]        [:COMPETITOR_OF]
          │                     │                     │
          ▼                     ▼                     ▼
   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
   │   Company   │       │   Company   │       │   Company   │
   └─────────────┘       └─────────────┘       └─────────────┘
          │
   [:OPERATES_IN] ──► [ Region / Country ]
          │
   [:EXPOSED_TO]  ──► [ RiskFactor / Commodity / Event / Policy ]
          │
   [:SUBSIDIARY_OF] ──► [ Parent Company ]
```

### Supported Relation Types
1. `SUBSIDIARY_OF`: Corporate ownership derived from Exhibit 21.
2. `SUPPLIER_OF`: Supply chain relationships (equipment, foundry, packaging, raw materials).
3. `CUSTOMER_OF`: Direct commercial revenue customers.
4. `COMPETITOR_OF`: Market rivalry across product lines.
5. `OPERATES_IN`: Geographic manufacturing, fab locations, or operational facilities.
6. `EXPOSED_TO`: Geographic, commodity (e.g., neon, silicon), geopolitical, or supply chain bottlenecks.

---

## 4. Multi-Stage Implementation Plan

This project is delivered via structured, staged commits:

- **Stage 1: Project Foundation & Schema Architecture** (Current)
  - Virtual environment, dependencies, configuration (`config/default.yaml`), Pydantic domain models (`EntityNode`, `ExtractedTriple`, `DocumentMetadata`, `Citation`), and schema unit tests.
- **Stage 2: SEC EDGAR Ingestion & Section-Aware Chunking**
  - Automated SEC EDGAR 10-K fetcher/parser with CIK lookup, HTML/text cleaner, Item 1 / 1A / Exhibit 21 extractors, and chunking with document provenance.
- **Stage 3: LLM Triple Extraction, Entity Resolution & Knowledge Graph Store**
  - Structured extraction pipeline with evidence spans; entity resolution and alias clustering; NetworkX + SQLite property graph store with full edge provenance.
- **Stage 4: Retrieval Systems & Grounded Generation**
  - Implement B0 (Zero-shot), B1 (Vector RAG), S1 (GraphRAG k-hop traversal), S2 (Hybrid RAG), and grounded answer generation with inline citations and abstain capability.
- **Stage 5: Evaluation Benchmark & Cost-Benefit Analysis**
  - 150-question stratified multi-hop benchmark (1-hop, 2-hop, 3+ hop; lookup, chain, exposure, aggregation, path); metrics engine (Set F1, EM, Fact Recall, Context Precision, Hallucination Rate, Cost & Latency, Break-even calculator).
- **Stage 6: Interactive Streamlit Demo & Technical Report**
  - Visual subgraph explorer, side-by-side system comparator, citation inspector, and publication-ready technical report.

---

## 5. Quickstart

### Prerequisites
- Python 3.10+
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/rishang19dx/LSEG-GraphRAG.git
cd LSEG-GraphRAG

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Running Tests
```bash
pytest tests/ -v
```

---

## 6. Project Structure

```
LSEG-GraphRAG/
├── README.md                          # Project documentation and guide
├── pyproject.toml                     # Python package specifications
├── requirements.txt                   # Pinned project dependencies
├── .env.example                       # Environment variable template
├── config/
│   └── default.yaml                   # Central configuration & company universe
├── data/
│   ├── raw/                           # Ingested 10-Ks and news
│   ├── processed/                     # Parsed sections and chunk index
│   ├── graphs/                        # Serialized knowledge graph snapshots
│   └── benchmark/                     # Multi-hop question benchmark datasets
├── src/
│   └── lseg_graphrag/
│       ├── config.py                  # Pydantic configuration loader
│       ├── schema/                    # Entities, relations, provenance models
│       ├── ingestion/                 # SEC EDGAR fetchers, section chunkers
│       ├── extraction/                # Triple extractor & LLM backends
│       ├── resolution/                # Entity normalization & disambiguation
│       ├── graph/                     # Property graph store & k-hop traverser
│       ├── retrieval/                 # Vector, Graph, and Hybrid retrieval
│       ├── generation/                # Grounded answerer & citation generator
│       └── evaluation/                # Benchmark runner, metrics, cost model
├── tests/                             # Unit and integration test suite
├── app/                               # Streamlit interactive demo dashboard
└── scripts/                           # Pipeline CLI entrypoints
```

---

## 7. License
Distributed under the MIT License. See `LICENSE` for details.
