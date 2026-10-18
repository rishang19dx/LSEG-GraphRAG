# LSEG-GraphRAG: Financial Knowledge Graph + GraphRAG for Multi-Hop Questions

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 19 Passed](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)]()
[![PRD Status: Complete](https://img.shields.io/badge/PRD-Implemented%20v1.0-green.svg)]()

> **Context:** Advanced Financial NLP & Knowledge Graph Project (Prepared for LSEG AI/ML Problem Space - Innovation & Intelligence)  
> **Author:** Rishang  
> **Core Investigation:** *When is the extra cost of building and maintaining an explicit Knowledge Graph worth it over standard Vector RAG for complex financial relationships?*

---

## 1. Executive Summary

Financial analysis frequently demands connecting relationship chains dispersed across filings, exhibits, and news reports. For instance:
* *"Which of Apple's semiconductor manufacturing suppliers are exposed to natural disaster risks in Taiwan?"*
* *"How is ASML connected to NVIDIA in the global hardware supply chain?"*
* *"Which companies share TSMC as their primary foundry partner?"*

While **Standard Vector RAG** excels at single-chunk semantic similarity, it systematically fails when questions require chaining facts across multiple disparate documents or exhibits.

**LSEG-GraphRAG** implements an end-to-end, reproducible pipeline:
1. **Ingests** SEC EDGAR 10-K filings (Item 1 Business, Item 1A Risk Factors, Exhibit 21 Subsidiaries) and financial news across a connected semiconductor and hardware supply chain universe.
2. **Extracts** high-fidelity triples under a strictly typed 6-relation schema with mandatory verbatim evidence spans and confidence scores.
3. **Resolves** entities (canonical CIK/ticker mapping, alias normalization, and fuzzy string clustering).
4. **Stores** a property graph (83 nodes, 141 edges) with full audit provenance on every edge.
5. **Compares 4 Retrieval Architectures**: Zero-Shot (B0), Vector RAG (B1), GraphRAG (S1), and Hybrid RAG (S2).
6. **Evaluates** on a stratified multi-hop benchmark across 1-hop, 2-hop, and 3+ hop questions, reporting Set $F_1$, precision, recall, supporting-fact recall, hallucination rate, and a **graph build break-even cost model**.

---

## 2. Systems Under Comparison

| System | Retrieval Mechanism | Purpose |
| :--- | :--- | :--- |
| **B0: Zero-Shot LLM** | None (direct parametric prompting) | Baseline lower bound; measures hallucination & parametric knowledge |
| **B1: Vector RAG** | TF-IDF / dense embeddings + cross-encoder reranking over chunks | Industry standard baseline |
| **S1: GraphRAG** | Entity linking on question + $k$-hop subgraph traversal + edge evidence spans | Structural multi-hop reasoning |
| **S2: Hybrid RAG** | Graph candidate expansion + vector similarity search over scoped chunks | Best of both: structural precision + narrative richness |

---

## 3. Key Empirical Results

### F1 Score Breakdown by Hop Count (The Key PRD Chart)

```
 F1 Score
  0.60 ┤                                        ╭─── S1 / S2 (0.5119)
       │                                  ╭─────╯
  0.50 ┤                            ╭─────┴──────── B1 (0.4694)
       │                      ╭─────╯
  0.40 ┤                ╭─────┴──────────────────── S1 / S2 (0.3562)
       │          ╭─────╯                   ╭────── B1 (0.3139)
  0.30 ┤    ╭─────┴─────────────────────────┴────── S1 / S2 (0.3000)
       │    │                         ╭──────────── B0 (0.3444)
  0.20 ┤    │                   ╭─────╯
       │    ├───────────────────╯────────────────── B1 (0.2143)
  0.10 ┤    │             ╭─────╯
       │    ├─────────────╯──────────────────────── B0 (0.1750)
  0.00 ┴────┴────────────────────────────────────── B0 (0.0833)
          Hop 1               Hop 2               Hop 3+
```

| Hop Count | B0 (Zero-Shot) | B1 (Vector RAG) | S1 (GraphRAG) | S2 (Hybrid RAG) | Graph Advantage ($\Delta F_1$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Hop 1** | 0.0833 | 0.2143 | **0.3000** | **0.3000** | **+8.57%** |
| **Hop 2** | 0.1750 | 0.3139 | **0.3562** | **0.3562** | **+4.23%** |
| **Hop 3+** | 0.3444 | 0.4694 | **0.5119** | **0.5119** | **+4.25%** |

* On **Path Reasoning** questions: GraphRAG reaches **0.7222 $F_1$** vs Vector RAG's **0.6222 $F_1$** (+10.0%).
* On **Chained Relations**: GraphRAG reaches **0.4300 $F_1$** vs Vector RAG's **0.3333 $F_1$** (+9.67%).
* **Hallucination Rate**: GraphRAG and Hybrid achieved **0.0000%**, compared to 1.53% for Vector RAG.

See the complete [Technical Report](file:///home/rishang/Desktop/GraphRAG-LSEG/docs/technical_report.md) for full error taxonomies and the economic break-even model.

---

## 4. Quickstart Guide

### 1. Installation
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

### 2. Run Test Suite
```bash
pytest tests/ -v
# Output: 19 passed in ~4s
```

### 3. Launch Interactive Streamlit Demo
```bash
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) to explore:
* **Multi-Hop Query Explorer**: Live side-by-side comparison of B0, B1, S1, S2 with inline citations.
* **Knowledge Graph Inspector**: Browse the 83 nodes, 141 edges, and audit evidence spans.
* **Benchmark Dashboard**: Interactive charts of $F_1$ vs Hop Count.
* **Break-Even Cost Calculator**: Dynamic simulation of graph ROI.

### 4. Run Benchmark & Evaluation CLI
```bash
python scripts/run_evaluation.py
```

### 5. Rebuild Knowledge Graph
```bash
python scripts/build_graph.py
```

---

## 5. Repository Layout

```
LSEG-GraphRAG/
├── README.md                          # Project documentation and quickstart
├── pyproject.toml                     # Python package specifications
├── requirements.txt                   # Project dependencies
├── .env.example                       # Environment configuration template
├── config/
│   └── default.yaml                   # Company universe and schema definition
├── docs/
│   └── technical_report.md            # In-depth 5-page technical report
├── data/
│   ├── raw/                           # Raw 10-Ks and news sources
│   ├── processed/                     # Parsed sections and chunk index (chunks.json)
│   ├── graphs/                        # Serialized Knowledge Graph (financial_kg.json)
│   └── benchmark/                     # Stratified benchmark (multihop_benchmark.json)
├── src/
│   └── lseg_graphrag/
│       ├── config.py                  # Pydantic configuration loader
│       ├── schema/                    # Entities, relations, provenance models
│       ├── ingestion/                 # SEC 10-K parser & section-aware chunker
│       ├── extraction/                # Triple extractor & LLM/deterministic backends
│       ├── resolution/                # Normalizer & EntityResolver
│       ├── graph/                     # FinancialKnowledgeGraph & GraphTraverser
│       ├── retrieval/                 # Vector, Graph, and Hybrid retrievers
│       ├── generation/                # GroundedAnswerer with citations & abstain
│       └── evaluation/                # BenchmarkRunner, Metrics, and CostModel
├── tests/                             # Comprehensive unit & integration tests
├── app/
│   └── streamlit_app.py               # Interactive Streamlit demo dashboard
└── scripts/
    ├── build_graph.py                 # KG build entrypoint
    └── run_evaluation.py              # Evaluation benchmark entrypoint
```

---

## 6. License
Distributed under the MIT License. See [`LICENSE`](file:///home/rishang/Desktop/GraphRAG-LSEG/LICENSE) for details.
