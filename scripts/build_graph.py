"""CLI pipeline script to extract triples from corpus and build the Knowledge Graph."""

import json
from pathlib import Path
from lseg_graphrag.schema.provenance import ChunkMetadata
from lseg_graphrag.extraction.extractor import TripleExtractor
from lseg_graphrag.extraction.llm_backends import RuleBasedFinancialExtractor
from lseg_graphrag.resolution.resolver import EntityResolver
from lseg_graphrag.graph.store import FinancialKnowledgeGraph


def build_knowledge_graph_from_corpus(
    chunks_path: str = "data/processed/chunks.json",
    output_graph_path: str = "data/graphs/financial_kg.json",
) -> FinancialKnowledgeGraph:
    """Load chunks, extract grounded triples, resolve entities, and persist knowledge graph."""
    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = [ChunkMetadata(**c) for c in chunks_data]
    print(f"Loaded {len(chunks)} chunks.")

    extractor = TripleExtractor(backend=RuleBasedFinancialExtractor(), confidence_threshold=0.70)
    resolver = EntityResolver()
    kg = FinancialKnowledgeGraph()

    total_triples_extracted = 0
    print("Extracting grounded triples and building Knowledge Graph...")
    for chunk in chunks:
        res = extractor.extract_from_chunk(chunk)
        for triple in res.triples:
            head_node, tail_node = resolver.resolve_triple(triple)
            kg.add_triple(head_node, tail_node, triple)
            total_triples_extracted += 1

    stats = kg.get_stats()
    print("\n=== Knowledge Graph Build Complete ===")
    print(f"Nodes in Graph: {stats['num_nodes']}")
    print(f"Edges in Graph: {stats['num_edges']}")
    print(f"Total Validated Triples Added: {total_triples_extracted}")
    print("\nRelations Distribution:")
    for rel, count in stats["relation_distribution"].items():
        print(f"  - {rel}: {count}")

    print("\nNode Types Distribution:")
    for ntype, count in stats["node_type_distribution"].items():
        print(f"  - {ntype}: {count}")

    kg.save_to_json(output_graph_path)
    print(f"\nPersisted Knowledge Graph to {output_graph_path}")
    return kg


if __name__ == "__main__":
    build_knowledge_graph_from_corpus()
