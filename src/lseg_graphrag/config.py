"""Configuration loader and schema validator for LSEG-GraphRAG."""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from pydantic import BaseModel, Field


class CompanyConfig(BaseModel):
    symbol: str
    name: str
    cik: str
    tier: Optional[str] = None


class RelationSchemaConfig(BaseModel):
    name: str
    description: str
    head_types: List[str]
    tail_types: List[str]


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    respect_paragraphs: bool = True


class IngestionConfig(BaseModel):
    form_type: str = "10-K"
    sections: List[str] = ["item_1", "item_1a", "exhibit_21"]
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)


class ExtractionConfig(BaseModel):
    confidence_threshold: float = 0.70
    require_evidence_span: bool = True
    batch_size: int = 5


class ResolutionConfig(BaseModel):
    similarity_threshold: float = 0.85
    use_alias_dictionary: bool = True
    use_llm_disambiguation: bool = True


class VectorRetrievalConfig(BaseModel):
    top_k: int = 5
    similarity_metric: str = "cosine"


class GraphRetrievalConfig(BaseModel):
    max_hops: int = 3
    max_subgraph_nodes: int = 50
    direction: str = "both"


class HybridRetrievalConfig(BaseModel):
    top_k_chunks: int = 5
    graph_candidate_depth: int = 2
    graph_weight: float = 0.5
    vector_weight: float = 0.5


class RetrievalConfig(BaseModel):
    vector: VectorRetrievalConfig = Field(default_factory=VectorRetrievalConfig)
    graph: GraphRetrievalConfig = Field(default_factory=GraphRetrievalConfig)
    hybrid: HybridRetrievalConfig = Field(default_factory=HybridRetrievalConfig)


class CostModelConfig(BaseModel):
    prompt_token_cost_per_million: float = 0.150
    completion_token_cost_per_million: float = 0.600


class EvaluationConfig(BaseModel):
    hop_stratification: List[int] = [1, 2, 3]
    question_types: List[str] = [
        "direct_lookup",
        "chained_relation",
        "multi_hop_exposure",
        "aggregation",
        "path_reasoning",
    ]
    cost_model: CostModelConfig = Field(default_factory=CostModelConfig)


class AppConfig(BaseModel):
    project_name: str = "LSEG-GraphRAG"
    version: str = "0.1.0"
    universe_sector: str = "Semiconductors"
    companies: List[CompanyConfig] = Field(default_factory=list)
    schema_relations: List[RelationSchemaConfig] = Field(default_factory=list)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def load_from_yaml(cls, path: str = "config/default.yaml") -> "AppConfig":
        """Load configuration from YAML file, with fallbacks."""
        config_path = Path(path)
        if not config_path.is_file():
            # Try finding it relative to project root
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / path

        if not config_path.is_file():
            # Return default if not found
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            project_name=raw.get("project", {}).get("name", "LSEG-GraphRAG"),
            version=raw.get("project", {}).get("version", "0.1.0"),
            universe_sector=raw.get("universe", {}).get("sector", "Semiconductors"),
            companies=[CompanyConfig(**c) for c in raw.get("universe", {}).get("companies", [])],
            schema_relations=[
                RelationSchemaConfig(**r) for r in raw.get("schema", {}).get("relation_types", [])
            ],
            ingestion=IngestionConfig(**raw.get("ingestion", {})),
            extraction=ExtractionConfig(**raw.get("extraction", {})),
            resolution=ResolutionConfig(**raw.get("resolution", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )


# Global default configuration instance
default_config = AppConfig.load_from_yaml()
