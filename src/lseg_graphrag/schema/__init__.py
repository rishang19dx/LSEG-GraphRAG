"""Schema package exposing entity, relation, and provenance definitions."""

from lseg_graphrag.schema.entities import EntityType, EntityNode
from lseg_graphrag.schema.relations import RelationType, Triple, ExtractedTriple, ExtractionResult
from lseg_graphrag.schema.provenance import DocumentMetadata, ChunkMetadata, Citation

__all__ = [
    "EntityType",
    "EntityNode",
    "RelationType",
    "Triple",
    "ExtractedTriple",
    "ExtractionResult",
    "DocumentMetadata",
    "ChunkMetadata",
    "Citation",
]
