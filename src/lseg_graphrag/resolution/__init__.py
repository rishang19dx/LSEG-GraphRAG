"""Entity resolution package."""

from lseg_graphrag.resolution.normalizer import normalize_company_name
from lseg_graphrag.resolution.resolver import EntityResolver

__all__ = [
    "normalize_company_name",
    "EntityResolver",
]
