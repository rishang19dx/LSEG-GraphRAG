"""Entity data models and types for the Financial Knowledge Graph."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    COMPANY = "Company"
    SUBSIDIARY = "Subsidiary"
    REGION = "Region"
    COUNTRY = "Country"
    RISK_FACTOR = "RiskFactor"
    COMMODITY = "Commodity"
    EVENT = "Event"
    REGULATORY_POLICY = "RegulatoryPolicy"
    INDUSTRY = "Industry"


class EntityNode(BaseModel):
    """Represents a normalized canonical node in the knowledge graph."""
    id: str = Field(..., description="Unique canonical identifier, e.g. 'company:nvda' or 'region:taiwan'")
    name: str = Field(..., description="Canonical display name")
    entity_type: EntityType = Field(default=EntityType.COMPANY, description="Category of the entity")
    aliases: List[str] = Field(default_factory=list, description="Known aliases, ticker symbols, or variants")
    ticker: Optional[str] = Field(default=None, description="Stock ticker symbol if applicable")
    cik: Optional[str] = Field(default=None, description="SEC Central Index Key if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g. tier, sector)")

    def matches_mention(self, mention: str) -> bool:
        """Check if a text mention matches the entity name or any alias."""
        m_lower = mention.strip().lower()
        if m_lower == self.name.lower():
            return True
        if self.ticker and m_lower == self.ticker.lower():
            return True
        return any(m_lower == alias.lower() for alias in self.aliases)
