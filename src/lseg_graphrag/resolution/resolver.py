"""Entity resolution engine linking mentions to canonical entities."""

from typing import Dict, List, Optional, Tuple, Set
from difflib import SequenceMatcher
from lseg_graphrag.schema.entities import EntityNode, EntityType
from lseg_graphrag.schema.relations import ExtractedTriple
from lseg_graphrag.resolution.normalizer import normalize_company_name


class EntityResolver:
    """Resolves entity mentions to canonical entities using dictionaries and fuzzy similarity."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.canonical_entities: Dict[str, EntityNode] = {}
        self.mention_to_id: Dict[str, str] = {}
        self._initialize_seed_entities()

    def _initialize_seed_entities(self):
        """Seed canonical entities for the core semiconductor and electronics ecosystem."""
        seed_data = [
            EntityNode(
                id="company:nvda",
                name="NVIDIA Corporation",
                entity_type=EntityType.COMPANY,
                aliases=["NVIDIA", "Nvidia Corp", "NVDA", "Nvidia"],
                ticker="NVDA",
                cik="0001045810",
                metadata={"tier": "Fabless Chip Designer", "sector": "Semiconductors"},
            ),
            EntityNode(
                id="company:tsm",
                name="Taiwan Semiconductor Manufacturing Company",
                entity_type=EntityType.COMPANY,
                aliases=["TSMC", "Taiwan Semi", "TSMC Limited", "TSM", "Taiwan Semiconductor"],
                ticker="TSM",
                cik="0001046179",
                metadata={"tier": "Pure-play Foundry", "sector": "Semiconductors"},
            ),
            EntityNode(
                id="company:aapl",
                name="Apple Inc.",
                entity_type=EntityType.COMPANY,
                aliases=["Apple", "Apple Computer", "AAPL"],
                ticker="AAPL",
                cik="0000320193",
                metadata={"tier": "OEM / Consumer Electronics", "sector": "Technology"},
            ),
            EntityNode(
                id="company:asml",
                name="ASML Holding N.V.",
                entity_type=EntityType.COMPANY,
                aliases=["ASML", "ASML Holding", "ASML NV"],
                ticker="ASML",
                cik="0000937966",
                metadata={"tier": "Lithography Equipment", "sector": "Semiconductor Equipment"},
            ),
            EntityNode(
                id="company:amd",
                name="Advanced Micro Devices, Inc.",
                entity_type=EntityType.COMPANY,
                aliases=["AMD", "Advanced Micro Devices"],
                ticker="AMD",
                cik="0000002488",
                metadata={"tier": "Fabless Chip Designer", "sector": "Semiconductors"},
            ),
            EntityNode(
                id="company:intc",
                name="Intel Corporation",
                entity_type=EntityType.COMPANY,
                aliases=["Intel", "Intel Corp", "INTC"],
                ticker="INTC",
                cik="0000050863",
                metadata={"tier": "Integrated Device Manufacturer", "sector": "Semiconductors"},
            ),
            EntityNode(
                id="company:qcom",
                name="Qualcomm Incorporated",
                entity_type=EntityType.COMPANY,
                aliases=["Qualcomm", "QCOM"],
                ticker="QCOM",
                cik="0000804328",
                metadata={"tier": "Fabless Communications", "sector": "Semiconductors"},
            ),
            EntityNode(
                id="company:mu",
                name="Micron Technology, Inc.",
                entity_type=EntityType.COMPANY,
                aliases=["Micron", "Micron Technology", "MU"],
                ticker="MU",
                cik="0000723125",
                metadata={"tier": "Memory & Storage", "sector": "Semiconductors"},
            ),
            EntityNode(
                id="company:amat",
                name="Applied Materials, Inc.",
                entity_type=EntityType.COMPANY,
                aliases=["Applied Materials", "AMAT"],
                ticker="AMAT",
                cik="0000006951",
                metadata={"tier": "Semiconductor Equipment", "sector": "Semiconductor Equipment"},
            ),
            EntityNode(
                id="company:lrcx",
                name="Lam Research Corporation",
                entity_type=EntityType.COMPANY,
                aliases=["Lam Research", "LRCX"],
                ticker="LRCX",
                cik="0000707549",
                metadata={"tier": "Semiconductor Equipment", "sector": "Semiconductor Equipment"},
            ),
            EntityNode(
                id="company:foxconn",
                name="Hon Hai Precision Industry (Foxconn)",
                entity_type=EntityType.COMPANY,
                aliases=["Foxconn", "Hon Hai", "Hon Hai Precision Industry", "Foxconn Technology Group"],
                ticker="2317.TW",
                cik=None,
                metadata={"tier": "Contract Electronics Manufacturer", "sector": "Manufacturing"},
            ),
            EntityNode(
                id="company:carl_zeiss",
                name="Carl Zeiss SMT GmbH",
                entity_type=EntityType.COMPANY,
                aliases=["Carl Zeiss", "Zeiss SMT", "Carl Zeiss SMT"],
                ticker=None,
                cik=None,
                metadata={"tier": "Optics Supplier", "sector": "Precision Optics"},
            ),
            EntityNode(
                id="region:taiwan",
                name="Taiwan",
                entity_type=EntityType.REGION,
                aliases=["ROC", "Republic of China", "Hsinchu", "Taichung", "Tainan"],
            ),
            EntityNode(
                id="region:united_states",
                name="United States",
                entity_type=EntityType.REGION,
                aliases=["USA", "U.S.", "United States of America", "US"],
            ),
            EntityNode(
                id="region:china",
                name="China",
                entity_type=EntityType.REGION,
                aliases=["PRC", "People's Republic of China", "Shanghai", "Shenzhen", "Zhengzhou"],
            ),
            EntityNode(
                id="region:netherlands",
                name="Netherlands",
                entity_type=EntityType.REGION,
                aliases=["The Netherlands", "Holland", "Veldhoven"],
            ),
            EntityNode(
                id="risk:taiwan_geopolitical",
                name="Taiwan Geopolitical Risk",
                entity_type=EntityType.RISK_FACTOR,
                aliases=["Cross-strait tensions", "Taiwan risk"],
            ),
            EntityNode(
                id="risk:natural_disaster",
                name="Natural Disaster",
                entity_type=EntityType.RISK_FACTOR,
                aliases=["Earthquake", "Seismic event", "Typhoon", "Hualien earthquake"],
            ),
            EntityNode(
                id="risk:us_export_controls",
                name="US BIS Export Restrictions",
                entity_type=EntityType.REGULATORY_POLICY,
                aliases=["Export control", "BIS restrictions", "Commerce export controls", "Trade restrictions"],
            ),
            EntityNode(
                id="risk:cowos_packaging",
                name="CoWoS Packaging Bottleneck",
                entity_type=EntityType.RISK_FACTOR,
                aliases=["Advanced packaging shortage", "CoWoS capacity", "Packaging bottleneck"],
            ),
        ]

        for entity in seed_data:
            self.register_entity(entity)

    def register_entity(self, entity: EntityNode):
        """Register a canonical entity and index its aliases."""
        self.canonical_entities[entity.id] = entity
        self.mention_to_id[entity.name.lower()] = entity.id
        if entity.ticker:
            self.mention_to_id[entity.ticker.lower()] = entity.id
        for alias in entity.aliases:
            self.mention_to_id[alias.lower()] = entity.id

    def resolve(self, mention: str, expected_type: str = "Company") -> EntityNode:
        """Resolve an entity mention to a canonical EntityNode."""
        mention_str = mention.strip()
        m_lower = mention_str.lower()

        # 1. Exact alias / ticker match
        if m_lower in self.mention_to_id:
            return self.canonical_entities[self.mention_to_id[m_lower]]

        # 2. Normalized company name match
        norm = normalize_company_name(mention_str)
        if norm:
            for entity in self.canonical_entities.values():
                if norm == normalize_company_name(entity.name):
                    self.mention_to_id[m_lower] = entity.id
                    return entity
                for alias in entity.aliases:
                    if norm == normalize_company_name(alias):
                        self.mention_to_id[m_lower] = entity.id
                        return entity

        # 3. Fuzzy similarity match
        best_score = 0.0
        best_match: Optional[EntityNode] = None

        for entity in self.canonical_entities.values():
            # Check against name
            score = SequenceMatcher(None, m_lower, entity.name.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = entity

            # Check against aliases
            for alias in entity.aliases:
                alias_score = SequenceMatcher(None, m_lower, alias.lower()).ratio()
                if alias_score > best_score:
                    best_score = alias_score
                    best_match = entity

        if best_match and best_score >= self.similarity_threshold:
            self.mention_to_id[m_lower] = best_match.id
            return best_match

        # 4. Unknown entity: generate new canonical entity dynamically
        clean_id_suffix = norm.replace(" ", "_") if norm else m_lower.replace(" ", "_")
        prefix = expected_type.lower()
        new_id = f"{prefix}:{clean_id_suffix}"

        entity_type_enum = EntityType.COMPANY
        try:
            entity_type_enum = EntityType(expected_type)
        except ValueError:
            pass

        new_entity = EntityNode(
            id=new_id,
            name=mention_str,
            entity_type=entity_type_enum,
            aliases=[mention_str],
        )
        self.register_entity(new_entity)
        return new_entity

    def resolve_triple(self, triple: ExtractedTriple) -> Tuple[EntityNode, EntityNode]:
        """Resolve head and tail of an extracted triple to canonical EntityNodes."""
        head_node = self.resolve(triple.head, expected_type=triple.head_type)
        tail_node = self.resolve(triple.tail, expected_type=triple.tail_type)
        return head_node, tail_node
