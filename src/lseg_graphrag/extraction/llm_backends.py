"""LLM and deterministic rule-based backends for knowledge graph extraction."""

import os
import re
import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from lseg_graphrag.schema.relations import ExtractedTriple, RelationType
from lseg_graphrag.schema.provenance import ChunkMetadata
from lseg_graphrag.extraction.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT_TEMPLATE


class BaseLLMBackend(ABC):
    """Abstract interface for triple extraction backends."""

    @abstractmethod
    def extract(self, chunk: ChunkMetadata) -> Tuple[List[ExtractedTriple], int, float]:
        """Extract triples from a chunk. Returns (triples, token_count, latency_ms)."""
        pass


class OpenAIBackend(BaseLLMBackend):
    """Extraction backend using OpenAI or OpenAI-compatible API (Groq, Ollama, vLLM)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None
        except ImportError:
            self.client = None

    def extract(self, chunk: ChunkMetadata) -> Tuple[List[ExtractedTriple], int, float]:
        if not self.client:
            raise ValueError("OpenAI client is not initialized or OPENAI_API_KEY is missing.")

        user_content = EXTRACTION_USER_PROMPT_TEMPLATE.format(
            doc_id=chunk.doc_id,
            section=chunk.section,
            company_symbol=chunk.company_symbol,
            chunk_text=chunk.text,
        )

        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        latency_ms = (time.time() - start_time) * 1000.0
        tokens = response.usage.total_tokens if response.usage else 0

        raw_json = response.choices[0].message.content
        data = json.loads(raw_json)
        raw_triples = data.get("triples", [])

        extracted: List[ExtractedTriple] = []
        for r in raw_triples:
            try:
                rel = RelationType(r["relation"].strip().upper())
                extracted.append(
                    ExtractedTriple(
                        head=r["head"].strip(),
                        head_type=r.get("head_type", "Company"),
                        relation=rel,
                        tail=r["tail"].strip(),
                        tail_type=r.get("tail_type", "Company"),
                        evidence_span=r["evidence_span"].strip(),
                        confidence=float(r.get("confidence", 0.9)),
                        source_doc_id=chunk.doc_id,
                        source_section=chunk.section,
                        chunk_id=chunk.chunk_id,
                        filing_date=chunk.metadata.get("filing_date"),
                    )
                )
            except Exception:
                continue

        return extracted, tokens, latency_ms


class RuleBasedFinancialExtractor(BaseLLMBackend):
    """Deterministic, high-precision extractor for financial filings and Exhibit 21 tables."""

    def __init__(self):
        # Known company entity map for high precision recognition
        self.known_companies = {
            "nvidia": "NVIDIA",
            "nvda": "NVIDIA",
            "taiwan semiconductor manufacturing company": "TSMC",
            "tsmc": "TSMC",
            "apple": "Apple",
            "asml": "ASML",
            "advanced micro devices": "AMD",
            "amd": "AMD",
            "intel": "Intel",
            "qualcomm": "Qualcomm",
            "broadcom": "Broadcom",
            "micron technology": "Micron",
            "micron": "Micron",
            "applied materials": "Applied Materials",
            "lam research": "Lam Research",
            "kla": "KLA Corporation",
            "carl zeiss": "Carl Zeiss",
            "foxconn": "Foxconn",
            "hon hai": "Foxconn",
            "sk hynix": "SK Hynix",
            "samsung electronics": "Samsung Electronics",
            "samsung": "Samsung Electronics",
            "microsoft": "Microsoft",
            "amazon": "Amazon",
            "alphabet": "Alphabet",
            "meta platforms": "Meta",
        }

    def extract(self, chunk: ChunkMetadata) -> Tuple[List[ExtractedTriple], int, float]:
        start_time = time.time()
        triples: List[ExtractedTriple] = []
        text = chunk.text

        # 1. Exhibit 21 Subsidiary Extraction
        if "exhibit_21" in chunk.section.lower():
            subs_triples = self._extract_subsidiaries(chunk)
            triples.extend(subs_triples)

        # 2. Supplier / Customer Extraction
        supply_triples = self._extract_supply_relationships(chunk)
        triples.extend(supply_triples)

        # 3. Competitor Extraction
        comp_triples = self._extract_competitors(chunk)
        triples.extend(comp_triples)

        # 4. Operates In (Geographic Locations)
        geo_triples = self._extract_operations(chunk)
        triples.extend(geo_triples)

        # 5. Risk / Commodity / Event Exposure
        risk_triples = self._extract_exposures(chunk)
        triples.extend(risk_triples)

        latency_ms = (time.time() - start_time) * 1000.0
        # Simulated token count
        token_count = max(1, len(chunk.text) // 4)

        return triples, token_count, latency_ms

    def _extract_subsidiaries(self, chunk: ChunkMetadata) -> List[ExtractedTriple]:
        triples = []
        parent_name = chunk.company_symbol

        # Match lines like "1. Mellanox Technologies, Ltd. - Jurisdiction: Israel"
        lines = chunk.text.splitlines()
        for line in lines:
            sub_match = re.search(r"^\s*(?:\d+[\.\)]|\-)\s*([^–\-]+?)\s*(?:–|-|\()\s*Jurisdiction:?\s*([^,\)\n]+)", line, re.I)
            if sub_match:
                sub_name = sub_match.group(1).strip()
                country = sub_match.group(2).strip()
                span = line.strip()

                # Subsidiary of parent
                triples.append(
                    ExtractedTriple(
                        head=sub_name,
                        head_type="Subsidiary",
                        relation=RelationType.SUBSIDIARY_OF,
                        tail=parent_name,
                        tail_type="Company",
                        evidence_span=span,
                        confidence=0.98,
                        source_doc_id=chunk.doc_id,
                        source_section=chunk.section,
                        chunk_id=chunk.chunk_id,
                        filing_date=chunk.metadata.get("filing_date"),
                    )
                )

                # Subsidiary operates in country
                if country:
                    triples.append(
                        ExtractedTriple(
                            head=sub_name,
                            head_type="Subsidiary",
                            relation=RelationType.OPERATES_IN,
                            tail=country,
                            tail_type="Country",
                            evidence_span=span,
                            confidence=0.95,
                            source_doc_id=chunk.doc_id,
                            source_section=chunk.section,
                            chunk_id=chunk.chunk_id,
                            filing_date=chunk.metadata.get("filing_date"),
                        )
                    )
        return triples

    def _extract_supply_relationships(self, chunk: ChunkMetadata) -> List[ExtractedTriple]:
        triples = []
        text = chunk.text
        reporting_co = self.known_companies.get(chunk.company_symbol.lower(), chunk.company_symbol)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sent in sentences:
            s_lower = sent.lower()

            # Supplier patterns
            # e.g., "TSMC manufactures our advanced semiconductor wafers..." or "ASML supplies TSMC"
            if any(w in s_lower for w in ["manufactures", "supplies", "supplier", "partner", "procure", "outsource"]):
                found_companies = []
                for cand, name in self.known_companies.items():
                    if cand in s_lower and name not in found_companies:
                        found_companies.append(name)

                # Case A: Two or more explicit companies mentioned in the sentence
                if len(found_companies) >= 2:
                    for i in range(len(found_companies)):
                        for j in range(len(found_companies)):
                            if i != j:
                                head_name, tail_name = found_companies[i], found_companies[j]
                                # Check direction indicators
                                if any(w in s_lower for w in ["manufactures for", "supplier of", "supplier to", "supplies", "partnering with", "relies on", "rely on"]):
                                    triples.append(
                                        ExtractedTriple(
                                            head=head_name,
                                            head_type="Company",
                                            relation=RelationType.SUPPLIER_OF,
                                            tail=tail_name,
                                            tail_type="Company",
                                            evidence_span=sent.strip(),
                                            confidence=0.92,
                                            source_doc_id=chunk.doc_id,
                                            source_section=chunk.section,
                                            chunk_id=chunk.chunk_id,
                                            filing_date=chunk.metadata.get("filing_date"),
                                        )
                                    )

                # Case B: One external company mentioned alongside first-person filer reference ("our", "we", "us")
                elif len(found_companies) == 1:
                    partner = found_companies[0]
                    if partner != reporting_co and any(w in s_lower for w in ["our", "we", "us"]):
                        # If "partner manufactures our..." or "we rely on partner for..."
                        if any(w in s_lower for w in ["manufactures our", "supplier", "supplies", "rely on", "relies on", "partnering with", "procure from", "outsource"]):
                            triples.append(
                                ExtractedTriple(
                                    head=partner,
                                    head_type="Company",
                                    relation=RelationType.SUPPLIER_OF,
                                    tail=reporting_co,
                                    tail_type="Company",
                                    evidence_span=sent.strip(),
                                    confidence=0.94,
                                    source_doc_id=chunk.doc_id,
                                    source_section=chunk.section,
                                    chunk_id=chunk.chunk_id,
                                    filing_date=chunk.metadata.get("filing_date"),
                                )
                            )

            # Customer patterns: e.g. "Our customers include Apple, NVIDIA..."
            if "customer" in s_lower:
                reporting_co = self.known_companies.get(chunk.company_symbol.lower(), chunk.company_symbol)
                for cust_cand, cust_name in self.known_companies.items():
                    if cust_cand in s_lower and cust_name != reporting_co:
                        triples.append(
                            ExtractedTriple(
                                head=cust_name,
                                head_type="Company",
                                relation=RelationType.CUSTOMER_OF,
                                tail=reporting_co,
                                tail_type="Company",
                                evidence_span=sent.strip(),
                                confidence=0.93,
                                source_doc_id=chunk.doc_id,
                                source_section=chunk.section,
                                chunk_id=chunk.chunk_id,
                                filing_date=chunk.metadata.get("filing_date"),
                            )
                        )
                        # Symmetrical supplier relation
                        triples.append(
                            ExtractedTriple(
                                head=reporting_co,
                                head_type="Company",
                                relation=RelationType.SUPPLIER_OF,
                                tail=cust_name,
                                tail_type="Company",
                                evidence_span=sent.strip(),
                                confidence=0.93,
                                source_doc_id=chunk.doc_id,
                                source_section=chunk.section,
                                chunk_id=chunk.chunk_id,
                                filing_date=chunk.metadata.get("filing_date"),
                            )
                        )
        return triples

    def _extract_competitors(self, chunk: ChunkMetadata) -> List[ExtractedTriple]:
        triples = []
        reporting_co = self.known_companies.get(chunk.company_symbol.lower(), chunk.company_symbol)
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)

        for sent in sentences:
            s_lower = sent.lower()
            if "compete" in s_lower or "competition" in s_lower or "competitor" in s_lower:
                for comp_cand, comp_name in self.known_companies.items():
                    if comp_cand in s_lower and comp_name != reporting_co:
                        triples.append(
                            ExtractedTriple(
                                head=reporting_co,
                                head_type="Company",
                                relation=RelationType.COMPETITOR_OF,
                                tail=comp_name,
                                tail_type="Company",
                                evidence_span=sent.strip(),
                                confidence=0.90,
                                source_doc_id=chunk.doc_id,
                                source_section=chunk.section,
                                chunk_id=chunk.chunk_id,
                                filing_date=chunk.metadata.get("filing_date"),
                            )
                        )
        return triples

    def _extract_operations(self, chunk: ChunkMetadata) -> List[ExtractedTriple]:
        triples = []
        reporting_co = self.known_companies.get(chunk.company_symbol.lower(), chunk.company_symbol)
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)

        regions = ["Taiwan", "United States", "China", "Netherlands", "Japan", "Germany", "Israel", "India", "Ireland"]
        for sent in sentences:
            s_lower = sent.lower()
            if any(w in s_lower for w in ["located in", "facilities in", "fabs in", "manufacturing in", "operations in", "campus in"]):
                for reg in regions:
                    if reg.lower() in s_lower:
                        triples.append(
                            ExtractedTriple(
                                head=reporting_co,
                                head_type="Company",
                                relation=RelationType.OPERATES_IN,
                                tail=reg,
                                tail_type="Region",
                                evidence_span=sent.strip(),
                                confidence=0.88,
                                source_doc_id=chunk.doc_id,
                                source_section=chunk.section,
                                chunk_id=chunk.chunk_id,
                                filing_date=chunk.metadata.get("filing_date"),
                            )
                        )
        return triples

    def _extract_exposures(self, chunk: ChunkMetadata) -> List[ExtractedTriple]:
        triples = []
        reporting_co = self.known_companies.get(chunk.company_symbol.lower(), chunk.company_symbol)
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)

        risk_keywords = {
            "earthquake": ("Natural Disaster", "RiskFactor"),
            "seismic": ("Natural Disaster", "RiskFactor"),
            "typhoon": ("Natural Disaster", "RiskFactor"),
            "export control": ("US BIS Export Restrictions", "RegulatoryPolicy"),
            "export restriction": ("US BIS Export Restrictions", "RegulatoryPolicy"),
            "taiwan": ("Taiwan Geopolitical Risk", "RiskFactor"),
            "china": ("China Trade Restrictions", "RegulatoryPolicy"),
            "packaging": ("CoWoS Packaging Bottleneck", "RiskFactor"),
            "neon": ("Neon Gas Supply Bottleneck", "Commodity"),
            "silicon": ("Silicon Wafer Shortage", "Commodity"),
        }

        for sent in sentences:
            s_lower = sent.lower()
            if any(w in s_lower for w in ["exposed to", "disruption", "risk", "shortage", "restriction", "susceptible"]):
                for kw, (risk_label, r_type) in risk_keywords.items():
                    if kw in s_lower:
                        triples.append(
                            ExtractedTriple(
                                head=reporting_co,
                                head_type="Company",
                                relation=RelationType.EXPOSED_TO,
                                tail=risk_label,
                                tail_type=r_type,
                                evidence_span=sent.strip(),
                                confidence=0.85,
                                source_doc_id=chunk.doc_id,
                                source_section=chunk.section,
                                chunk_id=chunk.chunk_id,
                                filing_date=chunk.metadata.get("filing_date"),
                            )
                        )
        return triples
