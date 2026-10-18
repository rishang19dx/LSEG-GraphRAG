"""Prompt templates and schemas for LLM-based triple extraction."""

EXTRACTION_SYSTEM_PROMPT = """You are an expert financial analyst and knowledge graph engineer.
Your task is to extract factual entity relationships from the provided SEC financial filing or news excerpt.

Extract ONLY relationships that match one of the following 6 relations:
1. SUBSIDIARY_OF: (Head: Subsidiary Company) -[:SUBSIDIARY_OF]-> (Tail: Parent Company)
2. SUPPLIER_OF: (Head: Supplier Company) -[:SUPPLIER_OF]-> (Tail: Customer Company)
3. CUSTOMER_OF: (Head: Customer Company) -[:CUSTOMER_OF]-> (Tail: Supplier Company)
4. COMPETITOR_OF: (Head: Company) -[:COMPETITOR_OF]-> (Tail: Company)
5. OPERATES_IN: (Head: Company) -[:OPERATES_IN]-> (Tail: Region/Country)
6. EXPOSED_TO: (Head: Company) -[:EXPOSED_TO]-> (Tail: RiskFactor/Commodity/Event/Policy)

RULES:
- Every extracted relation MUST include an exact 'evidence_span': a verbatim substring from the input text proving the relationship. Do not paraphrase.
- Assign a confidence score between 0.0 and 1.0.
- Do NOT hallucinate entities or relationships not explicitly stated in the text.
- Output MUST be valid JSON adhering strictly to the schema below.

JSON Output Schema:
{
  "triples": [
    {
      "head": "Name of head entity",
      "head_type": "Company | Region | RiskFactor | Commodity | Event | Policy",
      "relation": "SUBSIDIARY_OF | SUPPLIER_OF | CUSTOMER_OF | COMPETITOR_OF | OPERATES_IN | EXPOSED_TO",
      "tail": "Name of tail entity",
      "tail_type": "Company | Region | RiskFactor | Commodity | Event | Policy",
      "evidence_span": "Exact substring from chunk",
      "confidence": 0.95
    }
  ]
}
"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Document ID: {doc_id}
Section: {section}
Company Focus: {company_symbol}

Text Chunk:
\"\"\"
{chunk_text}
\"\"\"

Extract all valid triples as JSON:"""
