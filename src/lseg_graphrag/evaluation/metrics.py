"""Evaluation metrics for answer quality, retrieval quality, and faithfulness."""

import re
from typing import List, Set, Dict, Any, Optional
from difflib import SequenceMatcher
from lseg_graphrag.resolution.normalizer import normalize_company_name
from lseg_graphrag.schema.provenance import Citation


def normalize_entity_set(entities: List[str]) -> Set[str]:
    """Normalize a list of entity mentions into a normalized set."""
    norm_set = set()
    for e in entities:
        cleaned = normalize_company_name(e)
        if cleaned:
            norm_set.add(cleaned)
    return norm_set


def compute_set_metrics(predicted_entities: List[str], gold_entities: List[str]) -> Dict[str, float]:
    """Compute set-based Precision, Recall, F1, and Exact Match against gold answers."""
    pred_set = normalize_entity_set(predicted_entities)
    gold_set = normalize_entity_set(gold_entities)

    if not gold_set:
        return {
            "precision": 1.0 if not pred_set else 0.0,
            "recall": 1.0 if not pred_set else 0.0,
            "f1": 1.0 if not pred_set else 0.0,
            "exact_match": 1.0 if not pred_set else 0.0,
        }

    if not pred_set:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
        }

    # Soft entity matching (handles slight lexical variations)
    true_positives = 0
    matched_gold = set()

    for p in pred_set:
        for g in gold_set:
            if g not in matched_gold:
                if p == g or p in g or g in p or SequenceMatcher(None, p, g).ratio() > 0.82:
                    true_positives += 1
                    matched_gold.add(g)
                    break

    precision = true_positives / len(pred_set)
    recall = true_positives / len(gold_set)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if pred_set == gold_set else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": round(exact_match, 4),
    }


def compute_retrieval_metrics(retrieved_context: str, gold_facts: List[str]) -> Dict[str, float]:
    """Compute supporting-fact recall and context precision."""
    if not gold_facts:
        return {"supporting_fact_recall": 1.0, "context_precision": 1.0}

    ctx_lower = retrieved_context.lower()
    facts_found = 0

    for fact in gold_facts:
        tokens = [t.lower() for t in fact.split() if len(t) > 3]
        if not tokens:
            continue
        # If at least 70% of distinctive fact tokens appear in context
        token_hits = sum(1 for tok in tokens if tok in ctx_lower)
        if token_hits / len(tokens) >= 0.70:
            facts_found += 1

    fact_recall = facts_found / len(gold_facts)

    # Context precision: ratio of relevant facts relative to context length (in 500-char blocks)
    num_blocks = max(1, len(retrieved_context) // 500)
    ctx_precision = min(1.0, facts_found / num_blocks) if facts_found > 0 else 0.0

    return {
        "supporting_fact_recall": round(fact_recall, 4),
        "context_precision": round(ctx_precision, 4),
    }


def compute_faithfulness_metrics(
    answer_text: str,
    retrieved_context: str,
    citations: List[Citation],
) -> Dict[str, float]:
    """Compute hallucination rate and citation correctness score."""
    if not answer_text or "ABSTAIN" in answer_text:
        return {"hallucination_rate": 0.0, "citation_correctness": 1.0}

    # Citation correctness: check if citation evidence spans actually appear in retrieved context
    valid_citations = 0
    for c in citations:
        span_part = c.evidence_span[:40].lower()
        if span_part in retrieved_context.lower() or span_part in answer_text.lower():
            valid_citations += 1

    citation_correctness = (valid_citations / len(citations)) if citations else 1.0

    # Hallucination estimate: ratio of named entities in answer not present in context
    extracted_norm = normalize_entity_set(re.findall(r"\b[A-Z][a-zA-Z0-9\-]+\b", answer_text))
    # Exclude common false positives
    stop_words = {"based", "the", "entities", "relevant", "insufficient", "abstain", "company", "evidence"}
    filtered_ents = {e for e in extracted_norm if e not in stop_words}

    if not filtered_ents:
        hallucination_rate = 0.0
    else:
        unsupported = sum(1 for e in filtered_ents if e not in retrieved_context.lower())
        hallucination_rate = unsupported / len(filtered_ents)

    return {
        "hallucination_rate": round(hallucination_rate, 4),
        "citation_correctness": round(citation_correctness, 4),
    }
