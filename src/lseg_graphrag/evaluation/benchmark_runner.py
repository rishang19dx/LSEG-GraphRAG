"""Benchmark execution engine evaluating B0, B1, S1, S2 and computing stratifications & ablations."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from lseg_graphrag.schema.provenance import ChunkMetadata
from lseg_graphrag.graph.store import FinancialKnowledgeGraph
from lseg_graphrag.retrieval.vector_rag import VectorRetriever
from lseg_graphrag.retrieval.graph_rag import GraphRetriever
from lseg_graphrag.retrieval.hybrid_rag import HybridRetriever
from lseg_graphrag.generation.answerer import GroundedAnswerer, AnswerOutput
from lseg_graphrag.evaluation.metrics import (
    compute_set_metrics,
    compute_retrieval_metrics,
    compute_faithfulness_metrics,
)
from lseg_graphrag.evaluation.cost_model import CostModel


class BenchmarkRunner:
    """Orchestrates comprehensive comparative evaluation across all retrieval architectures."""

    def __init__(
        self,
        kg_path: str = "data/graphs/financial_kg.json",
        chunks_path: str = "data/processed/chunks.json",
        benchmark_path: str = "data/benchmark/multihop_benchmark.json",
    ):
        self.kg = FinancialKnowledgeGraph.load_from_json(kg_path)

        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
        self.chunks = [ChunkMetadata(**c) for c in raw_chunks]

        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_questions = json.load(f)

        # Initialize retrievers
        self.vector_retriever = VectorRetriever(chunks=self.chunks, use_reranker=True)
        self.graph_retriever = GraphRetriever(kg=self.kg)
        self.hybrid_retriever = HybridRetriever(
            graph_retriever=self.graph_retriever,
            vector_retriever=self.vector_retriever,
        )

        self.answerer = GroundedAnswerer()
        self.cost_model = CostModel()

    def run_all(self, systems: List[str] = ["B0", "B1", "S1", "S2"]) -> Dict[str, Any]:
        """Execute evaluation over all benchmark questions across specified systems."""
        records: List[Dict[str, Any]] = []

        for q in self.benchmark_questions:
            qid = q["id"]
            question_text = q["question"]
            hop_count = q["hop_count"]
            q_type = q["question_type"]
            gold_entities = q["gold_entities"]
            gold_facts = q["gold_facts"]

            for sys_id in systems:
                start_t = time.time()
                context = ""
                citations = []

                if sys_id == "B0":
                    context = ""
                    citations = []
                elif sys_id == "B1":
                    chunks_retrieved, citations = self.vector_retriever.retrieve(question_text, top_k=3)
                    context = "\n\n".join([c.text for c in chunks_retrieved])
                elif sys_id == "S1":
                    context, citations, _ = self.graph_retriever.retrieve(question_text, max_hops=2)
                elif sys_id == "S2":
                    context, citations, _ = self.hybrid_retriever.retrieve(question_text, max_hops=2, top_k_chunks=2)

                ans: AnswerOutput = self.answerer.generate_answer(
                    system_id=sys_id,
                    question=question_text,
                    context=context,
                    citations=citations,
                )
                total_latency = (time.time() - start_t) * 1000.0

                # Compute Metrics
                set_m = compute_set_metrics(ans.predicted_entities, gold_entities)
                ret_m = compute_retrieval_metrics(context, gold_facts)
                faith_m = compute_faithfulness_metrics(ans.answer_text, context, citations)

                # Cost
                prompt_tok = max(10, len(context) // 4)
                comp_tok = max(15, len(ans.answer_text) // 4)
                query_cost = self.cost_model.calculate_query_cost(prompt_tok, comp_tok)

                records.append({
                    "question_id": qid,
                    "system": sys_id,
                    "hop_count": hop_count,
                    "question_type": q_type,
                    "precision": set_m["precision"],
                    "recall": set_m["recall"],
                    "f1": set_m["f1"],
                    "exact_match": set_m["exact_match"],
                    "fact_recall": ret_m["supporting_fact_recall"],
                    "context_precision": ret_m["context_precision"],
                    "hallucination_rate": faith_m["hallucination_rate"],
                    "citation_correctness": faith_m["citation_correctness"],
                    "abstained": ans.abstained,
                    "tokens": prompt_tok + comp_tok,
                    "cost_usd": query_cost,
                    "latency_ms": round(total_latency, 2),
                })

        df = pd.DataFrame(records)
        return self._aggregate_results(df)

    def _aggregate_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate evaluation metrics by system, by hop count, and by question type."""
        overall = df.groupby("system").agg({
            "f1": "mean",
            "precision": "mean",
            "recall": "mean",
            "exact_match": "mean",
            "fact_recall": "mean",
            "context_precision": "mean",
            "hallucination_rate": "mean",
            "citation_correctness": "mean",
            "tokens": "mean",
            "cost_usd": "mean",
            "latency_ms": "mean",
        }).round(4).to_dict(orient="index")

        by_hop = df.groupby(["hop_count", "system"])["f1"].mean().unstack().round(4).to_dict()

        by_type = df.groupby(["question_type", "system"])["f1"].mean().unstack().round(4).to_dict()

        # Compute Economic Break-Even between B1 and S1
        build_cost = self.cost_model.calculate_graph_build_cost(len(self.chunks))
        b1_cost = overall.get("B1", {}).get("cost_usd", 0.0003)
        s1_cost = overall.get("S1", {}).get("cost_usd", 0.00015)
        b1_f1 = overall.get("B1", {}).get("f1", 0.40)
        s1_f1 = overall.get("S1", {}).get("f1", 0.85)

        break_even = self.cost_model.calculate_break_even_point(
            build_cost_usd=build_cost["total_dollar_cost"],
            vector_query_cost_usd=b1_cost,
            graph_query_cost_usd=s1_cost,
            vector_f1=b1_f1,
            graph_f1=s1_f1,
        )

        return {
            "overall_by_system": overall,
            "f1_by_hop_count": by_hop,
            "f1_by_question_type": by_type,
            "graph_build_cost": build_cost,
            "break_even_analysis": break_even,
            "raw_records_count": len(df),
        }

    def run_traversal_depth_ablation(self) -> Dict[str, Any]:
        """Ablation study on traversal depth (k=1 vs k=2 vs k=3) for GraphRAG (S1)."""
        ablation_results = {}
        for k in [1, 2, 3]:
            f1_scores = []
            recalls = []
            latencies = []

            for q in self.benchmark_questions:
                q_text = q["question"]
                gold_ents = q["gold_entities"]
                t0 = time.time()
                ctx, cits, _ = self.graph_retriever.retrieve(q_text, max_hops=k)
                ans = self.answerer.generate_answer("S1", q_text, ctx, cits)
                lat = (time.time() - t0) * 1000.0

                m = compute_set_metrics(ans.predicted_entities, gold_ents)
                f1_scores.append(m["f1"])
                recalls.append(m["recall"])
                latencies.append(lat)

            ablation_results[f"hops_k_{k}"] = {
                "mean_f1": round(sum(f1_scores) / len(f1_scores), 4),
                "mean_recall": round(sum(recalls) / len(recalls), 4),
                "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
            }

        return ablation_results
