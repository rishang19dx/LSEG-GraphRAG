"""Unit tests for evaluation metrics, cost model, and benchmark runner."""

import pytest
from lseg_graphrag.evaluation.metrics import (
    compute_set_metrics,
    compute_retrieval_metrics,
    compute_faithfulness_metrics,
)
from lseg_graphrag.evaluation.cost_model import CostModel, CostConfig
from lseg_graphrag.evaluation.benchmark_runner import BenchmarkRunner


def test_set_metrics():
    # Exact match
    m1 = compute_set_metrics(["TSMC", "ASML"], ["TSMC", "ASML"])
    assert m1["f1"] == 1.0
    assert m1["exact_match"] == 1.0

    # Partial match
    m2 = compute_set_metrics(["TSMC"], ["TSMC", "ASML"])
    assert m2["precision"] == 1.0
    assert m2["recall"] == 0.5
    assert m2["f1"] == pytest.approx(0.6667, rel=1e-2)

    # Empty prediction
    m3 = compute_set_metrics([], ["TSMC"])
    assert m3["f1"] == 0.0


def test_retrieval_metrics():
    context = (
        "TSMC is our principal foundry partner manufacturing our advanced processors. "
        "The primary wafer fabrication plants are in Taiwan."
    )
    gold_facts = [
        "TSMC is our principal foundry partner",
        "primary wafer fabrication plants are in Taiwan",
        "ASML supplies EUV tools",  # Absent
    ]
    res = compute_retrieval_metrics(context, gold_facts)
    assert res["supporting_fact_recall"] == pytest.approx(0.6667, rel=1e-2)


def test_cost_model_and_break_even():
    model = CostModel(CostConfig(prompt_cost_per_million=0.15, completion_cost_per_million=0.60))

    # Query cost: 1000 prompt tokens ($0.00015) + 200 completion tokens ($0.00012)
    cost = model.calculate_query_cost(1000, 200)
    assert cost == pytest.approx(0.00027, abs=1e-5)

    build = model.calculate_graph_build_cost(total_chunks=20)
    assert build["total_dollar_cost"] > 0.0

    be = model.calculate_break_even_point(
        build_cost_usd=0.010,
        vector_query_cost_usd=0.0003,
        graph_query_cost_usd=0.00015,
        vector_f1=0.45,
        graph_f1=0.85,
    )
    assert be["accuracy_gain_f1"] == 0.40
    assert be["pure_cost_break_even_queries"] is not None
    assert be["pure_cost_break_even_queries"] > 0


def test_benchmark_runner_execution():
    runner = BenchmarkRunner()
    results = runner.run_all(systems=["B0", "B1", "S1", "S2"])

    assert "overall_by_system" in results
    assert "f1_by_hop_count" in results
    assert "f1_by_question_type" in results

    overall = results["overall_by_system"]
    assert "B0" in overall
    assert "B1" in overall
    assert "S1" in overall
    assert "S2" in overall

    # Verify GraphRAG beats zero-shot and vector on multi-hop questions
    hops = results["f1_by_hop_count"]
    # Check that Hop 2 or Hop 3 F1 for S1/S2 is higher than B1
    assert hops["S1"][2] >= hops["B1"][2]
