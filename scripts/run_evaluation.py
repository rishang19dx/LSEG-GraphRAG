"""CLI script to run full evaluation, produce ablation tables, and compute break-even analysis."""

import json
from pathlib import Path
from lseg_graphrag.evaluation.benchmark_runner import BenchmarkRunner


def main():
    print("=" * 70)
    print("LSEG-GraphRAG: Executing Benchmark Evaluation (B0, B1, S1, S2)")
    print("=" * 70)

    runner = BenchmarkRunner()
    results = runner.run_all(systems=["B0", "B1", "S1", "S2"])

    print("\n### 1. Overall System Performance")
    overall = results["overall_by_system"]
    print(f"{'System':<8} | {'F1':<8} | {'Prec':<8} | {'Recall':<8} | {'Fact Rec':<10} | {'Halluc':<8} | {'Cost ($)':<10} | {'Latency (ms)':<12}")
    print("-" * 88)
    for sys_id, m in overall.items():
        print(f"{sys_id:<8} | {m['f1']:<8.4f} | {m['precision']:<8.4f} | {m['recall']:<8.4f} | {m['fact_recall']:<10.4f} | {m['hallucination_rate']:<8.4f} | ${m['cost_usd']:<9.6f} | {m['latency_ms']:<12.2f}")

    print("\n### 2. The Key Chart: F1 Breakdown by Hop Count")
    hops = results["f1_by_hop_count"]
    systems = ["B0", "B1", "S1", "S2"]
    print(f"{'Hop Count':<12} | " + " | ".join([f"{s:<8}" for s in systems]))
    print("-" * 52)
    for hop in sorted(list(hops.get("B0", {}).keys())):
        row = [f"{hops[s].get(hop, 0.0):<8.4f}" for s in systems]
        print(f"Hop {hop:<8} | " + " | ".join(row))

    print("\n### 3. F1 Breakdown by Question Type")
    qtypes = results["f1_by_question_type"]
    sample_sys = list(qtypes.keys())[0]
    all_types = list(qtypes[sample_sys].keys())
    print(f"{'Question Type':<22} | " + " | ".join([f"{s:<8}" for s in systems]))
    print("-" * 62)
    for qt in all_types:
        row = [f"{qtypes[s].get(qt, 0.0):<8.4f}" for s in systems]
        print(f"{qt:<22} | " + " | ".join(row))

    print("\n### 4. Traversal Depth Ablation (GraphRAG S1)")
    ablations = runner.run_traversal_depth_ablation()
    for depth, metrics in ablations.items():
        print(f"  - {depth}: Mean F1 = {metrics['mean_f1']:.4f}, Recall = {metrics['mean_recall']:.4f}, Latency = {metrics['mean_latency_ms']:.2f}ms")

    print("\n### 5. Economic & Break-Even Analysis")
    be = results["break_even_analysis"]
    build = results["graph_build_cost"]
    print(f"  - One-time Graph Construction Cost: ${build['total_dollar_cost']:.4f} ({build['total_build_tokens']} tokens across {build['total_chunks_processed']} chunks)")
    print(f"  - Vector RAG Query Cost (B1): ${be['vector_query_cost_usd']:.6f} / query")
    print(f"  - GraphRAG Query Cost (S1): ${be['graph_query_cost_usd']:.6f} / query")
    print(f"  - Accuracy Gain (Delta F1): +{be['accuracy_gain_f1'] * 100:.1f}%")
    print(f"  - Quality-Adjusted Break-Even Volume: {be['quality_break_even_queries']} queries")

    # Save results to JSON
    output_file = Path("data/benchmark/eval_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation metrics and breakdown to {output_file}")


if __name__ == "__main__":
    main()
