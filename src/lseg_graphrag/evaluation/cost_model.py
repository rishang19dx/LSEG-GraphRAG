"""Cost, latency, and break-even economic analysis model."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class CostConfig(BaseModel):
    """Token pricing configuration per 1,000,000 tokens."""
    prompt_cost_per_million: float = 0.150      # $0.15 per 1M tokens (gpt-4o-mini)
    completion_cost_per_million: float = 0.600  # $0.60 per 1M tokens


class CostModel:
    """Computes dollar costs, query latencies, and knowledge graph break-even points."""

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()

    def calculate_query_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate total dollar cost for a query."""
        prompt_cost = (prompt_tokens / 1_000_000) * self.config.prompt_cost_per_million
        completion_cost = (completion_tokens / 1_000_000) * self.config.completion_cost_per_million
        return round(prompt_cost + completion_cost, 6)

    def calculate_graph_build_cost(
        self,
        total_chunks: int,
        avg_tokens_per_chunk: int = 400,
        extraction_model_pricing: Optional[CostConfig] = None,
    ) -> Dict[str, Any]:
        """Compute one-time graph construction cost."""
        pricing = extraction_model_pricing or self.config
        total_prompt_tokens = total_chunks * avg_tokens_per_chunk
        total_completion_tokens = total_chunks * 150  # ~150 tokens per JSON output

        dollar_cost = (
            (total_prompt_tokens / 1_000_000) * pricing.prompt_cost_per_million +
            (total_completion_tokens / 1_000_000) * pricing.completion_cost_per_million
        )

        return {
            "total_chunks_processed": total_chunks,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_build_tokens": total_prompt_tokens + total_completion_tokens,
            "total_dollar_cost": round(dollar_cost, 4),
        }

    def calculate_break_even_point(
        self,
        build_cost_usd: float,
        vector_query_cost_usd: float,
        graph_query_cost_usd: float,
        vector_f1: float,
        graph_f1: float,
    ) -> Dict[str, Any]:
        """Compute query volume break-even points for pure cost and quality-adjusted value."""
        cost_savings_per_query = vector_query_cost_usd - graph_query_cost_usd

        # Pure token cost break-even
        if cost_savings_per_query > 0:
            pure_cost_break_even = int(build_cost_usd / cost_savings_per_query)
        else:
            pure_cost_break_even = None  # Graph query is more expensive or equal

        # Value/Quality-adjusted break-even
        # Cost per effective unit of F1 accuracy
        vector_effective_cost = vector_query_cost_usd / max(0.01, vector_f1)
        graph_effective_cost = graph_query_cost_usd / max(0.01, graph_f1)
        quality_savings_per_query = vector_effective_cost - graph_effective_cost

        quality_break_even = (
            int(build_cost_usd / quality_savings_per_query)
            if quality_savings_per_query > 0 else None
        )

        return {
            "build_cost_usd": build_cost_usd,
            "vector_query_cost_usd": vector_query_cost_usd,
            "graph_query_cost_usd": graph_query_cost_usd,
            "cost_delta_per_query": round(cost_savings_per_query, 6),
            "pure_cost_break_even_queries": pure_cost_break_even,
            "quality_break_even_queries": quality_break_even,
            "accuracy_gain_f1": round(graph_f1 - vector_f1, 4),
        }
