"""Token pricing and a hard spend ceiling.

The ceiling is checked before every model call, not after, so an overnight run cannot
quietly spend more than it was told to. Prices are USD per million tokens and are
recorded in every result file alongside the numbers they produced, because they change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per million tokens, (prompt, completion).
#
# The Together figures were read from that provider's own /v1/models endpoint on
# 2026-08-01; the OpenAI figures come from its published price list on the same date,
# because the OpenAI API does not expose pricing. Prices move, so the table used for a
# run is copied into that run's manifest.json next to the costs it produced. Anything not
# listed is priced at UNKNOWN_PRICE and the cost is an upper-bound estimate.
PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5": (1.25, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    # Together, open weight. Read from that provider's /v1/models endpoint on 2026-08-01,
    # which is authoritative rather than transcribed from a price page.
    "openai/gpt-oss-20b": (0.05, 0.20),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "Qwen/Qwen3.5-9B": (0.17, 0.25),
    "Qwen/Qwen3.7-Plus": (0.32, 1.28),
    "Qwen/Qwen3.7-Max": (1.25, 3.75),
    "Qwen/Qwen3.6-Plus": (0.50, 3.00),
    "Qwen/Qwen3.5-397B-A17B": (0.60, 3.60),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": (0.15, 1.50),
    "zai-org/GLM-4.7": (0.45, 2.00),
    "zai-org/GLM-5": (1.00, 3.20),
    "zai-org/GLM-5.2": (1.40, 4.40),
    "moonshotai/Kimi-K2.6": (1.20, 4.50),
    "moonshotai/Kimi-K2.5-fp4": (0.50, 2.80),
    "deepseek-ai/DeepSeek-V3.1": (0.60, 1.70),
    "deepseek-ai/DeepSeek-V4-Pro": (1.74, 3.48),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
    "MiniMaxAI/MiniMax-M2.7": (0.30, 1.20),
}

UNKNOWN_PRICE = (1.00, 3.00)


def price_for(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    for key, value in PRICES.items():
        if key.endswith(model) or model.endswith(key):
            return value
    return UNKNOWN_PRICE


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = price_for(model)
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    """Shared spend counter with a hard ceiling."""

    limit_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    per_model: dict[str, float] = field(default_factory=dict)

    def check(self) -> None:
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(
                f"spend ceiling reached: ${self.spent_usd:.4f} of ${self.limit_usd:.2f}"
            )

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        delta = cost_usd(model, prompt_tokens, completion_tokens)
        self.spent_usd += delta
        self.calls += 1
        self.per_model[model] = self.per_model.get(model, 0.0) + delta
        return delta

    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)


def estimate_run_cost(
    models: list[str],
    episodes_per_model: int,
    avg_prompt_tokens: int = 5200,
    avg_completion_tokens: int = 420,
    avg_model_calls_per_episode: int = 7,
) -> dict[str, float]:
    """Rough pre-run estimate, printed before anything is spent.

    The prompt-token figure assumes the conversation grows as tool results accumulate,
    so it is the average over the calls in an episode rather than the system prompt size.
    """
    out: dict[str, float] = {}
    total = 0.0
    for model in models:
        per_episode = cost_usd(
            model,
            avg_prompt_tokens * avg_model_calls_per_episode,
            avg_completion_tokens * avg_model_calls_per_episode,
        )
        model_total = per_episode * episodes_per_model
        out[model] = model_total
        total += model_total
    out["__total__"] = total
    return out
