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
    # OpenRouter, read from its /v1/models endpoint on 2026-08-01. Keys are qualified with
    # the provider because the same model is not the same price everywhere: gpt-oss-20b is
    # $0.05/$0.20 on Together and $0.030/$0.130 here, and an unqualified lookup would report
    # whichever it found first.
    "openai/gpt-oss-20b@openrouter": (0.030, 0.130),
    "openai/gpt-oss-120b@openrouter": (0.037, 0.170),
    "qwen/qwen3-235b-a22b-2507@openrouter": (0.150, 0.598),
    "qwen/qwen3-max@openrouter": (0.780, 3.900),
    "deepseek/deepseek-chat-v3.1@openrouter": (0.250, 0.950),
    "moonshotai/kimi-k2-0905@openrouter": (0.600, 2.500),
    "z-ai/glm-4.6@openrouter": (0.500, 2.000),
    "google/gemini-2.5-flash@openrouter": (0.300, 2.500),
    "anthropic/claude-haiku-4.5@openrouter": (1.000, 5.000),
    "mistralai/mistral-medium-3.1@openrouter": (0.400, 2.000),
    "openai/gpt-5-mini@openrouter": (0.250, 2.000),
}

UNKNOWN_PRICE = (1.00, 3.00)


def price_for(model: str, provider: str = "") -> tuple[float, float]:
    """Price for a model, preferring the provider-qualified entry.

    Falls back to UNKNOWN_PRICE rather than guessing, so an unpriced model shows up as an
    upper bound in the manifest instead of quietly reporting someone else's rate.
    """
    # Self-hosted inference has no per-token price. Reporting one would be fiction, and
    # pricing a local vLLM server at the UNKNOWN_PRICE fallback would make a free run look
    # like it cost tens of dollars.
    if provider == "local":
        return (0.0, 0.0)
    if provider and f"{model}@{provider}" in PRICES:
        return PRICES[f"{model}@{provider}"]
    if model in PRICES:
        return PRICES[model]
    return UNKNOWN_PRICE


def cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int, provider: str = ""
) -> float:
    prompt_price, completion_price = price_for(model, provider)
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
        # A ceiling at or below zero means there is nothing to meter, which is the case for a
        # self-hosted server. Without this, a run against a local vLLM instance aborted on its
        # first call because spent (0) was already equal to the limit (0).
        if self.limit_usd <= 0:
            return
        if self.spent_usd >= self.limit_usd:
            raise BudgetExceeded(
                f"spend ceiling reached: ${self.spent_usd:.4f} of ${self.limit_usd:.2f}"
            )

    def add(
        self, model: str, prompt_tokens: int, completion_tokens: int, provider: str = ""
    ) -> float:
        delta = cost_usd(model, prompt_tokens, completion_tokens, provider)
        self.spent_usd += delta
        self.calls += 1
        self.per_model[model] = self.per_model.get(model, 0.0) + delta
        return delta

    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)


def estimate_run_cost(
    models: list[tuple[str, str]],
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
    for model, provider in models:
        per_episode = cost_usd(
            model,
            avg_prompt_tokens * avg_model_calls_per_episode,
            avg_completion_tokens * avg_model_calls_per_episode,
            provider,
        )
        model_total = per_episode * episodes_per_model
        out[model] = model_total
        total += model_total
    out["__total__"] = total
    return out
