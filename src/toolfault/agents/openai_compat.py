"""Adapter for any OpenAI-compatible chat-completions endpoint.

Covers OpenAI, Together, Fireworks, Groq, and a local vLLM or SGLang server. The only
provider-specific handling is the parameter naming quirk in the gpt-5 family, which is
kept in one place and documented rather than spread through the harness.

Transport failures are retried with backoff and are not treated as agent behaviour. A
failed HTTP call is a harness problem, so the episode records an error and is excluded
from the metrics rather than being scored as an agent mistake.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..budget import Budget
from .base import Context, ModelReply, ToolCall

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

# Some providers return a retryable status code for a condition that is not retryable.
# An exhausted balance arrives as 429 from OpenAI and as 402 from Together, and retrying it
# five times per episode across a grid wastes minutes and logs nothing useful. Fitting,
# given what this benchmark measures.
TERMINAL_BODY_MARKERS = (
    "insufficient_quota",
    "credit_balance_exhausted",
    "no credits remaining",
    "payment required",
    "invalid_api_key",
    "invalid api key",
    "account is not active",
)


def _uses_completion_token_param(model: str) -> bool:
    head = model.split("/")[-1].lower()
    return head.startswith(("gpt-5", "o1", "o3", "o4"))


def _supports_temperature(model: str) -> bool:
    return not _uses_completion_token_param(model)


@dataclass
class OpenAICompatAgent:
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    budget: Budget | None = None
    temperature: float = 0.0
    max_output_tokens: int = 1200
    timeout: float = 180.0
    max_retries: int = 5
    seed: int | None = 20260801
    name: str = field(default="")
    _client: httpx.Client | None = field(default=None, repr=False)
    last_error: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.model

    # ------------------------------------------------------------------ wire

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url.rstrip("/"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        if _uses_completion_token_param(self.model):
            payload["max_completion_tokens"] = self.max_output_tokens
        else:
            payload["max_tokens"] = self.max_output_tokens
        if _supports_temperature(self.model):
            payload["temperature"] = self.temperature
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        delay = 2.0
        last: str | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._http().post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last = f"transport: {exc.__class__.__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response.json(), None
                last = f"http {response.status_code}: {response.text[:400]}"
                if response.status_code not in RETRY_STATUS:
                    return None, last
                lowered = last.lower()
                if any(marker in lowered for marker in TERMINAL_BODY_MARKERS):
                    return None, last
            if attempt < self.max_retries - 1:
                time.sleep(delay + random.uniform(0, 0.75))
                delay = min(delay * 2, 30.0)
        return None, last

    # -------------------------------------------------------------- protocol

    def reply(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], ctx: Context
    ) -> ModelReply:
        if self.budget is not None:
            self.budget.check()

        body, error = self._post(self._payload(messages, tools))
        if body is None:
            self.last_error = error
            return ModelReply(error=error or "unknown request failure")

        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if self.budget is not None:
            self.budget.add(self.model, prompt_tokens, completion_tokens)

        choices = body.get("choices") or []
        if not choices:
            return ModelReply(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error="response carried no choices",
            )

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some providers return content parts
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )

        calls: list[ToolCall] = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function") or {}
            name = function.get("name") or ""
            arguments = function.get("arguments")
            args: dict[str, Any]
            if isinstance(arguments, dict):
                args = arguments
            else:
                try:
                    parsed = json.loads(arguments or "{}")
                    args = parsed if isinstance(parsed, dict) else {"value": parsed}
                except json.JSONDecodeError:
                    # A model that emits unparseable arguments is agent behaviour, not a
                    # transport fault, so it flows through to the tool as bad arguments.
                    args = {"__unparsed__": str(arguments)[:400]}
            calls.append(
                ToolCall(call_id=raw.get("id") or f"call-{index}", name=name, args=args)
            )

        return ModelReply(
            content=content if isinstance(content, str) else None,
            tool_calls=calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
