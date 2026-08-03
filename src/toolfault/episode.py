"""The episode loop: one task, one condition, one agent, one trace."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .agents.base import Agent, Context, ModelReply
from .budget import cost_usd
from .faults import FaultInjector
from .scoring import score
from .tasks import MAX_TOOL_CALLS, SYSTEM_PROMPT, Task
from .tools import TERMINAL_TOOLS, TOOL_SCHEMAS, Registry
from .world import World

MAX_NUDGES = 2


@dataclass
class Episode:
    run_id: str
    agent: str
    model: str
    task_id: str
    task_type: str
    condition: str
    world_seed: int
    terminal: str = "no_terminal"
    answer: str | None = None
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    escalation_reason: str | None = None
    ground_truth: str = ""
    outcome: str = "no_answer"
    expected_action: str = "answer"
    resolved_correctly: bool = False
    unsafe: bool = False
    over_escalated: bool = False
    flagged: bool = False
    tool_call_count: int = 0
    distinct_tools: int = 0
    fault_fired: bool = False
    fault_side: str = ""
    fault_events: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        """Everything except the transcript, for tabular analysis."""
        row = asdict(self)
        row.pop("messages", None)
        row.pop("tool_log", None)
        return row


def _openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": schema} for schema in TOOL_SCHEMAS]


def _assistant_message(reply: ModelReply) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.args)},
            }
            for call in reply.tool_calls
        ]
    return message


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, out))


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def run_episode(
    world: World,
    task: Task,
    condition: str,
    agent: Agent,
    *,
    run_id: str = "local",
    model: str = "",
    provider: str = "",
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> Episode:
    injector = FaultInjector(
        world=world, condition=condition, task_type=task.task_type, task_id=task.task_id
    )
    registry = Registry(world=world, injector=injector)
    ctx = Context(task=task, world=world, condition=condition)

    episode = Episode(
        run_id=run_id,
        agent=getattr(agent, "name", agent.__class__.__name__),
        model=model,
        task_id=task.task_id,
        task_type=task.task_type,
        condition=condition,
        world_seed=world.seed,
        ground_truth=task.answer,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.question},
    ]
    tools = _openai_tools()
    started = time.perf_counter()
    nudges = 0
    # A few extra model turns beyond the tool budget, so an agent that runs out of read
    # calls still gets the chance to submit or escalate rather than being cut off.
    max_model_steps = max_tool_calls + 4
    steps = 0

    while steps < max_model_steps:
        steps += 1
        reply = agent.reply(messages, tools, ctx)
        episode.prompt_tokens += reply.prompt_tokens
        episode.completion_tokens += reply.completion_tokens

        if reply.error:
            episode.error = reply.error
            episode.terminal = "error"
            break

        messages.append(_assistant_message(reply))

        if not reply.tool_calls:
            nudges += 1
            if nudges > MAX_NUDGES:
                episode.terminal = "no_terminal"
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You did not call a tool. Continue the procedure, and finish with "
                        "exactly one call to submit or escalate."
                    ),
                }
            )
            continue

        terminated = False
        for call in reply.tool_calls:
            if call.name in TERMINAL_TOOLS:
                episode.terminal = call.name
                if call.name == "submit":
                    episode.answer = str(call.args.get("answer", "")).strip()
                    episode.confidence = _as_float(call.args.get("confidence"))
                    episode.evidence = _as_list(call.args.get("evidence"))
                    episode.flags = _as_list(call.args.get("data_quality_flags"))
                else:
                    episode.escalation_reason = str(call.args.get("reason", "")).strip()
                    episode.flags = _as_list(call.args.get("data_quality_flags"))
                    episode.confidence = _as_float(call.args.get("confidence"))
                terminated = True
                break

            if len(registry.calls) >= max_tool_calls:
                result: dict[str, Any] = {
                    "error": "budget_exhausted",
                    "message": (
                        f"Tool call budget of {max_tool_calls} is spent. Call submit or "
                        "escalate now."
                    ),
                }
            else:
                result = registry.call(call.name, call.args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        if terminated:
            break

    episode.wall_seconds = round(time.perf_counter() - started, 3)
    episode.cost_usd = cost_usd(
        model, episode.prompt_tokens, episode.completion_tokens, provider
    )
    episode.tool_call_count = len(registry.calls)
    episode.distinct_tools = len({c["tool"] for c in registry.calls})
    episode.fault_fired = injector.fired
    episode.fault_events = injector.events
    episode.fault_side = injector.side if condition != "clean" else ""
    episode.flagged = bool(episode.flags)
    episode.messages = messages
    episode.tool_log = registry.calls
    score(episode, task, condition)
    return episode
