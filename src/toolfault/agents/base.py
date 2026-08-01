"""The agent interface.

One protocol covers both API-backed models and the two hand-written reference policies,
so the harness runs all of them through exactly the same episode loop, the same tool
surface and the same fault injector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..tasks import Task
from ..world import World


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict[str, Any]


@dataclass
class ModelReply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


@dataclass
class Context:
    """Everything an agent may look at.

    API-backed agents ignore ``world``. The reference policies use it only to know which
    task they are solving; they still have to obtain every fact through the tools, which
    is the point of having them.
    """

    task: Task
    world: World
    condition: str


class Agent(Protocol):
    name: str

    def reply(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx: Context,
    ) -> ModelReply: ...
