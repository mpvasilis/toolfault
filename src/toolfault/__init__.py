"""toolfault: a fault-injection benchmark for tool-using LLM agents.

The agent is given a compliance task and a set of registry tools. The tools then
misbehave in one of thirteen documented ways, one per episode, and the harness measures
what the agent does about it. Every fault has a recovery path, and a hand-written
reference policy resolves all fourteen conditions, so the ceiling is known.
"""

from .budget import Budget, cost_usd
from .episode import Episode, run_episode
from .faults import CONDITIONS, FAULTS, FaultInjector, FaultSpec
from .report import summarize, write_report
from .tasks import Task, generate_tasks
from .tools import TOOL_SCHEMAS, Registry
from .world import World, build_world

__version__ = "0.1.0"

__all__ = [
    "Budget",
    "CONDITIONS",
    "Episode",
    "FAULTS",
    "FaultInjector",
    "FaultSpec",
    "Registry",
    "TOOL_SCHEMAS",
    "Task",
    "World",
    "build_world",
    "cost_usd",
    "generate_tasks",
    "run_episode",
    "summarize",
    "write_report",
]
