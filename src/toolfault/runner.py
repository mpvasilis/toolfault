"""Grid runner: agents x tasks x conditions, with a hard spend ceiling."""

from __future__ import annotations

import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents.base import Agent
from .agents.openai_compat import OpenAICompatAgent
from .agents.scripted import NaiveAgent, OracleAgent
from .budget import Budget, BudgetExceeded, estimate_run_cost, price_for
from .episode import Episode, run_episode
from .faults import CONDITIONS
from .tasks import Task, generate_tasks
from .world import World, build_world

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "huggingface": ("https://router.huggingface.co/v1", "HF_TOKEN"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "local": ("http://127.0.0.1:8000/v1", "LOCAL_API_KEY"),
}


@dataclass
class AgentSpec:
    kind: str
    model: str = ""
    label: str = ""
    base_url: str = ""
    api_key_env: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 1200

    def resolved_label(self) -> str:
        return self.label or self.model or self.kind

    def is_paid(self) -> bool:
        return self.kind not in ("oracle", "naive")


@dataclass
class RunConfig:
    run_id: str = "local"
    world_seed: int = 7
    tasks_per_type: int = 12
    task_seed: int = 11
    repeats: int = 1
    max_tool_calls: int = 18
    budget_usd: float = 5.0
    workers: int = 6
    conditions: tuple[str, ...] = CONDITIONS
    task_types: tuple[str, ...] = ("verify_entity", "ubo_chain", "revenue_eur", "sanctions_screen")
    save_traces: str = "failures"  # all | failures | none
    trace_sample: float = 0.1
    out_dir: str = "results"
    agents: list[AgentSpec] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RunConfig:
        agents = [AgentSpec(**spec) for spec in data.pop("agents", [])]
        conditions = data.pop("conditions", None)
        task_types = data.pop("task_types", None)
        config = RunConfig(**data, agents=agents)
        if conditions:
            config.conditions = tuple(conditions)
        if task_types:
            config.task_types = tuple(task_types)
        return config


def make_agent(spec: AgentSpec, budget: Budget) -> Agent:
    """A fresh agent per episode. The scripted policies carry generator state."""
    if spec.kind == "oracle":
        return OracleAgent()
    if spec.kind == "naive":
        return NaiveAgent()
    base_url, key_env = PROVIDERS.get(spec.kind, ("", ""))
    base_url = spec.base_url or base_url
    key_env = spec.api_key_env or key_env
    if not base_url:
        raise ValueError(f"agent {spec.resolved_label()}: no base_url and unknown kind {spec.kind!r}")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise RuntimeError(f"agent {spec.resolved_label()}: environment variable {key_env} is not set")
    return OpenAICompatAgent(
        model=spec.model,
        api_key=api_key,
        base_url=base_url,
        budget=budget,
        temperature=spec.temperature,
        max_output_tokens=spec.max_output_tokens,
        name=spec.resolved_label(),
    )


def plan(config: RunConfig, tasks: list[Task]) -> list[tuple[AgentSpec, Task, str, int]]:
    """Every (agent, task, condition, repeat) cell the run will execute."""
    cells: list[tuple[AgentSpec, Task, str, int]] = []
    for spec in config.agents:
        for task in tasks:
            if task.task_type not in config.task_types:
                continue
            for condition in config.conditions:
                if condition not in task.supports:
                    continue
                for repeat in range(config.repeats):
                    cells.append((spec, task, condition, repeat))
    return cells


def run_grid(config: RunConfig, *, verbose: bool = True) -> dict[str, Any]:
    world = build_world(seed=config.world_seed)
    tasks = generate_tasks(world, n_per_type=config.tasks_per_type, seed=config.task_seed)
    cells = plan(config, tasks)

    paid_specs = [s for s in config.agents if s.is_paid()]
    paid_cells = sum(1 for spec, *_ in cells if spec.is_paid())
    if verbose:
        print(f"[toolfault] world seed {config.world_seed}, {len(tasks)} tasks, {len(cells)} episodes", file=sys.stderr)
        if paid_specs:
            per_model = paid_cells // max(1, len(paid_specs))
            estimate = estimate_run_cost([s.model for s in paid_specs], per_model)
            print(
                f"[toolfault] {paid_cells} paid episodes, estimated "
                f"${estimate['__total__']:.2f}, ceiling ${config.budget_usd:.2f}",
                file=sys.stderr,
            )

    budget = Budget(limit_usd=config.budget_usd)
    out_root = Path(config.out_dir) / config.run_id
    trace_dir = out_root / "traces"
    out_root.mkdir(parents=True, exist_ok=True)
    if config.save_traces != "none":
        trace_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    sampler = random.Random(config.world_seed)
    stopped = False

    def execute(cell: tuple[AgentSpec, Task, str, int]) -> Episode | None:
        spec, task, condition, repeat = cell
        agent = make_agent(spec, budget)
        try:
            episode = run_episode(
                world,
                task,
                condition,
                agent,
                run_id=config.run_id,
                model=spec.model or spec.kind,
                max_tool_calls=config.max_tool_calls,
            )
        finally:
            close = getattr(agent, "close", None)
            if callable(close):
                close()
        episode.agent = spec.resolved_label()
        return episode

    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        futures = {pool.submit(execute, cell): cell for cell in cells}
        done = 0
        for future in as_completed(futures):
            spec, task, condition, _ = futures[future]
            try:
                episode = future.result()
            except BudgetExceeded as exc:
                stopped = True
                if verbose:
                    print(f"[toolfault] {exc}", file=sys.stderr)
                continue
            except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the grid
                if verbose:
                    print(
                        f"[toolfault] cell failed: {spec.resolved_label()} {task.task_id} "
                        f"{condition}: {exc.__class__.__name__}: {exc}",
                        file=sys.stderr,
                    )
                continue
            if episode is None:
                continue

            with lock:
                rows.append(episode.to_row())
                done += 1
                if verbose and done % 25 == 0:
                    print(
                        f"[toolfault] {done}/{len(cells)} episodes, ${budget.spent_usd:.3f} spent",
                        file=sys.stderr,
                    )
                keep = config.save_traces == "all" or (
                    config.save_traces == "failures"
                    and (episode.unsafe or episode.error or sampler.random() < config.trace_sample)
                )
                if keep and config.save_traces != "none":
                    path = (
                        trace_dir
                        / f"{episode.agent.replace('/', '_')}__{episode.task_id}__{condition}.json"
                    )
                    path.write_text(
                        json.dumps(
                            {
                                "row": episode.to_row(),
                                "messages": episode.messages,
                                "tool_log": episode.tool_log,
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )

    episodes_path = out_root / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (r["agent"], r["task_id"], r["condition"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "run_id": config.run_id,
        "world_seed": config.world_seed,
        "task_seed": config.task_seed,
        "tasks_per_type": config.tasks_per_type,
        "tasks": len(tasks),
        "repeats": config.repeats,
        "max_tool_calls": config.max_tool_calls,
        "conditions": list(config.conditions),
        "task_types": list(config.task_types),
        "episodes_planned": len(cells),
        "episodes_completed": len(rows),
        "budget_usd": config.budget_usd,
        "spent_usd": round(budget.spent_usd, 4),
        "spend_by_model": {k: round(v, 4) for k, v in budget.per_model.items()},
        "model_calls": budget.calls,
        "stopped_on_budget": stopped,
        # The price table travels with the numbers it produced, because prices move.
        "prices_usd_per_million": {
            spec.model: list(price_for(spec.model))
            for spec in config.agents
            if spec.is_paid() and spec.model
        },
        "agents": [
            {"label": s.resolved_label(), "kind": s.kind, "model": s.model} for s in config.agents
        ],
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if verbose:
        print(
            f"[toolfault] wrote {len(rows)} episodes to {episodes_path}, "
            f"spent ${budget.spent_usd:.3f}",
            file=sys.stderr,
        )
    return manifest


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_reference_world() -> tuple[World, list[Task]]:
    world = build_world()
    return world, generate_tasks(world)
