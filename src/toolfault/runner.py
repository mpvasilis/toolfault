"""Grid runner: agents x tasks x conditions, with an estimated spend stop threshold."""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .agents.base import Agent
from .agents.degenerate import (
    AlwaysEscalateAgent,
    ConfidentGuessAgent,
    ModalAnswerAgent,
    OneCallAgent,
    SecondarySourceOnlyAgent,
)
from .agents.openai_compat import OpenAICompatAgent
from .agents.scripted import NaiveAgent, OracleAgent
from .budget import Budget, BudgetExceeded, estimate_run_cost, price_for
from .episode import Episode, run_episode
from .faults import CONDITIONS, TASK_TYPES
from .tasks import Task, generate_tasks
from .world import World, build_world

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "huggingface": ("https://router.huggingface.co/v1", "HF_TOKEN"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "local": ("http://127.0.0.1:8000/v1", "LOCAL_API_KEY"),
}
SCRIPTED_AGENTS = {
    "oracle": OracleAgent,
    "naive": NaiveAgent,
    "always-escalate": AlwaysEscalateAgent,
    "modal-answer": ModalAnswerAgent,
    "one-call": OneCallAgent,
    "confident-guess": ConfidentGuessAgent,
    "secondary-only": SecondarySourceOnlyAgent,
}


@dataclass
class AgentSpec:
    kind: str
    model: str = ""
    label: str = ""
    base_url: str = ""
    api_key_env: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 4000
    seed: int | None = 20260801

    def resolved_label(self) -> str:
        return self.label or self.model or self.kind

    def is_paid(self) -> bool:
        return self.kind not in SCRIPTED_AGENTS and self.kind != "local"


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
    task_types: tuple[str, ...] = TASK_TYPES
    save_traces: str = "failures"  # all | failures | none
    trace_sample: float = 0.1
    out_dir: str = "results"
    agents: list[AgentSpec] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RunConfig:
        data = dict(data)
        agents = [AgentSpec(**spec) for spec in data.pop("agents", [])]
        conditions = data.pop("conditions", None)
        task_types = data.pop("task_types", None)
        config = RunConfig(**data, agents=agents)
        if conditions is not None:
            config.conditions = tuple(conditions)
        if task_types is not None:
            config.task_types = tuple(task_types)
        return config

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id) or self.run_id.endswith("."):
            raise ValueError("run_id must be a directory name made of letters, digits, _, - or .")
        if min(self.repeats, self.tasks_per_type, self.workers, self.max_tool_calls) < 1:
            raise ValueError("repeats, tasks_per_type, workers and max_tool_calls must be positive")
        for name, selected, supported in (
            ("conditions", self.conditions, CONDITIONS),
            ("task_types", self.task_types, TASK_TYPES),
        ):
            if not selected or set(selected) - set(supported) or len(set(selected)) != len(selected):
                raise ValueError(f"{name} must contain distinct known values")
        if self.save_traces not in ("all", "failures", "none") or not 0 <= self.trace_sample <= 1:
            raise ValueError("invalid save_traces or trace_sample")
        labels = [spec.resolved_label().casefold() for spec in self.agents]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("configure at least one agent and give each agent a distinct label")
        if not math.isfinite(self.budget_usd) or self.budget_usd < 0:
            raise ValueError("budget_usd must be finite and nonnegative")
        if self.budget_usd == 0 and any(spec.is_paid() for spec in self.agents):
            raise ValueError("remote model agents require a positive budget_usd")
        for spec in self.agents:
            if spec.kind not in SCRIPTED_AGENTS and (not spec.model or spec.max_output_tokens < 1):
                raise ValueError(f"agent {spec.resolved_label()}: model and positive max_output_tokens required")


def _connection(spec: AgentSpec) -> tuple[str, str]:
    base_url, key_env = PROVIDERS.get(spec.kind, ("", ""))
    base_url = spec.base_url or base_url
    key_env = spec.api_key_env or key_env
    if not base_url:
        raise ValueError(f"agent {spec.resolved_label()}: no base_url and unknown kind {spec.kind!r}")
    api_key = os.environ.get(key_env, "")
    if not api_key and (spec.kind != "local" or spec.api_key_env):
        raise RuntimeError(f"agent {spec.resolved_label()}: environment variable {key_env} is not set")
    return base_url, api_key


def make_agent(spec: AgentSpec, budget: Budget, repeat: int = 0) -> Agent:
    """A fresh agent per episode. The scripted policies carry generator state."""
    if spec.kind in SCRIPTED_AGENTS:
        return SCRIPTED_AGENTS[spec.kind]()
    base_url, api_key = _connection(spec)
    return OpenAICompatAgent(
        model=spec.model,
        api_key=api_key,
        base_url=base_url,
        budget=budget,
        temperature=spec.temperature,
        max_output_tokens=spec.max_output_tokens,
        seed=spec.seed + repeat if spec.seed is not None else None,
        provider=spec.kind,
        name=spec.resolved_label(),
    )


def plan(config: RunConfig, tasks: list[Task]) -> list[tuple[AgentSpec, Task, str, int]]:
    """Every (agent, task, condition, repeat) cell the run will execute."""
    config.validate()
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
    if not cells:
        raise ValueError("no tasks support the selected task types and conditions")
    return cells


def estimate_grid_cost(cells: list[tuple[AgentSpec, Task, str, int]]) -> tuple[int, float]:
    counts = Counter((spec.model, spec.kind) for spec, *_ in cells if spec.is_paid())
    estimate = sum(
        estimate_run_cost([(model, kind)], count)["__total__"]
        for (model, kind), count in counts.items()
    )
    return sum(counts.values()), estimate


def run_grid(config: RunConfig, *, verbose: bool = True) -> dict[str, Any]:
    world = build_world(seed=config.world_seed)
    tasks = generate_tasks(world, n_per_type=config.tasks_per_type, seed=config.task_seed)
    cells = plan(config, tasks)
    for spec in config.agents:
        if spec.kind not in SCRIPTED_AGENTS:
            _connection(spec)

    if verbose:
        print(f"[toolfault] world seed {config.world_seed}, {len(tasks)} tasks, {len(cells)} episodes", file=sys.stderr)
        paid_cells, estimate = estimate_grid_cost(cells)
        if paid_cells:
            print(
                f"[toolfault] {paid_cells} paid episodes, estimated "
                f"${estimate:.2f}, stop threshold ${config.budget_usd:.2f}",
                file=sys.stderr,
            )

    budget = Budget(limit_usd=config.budget_usd)
    out_root = Path(config.out_dir) / config.run_id
    trace_dir = out_root / "traces"
    try:
        out_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"run directory exists at {out_root}. Set a new --run-id or --out-dir.") from None
    if config.save_traces != "none":
        trace_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    stopped = False
    failed_cells = 0

    def execute(cell: tuple[AgentSpec, Task, str, int]) -> Episode:
        spec, task, condition, repeat = cell
        agent = make_agent(spec, budget, repeat)
        try:
            episode = run_episode(
                world,
                task,
                condition,
                agent,
                run_id=config.run_id,
                model=spec.model or spec.kind,
                provider=spec.kind,
                max_tool_calls=config.max_tool_calls,
            )
        finally:
            close = getattr(agent, "close", None)
            if callable(close):
                close()
        episode.agent = spec.resolved_label()
        episode.repeat = repeat
        episode.task_seed = config.task_seed
        return episode

    with ThreadPoolExecutor(max_workers=max(1, config.workers)) as pool:
        futures = {pool.submit(execute, cell): cell for cell in cells}
        done = 0
        for future in as_completed(futures):
            spec, task, condition, repeat = futures[future]
            try:
                episode = future.result()
            except BudgetExceeded as exc:
                stopped = True
                if verbose:
                    print(f"[toolfault] {exc}", file=sys.stderr)
                continue
            except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the grid
                failed_cells += 1
                episode = Episode(
                    run_id=config.run_id,
                    agent=spec.resolved_label(),
                    model=spec.model or spec.kind,
                    task_id=task.task_id,
                    task_type=task.task_type,
                    condition=condition,
                    world_seed=config.world_seed,
                    repeat=repeat,
                    task_seed=config.task_seed,
                    ground_truth=task.answer,
                    terminal="error",
                    outcome="error",
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            row = episode.to_row()
            rows.append(row)
            done += 1
            if verbose and done % 25 == 0:
                print(
                    f"[toolfault] {done}/{len(cells)} episodes, ${budget.spent_usd:.3f} metered",
                    file=sys.stderr,
                )
            identity = (config.world_seed, config.task_seed, episode.agent, task.task_id, condition, repeat)
            keep = config.save_traces == "all" or (
                config.save_traces == "failures"
                and (
                    episode.unsafe or episode.error
                    or random.Random(json.dumps(identity)).random() < config.trace_sample
                )
            )
            if keep:
                label = quote(episode.agent, safe="")
                path = trace_dir / f"{label}__{task.task_id}__{condition}__r{repeat}.json"
                path.write_text(
                    json.dumps(
                        {"row": row, "messages": episode.messages, "tool_log": episode.tool_log},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

    episodes_path = out_root / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (r["agent"], r["task_id"], r["condition"], r["repeat"])):
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
        "episodes_with_errors": sum(row["outcome"] == "error" for row in rows),
        "harness_exceptions": failed_cells,
        "budget_usd": config.budget_usd,
        "spent_usd": round(budget.spent_usd, 4),
        "spend_by_model": {k: round(v, 4) for k, v in budget.per_model.items()},
        "model_calls": budget.calls,
        "stopped_on_budget": stopped,
        # The price table travels with the numbers it produced, because prices move.
        "prices_usd_per_million": {
            spec.model: list(price_for(spec.model, spec.kind))
            for spec in config.agents
            if spec.is_paid() and spec.model
        },
        "agents": [
            {
                "label": s.resolved_label(),
                "kind": s.kind,
                "model": s.model,
                "temperature": s.temperature,
                "max_output_tokens": s.max_output_tokens,
                "seed": s.seed,
                "price_usd_per_million": list(price_for(s.model, s.kind)) if s.is_paid() else None,
            }
            for s in config.agents
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
