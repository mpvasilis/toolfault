"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .faults import FAULTS, applicable_conditions
from .report import summarize, write_report
from .runner import RunConfig, load_rows, run_grid
from .tasks import generate_tasks
from .world import build_world


def _load_env_file(path: str) -> int:
    loaded = 0
    file = Path(path)
    if not file.exists():
        print(f"[toolfault] no env file at {path}", file=sys.stderr)
        return 0
    for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and not os.environ.get(key):
            os.environ[key] = value
            loaded += 1
    return loaded


def _load_config(path: str) -> RunConfig:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RunConfig.from_dict(data)


def cmd_run(args: argparse.Namespace) -> int:
    if args.env_file:
        count = _load_env_file(args.env_file)
        print(f"[toolfault] loaded {count} variables from {args.env_file}", file=sys.stderr)
    config = _load_config(args.config)
    if args.run_id:
        config.run_id = args.run_id
    if args.budget is not None:
        config.budget_usd = args.budget
    if args.workers is not None:
        config.workers = args.workers
    if args.save_traces:
        config.save_traces = args.save_traces
    if args.only:
        keep = set(args.only)
        config.agents = [s for s in config.agents if s.resolved_label() in keep or s.kind in keep]
        if not config.agents:
            print(f"[toolfault] --only {sorted(keep)} matched no agent in the config", file=sys.stderr)
            return 2

    if args.dry_run:
        world = build_world(seed=config.world_seed)
        tasks = generate_tasks(world, n_per_type=config.tasks_per_type, seed=config.task_seed)
        from .runner import plan

        cells = plan(config, tasks)
        paid = sum(1 for spec, *_ in cells if spec.is_paid())
        print(json.dumps({
            "run_id": config.run_id,
            "tasks": len(tasks),
            "episodes": len(cells),
            "paid_episodes": paid,
            "agents": [s.resolved_label() for s in config.agents],
            "budget_usd": config.budget_usd,
        }, indent=2))
        return 0

    manifest = run_grid(config)
    rows = load_rows(Path(config.out_dir) / config.run_id / "episodes.jsonl")
    summary = summarize(rows)
    paths = write_report(summary, Path(config.out_dir) / config.run_id)
    print(json.dumps({"manifest": manifest, "report": str(paths["report"])}, indent=2))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run both reference policies over the full grid. No API key, no cost."""
    config = RunConfig(
        run_id=args.run_id,
        world_seed=args.world_seed,
        tasks_per_type=args.tasks_per_type,
        budget_usd=0.0,
        workers=args.workers,
        save_traces="failures" if args.keep_traces else "none",
        out_dir=args.out_dir,
    )
    from .runner import AgentSpec

    config.agents = [AgentSpec(kind="oracle"), AgentSpec(kind="naive")]
    run_grid(config, verbose=not args.quiet)
    rows = load_rows(Path(config.out_dir) / config.run_id / "episodes.jsonl")
    summary = summarize(rows)
    write_report(summary, Path(config.out_dir) / config.run_id)

    oracle = summary["agents"].get("oracle")
    if oracle is None:
        print("[toolfault] the oracle produced no episodes", file=sys.stderr)
        return 1

    failures = []
    for condition, cell in oracle["conditions"].items():
        if cell["resolved"].value < 1.0:
            failures.append((condition, cell["resolved"].value, cell["n"]))

    naive = summary["agents"].get("naive", {})
    print(f"oracle episodes: {oracle['episodes']}")
    print(f"oracle clean: {oracle['headline']['clean_resolved'].with_ci()}")
    print(f"oracle under faults: {oracle['headline']['fault_resolved'].with_ci()}")
    if naive:
        print(f"naive clean: {naive['headline']['clean_resolved'].with_ci()}")
        print(f"naive under silent faults: {naive['headline']['silent_resolved'].with_ci()}")
        print(f"naive unsafe on silent faults: {naive['headline']['silent_unsafe'].with_ci()}")

    if failures:
        print("\nSOLVABILITY FAILURE: the reference policy did not resolve every condition.")
        for condition, value, n in sorted(failures):
            print(f"  {condition}: {value:.3f} over n={n}")
        rows_by_cell = [
            r for r in rows
            if r["agent"] == "oracle" and r["condition"] in {c for c, _, _ in failures}
            and not r["resolved_correctly"]
        ]
        for row in rows_by_cell[:12]:
            print(
                f"    {row['task_id']} {row['condition']}: terminal={row['terminal']} "
                f"answer={row['answer']!r} truth={row['ground_truth']!r} "
                f"flags={row['flags']}"
            )
        return 1

    print("\nOK: the reference policy resolves every condition. The benchmark is solvable.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = load_rows(args.episodes)
    summary = summarize(rows)
    out = args.out or str(Path(args.episodes).parent)
    paths = write_report(summary, out)
    if not args.no_charts:
        from .charts import write_charts

        for chart in write_charts(summary, out):
            print(str(chart))
    print(str(paths["report"]))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    data: dict[str, Any] = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    row = data.get("row", {})
    print(f"agent      : {row.get('agent')}")
    print(f"task       : {row.get('task_id')} ({row.get('task_type')})")
    print(f"condition  : {row.get('condition')}")
    print(f"truth      : {row.get('ground_truth')}")
    print(f"answer     : {row.get('answer')}  confidence={row.get('confidence')}")
    print(f"outcome    : {row.get('outcome')}  unsafe={row.get('unsafe')}")
    print(f"flags      : {row.get('flags')}")
    print(f"fault fired: {row.get('fault_fired')}  events={row.get('fault_events')}")
    print("-" * 78)
    for call in data.get("tool_log", []):
        payload = json.dumps(call.get("result"), ensure_ascii=False)
        if len(payload) > args.width:
            payload = payload[: args.width] + " ..."
        marker = "ok " if call.get("ok") else "ERR"
        print(f"{marker} {call['tool']}({json.dumps(call['args'], ensure_ascii=False)})")
        print(f"    -> {payload}")
    if args.messages:
        print("-" * 78)
        for message in data.get("messages", []):
            content = (message.get("content") or "")[: args.width]
            print(f"[{message['role']}] {content}")
    return 0


def cmd_faults(args: argparse.Namespace) -> int:
    if args.markdown:
        print("| fault | family | recovery | applies to | detected by |")
        print("|---|---|---|---|---|")
        for spec in FAULTS:
            print(
                f"| `{spec.fault_id}` | {spec.family} | {spec.recovery} | "
                f"{', '.join(spec.applies_to)} | {spec.recovery_hint} |"
            )
        return 0
    for spec in FAULTS:
        print(f"{spec.fault_id:22} {spec.family:12} {spec.recovery:14} {spec.summary}")
        print(f"{'':22} recovery: {spec.recovery_hint}")
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    world = build_world(seed=args.world_seed)
    tasks = generate_tasks(world, n_per_type=args.tasks_per_type, seed=args.task_seed)
    print(f"world seed {world.seed}: {len(world.entities)} entities, {len(world.persons)} persons")
    print(f"{len(tasks)} tasks\n")
    by_type = Counter(t.task_type for t in tasks)
    for task_type, count in sorted(by_type.items()):
        answers = Counter(t.answer for t in tasks if t.task_type == task_type)
        top = ", ".join(f"{k}={v}" for k, v in answers.most_common(4))
        print(f"{task_type:18} n={count:3}  conditions={len(applicable_conditions(task_type)):2}  {top}")
    print()
    supports = Counter()
    for task in tasks:
        for condition in task.supports:
            supports[condition] += 1
    for condition, count in sorted(supports.items()):
        print(f"  {condition:24} supported by {count} tasks")
    if args.verbose:
        print()
        for task in tasks:
            print(f"{task.task_id:14} {task.answer:22} {task.question[:90]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="toolfault", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a benchmark grid from a config file")
    run.add_argument("--config", required=True)
    run.add_argument("--run-id")
    run.add_argument("--budget", type=float, help="hard USD ceiling, overrides the config")
    run.add_argument("--workers", type=int)
    run.add_argument("--save-traces", choices=["all", "failures", "none"])
    run.add_argument("--only", nargs="*", help="restrict to these agent labels or kinds")
    run.add_argument("--env-file", help="read API keys from this file if not already in the env")
    run.add_argument("--dry-run", action="store_true", help="print the plan and the cost estimate")
    run.set_defaults(func=cmd_run)

    selftest = sub.add_parser(
        "selftest", help="prove the benchmark is solvable using the free reference policies"
    )
    selftest.add_argument("--run-id", default="selftest")
    selftest.add_argument("--world-seed", type=int, default=7)
    selftest.add_argument("--tasks-per-type", type=int, default=12)
    selftest.add_argument("--workers", type=int, default=8)
    selftest.add_argument("--out-dir", default="runs")
    selftest.add_argument("--keep-traces", action="store_true")
    selftest.add_argument("--quiet", action="store_true")
    selftest.set_defaults(func=cmd_selftest)

    report = sub.add_parser("report", help="rebuild the report from an episodes.jsonl")
    report.add_argument("--episodes", required=True)
    report.add_argument("--out")
    report.add_argument("--no-charts", action="store_true")
    report.set_defaults(func=cmd_report)

    replay = sub.add_parser("replay", help="print one saved trace")
    replay.add_argument("trace")
    replay.add_argument("--width", type=int, default=400)
    replay.add_argument("--messages", action="store_true")
    replay.set_defaults(func=cmd_replay)

    faults = sub.add_parser("faults", help="print the fault taxonomy")
    faults.add_argument("--markdown", action="store_true")
    faults.set_defaults(func=cmd_faults)

    tasks = sub.add_parser("tasks", help="print the task set")
    tasks.add_argument("--world-seed", type=int, default=7)
    tasks.add_argument("--tasks-per-type", type=int, default=12)
    tasks.add_argument("--task-seed", type=int, default=11)
    tasks.add_argument("--verbose", action="store_true")
    tasks.set_defaults(func=cmd_tasks)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
