"""Recheck the published silent-fault results and stress-test the scripted policies."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from toolfault.report import calibration_of, family_of, routing_of, usable
from toolfault.runner import AgentSpec, RunConfig, load_rows, run_grid
from toolfault.stats import auroc

WORLD_SEEDS = (7, 13, 17, 41, 53, 79)
AGENTS = ("oracle", "naive", "secondary-only")
BOOTSTRAP_DRAWS = 2_000


def bootstrap_auroc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resample whole task IDs, keeping their dependent fault episodes together."""
    clusters = defaultdict(list)
    for row in rows:
        confidence = row.get("confidence")
        if row["terminal"] == "submit" and type(confidence) in (int, float) and 0 <= confidence <= 1:
            clusters[(row["world_seed"], row["task_id"])].append(
                (confidence, row["outcome"] == "correct")
            )
    tasks = [clusters[key] for key in sorted(clusters)]
    rng = random.Random(20261111)
    estimates = []
    for _ in range(BOOTSTRAP_DRAWS):
        pairs = [pair for task in rng.choices(tasks, k=len(tasks)) for pair in task]
        estimate = auroc([c for c, _ in pairs], [ok for _, ok in pairs])
        if math.isfinite(estimate):
            estimates.append(estimate)
    estimates.sort()
    interval = None
    if len(estimates) >= 0.95 * BOOTSTRAP_DRAWS:
        interval = [
            estimates[int((len(estimates) - 1) * quantile)] for quantile in (0.025, 0.975)
        ]
    return {"tasks": len(tasks), "valid_draws": len(estimates), "interval_95": interval}


def published_audit(path: Path) -> dict[str, Any]:
    all_rows = load_rows(path)
    rows = usable(all_rows)
    by_agent = defaultdict(list)
    for row in rows:
        if family_of(row["condition"]) == "silent":
            by_agent[row["agent"]].append(row)
    agents = {}
    for agent, silent in sorted(by_agent.items()):
        calibration = calibration_of(silent)
        auc = calibration["auroc"]
        agents[agent] = {
            "silent_episodes": len(silent),
            "faults_fired": sum(r["fault_fired"] for r in silent),
            "resolved": sum(r["resolved_correctly"] for r in silent) / len(silent),
            "valid_confidence_submissions": calibration["n_submitted"],
            "auroc": auc if math.isfinite(auc) else None,
            "bootstrap": bootstrap_auroc(silent),
            "routing": routing_of(silent),
        }
    return {"episodes": len(all_rows), "errors": len(all_rows) - len(rows), "agents": agents}


def world_audit(out: Path) -> list[dict[str, Any]]:
    results = []
    for seed in WORLD_SEEDS:
        run_id = f"world-{seed}"
        manifest = run_grid(
            RunConfig(
                run_id=run_id,
                out_dir=str(out),
                world_seed=seed,
                tasks_per_type=16,
                budget_usd=0,
                save_traces="none",
                agents=[AgentSpec(kind=agent) for agent in AGENTS],
            ),
            verbose=False,
        )
        rows = load_rows(out / run_id / "episodes.jsonl")
        assert manifest["episodes_planned"] == len(rows) and not manifest["episodes_with_errors"]
        for agent in AGENTS:
            arm = [r for r in rows if r["agent"] == agent]
            silent = [r for r in arm if family_of(r["condition"]) == "silent"]
            results.append({
                "world_seed": seed,
                "agent": agent,
                "episodes": len(arm),
                "resolved": sum(r["resolved_correctly"] for r in arm),
                "silent_episodes": len(silent),
                "silent_resolved": sum(r["resolved_correctly"] for r in silent) / len(silent),
                "silent_unsafe": sum(r["unsafe"] for r in silent) / len(silent),
            })
    return results


def render(published: dict[str, Any], worlds: list[dict[str, Any]]) -> str:
    lines = [
        "# ToolFault experiment audit",
        "",
        f"Published grid contains {published['episodes']:,} episodes, "
        f"including {published['errors']} endpoint errors.",
        "",
        "## Silent faults in the published model runs",
        "",
        "The percentile intervals resample task IDs 2,000 times and preserve the fault "
        "episodes within each task. They quantify variation across tasks in the published "
        "world. AUROC measures answer correctness among valid confidence submissions. "
        "Review and unsafe rates use all scorable silent episodes.",
        "",
        "| Agent | Fault fired | Valid submissions | AUROC [95% interval] | Review, no gate → ≥0.95 | Unsafe, no gate → ≥0.95 |",
        "|---|---|---|---|---|---|",
    ]
    for agent, data in published["agents"].items():
        baseline, *_, gate = data["routing"]
        auc = "n/a" if data["auroc"] is None else f"{data['auroc']:.3f}"
        if interval := data["bootstrap"]["interval_95"]:
            auc += f" [{interval[0]:.3f}, {interval[1]:.3f}]"
        lines.append(
            f"| {agent} | {data['faults_fired']}/{data['silent_episodes']} | "
            f"{data['valid_confidence_submissions']} | {auc} | "
            f"{baseline['review_rate']:.1%} → {gate['review_rate']:.1%} | "
            f"{baseline['unsafe_rate']:.1%} → {gate['unsafe_rate']:.1%} |"
        )
    lines.extend([
        "",
        "The threshold is a retrospective illustration. Review rates assume any "
        "diverted submission goes to a reviewer. The audit does not measure human "
        "review time or quality.",
        "",
        "## Scripted policies across six worlds",
        "",
        "| World seed | Agent | Resolved, all | Resolved, silent | Unsafe, silent |",
        "|---|---|---|---|---|",
    ])
    for row in worlds:
        lines.append(
            f"| {row['world_seed']} | {row['agent']} | {row['resolved']}/{row['episodes']} | "
            f"{row['silent_resolved']:.1%} | {row['silent_unsafe']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, default=Path("results/bench-v1/episodes.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/audit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    published = published_audit(args.published)
    worlds = world_audit(args.out)
    (args.out / "audit.json").write_text(
        json.dumps({"published": published, "worlds": worlds}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report = args.out / "AUDIT.md"
    report.write_text(render(published, worlds), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
