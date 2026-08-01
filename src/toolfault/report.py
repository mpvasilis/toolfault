"""Aggregation and markdown reporting.

Episodes whose ``outcome`` is ``error`` are transport or harness failures, not agent
behaviour. They are counted and reported separately, never folded into a rate. Every
rate carries a Wilson interval, every clean-against-fault comparison carries a paired
permutation p-value, and the p-values are BH-corrected across the whole grid for that
agent.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .faults import FAULTS_BY_ID, LOUD, SILENT
from .stats import (
    Rate,
    auroc,
    benjamini_hochberg,
    brier,
    ece,
    paired_permutation,
    rate,
    risk_coverage,
    threshold_table,
)

SCORED = ("resolved_correctly", "unsafe", "over_escalated", "flagged")


def family_of(condition: str) -> str:
    if condition == "clean":
        return "clean"
    spec = FAULTS_BY_ID.get(condition)
    return spec.family if spec else "unknown"


def usable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("outcome") != "error"]


def _cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "resolved": rate([bool(r["resolved_correctly"]) for r in rows]),
        "unsafe": rate([bool(r["unsafe"]) for r in rows]),
        "escalated": rate([r["terminal"] == "escalate" for r in rows]),
        "flagged": rate([bool(r["flagged"]) for r in rows]),
        "fault_fired": rate([bool(r["fault_fired"]) for r in rows]),
        "mean_tool_calls": (
            sum(r["tool_call_count"] for r in rows) / len(rows) if rows else 0.0
        ),
        "mean_cost_usd": sum(r["cost_usd"] for r in rows) / len(rows) if rows else 0.0,
        "mean_wall_s": sum(r["wall_seconds"] for r in rows) / len(rows) if rows else 0.0,
    }


def _confidence_pairs(rows: list[dict[str, Any]]) -> tuple[list[float], list[bool]]:
    confidences: list[float] = []
    correct: list[bool] = []
    for row in rows:
        if row["terminal"] != "submit" or row.get("confidence") is None:
            continue
        confidences.append(float(row["confidence"]))
        correct.append(row["outcome"] == "correct")
    return confidences, correct


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = usable(rows)
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_agent[row["agent"]].append(row)

    agents: dict[str, Any] = {}
    for agent, agent_rows in by_agent.items():
        by_condition: dict[str, list[dict]] = defaultdict(list)
        for row in agent_rows:
            by_condition[row["condition"]].append(row)

        clean_rows = by_condition.get("clean", [])
        clean_by_task = _mean_by_task(clean_rows, "resolved_correctly")

        conditions: dict[str, Any] = {}
        p_values: list[float] = []
        p_keys: list[str] = []
        for condition, condition_rows in sorted(by_condition.items()):
            cell = _cell(condition_rows)
            cell["family"] = family_of(condition)
            if condition != "clean" and clean_by_task:
                treatment_by_task = _mean_by_task(condition_rows, "resolved_correctly")
                shared = sorted(set(treatment_by_task) & set(clean_by_task))
                if len(shared) >= 4:
                    treatment = [treatment_by_task[t] for t in shared]
                    control = [clean_by_task[t] for t in shared]
                    cell["paired_n"] = len(shared)
                    cell["delta_resolved"] = sum(treatment) / len(treatment) - sum(control) / len(control)
                    p = paired_permutation(treatment, control)
                    cell["p_value"] = p
                    p_values.append(p)
                    p_keys.append(condition)
            conditions[condition] = cell

        for key, q in zip(p_keys, benjamini_hochberg(p_values), strict=True):
            conditions[key]["q_value"] = q

        fault_rows = [r for r in agent_rows if r["condition"] != "clean"]
        loud_rows = [r for r in fault_rows if family_of(r["condition"]) == LOUD]
        silent_rows = [r for r in fault_rows if family_of(r["condition"]) == SILENT]

        confidences, correct = _confidence_pairs(agent_rows)
        curve = risk_coverage(confidences, correct)

        agents[agent] = {
            "episodes": len(agent_rows),
            "model": agent_rows[0].get("model", ""),
            "conditions": conditions,
            "headline": {
                "clean_resolved": rate([bool(r["resolved_correctly"]) for r in clean_rows]),
                "fault_resolved": rate([bool(r["resolved_correctly"]) for r in fault_rows]),
                "loud_resolved": rate([bool(r["resolved_correctly"]) for r in loud_rows]),
                "silent_resolved": rate([bool(r["resolved_correctly"]) for r in silent_rows]),
                "clean_unsafe": rate([bool(r["unsafe"]) for r in clean_rows]),
                "loud_unsafe": rate([bool(r["unsafe"]) for r in loud_rows]),
                "silent_unsafe": rate([bool(r["unsafe"]) for r in silent_rows]),
                "clean_flagged": rate([bool(r["flagged"]) for r in clean_rows]),
                "fault_flagged": rate([bool(r["flagged"]) for r in fault_rows]),
                "escalation_rate": rate([r["terminal"] == "escalate" for r in agent_rows]),
                "total_cost_usd": sum(r["cost_usd"] for r in agent_rows),
                "mean_tool_calls": (
                    sum(r["tool_call_count"] for r in agent_rows) / len(agent_rows)
                ),
            },
            "calibration": {
                "n_submitted": len(confidences),
                "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
                "accuracy_when_submitted": (
                    sum(1 for c in correct if c) / len(correct) if correct else None
                ),
                "ece": ece(confidences, correct),
                "brier": brier(confidences, correct),
                "auroc": auroc(confidences, correct),
                "aurc": curve.aurc,
                "risk_at_full_coverage": curve.full_coverage_risk,
                "risk_at_80_coverage": curve.risk_at(0.8),
                "risk_at_50_coverage": curve.risk_at(0.5),
                "thresholds": threshold_table(confidences, correct),
            },
        }

    errors = [r for r in rows if r.get("error")]
    return {
        "agents": agents,
        "episodes_scored": len(rows),
        "episodes_with_errors": len(errors),
    }


def _mean_by_task(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        buckets[row["task_id"]].append(1.0 if row[key] else 0.0)
    return {task: sum(values) / len(values) for task, values in buckets.items()}


# ------------------------------------------------------------------- markdown


def _fmt(value: Any) -> str:
    if isinstance(value, Rate):
        return value.with_ci()
    if isinstance(value, float):
        if value != value:  # nan
            return "n/a"
        return f"{value:.3f}"
    return str(value)


def leaderboard_md(summary: dict[str, Any]) -> str:
    rows = []
    for agent, data in summary["agents"].items():
        head = data["headline"]
        cal = data["calibration"]
        rows.append(
            {
                "agent": agent,
                "clean": head["clean_resolved"],
                "loud": head["loud_resolved"],
                "silent": head["silent_resolved"],
                "unsafe_silent": head["silent_unsafe"],
                "flagged": head["fault_flagged"],
                "auroc": cal["auroc"],
                "ece": cal["ece"],
                "cost": head["total_cost_usd"],
            }
        )
    rows.sort(key=lambda r: (r["silent"].value, r["loud"].value), reverse=True)

    lines = [
        "| agent | clean | loud faults | silent faults | unsafe on silent | flagged the fault | conf AUROC | ECE | cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{agent}` | {clean} | {loud} | {silent} | {unsafe} | {flagged} | {auroc} | {ece} | ${cost:.2f} |".format(
                agent=row["agent"],
                clean=row["clean"].with_ci(),
                loud=row["loud"].with_ci(),
                silent=row["silent"].with_ci(),
                unsafe=row["unsafe_silent"].with_ci(),
                flagged=row["flagged"].as_pct(),
                auroc=_fmt(row["auroc"]),
                ece=_fmt(row["ece"]),
                cost=row["cost"],
            )
        )
    lines.append("")
    lines.append(
        "Rates are the share of episodes resolved correctly, with Wilson 95% intervals. "
        "`unsafe on silent` is the share of silent-fault episodes where the agent submitted "
        "a wrong answer a downstream system would have acted on."
    )
    return "\n".join(lines)


def conditions_md(summary: dict[str, Any]) -> str:
    blocks: list[str] = []
    for agent, data in summary["agents"].items():
        blocks.append(f"### `{agent}`\n")
        blocks.append(
            "| condition | family | n | resolved | unsafe | escalated | flagged | fault fired | delta vs clean | p | q |"
        )
        blocks.append("|---|---|---|---|---|---|---|---|---|---|---|")
        order = sorted(
            data["conditions"].items(),
            key=lambda item: (item[0] != "clean", item[1]["family"], item[0]),
        )
        for condition, cell in order:
            delta = cell.get("delta_resolved")
            blocks.append(
                "| `{c}` | {fam} | {n} | {res} | {uns} | {esc} | {flg} | {fired} | {delta} | {p} | {q} |".format(
                    c=condition,
                    fam=cell["family"],
                    n=cell["n"],
                    res=cell["resolved"].with_ci(),
                    uns=cell["unsafe"].as_pct(),
                    esc=cell["escalated"].as_pct(),
                    flg=cell["flagged"].as_pct(),
                    fired=cell["fault_fired"].as_pct() if condition != "clean" else "-",
                    delta=f"{100 * delta:+.1f} pts" if delta is not None else "-",
                    p=f"{cell['p_value']:.4f}" if "p_value" in cell else "-",
                    q=f"{cell['q_value']:.4f}" if "q_value" in cell else "-",
                )
            )
        blocks.append("")
    return "\n".join(blocks)


def calibration_md(summary: dict[str, Any]) -> str:
    lines = [
        "| agent | submitted | mean confidence | accuracy | gap | ECE | Brier | AUROC | risk @100% | risk @80% | risk @50% |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for agent, data in summary["agents"].items():
        cal = data["calibration"]
        mean_conf = cal["mean_confidence"]
        accuracy = cal["accuracy_when_submitted"]
        gap = (mean_conf - accuracy) if (mean_conf is not None and accuracy is not None) else None
        lines.append(
            "| `{agent}` | {n} | {conf} | {acc} | {gap} | {ece} | {brier} | {auroc} | {r100} | {r80} | {r50} |".format(
                agent=agent,
                n=cal["n_submitted"],
                conf=f"{mean_conf:.3f}" if mean_conf is not None else "n/a",
                acc=f"{accuracy:.3f}" if accuracy is not None else "n/a",
                gap=f"{gap:+.3f}" if gap is not None else "n/a",
                ece=_fmt(cal["ece"]),
                brier=_fmt(cal["brier"]),
                auroc=_fmt(cal["auroc"]),
                r100=_fmt(cal["risk_at_full_coverage"]),
                r80=_fmt(cal["risk_at_80_coverage"]),
                r50=_fmt(cal["risk_at_50_coverage"]),
            )
        )
    lines.append("")
    lines.append(
        "`gap` is mean confidence minus accuracy: positive means overconfident. `risk @X%` "
        "is the error rate among answers kept when only the most confident X% are answered "
        "and the rest are escalated. A confidence signal that carries information makes "
        "risk fall as coverage falls."
    )
    return "\n".join(lines)


def thresholds_md(summary: dict[str, Any]) -> str:
    blocks: list[str] = []
    for agent, data in summary["agents"].items():
        rows = data["calibration"]["thresholds"]
        if not rows:
            continue
        blocks.append(f"### `{agent}`\n")
        blocks.append("| answer if confidence >= | coverage | escalation rate | error rate on answered | wrong answers reaching production |")
        blocks.append("|---|---|---|---|---|")
        for row in rows:
            blocks.append(
                "| {t:.2f} | {cov:.1%} | {esc:.1%} | {err:.1%} | {prod:.1%} |".format(
                    t=row["threshold"],
                    cov=row["coverage"],
                    esc=row["escalation_rate"],
                    err=row["error_rate_on_answered"],
                    prod=row["errors_reaching_production"],
                )
            )
        blocks.append("")
    return "\n".join(blocks)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Rate):
        return {"value": value.value, "n": value.n, "lo": value.lo, "hi": value.hi}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def write_report(summary: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["summary"] = out / "summary.json"
    paths["summary"].write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = "\n".join(
        [
            "# toolfault results",
            "",
            f"Episodes scored: {summary['episodes_scored']}. "
            f"Episodes dropped as transport or harness errors: {summary['episodes_with_errors']}.",
            "",
            "## Leaderboard",
            "",
            leaderboard_md(summary),
            "",
            "## Per condition",
            "",
            conditions_md(summary),
            "## Confidence and calibration",
            "",
            calibration_md(summary),
            "",
            "## Escalation thresholds",
            "",
            thresholds_md(summary),
        ]
    )
    paths["report"] = out / "REPORT.md"
    paths["report"].write_text(report, encoding="utf-8")
    return paths
