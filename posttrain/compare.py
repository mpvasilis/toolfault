"""Compare the base and tuned arms, split by trained against held-out conditions.

The comparison that matters is not the overall delta. It is whether a gain on the eight
trained conditions carries over to the six that were held out, because a model that only
improved where it was shown the answer has memorised a recovery, not learned to verify.

Pairing is per condition and per task id, so the permutation test compares the same company
under the same fault with and without the adapter. p-values are BH-corrected across the
fourteen conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toolfault.faults import FAULTS_BY_ID  # noqa: E402
from toolfault.report import family_of, usable  # noqa: E402
from toolfault.stats import (  # noqa: E402
    auroc,
    benjamini_hochberg,
    ece,
    paired_permutation,
    rate,
    risk_coverage,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import HELD_OUT_CONDITIONS, PREDICTIONS, TRAIN_CONDITIONS  # noqa: E402


def load(path: str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return usable(rows)


def by_task(rows: list[dict[str, Any]], condition: str, key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] == condition:
            buckets[row["task_id"]].append(1.0 if row[key] else 0.0)
    return {task: sum(v) / len(v) for task, v in buckets.items()}


def arm_summary(rows: list[dict[str, Any]], conditions: tuple[str, ...]) -> dict[str, Any]:
    subset = [r for r in rows if r["condition"] in conditions]
    confidences = [
        float(r["confidence"])
        for r in subset
        if r["terminal"] == "submit" and r.get("confidence") is not None
    ]
    correct = [
        r["outcome"] == "correct"
        for r in subset
        if r["terminal"] == "submit" and r.get("confidence") is not None
    ]
    curve = risk_coverage(confidences, correct)
    return {
        "n": len(subset),
        "resolved": rate([bool(r["resolved_correctly"]) for r in subset]),
        "unsafe": rate([bool(r["unsafe"]) for r in subset]),
        "escalated": rate([r["terminal"] == "escalate" for r in subset]),
        "flagged": rate([bool(r["flagged"]) for r in subset]),
        "mean_tool_calls": (
            sum(r["tool_call_count"] for r in subset) / len(subset) if subset else 0.0
        ),
        "ece": ece(confidences, correct),
        "auroc": auroc(confidences, correct),
        "risk_at_80_coverage": curve.risk_at(0.8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--out", default="results/posttrain")
    args = parser.parse_args()

    base = load(args.base)
    tuned = load(args.tuned)
    if not base or not tuned:
        print("one of the arms produced no scorable episodes", file=sys.stderr)
        return 1

    conditions = sorted({r["condition"] for r in base} & {r["condition"] for r in tuned})
    per_condition: dict[str, Any] = {}
    p_values: list[float] = []
    p_keys: list[str] = []

    for condition in conditions:
        base_by_task = by_task(base, condition, "resolved_correctly")
        tuned_by_task = by_task(tuned, condition, "resolved_correctly")
        shared = sorted(set(base_by_task) & set(tuned_by_task))
        cell: dict[str, Any] = {
            "family": family_of(condition),
            "split": "trained" if condition in TRAIN_CONDITIONS else "held_out",
            "recovery": (
                FAULTS_BY_ID[condition].recovery if condition in FAULTS_BY_ID else "n/a"
            ),
            "prediction": PREDICTIONS.get(condition),
            "paired_n": len(shared),
            "base_resolved": rate(
                [bool(r["resolved_correctly"]) for r in base if r["condition"] == condition]
            ),
            "tuned_resolved": rate(
                [bool(r["resolved_correctly"]) for r in tuned if r["condition"] == condition]
            ),
            "base_unsafe": rate([bool(r["unsafe"]) for r in base if r["condition"] == condition]),
            "tuned_unsafe": rate([bool(r["unsafe"]) for r in tuned if r["condition"] == condition]),
        }
        if len(shared) >= 4:
            treatment = [tuned_by_task[t] for t in shared]
            control = [base_by_task[t] for t in shared]
            cell["delta"] = sum(treatment) / len(treatment) - sum(control) / len(control)
            p = paired_permutation(treatment, control)
            cell["p_value"] = p
            p_values.append(p)
            p_keys.append(condition)
        per_condition[condition] = cell

    for key, q in zip(p_keys, benjamini_hochberg(p_values), strict=True):
        per_condition[key]["q_value"] = q

    trained_present = tuple(c for c in conditions if c in TRAIN_CONDITIONS)
    held_out_present = tuple(c for c in conditions if c in HELD_OUT_CONDITIONS)

    summary = {
        "arms": {
            "base": {
                "trained_conditions": arm_summary(base, trained_present),
                "held_out_conditions": arm_summary(base, held_out_present),
            },
            "tuned": {
                "trained_conditions": arm_summary(tuned, trained_present),
                "held_out_conditions": arm_summary(tuned, held_out_present),
            },
        },
        "per_condition": per_condition,
        "train_conditions": list(TRAIN_CONDITIONS),
        "held_out_conditions": list(HELD_OUT_CONDITIONS),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def jsonable(value: Any) -> Any:
        if hasattr(value, "with_ci"):
            return {"value": value.value, "n": value.n, "lo": value.lo, "hi": value.hi}
        if isinstance(value, dict):
            return {k: jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [jsonable(v) for v in value]
        if isinstance(value, float) and value != value:
            return None
        return value

    (out / "comparison.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Post-training: base against tuned",
        "",
        "## Split summary",
        "",
        "| split | arm | n | resolved | unsafe | escalated | flagged | tool calls | ECE | AUROC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split in ("trained_conditions", "held_out_conditions"):
        for arm in ("base", "tuned"):
            data = summary["arms"][arm][split]
            lines.append(
                "| {split} | {arm} | {n} | {res} | {uns} | {esc} | {flg} | {tc:.1f} | {ece} | {auroc} |".format(
                    split=split.replace("_conditions", ""),
                    arm=arm,
                    n=data["n"],
                    res=data["resolved"].with_ci(),
                    uns=data["unsafe"].as_pct(),
                    esc=data["escalated"].as_pct(),
                    flg=data["flagged"].as_pct(),
                    tc=data["mean_tool_calls"],
                    ece=f"{data['ece']:.3f}" if data["ece"] == data["ece"] else "n/a",
                    auroc=f"{data['auroc']:.3f}" if data["auroc"] == data["auroc"] else "n/a",
                )
            )

    lines += [
        "",
        "## Per condition",
        "",
        "| condition | split | family | n | base | tuned | delta | p | q | prediction |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for condition in sorted(per_condition, key=lambda c: (per_condition[c]["split"], c)):
        cell = per_condition[condition]
        delta = cell.get("delta")
        lines.append(
            "| `{c}` | {split} | {fam} | {n} | {b} | {t} | {d} | {p} | {q} | {pred} |".format(
                c=condition,
                split=cell["split"],
                fam=cell["family"],
                n=cell["paired_n"],
                b=cell["base_resolved"].with_ci(),
                t=cell["tuned_resolved"].with_ci(),
                d=f"{100 * delta:+.1f} pts" if delta is not None else "-",
                p=f"{cell['p_value']:.4f}" if "p_value" in cell else "-",
                q=f"{cell['q_value']:.4f}" if "q_value" in cell else "-",
                pred=cell.get("prediction") or "-",
            )
        )
    lines.append("")
    lines.append(
        "Predictions were written into the dataset manifest before training and are "
        "reproduced here unchanged."
    )
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
