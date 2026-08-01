"""Build the SFT dataset by distilling the reference policy.

The teacher is the ``oracle`` policy, not a frontier model. That is a deliberate choice:
on this task the oracle is better than any model at exactly the thing being taught, it is
free, and its behaviour is inspectable line by line, so a trace that looks wrong can be
traced to a branch in the source rather than to a sampling temperature.

Two splits, both frozen before any training runs.

**Fault split.** Eight conditions go into training and six are held out entirely. The
held-out set is not a random sample. It is chosen so that each held-out condition tests a
different kind of transfer:

| held out | family | what transfer it tests |
|---|---|---|
| ``rate_limit`` | loud | near transfer: a new error code, a recovery already trained |
| ``schema_drift`` | loud | a recovery mode absent from training entirely (``tolerate``) |
| ``stale_record`` | silent | a new tell, a recovery already trained (``second_source``) |
| ``contradiction`` | silent | a novel trigger for an action that was trained (escalation) |
| ``unit_shift`` | silent | a new magnitude tell, recovery already trained |
| ``injected_instruction`` | adversarial | an entire family never seen |

``auth_expired`` stays in training on purpose, because it is the only trained condition
where escalating is correct. Without it, ``contradiction`` would be asking the model to
invent an action it has never been shown, which tests nothing.

**World split.** Training traces come from a different world seed than the benchmark. No
company name, person id, officer name or filing amount is shared between the two, so a
model cannot score by memorising the registry.

The predictions are written into the manifest before training, so the results can be read
against what was expected rather than against what turned out convenient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toolfault.agents.scripted import OracleAgent  # noqa: E402
from toolfault.episode import run_episode  # noqa: E402
from toolfault.tasks import generate_tasks  # noqa: E402
from toolfault.tools import TOOL_SCHEMAS  # noqa: E402
from toolfault.world import build_world  # noqa: E402

TRAIN_CONDITIONS = (
    "clean",
    "timeout",
    "http_500",
    "malformed_json",
    "auth_expired",
    "empty_result",
    "truncated_list",
    "wrong_entity",
)

HELD_OUT_CONDITIONS = (
    "rate_limit",
    "schema_drift",
    "stale_record",
    "contradiction",
    "unit_shift",
    "injected_instruction",
)

PREDICTIONS = {
    "rate_limit": "recovers; the retry behaviour is trained and only the error code is new",
    "schema_drift": "little or no gain; the tolerate recovery mode is absent from training",
    "stale_record": "partial gain; the recovery is trained but the tell is new",
    "contradiction": "the interesting cell; escalation is trained but this trigger is not",
    "unit_shift": "partial gain; cross-checking the document is trained on other faults",
    "injected_instruction": "little or no gain; the family is entirely unseen",
}


def balance(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    """Cap each condition so the loud faults stop drowning out the silent ones.

    Without this the mix comes out about 88% loud, because every task supports every loud
    fault while the silent ones need a particular state of the world. A model trained on
    that mix learns to retry, which is the cheap half of the behaviour. Capping is done per
    condition and the survivors are chosen by a seeded shuffle, so the set stays
    reproducible.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row["meta"]["condition"], []).append(row)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for condition in sorted(buckets):
        bucket = buckets[condition]
        rng.shuffle(bucket)
        out.extend(bucket[:cap])
    out.sort(key=lambda r: (r["meta"]["condition"], r["meta"]["world_seed"], r["meta"]["task_id"]))
    return out


def build_examples(
    world_seed: int, tasks_per_type: int, task_seed: int, conditions: tuple[str, ...]
) -> list[dict[str, Any]]:
    world = build_world(seed=world_seed)
    tasks = generate_tasks(world, n_per_type=tasks_per_type, seed=task_seed)
    examples: list[dict[str, Any]] = []
    skipped = 0

    for task in tasks:
        for condition in conditions:
            if condition not in task.supports:
                continue
            episode = run_episode(world, task, condition, OracleAgent(), model="oracle")
            # Only demonstrations that were actually right become training data. The oracle
            # resolves everything, so this filter should never fire; it stays because a
            # dataset builder that cannot fail silently is worth the three lines.
            if not episode.resolved_correctly:
                skipped += 1
                continue
            examples.append(
                {
                    "messages": episode.messages,
                    "tools": [{"type": "function", "function": s} for s in TOOL_SCHEMAS],
                    "meta": {
                        "task_id": f"w{world_seed}-{task.task_id}",
                        "task_type": task.task_type,
                        "condition": condition,
                        "world_seed": world_seed,
                        "terminal": episode.terminal,
                        "tool_calls": episode.tool_call_count,
                        "fault_fired": episode.fault_fired,
                        "flags": episode.flags,
                    },
                }
            )
    if skipped:
        print(f"[posttrain] dropped {skipped} unresolved demonstrations", file=sys.stderr)
    return examples


def digest(rows: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(json.dumps(row["messages"], sort_keys=True, ensure_ascii=False).encode())
    return hasher.hexdigest()[:16]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="posttrain/data")
    parser.add_argument(
        "--world-seeds",
        default="23,29,31",
        help="training worlds, comma separated; none may be the benchmark world",
    )
    parser.add_argument("--benchmark-world-seed", type=int, default=7)
    parser.add_argument("--tasks-per-type", type=int, default=40)
    parser.add_argument("--task-seed", type=int, default=101)
    parser.add_argument(
        "--cap-per-condition",
        type=int,
        default=110,
        help="maximum demonstrations per condition, to keep the loud faults from dominating",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=5)
    args = parser.parse_args()

    world_seeds = [int(s) for s in args.world_seeds.split(",") if s.strip()]
    if args.benchmark_world_seed in world_seeds:
        parser.error("no training world may be the benchmark world; that is the whole point")

    out = Path(args.out)
    raw: list[dict[str, Any]] = []
    for seed in world_seeds:
        produced = build_examples(seed, args.tasks_per_type, args.task_seed, TRAIN_CONDITIONS)
        print(f"[posttrain] world {seed}: {len(produced)} demonstrations", file=sys.stderr)
        raw.extend(produced)
    # Several worlds, because the silent faults need a particular state of the world and one
    # registry does not contain enough of it. Three worlds triple the supply of the rare
    # conditions without touching the benchmark world.
    rows = balance(raw, args.cap_per_condition, args.split_seed)
    if not rows:
        print("[posttrain] no demonstrations produced", file=sys.stderr)
        return 1

    # Validation is split by task id, never by episode. Splitting by episode would put the
    # same company in train and val under two different faults, and the reported validation
    # loss would be measuring memorisation.
    task_ids = sorted({row["meta"]["task_id"] for row in rows})
    rng = random.Random(args.split_seed)
    rng.shuffle(task_ids)
    n_val = max(1, int(len(task_ids) * args.val_fraction))
    val_ids = set(task_ids[:n_val])

    train = [r for r in rows if r["meta"]["task_id"] not in val_ids]
    val = [r for r in rows if r["meta"]["task_id"] in val_ids]

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)

    by_condition: dict[str, int] = {}
    for row in train:
        key = row["meta"]["condition"]
        by_condition[key] = by_condition.get(key, 0) + 1
    escalations = sum(1 for r in train if r["meta"]["terminal"] == "escalate")

    manifest = {
        "teacher": "toolfault oracle reference policy",
        "training_world_seeds": world_seeds,
        "benchmark_world_seed": args.benchmark_world_seed,
        "cap_per_condition": args.cap_per_condition,
        "demonstrations_before_capping": len(raw),
        "task_seed": args.task_seed,
        "tasks_per_type": args.tasks_per_type,
        "train_conditions": list(TRAIN_CONDITIONS),
        "held_out_conditions": list(HELD_OUT_CONDITIONS),
        "predictions_before_training": PREDICTIONS,
        "train_examples": len(train),
        "val_examples": len(val),
        "val_task_ids": sorted(val_ids),
        "train_by_condition": by_condition,
        "train_escalations": escalations,
        "train_digest": digest(train),
        "val_digest": digest(val),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
