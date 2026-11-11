"""Integrity checks for paired base-versus-tuned evaluation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from toolfault.episode import Episode

COMPARE = Path(__file__).resolve().parents[1] / "posttrain" / "compare.py"


def _row(task_id="verify-0000", task_seed=11):
    return Episode(
        run_id="test",
        agent="test-agent",
        model="test-model",
        world_seed=7,
        task_seed=task_seed,
        task_id=task_id,
        task_type="verify_entity",
        condition="clean",
        terminal="submit",
        outcome="correct",
        resolved_correctly=True,
    ).to_row()


def _compare(tmp_path, base, tuned):
    for name, rows in (("base", base), ("tuned", tuned)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return subprocess.run(
        [
            sys.executable, str(COMPARE),
            "--base", str(tmp_path / "base.jsonl"),
            "--tuned", str(tmp_path / "tuned.jsonl"),
            "--out", str(tmp_path / "comparison"),
        ],
        capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("field,value", [
    ("world_seed", 19), ("task_seed", 19), ("task_id", "verify-0001"),
    ("condition", "timeout"), ("repeat", 1),
    ("ground_truth", "different answer"), ("task_type", "other-task"),
])
def test_compare_rejects_mismatched_cells(tmp_path, field, value):
    base, tuned = _row(), _row()
    tuned[field] = value
    result = _compare(tmp_path, [base], [tuned])
    assert result.returncode == 2, result.stderr
    assert "identical tasks, conditions and repeat IDs" in result.stderr
    assert not (tmp_path / "comparison").exists()


def test_compare_rejects_duplicate_episode_identities(tmp_path):
    result = _compare(tmp_path, [_row(), _row()], [_row()])
    assert result.returncode == 2
    assert "duplicate episode identities" in result.stderr


@pytest.mark.parametrize("omit_seed", [False, True])
def test_compare_does_not_match_unknown_task_seed_to_zero(tmp_path, omit_seed):
    base = _row(task_seed=None)
    if omit_seed:
        del base["task_seed"]
    result = _compare(tmp_path, [base], [_row(task_seed=0)])
    assert result.returncode == 2, result.stderr
    assert "identical tasks, conditions and repeat IDs" in result.stderr


@pytest.mark.parametrize("seeds", [(11, 19), (None, 0)])
def test_compare_counts_distinct_task_seeds_separately(tmp_path, seeds):
    rows = [_row(task_seed=seed) for seed in seeds]
    result = _compare(tmp_path, rows, rows)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "comparison" / "comparison.json").read_text())
    assert summary["per_condition"]["clean"]["paired_n"] == 2


@pytest.mark.parametrize("base_error,tuned_error", [(True, False), (False, True), (True, True)])
def test_compare_excludes_error_pairs_from_both_arms(tmp_path, base_error, tuned_error):
    base = [_row("verify-0000"), _row("verify-0001")]
    tuned = [_row("verify-0000"), _row("verify-0001")]
    for rows, failed in ((base, base_error), (tuned, tuned_error)):
        if failed:
            rows[1]["outcome"] = "error"
    result = _compare(tmp_path, base, tuned)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "comparison" / "comparison.json").read_text())
    assert summary["pairs_excluded_on_error"] == 1
    assert summary["per_condition"]["clean"]["paired_n"] == 1
    for arm in ("base", "tuned"):
        assert summary["arms"][arm]["trained_conditions"]["n"] == 1
