"""Regression tests for persisted episode identities and reporting denominators."""

from pathlib import Path

import httpx
import pytest

from toolfault.episode import Episode, _as_float
from toolfault.report import pass_hat_k, routing_of, summarize
from toolfault.runner import AgentSpec, RunConfig, load_rows, run_grid


def test_run_persists_each_repeat_and_protects_artifacts(tmp_path):
    config = RunConfig(
        out_dir=str(tmp_path),
        tasks_per_type=1,
        conditions=("clean",),
        repeats=3,
        save_traces="all",
        agents=[AgentSpec(kind="oracle")],
    )
    manifest = run_grid(config, verbose=False)
    root = Path(config.out_dir) / config.run_id
    rows = load_rows(root / "episodes.jsonl")
    assert manifest["episodes_completed"] == len(rows) == 12
    assert len(list((root / "traces").glob("*.json"))) == 12
    assert [row["repeat"] for row in rows[:3]] == [0, 1, 2]
    assert pass_hat_k(rows, 3).successes == 4
    assert pass_hat_k(rows[1:], 3).n == 3
    assert pass_hat_k([rows[0], rows[0]], 2) is None
    with pytest.raises(FileExistsError):
        run_grid(config, verbose=False)


def test_run_checks_remote_credentials_before_creating_artifacts(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = RunConfig(
        out_dir=str(tmp_path),
        tasks_per_type=1,
        agents=[AgentSpec(kind="openrouter", model="example/model")],
    )
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        run_grid(config, verbose=False)
    assert not (tmp_path / config.run_id).exists()


def _report_row(task_seed, task_id, condition, resolved, repeat=0):
    return Episode(
        run_id="test",
        agent="test-agent",
        model="test-model",
        world_seed=7,
        task_seed=task_seed,
        task_id=task_id,
        task_type="verify_entity",
        condition=condition,
        repeat=repeat,
        terminal="submit",
        outcome="correct" if resolved else "incorrect",
        resolved_correctly=resolved,
        unsafe=not resolved,
    ).to_row()


@pytest.mark.parametrize("seeds", [(11, 19), (None, 0)])
def test_pass_hat_k_keeps_task_seeds_distinct(seeds):
    rows = [
        _report_row(seed, "verify-0000", "clean", index == 0, repeat)
        for index, seed in enumerate(seeds)
        for repeat in (0, 1)
    ]
    result = pass_hat_k(rows, 2)
    assert result.n == 2
    assert result.successes == 1
    assert pass_hat_k(list(reversed(rows)), 2) == result


def test_pass_hat_k_does_not_combine_incomplete_repeats_across_task_seeds():
    rows = [
        _report_row(11, "verify-0000", "clean", True, repeat=0),
        _report_row(19, "verify-0000", "clean", True, repeat=1),
    ]
    assert pass_hat_k(rows, 2) is None


@pytest.mark.parametrize("seeds", [(11, 19), (None, 0)])
def test_paired_comparisons_keep_task_seeds_distinct(seeds):
    rows = [
        _report_row(seed, f"verify-{task}", condition, (index == 0) == (condition == "clean"))
        for index, seed in enumerate(seeds)
        for task in range(4)
        for condition in ("clean", "timeout")
    ]
    # Historical rows omit task_seed; unknown identity must not alias a real seed of zero.
    for row in rows:
        if row["task_seed"] is None:
            del row["task_seed"]
    cell = summarize(rows)["agents"]["test-agent"]["conditions"]["timeout"]
    assert cell["paired_n"] == 8
    assert cell["delta_resolved"] == 0


def test_paired_comparisons_require_a_control_with_the_same_task_seed():
    rows = [
        _report_row(seed, f"verify-{task}", condition, resolved)
        for task in range(4)
        for seed, condition, resolved in (
            (11, "clean", False), (11, "timeout", True), (19, "timeout", False)
        )
    ]
    cell = summarize(rows)["agents"]["test-agent"]["conditions"]["timeout"]
    assert cell["paired_n"] == 4
    assert cell["delta_resolved"] == 1


def test_errors_are_counted_before_filtering():
    summary = summarize([{"agent": "endpoint", "outcome": "error", "error": "transport"}])
    assert summary["episodes_scored"] == 0
    assert summary["episodes_with_errors"] == 1
    assert summary["errors_by_agent"] == {"endpoint": 1}


def test_routing_counts_unsafe_correct_guesses_and_existing_escalations():
    rows = [
        {"terminal": "submit", "confidence": 0.95, "outcome": "correct", "unsafe": True},
        {"terminal": "submit", "confidence": None, "unsafe": False},
        {"terminal": "escalate"},
        {"terminal": "no_terminal"},
    ]
    baseline, at_zero, *_ = routing_of(rows)
    assert baseline["coverage"] == 0.50
    assert baseline["review_rate"] == 0.25
    assert at_zero["coverage"] == 0.25
    assert at_zero["review_rate"] == 0.50
    assert at_zero["unsafe_rate"] == 0.25
    assert at_zero["unsafe_given_accepted"] == 1
    assert at_zero["unfinished_rate"] == 0.25


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1, None])
def test_invalid_confidence_stays_missing(value):
    assert _as_float(value) is None


def test_local_endpoint_errors_are_recorded_without_an_api_key(monkeypatch, tmp_path):
    def endpoint(request):
        assert "authorization" not in request.headers
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, content=b"<html>unexpected response</html>")

    client = httpx.Client

    monkeypatch.delenv("LOCAL_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(
        **kwargs, transport=httpx.MockTransport(endpoint)
    ))
    config = RunConfig(
        out_dir=str(tmp_path),
        tasks_per_type=1,
        task_types=("verify_entity",),
        conditions=("clean",),
        budget_usd=0,
        agents=[AgentSpec(kind="local", model="mock-model")],
    )
    manifest = run_grid(config, verbose=False)
    rows = load_rows(tmp_path / config.run_id / "episodes.jsonl")
    assert manifest["episodes_with_errors"] == 1
    assert manifest["harness_exceptions"] == 0
    assert rows[0]["error"] == "invalid JSON in model endpoint response"
