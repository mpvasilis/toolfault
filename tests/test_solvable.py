"""The load-bearing test.

Every condition has to be winnable, and every fault has to actually fire. A benchmark
whose ceiling is unknown does not measure anything, and a fault that never lands would
show up as a reassuring "no effect" in the results.
"""

from __future__ import annotations

import pytest

from toolfault.agents.scripted import NaiveAgent, OracleAgent
from toolfault.episode import run_episode
from toolfault.faults import CONDITIONS, FAULTS, FAULTS_BY_ID, SILENT
from toolfault.tasks import generate_tasks
from toolfault.world import build_world

WORLD = build_world(seed=7)
TASKS = generate_tasks(WORLD, n_per_type=8, seed=11)


def _episodes(condition: str, agent_factory):
    for task in TASKS:
        if condition not in task.supports:
            continue
        yield task, run_episode(WORLD, task, condition, agent_factory())


@pytest.mark.parametrize("condition", CONDITIONS)
def test_oracle_resolves_every_condition(condition):
    failures = []
    total = 0
    for task, episode in _episodes(condition, OracleAgent):
        total += 1
        if not episode.resolved_correctly:
            failures.append(
                f"{task.task_id}: terminal={episode.terminal} answer={episode.answer!r} "
                f"truth={task.answer!r} flags={episode.flags}"
            )
    assert total > 0, f"no task supports {condition}"
    assert not failures, f"{condition} unresolved in {len(failures)}/{total}:\n" + "\n".join(
        failures[:6]
    )


@pytest.mark.parametrize("condition", [f.fault_id for f in FAULTS])
def test_every_fault_actually_fires(condition):
    """At least one supporting task must see the fault land under the reference policy."""
    fired = 0
    total = 0
    for _, episode in _episodes(condition, OracleAgent):
        total += 1
        fired += 1 if episode.fault_fired else 0
    assert total > 0
    assert fired > 0, f"{condition} never fired across {total} episodes"


def test_oracle_never_acts_unsafely():
    for condition in CONDITIONS:
        for task, episode in _episodes(condition, OracleAgent):
            assert not episode.unsafe, f"{task.task_id} {condition} scored unsafe"


def test_naive_is_perfect_on_the_clean_condition():
    """Otherwise a drop under a fault could not be attributed to the fault."""
    for task, episode in _episodes("clean", NaiveAgent):
        assert episode.resolved_correctly, (
            f"{task.task_id}: answer={episode.answer!r} truth={task.answer!r}"
        )


def test_silent_faults_break_the_uncorroborated_policy():
    """Each silent fault must move the naive policy. One that does not is not a fault."""
    inert = []
    for spec in FAULTS:
        if spec.family != SILENT:
            continue
        outcomes = [e.resolved_correctly for _, e in _episodes(spec.fault_id, NaiveAgent)]
        if outcomes and all(outcomes):
            inert.append(spec.fault_id)
    assert not inert, f"silent faults with no effect on the naive policy: {inert}"


def test_escalate_only_conditions_are_labelled_that_way():
    for condition in ("auth_expired", "contradiction"):
        assert FAULTS_BY_ID[condition].recovery == "escalate_only"
        seen = False
        for _, episode in _episodes(condition, OracleAgent):
            if episode.fault_fired:
                seen = True
                assert episode.terminal == "escalate"
                assert episode.expected_action == "escalate"
        assert seen


def test_oracle_flags_every_fault_that_corrupted_the_data():
    """Detection is machine-checkable, so it has to be exercised.

    Scoped to faults that changed what the data said or destroyed the evidence. A timeout
    or a 500 that a retry cleared produced correct data in the end, and
    ``data_quality_flags`` is a statement about the data, not a transport log. Flagging a
    successfully retried transient error would be noise, and demanding it would penalise
    the normal behaviour of retrying quietly.
    """
    for spec in FAULTS:
        if spec.recovery == "retry":
            continue
        flagged = [e.flagged for _, e in _episodes(spec.fault_id, OracleAgent) if e.fault_fired]
        assert flagged, f"{spec.fault_id} never fired"
        assert any(flagged), f"{spec.fault_id} corrupted the data but was never flagged"


def test_retried_transient_faults_are_not_flagged_as_data_problems():
    """The other half of the contract above: clean data after a retry means no flag."""
    for spec in FAULTS:
        if spec.recovery != "retry":
            continue
        for task, episode in _episodes(spec.fault_id, OracleAgent):
            assert not episode.flags, (
                f"{task.task_id} {spec.fault_id}: retry produced clean data but flagged "
                f"{episode.flags}"
            )


def test_clean_condition_produces_no_false_flags():
    for task, episode in _episodes("clean", OracleAgent):
        assert not episode.flags, f"{task.task_id} flagged {episode.flags} on clean data"
