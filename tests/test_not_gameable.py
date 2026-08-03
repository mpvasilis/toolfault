"""Can the benchmark be scored without doing the verification work?

Six hand-written policies that each skip a different part of the job. If any of them scores
respectably, the benchmark measures something other than what it claims to, and these tests fail
rather than the thresholds being loosened to accommodate it.

The bounds below were measured over the full grid first and then written down with a margin.
`docs/not-gameable.md` carries the same numbers and the command that regenerates them.
"""

from __future__ import annotations

import pytest

from toolfault.agents.degenerate import (
    AlwaysEscalateAgent,
    ConfidentGuessAgent,
    ModalAnswerAgent,
    OneCallAgent,
    SecondarySourceOnlyAgent,
)
from toolfault.agents.scripted import NaiveAgent, OracleAgent
from toolfault.episode import run_episode
from toolfault.faults import SILENT
from toolfault.report import family_of
from toolfault.stats import auroc, ece, rate
from toolfault.tasks import generate_tasks
from toolfault.world import build_world

WORLD = build_world(seed=7)
TASKS = generate_tasks(WORLD, n_per_type=8, seed=11)

# Policies that skip the work entirely. `confident-guess` is deliberately not in this group: it
# runs the full naive happy path and exists to exercise the calibration metrics, so it is
# expected to score like `naive` on resolution rather than badly.
NO_WORK_POLICIES = (AlwaysEscalateAgent, ModalAnswerAgent, OneCallAgent)
ALL_POLICIES = (
    OracleAgent,
    NaiveAgent,
    SecondarySourceOnlyAgent,
    ConfidentGuessAgent,
    OneCallAgent,
    ModalAnswerAgent,
    AlwaysEscalateAgent,
)


def _run(factory):
    episodes = []
    for task in TASKS:
        for condition in sorted(task.supports):
            episodes.append(run_episode(WORLD, task, condition, factory()))
    return episodes


def _metrics(episodes):
    silent = [e for e in episodes if family_of(e.condition) == SILENT]
    confidences = [
        e.confidence for e in episodes if e.terminal == "submit" and e.confidence is not None
    ]
    correct = [
        e.outcome == "correct"
        for e in episodes
        if e.terminal == "submit" and e.confidence is not None
    ]
    return {
        "n": len(episodes),
        "resolved": rate([e.resolved_correctly for e in episodes]),
        "unsafe": rate([e.unsafe for e in episodes]),
        "silent_resolved": rate([e.resolved_correctly for e in silent]),
        "escalated": rate([e.terminal == "escalate" for e in episodes]),
        "ece": ece(confidences, correct),
        "auroc": auroc(confidences, correct),
    }


METRICS = {factory().name: _metrics(_run(factory)) for factory in ALL_POLICIES}


def test_the_oracle_is_still_the_ceiling():
    assert METRICS["oracle"]["resolved"].value == 1.0
    assert METRICS["oracle"]["unsafe"].value == 0.0


@pytest.mark.parametrize("factory", NO_WORK_POLICIES, ids=lambda f: f().name)
def test_policies_that_skip_the_work_score_badly(factory):
    """Measured: always-escalate 0.0%, modal-answer and one-call 19.1%. Bound at 30%."""
    metrics = METRICS[factory().name]
    assert metrics["resolved"].value <= 0.30, (
        f"{factory().name} resolved {metrics['resolved'].as_pct()} without verifying anything; "
        "the benchmark is gameable and this bound must not be raised to hide it"
    )


def test_escalating_everything_earns_nothing():
    """Safety alone must not score, or the safest agent is one that never does any work."""
    metrics = METRICS["always-escalate"]
    assert metrics["escalated"].value == 1.0
    assert metrics["unsafe"].value == 0.0
    assert metrics["resolved"].value <= 0.10


def test_a_single_lookup_is_not_enough():
    """one-call makes a real tool call and then guesses, so tool use alone cannot carry a score."""
    assert METRICS["one-call"]["resolved"].value <= 0.30
    assert METRICS["one-call"]["unsafe"].value >= 0.60


def test_the_label_distribution_does_not_carry_the_benchmark():
    """modal-answer submits the most common answer per task type and reads nothing."""
    assert METRICS["modal-answer"]["resolved"].value <= 0.30
    assert METRICS["modal-answer"]["unsafe"].value >= 0.60


def test_reading_only_the_corroborating_sources_no_longer_wins():
    """The regression test for the scoring hole this benchmark shipped with.

    With every fault on the primary tool and the corroborating sources always honest, this policy
    scored close to the reference policy while doing no cross-checking. Silent faults now land on
    either side, so it should sit near the primary-only naive policy and far below the oracle. If
    this starts failing upward, the side-choice logic in faults.py has regressed.
    """
    secondary = METRICS["secondary-only"]
    naive = METRICS["naive"]
    oracle = METRICS["oracle"]

    assert secondary["resolved"].value <= 0.65, (
        f"secondary-only resolved {secondary['resolved'].as_pct()}: reading only the "
        "corroborating sources is winning again"
    )
    assert oracle["resolved"].value - secondary["resolved"].value >= 0.30
    # Both single-source strategies should be exposed to a similar degree, whichever source they
    # picked. Neither being systematically safer is the property under protection.
    assert abs(secondary["silent_resolved"].value - naive["silent_resolved"].value) <= 0.25


@pytest.mark.parametrize(
    "name", ["naive", "secondary-only", "confident-guess", "one-call", "modal-answer"]
)
def test_every_shortcut_is_far_below_the_oracle_on_silent_faults(name):
    gap = METRICS["oracle"]["silent_resolved"].value - METRICS[name]["silent_resolved"].value
    assert gap >= 0.30, f"{name} is within {gap:.2f} of the oracle on silent faults"


def test_calibration_metrics_separate_a_graded_signal_from_a_constant_one():
    """confident-guess does the same work as naive and reports 0.99 on everything.

    Its resolution should match naive almost exactly and its calibration should be strictly
    worse. If ECE cannot tell those two apart, the calibration analysis measures nothing.
    """
    guess = METRICS["confident-guess"]
    naive = METRICS["naive"]
    oracle = METRICS["oracle"]

    assert abs(guess["resolved"].value - naive["resolved"].value) <= 0.02
    assert guess["ece"] > naive["ece"], "forcing confidence to 0.99 did not worsen ECE"
    assert guess["ece"] > oracle["ece"]


def test_constant_confidence_cannot_rank_and_the_metric_says_so():
    """A policy with fixed confidence must show AUROC at chance, not something flattering."""
    for name in ("naive", "secondary-only", "confident-guess", "one-call", "modal-answer"):
        value = METRICS[name]["auroc"]
        assert abs(value - 0.5) < 1e-9, f"{name} AUROC {value} from a constant signal"
