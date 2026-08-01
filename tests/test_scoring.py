"""Answer parsing and outcome labelling."""

from __future__ import annotations

import pytest

from toolfault.scoring import answer_matches
from toolfault.tasks import Task, expected_action


def _task(answer: str, kind: str) -> Task:
    return Task(
        task_id="t",
        task_type="verify_entity" if kind == "categorical" else "revenue_eur",
        question="",
        answer=answer,
        answer_kind=kind,
        entity_id="E-00001",
        supports=frozenset({"clean"}),
        meta={},
    )


@pytest.mark.parametrize(
    "given",
    ["approve", "Approve", " APPROVE ", "approve.", '"approve"', "approved"],
)
def test_categorical_accepts_reasonable_spellings(given):
    assert answer_matches(_task("approve", "categorical"), given)


@pytest.mark.parametrize("given", ["reject", "insufficient_evidence", "", None, "maybe"])
def test_categorical_rejects_other_answers(given):
    assert not answer_matches(_task("approve", "categorical"), given)


def test_insufficient_evidence_aliases():
    task = _task("insufficient_evidence", "categorical")
    for given in ("insufficient_evidence", "insufficient evidence", "Insufficient-Evidence"):
        assert answer_matches(task, given)


@pytest.mark.parametrize(
    "given",
    ["12340000", "12,340,000", "EUR 12340000", "12340000 EUR", "12.340.000", " 12340000 "],
)
def test_amount_accepts_separators_and_currency_noise(given):
    assert answer_matches(_task("12340000", "eur_amount"), given)


def test_amount_tolerance_is_one_percent():
    task = _task("1000000", "eur_amount")
    assert answer_matches(task, "1005000")
    assert not answer_matches(task, "1020000")


def test_amount_rejects_a_thousandfold_error():
    """The unit_shift condition depends on this not being forgiven."""
    assert not answer_matches(_task("12340000", "eur_amount"), "12340")


def test_person_id_normalises_padding():
    task = Task(
        task_id="t",
        task_type="ubo_chain",
        question="",
        answer="P-00042",
        answer_kind="person_id",
        entity_id=None,
        supports=frozenset({"clean"}),
        meta={},
    )
    for given in ("P-00042", "p-00042", "P-42", "the owner is P-00042", "P00042"):
        assert answer_matches(task, given)
    assert not answer_matches(task, "P-00043")
    assert not answer_matches(task, "unknown")


def test_expected_action_only_demands_escalation_when_the_fault_fired():
    task = _task("approve", "categorical")
    assert expected_action(task, "auth_expired", fault_fired=True) == "escalate"
    assert expected_action(task, "auth_expired", fault_fired=False) == "answer"
    assert expected_action(task, "timeout", fault_fired=True) == "answer"
    assert expected_action(task, "clean", fault_fired=False) == "answer"
