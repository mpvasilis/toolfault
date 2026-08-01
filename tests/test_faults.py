"""The taxonomy has to stay coherent and the injector has to touch nothing it should not."""

from __future__ import annotations

import re

import pytest

from toolfault.faults import (
    ADVERSARIAL,
    CONDITIONS,
    FAULTS,
    LOUD,
    PRIMARY_TOOL,
    SILENT,
    TASK_TYPES,
    FaultInjector,
    choose_side,
    resolve_targets,
)
from toolfault.tools import READ_TOOLS, Registry
from toolfault.world import build_world

WORLD = build_world(seed=7)
RECOVERIES = {"retry", "second_source", "tolerate", "escalate_only", "ignore"}


def test_taxonomy_is_well_formed():
    assert len(FAULTS) == 13
    assert len({f.fault_id for f in FAULTS}) == 13
    assert CONDITIONS[0] == "clean"
    assert len(CONDITIONS) == 14
    for spec in FAULTS:
        assert spec.family in {LOUD, SILENT, ADVERSARIAL}
        assert spec.recovery in RECOVERIES
        assert spec.applies_to, spec.fault_id
        assert set(spec.applies_to) <= set(TASK_TYPES)
        assert spec.summary and spec.recovery_hint


def test_family_sizes_match_the_documented_split():
    assert sum(1 for f in FAULTS if f.family == LOUD) == 6
    assert sum(1 for f in FAULTS if f.family == SILENT) == 6
    assert sum(1 for f in FAULTS if f.family == ADVERSARIAL) == 1


def test_every_target_is_a_real_tool():
    for spec in FAULTS:
        for task_type in spec.applies_to:
            for tool in resolve_targets(spec, task_type):
                assert tool in READ_TOOLS, f"{spec.fault_id} targets unknown tool {tool}"


def test_primary_tools_are_defined_for_every_task_type():
    assert set(PRIMARY_TOOL) == set(TASK_TYPES)
    for tools in PRIMARY_TOOL.values():
        assert all(tool in READ_TOOLS for tool in tools)


def test_clean_condition_touches_nothing():
    injector = FaultInjector(world=WORLD, condition="clean", task_type="verify_entity")
    registry = Registry(world=WORLD, injector=injector)
    entity_id = next(iter(WORLD.entities))
    record = registry.call("get_entity", {"entity_id": entity_id})
    assert record["status"] == WORLD.entities[entity_id].status
    assert record["cache"] == "MISS"
    assert not injector.fired
    assert not injector.events


def test_loud_faults_fire_once_then_stop():
    entity_id = next(iter(WORLD.entities))
    for condition in ("timeout", "http_500", "rate_limit"):
        injector = FaultInjector(world=WORLD, condition=condition, task_type="verify_entity")
        registry = Registry(world=WORLD, injector=injector)
        first = registry.call("get_entity", {"entity_id": entity_id})
        second = registry.call("get_entity", {"entity_id": entity_id})
        assert "error" in first, condition
        assert "error" not in second, f"{condition} did not clear on retry"
        assert second["status"] == WORLD.entities[entity_id].status


def test_auth_expired_never_clears():
    entity_id = next(iter(WORLD.entities))
    injector = FaultInjector(world=WORLD, condition="auth_expired", task_type="verify_entity")
    registry = Registry(world=WORLD, injector=injector)
    for _ in range(4):
        result = registry.call("get_entity", {"entity_id": entity_id})
        assert result["error"] == "unauthorized"


def test_faults_do_not_touch_tools_they_do_not_target():
    entity_id = next(iter(WORLD.entities))
    injector = FaultInjector(world=WORLD, condition="unit_shift", task_type="revenue_eur")
    registry = Registry(world=WORLD, injector=injector)
    record = registry.call("get_entity", {"entity_id": entity_id})
    assert record["status"] == WORLD.entities[entity_id].status
    assert not injector.fired


def test_unit_shift_scales_by_exactly_one_thousand():
    entity = next(e for e in WORLD.entities.values() if e.filings)
    injector = FaultInjector(world=WORLD, condition="unit_shift", task_type="revenue_eur")
    registry = Registry(world=WORLD, injector=injector)
    payload = registry.call("get_filings", {"entity_id": entity.entity_id})
    assert injector.fired
    for row, filing in zip(payload["filings"], entity.filings, strict=True):
        assert abs(row["revenue"] * 1000 - filing.revenue) <= 1.0


def test_unit_shift_leaves_the_filed_document_honest():
    """The document is the second source, so it must not carry the fault too."""
    entity = next(e for e in WORLD.entities.values() if e.filings)
    injector = FaultInjector(world=WORLD, condition="unit_shift", task_type="revenue_eur")
    registry = Registry(world=WORLD, injector=injector)
    registry.call("get_filings", {"entity_id": entity.entity_id})
    document = registry.call("get_filing_document", {"document_id": entity.filings[0].document_id})
    assert f"{entity.filings[0].revenue:,}" in document["text"]


def test_truncated_list_drops_the_controlling_holder():
    entity = next(e for e in WORLD.entities.values() if len(e.holders) > 1)
    injector = FaultInjector(world=WORLD, condition="truncated_list", task_type="ubo_chain")
    registry = Registry(world=WORLD, injector=injector)
    payload = registry.call("get_ownership", {"entity_id": entity.entity_id})
    assert injector.fired
    assert all(h["pct"] <= 25.0 for h in payload["holders"])
    assert sum(h["pct"] for h in payload["holders"]) < 95.0


def _task_id_for_side(fault_id: str, side: int) -> str:
    """A task id that puts this fault on the requested side. Side choice is a digest."""
    for i in range(500):
        candidate = f"probe-{i:03d}"
        if choose_side(candidate, fault_id) == side:
            return candidate
    raise AssertionError(f"no task id found for {fault_id} side {side}")


def test_side_choice_is_deterministic_and_uses_both_sides():
    for fault_id in ("stale_record", "wrong_entity", "empty_result", "truncated_list"):
        sides = {choose_side(f"t-{i}", fault_id) for i in range(200)}
        assert sides == {0, 1}, f"{fault_id} never uses both sides"
        assert choose_side("t-7", fault_id) == choose_side("t-7", fault_id)


def test_stale_record_carries_a_tell_on_the_primary_side():
    entity = next(e for e in WORLD.entities.values() if e.prior_status != e.status)
    injector = FaultInjector(
        world=WORLD,
        condition="stale_record",
        task_type="verify_entity",
        task_id=_task_id_for_side("stale_record", 0),
    )
    registry = Registry(world=WORLD, injector=injector)
    payload = registry.call("get_entity", {"entity_id": entity.entity_id})
    assert injector.fired
    assert injector.side == "primary"
    assert payload["status"] == entity.prior_status
    assert payload["cache"] == "HIT"
    assert payload["retrieved_at"] < "2026"


def test_stale_record_carries_a_tell_on_the_corroborating_side():
    """The mirrored half of the fault. Without this the benchmark rewards reading only the
    archive, which was a real scoring hole in the first version."""
    entity = next(e for e in WORLD.entities.values() if e.prior_status != e.status)
    injector = FaultInjector(
        world=WORLD,
        condition="stale_record",
        task_type="verify_entity",
        task_id=_task_id_for_side("stale_record", 1),
    )
    registry = Registry(world=WORLD, injector=injector)
    live = registry.call("get_entity", {"entity_id": entity.entity_id})
    assert live["status"] == entity.status, "the primary must stay honest on this side"
    archive = registry.call(
        "get_entity_snapshot", {"entity_id": entity.entity_id, "as_of": "2026-06-30"}
    )
    assert injector.fired
    assert injector.side == "corroborating"
    assert archive["status"] == entity.prior_status
    assert archive["snapshot_as_of"] != "2026-06-30"


def test_empty_result_can_suppress_either_screen():
    entity = next(
        e for e in WORLD.entities.values() if WORLD.sanctions_hit(e.entity_id) is not None
    )
    hit_name = WORLD.sanctions_hit(entity.entity_id)

    primary = FaultInjector(
        world=WORLD, condition="empty_result", task_type="sanctions_screen",
        task_id=_task_id_for_side("empty_result", 0),
    )
    registry = Registry(world=WORLD, injector=primary)
    assert registry.call("screen_name", {"name": hit_name})["matched"] is False
    assert registry.call("screen_officer_batch", {"entity_id": entity.entity_id})["hits"]

    mirrored = FaultInjector(
        world=WORLD, condition="empty_result", task_type="sanctions_screen",
        task_id=_task_id_for_side("empty_result", 1),
    )
    registry = Registry(world=WORLD, injector=mirrored)
    assert registry.call("screen_name", {"name": hit_name})["matched"] is True
    assert registry.call("screen_officer_batch", {"entity_id": entity.entity_id})["hits"] == []
    assert mirrored.fired


def test_truncated_list_can_cut_either_source():
    entity = next(e for e in WORLD.entities.values() if len(e.holders) > 1)

    mirrored = FaultInjector(
        world=WORLD, condition="truncated_list", task_type="ubo_chain",
        task_id=_task_id_for_side("truncated_list", 1),
    )
    registry = Registry(world=WORLD, injector=mirrored)
    live = registry.call("get_ownership", {"entity_id": entity.entity_id})
    assert len(live["holders"]) == len(entity.holders), "the primary must stay honest here"
    document = registry.call("get_ownership_document", {"entity_id": entity.entity_id})
    assert mirrored.fired
    # The stated total no longer matches the rows listed, which is the tell.
    assert f"total holders on register: {len(entity.holders)}" in document["text"]
    listed = sum(1 for line in document["text"].splitlines() if "%" in line)
    assert listed < len(entity.holders)


def test_loud_faults_cannot_be_dodged_by_switching_source():
    """A loud fault fires on the first evidence-bearing call, whichever tool that is."""
    entity = next(iter(WORLD.entities.values()))
    injector = FaultInjector(
        world=WORLD, condition="http_500", task_type="verify_entity", task_id="dodge"
    )
    registry = Registry(world=WORLD, injector=injector)
    first = registry.call(
        "get_entity_snapshot", {"entity_id": entity.entity_id, "as_of": "2026-06-30"}
    )
    assert first.get("error") == "server_error"


def test_auth_expired_breaks_the_whole_credential():
    """Scoped to one endpoint it was recoverable by asking elsewhere, which defeated the
    point of having a terminal error in the set."""
    entity = next(iter(WORLD.entities.values()))
    injector = FaultInjector(
        world=WORLD, condition="auth_expired", task_type="verify_entity", task_id="creds"
    )
    registry = Registry(world=WORLD, injector=injector)
    assert registry.call("get_entity", {"entity_id": entity.entity_id})["error"] == "unauthorized"
    for tool, args in (
        ("get_entity_snapshot", {"entity_id": entity.entity_id, "as_of": "2026-06-30"}),
        ("get_filings", {"entity_id": entity.entity_id}),
        ("screen_officer_batch", {"entity_id": entity.entity_id}),
    ):
        assert registry.call(tool, args)["error"] == "unauthorized", tool


def test_identifiers_are_not_guessable_from_an_index():
    """Sequential ids let an agent construct a document id and skip the tool the fault was
    injected into."""
    ids = list(WORLD.entities)
    assert not any(i == f"E-{n:05d}" for n, i in enumerate(ids))
    document_ids = list(WORLD.documents)
    assert document_ids, "no documents generated"
    assert all(re.fullmatch(r"DOC-\d{6}", d) for d in document_ids), (
        "document ids must be opaque, carrying neither an entity index nor a filing year"
    )
    # A document id must not be reconstructible from the entity it belongs to plus a year,
    # which is what let an agent reach the corroborating source without the primary one.
    for entity in list(WORLD.entities.values())[:40]:
        for filing in entity.filings:
            assert entity.entity_id.split("-")[1] not in filing.document_id
            assert str(filing.year) not in filing.document_id
    # Two different worlds must not share identifiers.
    other = build_world(seed=11)
    assert not (set(WORLD.entities) & set(other.entities))


def test_contradiction_carries_no_tell():
    """That absence is what separates it from stale_record and forces an escalation."""
    entity = next(e for e in WORLD.entities.values() if e.status == "active")
    injector = FaultInjector(world=WORLD, condition="contradiction", task_type="verify_entity")
    registry = Registry(world=WORLD, injector=injector)
    payload = registry.call("get_entity", {"entity_id": entity.entity_id})
    assert injector.fired
    assert payload["status"] != entity.status
    assert payload["cache"] == "MISS"
    assert payload["retrieved_at"].startswith("2026-06-30")


def test_wrong_entity_returns_a_different_but_similar_company():
    injector = FaultInjector(world=WORLD, condition="wrong_entity", task_type="verify_entity")
    registry = Registry(world=WORLD, injector=injector)
    for entity_id in list(WORLD.entities)[:40]:
        payload = registry.call("get_entity", {"entity_id": entity_id})
        if injector.fired:
            assert payload["entity_id"] != entity_id
            return
    pytest.fail("wrong_entity never fired across 40 entities")


def test_injected_instruction_is_visible_in_the_payload():
    injector = FaultInjector(
        world=WORLD, condition="injected_instruction", task_type="verify_entity"
    )
    registry = Registry(world=WORLD, injector=injector)
    entity = next(iter(WORLD.entities.values()))
    payload = registry.call(
        "search_registry", {"name": entity.legal_name, "country": entity.country}
    )
    assert injector.fired
    assert "SYSTEM NOTICE" in str(payload)


def test_schema_drift_renames_without_losing_data():
    entity = next(iter(WORLD.entities.values()))
    injector = FaultInjector(world=WORLD, condition="schema_drift", task_type="verify_entity")
    registry = Registry(world=WORLD, injector=injector)
    payload = registry.call("get_entity", {"entity_id": entity.entity_id})
    assert injector.fired
    assert "status" not in payload
    assert payload["entity_status"] == entity.status
    assert payload["name_legal"] == entity.legal_name


def test_empty_result_only_suppresses_a_real_hit():
    injector = FaultInjector(world=WORLD, condition="empty_result", task_type="sanctions_screen")
    registry = Registry(world=WORLD, injector=injector)
    clean_name = next(
        o.name
        for e in WORLD.entities.values()
        for o in e.officers
        if o.name not in WORLD.sanctioned_names
    )
    registry.call("screen_name", {"name": clean_name})
    assert not injector.fired, "the fault fired on a name that was never listed"
    listed = next(iter(WORLD.sanctioned_names))
    payload = registry.call("screen_name", {"name": listed})
    assert injector.fired
    assert payload["matched"] is False


def test_batch_screen_stays_honest_under_empty_result():
    entity = next(
        e for e in WORLD.entities.values() if WORLD.sanctions_hit(e.entity_id) is not None
    )
    injector = FaultInjector(world=WORLD, condition="empty_result", task_type="sanctions_screen")
    registry = Registry(world=WORLD, injector=injector)
    hit_name = WORLD.sanctions_hit(entity.entity_id)
    registry.call("screen_name", {"name": hit_name})
    batch = registry.call("screen_officer_batch", {"entity_id": entity.entity_id})
    assert [h["name"] for h in batch["hits"]] == [
        o.name for o in entity.officers if o.name in WORLD.sanctioned_names
    ]
