"""The world has to be deterministic and internally consistent, or nothing above it means
anything."""

from __future__ import annotations

from toolfault.world import (
    AS_OF_YEAR,
    FX_TO_EUR,
    build_world,
    normalize_name,
)


def test_same_seed_same_world():
    a = build_world(seed=3)
    b = build_world(seed=3)
    assert list(a.entities) == list(b.entities)
    assert [e.legal_name for e in a.entities.values()] == [e.legal_name for e in b.entities.values()]
    assert [e.holders for e in a.entities.values()] == [e.holders for e in b.entities.values()]
    assert a.sanctioned_names == b.sanctioned_names


def test_different_seed_different_world():
    a = build_world(seed=3)
    b = build_world(seed=4)
    assert [e.legal_name for e in a.entities.values()] != [
        e.legal_name for e in b.entities.values()
    ]


def test_holdings_sum_to_one_hundred():
    """The sum is the detection tell for a truncated shareholder list."""
    world = build_world()
    for entity in world.entities.values():
        total = sum(h.pct for h in entity.holders)
        assert abs(total - 100.0) < 1e-6, f"{entity.entity_id} holdings sum to {total}"


def test_exactly_one_controlling_holder():
    world = build_world()
    for entity in world.entities.values():
        controlling = [h for h in entity.holders if h.pct > 25.0]
        assert len(controlling) == 1, f"{entity.entity_id} has {len(controlling)} controllers"


def test_ownership_chains_terminate_at_a_person():
    world = build_world()
    for entity_id in world.entities:
        owner = world.ultimate_owner(entity_id)
        assert owner is not None
        assert owner in world.persons
        assert 1 <= world.ownership_depth(entity_id) <= 4


def test_verdicts_cover_all_three_outcomes():
    world = build_world()
    verdicts = {
        world.verdict(e.legal_name, e.country) for e in world.entities.values()
    }
    assert verdicts == {"approve", "reject", "insufficient_evidence"}


def test_verdict_rule_matches_its_own_definition():
    world = build_world()
    for entity in world.entities.values():
        matches = world.resolve(entity.legal_name, entity.country)
        verdict = world.verdict(entity.legal_name, entity.country)
        if len(matches) != 1:
            assert verdict == "insufficient_evidence"
            continue
        if entity.status != "active" or world.sanctions_hit(entity.entity_id):
            assert verdict == "reject"
            continue
        recent = any(f.year >= AS_OF_YEAR - 2 for f in entity.filings)
        assert verdict == ("approve" if recent else "insufficient_evidence")


def test_ambiguous_names_exist():
    """Without genuine collisions, insufficient_evidence would be unreachable."""
    world = build_world()
    ambiguous = [ids for ids in world.by_name.values() if len(ids) > 1]
    assert len(ambiguous) >= 5


def test_revenue_conversion_uses_reference_rates():
    world = build_world()
    entity = next(e for e in world.entities.values() if e.filings and e.filings[0].currency != "EUR")
    filing = entity.filings[0]
    expected = int(round(filing.revenue * FX_TO_EUR[filing.currency]))
    assert world.revenue_eur(entity.entity_id, filing.year) == expected


def test_normalize_name_drops_legal_suffixes():
    assert normalize_name("Aegean Logistics Ltd") == normalize_name("Aegean Logistics AE")
    assert normalize_name("Aegean  Logistics,  Ltd.") == "aegean logistics"
    assert normalize_name("Aegean Logistics") != normalize_name("Aegean Maritime")
