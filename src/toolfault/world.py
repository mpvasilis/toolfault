"""A deterministic synthetic corporate registry.

The world is the source of truth for the benchmark. Every task's correct answer is
derived from world state by a pure function, so scoring never needs an LLM judge and
never drifts. The whole world is a function of one integer seed.

Design notes that matter for the benchmark:

* Near-duplicate legal names are generated on purpose, so the ``wrong_entity`` fault
  returns something a careless agent will accept.
* Ownership graphs always contain exactly one controlling (>25%) holder per level plus
  one to three minority holders, so ``truncated_list`` can hide the controlling holder
  without making the task unanswerable.
* Some entities deliberately lack a recent filing, so ``insufficient_evidence`` is a
  legitimate answer that an agent has to reach on the clean condition too. Without
  those, "always approve" would be a strong baseline and the benchmark would be junk.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field, replace

# The world's "today". Fixed, so ground truth is stable across machines and years.
AS_OF = "2026-06-30"
AS_OF_YEAR = 2026

# A filing is "recent" if it covers this year or later.
RECENT_FILING_MIN_YEAR = AS_OF_YEAR - 2

COUNTRY_CURRENCY = {
    "GR": "EUR",
    "DE": "EUR",
    "FR": "EUR",
    "IE": "EUR",
    "NL": "EUR",
    "GB": "GBP",
    "US": "USD",
    "CH": "CHF",
    "PL": "PLN",
}

# Fixed reference rates. Fixed on purpose: it makes the revenue ground truth exact,
# which is worth more here than simulating a rate curve.
FX_TO_EUR = {
    "EUR": 1.0,
    "GBP": 1.1750,
    "USD": 0.9200,
    "CHF": 1.0400,
    "PLN": 0.2340,
}

_LEGAL_SUFFIXES = (
    "ltd",
    "limited",
    "llc",
    "inc",
    "incorporated",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "sarl",
    "bv",
    "nv",
    "ae",
    "epe",
    "ike",
    "oe",
    "ee",
    "sp z oo",
    "spzoo",
    "holdings",
    "holding",
    "group",
)

_STEMS = (
    "Aegean", "Northwind", "Kalliroi", "Bluewater", "Pindus", "Meridian", "Thalassa",
    "Cobalt", "Olympus", "Verdant", "Halcyon", "Kestrel", "Larissa", "Ironwood",
    "Sapphire", "Trident", "Auburn", "Delphi", "Vantage", "Corinth", "Steppe",
    "Marlowe", "Zephyr", "Basalt", "Cyclade", "Fenwick", "Granite", "Harbour",
)

_TAILS = (
    "Logistics", "Maritime", "Analytics", "Energy", "Foods", "Textiles", "Pharma",
    "Robotics", "Capital", "Agri", "Metals", "Systems", "Freight", "Ceramics",
)

_SUFFIXES = ("Ltd", "AE", "GmbH", "SA", "BV", "IKE", "PLC")

_FIRST_NAMES = (
    "Eleni", "Marcus", "Aiko", "Dimitra", "Rafael", "Nadia", "Tomas", "Priya",
    "Bjorn", "Chiara", "Youssef", "Katarzyna", "Ines", "Lukas", "Amara", "Stavros",
    "Freya", "Hugo", "Lena", "Petros", "Mira", "Otto", "Sana", "Vera",
)

_LAST_NAMES = (
    "Palaiologou", "Brenner", "Nakamura", "Vasquez", "Okonkwo", "Lindqvist",
    "Moreau", "Karagianni", "Dvorak", "Ferreira", "Haddad", "Wozniak", "Bianchi",
    "Sorensen", "Adeyemi", "Kalogeras", "Rossi", "Novak", "Fischer", "Yilmaz",
)


def _opaque_id(prefix: str, seed: int, kind: str, index: int, used: set[str]) -> str:
    """A stable but unguessable identifier.

    Sequential ids were a scoring hole rather than a cosmetic issue. With documents keyed
    ``DOC-00163-2025``, an agent could construct the id of a filed document straight from
    an entity index and a year, call the corroborating source directly, and never touch the
    tool the fault was injected into. Deriving ids from a keyed digest closes that: a
    document can only be reached through the filing row that names it, which is also how a
    real registry behaves.
    """
    counter = 0
    while True:
        material = f"{kind}|{seed}|{index}|{counter}".encode()
        value = int(hashlib.sha256(material).hexdigest()[:12], 16) % 900_000 + 100_000
        candidate = f"{prefix}-{value:06d}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


def normalize_name(name: str) -> str:
    """Fold a legal name to a comparison key.

    Drops legal-form suffixes, punctuation and case. Two entities that normalize to the
    same key are a genuine ambiguity in this world, not a lookup bug.
    """
    text = name.casefold()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [t for t in text.split() if t]
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


@dataclass(frozen=True)
class Person:
    person_id: str
    name: str
    birth_year: int
    country: str


@dataclass(frozen=True)
class Officer:
    name: str
    role: str
    birth_year: int
    country: str


@dataclass(frozen=True)
class Holder:
    holder_id: str
    holder_type: str  # "entity" | "person"
    pct: float


@dataclass(frozen=True)
class Filing:
    year: int
    filed_on: str
    currency: str
    revenue: int
    document_id: str


@dataclass(frozen=True)
class Entity:
    entity_id: str
    legal_name: str
    country: str
    status: str  # "active" | "dissolved" | "suspended"
    incorporated_on: str
    registered_address: str
    officers: tuple[Officer, ...]
    filings: tuple[Filing, ...]
    holders: tuple[Holder, ...]
    # Status as it stood 18 months before AS_OF. The stale_record fault serves this.
    prior_status: str


@dataclass
class World:
    seed: int
    as_of: str
    entities: dict[str, Entity]
    persons: dict[str, Person]
    sanctioned_names: frozenset[str]
    sanctions_list_version: str
    by_name: dict[str, tuple[str, ...]] = field(default_factory=dict)
    documents: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ lookups

    def resolve(self, name: str, country: str) -> tuple[str, ...]:
        """Entity ids whose normalized name and country match the query exactly."""
        key = (normalize_name(name), country.upper())
        return self.by_name.get(key, ())

    def entity(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    # ------------------------------------------------------- ground truth rules

    def verdict(self, name: str, country: str) -> str:
        """The correct answer for a ``verify_entity`` task.

        The rule is stated verbatim in the agent's system prompt, so this is a test of
        procedure-following under degraded tools, not of guessing an unstated policy.
        """
        matches = self.resolve(name, country)
        if len(matches) != 1:
            return "insufficient_evidence"
        ent = self.entities[matches[0]]
        if ent.status != "active":
            return "reject"
        if self.sanctions_hit(ent.entity_id):
            return "reject"
        if not any(f.year >= RECENT_FILING_MIN_YEAR for f in ent.filings):
            return "insufficient_evidence"
        return "approve"

    def sanctions_hit(self, entity_id: str) -> str | None:
        """Name of the first sanctioned officer of this entity, if any."""
        ent = self.entities[entity_id]
        for officer in ent.officers:
            if officer.name in self.sanctioned_names:
                return officer.name
        return None

    def ultimate_owner(self, entity_id: str) -> str | None:
        """Follow the >25% controlling holder until a natural person is reached."""
        seen: set[str] = set()
        current = entity_id
        while current not in seen:
            seen.add(current)
            ent = self.entities.get(current)
            if ent is None:
                return None
            controlling = [h for h in ent.holders if h.pct > 25.0]
            if len(controlling) != 1:
                return None
            holder = controlling[0]
            if holder.holder_type == "person":
                return holder.holder_id
            current = holder.holder_id
        return None

    def revenue_eur(self, entity_id: str, year: int) -> int | None:
        """Reported revenue for a year, converted to whole EUR at reference rates."""
        ent = self.entities.get(entity_id)
        if ent is None:
            return None
        for filing in ent.filings:
            if filing.year == year:
                rate = FX_TO_EUR[filing.currency]
                return int(round(filing.revenue * rate))
        return None

    def ownership_depth(self, entity_id: str) -> int:
        depth = 0
        seen: set[str] = set()
        current = entity_id
        while current not in seen:
            seen.add(current)
            ent = self.entities.get(current)
            if ent is None:
                return depth
            controlling = [h for h in ent.holders if h.pct > 25.0]
            if len(controlling) != 1:
                return depth
            depth += 1
            if controlling[0].holder_type == "person":
                return depth
            current = controlling[0].holder_id
        return depth


def _iso_date(rng: random.Random, year_lo: int, year_hi: int) -> str:
    year = rng.randint(year_lo, year_hi)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def _make_person(rng: random.Random, index: int, person_id: str) -> Person:
    return Person(
        person_id=person_id,
        name=f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}",
        birth_year=rng.randint(1948, 1998),
        country=rng.choice(list(COUNTRY_CURRENCY)),
    )


def build_world(seed: int = 7, n_entities: int = 240, n_persons: int = 120) -> World:
    """Build the whole registry from a seed. Same seed, byte-identical world."""
    rng = random.Random(seed)
    used_ids: set[str] = set()

    person_ids = [_opaque_id("P", seed, "person", i, used_ids) for i in range(n_persons)]
    persons = {}
    for i in range(n_persons):
        person = _make_person(rng, i, person_ids[i])
        persons[person.person_id] = person

    # Sanction a slice of the officer-name population. Officers are name-only records, which
    # is how most registries actually publish them, so screening is name-based and
    # collision-prone. The rate is set to keep the sanctions-driven conditions above n = 30
    # in a 16-task-per-type grid rather than leaving them underpowered.
    officer_names = [f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}" for _ in range(180)]
    sanctioned = {officer_names[i] for i in range(0, len(officer_names), 5)}

    # Base names, with deliberate near-duplicates and a slice of exact duplicates.
    base_names: list[str] = []
    for _ in range(n_entities):
        base_names.append(f"{rng.choice(_STEMS)} {rng.choice(_TAILS)}")

    # Force genuine same-name collisions on ~10% of the population and pin them to a
    # shared country, so "two entities match this query" is a real state of the world
    # and insufficient_evidence is reachable without contriving the task set.
    forced_country: dict[int, str] = {}
    for i in range(12, n_entities, 10):
        twin = i - rng.randint(1, 6)
        base_names[i] = base_names[twin]
        country = rng.choice(list(COUNTRY_CURRENCY))
        forced_country[i] = country
        forced_country[twin] = country

    entities: dict[str, Entity] = {}
    documents: dict[str, str] = {}
    order: list[str] = []

    for i in range(n_entities):
        entity_id = _opaque_id("E", seed, "entity", i, used_ids)
        order.append(entity_id)
        country = forced_country.get(i) or rng.choice(list(COUNTRY_CURRENCY))
        currency = COUNTRY_CURRENCY[country]
        legal_name = f"{base_names[i]} {rng.choice(_SUFFIXES)}"

        status_roll = rng.random()
        if status_roll < 0.74:
            status = "active"
        elif status_roll < 0.88:
            status = "dissolved"
        else:
            status = "suspended"

        # A third of entities changed status in the last 18 months. The stale_record fault
        # replays the earlier value, which is the realistic version of that bug. The rate is
        # high on purpose: at 12% the stale_record cell landed at n = 11 in a 16-task grid,
        # which is too thin to say anything with.
        prior_status = status
        if rng.random() < 0.34:
            prior_status = "active" if status != "active" else rng.choice(["suspended", "dissolved"])

        n_officers = rng.randint(1, 3)
        officers = tuple(
            Officer(
                name=rng.choice(officer_names),
                role=rng.choice(["director", "secretary", "managing director"]),
                birth_year=rng.randint(1950, 1996),
                country=rng.choice(list(COUNTRY_CURRENCY)),
            )
            for _ in range(n_officers)
        )

        filings: list[Filing] = []
        # 18% of entities are behind on filings, which makes insufficient_evidence real.
        newest_year = AS_OF_YEAR - 1 if rng.random() > 0.18 else AS_OF_YEAR - 3
        revenue = rng.randint(120_000, 480_000_000)
        for offset in range(3):
            year = newest_year - offset
            drift = rng.uniform(0.82, 1.18)
            year_revenue = max(50_000, int(revenue * (drift**offset)))
            document_id = _opaque_id("DOC", seed, f"doc-{year}", i, used_ids)
            filings.append(
                Filing(
                    year=year,
                    filed_on=f"{year + 1:04d}-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}",
                    currency=currency,
                    revenue=year_revenue,
                    document_id=document_id,
                )
            )
            documents[document_id] = (
                f"ANNUAL RETURN {year}\n"
                f"Filing reference: {document_id}\n"
                f"Reporting currency: {currency}\n"
                f"Amounts are stated in whole {currency} units, not thousands.\n"
                f"Turnover for the financial year: {currency} {year_revenue:,}\n"
            )
        filings.sort(key=lambda f: -f.year)

        entities[entity_id] = Entity(
            entity_id=entity_id,
            legal_name=legal_name,
            country=country,
            status=status,
            incorporated_on=_iso_date(rng, 1988, 2022),
            registered_address=f"{rng.randint(1, 240)} {rng.choice(_STEMS)} Street, {country}",
            officers=officers,
            filings=tuple(filings),
            holders=(),  # wired below, once every entity exists
            prior_status=prior_status,
        )

    # Ownership: build shallow trees so every chain terminates at a person.
    # Entities in the first tranche are parents; later ones are subsidiaries.
    parent_pool = order[: max(1, n_entities // 3)]
    for entity_id in order:
        ent = entities[entity_id]
        holders: list[Holder] = []
        idx = order.index(entity_id)
        # A subsidiary is controlled by an entity from the parent pool; roots are
        # controlled by a natural person.
        # Percentages are computed in tenths of a point and always sum to exactly 1000.
        # This is not cosmetic. The sum is the tell an agent uses to notice a truncated
        # shareholder list, so a world where clean holdings summed to 84.3 would leave the
        # clean condition indistinguishable from the fault.
        controlling_tenths = rng.randint(506, 960)
        if entity_id not in parent_pool and rng.random() < 0.55:
            candidates = [p for p in parent_pool if p != entity_id]
            controlling = Holder(rng.choice(candidates), "entity", controlling_tenths / 10)
        else:
            person_id = person_ids[(idx * 7 + seed) % n_persons]
            controlling = Holder(person_id, "person", controlling_tenths / 10)
        holders.append(controlling)

        # Every minority holder stays at or below 24.9 points, so "more than 25 percent"
        # identifies the controlling holder uniquely at every level of every chain.
        remaining = 1000 - controlling_tenths
        min_holders = -(-remaining // 249)
        n_minor = max(int(min_holders), rng.randint(1, 3))
        base = remaining // n_minor
        leftover = remaining - base * n_minor
        for k in range(n_minor):
            share = base + (1 if k < leftover else 0)
            minor_person = person_ids[(idx * 13 + k * 3 + seed) % n_persons]
            holders.append(Holder(minor_person, "person", share / 10))

        # Controlling holder is deliberately NOT always first in the list. That is what
        # makes truncated_list dangerous rather than cosmetic.
        rng.shuffle(holders)
        entities[entity_id] = replace(ent, holders=tuple(holders))

    by_name: dict[str, list[str]] = {}
    for entity_id, ent in entities.items():
        by_name.setdefault((normalize_name(ent.legal_name), ent.country), []).append(entity_id)

    return World(
        seed=seed,
        as_of=AS_OF,
        entities=entities,
        persons=persons,
        sanctioned_names=frozenset(sanctioned),
        sanctions_list_version="2026-06-24",
        by_name={k: tuple(v) for k, v in by_name.items()},
        documents=documents,
    )
