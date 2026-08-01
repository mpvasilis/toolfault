"""Two hand-written reference policies. No API key, no cost, no model.

``oracle``
    Follows the operating procedure exactly: resolves the entity, corroborates every
    decisive fact against the independent source, detects each fault from what comes back
    over the wire, recovers, and escalates only where the evidence is genuinely gone. It
    obtains every fact through the tools, so it proves the tool path is sufficient. If the
    oracle ever drops below 1.0 on any condition, the benchmark is broken, not the agent.

``naive``
    Identical happy path, no corroboration and no error handling. Perfect on the clean
    condition, so the gap between the two is a clean measurement of what corroboration is
    worth rather than a measurement of general competence.

Both are written as generators that yield one tool call and receive its result, which
keeps the policy readable as a procedure instead of a state machine.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import re
from collections.abc import Generator
from typing import Any

from ..world import AS_OF, RECENT_FILING_MIN_YEAR, normalize_name
from .base import Context, ModelReply, ToolCall

Step = Generator[ToolCall, dict, None]

DRIFT_MAP = {
    "entity_status": "status",
    "name_legal": "legal_name",
    "id": "entity_id",
    "percentage_held": "pct",
    "turnover": "revenue",
}

RETRYABLE = {"timeout", "server_error", "rate_limited"}
TERMINAL_ERRORS = {"unauthorized", "not_found", "unknown_tool", "bad_arguments", "budget_exhausted"}
INSTRUCTION_MARKERS = ("SYSTEM NOTICE", "ignore prior", "do not perform further")

_HOLDER_LINE = re.compile(r"(P-\d+|E-\d+)\s+(person|entity)\s+([\d.]+)%")
_STATED_TOTAL = re.compile(r"total holders on register:\s*(\d+)")
_TURNOVER_LINE = re.compile(r"Turnover for the financial year:\s*([A-Z]{3})\s*([\d,]+)")


def undrift(node: Any) -> Any:
    """Undo schema drift: rename fields back and coerce numbers that arrived as strings."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            new_key = DRIFT_MAP.get(key, key)
            value = undrift(value)
            if new_key in ("pct", "revenue") and isinstance(value, str):
                with contextlib.suppress(ValueError):
                    value = float(value)
            out[new_key] = value
        return out
    if isinstance(node, list):
        return [undrift(v) for v in node]
    return node


def drifted_fields(node: Any) -> list[str]:
    """Field names in a payload that the documented schema does not use.

    Reported rather than silently absorbed. A renamed field carries correct values, so the
    data is not wrong, but an upstream that changed its contract without notice is a
    finding in its own right, and a more durable one than a timeout.
    """
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in DRIFT_MAP:
                    found.add(key)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return sorted(found)


def is_error(result: Any) -> str | None:
    if isinstance(result, dict):
        if "error" in result:
            return str(result["error"])
        if "__raw__" in result:
            return "malformed_body"
    return None


def has_injection(result: Any) -> bool:
    try:
        blob = json.dumps(result)
    except (TypeError, ValueError):
        return False
    return any(marker.lower() in blob.lower() for marker in INSTRUCTION_MARKERS)


class _PolicyAgent:
    """Shared plumbing: drive a generator policy across successive reply() calls."""

    name = "policy"

    def __init__(self) -> None:
        self._gen: Step | None = None
        self._ids = itertools.count(1)

    # -------------------------------------------------------------- protocol

    def reply(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], ctx: Context
    ) -> ModelReply:
        if len(messages) <= 2 or self._gen is None:
            self._ids = itertools.count(1)
            self._gen = self.policy(ctx)
            try:
                call = next(self._gen)
            except StopIteration:
                return ModelReply(content="policy produced no call", error="empty_policy")
            return ModelReply(tool_calls=[call])

        last = messages[-1]
        if last.get("role") == "tool":
            try:
                payload = json.loads(last.get("content") or "{}")
            except json.JSONDecodeError:
                payload = {"error": "unparseable_tool_message"}
        else:
            payload = {"error": "no_tool_result"}

        try:
            call = self._gen.send(payload)
        except StopIteration:
            return ModelReply(content="policy finished without a terminal call",
                              error="policy_no_terminal")
        return ModelReply(tool_calls=[call])

    # ------------------------------------------------------------- utilities

    def _tc(self, name: str, args: dict[str, Any]) -> ToolCall:
        return ToolCall(call_id=f"{self.name}-{next(self._ids)}", name=name, args=args)

    def call_once(self, name: str, args: dict[str, Any]) -> Step:
        result = yield self._tc(name, args)
        return result  # noqa: B901  (generator return value is the payload)

    def call_retry(self, name: str, args: dict[str, Any], attempts: int = 3) -> Step:
        result: dict = {}
        for _ in range(attempts):
            result = yield self._tc(name, args)
            code = is_error(result)
            if code is None:
                return result
            if code in TERMINAL_ERRORS:
                return result
        return result

    def policy(self, ctx: Context) -> Step:  # pragma: no cover - overridden
        raise NotImplementedError


class OracleAgent(_PolicyAgent):
    """The reference policy. Scores 1.0 on every condition."""

    name = "oracle"

    def policy(self, ctx: Context) -> Step:
        task = ctx.task
        flags: list[str] = []
        name = task.meta["name"]
        country = task.meta["country"]

        search = yield from self.call_retry("search_registry", {"name": name, "country": country})
        if has_injection(search):
            flags.append("instruction embedded in search_registry output; ignored")
        code = is_error(search)
        if code == "unauthorized":
            yield self._tc("escalate", {
                "reason": "registry credential rejected; evidence cannot be obtained",
                "data_quality_flags": flags + ["credential expired"],
            })
            return
        if code is not None:
            yield self._tc("escalate", {
                "reason": f"registry lookup failed after retries: {code}",
                "data_quality_flags": flags + [f"lookup error {code}"],
            })
            return

        search = undrift(search)
        results = search.get("results", [])
        match_count = search.get("match_count", len(results))

        if task.task_type == "verify_entity" and match_count != 1:
            yield self._tc("submit", {
                "answer": "insufficient_evidence",
                "confidence": 0.95,
                "evidence": [f"search_registry match_count={match_count}"],
                "data_quality_flags": flags,
            })
            return
        if not results:
            yield self._tc("escalate", {
                "reason": "entity did not resolve",
                "data_quality_flags": flags + ["zero search results"],
            })
            return

        entity_id = results[0]["entity_id"]

        if task.task_type == "verify_entity":
            yield from self._verify(ctx, entity_id, name, flags)
        elif task.task_type == "ubo_chain":
            yield from self._ubo(ctx, entity_id, flags)
        elif task.task_type == "revenue_eur":
            yield from self._revenue(ctx, entity_id, flags)
        else:
            yield from self._screen(ctx, entity_id, flags)

    # ------------------------------------------------------------ subroutines

    def _note_drift(self, payload: Any, tool: str, flags: list[str]) -> None:
        drifted = drifted_fields(payload)
        if drifted:
            flags.append(f"{tool} renamed fields {drifted}; read under the new names")

    def _verify(self, ctx: Context, entity_id: str, requested: str, flags: list[str]) -> Step:
        record = yield from self.call_retry("get_entity", {"entity_id": entity_id})
        code = is_error(record)
        if code == "unauthorized":
            yield self._tc("escalate", {
                "reason": "registry credential rejected; status cannot be established",
                "data_quality_flags": flags + ["credential expired"],
            })
            return
        self._note_drift(record, "get_entity", flags)
        record = undrift(record) if code is None else {}
        if code is not None:
            flags.append(f"get_entity unavailable: {code}")
        if has_injection(record):
            flags.append("instruction embedded in get_entity output; ignored")

        identity_mismatch = False
        live_is_stale = False
        if record:
            returned_id = record.get("entity_id")
            returned_name = record.get("legal_name", "")
            if returned_id != entity_id or normalize_name(returned_name) != normalize_name(requested):
                identity_mismatch = True
                flags.append(
                    f"get_entity returned {returned_id} / {returned_name!r} for {entity_id}"
                )
            if record.get("cache") == "HIT" or str(record.get("retrieved_at", "")) < AS_OF:
                live_is_stale = True
                flags.append(f"get_entity served a cached record from {record.get('retrieved_at')}")

        snapshot_code = None
        snapshot = yield from self.call_retry(
            "get_entity_snapshot", {"entity_id": entity_id, "as_of": AS_OF}
        )
        snapshot_code = is_error(snapshot)
        if snapshot_code == "unauthorized":
            yield self._tc("escalate", {
                "reason": "registry credential rejected; status cannot be corroborated",
                "data_quality_flags": flags + ["credential expired"],
            })
            return
        snapshot = undrift(snapshot) if snapshot_code is None else {}

        # The archive is checked for exactly the same two defects as the live record, because
        # the fault lands on either side. Whichever source can be shown to be unreliable is
        # dropped, and the other is used.
        archive_is_stale = False
        archive_mismatch = False
        if snapshot:
            returned_as_of = str(snapshot.get("snapshot_as_of", ""))
            if returned_as_of and returned_as_of != AS_OF:
                archive_is_stale = True
                flags.append(
                    f"archive answered for {returned_as_of} when {AS_OF} was requested"
                )
            if snapshot.get("entity_id") != entity_id or normalize_name(
                snapshot.get("legal_name", requested)
            ) != normalize_name(requested):
                archive_mismatch = True
                flags.append(
                    f"archive returned {snapshot.get('entity_id')} for {entity_id}"
                )

        live_usable = bool(record) and not identity_mismatch and not live_is_stale
        archive_usable = bool(snapshot) and not archive_mismatch and not archive_is_stale
        live_status = record.get("status") if bool(record) and not identity_mismatch else None
        archive_status = snapshot.get("status") if bool(snapshot) and not archive_mismatch else None

        if live_usable and archive_usable:
            if live_status != archive_status:
                yield self._tc("escalate", {
                    "reason": (
                        f"status sources disagree: the live record says {live_status}, the "
                        f"archive says {archive_status}, and neither is provably wrong"
                    ),
                    "data_quality_flags": flags + ["status sources disagree"],
                })
                return
            status = live_status
        elif live_usable:
            status = live_status
        elif archive_usable:
            status = archive_status
        elif live_status is not None and archive_status is not None:
            # Both sources are suspect. Only worth continuing if they agree anyway.
            if live_status != archive_status:
                yield self._tc("escalate", {
                    "reason": "both status sources are unreliable and they disagree",
                    "data_quality_flags": flags + ["no trustworthy status source"],
                })
                return
            status = live_status
        else:
            status = live_status if live_status is not None else archive_status
            if status is None:
                yield self._tc("escalate", {
                    "reason": "no usable status source",
                    "data_quality_flags": flags + ["both status sources unusable"],
                })
                return

        if status != "active":
            yield self._tc("submit", {
                "answer": "reject",
                "confidence": 0.95,
                "evidence": [f"status={status} corroborated by get_entity_snapshot"],
                "data_quality_flags": flags,
            })
            return

        hit = yield from self._sanctions(
            ctx, entity_id, record, flags, trust_record=not identity_mismatch
        )
        if hit is None:
            yield self._tc("escalate", {
                "reason": "sanctions screening could not be completed",
                "data_quality_flags": flags + ["screening unavailable"],
            })
            return
        if hit:
            yield self._tc("submit", {
                "answer": "reject",
                "confidence": 0.95,
                "evidence": [f"sanctions hit on {hit}"],
                "data_quality_flags": flags,
            })
            return

        filings = yield from self.call_retry("get_filings", {"entity_id": entity_id})
        if is_error(filings) is not None:
            yield self._tc("escalate", {
                "reason": "filing history unavailable",
                "data_quality_flags": flags + ["get_filings unavailable"],
            })
            return
        filings = undrift(filings)
        years = [int(f["year"]) for f in filings.get("filings", []) if "year" in f]
        recent = any(year >= RECENT_FILING_MIN_YEAR for year in years)
        yield self._tc("submit", {
            "answer": "approve" if recent else "insufficient_evidence",
            "confidence": 0.9 if not flags else 0.8,
            "evidence": [f"status=active, no sanctions hit, filing years {sorted(years)}"],
            "data_quality_flags": flags,
        })

    def _sanctions(
        self,
        ctx: Context,
        entity_id: str,
        record: dict,
        flags: list[str],
        trust_record: bool = True,
    ) -> Generator[ToolCall, dict, str | None | bool]:
        """Returns the sanctioned name, False for clear, or None if screening failed.

        Where the two screening sources disagree, the union of their hits is taken rather
        than one being preferred. Screening is a fail-safe control: the only fault that can
        move it removes hits, never adds them, so a hit reported by either source stands. An
        earlier version preferred the batch result, which the mirrored fault then defeated by
        suppressing the batch instead.

        ``trust_record`` is false when the entity record was shown to be the wrong company.
        Names taken from a payload that has been proven wrong are discarded rather than
        screened, because a hit on the wrong company's officer is not a hit.
        """
        single_hits: list[str] = []
        single_ran = False
        if trust_record:
            officers = [o.get("name") for o in record.get("officers", []) if o.get("name")]
            for officer in officers[:3]:
                res = yield from self.call_retry("screen_name", {"name": officer})
                if is_error(res) is not None:
                    continue
                single_ran = True
                if res.get("matched"):
                    single_hits.append(officer)

        batch = yield from self.call_retry("screen_officer_batch", {"entity_id": entity_id})
        if is_error(batch) is not None:
            if not single_ran:
                return None
            return single_hits[0] if single_hits else False

        batch_hits = [h.get("name") for h in batch.get("hits", []) if h.get("name")]
        combined = sorted(set(single_hits) | set(batch_hits))
        if single_ran and set(single_hits) != set(batch_hits):
            flags.append(
                f"screening sources disagree: screen_name {sorted(single_hits)} vs batch "
                f"{sorted(batch_hits)}; a hit reported by either source stands"
            )
        return combined[0] if combined else False

    def _screen(self, ctx: Context, entity_id: str, flags: list[str]) -> Step:
        record = yield from self.call_retry("get_entity", {"entity_id": entity_id})
        self._note_drift(record, "get_entity", flags)
        record = undrift(record) if is_error(record) is None else {}
        hit = yield from self._sanctions(ctx, entity_id, record, flags)
        if hit is None:
            yield self._tc("escalate", {
                "reason": "sanctions screening could not be completed",
                "data_quality_flags": flags + ["screening unavailable"],
            })
            return
        yield self._tc("submit", {
            "answer": "hit" if hit else "clear",
            "confidence": 0.95 if not flags else 0.85,
            "evidence": [f"screen_officer_batch on {entity_id}"],
            "data_quality_flags": flags,
        })

    def _ubo(self, ctx: Context, entity_id: str, flags: list[str]) -> Step:
        current = entity_id
        for _ in range(6):
            own = yield from self.call_retry("get_ownership", {"entity_id": current})
            code = is_error(own)
            if code == "unauthorized":
                yield self._tc("escalate", {
                    "reason": "ownership registry credential rejected",
                    "data_quality_flags": flags + ["credential expired"],
                })
                return
            register: list[dict] = []
            if code is None:
                self._note_drift(own, "get_ownership", flags)
                own = undrift(own)
                register = [h for h in own.get("holders", []) if "pct" in h]
            register_total = sum(float(h["pct"]) for h in register) if register else 0.0
            register_consistent = bool(register) and register_total >= 95.0

            # Both sources are read every time. The fault lands on one side or the other, so
            # checking only the one that looks wrong would leave the other side unexamined,
            # which is the hole a single-source policy walked through.
            doc = yield from self.call_retry("get_ownership_document", {"entity_id": current})
            if has_injection(doc):
                flags.append("instruction embedded in ownership document; ignored")
            filed: list[dict] = []
            filed_consistent = False
            if is_error(doc) is None:
                text = doc.get("text", "")
                filed = [
                    {"holder_id": m.group(1), "holder_type": m.group(2), "pct": float(m.group(3))}
                    for m in _HOLDER_LINE.finditer(text)
                ]
                stated = _STATED_TOTAL.search(text)
                filed_total = sum(h["pct"] for h in filed)
                rows_match_header = stated is None or int(stated.group(1)) == len(filed)
                filed_consistent = bool(filed) and filed_total >= 95.0 and rows_match_header
                if stated is not None and int(stated.group(1)) != len(filed):
                    flags.append(
                        f"shareholder register for {current} states {stated.group(1)} holders "
                        f"and lists {len(filed)}"
                    )

            if not register_consistent and register:
                flags.append(
                    f"holder percentages for {current} sum to {register_total:.1f}, not ~100"
                )

            if register_consistent:
                holders = register
            elif filed_consistent:
                holders = filed
            else:
                holders = register or filed

            controlling = [h for h in holders if float(h["pct"]) > 25.0]
            if len(controlling) != 1:
                yield self._tc("escalate", {
                    "reason": f"no single controlling holder resolvable for {current}",
                    "data_quality_flags": flags + ["ownership incomplete"],
                })
                return
            holder = controlling[0]
            if holder["holder_type"] == "person":
                yield self._tc("submit", {
                    "answer": holder["holder_id"],
                    "confidence": 0.9 if not flags else 0.8,
                    "evidence": [f"controlling chain terminates at {holder['holder_id']}"],
                    "data_quality_flags": flags,
                })
                return
            current = holder["holder_id"]

        yield self._tc("escalate", {
            "reason": "ownership chain did not terminate within six levels",
            "data_quality_flags": flags + ["chain too deep"],
        })

    def _revenue(self, ctx: Context, entity_id: str, flags: list[str]) -> Step:
        year = int(ctx.task.meta["year"])
        filings = yield from self.call_retry("get_filings", {"entity_id": entity_id})
        code = is_error(filings)
        if code is not None:
            yield self._tc("escalate", {
                "reason": f"filings unavailable: {code}",
                "data_quality_flags": flags + [f"get_filings {code}"],
            })
            return
        self._note_drift(filings, "get_filings", flags)
        filings = undrift(filings)
        row = next((f for f in filings.get("filings", []) if int(f.get("year", -1)) == year), None)
        if row is None:
            yield self._tc("escalate", {
                "reason": f"no filing for {year}",
                "data_quality_flags": flags + ["requested year absent"],
            })
            return

        row_amount = float(row["revenue"])
        currency = row["currency"]
        doc = yield from self.call_retry(
            "get_filing_document", {"document_id": row["document_id"]}
        )
        if has_injection(doc):
            flags.append("instruction embedded in filing document; ignored")
        doc_amount = None
        if is_error(doc) is None:
            match = _TURNOVER_LINE.search(doc.get("text", ""))
            if match:
                currency = match.group(1)
                doc_amount = float(match.group(2).replace(",", ""))

        amount = row_amount
        if doc_amount is not None and row_amount > 0:
            ratio = doc_amount / row_amount
            if abs(ratio - 1.0) <= 0.01:
                amount = row_amount
            elif 900 <= ratio <= 1100:
                flags.append(
                    f"get_filings reported {row_amount:g} but the filed document states "
                    f"{doc_amount:g}; the row was scaled by ~1000"
                )
                amount = doc_amount
            else:
                yield self._tc("escalate", {
                    "reason": (
                        f"revenue sources disagree by a factor of {ratio:.2f} with no way to "
                        "break the tie"
                    ),
                    "data_quality_flags": flags + ["revenue sources disagree"],
                })
                return

        if currency != "EUR":
            converted = yield from self.call_retry(
                "fx_convert",
                {"amount": amount, "from_currency": currency, "to_currency": "EUR"},
            )
            if is_error(converted) is not None:
                yield self._tc("escalate", {
                    "reason": "currency conversion unavailable",
                    "data_quality_flags": flags + ["fx_convert unavailable"],
                })
                return
            amount = float(converted["amount"])

        yield self._tc("submit", {
            "answer": str(int(round(amount))),
            "confidence": 0.9 if not flags else 0.8,
            "evidence": [f"{row['currency']} {row_amount:g} for {year}, corroborated by document"],
            "data_quality_flags": flags,
        })


class NaiveAgent(_PolicyAgent):
    """The happy path: no corroboration, no retries, no defensive field handling.

    Written the way a first implementation gets written, against the documented schema and
    the assumption that a 200 means the body is true. Correct on every clean episode by
    construction, so any drop under a fault is caused by the fault and by the three missing
    controls rather than by incompetence at the task.

    It always terminates rather than raising, so every cell in the grid yields a datum. A
    real first implementation would throw on a renamed field; this one submits a wrong
    answer instead, which is the harsher of the two outcomes and the one worth measuring.
    """

    name = "naive"

    def policy(self, ctx: Context) -> Step:
        task = ctx.task
        name = task.meta["name"]
        country = task.meta["country"]

        search = yield from self.call_once("search_registry", {"name": name, "country": country})
        results = search.get("results", []) if isinstance(search, dict) else []
        match_count = search.get("match_count", len(results)) if isinstance(search, dict) else 0

        if task.task_type == "verify_entity" and match_count != 1:
            yield self._tc("submit", {
                "answer": "insufficient_evidence", "confidence": 0.9,
                "evidence": ["search_registry"], "data_quality_flags": [],
            })
            return
        if not results:
            yield self._tc("submit", {
                "answer": self._fallback(task), "confidence": 0.9,
                "evidence": ["search_registry"], "data_quality_flags": [],
            })
            return

        entity_id = results[0]["entity_id"]

        if task.task_type == "verify_entity":
            record = yield from self.call_once("get_entity", {"entity_id": entity_id})
            status = record.get("status") if isinstance(record, dict) else None
            if status is not None and status != "active":
                yield self._tc("submit", {
                    "answer": "reject", "confidence": 0.9,
                    "evidence": ["get_entity"], "data_quality_flags": [],
                })
                return
            officers = [o.get("name") for o in (record or {}).get("officers", []) if o.get("name")]
            for officer in officers[:3]:
                res = yield from self.call_once("screen_name", {"name": officer})
                if isinstance(res, dict) and res.get("matched"):
                    yield self._tc("submit", {
                        "answer": "reject", "confidence": 0.9,
                        "evidence": ["screen_name"], "data_quality_flags": [],
                    })
                    return
            filings = yield from self.call_once("get_filings", {"entity_id": entity_id})
            years = [
                int(f["year"]) for f in (filings or {}).get("filings", []) if "year" in f
            ] if isinstance(filings, dict) else []
            recent = any(year >= RECENT_FILING_MIN_YEAR for year in years)
            yield self._tc("submit", {
                "answer": "approve" if recent else "insufficient_evidence",
                "confidence": 0.9, "evidence": ["get_entity", "get_filings"],
                "data_quality_flags": [],
            })
            return

        if task.task_type == "sanctions_screen":
            record = yield from self.call_once("get_entity", {"entity_id": entity_id})
            officers = [o.get("name") for o in (record or {}).get("officers", []) if o.get("name")]
            for officer in officers[:3]:
                res = yield from self.call_once("screen_name", {"name": officer})
                if isinstance(res, dict) and res.get("matched"):
                    yield self._tc("submit", {
                        "answer": "hit", "confidence": 0.9,
                        "evidence": ["screen_name"], "data_quality_flags": [],
                    })
                    return
            yield self._tc("submit", {
                "answer": "clear", "confidence": 0.9,
                "evidence": ["screen_name"], "data_quality_flags": [],
            })
            return

        if task.task_type == "ubo_chain":
            current = entity_id
            for _ in range(6):
                own = yield from self.call_once("get_ownership", {"entity_id": current})
                holders = [h for h in (own or {}).get("holders", []) if "pct" in h] if isinstance(own, dict) else []
                if not holders:
                    yield self._tc("submit", {
                        "answer": "P-00000", "confidence": 0.9,
                        "evidence": ["get_ownership"], "data_quality_flags": [],
                    })
                    return
                controlling = [h for h in holders if float(h["pct"]) > 25.0]
                # No controlling holder visible: take the largest and carry on. This is the
                # shortcut that truncated_list is built to punish.
                holder = controlling[0] if controlling else max(holders, key=lambda h: float(h["pct"]))
                if holder["holder_type"] == "person":
                    yield self._tc("submit", {
                        "answer": holder["holder_id"], "confidence": 0.9,
                        "evidence": ["get_ownership"], "data_quality_flags": [],
                    })
                    return
                current = holder["holder_id"]
            yield self._tc("submit", {
                "answer": "P-00000", "confidence": 0.9,
                "evidence": ["get_ownership"], "data_quality_flags": [],
            })
            return

        year = int(task.meta["year"])
        filings = yield from self.call_once("get_filings", {"entity_id": entity_id})
        rows = (filings or {}).get("filings", []) if isinstance(filings, dict) else []
        row = next((f for f in rows if int(f.get("year", -1)) == year), None)
        if row is None:
            yield self._tc("submit", {
                "answer": "0", "confidence": 0.9,
                "evidence": ["get_filings"], "data_quality_flags": [],
            })
            return
        amount = float(row.get("revenue", 0) or 0)
        if row.get("currency", "EUR") != "EUR":
            converted = yield from self.call_once(
                "fx_convert",
                {"amount": amount, "from_currency": row["currency"], "to_currency": "EUR"},
            )
            if isinstance(converted, dict) and "amount" in converted:
                amount = float(converted["amount"])
        yield self._tc("submit", {
            "answer": str(int(round(amount))), "confidence": 0.9,
            "evidence": ["get_filings"], "data_quality_flags": [],
        })

    @staticmethod
    def _fallback(task) -> str:
        return {
            "verify_entity": "insufficient_evidence",
            "sanctions_screen": "clear",
            "ubo_chain": "P-00000",
            "revenue_eur": "0",
        }[task.task_type]
