"""Hand-written baselines that test whether the benchmark can be gamed.

``always-escalate`` isolates the absence of verification and decision making by escalating
without reading evidence.

``modal-answer`` isolates the absence of tool use by submitting the most common answer for each
task type.

``one-call`` isolates superficial tool use by performing one registry lookup and ignoring its
result.

``confident-guess`` follows the naive happy path but isolates calibration by claiming near
certainty and suppressing every anomaly flag. It does the same amount of work as ``naive``, so it
is not expected to score badly on resolution; it exists to check that the calibration metrics can
tell a graded confidence signal from a maximally overconfident one.

``secondary-only`` is the important one. It reads only the corroborating sources and never the
primary tool, which was a scoring exploit in the first version of this benchmark: with every fault
landing on the primary tool and the corroborating sources always honest, this policy scored close
to the reference policy while doing no cross-checking. It is kept as a published baseline so the
fix stays fixed. If it ever climbs back toward the oracle, the side-choice logic in ``faults.py``
has regressed.
"""

from __future__ import annotations

from typing import Any

from ..world import AS_OF, RECENT_FILING_MIN_YEAR
from .base import Context, ToolCall
from .scripted import _HOLDER_LINE, NaiveAgent, Step, _PolicyAgent, undrift

MODAL_ANSWERS = {
    "verify_entity": "approve",
    "sanctions_screen": "clear",
    "ubo_chain": "P-00000",
    "revenue_eur": "0",
}


class AlwaysEscalateAgent(_PolicyAgent):
    """Escalate without reviewing evidence."""

    name = "always-escalate"

    def policy(self, ctx: Context) -> Step:
        yield self._tc(
            "escalate",
            {"reason": "no evidence reviewed", "data_quality_flags": []},
        )


class ModalAnswerAgent(_PolicyAgent):
    """Submit the modal answer for the task type without reading evidence."""

    name = "modal-answer"

    def policy(self, ctx: Context) -> Step:
        yield self._tc(
            "submit",
            {
                "answer": MODAL_ANSWERS[ctx.task.task_type],
                "confidence": 0.9,
                "evidence": [],
                "data_quality_flags": [],
            },
        )


class OneCallAgent(_PolicyAgent):
    """Perform one registry lookup, ignore it, then submit the modal answer."""

    name = "one-call"

    def policy(self, ctx: Context) -> Step:
        yield from self.call_once(
            "search_registry",
            {"name": ctx.task.meta["name"], "country": ctx.task.meta["country"]},
        )
        yield self._tc(
            "submit",
            {
                "answer": MODAL_ANSWERS[ctx.task.task_type],
                "confidence": 0.9,
                "evidence": [],
                "data_quality_flags": [],
            },
        )


class ConfidentGuessAgent(NaiveAgent):
    """Run the naive policy while reporting near certainty and no anomaly flags."""

    name = "confident-guess"

    def _tc(self, name: str, args: dict[str, Any]) -> ToolCall:
        if name == "submit":
            args = {**args, "confidence": 0.99, "data_quality_flags": []}
        return super()._tc(name, args)


class SecondarySourceOnlyAgent(_PolicyAgent):
    """Trust only the corroborating sources, never the primary tool.

    The policy an adversarial review found could beat the first version of this benchmark. It is
    single-source credulity with the sources swapped: it does no reconciliation, it just prefers
    the archive, the batch screen and the filed register over the live endpoints. Since silent
    faults now land on either side, it is exposed on roughly the same share of episodes as the
    ``naive`` policy that reads only primary sources, which is the property being protected.

    On the revenue task there is no secondary-only path at all, because a filed document can only
    be reached through the filing row that names it and the identifiers are digests. That task
    therefore falls back to reading the filing row, and this is stated rather than hidden.
    """

    name = "secondary-only"

    def policy(self, ctx: Context) -> Step:
        task = ctx.task
        search = yield from self.call_once(
            "search_registry",
            {"name": task.meta["name"], "country": task.meta["country"]},
        )
        results = search.get("results", []) if isinstance(search, dict) else []
        match_count = search.get("match_count", len(results)) if isinstance(search, dict) else 0

        if task.task_type == "verify_entity" and match_count != 1:
            yield self._submit("insufficient_evidence")
            return
        if not results:
            yield self._submit(MODAL_ANSWERS[task.task_type])
            return
        entity_id = results[0]["entity_id"]

        if task.task_type == "verify_entity":
            snapshot = yield from self.call_once(
                "get_entity_snapshot", {"entity_id": entity_id, "as_of": AS_OF}
            )
            status = snapshot.get("status") if isinstance(snapshot, dict) else None
            if status is not None and status != "active":
                yield self._submit("reject")
                return
            batch = yield from self.call_once(
                "screen_officer_batch", {"entity_id": entity_id}
            )
            if isinstance(batch, dict) and batch.get("hits"):
                yield self._submit("reject")
                return
            filings = undrift((yield from self.call_once("get_filings", {"entity_id": entity_id})))
            years = [
                int(f["year"]) for f in (filings or {}).get("filings", []) if "year" in f
            ] if isinstance(filings, dict) else []
            recent = any(year >= RECENT_FILING_MIN_YEAR for year in years)
            yield self._submit("approve" if recent else "insufficient_evidence")
            return

        if task.task_type == "sanctions_screen":
            batch = yield from self.call_once(
                "screen_officer_batch", {"entity_id": entity_id}
            )
            hits = batch.get("hits") if isinstance(batch, dict) else None
            yield self._submit("hit" if hits else "clear")
            return

        if task.task_type == "ubo_chain":
            current = entity_id
            for _ in range(6):
                doc = yield from self.call_once(
                    "get_ownership_document", {"entity_id": current}
                )
                text = doc.get("text", "") if isinstance(doc, dict) else ""
                holders = [
                    {"holder_id": m.group(1), "holder_type": m.group(2), "pct": float(m.group(3))}
                    for m in _HOLDER_LINE.finditer(text)
                ]
                controlling = [h for h in holders if h["pct"] > 25.0]
                if not controlling:
                    yield self._submit("P-00000")
                    return
                holder = controlling[0]
                if holder["holder_type"] == "person":
                    yield self._submit(holder["holder_id"])
                    return
                current = holder["holder_id"]
            yield self._submit("P-00000")
            return

        # revenue_eur: no corroborating-only path exists, so it reads the filing row.
        filings = undrift((yield from self.call_once("get_filings", {"entity_id": entity_id})))
        rows = (filings or {}).get("filings", []) if isinstance(filings, dict) else []
        row = next((f for f in rows if int(f.get("year", -1)) == int(task.meta["year"])), None)
        if row is None:
            yield self._submit("0")
            return
        amount = float(row.get("revenue", 0) or 0)
        if row.get("currency", "EUR") != "EUR":
            converted = yield from self.call_once(
                "fx_convert",
                {"amount": amount, "from_currency": row["currency"], "to_currency": "EUR"},
            )
            if isinstance(converted, dict) and "amount" in converted:
                amount = float(converted["amount"])
        yield self._submit(str(int(round(amount))))

    def _submit(self, answer: str) -> ToolCall:
        return self._tc(
            "submit",
            {
                "answer": answer,
                "confidence": 0.9,
                "evidence": ["corroborating sources only"],
                "data_quality_flags": [],
            },
        )
