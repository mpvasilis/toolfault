"""Statistics, in pure Python.

No numpy, no scipy. The whole module is a few hundred lines of arithmetic, it installs
anywhere, and every number in the published tables can be traced to a function here that
a reader can check by eye. For a benchmark that argues about rigour, that trade is worth
more than the speed.

What is used and why:

* **Wilson score interval** for every rate. Normal-approximation intervals are wrong at
  the small per-cell counts a fault-conditioned grid produces, and they happily report
  bounds outside [0, 1].
* **Paired sign-flip permutation test** for clean against a fault. The same task ids run
  under both conditions, so the pairing is real and the test needs no distributional
  assumption.
* **Benjamini-Hochberg** across the condition grid. Thirteen faults times several models
  is enough comparisons that uncorrected p-values would manufacture findings.
* **ECE, Brier, AUROC and the risk-coverage curve** for the confidence analysis, because
  "is the number calibrated" and "does the number rank correctly" are different
  questions and an agent can pass one while failing the other.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True)
class Rate:
    successes: int
    n: int
    lo: float
    hi: float

    @property
    def value(self) -> float:
        return self.successes / self.n if self.n else 0.0

    def as_pct(self) -> str:
        if not self.n:
            return "n/a"
        return f"{100 * self.value:.1f}%"

    def with_ci(self) -> str:
        if not self.n:
            return "n/a"
        return f"{100 * self.value:.1f}% [{100 * self.lo:.0f}-{100 * self.hi:.0f}]"


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> Rate:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return Rate(0, 0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Rounded before clamping: at p = 0 the exact lower bound is 0, and float error
    # otherwise leaves a bound of 5e-17 sitting below a point estimate of 0.
    lo = max(0.0, round(centre - half, 12))
    hi = min(1.0, round(centre + half, 12))
    return Rate(successes, n, lo, hi)


def rate(flags: list[bool]) -> Rate:
    return wilson(sum(1 for f in flags if f), len(flags))


def paired_permutation(
    treatment: list[float], control: list[float], iterations: int = 20000, seed: int = 20260801
) -> float:
    """Two-sided p-value for a paired difference under random sign flips.

    ``treatment[i]`` and ``control[i]`` must be the same task under two conditions.
    """
    if len(treatment) != len(control):
        raise ValueError("paired_permutation needs equal-length paired samples")
    diffs = [t - c for t, c in zip(treatment, control, strict=True)]
    non_zero = [d for d in diffs if d != 0.0]
    if not non_zero:
        return 1.0
    observed = abs(sum(diffs) / len(diffs))
    rng = random.Random(seed)
    hits = 0
    n = len(diffs)
    for _ in range(iterations):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= observed - 1e-12:
            hits += 1
    # Add-one correction: with a finite permutation count, never report p = 0.
    return (hits + 1) / (iterations + 1)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """BH step-up adjusted p-values (q-values), returned in the input order."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        candidate = pvalues[index] * m / rank
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


# --------------------------------------------------------------- calibration


def brier(confidences: list[float], correct: list[bool]) -> float:
    if not confidences:
        return float("nan")
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in zip(confidences, correct, strict=True)) / len(
        confidences
    )


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error with equal-mass bins that never split a tie.

    Equal-mass rather than equal-width: agents pile almost every prediction into the top
    of the range, and equal-width bins hide that behind empty buckets.

    Not splitting ties matters more than it sounds. Agents report the same confidence over
    and over, typically 0.9. If a run of identical values is cut across two bins, each
    fragment gets its own local accuracy and the error is computed against a value the
    model never distinguished, which inflates ECE. Twenty predictions at 0.9 that are right
    half the time are one bin with an error of 0.4, not ten bins averaging 0.5.
    """
    n = len(confidences)
    if n == 0:
        return float("nan")
    pairs = sorted(zip(confidences, correct, strict=True), key=lambda pair: pair[0])
    bins = max(1, min(bins, n))
    total = 0.0
    start = 0
    for b in range(bins):
        if start >= n:
            break
        end = round((b + 1) * n / bins)
        if end <= start:
            continue
        # Extend past the cut until the confidence value changes.
        while end < n and pairs[end][0] == pairs[end - 1][0]:
            end += 1
        chunk = pairs[start:end]
        mean_conf = sum(c for c, _ in chunk) / len(chunk)
        accuracy = sum(1 for _, ok in chunk if ok) / len(chunk)
        total += (len(chunk) / n) * abs(mean_conf - accuracy)
        start = end
    return total


def auroc(scores: list[float], labels: list[bool]) -> float:
    """AUROC via rank sum, with ties averaged. Returns nan if one class is absent."""
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    positive_rank_sum = sum(r for r, label in zip(ranks, labels, strict=True) if label)
    u = positive_rank_sum - positives * (positives + 1) / 2
    return u / (positives * negatives)


@dataclass(frozen=True)
class RiskCoverage:
    points: tuple[tuple[float, float], ...]  # (coverage, risk)
    aurc: float
    full_coverage_risk: float

    def risk_at(self, coverage: float) -> float:
        """Risk at the greatest achievable coverage at or below the target."""
        return next(
            (risk for cov, risk in reversed(self.points) if cov <= coverage + 1e-9),
            math.nan,
        )


def risk_coverage(confidences: list[float], correct: list[bool]) -> RiskCoverage:
    """Selective-prediction curve: answer the most confident fraction, escalate the rest.

    Each point is achievable with a confidence threshold. Equal scores enter together.
    AURC uses a right-endpoint step integral over the resulting coverage increments.
    """
    n = len(confidences)
    if n == 0:
        return RiskCoverage((), float("nan"), float("nan"))
    ordered = sorted(zip(confidences, correct, strict=True), key=lambda pair: -pair[0])
    points: list[tuple[float, float]] = []
    seen = 0
    errors = 0
    aurc = 0.0
    for _, group in groupby(ordered, key=lambda pair: pair[0]):
        batch = list(group)
        seen += len(batch)
        errors += sum(not ok for _, ok in batch)
        risk = errors / seen
        points.append((seen / n, risk))
        aurc += len(batch) / n * risk
    return RiskCoverage(tuple(points), aurc, points[-1][1])


def threshold_table(
    confidences: list[float], correct: list[bool], thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
) -> list[dict[str, float]]:
    """Answer when confidence >= tau, escalate otherwise. One row per threshold."""
    rows: list[dict[str, float]] = []
    n = len(confidences)
    for tau in thresholds:
        answered = [(c, ok) for c, ok in zip(confidences, correct, strict=True) if c >= tau]
        escalated = n - len(answered)
        errors = sum(1 for _, ok in answered if not ok)
        rows.append(
            {
                "threshold": tau,
                "coverage": len(answered) / n if n else 0.0,
                "escalation_rate": escalated / n if n else 0.0,
                "error_rate_on_answered": errors / len(answered) if answered else 0.0,
                "errors_reaching_production": errors / n if n else 0.0,
            }
        )
    return rows
