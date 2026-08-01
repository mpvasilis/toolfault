"""Statistics tests, checked against values computed by hand or by definition."""

from __future__ import annotations

import math

from toolfault.stats import (
    auroc,
    benjamini_hochberg,
    brier,
    ece,
    paired_permutation,
    risk_coverage,
    threshold_table,
    wilson,
)


def test_wilson_matches_published_value():
    # 8 of 10, Wilson 95%: roughly [0.49, 0.94]
    interval = wilson(8, 10)
    assert abs(interval.value - 0.8) < 1e-9
    assert abs(interval.lo - 0.4901) < 0.002
    assert abs(interval.hi - 0.9433) < 0.002


def test_wilson_stays_inside_the_unit_interval():
    for successes, n in ((0, 3), (3, 3), (0, 1), (1, 1), (1, 200)):
        interval = wilson(successes, n)
        assert 0.0 <= interval.lo <= interval.value <= interval.hi <= 1.0


def test_wilson_handles_empty():
    interval = wilson(0, 0)
    assert interval.n == 0
    assert interval.with_ci() == "n/a"


def test_permutation_finds_a_real_effect():
    treatment = [0.0] * 20
    control = [1.0] * 20
    assert paired_permutation(treatment, control, iterations=4000) < 0.01


def test_permutation_finds_nothing_in_noise():
    treatment = [1.0, 0.0] * 10
    control = [1.0, 0.0] * 10
    assert paired_permutation(treatment, control, iterations=4000) == 1.0


def test_permutation_never_returns_zero():
    treatment = [0.0] * 40
    control = [1.0] * 40
    assert paired_permutation(treatment, control, iterations=1000) > 0.0


def test_bh_is_monotone_and_bounded():
    raw = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    adjusted = benjamini_hochberg(raw)
    assert all(0.0 <= q <= 1.0 for q in adjusted)
    assert all(q >= p for q, p in zip(adjusted, raw, strict=True))
    assert adjusted == sorted(adjusted)


def test_bh_preserves_input_order():
    adjusted = benjamini_hochberg([0.5, 0.01, 0.2])
    assert adjusted[1] < adjusted[2] < adjusted[0]


def test_auroc_perfect_and_random():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5


def test_auroc_needs_both_classes():
    assert math.isnan(auroc([0.9, 0.8], [True, True]))


def test_brier_and_ece_at_the_extremes():
    assert brier([1.0, 1.0], [True, True]) == 0.0
    assert brier([0.0, 0.0], [True, True]) == 1.0
    assert abs(ece([1.0] * 10, [True] * 10)) < 1e-9
    assert abs(ece([1.0] * 10, [False] * 10) - 1.0) < 1e-9


def test_ece_detects_overconfidence():
    confidences = [0.9] * 10
    correct = [True] * 5 + [False] * 5
    assert abs(ece(confidences, correct) - 0.4) < 1e-9


def test_risk_coverage_falls_when_confidence_is_informative():
    confidences = [0.95, 0.9, 0.85, 0.4, 0.3, 0.2]
    correct = [True, True, True, False, False, False]
    curve = risk_coverage(confidences, correct)
    assert curve.full_coverage_risk == 0.5
    assert curve.risk_at(0.5) == 0.0
    assert curve.aurc < curve.full_coverage_risk


def test_risk_coverage_stays_flat_when_confidence_is_noise():
    confidences = [0.9] * 8
    correct = [True, False] * 4
    curve = risk_coverage(confidences, correct)
    assert abs(curve.risk_at(0.5) - 0.5) < 0.26  # ties, so it can only wobble


def test_threshold_table_accounts_for_every_episode():
    confidences = [0.95, 0.6, 0.4]
    correct = [True, False, False]
    for row in threshold_table(confidences, correct, thresholds=(0.5, 0.9)):
        assert abs(row["coverage"] + row["escalation_rate"] - 1.0) < 1e-9
        assert row["errors_reaching_production"] <= row["error_rate_on_answered"] + 1e-9
