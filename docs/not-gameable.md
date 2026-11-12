# Shortcut baseline evaluation

Simple policies help identify shortcuts in a benchmark's scoring rules and data. This page
measures guessing, unconditional escalation and reliance on a single source.
`tests/test_not_gameable.py` checks bounds on these baselines. The checks cover the specified
policies and do not establish that every possible shortcut has been excluded.

Measured over the full grid, 624 episodes per policy, world seed 7, task seed 11, 16 tasks per
type. Zero cost, no API key.

| policy | resolved | unsafe | silent faults | silent unsafe | escalated | ECE | conf AUROC |
|---|---|---|---|---|---|---|---|
| `oracle` | **100.0%** [99-100] | 0.0% | **100.0%** [97-100] | 0.0% | 13.8% | 0.104 | n/a |
| `secondary-only` | 52.9% [49-57] | 47.1% | 46.4% [37-56] | 53.6% | 0.0% | 0.357 | 0.500 |
| `naive` | 49.4% [45-53] | 50.6% | 48.2% [39-57] | 51.8% | 0.0% | 0.390 | 0.500 |
| `confident-guess` | 49.4% [45-53] | 50.6% | 48.2% [39-57] | 51.8% | 0.0% | **0.480** | 0.500 |
| `one-call` | 19.1% [16-22] | 80.9% | 13.4% [8-21] | 86.6% | 0.0% | 0.709 | 0.500 |
| `modal-answer` | 19.1% [16-22] | 80.9% | 13.4% [8-21] | 86.6% | 0.0% | 0.709 | 0.500 |
| `always-escalate` | 0.0% [0-1] | 0.0% | 0.0% [0-3] | 0.0% | 100.0% | n/a | n/a |

```bash
pytest tests/test_not_gameable.py -q      # the bounds, enforced
toolfault selftest --tasks-per-type 16    # the reference policies, same grid
```

## Baseline definitions and results

**`always-escalate`** escalates immediately without reading any evidence. It achieves **0.0%**
resolution and **0.0%** unsafe submissions in this grid. Reporting both metrics distinguishes
a policy that avoids unsafe answers from one that also completes the tasks.

**`modal-answer`** submits the most common answer for each task type and calls no tools at all. At
**19.1%** resolved and **80.9%** unsafe, this policy measures performance from a fixed guess
without evidence retrieval.

**`one-call`** makes one real `search_registry` call, ignores the result, and then guesses. It scores
identically to `modal-answer` on every episode. The scorer does not reward tool-call volume.

**`secondary-only`** reads only the corroborating sources and never the
primary tool: the archive instead of the live record, the batch screen instead of the per-name screen,
the filed register instead of the ownership endpoint. It does no reconciliation, it just prefers the
other source.

In the first version, faults affected only the primary tool. The corroborating sources were
uncorrupted, so `secondary-only` scored close to the reference policy without cross-checking.
Silent faults now land on either side, chosen per episode from a digest of the task id, and
the result is the table above. `secondary-only` at 52.9% sits within four points of `naive` at 49.4%,
and both are more than forty points below the oracle. Whichever single source an agent decides to
trust, these two policies lose about half of the silent-fault episodes. The reference policy
reconciles the evidence and recovers all evaluated episodes. This is the behaviour the
benchmark is designed to reward.

`test_reading_only_the_corroborating_sources_no_longer_wins` is the regression test, and it fails if
this row climbs back toward the oracle.

**`confident-guess`** follows the `naive` policy but reports confidence 0.99 and no quality flags
on every episode. Resolution remains identical, while ECE changes from 0.390 to **0.480**.
This control measures how calibration error changes when confidence increases without a
corresponding change in accuracy.

## Two limits of the confidence column

The five submitting shortcut policies report constant confidence, so their AUROC is 0.500.
This confirms that the metric assigns chance ranking to a constant signal when both correct
and incorrect submissions occur. It is a control for the metric, rather than a model finding.

The oracle's AUROC is `n/a` rather than a number because it makes no errors, and AUROC is undefined
with one class absent. Its ECE of 0.104 reflects under-confidence rather than error: it states 0.92
on clean episodes and 0.87 on silent-fault episodes while being right every time, because it lowers
its own confidence when it raises a data-quality flag.

## What this does not rule out

- **A policy tuned against these specific tells.** An agent hard-coded to check `cache: HIT`, sum
  percentages and compare `snapshot_as_of` would score well without generalising to a fault type
  outside the set. Measuring that is what the held-out fault split in
  [the post-training experiment](../posttrain/README.md) is for.
- **Memorisation of this world.** The table above uses world seed 7. The
  [six-world audit](../results/audit-2026-11-11/AUDIT.md) extends the oracle and single-source
  checks to additional generated worlds. Model generalisation still requires model reruns.
- **Overfitting to one domain.** It is a corporate registry. A policy tuned for it may not transfer.
