# Can this benchmark be scored without doing the work?

A benchmark that reports a number without publishing what a lazy policy scores has not shown you
anything. This page answers "could an agent get a respectable score by cheating", and
`tests/test_not_gameable.py` fails the build if the answer changes.

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

## What each baseline rules out

**`always-escalate`** escalates immediately without reading anything: **0.0%** resolved, **0.0%**
unsafe. This is the one that matters most to the metric design. Escalation is a safe outcome in this
benchmark, so if safety alone earned a score, the highest-scoring agent would be one that never does
any work. It earns nothing. Safety and competence are measured separately rather than collapsed into
one number.

**`modal-answer`** submits the most common answer for each task type and calls no tools at all. At
**19.1%** resolved and **80.9%** unsafe, the label distribution does not carry the benchmark. This
is the check that the task set is balanced enough that guessing fails.

**`one-call`** makes one real `search_registry` call, ignores the result, and then guesses. It scores
identically to `modal-answer`, to the episode. That identity is the point: making a tool call and not
using it is worth exactly zero, so a score cannot be earned through tool-call volume.

**`secondary-only`** is the important one. It reads only the corroborating sources and never the
primary tool: the archive instead of the live record, the batch screen instead of the per-name screen,
the filed register instead of the ownership endpoint. It does no reconciliation, it just prefers the
other source.

This policy was a genuine exploit. In the first version of this benchmark every fault landed on the
primary tool and the corroborating sources were always honest, so `secondary-only` scored close to
the reference policy while doing no cross-checking at all: single-source credulity with the sources
swapped. Silent faults now land on either side, chosen per episode from a digest of the task id, and
the result is the table above. `secondary-only` at 52.9% sits within four points of `naive` at 49.4%,
and both are more than forty points below the oracle. Whichever single source an agent decides to
trust, it loses about half of the silent-fault episodes. Only reconciliation recovers all of them,
which is the property the benchmark is meant to measure.

`test_reading_only_the_corroborating_sources_no_longer_wins` is the regression test, and it fails if
this row climbs back toward the oracle.

**`confident-guess`** runs the full `naive` policy and reports 0.99 confidence with empty flags on
every episode. Its resolution is identical to `naive`, to the episode, which is expected because it
does the same work. It is a calibration control rather than a shortcut, and it earns its place by
answering whether the calibration metrics can tell a graded confidence signal from a constant one.
They can: ECE moves from 0.390 to **0.480** while resolution does not move at all.

## Two limits of the confidence column

Every policy here reports confidence from a fixed rule, so every AUROC in the table is exactly 0.500.
That is the correct output for a constant signal, and it is not a finding about anything. It does
confirm the metric is not inventing structure that is absent, which is worth establishing before
reading a model's AUROC.

The oracle's AUROC is `n/a` rather than a number because it makes no errors, and AUROC is undefined
with one class absent. Its ECE of 0.104 reflects under-confidence rather than error: it states 0.92
on clean episodes and 0.87 on silent-fault episodes while being right every time, because it lowers
its own confidence when it raises a data-quality flag.

## What this does not rule out

- **A policy tuned against these specific tells.** An agent hard-coded to check `cache: HIT`, sum
  percentages and compare `snapshot_as_of` would score well without generalising to a fault type
  outside the set. Measuring that is what the held-out fault split in
  [the post-training experiment](../posttrain/README.md) is for.
- **Memorisation of this world.** Every number here comes from world seed 7. Results should be
  reported on a second seed too; `--world-seed` takes one, and ground truth is derived rather than
  stored, so any seed works.
- **Overfitting to one domain.** It is a corporate registry. A policy tuned for it may not transfer.
