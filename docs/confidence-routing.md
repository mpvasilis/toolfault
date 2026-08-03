# Confidence routing: what an escalation threshold actually buys

Confidence-gated escalation is the dominant pattern for agent human-handoff: when the agent's
confidence drops below a threshold, route the case to a person. This document is about whether
that works, how to measure it, and what it costs.

The short version: a threshold policy is only worth having if the confidence signal separates
right from wrong **on the episodes where the data was corrupted**. Aggregate calibration cannot
tell you that, and aggregate calibration is what almost everyone reports.

## Why aggregate calibration is the wrong instrument

Two agents can report the same overall ECE and be completely different products.

Agent A is a little overconfident everywhere. Its confidence still ranks correct answers above
wrong ones, so a threshold catches most of its errors and lets most of its correct answers
through.

Agent B is well calibrated on clean data and reports 0.95 on every episode where a tool quietly
lied to it. Its aggregate ECE looks fine, because clean episodes dominate the run. Its threshold
policy is worthless, because the failures it needs to catch sit at the top of the confidence
distribution alongside the successes.

Distinguishing them takes two numbers that measure different things:

- **ECE** answers "is 0.9 worth 0.9". A calibration question.
- **AUROC of confidence against correctness** answers "does higher confidence mean more likely
  right". A ranking question, and the one a threshold policy depends on.

An agent can pass one and fail the other. A miscalibrated but well-ranked signal is fixable with
a scaling constant. A well-calibrated but unranked signal is not fixable at all, because there
is no threshold that separates anything.

That is why this repo reports calibration **per fault family**, not once per run:

```bash
toolfault report --episodes results/bench-v1/episodes.jsonl
# see the "Calibration by fault family" table in REPORT.md
```

The table gives ECE, AUROC and risk at 80% coverage separately for clean, loud, silent and
adversarial episodes. The silent row is the one that decides whether a gate is real.

## The risk-coverage curve, and why it is the honest picture

Sweep the threshold from 0 to 1. At each point, **coverage** is the fraction of cases the agent
answers and **risk** is the error rate among those answers. Plot risk against coverage.

- A signal that carries information: risk falls as coverage falls. Answering only your most
  confident 60% gives a much lower error rate than answering everything.
- A signal that carries none: the curve is flat. Escalating 40% of cases buys nothing, because
  the 40% escalated has the same error rate as the 60% kept.

The area under this curve (AURC) summarises it in one number, and unlike accuracy it is
sensitive to exactly the property a gate needs. `stats.py` computes the curve, the AURC, and
risk at fixed coverage points.

## The table that makes it a decision

Coverage curves are for reading. This is for deciding. Per threshold:

| column | meaning |
|---|---|
| coverage | share of cases answered autonomously |
| escalation rate | share routed to a human, so, reviewer load |
| error rate on answered | error rate among the cases you did act on |
| wrong answers reaching production | share of **all** cases that became a wrong autonomous action |

The last column is the one to optimise, and it is not the same as the third. A policy can have a
low error rate on the cases it answers and still push plenty of wrong actions through, if
coverage is high. Conversely a very low threshold looks safe on that column while escalating so
much that the agent is not doing anything.

Pick the threshold from three numbers: the measured curve, the cost of a reviewer minute, and
the cost of a wrong action. In a regulated context the third number is not linear, and that
asymmetry is the whole reason a gate exists.

`toolfault report` emits this table per agent under "Escalation thresholds".

## What five models actually do

From the published grid: 4,992 episodes, 624 per agent, world seed 7, task seed 11, 16 tasks per
type. Confidence AUROC against correctness, computed **separately per fault family** on the episodes
where the agent submitted an answer.

| model | clean | loud | **silent** | adversarial | silent gap | silent ECE |
|---|---|---|---|---|---|---|
| `deepseek-v3.1` | n/a | 0.662 | **n/a** | n/a | -0.093 | 0.093 |
| `gpt-oss-20b` | 0.202 | 0.316 | **0.636** | 0.833 | +0.275 | 0.275 |
| `kimi-k2-0905` | 0.158 | 0.382 | **0.448** | 0.415 | +0.096 | 0.096 |
| `qwen3-235b-a22b` | 0.414 | 0.339 | **0.365** | 0.157 | +0.147 | 0.167 |
| `gpt-oss-120b` | 0.117 | 0.164 | **0.234** | 0.391 | +0.184 | 0.257 |

Read the silent column, because that is the one a review gate has to catch.

**Three of five models sit at or below 0.5.** `gpt-oss-120b` at 0.234 and `qwen3-235b-a22b` at 0.365
are not merely uninformative, they are inverted: on silent-fault episodes, the answers they were
*more* confident about were *more* likely to be wrong. A threshold policy built on those numbers
does not fail gracefully. It preferentially forwards the wrong answers to production and escalates
the right ones. `kimi-k2` at 0.448 is indistinguishable from a coin toss.

Only `gpt-oss-20b` (0.636) carries usable signal, and it is the smallest model in the grid, which is
not the direction anyone would predict.

**`deepseek-v3.1` shows n/a for a reason worth stating.** Of 112 silent-fault episodes it submitted
an answer on only 32, and every one of those 32 was correct. AUROC is undefined with one class
absent. It escalated 71.4% of silent-fault episodes and raised a data-quality flag on 99.1% of them.
That is a coherent and defensible policy: notice almost everything, hand almost everything to a
human, and be right when you do answer. Its unsafe rate on silent faults is 5.4%, the lowest of any
model. Its resolution is also the lowest, 46.4%, because escalation is not a resolution.

The comparison that makes the point: the `secondary-only` baseline also resolves 46.4% of
silent-fault episodes. Its unsafe rate is 53.6%. Identical accuracy, tenfold difference in what
reaches production. Any evaluation reporting one number would rank these two agents the same.

**The overconfidence gap widens on exactly the wrong slice.** Every model's mean-confidence-minus-
accuracy gap is larger on silent faults than on clean data: `gpt-oss-20b` goes from +0.051 to
+0.275, `gpt-oss-120b` from +0.112 to +0.184, `qwen3-235b` from +0.030 to +0.147. Aggregate
calibration would average that away against the clean and loud slices, which are the bulk of any
run.

### What this means for a gate

A confidence threshold is only a control if AUROC on the degraded slice is meaningfully above 0.5,
and on this benchmark it usually is not. The practical consequences:

1. **Measure AUROC per fault condition before shipping a threshold**, not overall. The aggregate
   numbers in the leaderboard column (0.412 for `kimi-k2`, 0.620 for `deepseek`) are dominated by
   clean and loud episodes and would have hidden all of the above.
2. **Prefer the deterministic signals.** `deepseek` reaches a 5.4% unsafe rate through flagging and
   escalation, not through calibration. Structural checks and source disagreement did that work.
3. **Do not port a threshold across models.** The five models here disagree about the sign of the
   relationship, not just its magnitude.

## What the reference policies show, and what they cannot

From the committed reference run, 624 episodes per policy, world seed 7, task seed 11, 16 tasks
per type:

| policy | slice | submitted | mean conf | accuracy | gap | ECE | AUROC |
|---|---|---|---|---|---|---|---|
| `naive` | clean | 64 | 0.900 | 1.000 | -0.100 | 0.100 | n/a |
| `naive` | loud | 384 | 0.900 | 0.349 | **+0.551** | 0.551 | **0.500** |
| `naive` | silent | 112 | 0.900 | 0.500 | **+0.400** | 0.400 | **0.500** |
| `oracle` | clean | 64 | 0.921 | 1.000 | -0.079 | 0.079 | n/a |
| `oracle` | loud | 325 | 0.910 | 1.000 | -0.090 | 0.090 | n/a |
| `oracle` | silent | 85 | 0.866 | 1.000 | -0.134 | 0.134 | n/a |

Read this carefully, because it is a demonstration of the instrument rather than a finding about
models.

The `naive` policy reports a constant 0.900. Its AUROC is therefore exactly 0.5 by construction:
constant confidence cannot rank anything. Its overconfidence gap grows from -0.100 on clean data
to +0.551 under loud faults, which is what a fixed self-report looks like when accuracy moves and
the number does not. It is the worst case for a threshold gate, and it is what a real agent
approximates whenever it answers 0.95 out of habit.

The `oracle` is the useful comparison. Its accuracy is 1.0 everywhere, and its stated confidence
still declines with the severity of what it found: 0.921 clean, 0.910 loud, 0.866 silent, 0.838
adversarial. It lowers its own confidence when it raises a data-quality flag. AUROC is undefined
because there are no errors to rank, which is the correct output and not a gap in the table.

**What this cannot show:** whether real models rank their errors. Both reference policies report
confidence from a fixed rule, so neither can produce an informative AUROC. That row needs the
model grid, and it is the number this document exists to report. It lands with `bench-v1`.

## Cheaper signals that should gate first

The model's self-report should be the last gate, not the first, because three better signals are
deterministic and free:

1. **Structural checks.** Percentages that do not sum to 100. An amount three orders of magnitude
   from its own prior year. A returned identifier that is not the one requested. A `retrieved_at`
   older than the as-of date asked for. Each of these is a few lines of code, each catches a
   specific silent fault outright, and none of them can be talked out of it.
2. **Source disagreement.** Two independent sources differing on a decisive fact is a hard
   escalation trigger that needs no probability at all.
3. **Provenance gaps.** A decisive field with a single source, or with a stale one, escalates on
   the strength of its metadata.

The benchmark exposes this directly through `data_quality_flags`, which is machine-checkable, has
a measured false-positive rate on the clean condition, and does not require trusting a number the
model made up. In the reference run the `oracle` flags on 0% of clean episodes and on the large
majority of episodes where a fault corrupted the data; `naive` flags on nothing, ever, which is
what a policy with no structural checks looks like.

## Practical notes

- **Verbalised confidence has coarse resolution.** Models emit 0.9 and 0.95 far more than they
  emit 0.87. Equal-mass calibration bins have to avoid splitting those ties or ECE is inflated by
  an artefact of the binning. This repo's `ece()` extends each bin past its cut until the
  confidence value changes, and a test pins the behaviour. It was a real bug, and it would have
  corrupted every published ECE number.
- **Recalibrate per model.** Thresholds do not transfer between models, and
  [Act or Escalate](https://arxiv.org/pdf/2604.08588) reports exactly this: implicit cost
  thresholds are a model-specific property. Treat a model swap as invalidating your gate.
- **Silent recovery is not the same as safe recovery.** In the smoke run, the model recovered a
  revenue figure that a tool had scaled by a thousand, and reported no flag. It got the right
  answer and told the operator nothing, so the broken upstream stayed broken. Recovery and
  reporting are separate metrics here for that reason.
- **A threshold is not a substitute for a tier.** No confidence level should authorise an
  irreversible action autonomously. See [production agents](production-agents.md) section 3.

## Reproducing

```bash
toolfault selftest --tasks-per-type 16 --run-id reference --out-dir results
toolfault report --episodes results/reference/episodes.jsonl
```

Free, no API key. The tables above come from that run.
