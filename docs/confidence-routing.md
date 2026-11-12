# Confidence routing under tool failures

Confidence-gated escalation routes a case to a person when the agent's stated confidence
drops below a threshold. This document measures its effect on accepted unsafe submissions
and review volume in the published benchmark.

A useful threshold needs an informative confidence signal **on episodes where the data was
corrupted**. Aggregate calibration can conceal differences between clean and faulty conditions.

## Calibration and confidence ranking

Agents with similar aggregate calibration can respond differently to a confidence threshold.
A confidence signal may rank correct answers above incorrect answers even when its numerical
values are miscalibrated. Conversely, high confidence on both correct and incorrect submissions
limits the errors that a threshold can divert.

- **ECE** measures the difference between stated confidence and observed accuracy within bins.
- **Confidence AUROC** measures whether correct submissions receive higher confidence than
  incorrect submissions.

An agent can perform well on one measure and poorly on the other. A miscalibrated but well-ranked
signal may benefit from recalibration on held-out data. A signal that assigns the same score to
correct and incorrect answers cannot separate them through a threshold alone.

That is why this repo reports calibration **per fault family**, not once per run:

```bash
toolfault report --episodes results/bench-v1/episodes.jsonl
# see the "Calibration by fault family" table in REPORT.md
```

The table gives ECE, AUROC and risk at up to 80% coverage separately for clean, loud, silent and
adversarial episodes. Each slice reveals how the gate behaves under that class of faults.

## The risk-coverage curve

The curve retains submissions with valid confidence at or above each distinct score.
Coverage is the share of those submissions retained and risk is the share with incorrect answers.
Tied scores enter together because a confidence threshold has no way to distinguish them.
`stats.py` reports risk at the largest attainable nonempty coverage at or below a target.
AURC integrates risk across the resulting coverage increments.

## Routing metrics

The operational table uses every scorable episode in a slice.

| column | meaning |
|---|---|
| accepted | share of episodes with a retained autonomous submission |
| sent to review | original escalations plus submissions diverted by the gate |
| unfinished | share without a terminal tool call |
| unsafe accepted / all | accepted unsafe submissions as a share of all scorable episodes |
| unsafe / accepted | share of retained submissions that were unsafe |

`toolfault report` emits this table for all scorable episodes and for the silent-fault slice.
The review rate estimates review volume. The experiment records no human review times or outcomes.

## Published model results

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

The silent column isolates episodes in which a tool can return plausible but unreliable data.

The task-cluster bootstrap gives a 95% interval of 0.101–0.363 for GPT-OSS-120B, consistent
with higher confidence on incorrect submissions in this world. The intervals for Qwen3-235B
(0.144–0.601) and Kimi K2 (0.262–0.578) include 0.5. GPT-OSS-20B has an interval of
0.500–0.758. The estimates use a single synthetic world and cover submissions with valid confidence.

**`deepseek-v3.1` shows n/a for a reason worth stating.** Of 112 silent-fault episodes it submitted
an answer on only 32, and every one of those 32 was correct. AUROC is undefined with one class
absent. Six correct answers were still unsafe because the episode required escalation. It
escalated 71.4% of silent-fault episodes and raised a data-quality flag on 99.1% of them. Its
silent-fault unsafe rate was 5.4% and its resolution rate was 46.4%. Required escalations
count as resolutions.

The `secondary-only` baseline also resolves 46.4% of silent-fault episodes, with an unsafe rate
of 53.6%. The same resolution rate therefore accompanies substantially different safety and
review outcomes in this benchmark.

**Four of five models have a larger positive overconfidence gap under silent faults.** The
mean-confidence-minus-accuracy gap for `gpt-oss-20b` rises from +0.051 on clean episodes to
+0.275 on silent faults, `gpt-oss-120b` from +0.112 to +0.184, and `qwen3-235b` from +0.030
to +0.147. DeepSeek remains underconfident among its submissions, with a signed gap changing
from −0.023 to −0.093. Aggregate calibration can conceal these condition-specific differences.

### What this means for a gate

The same threshold has different review and safety effects across models. For GPT-OSS-20B,
a retrospective gate at 0.95 would divert 9.8 percentage points more episodes to review and
reduce accepted unsafe submissions by 4.5 points. For Qwen3-235B it would divert 4.5 points
more episodes with the observed unsafe rate still at 8.0%. Reproduce these calculations
with `uv run python experiments/audit.py --out runs/audit`.

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

These policies provide controls for the calibration metrics.

The `naive` policy reports a constant 0.900. Its AUROC is 0.5 on slices containing both correct
and incorrect submissions, because constant confidence cannot rank them. Its overconfidence
gap changes from -0.100 on clean episodes to +0.551 under loud faults as accuracy declines.
A confidence threshold either retains or diverts every submission from this policy.

The `oracle` is the useful comparison. Its accuracy is 1.0 everywhere, and its stated confidence
still declines with the severity of what it found: 0.921 clean, 0.910 loud, 0.866 silent, 0.838
adversarial. It lowers its own confidence when it raises a data-quality flag. AUROC is undefined
because there are no errors to rank, which is the correct output and not a gap in the table.

**What this cannot show:** whether model confidence ranks model errors. These reference policies
provide controls for the metrics. The model observations are reported separately in `bench-v1`.

## Deterministic checks

Structured outputs also permit checks that operate independently of stated confidence.
The benchmark includes three types.

1. **Structural checks.** Percentages that do not sum to 100. An amount three orders of magnitude
   from its own prior year. A returned identifier that is not the one requested. A `retrieved_at`
   older than the as-of date asked for. These checks can identify specific inconsistencies
   when the required fields and metadata are available.
2. **Source disagreement.** Two independent sources differing on a decisive fact is a hard
   escalation trigger that needs no probability at all.
3. **Provenance gaps.** A decisive field with a single source, or with a stale one, escalates on
   the strength of its metadata.

The benchmark records `data_quality_flags` and measures how often at least one flag appears,
including a false-positive rate on clean episodes. These are agent declarations. The scorer
does not verify that a flag identifies the actual fault. In the reference run, the `oracle`
flags on 0% of clean episodes and on most episodes where a fault corrupted the data. The
`naive` policy raises no flags.

## Practical notes

- **Verbalised confidence has coarse resolution.** Models emit 0.9 and 0.95 far more than they
  emit 0.87. Equal-mass calibration bins have to avoid splitting those ties or ECE is inflated by
  an artefact of the binning. This repo's `ece()` extends each bin past its cut until the
  confidence value changes. A regression test checks that equal confidence values remain in
  the same bin.
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
toolfault selftest --tasks-per-type 16 --run-id reference-new --out-dir runs
toolfault report --episodes runs/reference-new/episodes.jsonl
```

The reference-policy check needs no API key. Use an unused run ID for each execution.
The model tables use the committed `results/bench-v1/episodes.jsonl`.
