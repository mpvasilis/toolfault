# Related work, and what is left for this benchmark to contribute

Written after building the thing, which is the wrong order. The first version of this repo
was going to claim that agents recover from loud tool failures and fail on silent ones while
their confidence does not move. That claim is already published, by more than one group. What
follows is the honest map of the surrounding work and the narrower thing that is actually
left.

## The claim I was going to make, already made

**ToolMaze / "When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM
Agents"** ([arXiv 2606.05806](https://arxiv.org/abs/2606.05806)) organises tool perturbations
on a 2x2 of explicit against implicit and transient against permanent. Their explicit and
implicit axis is the same cut as the loud and silent families here. They report that implicit
semantic failures produce the sharpest drops, and they name the cause: systemic over-trust in
corrupted outputs. Perturbation Recovery Rate falls by roughly 37 points in those conditions.

So the loud-against-silent finding is prior work. This repo cites it rather than rediscovering
it, and the taxonomy here should be read as an independent implementation of a distinction
that paper established, not as a new one.

**ReliabilityBench** ([arXiv 2601.06112](https://arxiv.org/abs/2601.06112)) applies
chaos-engineering-style fault injection with, in its own words, timeouts, rate limits, partial
responses and schema drift. That is four of the six loud faults here, near enough exactly. It
reports a reliability surface over three axes: consistency via pass^k across repeated
executions, robustness to task perturbation, and fault tolerance under tool failure. It covers
two models and two agent architectures across four domains over 1,280 episodes.

**ToolMisuseBench** ([arXiv 2604.01508](https://arxiv.org/html/2604.01508v1)) is an offline
deterministic benchmark with replayable fault injection over CRUD, retrieval, file and
scheduling environments, 6,800 tasks, reporting success, invalid call behaviour, policy
violations, recovery quality and budgeted efficiency. It explicitly argues for per-fault
conditioned analysis on the grounds that patterns disappear in aggregate averages, which is
the same argument made here for reporting per condition rather than as one score.

Also in this area: **MAS-FIRE** ([arXiv 2602.19843](https://arxiv.org/html/2602.19843v1)) for
multi-agent systems, **AgentProp-Bench** ([arXiv 2604.16706](https://arxiv.org/html/2604.16706))
on judge reliability and propagation cascades from parameter-level injection, **StressWeb**
([arXiv 2604.16385](https://arxiv.org/pdf/2604.16385)) for web agents under interaction
variability, and **Self-Healing Agentic Orchestrators**
([arXiv 2606.01416](https://arxiv.org/html/2606.01416v1)) on orchestration policies under
matched fault conditions.

## Reliability metrics

**tau-bench** ([OpenReview](https://openreview.net/forum?id=roNSXZpUDN)) introduced pass^k, the
probability that an agent succeeds on all k attempts, and showed the reliability gap it
exposes: 61% pass@1 against 25% pass^8 on retail tasks. tau2-bench extends it with more
domains. pass^k is the right vocabulary for run-to-run variance and this repo adopts it rather
than inventing a substitute.

**Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents**
([arXiv 2603.29231](https://arxiv.org/pdf/2603.29231)), **Towards a Science of AI Agent
Reliability** ([arXiv 2602.16666](https://arxiv.org/html/2602.16666v1)) and the **Holistic
Agent Leaderboard** ([arXiv 2510.11977](https://arxiv.org/pdf/2510.11977)) argue the general
case that headline single-run scores mislead.

## Confidence, calibration and escalation

**The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents**
([arXiv 2601.07264](https://arxiv.org/pdf/2601.07264)) is the closest work on the calibration
side. It finds that tool use has systematic but heterogeneous effects on agent calibration,
against the assumption that tool augmentation uniformly helps or hurts.

**Agentic Confidence Calibration** ([arXiv 2601.15778](https://arxiv.org/abs/2601.15778))
introduces Holistic Trajectory Calibration, extracting process-level features across a whole
trajectory rather than scoring one final answer.

**Act or Escalate? Evaluating Escalation Behavior in Automation with Language Models**
([arXiv 2604.08588](https://arxiv.org/pdf/2604.08588)) frames acting against escalating as a
decision under uncertainty and evaluates it across five domains. It finds self-estimates
miscalibrated in model-specific ways, finds marked differences in the implicit cost thresholds
models use, and concludes that escalation behaviour is a model-specific property that should
be characterised before deployment. It does not inject tool failures.

**ConfidenceBench** ([arXiv 2607.20526](https://arxiv.org/html/2607.20526v1)) benchmarks
calibration in language models generally. Risk-coverage curves, AURC and ECE are standard
instruments in this literature and are used here as such, not as contributions.

## What is left

Every piece above holds one half of the question. The fault-injection benchmarks measure
recovery and do not measure confidence: ReliabilityBench reports no confidence signal, and
ToolMaze reports recovery rate without calibration. The calibration and escalation work
measures confidence and does not inject tool faults: Act or Escalate characterises escalation
thresholds on clean inputs.

The gap between them is the question this repo is for.

**When a tool silently returns plausible wrong data, does the agent's own confidence signal
move, and can that signal be used to route those specific episodes to a human at a cost
anyone would accept?**

That is a joint measurement, and the three things needed to answer it are the three things
this repo adds:

1. **Calibration reported per fault condition, not in aggregate.** ECE, AUROC and a
   risk-coverage curve computed separately for clean, loud, silent and adversarial episodes.
   Aggregate calibration hides the case that matters: a model can be adequately calibrated
   overall and completely uninformative exactly where the data was corrupted.
2. **Escalation as a scored outcome with a cost, not as a failure.** The metric set separates
   a wrong answer that reaches production from an escalation that costs a human review, so a
   threshold policy can be priced. `docs/confidence-routing.md` reports the resulting
   coverage against error-rate table.
3. **A known ceiling and a proof the benchmark cannot be gamed.** A hand-written reference
   policy resolves all fourteen conditions at 1.0 using only the tools, enforced on every push
   in CI, so a low score is agent failure and not an unwinnable condition. ReliabilityBench
   offers no such evidence, and none of the benchmarks above publish degenerate baselines.
   Against that: `docs/not-gameable.md`.

Two smaller differences worth stating. Scoring here is entirely programmatic, derived from
world state by pure functions, with no LLM judge anywhere; AgentProp-Bench exists partly
because judge reliability is a live problem. And the whole leaderboard is open-weight models
through a public API, so a stranger can rerun every number for about twenty dollars, which is
not true of results reported on closed models.

## What this repo does not do

- **No multi-turn user simulation.** tau-bench's simulated user is a large part of what makes
  it realistic and none of it is here. These tasks are single-shot instructions.
- **No compound faults.** One fault per episode, for attribution. ToolMaze's transient against
  permanent axis is partly covered by the retry against second-source recovery split, but the
  interaction of two simultaneous faults is not measured.
- **No agent architectures.** ReliabilityBench varies ReAct against Reflexion. This varies only
  the model behind a fixed loop.
- **One domain.** A synthetic corporate registry. ToolMisuseBench spans four environments.
- **No claim of scale.** Hundreds of episodes per model, not thousands.
