# Related benchmarks and motivation

ToolFault measures whether an agent can recover from faulty tool responses, recognise when
the evidence is insufficient, and express confidence that helps separate safe submissions
from cases needing review. A successful tool call can still return the wrong entity, an
outdated record, an incomplete list or an incorrect unit. These failures motivate evaluating
the agent's final decision together with the quality of its confidence signal.

Tool failures and agent reliability are established research problems. ToolMaze studies
replanning under explicit and implicit perturbations, including the difficulty of recovering
from plausible but corrupted outputs. ReliabilityBench combines repeated execution, task
perturbations and injected API failures. ToolFault builds on that motivation with a controlled
measurement of recovery, unsafe submissions, calibration and retrospective review routing
within the same episodes. [ToolMaze](https://arxiv.org/abs/2606.05806),
[ReliabilityBench](https://arxiv.org/abs/2601.06112).

## Comparison with related benchmarks

The table compares evaluation targets and experimental controls. It is not a score comparison,
since the tasks, agents, fault distributions and scoring rules differ.

| Benchmark | Evaluation setting | Main measurements | Connection to ToolFault |
|---|---|---|---|
| [tau-bench](https://arxiv.org/abs/2406.12045) and [tau2-bench](https://arxiv.org/abs/2506.07982) | Conversations with simulated users, domain policies and tools. tau2 adds a shared environment in which both user and agent can act. | Goal-state completion, repeated-trial reliability through pass^k, and communication and coordination failures. | Establish the importance of reliable outcomes over multiple attempts. ToolFault uses fixed user instructions and varies tool-response conditions. |
| [StableToolBench](https://arxiv.org/abs/2403.07714) | A virtual API server combining cached responses and API simulators, with an evaluation system derived from ToolBench. | Solvable pass and win rates, stability across changes in API availability and evaluation. | Addresses reproducibility of tool evaluation. ToolFault controls its synthetic world directly and deliberately introduces faults into otherwise deterministic tools. |
| [AgentDojo](https://arxiv.org/abs/2406.13352) | An extensible environment for tools operating on untrusted data, with realistic user tasks and prompt-injection attacks and defenses. | User-task success and the success of adversarial objectives. | Provides a broader adversarial evaluation setting. ToolFault includes one instruction-injection condition alongside operational and data-quality faults. |
| [ToolMaze](https://arxiv.org/abs/2606.05806) | Tool graphs with controlled topological complexity and explicit/implicit, transient/permanent perturbations. | Task performance and perturbation recovery, with analysis of replanning behavior. | Closely related motivation for separating visible errors from semantic corruption. ToolFault examines the accompanying confidence and escalation decisions in a registry workflow. |
| [ReliabilityBench](https://arxiv.org/abs/2601.06112) | Four domains, repeated runs, task perturbations, controlled tool failures and two agent architectures. | A reliability surface over repetition, perturbation intensity and fault intensity. | Motivates controlled fault injection and repeated execution. ToolFault holds the agent loop fixed and reports confidence and routing outcomes by fault condition. |
| [ToolFault-Bench, Atharva Vichare](https://huggingface.co/datasets/ATTY57/toolfault-bench) | 522 synthetic trajectories across six domains, each ending at a decision point after a potentially faulty result. | Fault classification, next-action choice and critical recovery decisions, including whether a timed-out write can safely be retried. | A separately maintained project with a similar name. ToolFault executes complete episodes with read-only tools and scores the resulting submission or escalation. |

## Research questions

ToolFault's measurement focus is the relationship between corrupted evidence, the agent's
decision and its reported confidence. The benchmark asks three questions.

1. **How does recovery change with the fault?** Matched tasks run under clean, loud, silent
   and adversarial conditions. Resolution, unsafe submission, escalation and unfinished
   rates reveal different consequences of a failure. Required escalations can resolve a
   task. A correct guess can still be unsafe when the evidence required escalation.
2. **Does confidence remain informative under corrupted evidence?** Calibration and ranking
   metrics are reported by fault family and condition. ECE and Brier score describe confidence
   against correctness. AUROC and risk-coverage curves describe whether confidence can rank
   correct submissions above incorrect ones. Equal confidence scores enter a threshold
   together. These statistics cover submissions with valid confidence, so their sample
   sizes and missing-confidence counts matter.
3. **What does a confidence threshold change operationally?** Retrospective routing keeps
   existing escalations, sends low-confidence or missing-confidence submissions to review,
   and retains unfinished episodes as a separate outcome. It reports autonomous acceptance,
   review volume and accepted unsafe submissions over all scorable episodes. This exposes
   the tradeoff that an accuracy or calibration score alone cannot express.

The definitions and worked results are in [confidence routing](confidence-routing.md).
The routing calculation estimates review volume. Human review time, reviewer accuracy and
monetary review costs have not been measured.

## Experimental strengths

- **Inspectable ground truth.** Tasks and answers derive from a seeded synthetic world.
  Scoring is programmatic, with explicit rules for answer correctness, safe escalation and
  unsafe submission. No model judge is needed to reproduce the scores.
- **Controlled comparisons.** The same task can be evaluated across applicable fault
  conditions. The injector can corrupt either the primary or corroborating source, making
  a fixed preference for one source an explicit baseline to test.
- **Reference policies and shortcut checks.** A tool-using reference policy establishes
  solvability on the evaluated grids. Single-source, guessing and unconditional-escalation
  baselines test specific shortcuts. Their results are regression checks, not a proof that
  every shortcut has been excluded. See [baseline analysis](not-gameable.md).
- **Auditable results.** Episode records, manifests and report generation support inspection
  of individual failures and recalculation of aggregate metrics. Additional world seeds and
  task-cluster uncertainty estimates help distinguish an observed result from a conclusion
  that would require more independent tasks or worlds.

These controls make ToolFault useful for comparing recovery policies, confidence prompts and
training interventions under a declared failure distribution. The contribution is the
combined measurement and reproducible test environment. Fault injection, calibration metrics
and escalation are established methods.

## Scope and next experiments

The current environment is a synthetic corporate registry with four task types, read-only
tools, a fixed agent loop and one fault condition per episode. It does not measure interaction
with a changing user, simultaneous fault types or recovery from side effects such as duplicate
payments. Its instruction-injection condition does not cover an adaptive attack suite.
Evidence citations are recorded but are not validated for semantic support, and a declared
data-quality flag does not establish that the agent identified the correct fault.

The published model grid uses one world seed. The additional scripted-policy audit across
worlds validates the environment and reference policies, but does not establish model
generalisation across worlds. Priorities for extending the evidence are model reruns on
independent world seeds, repeated attempts on matched tasks, thresholds selected on a
separate calibration split, and held-out fault evaluations. The
[post-training experiment](../posttrain/README.md) provides a training and evaluation path.
Its dataset and preprocessing checks should be distinguished from completed training or
measured improvements in a trained model.
