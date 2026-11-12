# Agent controls for tool failures

This document describes validation, escalation and logging controls for agents that consume
external tool responses. ToolFault measures some of these behaviours in a synthetic registry.
The production policies below are design proposals. The benchmark does not validate their
effectiveness in a deployed system or measure human-review outcomes.

## 1. What the data says, and what it does not

In the published model grid, resolution falls under silent faults for all five models.
Confidence-based routing has different safety and review-volume effects across those models.
The measurements are conditional on one synthetic world. See
[confidence routing](confidence-routing.md) for the results and uncertainty estimates, and
[related work](related-work.md) for the comparison with other benchmarks.

The `naive` reference policy scores 100% on clean and instruction-injection episodes in the
reference grid. It ignores the injected text, but loses about half of the silent-fault
episodes because it trusts one source.

This scripted baseline ignores instruction text and has no cross-checking procedure. The
scripted oracle demonstrates that deterministic code can also reconcile structured sources.
The relevant distinction is which inputs a component interprets and which checks it performs.

## 2. Separate model outputs from enforced controls

Specify which outputs the model may produce and which checks must pass before an action runs.

**Keep in code:** the decision rule, the corroboration requirement, retry and backoff policy,
the escalation threshold, idempotency, budget limits and audit logging. These controls can be
implemented and tested explicitly. Their guarantees still depend on validating untrusted data
and enforcing the checks at the action boundary.

**Leave to the model:** reading unstructured documents, resolving names and near-duplicates,
summarising evidence, deciding which tool to call next within an allowed set, and drafting the
reviewer's explanation.

Record the version of each decision rule and the evidence used to evaluate it. Test that an
instruction embedded in a tool response cannot bypass the required checks. Computing a verdict
in code still requires validation of any structured inputs derived from model output.

## 3. Action permissions and review requirements

The following example policy assigns review requirements according to an action's consequences
and reversibility. Deployment-specific permissions and review capacity require separate validation.

| tier | example | autonomy | gate |
|---|---|---|---|
| read-only | look up a filing, summarise a document | full | none, but logged |
| reversible write | draft a note, attach a document, tag a case | full | logged, undo available |
| consequential | approve onboarding, release a payment, reject an application | never autonomous | human decision on the agent's evidence |
| irreversible | delete records, notify a regulator, close an account | never agent-initiated | human initiates, agent assists |

Declare permissions when registering each tool and enforce them at execution. Evaluate sequences
of actions as well. Several reversible writes may together constitute a consequential approval.

## 4. Tool contracts that make failures visible

Response schemas and metadata can expose several fault types used in this benchmark. The
following checks detect specific inconsistencies, but still depend on the reliability of the
metadata and the source.

| failure | response contract and check |
|---|---|
| stale record | every payload carries `retrieved_at` and a cache indicator; the caller compares against the as-of date it asked for |
| truncated list | list responses carry `total` and `has_more`; test that callers retrieve remaining pages when required |
| unit shift | amounts are objects, `{value, currency, scale}`, never bare numbers |
| wrong entity | every response echoes the identifier that was requested, and the caller asserts it matches |
| schema drift | responses are versioned and validated at the boundary; an unknown field shape fails loudly rather than reading as absent |
| contradiction | provenance on every field, so a disagreement names its two sources instead of surfacing as one value |

Validate that returned records match the requested identifier, period and unit. When
corroborating a fact, check whether the sources share upstream data or caches. Shared failures
can produce matching but incorrect responses. ToolFault injects faults into either source
and measures policies that trust each source without reconciliation.

## 5. Retry policy

Classify errors using the endpoint's status codes and response body, and enforce retry limits
in the adapter.

- **Transient** (timeout, 502, 503, 429): retry with exponential backoff and jitter, cap the
  attempts, respect `Retry-After`, and make every write idempotent with a caller-supplied key
  so a retry after an ambiguous timeout cannot double-execute.
- **Terminal** (401, 403, 404, schema violation, quota exhausted): do not retry. Escalate or
  fail. An agent that retries a rejected credential until its budget is gone has converted a
  clear error into a silent non-completion.

The model adapter previously retried an exhausted API balance because the provider returned
HTTP 429. It now checks the error body for terminal quota failures before retrying.

## 6. The escalation gate

The simulated policy accepts submissions at or above a confidence threshold and sends lower
or missing-confidence submissions to review. Existing escalations remain in the review volume.

**Measure confidence by fault condition.** Report AUROC, uncertainty and the risk–coverage
curve for each relevant slice. AUROC near 0.5 indicates weak overall ranking, but does not
determine the error rate at every threshold. Inspect the measured routing outcomes directly.
See [confidence routing](confidence-routing.md) for definitions and results.

**Measure review capacity and error costs.** Threshold selection needs an estimate of review
workload and the consequences of accepted errors. Choose it on a calibration split and assess
it on held-out data. Reevaluate it when the model or task distribution changes.

**Combine confidence with explicit checks.** Candidate controls include:

1. **Structural checks.** Percentages that do not sum, an amount three orders of magnitude off
   its own history, a returned id that is not the requested id. These are deterministic, and
   they catch several of the silent faults outright.
2. **Source disagreement.** Two independent sources differing is a hard escalation trigger and
   needs no confidence estimate at all.
3. **Provenance gaps.** A decisive field with one source, or with a stale source, escalates.
4. **Verbalised confidence.** Last, and only where the three above are silent.

Log the result of each check so its false-positive rate and contribution can be evaluated.

## 7. Episode logging

A production trace can record the following fields per episode:

- the task, the model and its version, the prompt template hash, the decoding parameters;
- **every tool call, its arguments, its full response, and its latency**;
- the provenance of each fact that entered the decision;
- the structural checks that ran and what they returned;
- the confidence, the threshold in force, and the routing outcome;
- the terminal action, and whether a human changed it.

These records support investigation of tool errors and how an agent used the returned data.
Reviewer corrections can supply candidate training labels after quality checks.

Sampling: keep every escalation, every case a human overturned, and a fixed fraction of clean
completions. The clean sample is what tells you your gate is not quietly drifting.

## 8. Evaluation as a release gate

Run the evaluation as part of the release process and define acceptance criteria before
reviewing the results:

- **A frozen held-out set**, defined before the change you are evaluating, with its size stated.
- **Fault conditions in the gate, not only happy paths.** A change that improves clean accuracy
  and doubles the confidently-wrong rate is a regression, and only a fault-conditioned eval sees
  it.
- **Programmatic scoring where ground truth is available.** For model-judged outcomes,
  document the judge and assess its agreement with independently reviewed examples.
- **A solvability check.** This benchmark requires its reference policy to resolve every
  scheduled condition in the CI grid. A production evaluation can use scripted reference
  behaviour over recorded fixtures where applicable.
- **Separate unsafe-submission criteria.** A gain in resolution should not conceal an
  increase in accepted unsafe answers.

## 9. What this design does not solve

- **Compound failures.** One fault at a time is a benchmark simplification. In production two
  things break together, and corroboration across two simultaneously degraded sources gives you
  a confident agreement on a wrong answer.
- **Adversarial tool operators.** Prompt injection in tool output is in the benchmark. A tool
  provider who tailors corruption to your specific checks is not.
- **Long horizons.** Single-shot tasks. Errors that compound across a hundred steps, and the
  memory and replanning problems that come with them, are a different design document.
- **Reviewer capacity.** Escalations can exceed available review capacity. The benchmark
  measures referral volume but does not model queues or response times.
- **Human-review quality.** Reviewers may accept incorrect summaries. The effect of showing
  source evidence and disagreements needs a separate study.
