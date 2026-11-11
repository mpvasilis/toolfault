# Post-training experiment

**Status — preprocessing validated. GPU training and matched base/adapter evaluation are
pending.** Dataset generation and the CPU tokenizer preflight have been run. This document
reports those checks and the planned evaluation protocol. It contains no measured
fine-tuning improvement.

The experiment asks whether supervised fine-tuning on selected fault types improves recovery
on fault types excluded from training. Both the benefit of training and the extent of transfer
are open questions. The fault split and qualitative hypotheses are recorded in
[`data/manifest.json`](data/manifest.json) before training.

## Reference-policy demonstrations

The `oracle` policy supplies demonstrations. It resolves every episode in the evaluated
reference grids using the available tools, and its recovery branches are inspectable in
[`scripted.py`](../src/toolfault/agents/scripted.py). Dataset generation runs on CPU without
model API calls. The builder excludes unresolved demonstrations and reports their count.

## Splits and hypotheses

Eight conditions are used for training, including `clean`. Six fault conditions are held out.
The held-out set is selected to probe specific kinds of transfer.

| Held out | Family | Transfer question | Hypothesis recorded before training |
|---|---|---|---|
| `rate_limit` | loud | Can retry behavior transfer to a new error code? | recovery transfers |
| `schema_drift` | loud | Can the model handle a recovery mode absent from demonstrations? | little or no gain |
| `stale_record` | silent | Can a demonstrated recovery transfer to a new warning sign? | partial gain |
| `contradiction` | silent | Can escalation transfer to a new trigger? | evaluate escalation transfer |
| `unit_shift` | silent | Can cross-checking transfer to a new magnitude error? | partial gain |
| `injected_instruction` | adversarial | Can behavior transfer to an excluded fault family? | little or no gain |

`auth_expired` remains in training to provide examples in which escalation is the correct
action. This makes the `contradiction` condition a test of transferring that action to a
different trigger.

**World split.** Training uses world seeds 23, 29 and 31. Evaluation uses world seed 7.
The builder rejects a training seed equal to the declared benchmark seed. Different seeds
change the registry, but do not guarantee disjoint names, identifiers or values. Identifier
formats and the underlying world generator remain shared.

**Validation split.** All retained episodes for a world-prefixed task ID belong to the same
split. This prevents the same task from appearing in both training and validation under
different conditions. Related entities can still occur in different tasks, so validation
loss is not an independent-world generalisation result.

## Reproduced dataset

Run commands from the repository root.

```bash
python posttrain/build_dataset.py
```

The default build produces 2,733 demonstrations across three worlds. A cap of 110 examples
per condition is applied before splitting, producing **765 training and 98 validation
examples**.

| Training condition | Examples |
|---|---|
| `truncated_list` | 105 |
| `malformed_json` | 101 |
| `auth_expired` | 96 |
| `timeout` | 96 |
| `http_500` | 94 |
| `clean` | 98 |
| `wrong_entity` | 92 |
| `empty_result` | 83 |

Silent faults account for 280 of 765 training examples, or 36.6%. There are 91 training
demonstrations ending in escalation. The cap limits the contribution of conditions supported
by many tasks.

The JSONL files are gitignored and can be regenerated from the recorded world, task and split
seeds. The manifest records SHA-256 prefixes over the ordered message payloads. The reproduced
training digest is `2d1d01bbeab421f8` and the validation digest is `f351c0742ba1b3c1`.

## CPU preflight and provenance

The preflight checks that conversations render, assistant masks are nonempty, supervised tool
calls and arguments match the source, tool-response markers are excluded from supervised
text, and all sequences fit the selected context length.

```bash
python posttrain/validate_template.py --model Qwen/Qwen3-4B-Instruct-2507
python posttrain/validate_template.py --model Qwen/Qwen3-4B-Instruct-2507 \
  --data posttrain/data/val.jsonl
```

The completed CPU check used `transformers==5.22.2`, tokenizer files from
`Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, and TRL's
`qwen3_instruct_2507_training.jinja` at commit
`e5677dace3f8ab512ab40ba95fe8a887f5d37a45`. All **863 conversations** passed. The longest
sequence was **3,033 tokens**, below the 4,096-token training limit. The median supervised
share in training was **8.1%**.

These revisions describe the completed check. The default commands resolve model and template
versions through the installed libraries. To reproduce the same artifacts, use the pinned
tokenizer files as a local `--model` directory and pass the pinned template with `--template`
to both validators and the trainer. The runner forwards `TRAINING_TEMPLATE` to all three
commands. The CPU preflight does not validate GPU training or serving compatibility.

## Planned training

Defaults target `Qwen/Qwen3-4B-Instruct-2507` with QLoRA, 4-bit NF4 weights, rank 32,
alpha 64, a 4,096-token training context and two epochs. A single 24 GB GPU is the intended
target. Peak memory use and runtime remain to be measured.

```bash
python posttrain/train_qlora.py --model Qwen/Qwen3-4B-Instruct-2507
```

Loss is applied to assistant turns. Pass the same `--template` used in preflight when selecting
an explicit template. `--allow-full-loss` permits training on the complete sequence when an
assistant-only template cannot be selected. That changes the objective and must be reported
as a separate experimental configuration.

## Planned evaluation

The runner serves both arms with vLLM and uses the same harness, world seed 7, task seed 11,
task count and serving configuration. The tuned arm loads the adapter through LoRA support.
It does not merge and requantize the model.

```bash
RUN_PREFIX=posttrain bash posttrain/run_on_gpu.sh
```

The script evaluates the base model, builds the dataset, validates both splits, trains,
evaluates the adapter and writes the comparison. Set a fresh `RUN_PREFIX` for each experiment.
The default is a timestamped prefix. `SKIP_TRAIN=1` evaluates an existing adapter.

```bash
python posttrain/compare.py \
  --base results/posttrain-base/episodes.jsonl \
  --tuned results/posttrain-tuned/episodes.jsonl
```

`compare.py` requires matching episode identities and task ground truth. It reports endpoint
errors and excludes affected pairs from both arms. It separates trained and held-out
conditions, pairs results by task, and reports sign-flip permutation tests with
Benjamini–Hochberg correction. Report resolution, unsafe submission, escalation and confidence
metrics together, including pair counts and excluded errors.

## Interpreting the results

- Gains confined to trained conditions would provide limited evidence of transfer.
- Gains on `rate_limit`, `stale_record` or `unit_shift` would support transfer to new triggers
  within the evaluated recovery patterns.
- Gains on `contradiction` would support transfer of escalation to a new trigger.
- Gains on `schema_drift` or `injected_instruction` would differ from the recorded hypotheses
  and warrant inspection of traces and alternative explanations.
- Higher resolution accompanied by more unsafe submissions would expose a safety tradeoff.
- Null results should include effect estimates, uncertainty and sample sizes. A nonsignificant
  test alone does not establish the absence of an effect.

Even improvement across the held-out faults would remain evidence from one synthetic domain
and one evaluation world. Independent worlds, repeated training seeds and different task
families would be needed to assess broader generalisation.

## Runtime environment and accounting

The GPU scripts install dependencies with lower version bounds. They do not define a locked
training environment. Record the resolved packages, model revision, template, GPU, driver and
CUDA versions alongside each completed run. Check that the installed PyTorch and vLLM builds
match the host environment before starting the experiment.

`bootstrap.sh` installs C build tools when absent, fetches model weights before serving,
retries downloads, disables Hugging Face Xet transfers and provides a ModelScope fallback.
These are setup provisions, not evidence that the full pipeline has completed successfully.

Generated configurations use `kind: local` and `budget_usd: 0` for inference token accounting.
GPU rental, storage and electricity are separate expenses. No measured end-to-end runtime or
cost is reported for this experiment.

The instance-destruction watchdog is opt-in with `ENABLE_WATCHDOG=1`, a configured `vastai`
CLI and an explicit `VAST_INSTANCE_ID`. It is cancelled when the script exits. Verify instance
termination through the provider, since cleanup of local processes does not end rental billing.
