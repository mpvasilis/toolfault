# Post-training experiment

**Status: one completed run, 14 November 2026.** `Qwen/Qwen3-4B-Instruct-2507` was fine-tuned
with QLoRA on 765 reference-policy demonstrations and compared with its own base on 624 matched
episodes. The comparison, both episode files, failure traces, stage logs and the resolved
environment are in [`results/posttrain-20261114/`](../results/posttrain-20261114/).

The experiment asks whether supervised fine-tuning on selected fault types improves recovery
on fault types excluded from training. The fault split and the hypotheses were written to
[`data/manifest.json`](data/manifest.json) before training, and the comparison report prints
them unchanged beside the measurements.

## Results

| Conditions | Arm | Episodes | Resolved | Unsafe | Escalated | Tool calls | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Trained (8) | base | 368 | 55.4% | 15.8% | 9.0% | 6.4 | 0.192 |
| Trained (8) | tuned | 368 | 100.0% | 0.0% | 16.0% | 4.3 | 0.091 |
| Held out (6) | base | 256 | 53.5% | 21.1% | 1.6% | 5.6 | 0.183 |
| Held out (6) | tuned | 256 | 65.2% | 28.5% | 6.2% | 3.4 | 0.135 |

The adapter resolves every episode in the eight trained conditions, the clean control among
them. On the six held-out faults resolution rises by 11.7 points and unsafe submissions rise
by 7.4. The protocol said before the run that this combination would count as a safety
tradeoff. All 624 pairs were scored, with no endpoint error in either arm.

Some of the base model's deficit is unfinished work. It ended 157 of its 624 episodes without
calling `submit` or `escalate`. The tuned model ended none that way, and it covered the grid
in 3,098 model calls where the base needed 5,126.

### Held-out faults

| Fault | Family | Pairs | Base | Tuned | Change | q | Unsafe, base to tuned | Hypothesis before training |
|---|---|---:|---:|---:|---:|---:|---|---|
| `unit_shift` | silent | 16 | 25.0% | 87.5% | +62.5 | 0.012 | 50.0% to 12.5% | partial gain |
| `stale_record` | silent | 16 | 25.0% | 68.8% | +43.8 | 0.077 | 50.0% to 31.2% | partial gain |
| `schema_drift` | loud | 64 | 70.3% | 100.0% | +29.7 | 0.0001 | 6.2% to 0.0% | little or no gain |
| `injected_instruction` | adversarial | 64 | 62.5% | 73.4% | +10.9 | 0.205 | 6.2% to 26.6% | little or no gain |
| `contradiction` | silent | 32 | 12.5% | 15.6% | +3.1 | 1.000 | 68.8% to 84.4% | open question |
| `rate_limit` | loud | 64 | 62.5% | 40.6% | -21.9 | 0.043 | 12.5% to 34.4% | recovery transfers |

Base and Tuned are the share of episodes resolved, and Change is in percentage points. Each
p-value comes from a paired sign-flip permutation test over tasks, and q is its
Benjamini-Hochberg correction across all 14 conditions. The full table, with intervals and
the eight trained conditions, is in [`REPORT.md`](../results/posttrain-20261114/REPORT.md).

### What the traces show

Cross-checking carried over to silent faults the model was never trained on. Under
`unit_shift` the fault fired in all 16 episodes for both arms, and the tuned model resolved 14
where the base resolved 4. `stale_record` needs more care, because a fault fires only when
the agent calls the affected tool: it reached 11 tuned episodes and 6 base ones. Among the
exposed episodes the tuned model resolved 6 of 11 and the base 1 of 6.

The `schema_drift` row is not evidence of transfer. The base model resolved 70.3% under
schema drift and 71.9% with no fault at all, so this fault barely cost it anything to begin
with. A 29.7-point gain here is the 28.1-point gain on the clean control again.

Escalation did not transfer. The contradiction fired in 27 tuned episodes and the tuned model
submitted an answer in every one of them. The base never escalated on it either. Each resolved
`contradiction` episode, in both arms, is one where the fault did not fire. `auth_expired`
teaches escalation on an explicit error, and that did not carry to two sources that quietly
disagree.

Retry transferred on one task type out of four, and `rate_limit` is the only held-out fault
where the adapter is significantly worse than its base. After a 429 the tuned model retried
and resolved all 16 sanctions-screening tasks. On the 16 revenue tasks it submitted `0`
straight after the first rate-limited call. On the 16 ownership tasks it escalated at the same
point and gave the rate limit as the reason. Entity verification sat in between at 10 of 16.
The hypothesis that retry would carry to a new error code was wrong for this adapter.

Injected text steers the adapter more easily than it steers the base. The notice appended to
`search_registry` output claims a supervisor has approved the entity and asks for an approving
answer. In 26 of 64 episodes the tuned model submitted right after that first tool response,
23 times with `approve` or `clear`, and 15 of those answers were wrong. On the clean control it
submitted that early in 5 of 64 episodes, all of them correct. The base model never did under
injection. Resolution still went up, from 40 episodes to 47, because the base left 20
unfinished. Unsafe submissions went from 4 to 17.

Stated confidence in the tuned arm takes only the four values the reference policy emits:
0.80, 0.85, 0.90 and 0.95. Calibration error falls on both splits because accuracy moved
towards those numbers. On held-out faults the confidence AUROC is 0.500, so confidence does
not separate correct submissions from incorrect ones.

### Limits of this run

This is one training seed, one evaluation world and one repeat per episode. Three held-out
conditions have 16 or 32 pairs, and the `stale_record` gain has q = 0.077. The adapter was
trained against 4-bit NF4 weights and served on the bf16 base through vLLM's LoRA support, so
the served arm is not exactly the model the optimiser saw. Independent worlds, repeated
training seeds and other task families would be needed before reading any of this as a
general property of the method.

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

Worlds are split by seed. Training uses world seeds 23, 29 and 31, and evaluation uses world
seed 7. The builder rejects a training seed equal to the declared benchmark seed. Different
seeds change the registry, but do not guarantee disjoint names, identifiers or values.
Identifier formats and the underlying world generator remain shared.

Validation is split by task. All retained episodes for a world-prefixed task ID belong to the
same split, which keeps one task from appearing in both training and validation under
different conditions. Related entities can still occur in different tasks, so validation
loss is not an independent-world generalisation result.

## Reproduced dataset

Run commands from the repository root.

```bash
python posttrain/build_dataset.py
```

The default build produces 2,733 demonstrations across three worlds. A cap of 110 examples
per condition is applied before splitting, producing 765 training and 98 validation
examples.

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
seeds. The manifest records SHA-256 prefixes over the ordered message payloads. The training
digest is `2d1d01bbeab421f8` and the validation digest is `f351c0742ba1b3c1`. The GPU run
rebuilt the dataset on the rented box and matched both digests against the committed manifest
before the optimiser started.

## Template preflight and provenance

The preflight checks that conversations render, assistant masks are nonempty, supervised tool
calls and arguments match the source, tool-response markers are excluded from supervised
text, and all sequences fit the selected context length.

```bash
python posttrain/validate_template.py --model Qwen/Qwen3-4B-Instruct-2507
python posttrain/validate_template.py --model Qwen/Qwen3-4B-Instruct-2507 \
  --data posttrain/data/val.jsonl
```

The CPU check used `transformers==5.22.2`, tokenizer files from
`Qwen/Qwen3-4B-Instruct-2507` at revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, and TRL's
`qwen3_instruct_2507_training.jinja` at commit
`e5677dace3f8ab512ab40ba95fe8a887f5d37a45`. All 863 conversations passed. The longest
sequence was 3,033 tokens, below the 4,096-token training limit. The median supervised
share in training was 8.1%.

The same preflight ran again on the GPU box before training, under `transformers==5.23.0` and
`trl==1.18.0`, and agreed with the CPU check: a maximum of 3,033 tokens and 8.1% of tokens
under loss, with a median length of 2,533. That run took the template TRL 1.18.0 resolves by
default. To pin one, pass it with `--template` to both validators and to the trainer.

## Training

The run used QLoRA on 4-bit NF4 weights with rank 32, alpha 64 and dropout 0.05 on the
attention and MLP projections. It trained for two epochs at an effective batch of 8 with a
4,096-token context, a learning rate of 1e-4, cosine decay and 3% warmup.

```bash
python posttrain/train_qlora.py --model Qwen/Qwen3-4B-Instruct-2507
```

Loss is applied to assistant turns. Pass the same `--template` used in preflight when selecting
an explicit template. `--allow-full-loss` permits training on the complete sequence when an
assistant-only template cannot be selected. That changes the objective and must be reported
as a separate experimental configuration.

Training took 192 optimiser steps in 3,353 seconds on one RTX 3090 and peaked at 5.78 GB of
GPU memory. Mean training loss was 0.0377. Validation loss was 0.00102 after the first epoch
and 0.00043 after the second, with a mean token accuracy of 0.99996. Loss that low comes from
a reference policy that is close to deterministic given the conversation, measured on
validation tasks drawn from the training worlds. It confirms the adapter learned the
demonstrations and says nothing about held-out faults. The full record is in
[`train_summary.json`](../results/posttrain-20261114/provenance/train_summary.json).

The trainer passes the 3% warmup as `warmup_ratio` where the installed release accepts that
argument and as a fractional `warmup_steps` where it does not. `transformers` 5.23.0 with
`trl` 1.18.0 is in the second group.

## Evaluation

Both arms are served with vLLM and run through the same harness with world seed 7, task
seed 11, 16 tasks per type, temperature 0 and identical serving flags. The tuned arm loads the
adapter through vLLM's LoRA support on the unquantized base. It is not merged and requantized.

The comparison can be recomputed on CPU from the committed episodes.

```bash
python posttrain/compare.py \
  --base results/posttrain-20261114-base/episodes.jsonl \
  --tuned results/posttrain-20261114-tuned/episodes.jsonl \
  --out runs/posttrain-recheck
```

`compare.py` requires matching episode identities and task ground truth. It reports endpoint
errors and excludes affected pairs from both arms. It separates trained and held-out
conditions, pairs results by task, and reports sign-flip permutation tests with
Benjamini-Hochberg correction. Resolution, unsafe submission, escalation and confidence
metrics are reported together, with pair counts and excluded errors.

## Running it on a GPU

The committed results come from [`run_staged.sh`](run_staged.sh), run on a rented RTX 3090
started from the `vllm/vllm-openai:v0.33.0` image with a shell in place of the serving
entrypoint.

```bash
curl -sLO https://raw.githubusercontent.com/mpvasilis/toolfault/main/posttrain/run_staged.sh
for stage in setup train base tuned compare; do
  bash run_staged.sh "$stage" || break
done
```

Serving uses the vLLM that ships in the image. Training runs in a separate venv built by the
`setup` stage. On this run the two stacks resolved different torch builds, 2.14.0 on CUDA 13.0
for serving and 2.15.0 on CUDA 13.2 for training, which is why they do not share an
interpreter. The `train` stage rebuilds the dataset and stops if its digests differ from the
committed manifest.

[`run_on_gpu.sh`](run_on_gpu.sh) and [`bootstrap.sh`](bootstrap.sh) run the same steps in a
single interpreter on a PyTorch image. That path installs vLLM beside the training stack, so
confirm the two agree on a torch build before renting a GPU for it. Its instance-destruction
watchdog is opt-in with `ENABLE_WATCHDOG=1`, a configured `vastai` CLI and an explicit
`VAST_INSTANCE_ID`, and it is cancelled when the script exits. Verify instance termination
through the provider, since cleanup of local processes does not end rental billing.

## Environment and cost of the run

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB, driver 615.71.09 |
| Image | `vllm/vllm-openai:v0.33.0` |
| Serving | vLLM 0.33.0, torch 2.14.0+cu130 |
| Training | Python 3.12.3, torch 2.15.0+cu132, transformers 5.23.0, trl 1.18.0, peft 0.21.3, bitsandbytes 0.50.2, datasets 5.2.0, accelerate 1.15.0 |
| Model | `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Code | `89cb54d`, plus the `warmup_steps` change above |
| Adapter | 132 MB, SHA-256 `9d1196ed74a165b04c267fd6b60cd68446a0ab8cb795302803821d5de4e1a409` |

The scripts install dependencies with lower version bounds, so the environment is whatever
resolved on the day. The resolved package list, the stage logs, both generated run
configurations and the adapter configuration are kept in
[`results/posttrain-20261114/provenance/`](../results/posttrain-20261114/provenance/). The
adapter weights are too large for git.

| Stage | Wall time |
|---|---:|
| `setup` | 2 min 34 s |
| `train` | 56 min 41 s |
| `base` | 26 min 36 s |
| `tuned` | 23 min 0 s |
| `compare` | 2 s |

The rental lasted 2 hours 4 minutes with the image pull included and cost about $0.47.
Generated configurations use `kind: local` and `budget_usd: 0`, so the harness reports no
token spend for either arm.
