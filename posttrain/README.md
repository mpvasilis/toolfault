# Post-training: can a verification procedure be distilled, and does it generalise?

A reproducible fine-tuning experiment on the benchmark in this repo. It is pre-registered: the
splits are frozen and the predictions are written into the dataset manifest before training runs,
so the results can be read against what was expected rather than against what turned out
convenient.

**Status: built and pre-registered, not yet run.** The dataset builder works and its output is
described below from a real run. The training and evaluation scripts are written and not executed;
there are no results in this document, and there will not be until there are.

## The question

Not "does SFT improve the score". It does, and that would not be worth the GPU hour.

**Does training on some fault types teach a general verification procedure, or only the specific
recoveries that were demonstrated?**

That distinction is the whole value. A model that learned to cross-check `get_filings` against the
filed document has learned a lookup. A model that then cross-checks a source it was never shown
being cross-checked has learned something closer to skepticism. Only the second is worth deploying,
and only a held-out fault split can tell them apart.

## The teacher is the reference policy, not a frontier model

The `oracle` policy generates the training data. Three reasons, and the third is the real one:

1. It is better than any model at this specific task, resolving all 14 conditions at 1.0.
2. It is free, so the experiment costs a GPU hour rather than an API budget.
3. It is inspectable line by line. When a demonstration looks wrong it traces to a branch in
   `agents/scripted.py`, not to a sampling temperature.

## The splits, frozen before training

**Fault split.** Eight conditions train, six are held out. The held-out set is chosen so each cell
tests a different kind of transfer, not sampled at random:

| held out | family | what it tests | prediction |
|---|---|---|---|
| `rate_limit` | loud | near transfer: new error code, trained recovery | recovers |
| `schema_drift` | loud | a recovery mode absent from training (`tolerate`) | little or no gain |
| `stale_record` | silent | new tell, trained recovery (`second_source`) | partial gain |
| `contradiction` | silent | novel trigger for a trained action (escalation) | the interesting cell |
| `unit_shift` | silent | new magnitude tell, trained recovery | partial gain |
| `injected_instruction` | adversarial | an entire family never seen | little or no gain |

`auth_expired` stays in training deliberately. It is the only trained condition where escalating is
the correct action. Without it, `contradiction` would be asking the model to invent an action it has
never been shown, which tests nothing.

**World split.** Training traces come from world seeds 23, 29 and 31. The benchmark is world seed 7.
No company name, person id, officer name, filing amount or identifier is shared, so a model cannot
score by memorising the registry. `build_dataset.py` refuses to run if the two overlap.

**Validation split by task id, never by episode.** Splitting by episode would put the same company
in train and validation under two different faults, and the validation loss would be measuring
memorisation.

## The dataset, as actually built

```bash
python posttrain/build_dataset.py
```

2,736 demonstrations across three worlds, capped at 110 per condition, giving **777 train and 89
validation examples** with sha256 digests in `data/manifest.json`.

| condition | examples |
|---|---|
| `truncated_list` | 105 |
| `malformed_json` | 104 |
| `auth_expired` | 100 |
| `timeout` | 100 |
| `http_500` | 96 |
| `clean` | 93 |
| `wrong_entity` | 93 |
| `empty_result` | 86 |

The cap matters. Uncapped, the mix came out 88% loud faults, because every task supports every loud
fault while the silent ones need a particular state of the world. A model trained on that mix learns
to retry, which is the cheap half of the behaviour. Capping moves the silent share from 12% to 37%.
95 of the 777 demonstrations end in an escalation.

The JSONL is gitignored: it is 8 MB and regenerable from two seeds, and the manifest carries the
digests to verify a regenerated copy against.

## Training

QLoRA on one 24 GB card. Defaults target `Qwen/Qwen3-4B-Instruct-2507` in 4-bit nf4, rank 32,
alpha 64, 8192-token context, 2 epochs.

```bash
python posttrain/train_qlora.py --model Qwen/Qwen3-4B-Instruct-2507
```

Loss is applied to assistant turns only. This matters more here than in most SFT setups: roughly
70% of the tokens in one of these conversations are tool results, and training on them teaches the
model to predict registry payloads rather than to decide what to do about them. The script asks TRL
for the training variant of the model's chat template and **fails loudly if there is not one**
rather than silently training on everything; `--allow-full-loss` overrides that, and using it
changes what the experiment measures.

## Evaluation

Both arms are served with vLLM on the same box and evaluated by the same harness against the same
world seed and task seed. The only difference between them is the adapter.

```bash
bash posttrain/run_on_gpu.sh
```

That script serves the base model, evaluates it, builds the dataset, trains, serves the adapter,
evaluates again, and writes the comparison. It arms a watchdog that destroys the instance after
`HARD_STOP_MINUTES` (default 240) regardless of what state the run is in, because an idle rented
GPU still bills.

The adapter is served rather than merged. Merging into a 4-bit base dequantizes it to bf16, and the
arm being measured is the adapter, not a requantization of the base.

```bash
python posttrain/compare.py \
  --base results/posttrain-base/episodes.jsonl \
  --tuned results/posttrain-tuned/episodes.jsonl
```

`compare.py` splits every number by trained against held-out conditions, pairs on task id, runs a
sign-flip permutation test per condition, and BH-corrects across the grid. It prints the
pre-registered prediction next to each measured delta.

## How to read the outcome

- **Gain on trained conditions only.** The model memorised specific recoveries. Honest, publishable,
  and the least interesting result.
- **Gain on trained conditions and on `rate_limit` and `stale_record`.** Recoveries transfer to new
  triggers within a mode that was demonstrated. The useful result.
- **Gain on `contradiction`.** Escalation transferred to a trigger never shown. The strongest
  available result from this design.
- **Gain on `schema_drift` or `injected_instruction`.** Would contradict the prediction, since
  neither recovery mode appears in training at all. Would need explaining rather than celebrating.
- **A rise in the unsafe rate alongside a rise in accuracy.** A regression, and the reason
  `unsafe` is tracked separately from accuracy.

A null result is fully publishable and is planned for. "SFT on eight fault conditions moved
resolution on those conditions by X and did not move the six held-out conditions (permutation
p = Y)" is a better artifact than silence, and it is the honest shape of most transfer experiments.

## Cost

One 3090 or 4090 at roughly $0.11 to $0.35/hour on a spot marketplace, plus disk. Dataset
generation is CPU-only and free. Expect two to three hours end to end including both evaluation
arms, so a few euros.

Debug the whole stack on the cheapest card available before touching anything expensive.

## Environment, the parts that actually matter

Written down because each of these cost a rental attempt, and none of them is discoverable from
the error message alone.

**Match vLLM to the image's CUDA, or use the official vLLM image.** On a
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` base, an unpinned `pip install vllm` replaces the
image's `torch 2.7.1+cu128` with `torch 2.11.0+cu130`. vLLM then wants a CUDA 13 toolkit that the
image does not have, and the failure arrives from deep inside its memory-profiling pass as
`Failed to find C compiler`, then `libnvrtc.so.13: cannot open shared object file`, then
`Could not find nvcc`. None of those name the real problem. Use `vllm/vllm-openai` as the base, or
pin vLLM to a release whose torch matches the image.

**`-runtime` images have no C compiler.** Triton JIT-compiles attention kernels at first use and
needs `gcc`. `-devel` has it but is roughly twice the pull, and on a rented box the pull is the
slow part, so `bootstrap.sh` installs `build-essential` onto the smaller image. Note that
`--enforce-eager` does *not* remove this requirement.

**Fetch the model separately, with retries, and disable xet.** A rented box shares its address
with every other tenant, so unauthenticated Hugging Face requests are often already rate limited
(`0/500 requests remaining`). Separately, the xet storage backend fails on some hosts with a
transport error rather than a retryable HTTP status, so `HF_HUB_DISABLE_XET=1` is set and
ModelScope is the fallback. Pulling the snapshot before serving keeps this out of the serving path
entirely.

**Self-hosted inference is not metered, and the harness has to know that.** A generated local
config sets `budget_usd: 0`, and a spend ceiling of zero used to mean the very first call aborted
because spent already equalled the limit. A ceiling at or below zero now means unmetered, and
`price_for` returns zero for the `local` provider rather than falling through to the unknown-price
upper bound, which would have reported a free run as costing tens of dollars.

**Marketplace hosts fail.** Of four instances rented, one stalled indefinitely on a single Docker
layer and one went offline mid-run. Neither is a code problem. Budget an attempt or two, keep the
total spend small per attempt, and destroy instances explicitly rather than trusting an in-container
watchdog: nothing running inside the container can stop the billing.
