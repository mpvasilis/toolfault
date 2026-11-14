#!/usr/bin/env bash
# The post-training experiment in five stages, for a box started from the official vLLM image.
# This is the script behind results/posttrain-20261114.
#
# Serving and training live in separate Python environments. vllm/vllm-openai ships a vLLM
# built against its own torch and CUDA, so serving uses the image as shipped. Training gets a
# venv of its own. Installing both into one interpreter is what replaced torch underneath vLLM
# on the earlier rentals. On the 2026-11-14 run the image carried torch 2.14.0 on CUDA 13.0,
# and the training stack resolved torch 2.15.0 on CUDA 13.2.
#
#   bash run_staged.sh setup     clone, build the training venv, fetch the model
#   bash run_staged.sh train     rebuild the dataset, check digests and template, train
#   bash run_staged.sh base      serve the base model, run the benchmark
#   bash run_staged.sh tuned     serve base plus adapter, run the benchmark
#   bash run_staged.sh compare   paired comparison of the two arms
#
# The script is all the box needs to start with. setup clones the repository into
# $WORK/toolfault and the later stages run from that clone. Each stage writes
# $WORK/logs/<stage>.ok or <stage>.fail, so a detached run can be polled over SSH without
# holding a connection open.
#
# RUN_PREFIX defaults to the UTC date. Set it by hand if the stages may straddle midnight,
# otherwise compare looks for a base run under yesterday's name.
set -euo pipefail

STAGE="${1:?usage: run_staged.sh setup|train|base|tuned|compare}"
WORK="${WORK:-/workspace}"
REPO="$WORK/toolfault"
VENV="$WORK/train-venv"
MODEL_DIR="$WORK/models/qwen3-4b"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-4B-Instruct-2507}"
REPO_URL="${REPO_URL:-https://github.com/mpvasilis/toolfault}"
RUN_PREFIX="${RUN_PREFIX:-posttrain-$(date -u +%Y%m%d)}"
ADAPTER_DIR="${ADAPTER_DIR:-posttrain/out/adapter}"
TASKS_PER_TYPE="${TASKS_PER_TYPE:-16}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
PY="$VENV/bin/python"

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export PIP_ROOT_USER_ACTION=ignore
export LOCAL_API_KEY="${LOCAL_API_KEY:-not-used-by-vllm}"

mkdir -p "$WORK/logs"
rm -f "$WORK/logs/$STAGE.ok" "$WORK/logs/$STAGE.fail"

stop_serving() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    unset VLLM_PID
  fi
}
finish() {
  local code=$?
  stop_serving
  if [[ $code -eq 0 ]]; then touch "$WORK/logs/$STAGE.ok"; else touch "$WORK/logs/$STAGE.fail"; fi
  echo "=== stage $STAGE exit $code at $(date -u +%FT%TZ) ==="
}
trap finish EXIT

echo "=== stage $STAGE start $(date -u +%FT%TZ) ==="

serve() {
  local name="$1"; shift
  echo "=== serving $name ==="
  # The image's own vllm, not the training venv. --enforce-eager for the reason given in
  # run_on_gpu.sh: a 4B model answering a few hundred short episodes is not throughput bound.
  vllm serve "$MODEL_DIR" \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    "$@" > "$WORK/logs/vllm-$name.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
      echo "[serve] $name up"
      curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 600; echo
      return 0
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[serve] $name exited early; last log lines:" >&2
      tail -n 60 "$WORK/logs/vllm-$name.log" >&2
      return 1
    fi
    sleep 5
  done
  echo "[serve] $name did not come up; last log lines:" >&2
  tail -n 60 "$WORK/logs/vllm-$name.log" >&2
  return 1
}

write_config() {
  local run_id="$1" model_arg="$2"
  cat > "configs/generated-$run_id.yaml" <<YAML
run_id: $run_id
world_seed: 7
task_seed: 11
tasks_per_type: $TASKS_PER_TYPE
repeats: 1
max_tool_calls: 18
budget_usd: 0   # self-hosted inference is not metered
workers: 8
save_traces: failures
trace_sample: 0.2
out_dir: results
agents:
  - kind: local
    model: $model_arg
    base_url: http://127.0.0.1:$PORT/v1
    api_key_env: LOCAL_API_KEY
    label: $run_id
YAML
}

case "$STAGE" in
setup)
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  df -h "$WORK" | tail -1
  if ! command -v git >/dev/null || ! command -v cc >/dev/null || ! command -v curl >/dev/null; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git build-essential curl
  fi
  echo "--- serving environment, as shipped ---"
  command -v vllm
  python3 -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"

  [[ -d "$REPO/.git" ]] || git clone --depth 1 "$REPO_URL" "$REPO"
  echo "toolfault commit: $(git -C "$REPO" rev-parse HEAD)"

  echo "--- training venv ---"
  if [[ ! -x "$PY" ]]; then
    if command -v uv >/dev/null; then uv venv "$VENV" --python 3.12 || uv venv "$VENV"; else python3 -m venv "$VENV"; fi
  fi
  if command -v uv >/dev/null; then INSTALL=(uv pip install --python "$PY"); else INSTALL=("$VENV/bin/pip" install -q); fi
  "${INSTALL[@]}" torch "transformers>=4.57" "trl>=0.24" "peft>=0.17" "datasets>=3.0" \
    "bitsandbytes>=0.45" "accelerate>=1.0" httpx pyyaml
  "${INSTALL[@]}" -e "$REPO"
  "$PY" - <<'PYEOF'
import torch, transformers, trl, peft, bitsandbytes, datasets, accelerate
print("train env: torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__,
      "bitsandbytes", bitsandbytes.__version__, "datasets", datasets.__version__,
      "accelerate", accelerate.__version__)
assert torch.cuda.is_available(), "training venv cannot see the GPU"
PYEOF
  if command -v uv >/dev/null; then
    uv pip freeze --python "$PY" > "$WORK/logs/train-env.freeze.txt"
  else
    "$VENV/bin/pip" freeze > "$WORK/logs/train-env.freeze.txt"
  fi

  echo "--- model ---"
  mkdir -p "$MODEL_DIR"
  weights_present() { [[ "$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]]; }
  for attempt in $(seq 1 10); do
    weights_present && break
    echo "[fetch] attempt $attempt"
    "$PY" - <<PYEOF || true
from huggingface_hub import snapshot_download
try:
    snapshot_download("$MODEL_REPO", local_dir="$MODEL_DIR", max_workers=2,
                      allow_patterns=["*.json", "*.txt", "*.safetensors", "*.jinja"])
    print("hf ok")
except Exception as exc:
    print("hf failed:", type(exc).__name__, str(exc)[:160].replace("\n", " "))
PYEOF
    weights_present || sleep 20
  done
  weights_present || { echo "FATAL: no model weights"; exit 1; }
  du -sh "$MODEL_DIR"
  ls "$MODEL_DIR"
  ;;

train)
  cd "$REPO"
  cp posttrain/data/manifest.json "$WORK/logs/manifest.committed.json"
  echo "--- dataset ---"
  "$PY" posttrain/build_dataset.py
  "$PY" - <<PYEOF
import json
old = json.load(open("$WORK/logs/manifest.committed.json"))
new = json.load(open("posttrain/data/manifest.json"))
keys = ["train_examples", "val_examples", "train_digest", "val_digest", "train_conditions", "held_out_conditions"]
bad = [k for k in keys if old.get(k) != new.get(k)]
for k in keys:
    print(f"  {k}: committed={old.get(k)!r} rebuilt={new.get(k)!r}")
if bad:
    raise SystemExit(f"rebuilt dataset differs from the committed manifest on: {bad}")
print("rebuilt dataset matches the committed manifest")
PYEOF
  echo "--- template preflight ---"
  "$PY" posttrain/validate_template.py --model "$MODEL_DIR"
  "$PY" posttrain/validate_template.py --model "$MODEL_DIR" --data posttrain/data/val.jsonl
  echo "--- training ---"
  "$PY" posttrain/train_qlora.py --model "$MODEL_DIR" --output-dir "$ADAPTER_DIR"
  ls -la "$ADAPTER_DIR"
  ;;

base)
  cd "$REPO"
  serve base
  write_config "$RUN_PREFIX-base" "$MODEL_DIR"
  "$VENV/bin/toolfault" run --config "configs/generated-$RUN_PREFIX-base.yaml"
  ;;

tuned)
  cd "$REPO"
  serve tuned --enable-lora --lora-modules "tuned=$REPO/$ADAPTER_DIR" --max-lora-rank 32
  write_config "$RUN_PREFIX-tuned" "tuned"
  "$VENV/bin/toolfault" run --config "configs/generated-$RUN_PREFIX-tuned.yaml"
  ;;

compare)
  cd "$REPO"
  "$PY" posttrain/compare.py \
    --base "results/$RUN_PREFIX-base/episodes.jsonl" \
    --tuned "results/$RUN_PREFIX-tuned/episodes.jsonl" \
    --out "results/$RUN_PREFIX"
  ls -la "results/$RUN_PREFIX"
  ;;

*)
  echo "unknown stage: $STAGE" >&2
  exit 2
  ;;
esac
