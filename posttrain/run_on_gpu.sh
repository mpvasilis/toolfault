#!/usr/bin/env bash
# The whole post-training experiment, start to finish, on one rented GPU box.
#
#   base model  -> serve -> evaluate on the benchmark
#   train QLoRA -> serve -> evaluate on the same benchmark, same seeds
#
# Both arms are evaluated by the same harness against the same world seed and the same task
# seed, so the only difference between them is the adapter. Nothing about the benchmark is
# regenerated between arms.
#
# Usage on the box:
#   bash posttrain/run_on_gpu.sh                    # full run
#   SKIP_TRAIN=1 bash posttrain/run_on_gpu.sh       # re-evaluate an adapter already trained
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
ADAPTER_DIR="${ADAPTER_DIR:-posttrain/out/adapter}"
RESULTS_DIR="${RESULTS_DIR:-results}"
TASKS_PER_TYPE="${TASKS_PER_TYPE:-16}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"

# An idle GPU still bills. This kills the box after HARD_STOP_MINUTES no matter what state
# the run is in, which is the difference between a forgotten instance costing two euros and
# costing a hundred.
HARD_STOP_MINUTES="${HARD_STOP_MINUTES:-240}"
if [[ "${ENABLE_WATCHDOG:-1}" == "1" ]]; then
  ( sleep $(( HARD_STOP_MINUTES * 60 )); echo "[watchdog] hard stop after ${HARD_STOP_MINUTES}m"; \
    command -v vastai >/dev/null && vastai destroy instance "${VAST_INSTANCE_ID:-0}" || shutdown -h now ) &
  echo "[watchdog] armed for ${HARD_STOP_MINUTES} minutes"
fi

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "=== install ==="
pip install -q -U "transformers>=4.57" "trl>=0.24" "peft>=0.17" "datasets>=3.0" \
  "bitsandbytes>=0.45" "accelerate>=1.0" "vllm>=0.11" >/dev/null
pip install -q -e . >/dev/null

serve() {
  local name="$1"; shift
  echo "=== serving ${name} ==="
  # Adapter served rather than merged. Merging a 4-bit base dequantizes it to bf16, and the
  # arm being measured is the adapter, not a requantization of the base.
  vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    "$@" > "vllm-${name}.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
      echo "[serve] ${name} up"
      return 0
    fi
    sleep 5
  done
  echo "[serve] ${name} did not come up; last log lines:" >&2
  tail -n 40 "vllm-${name}.log" >&2
  return 1
}

stop_serving() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 5
  fi
}
trap stop_serving EXIT

write_config() {
  local run_id="$1" model_arg="$2"
  cat > "configs/generated-${run_id}.yaml" <<YAML
run_id: ${run_id}
world_seed: 7
task_seed: 11
tasks_per_type: ${TASKS_PER_TYPE}
repeats: 1
max_tool_calls: 18
budget_usd: 0
workers: 8
save_traces: failures
trace_sample: 0.2
out_dir: ${RESULTS_DIR}
agents:
  - kind: local
    model: ${model_arg}
    base_url: http://127.0.0.1:${PORT}/v1
    api_key_env: LOCAL_API_KEY
    label: ${run_id}
YAML
}

export LOCAL_API_KEY="${LOCAL_API_KEY:-not-used-by-vllm}"

# ---------------------------------------------------------------- base arm
serve base
write_config "posttrain-base" "$MODEL"
toolfault run --config "configs/generated-posttrain-base.yaml"
stop_serving

# ---------------------------------------------------------------- training
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "=== building the dataset ==="
  python posttrain/build_dataset.py
  echo "=== training ==="
  python posttrain/train_qlora.py --model "$MODEL" --output-dir "$ADAPTER_DIR"
fi

# ---------------------------------------------------------------- tuned arm
serve tuned --enable-lora --lora-modules "tuned=${ADAPTER_DIR}" --max-lora-rank 32
write_config "posttrain-tuned" "tuned"
toolfault run --config "configs/generated-posttrain-tuned.yaml"
stop_serving

echo "=== comparison ==="
python posttrain/compare.py \
  --base "${RESULTS_DIR}/posttrain-base/episodes.jsonl" \
  --tuned "${RESULTS_DIR}/posttrain-tuned/episodes.jsonl" \
  --out "${RESULTS_DIR}/posttrain"
echo "done. results under ${RESULTS_DIR}/posttrain"
