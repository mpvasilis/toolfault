#!/usr/bin/env bash
# Unattended setup and run, from a bare rented GPU box to a finished comparison.
#
#   curl -sL https://raw.githubusercontent.com/mpvasilis/toolfault/main/posttrain/bootstrap.sh | bash
#
# Every workaround here was earned on a real rental, and each one cost an attempt:
#
#   * Hugging Face rate limits the shared datacenter address a rented box sits behind, so the
#     model is fetched separately with retries rather than inside the serving path.
#   * The HF xet storage backend fails on some hosts with a transport error rather than a
#     retryable HTTP status, so HF_HUB_DISABLE_XET is set. ModelScope is the fallback.
#   * SSH to a busy host drops on rapid successive connections, so this runs detached under
#     setsid and writes one log rather than being driven interactively.
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/mpvasilis/toolfault}"
WORK="${WORK:-/workspace}"
MODEL_DIR="${MODEL_DIR:-$WORK/models/qwen3-4b}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-4B-Instruct-2507}"
LOG="$WORK/bootstrap.log"

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_ROOT_USER_ACTION=ignore
export TOKENIZERS_PARALLELISM=false

exec > >(tee -a "$LOG") 2>&1
echo "=== bootstrap starting $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== clone ==="
cd "$WORK"
rm -rf toolfault
git clone --depth 1 "$REPO_URL" toolfault
cd "$WORK/toolfault"

echo "=== install ==="
pip install -q -U "transformers>=4.57" "trl>=0.24" "peft>=0.17" "datasets>=3.0" \
  "bitsandbytes>=0.45" "accelerate>=1.0" httpx pyyaml 2>&1 | tail -3
pip install -q vllm 2>&1 | tail -3
pip install -q -e . 2>&1 | tail -2
python - <<'PY'
import torch, transformers, trl, peft, vllm
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__)
print("vllm", vllm.__version__)
PY

echo "=== fetch model ==="
mkdir -p "$MODEL_DIR"
weights_present() { [ "$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)" -ge 1 ]; }
for attempt in $(seq 1 12); do
  weights_present && break
  echo "[fetch] attempt $attempt"
  python - <<PY
from huggingface_hub import snapshot_download
try:
    snapshot_download("$MODEL_REPO", local_dir="$MODEL_DIR", max_workers=2,
                      allow_patterns=["*.json", "*.txt", "*.safetensors"])
    print("hf ok")
except Exception as exc:
    print("hf failed:", type(exc).__name__, str(exc)[:140].replace("\n", " "))
PY
  sleep 20
done
if ! weights_present; then
  echo "[fetch] falling back to modelscope"
  pip install -q modelscope 2>&1 | tail -1
  python - <<PY
from modelscope import snapshot_download
print("modelscope", snapshot_download("$MODEL_REPO", local_dir="$MODEL_DIR"))
PY
fi
weights_present || { echo "FATAL: no model weights"; exit 1; }
du -sh "$MODEL_DIR"

echo "=== run the experiment ==="
export MODEL="$MODEL_DIR"
export TASKS_PER_TYPE="${TASKS_PER_TYPE:-16}"
export ENABLE_WATCHDOG=0
bash posttrain/run_on_gpu.sh

echo "=== bootstrap done $(date -u +%FT%TZ) ==="
