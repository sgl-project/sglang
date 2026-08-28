set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
# MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Base-7B}"
MODEL_PATH="${MODEL_PATH:-/root/.cache/modelscope/models/Dream-org--Dream-v0-Base-7B/snapshots/master}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export HF_ENDPOINT=https://hf-mirror.com
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"

export PYTHONPATH="python${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  test_dream_offline.py
  --model-path "${MODEL_PATH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
)
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  args+=(--attention-backend "${ATTENTION_BACKEND}")
fi

exec "${PYTHON_BIN}" "${args[@]}"
