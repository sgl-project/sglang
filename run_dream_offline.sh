set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Base-7B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"

export PYTHONPATH="python${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" test_dream_offline.py \
  --model-path "${MODEL_PATH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
