#!/usr/bin/env bash
# Shared evalscope client for the blog sweeps. Run AFTER a server script from
# gb300/ or b300/ is up on localhost:${PORT} (default 8002).
#
# Usage:
#   ./run_client.sh <model-path> <output-dir> <run-name>
#
#   <model-path>  model the server is serving; also selects the dataset
#                 tokenizer (nvidia/GLM-5.2-NVFP4 or nvidia/GLM-5.1-NVFP4;
#                 day-0 servers serve GLM-5.2, so pass nvidia/GLM-5.2-NVFP4)
#   <output-dir>  where results land (created if missing)
#   <run-name>    subdirectory name for this run (e.g. tp4, tep8)
#
# Example (one curve = one server config):
#   gb300/server_glm52_v0515_tp4.sh &            # terminal 1
#   ./run_client.sh nvidia/GLM-5.2-NVFP4 results/gb300/glm52_v0515 tp4
#
# What it does:
#   1. checks evalscope is installed (see evalscope-deps/README.md for the
#      one-time pinned install)
#   2. builds the OpenHands multi-turn dataset for <model-path> if missing
#      (74160/753 tokens first/subsequent turn, 13 turns, 128 conversations,
#      openscience padding) — first build takes ~10-20 min (streams two HF
#      datasets)
#   3. waits for the server's /health
#   4. runs the concurrency ladder c=1,2,4,8 in ONE evalscope invocation
#      (evalscope rotates dataset offsets between steps so every step replays
#      fresh conversations; OSL fixed at 220 via ignore_eos)
#
# Results: <output-dir>/<run-name>/parallel_*/benchmark_summary.json
set -euo pipefail

MODEL=${1:?usage: run_client.sh <model-path> <output-dir> <run-name>}
OUT=${2:?usage: run_client.sh <model-path> <output-dir> <run-name>}
NAME=${3:?usage: run_client.sh <model-path> <output-dir> <run-name>}
: "${PORT:=8002}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. evalscope present?
python3 - <<'PY' || { echo "evalscope missing — run evalscope-deps/scripts/install_evalscope_deps.sh once, then: PIP_NO_DEPS=1 pip install \"evalscope[all] @ git+https://github.com/modelscope/evalscope.git@acd09b44384d53174768bb1063f675420f76fae9\"" >&2; exit 1; }
import evalscope.perf.plugin.datasets.swe_smith  # noqa: F401  (the multi-turn plugin the sweep uses)
PY

# 2. dataset (per model tokenizer)
SLUG=$(echo "$MODEL" | tr '/' '-')
DATASET="$SCRIPT_DIR/datasets/openhand-${SLUG}.json"
if [ ! -s "$DATASET" ]; then
    mkdir -p "$SCRIPT_DIR/datasets"
    echo "=== building dataset for $MODEL (one-time, ~10-20 min) ==="
    python3 "$SCRIPT_DIR/build_openhands_padded_dataset.py" \
        --model "$MODEL" \
        --pad-source openscience \
        --first-turn-length 74160 \
        --subsequent-turn-length 753 \
        --num-turns 13 \
        --number 128 \
        --output-path "$DATASET"
else
    echo "reusing dataset $DATASET"
fi

# 3. wait for the server
echo "waiting for http://localhost:${PORT}/health ..."
for _ in $(seq 1 720); do
    curl -sf -m 3 "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
    sleep 5
done
curl -sf -m 3 "http://localhost:${PORT}/health" >/dev/null 2>&1 \
    || { echo "server never became healthy on port ${PORT}" >&2; exit 1; }

# 4. measured ladder (c = 1, 2, 4, 8)
mkdir -p "$OUT"
evalscope perf \
    --model "$MODEL" \
    --url "http://localhost:${PORT}/v1/chat/completions" \
    --api openai \
    --dataset swe_smith \
    --dataset-path "$DATASET" \
    --max-tokens 220 \
    --multi-turn \
    --number 4 8 8 16 \
    --parallel 1 2 4 8 \
    --extra-args '{"ignore_eos": true}' \
    --name "$NAME" \
    --outputs-dir "$OUT" \
    --no-timestamp

echo "done -> $OUT/$NAME/parallel_*/benchmark_summary.json"
