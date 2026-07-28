#!/bin/bash
# Validation script for GLM-5.2 EAGLE-3 TP4 × PP2 speculative decoding.
#
# Usage:
#   SERVER_URL=http://127.0.0.1:30000 bash scripts/validate_eagle3_pp2.sh
#
# The server must already be running. This script sends deterministic
# temperature=0 requests and checks HTTP status + token output.
#
# To launch the server, use the commands printed by this script with
# --print-launch-only.

set -euo pipefail

SERVER_URL="${SERVER_URL:-http://127.0.0.1:30000}"
TARGET_MODEL="${GLM52_MODEL:-zai-org/GLM-5.2-FP8}"
DRAFT_MODEL="${GLM52_EAGLE3_DRAFT:-}"
NUM_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [ -z "$DRAFT_MODEL" ]; then
    echo "ERROR: GLM52_EAGLE3_DRAFT must be set to a real EAGLE3 draft checkpoint."
    echo "       It must NOT equal the target model path."
    exit 1
fi

if [ "$TARGET_MODEL" = "$DRAFT_MODEL" ]; then
    echo "ERROR: Draft model path equals target model path."
    echo "       A separate trained EAGLE3 draft checkpoint is required."
    exit 1
fi

# --- Print launch commands if requested ---
if [ "${1:-}" = "--print-launch-only" ]; then
    echo "# Eager startup (first pass):"
    echo "SGLANG_ENABLE_PP_SPEC=1 \\"
    echo "SGLANG_GLM52_PP_DEBUG=1 \\"
    echo "SGLANG_PP_LAYER_PARTITION=40,38 \\"
    echo "CUDA_VISIBLE_DEVICES=$NUM_GPUS \\"
    echo "python -m sglang.launch_server \\"
    echo "  --model-path \"$TARGET_MODEL\" \\"
    echo "  --speculative-algorithm EAGLE3 \\"
    echo "  --speculative-draft-model-path \"$DRAFT_MODEL\" \\"
    echo "  --speculative-num-steps 4 \\"
    echo "  --speculative-eagle-topk 1 \\"
    echo "  --speculative-num-draft-tokens 5 \\"
    echo "  --tp-size 4 --pp-size 2 \\"
    echo "  --disable-overlap-schedule \\"
    echo "  --disable-cuda-graph \\"
    echo "  --trust-remote-code \\"
    echo "  --log-level debug \\"
    echo "  --host 0.0.0.0 --port 30000"
    exit 0
fi

# --- Send validation requests ---
echo "Sending validation requests to $SERVER_URL"

PROMPTS=(
    "Hello, how are you?"
    "Explain quantum computing in one sentence."
    "What is the capital of France?"
)

FAIL=0
for prompt in "${PROMPTS[@]}"; do
    echo "---"
    echo "Prompt: $prompt"

    RESPONSE=$(curl -s -w "\n%{http_code}\n%{time_total}" \
        -X POST "${SERVER_URL}/generate" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"${prompt}\", \"sampling_params\": {\"temperature\": 0, \"max_new_tokens\": 64}}")

    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    LATENCY=$(echo "$RESPONSE" | tail -2 | head -1)
    BODY=$(echo "$RESPONSE" | head -n -2)

    echo "HTTP status: $HTTP_CODE"
    echo "Latency: ${LATENCY}s"

    if [ "$HTTP_CODE" != "200" ]; then
        echo "FAIL: HTTP $HTTP_CODE"
        FAIL=1
        continue
    fi

    # Extract text from JSON response (basic extraction)
    TEXT=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin).get('text',''))" 2>/dev/null || echo "")
    if [ -z "$TEXT" ]; then
        echo "FAIL: empty response"
        FAIL=1
    else
        echo "Response: ${TEXT:0:200}..."
    fi
done

echo "---"
if [ "$FAIL" -eq 0 ]; then
    echo "All validation requests succeeded."
else
    echo "VALIDATION FAILED"
    exit 1
fi
