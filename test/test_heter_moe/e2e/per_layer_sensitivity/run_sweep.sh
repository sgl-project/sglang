#!/bin/bash
# Per-MoE-layer INT4 sensitivity sweep.
# For each layer L:
#   1. launch sglang with target_layers=[L], all 128 experts INT4-only
#   2. run WikiText-2 ppl via ppl_client
#   3. shut down server, record ppl_L
# Also runs a BF16 baseline (no --heter-precision-config).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/data/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}"
GPU="${GPU:-4}"
PORT="${PORT:-30210}"
NSAMPLES="${NSAMPLES:-32}"
SEQLEN="${SEQLEN:-2048}"
NUM_LAYERS="${NUM_LAYERS:-48}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results}"
CFG_DIR="${CFG_DIR:-${OUT_DIR}/configs}"
ENDPOINT="http://127.0.0.1:${PORT}"

mkdir -p "${OUT_DIR}" "${CFG_DIR}"

# Generate per-layer configs (idempotent).
conda run -n sglang python "${SCRIPT_DIR}/gen_configs.py" \
    --out_dir "${CFG_DIR}" --num_layers "${NUM_LAYERS}"

wait_ready() {
    local pid=$1
    for _ in $(seq 1 120); do
        if curl -sf "${ENDPOINT}/get_model_info" > /dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then return 1; fi
        sleep 5
    done
    return 1
}

kill_server() {
    local pid=$1
    # Kill the conda-run wrapper and its children.
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    # Kill anything still holding the port (sglang spawns subprocesses
    # that survive the parent kill).
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    # Wait for GPU memory to actually free.
    for _ in $(seq 1 12); do
        local used
        used=$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")
        if [[ "${used}" -lt 1000 ]]; then break; fi
        sleep 5
    done
}

launch_and_measure() {
    local tag=$1
    local extra_args=$2
    local out="${OUT_DIR}/${tag}.json"
    local log="${OUT_DIR}/${tag}.server.log"

    if [[ -f "${out}" ]]; then
        echo "[skip] ${tag} (already have ${out})"
        return 0
    fi

    echo "=== ${tag} ==="
    CUDA_VISIBLE_DEVICES="${GPU}" conda run -n sglang python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --port "${PORT}" --trust-remote-code \
        ${extra_args} > "${log}" 2>&1 &
    local pid=$!
    if ! wait_ready "$pid"; then
        echo "[FAIL] ${tag} server never became ready; see ${log}"
        kill_server "$pid"
        return 1
    fi

    conda run -n sglang python "${SCRIPT_DIR}/ppl_client.py" \
        --endpoint "${ENDPOINT}" \
        --tokenizer_path "${MODEL_PATH}" \
        --nsamples "${NSAMPLES}" --seqlen "${SEQLEN}" \
        --out "${out}"

    kill_server "$pid"
}

# --- 1. BF16 baseline ---
launch_and_measure "bf16_baseline" ""

# --- 2. Per-layer INT4 ---
for (( L=0; L<NUM_LAYERS; L++ )); do
    launch_and_measure "layer${L}" \
        "--heter-precision-config ${CFG_DIR}/heter_layer${L}.json"
done

# --- 3. Aggregate ---
conda run -n sglang python -c "
import glob, json, os
out_dir = '${OUT_DIR}'
results = {}
baseline_path = os.path.join(out_dir, 'bf16_baseline.json')
baseline = json.load(open(baseline_path))
ref_ppl = baseline['perplexity']
for p in sorted(glob.glob(os.path.join(out_dir, 'layer*.json'))):
    name = os.path.basename(p).replace('.json','')
    L = int(name.replace('layer',''))
    d = json.load(open(p))
    results[L] = {
        'perplexity': d['perplexity'],
        'ref_perplexity': ref_ppl,
        'ppl_increase': d['perplexity'] - ref_ppl,
        'ppl_increase_pct': (d['perplexity']/ref_ppl - 1) * 100,
        'tokens': d['tokens'],
    }
summary_path = os.path.join(out_dir, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump({'ref_perplexity': ref_ppl, 'per_layer': results}, f, indent=2)
print(f'wrote {summary_path} with {len(results)} layers')
"
