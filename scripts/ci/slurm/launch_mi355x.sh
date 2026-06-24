#!/usr/bin/env bash
# Launch a 2-node 1P1D disaggregation benchmark on the AMD MI355X `amd-sglang`
# Slurm cluster, then emit per-concurrency result JSONs that
# scripts/ci/slurm/process_result.py aggregates.
#
# salloc's two nodes and runs the Docker harness: prefill server on node A,
# decode server on node B, a standalone load balancer on the prefill node, then
# an sglang.bench_serving concurrency sweep over MORI.
#
# Required environment variables (set by the GitHub Actions workflow):
#   MODEL              - HuggingFace model id (table label / served model)
#   MODEL_PREFIX       - short prefix, e.g. dsv4flash
#   PRECISION          - fp8 / fp4
#   ISL, OSL           - input / output sequence lengths for the sweep
#   CONFIG_FILE        - path to the recipe YAML (relative to repo root)
#   RESULT_FILENAME    - prefix for the emitted result JSONs
#   MATRIX_CONFIG_NAME - matrix entry name (used in filenames/tags)
#   GITHUB_WORKSPACE   - set by GitHub Actions; where result JSONs are written
# Optional:
#   MODEL_PATH         - local snapshot dir (preferred over downloading MODEL)
#   SLURM_PARTITION    - default: amd-sglang
#   SLURM_NODELIST     - optional explicit 2-node pin (else scheduler chooses)
#   TIME_LIMIT         - salloc time limit, default 01:00:00

set -euo pipefail
set -x

: "${MODEL_PREFIX:?}"
: "${PRECISION:?}"
: "${ISL:?}"
: "${OSL:?}"
: "${CONFIG_FILE:?}"
: "${RESULT_FILENAME:?}"
: "${MATRIX_CONFIG_NAME:?}"
: "${GITHUB_WORKSPACE:?}"

SLURM_PARTITION="${SLURM_PARTITION:-amd-sglang}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-}}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: set MODEL_PATH (local snapshot) or MODEL" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse the recipe (runtime + bench + topology) into shell vars.
# ---------------------------------------------------------------------------
# Ensure PyYAML is available to the host python used for parsing.
python3 -c 'import yaml' 2>/dev/null || pip install pyyaml -q 2>/dev/null \
    || pip install --user pyyaml -q 2>/dev/null || true

# Emit KEY=value lines and eval them (robust single-level command substitution;
# avoids a nested read<<EOF/$(<<PY) heredoc that misparses on some shells).
RECIPE_VARS="$(python3 - "$CONFIG_FILE" <<'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1]))
rt = r["runtime"]; b = r["backend"]["sglang_config"]; bn = r["bench"]
def emit(k, v): print(f"{k}={v}")
emit("IMAGE", rt["image"])
emit("ATTN", rt["attention_backend"])
emit("IB", rt["ib_devices"])
emit("PPORT", rt["prefill_port"])
emit("DPORT", rt["decode_port"])
emit("PBOOT", rt["prefill_bootstrap_port"])
emit("DBOOT", rt["decode_bootstrap_port"])
emit("LBPORT", rt["lb_port"])
emit("MEMFRAC", rt["mem_fraction_static"])
emit("PAGE", rt["page_size"])
emit("MAXREQ", rt["max_running_requests"])
emit("CHUNK", rt["chunked_prefill_size"])
emit("SWA", rt["swa_full_tokens_ratio"])
emit("PTP", b["prefill"]["tensor-parallel-size"])
emit("DTP", b["decode"]["tensor-parallel-size"])
emit("CONCS", ",".join(str(c) for c in bn["concurrencies"]))
emit("NPF", bn["num_prompts_factor"])
emit("RRR", bn["random_range_ratio"])
PY
)"
if [[ -z "$RECIPE_VARS" ]]; then
    echo "ERROR: failed to parse recipe $CONFIG_FILE (empty output from python3/yaml)" >&2
    exit 1
fi
eval "$RECIPE_VARS"
# Optional image override from workflow_dispatch input.
if [[ -n "${IMAGE_OVERRIDE:-}" ]]; then
    IMAGE="$IMAGE_OVERRIDE"
fi
echo "recipe: image=$IMAGE attn=$ATTN ib=$IB ptp=$PTP dtp=$DTP concs=$CONCS isl=$ISL osl=$OSL"

# ---------------------------------------------------------------------------
# Shared NFS scratch (visible to login node + compute nodes). Raw bench output
# lands here; the launcher normalizes it into GITHUB_WORKSPACE afterwards.
# ---------------------------------------------------------------------------
WORKDIR="$HOME/.mi355x_ci/${MATRIX_CONFIG_NAME}"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/standalone_lb.py" "$WORKDIR/standalone_lb.py"

# DSV4-Flash-FP8 load-bearing env (see test/registered/amd/test_deepseek_v4_flash_fp8.py).
DSV4_ENV=(
  -e SGLANG_DEFAULT_THINKING=1 -e SGLANG_DSV4_REASONING_EFFORT=max
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=false -e SGLANG_USE_AITER=1
  -e SGLANG_USE_ROCM700A=1 -e SGLANG_OPT_USE_FUSED_COMPRESS=true
  -e SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
  -e SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
  -e SGLANG_OPT_FP8_WO_A_GEMM=false -e SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
  -e SGLANG_OPT_USE_TOPK_V2=false -e SGLANG_OPT_USE_AITER_INDEXER=true
  -e SGLANG_OPT_USE_TILELANG_INDEXER=false -e SGLANG_OPT_USE_TILELANG_MHC_PRE=false
  -e SGLANG_OPT_USE_TILELANG_MHC_POST=false -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
  -e SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false -e SGLANG_ROCM_USE_MULTI_STREAM=false
  -e AITER_BF16_FP8_MOE_BOUND=0 -e SGLANG_DSV4_FP4_EXPERTS=false
)
DSV4_ENV_STR="${DSV4_ENV[*]}"
MORI_ENV="-e MORI_DISABLE_AUTO_XGMI=1 -e NCCL_IB_HCA=ionic -e NCCL_IB_GID_INDEX=1 -e NCCL_CROSS_NIC=1"

COMMON_FLAGS="--trust-remote-code --tp $PTP --disable-radix-cache \
--attention-backend $ATTN --max-running-requests $MAXREQ --page-size $PAGE \
--mem-fraction-static $MEMFRAC --swa-full-tokens-ratio $SWA \
--chunked-prefill-size $CHUNK --disable-shared-experts-fusion \
--tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
--disaggregation-transfer-backend mori --disaggregation-ib-device $IB"

DOCKER_COMMON="--rm --network host --ipc host --shm-size 32g --privileged \
--security-opt seccomp=unconfined \
--device /dev/kfd --device /dev/dri --device /dev/infiniband \
-v /it-share:/it-share:ro -v $HOME:/host_home"

# ---------------------------------------------------------------------------
# Write per-role scripts that srun dispatches to each compute node.
# ---------------------------------------------------------------------------
cat > "$WORKDIR/prefill.sh" <<EOF
#!/bin/bash
docker rm -f mi355x_prefill 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_prefill \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $MORI_ENV $DSV4_ENV_STR \
  $IMAGE python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --host 0.0.0.0 --port $PPORT \
  $COMMON_FLAGS --disaggregation-mode prefill --disaggregation-bootstrap-port $PBOOT
EOF

cat > "$WORKDIR/decode.sh" <<EOF
#!/bin/bash
docker rm -f mi355x_decode 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_decode \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $MORI_ENV $DSV4_ENV_STR \
  $IMAGE python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --host 0.0.0.0 --port $DPORT \
  $COMMON_FLAGS --disaggregation-mode decode --disaggregation-bootstrap-port $DBOOT
EOF

# Bench script runs on the prefill node; \$PIP/\$DIP injected at srun time.
cat > "$WORKDIR/bench.sh" <<EOF
#!/bin/bash
set -e
PIP=\$1; DIP=\$2
docker rm -f mi355x_bench 2>/dev/null || true
docker run $DOCKER_COMMON --name mi355x_bench \
  -e PIP=\$PIP -e DIP=\$DIP \
  $IMAGE bash -lc '
    export PYTHONPATH=/sgl-workspace/sglang/python:\$PYTHONPATH
    echo "[wait] prefill"; for i in \$(seq 1 600); do curl -sf http://\$PIP:$PPORT/health >/dev/null && break; sleep 5; done
    echo "[wait] decode";  for i in \$(seq 1 600); do curl -sf http://\$DIP:$DPORT/health >/dev/null && break; sleep 5; done
    python3 /host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/standalone_lb.py \
      --prefill http://\$PIP:$PPORT $PBOOT --decode http://\$DIP:$DPORT \
      --host 0.0.0.0 --port $LBPORT &
    for i in \$(seq 1 30); do curl -sf http://127.0.0.1:$LBPORT/health >/dev/null && break; sleep 2; done
    for C in ${CONCS//,/ }; do
      echo "=== concurrency=\$C ==="
      OUT=/host_home/.mi355x_ci/${MATRIX_CONFIG_NAME}/raw_conc\${C}.json
      rm -f \$OUT
      python3 -m sglang.bench_serving --backend sglang \
        --host 127.0.0.1 --port $LBPORT --model $MODEL_PATH \
        --dataset-name random --random-input-len $ISL --random-output-len $OSL \
        --random-range-ratio $RRR --max-concurrency \$C \
        --num-prompts \$((C*$NPF)) --warmup-requests \$C \
        --output-file \$OUT || true
    done
  '
EOF
chmod +x "$WORKDIR"/*.sh

# ---------------------------------------------------------------------------
# Orchestration drive (runs inside the salloc allocation on the login node).
# ---------------------------------------------------------------------------
cat > "$WORKDIR/drive.sh" <<'DRIVE'
#!/bin/bash
set -x
WORKDIR="$1"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
PNODE="${NODES[0]}"; DNODE="${NODES[1]}"
PIP=$(getent ahostsv4 "$PNODE" | head -1 | awk '{print $1}')
DIP=$(getent ahostsv4 "$DNODE" | head -1 | awk '{print $1}')
echo "[drive] prefill=$PNODE($PIP) decode=$DNODE($DIP)"
srun --overlap -N1 --nodelist="$PNODE" bash "$WORKDIR/prefill.sh" > "$WORKDIR/prefill.log" 2>&1 &
srun --overlap -N1 --nodelist="$DNODE" bash "$WORKDIR/decode.sh"  > "$WORKDIR/decode.log"  2>&1 &
sleep 5
srun --overlap -N1 --nodelist="$PNODE" bash "$WORKDIR/bench.sh" "$PIP" "$DIP" > "$WORKDIR/bench.log" 2>&1
echo "[drive] bench done, tearing down"
srun --overlap -N1 --nodelist="$PNODE" docker kill mi355x_prefill >/dev/null 2>&1 || true
srun --overlap -N1 --nodelist="$DNODE" docker kill mi355x_decode  >/dev/null 2>&1 || true
DRIVE
chmod +x "$WORKDIR/drive.sh"

NODELIST_ARG=()
[[ -n "${SLURM_NODELIST:-}" ]] && NODELIST_ARG=(--nodelist="$SLURM_NODELIST")

salloc -p "$SLURM_PARTITION" -N2 "${NODELIST_ARG[@]}" -t "$TIME_LIMIT" \
    bash "$WORKDIR/drive.sh" "$WORKDIR"

echo "--- bench.log tail ---"; tail -40 "$WORKDIR/bench.log" || true

# ---------------------------------------------------------------------------
# Normalize raw bench_serving output -> process_result.py schema.
#
# bench_serving and process_result.py disagree on field names, so we remap the
# last JSON line of each raw file. If bench_serving ever renames an output
# field, the KeyError raised here (rather than a silently wrong table) is the
# signal to update this mapping. Field-by-field:
#
#   bench_serving key          ->  process_result.py key       (purpose)
#   --------------------------     ------------------------     -------------------------
#   max_concurrency            ->  max_concurrency             (sweep point; falls back to $C)
#   total_throughput           ->  total_token_throughput      (in+out tok/s, tput_per_gpu)
#   output_throughput          ->  output_throughput           (out tok/s, output_tput_per_gpu)
#   median_ttft_ms             ->  median_ttft_ms              (TTFT; /1000 -> s)
#   median_tpot_ms             ->  median_tpot_ms              (TPOT; -> interactivity)
#   median_e2e_latency_ms      ->  median_e2el_ms              (E2E latency; /1000 -> s)
#   (none; injected here)      ->  model_id                    (served model, from $MODEL_PATH)
# ---------------------------------------------------------------------------
TOTAL_GPUS=$((PTP + DTP))
PROCESSED=0
for C in ${CONCS//,/ }; do
    RAW="$WORKDIR/raw_conc${C}.json"
    [[ -f "$RAW" ]] || { echo "WARN: missing $RAW"; continue; }
    DEST="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${MATRIX_CONFIG_NAME}_conc${C}_gpus_${TOTAL_GPUS}_ctx_${PTP}_gen_${DTP}.json"
    MODEL_ID="$MODEL_PATH" python3 - "$RAW" "$DEST" "$C" <<'PY'
import json, os, sys
raw_path, dest, conc = sys.argv[1], sys.argv[2], int(sys.argv[3])
line = [l for l in open(raw_path).read().splitlines() if l.strip()][-1]
r = json.loads(line)
norm = {
    "max_concurrency": r.get("max_concurrency") or conc,
    "model_id": os.environ["MODEL_ID"],
    "total_token_throughput": r["total_throughput"],
    "output_throughput": r["output_throughput"],
    "median_ttft_ms": r["median_ttft_ms"],
    "median_tpot_ms": r["median_tpot_ms"],
    "median_e2el_ms": r["median_e2e_latency_ms"],
}
json.dump(norm, open(dest, "w"), indent=2)
print("normalized ->", dest)
PY
    PROCESSED=$((PROCESSED + 1))
done

if [[ "$PROCESSED" -eq 0 ]]; then
    echo "ERROR: no result files produced" >&2
    exit 1
fi
echo "Done. $PROCESSED result file(s) in $GITHUB_WORKSPACE."
