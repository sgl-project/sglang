#!/bin/bash
set -euo pipefail

# ============================================================
# Tiered Cache Benchmark: L1 / L1+L2 / L1+L2+L3
#
# DP=8 + EP=8 variant (--moe-a2a-backend mori auto-sets ep=tp).
# Uses write_through policy.
#
# With DP=8, each GPU runs an independent KV cache (attn TP=1).
# Load is distributed across 8 DP ranks, so each rank sees ~1/8
# of total traffic.  NUM_ROUNDS is raised to compensate for the
# lower per-rank pressure.
# ============================================================

# ---- Configurable parameters --------------------------------
MODEL_PATH="${MODEL_PATH:-/nfs/DeepSeek-V3}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-8}"
PAGE_SIZE="${PAGE_SIZE:-64}"

NUM_ROUNDS="${NUM_ROUNDS:-130}"
NUM_CLIENTS="${NUM_CLIENTS:-64}"
MAX_PARALLEL="${MAX_PARALLEL:-64}"
REQUEST_LENGTH="${REQUEST_LENGTH:-2048}"
SUB_QUESTION_INPUT_LENGTH="${SUB_QUESTION_INPUT_LENGTH:-430}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1}"
REQUEST_RATE="${REQUEST_RATE:-10}"
SEED="${SEED:-42}"

HICACHE_SIZE="${HICACHE_SIZE:-128}"                  # L2 128 GB/rank
UMBP_DRAM_BYTES="${UMBP_DRAM_BYTES:-68719476736}"    # L3 DRAM 64 GB/rank
UMBP_SSD_BYTES="${UMBP_SSD_BYTES:-103079215104}"     # L3 SSD  96 GB/rank
UMBP_SSD_DIR="${UMBP_SSD_DIR:-/tmp/umbp_ssd}"
UMBP_SSD_DURABILITY_MODE="${UMBP_SSD_DURABILITY_MODE:-relaxed}"
UMBP_COPY_TO_SSD_ASYNC="${UMBP_COPY_TO_SSD_ASYNC:-true}"
UMBP_SSD_WRITER_THREADS="${UMBP_SSD_WRITER_THREADS:-4}"

# SPDK backend (set UMBP_SSD_BACKEND=spdk_proxy to enable)
UMBP_SSD_BACKEND="${UMBP_SSD_BACKEND:-posix}"          # posix | spdk_proxy
UMBP_SPDK_NVME_PCI="${UMBP_SPDK_NVME_PCI:-}"           # e.g. 0000:89:00.0
UMBP_SPDK_PROXY_AUTO_START="${UMBP_SPDK_PROXY_AUTO_START:-true}"
UMBP_SPDK_PROXY_STARTUP_TIMEOUT_MS="${UMBP_SPDK_PROXY_STARTUP_TIMEOUT_MS:-60000}"
UMBP_MASTER_ADDRESS="${UMBP_MASTER_ADDRESS:-}"
UMBP_NODE_ADDRESS="${UMBP_NODE_ADDRESS:-}"
UMBP_IO_ENGINE_HOST="${UMBP_IO_ENGINE_HOST:-}"
UMBP_IO_ENGINE_PORT="${UMBP_IO_ENGINE_PORT:-}"
UMBP_PEER_SERVICE_PORT="${UMBP_PEER_SERVICE_PORT:-}"
UMBP_CACHE_REMOTE_FETCHES="${UMBP_CACHE_REMOTE_FETCHES:-true}"
UMBP_MASTER_AUTO_START="${UMBP_MASTER_AUTO_START:-true}"
UMBP_MASTER_BIN="${UMBP_MASTER_BIN:-}"
UMBP_MASTER_LISTEN="${UMBP_MASTER_LISTEN:-}"

SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-6000}"   # seconds
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-10800}"        # per-case timeout

WRITE_POLICY="write_through"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-mori}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
DUMMY_FORWARD="${DUMMY_FORWARD:-}"

# ---- Derived paths ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${SCRIPT_DIR}/results/tiered_cache_bench_dp8ep8/${TIMESTAMP}"
PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
export MORI_SHMEM_MODE=ISOLATION
export MORI_SHMEM_HEAP_SIZE=6G

# ---- Helpers ------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

bool_is_true() {
    local v="${1:-}"
    v="${v,,}"
    [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]
}

master_addr_host() {
    local addr="$1"
    echo "${addr%:*}"
}

master_addr_port() {
    local addr="$1"
    echo "${addr##*:}"
}

wait_for_tcp() {
    local host="$1"
    local port="$2"
    local timeout_s="$3"
    local label="$4"
    local elapsed=0
    local interval=2

    while (( elapsed < timeout_s )); do
        if python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
raise SystemExit(0)
PY
        then
            return 0
        fi
        sleep "$interval"
        elapsed=$(( elapsed + interval ))
    done

    log "ERROR: ${label} at ${host}:${port} not reachable within ${timeout_s}s."
    return 1
}

find_packaged_umbp_master() {
    python3 - <<'PY' 2>/dev/null || true
from pathlib import Path

try:
    import mori.umbp  # noqa: F401
    import os

    env_path = os.environ.get("UMBP_MASTER_BIN", "")
    if env_path and Path(env_path).is_file():
        print(env_path)
    else:
        import mori

        candidate = Path(mori.__file__).resolve().parent / "umbp_master"
        if candidate.is_file():
            print(candidate)
except Exception:
    pass
PY
}

validate_umbp_distributed_config() {
    if [[ -z "$UMBP_MASTER_ADDRESS" ]]; then
        return 0
    fi

    if [[ -z "$UMBP_IO_ENGINE_PORT" ]]; then
        log "ERROR: UMBP_MASTER_ADDRESS is set, but UMBP_IO_ENGINE_PORT is empty."
        return 1
    fi
    if [[ -z "$UMBP_PEER_SERVICE_PORT" ]]; then
        log "ERROR: UMBP_MASTER_ADDRESS is set, but UMBP_PEER_SERVICE_PORT is empty."
        return 1
    fi

    if [[ -z "$UMBP_NODE_ADDRESS" ]]; then
        log "WARNING: UMBP_NODE_ADDRESS is empty. UMBPStore will fall back to hostname resolution."
    fi

    return 0
}

MASTER_PID=""
MASTER_STARTED_BY_SCRIPT=0

kill_master() {
    if [[ -n "${MASTER_PID:-}" ]] && kill -0 "$MASTER_PID" 2>/dev/null; then
        log "Stopping UMBP master (PID $MASTER_PID)..."
        kill -TERM "$MASTER_PID" 2>/dev/null || true
        wait "$MASTER_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$MASTER_PID" 2>/dev/null || true
    fi
    MASTER_PID=""
    MASTER_STARTED_BY_SCRIPT=0
}

ensure_umbp_master() {
    if [[ -z "$UMBP_MASTER_ADDRESS" ]]; then
        return 0
    fi

    validate_umbp_distributed_config || return 1

    local host port check_host
    host="$(master_addr_host "$UMBP_MASTER_ADDRESS")"
    port="$(master_addr_port "$UMBP_MASTER_ADDRESS")"
    check_host="$host"
    if [[ "$check_host" == "0.0.0.0" ]]; then
        check_host="127.0.0.1"
    fi

    if wait_for_tcp "$check_host" "$port" 2 "UMBP master"; then
        log "Using existing UMBP master at ${UMBP_MASTER_ADDRESS}."
        return 0
    fi

    if ! bool_is_true "$UMBP_MASTER_AUTO_START"; then
        log "ERROR: UMBP master is not reachable at ${UMBP_MASTER_ADDRESS} and auto-start is disabled."
        return 1
    fi

    local mori_repo_root mori_build_dir master_bin master_listen master_log
    mori_repo_root="$(cd "${REPO_ROOT}/../mori" 2>/dev/null && pwd || true)"
    mori_build_dir="${MORI_BUILD_DIR:-${mori_repo_root}/build}"
    master_bin="${UMBP_MASTER_BIN:-${mori_build_dir}/src/umbp/umbp_master}"
    if [[ ! -x "$master_bin" ]]; then
        local packaged_master
        packaged_master="$(find_packaged_umbp_master)"
        if [[ -n "$packaged_master" ]]; then
            master_bin="$packaged_master"
        fi
    fi
    master_listen="${UMBP_MASTER_LISTEN:-0.0.0.0:${port}}"
    master_log="${RESULTS_DIR}/umbp_master.log"

    if [[ ! -x "$master_bin" ]]; then
        log "ERROR: UMBP master binary not found or not executable: $master_bin"
        log "Set UMBP_MASTER_BIN or MORI_BUILD_DIR explicitly."
        return 1
    fi

    log "Starting UMBP master: ${master_bin} ${master_listen}"
    "$master_bin" "$master_listen" > "$master_log" 2>&1 &
    MASTER_PID=$!
    MASTER_STARTED_BY_SCRIPT=1

    if ! wait_for_tcp "$check_host" "$port" 30 "UMBP master"; then
        log "ERROR: UMBP master failed to become ready. Check $master_log"
        kill_master
        return 1
    fi

    log "UMBP master is ready at ${UMBP_MASTER_ADDRESS}."
    return 0
}

kill_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (PID $SERVER_PID) and its children..."
        pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        sleep 1
        pkill -9 -P "$SERVER_PID" 2>/dev/null || true
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    kill_stale_sglang_procs
    wait_for_port_free
}

kill_stale_sglang_procs() {
    local pids
    pids=$(pgrep -f "sglang\.launch_server|sglang\.srt\." 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log "Killing stale sglang processes: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

wait_for_port_free() {
    local max_wait=60
    local elapsed=0
    local pids
    while (( elapsed < max_wait )); do
        pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
        if [[ -z "$pids" ]]; then
            return 0
        fi
        if (( elapsed == 0 )); then
            log "Waiting for port $PORT to be released (PIDs: $pids)..."
        fi
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
        elapsed=$(( elapsed + 2 ))
    done
    pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log "ERROR: Port $PORT still occupied after ${max_wait}s (PIDs: $pids)"
        return 1
    fi
}

clean_ssd_dir() {
    local retries=5
    for (( i=1; i<=retries; i++ )); do
        rm -rf "$UMBP_SSD_DIR" 2>/dev/null && return 0
        log "Retrying SSD dir cleanup ($i/$retries)..."
        sleep 2
    done
    log "WARNING: Could not fully remove $UMBP_SSD_DIR, proceeding anyway."
}

wait_for_server() {
    local url="http://localhost:${PORT}/v1/models"
    local elapsed=0
    local interval=30
    log "Waiting for server at $url (timeout ${SERVER_READY_TIMEOUT}s)..."
    while (( elapsed < SERVER_READY_TIMEOUT )); do
        if curl -sf "$url" > /dev/null 2>&1; then
            log "Server is ready (took ${elapsed}s)."
            return 0
        fi
        sleep "$interval"
        elapsed=$(( elapsed + interval ))
    done
    log "ERROR: Server did not become ready within ${SERVER_READY_TIMEOUT}s."
    return 1
}

run_benchmark() {
    local case_dir="$1"
    local case_tag="$2"
    local log_file="${case_dir}/bench.log"
    local metrics_file="${case_dir}/performance_metrics.jsonl"

    log "Starting benchmark (rounds=${NUM_ROUNDS}, clients=${NUM_CLIENTS}, max_parallel=${MAX_PARALLEL}, request_length=${REQUEST_LENGTH}, sub_question_input_length=${SUB_QUESTION_INPUT_LENGTH}, output_length=${OUTPUT_LENGTH}, request_rate=${REQUEST_RATE}, timeout ${BENCHMARK_TIMEOUT}s)..."
    timeout --signal=TERM --kill-after=30 "$BENCHMARK_TIMEOUT" \
        python "${SCRIPT_DIR}/bench_multiturn.py" \
            --model-path "$MODEL_PATH" \
            --port "$PORT" \
            --num-clients "$NUM_CLIENTS" \
            --max-parallel "$MAX_PARALLEL" \
            --num-rounds "$NUM_ROUNDS" \
            --request-length "$REQUEST_LENGTH" \
            --sub-question-input-length "$SUB_QUESTION_INPUT_LENGTH" \
            --output-length "$OUTPUT_LENGTH" \
            --request-rate "$REQUEST_RATE" \
            --disable-auto-run \
            --enable-round-barrier \
            --seed "$SEED" \
            --log-file "$metrics_file" \
            --tag "$case_tag" \
            2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    return "$exit_code"
}

# ---- Common server args for DP + EP -------------------------
DP_EP_ARGS=(
    --dp-size "$DP_SIZE"
    --enable-dp-attention
    --moe-a2a-backend "$MOE_A2A_BACKEND"
)
if [[ -n "$KV_CACHE_DTYPE" ]]; then
    DP_EP_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if bool_is_true "$DUMMY_FORWARD"; then
    DP_EP_ARGS+=(--dummy-forward)
fi

launch_server_case1() {
    python -m sglang.launch_server \
        --enable-cache-report --enable-metrics \
        --model-path "$MODEL_PATH" \
        --host 0.0.0.0 \
        --tp-size "$TP_SIZE" --page-size "$PAGE_SIZE" \
        "${DP_EP_ARGS[@]}" \
        "$@"
}

launch_server_case2() {
    python -m sglang.launch_server \
        --enable-cache-report --enable-metrics \
        --model-path "$MODEL_PATH" \
        --host 0.0.0.0 \
        --tp-size "$TP_SIZE" --page-size "$PAGE_SIZE" \
        "${DP_EP_ARGS[@]}" \
        --enable-hierarchical-cache \
        --hicache-size "$HICACHE_SIZE" \
        --hicache-write-policy "$WRITE_POLICY" \
        --hicache-mem-layout page_first \
        "$@"
}

launch_server_case3() {
    local spdk_fields=""
    if [[ "$UMBP_SSD_BACKEND" != "posix" ]]; then
        spdk_fields=", \"ssd_backend\": \"${UMBP_SSD_BACKEND}\""
        [[ -n "$UMBP_SPDK_NVME_PCI" ]] && \
            spdk_fields+=", \"spdk_nvme_pci_addr\": \"${UMBP_SPDK_NVME_PCI}\""
        spdk_fields+=", \"spdk_proxy_auto_start\": ${UMBP_SPDK_PROXY_AUTO_START}"
        spdk_fields+=", \"spdk_proxy_startup_timeout_ms\": ${UMBP_SPDK_PROXY_STARTUP_TIMEOUT_MS}"
        spdk_fields+=", \"spdk_proxy_tenant_id_base\": 0"
    fi
    local dist_fields=""
    if [[ -n "$UMBP_MASTER_ADDRESS" ]]; then
        dist_fields+=", \"master_address\": \"${UMBP_MASTER_ADDRESS}\""
        [[ -n "$UMBP_NODE_ADDRESS" ]] && \
            dist_fields+=", \"node_address\": \"${UMBP_NODE_ADDRESS}\""
        [[ -n "$UMBP_IO_ENGINE_HOST" ]] && \
            dist_fields+=", \"io_engine_host\": \"${UMBP_IO_ENGINE_HOST}\""
        [[ -n "$UMBP_IO_ENGINE_PORT" ]] && \
            dist_fields+=", \"io_engine_port\": \"${UMBP_IO_ENGINE_PORT}\""
        [[ -n "$UMBP_PEER_SERVICE_PORT" ]] && \
            dist_fields+=", \"peer_service_port\": \"${UMBP_PEER_SERVICE_PORT}\""
        dist_fields+=", \"cache_remote_fetches\": ${UMBP_CACHE_REMOTE_FETCHES}"
    fi
    local extra_config="{\"dram_capacity_bytes\": ${UMBP_DRAM_BYTES}, \"ssd_enabled\": true, \"ssd_storage_dir\": \"${UMBP_SSD_DIR}\", \"ssd_capacity_bytes\": ${UMBP_SSD_BYTES}, \"auto_promote_on_read\": true, \"eviction_policy\": \"prefix_aware_lru\", \"ssd_durability_mode\": \"${UMBP_SSD_DURABILITY_MODE}\", \"copy_to_ssd_async\": ${UMBP_COPY_TO_SSD_ASYNC}, \"ssd_writer_threads\": ${UMBP_SSD_WRITER_THREADS}${spdk_fields}${dist_fields}}"

    python -m sglang.launch_server \
        --enable-cache-report --enable-metrics \
        --model-path "$MODEL_PATH" \
        --host 0.0.0.0 \
        --tp-size "$TP_SIZE" --page-size "$PAGE_SIZE" \
        "${DP_EP_ARGS[@]}" \
        --enable-hierarchical-cache \
        --hicache-size "$HICACHE_SIZE" \
        --hicache-write-policy "$WRITE_POLICY" \
        --hicache-mem-layout page_first \
        --hicache-storage-backend umbp \
        --hicache-storage-backend-extra-config "$extra_config" \
        "$@"
}

# ---- Main ---------------------------------------------------
SERVER_PID=""
trap 'kill_server; kill_master; exit 130' INT TERM

read -r -a CASES <<< "${CASES_OVERRIDE:-case3:HBM_DRAM_SSD case2:HBM_DRAM case1:HBM_Only}"

log "======================================================"
log "Tiered Cache Benchmark (DP=${DP_SIZE} + EP via ${MOE_A2A_BACKEND})"
log "  Model:       $MODEL_PATH"
log "  TP:          $TP_SIZE"
log "  DP:          $DP_SIZE"
log "  MoE A2A:     $MOE_A2A_BACKEND"
log "  Rounds:      $NUM_ROUNDS"
log "  Clients:     $NUM_CLIENTS"
log "  Max parallel: $MAX_PARALLEL"
log "  Request len: $REQUEST_LENGTH"
log "  Sub-q input: $SUB_QUESTION_INPUT_LENGTH"
log "  Output len:  $OUTPUT_LENGTH"
log "  Request rate: $REQUEST_RATE"
log "  Write policy: $WRITE_POLICY"
log "  KV dtype:    ${KV_CACHE_DTYPE:-auto}"
if bool_is_true "$DUMMY_FORWARD"; then
log "  Dummy fwd:   ENABLED (dummy weights, no compute)"
fi
log "  L2 size:     ${HICACHE_SIZE} GB/rank"
log "  L3 DRAM:     $((UMBP_DRAM_BYTES / 1073741824)) GB/rank"
log "  L3 SSD:      $((UMBP_SSD_BYTES / 1073741824)) GB/rank"
log "  L3 SSD:      durability=${UMBP_SSD_DURABILITY_MODE}, async_copy=${UMBP_COPY_TO_SSD_ASYNC}, backend=${UMBP_SSD_BACKEND}"
if [[ "$UMBP_SSD_BACKEND" != "posix" ]]; then
log "  SPDK:        pci=${UMBP_SPDK_NVME_PCI:-auto}, auto_start=${UMBP_SPDK_PROXY_AUTO_START}"
fi
if [[ -n "$UMBP_MASTER_ADDRESS" ]]; then
log "  UMBP dist:   master=${UMBP_MASTER_ADDRESS} node=${UMBP_NODE_ADDRESS:-auto} io_port=${UMBP_IO_ENGINE_PORT} peer_port=${UMBP_PEER_SERVICE_PORT} auto_master=${UMBP_MASTER_AUTO_START}"
else
log "  UMBP dist:   disabled (case3 runs local-only UMBP)"
fi
log "  Results dir: $RESULTS_DIR"
log "======================================================"

OVERALL_START=$(date +%s)
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
mkdir -p "$RESULTS_DIR"

{
    echo "Tiered Cache Benchmark Summary (DP=${DP_SIZE} + EP)"
    echo "Started: $(date)"
    echo "Model: $MODEL_PATH  TP: $TP_SIZE  DP: $DP_SIZE  MoE: $MOE_A2A_BACKEND"
    echo "Rounds: $NUM_ROUNDS  Clients: $NUM_CLIENTS"
    echo "Max parallel: $MAX_PARALLEL  Request length: $REQUEST_LENGTH  Sub-question input length: $SUB_QUESTION_INPUT_LENGTH"
    echo "Output length: $OUTPUT_LENGTH  Request rate: $REQUEST_RATE"
    echo "Write policy: $WRITE_POLICY"
    echo "L2: ${HICACHE_SIZE} GB/rank  L3 DRAM: $((UMBP_DRAM_BYTES / 1073741824)) GB/rank  L3 SSD: $((UMBP_SSD_BYTES / 1073741824)) GB/rank"
    echo "L3 SSD: durability=${UMBP_SSD_DURABILITY_MODE}  async_copy=${UMBP_COPY_TO_SSD_ASYNC}  writer_threads=${UMBP_SSD_WRITER_THREADS}  backend=${UMBP_SSD_BACKEND}"
    echo ""
} > "$SUMMARY_FILE"

for entry in "${CASES[@]}"; do
    CASE_ID="${entry%%:*}"
    CASE_NAME="${entry##*:}"
    CASE_TAG="${CASE_ID}_${CASE_NAME}"
    CASE_DIR="${RESULTS_DIR}/${CASE_ID}_${CASE_NAME}"
    mkdir -p "$CASE_DIR"

    log "------------------------------------------------------"
    log "Starting ${CASE_NAME} (${CASE_ID})"
    log "------------------------------------------------------"

    # 1. Cleanup
    kill_server
    if [[ "$CASE_ID" == "case3" ]]; then
        log "Cleaning UMBP SSD dir: $UMBP_SSD_DIR"
        clean_ssd_dir
        if ! ensure_umbp_master; then
            log "SKIP ${CASE_NAME}: UMBP master is not available."
            echo "${CASE_NAME}: FAILED (umbp master unavailable)" >> "$SUMMARY_FILE"
            continue
        fi
    fi

    # 2. Launch server in background
    SERVER_LOG="${CASE_DIR}/server.log"
    log "Launching server for ${CASE_NAME}..."
    "launch_server_${CASE_ID}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    log "Server PID: $SERVER_PID"

    # 3. Wait for server to be ready
    if ! wait_for_server; then
        log "SKIP ${CASE_NAME}: server failed to start. Check $SERVER_LOG"
        echo "${CASE_NAME}: FAILED (server did not start)" >> "$SUMMARY_FILE"
        kill_server
        continue
    fi

    # 4. Run benchmark (in background so we can monitor server liveness)
    CASE_START=$(date +%s)
    SERVER_CRASHED=false

    run_benchmark "$CASE_DIR" "$CASE_TAG" &
    BENCH_PID=$!

    while kill -0 "$BENCH_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "ERROR: Server crashed during ${CASE_NAME} benchmark! Aborting..."
            SERVER_CRASHED=true
            pkill -TERM -P "$BENCH_PID" 2>/dev/null || true
            kill -TERM "$BENCH_PID" 2>/dev/null || true
            sleep 3
            pkill -9 -P "$BENCH_PID" 2>/dev/null || true
            kill -9 "$BENCH_PID" 2>/dev/null || true
            break
        fi
        sleep 10
    done

    BENCH_RC=0
    wait "$BENCH_PID" 2>/dev/null || BENCH_RC=$?
    CASE_END=$(date +%s)
    CASE_ELAPSED=$(( CASE_END - CASE_START ))

    if $SERVER_CRASHED; then
        log "ERROR: ${CASE_NAME} aborted — server crashed (${CASE_ELAPSED}s)."
        echo "${CASE_NAME}: SERVER_CRASH (${CASE_ELAPSED}s)" >> "$SUMMARY_FILE"
    elif (( BENCH_RC == 0 )); then
        log "${CASE_NAME} completed in ${CASE_ELAPSED}s."
        echo "${CASE_NAME}: PASSED (${CASE_ELAPSED}s)" >> "$SUMMARY_FILE"
    elif (( BENCH_RC == 124 )); then
        log "WARNING: ${CASE_NAME} timed out after ${BENCHMARK_TIMEOUT}s (ran ${CASE_ELAPSED}s)."
        echo "${CASE_NAME}: TIMEOUT (${CASE_ELAPSED}s)" >> "$SUMMARY_FILE"
    else
        log "WARNING: ${CASE_NAME} benchmark exited with code ${BENCH_RC} (${CASE_ELAPSED}s)."
        echo "${CASE_NAME}: ERROR rc=${BENCH_RC} (${CASE_ELAPSED}s)" >> "$SUMMARY_FILE"
    fi

    # 5. Stop server
    kill_server

    # 6. Post-cleanup for SSD
    if [[ "$CASE_ID" == "case3" ]]; then
        log "Cleaning UMBP SSD dir after test."
        clean_ssd_dir
    fi

    log "${CASE_NAME} done."

    sleep 30
done

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$(( OVERALL_END - OVERALL_START ))

{
    echo ""
    echo "Finished: $(date)"
    echo "Total elapsed: ${OVERALL_ELAPSED}s"
} >> "$SUMMARY_FILE"

log "======================================================"
log "All cases finished in ${OVERALL_ELAPSED}s."
log "Results: $RESULTS_DIR"
log "======================================================"
cat "$SUMMARY_FILE"

kill_master
