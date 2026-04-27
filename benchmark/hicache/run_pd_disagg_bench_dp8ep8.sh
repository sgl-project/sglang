#!/bin/bash
set -euo pipefail

# ============================================================
# PD Disaggregation Benchmark: Prefill / Decode split
#
# Run the SAME script on two machines with different --role:
#
#   Machine A (Prefill):
#     ./run_pd_disagg_bench_dp8ep8.sh --role prefill
#
#   Machine B (Decode):
#     PREFILL_URL=http://<prefill_ip>:30000 \
#       ./run_pd_disagg_bench_dp8ep8.sh --role decode
#
# DP=8 + EP=8 (mori), write_through policy.
# Cache tiers controlled via ENABLE_HICACHE / ENABLE_UMBP.
# Decode node auto-starts sglang_router by default.
# ============================================================

usage() {
    cat <<EOF
Usage: $0 --role <prefill|decode> [options]

Required:
  --role <prefill|decode>   Node role in PD disaggregation

Options:
  --help                    Show this help message

Environment variables (override defaults):
  MODEL_PATH                Model path (default: /nfs/DeepSeek-V3)
  TP_SIZE                   Tensor parallelism (default: 8)
  DP_SIZE                   Data parallelism (default: 8)

  # PD Disaggregation
  DISAGG_TRANSFER_BACKEND   Transfer backend (default: mori)
  DISAGG_BOOTSTRAP_PORT     Prefill bootstrap port (default: 8998)
  PREFILL_HOST              Prefill server bind address (default: 0.0.0.0)
  PREFILL_PORT              Prefill server port (default: 30000)
  DECODE_HOST               Decode server bind address (default: 0.0.0.0)
  DECODE_PORT               Decode server port (default: 30001)
  PREFILL_URL               Full URL to prefill server (decode needs this,
                            e.g. http://10.0.0.1:30000)

  # Router (decode node only)
  ENABLE_ROUTER             Auto-start router on decode node (default: true)
  ROUTER_HOST               Router bind address (default: 0.0.0.0)
  ROUTER_PORT               Router port (default: 8000)

  # KV Events (ZMQ publisher on SGLang + subscriber in UMBP)
  ENABLE_KV_EVENTS          Enable KV events publisher on sglang server (default: true)
  KV_EVENTS_PUBLISHER       Publisher backend (default: zmq)
  KV_EVENTS_ENDPOINT        ZMQ bind endpoint (default: tcp://*:5557)
  KV_EVENTS_TOPIC           ZMQ topic prefix (default: empty)

  # Cache tiers
  ENABLE_HICACHE            Enable L2 DRAM cache (default: true)
  ENABLE_UMBP               Enable L3 UMBP (DRAM+SSD) (default: true)
  HICACHE_SIZE              L2 DRAM size in GB/rank (default: 128)

  # Decode-specific
  MAX_RUNNING_REQUESTS      Max concurrent requests on decode (default: 128)

  # Multi-node distributed (per role, if nnodes>1)
  DIST_INIT_ADDR            --dist-init-addr for torch distributed
  NNODES                    Number of nodes for this role (default: 1)
  NODE_RANK                 This node's rank (default: 0)

Examples:
  # Prefill node with all cache tiers:
  ./run_pd_disagg_bench_dp8ep8.sh --role prefill

  # Decode node, connecting to prefill at 10.0.0.1:
  PREFILL_URL=http://10.0.0.1:30000 \\
    ./run_pd_disagg_bench_dp8ep8.sh --role decode

  # Prefill node, HBM only (no tiered cache):
  ENABLE_HICACHE=false ENABLE_UMBP=false \\
    ./run_pd_disagg_bench_dp8ep8.sh --role prefill
EOF
    exit 0
}

# ---- Argument parsing ---------------------------------------
PD_ROLE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --role)
            PD_ROLE="$2"; shift 2 ;;
        --help|-h)
            usage ;;
        *)
            echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$PD_ROLE" ]]; then
    echo "ERROR: --role is required (prefill or decode)"
    usage
fi
if [[ "$PD_ROLE" != "prefill" && "$PD_ROLE" != "decode" ]]; then
    echo "ERROR: --role must be 'prefill' or 'decode', got '$PD_ROLE'"
    exit 1
fi

# ---- Configurable parameters --------------------------------
MODEL_PATH="${MODEL_PATH:-/nfs/DeepSeek-V3}"
USE_DUMMY_WEIGHTS="${USE_DUMMY_WEIGHTS:-false}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-8}"
PAGE_SIZE="${PAGE_SIZE:-64}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"

# PD Disaggregation
DISAGG_TRANSFER_BACKEND="${DISAGG_TRANSFER_BACKEND:-mori}"
DISAGG_BOOTSTRAP_PORT="${DISAGG_BOOTSTRAP_PORT:-8998}"
PREFILL_HOST="${PREFILL_HOST:-0.0.0.0}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
DECODE_HOST="${DECODE_HOST:-0.0.0.0}"
DECODE_PORT="${DECODE_PORT:-30001}"
PREFILL_URL="${PREFILL_URL:-}"

# Router (decode node only)
ENABLE_ROUTER="${ENABLE_ROUTER:-true}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

# Cache tiers
ENABLE_HICACHE="${ENABLE_HICACHE:-true}"
ENABLE_UMBP="${ENABLE_UMBP:-true}"
HICACHE_SIZE="${HICACHE_SIZE:-128}"
WRITE_POLICY="write_through"

# UMBP (L3) config
UMBP_DRAM_BYTES="${UMBP_DRAM_BYTES:-68719476736}"
UMBP_SSD_BYTES="${UMBP_SSD_BYTES:-103079215104}"
UMBP_SSD_DIR="${UMBP_SSD_DIR:-/tmp/umbp_ssd}"
UMBP_SSD_DURABILITY_MODE="${UMBP_SSD_DURABILITY_MODE:-relaxed}"
UMBP_COPY_TO_SSD_ASYNC="${UMBP_COPY_TO_SSD_ASYNC:-true}"
UMBP_SSD_WRITER_THREADS="${UMBP_SSD_WRITER_THREADS:-4}"

# SPDK backend
UMBP_SSD_BACKEND="${UMBP_SSD_BACKEND:-posix}"
UMBP_SPDK_NVME_PCI="${UMBP_SPDK_NVME_PCI:-}"
UMBP_SPDK_PROXY_AUTO_START="${UMBP_SPDK_PROXY_AUTO_START:-true}"
UMBP_SPDK_PROXY_STARTUP_TIMEOUT_MS="${UMBP_SPDK_PROXY_STARTUP_TIMEOUT_MS:-60000}"

# UMBP distributed
UMBP_MASTER_ADDRESS="${UMBP_MASTER_ADDRESS:-}"
UMBP_NODE_ADDRESS="${UMBP_NODE_ADDRESS:-}"
UMBP_IO_ENGINE_HOST="${UMBP_IO_ENGINE_HOST:-}"
UMBP_IO_ENGINE_PORT="${UMBP_IO_ENGINE_PORT:-}"
UMBP_PEER_SERVICE_PORT="${UMBP_PEER_SERVICE_PORT:-}"
UMBP_CACHE_REMOTE_FETCHES="${UMBP_CACHE_REMOTE_FETCHES:-true}"
UMBP_MASTER_AUTO_START="${UMBP_MASTER_AUTO_START:-true}"

# KV Events
ENABLE_KV_EVENTS="${ENABLE_KV_EVENTS:-true}"
KV_EVENTS_PUBLISHER="${KV_EVENTS_PUBLISHER:-zmq}"
KV_EVENTS_ENDPOINT="${KV_EVENTS_ENDPOINT:-tcp://*:5557}"
KV_EVENTS_TOPIC="${KV_EVENTS_TOPIC:-}"
UMBP_MASTER_BIN="${UMBP_MASTER_BIN:-}"
UMBP_MASTER_LISTEN="${UMBP_MASTER_LISTEN:-}"

# Benchmark params (decode node only)
NUM_ROUNDS="${NUM_ROUNDS:-130}"
NUM_CLIENTS="${NUM_CLIENTS:-64}"
MAX_PARALLEL="${MAX_PARALLEL:-64}"
REQUEST_LENGTH="${REQUEST_LENGTH:-2048}"
SUB_QUESTION_INPUT_LENGTH="${SUB_QUESTION_INPUT_LENGTH:-430}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1}"
REQUEST_RATE="${REQUEST_RATE:-10}"
SEED="${SEED:-42}"

# Decode-specific
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"

# Multi-node torch distributed (per role)
DIST_INIT_ADDR="${DIST_INIT_ADDR:-}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

# Timeouts
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-6000}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-10800}"
PREFILL_WAIT_TIMEOUT="${PREFILL_WAIT_TIMEOUT:-6000}"

MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-mori}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

# Extra args forwarded verbatim to sglang.launch_server for both roles.  Split
# on whitespace so flag+value pairs stay together
# (e.g. EXTRA_SERVER_ARGS="--max-total-tokens 20000").
read -r -a EXTRA_SERVER_ARGS_ARR <<< "${EXTRA_SERVER_ARGS:-}"

# ---- Sanity: UMBP requires HICACHE ---------------------------
if bool_is_true "$ENABLE_UMBP" && ! bool_is_true "$ENABLE_HICACHE"; then
    log "WARNING: ENABLE_UMBP=true requires ENABLE_HICACHE=true. Forcing ENABLE_UMBP=false."
    ENABLE_UMBP=false
fi

# ---- Derived paths ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${SCRIPT_DIR}/results/pd_disagg_${PD_ROLE}/${TIMESTAMP}"
PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
export MORI_SHMEM_MODE=ISOLATION
export MORI_SHMEM_HEAP_SIZE=6G

# ---- Helpers ------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

bool_is_true() {
    local v="${1:-}"; v="${v,,}"
    [[ "$v" == "1" || "$v" == "true" || "$v" == "yes" || "$v" == "on" ]]
}

master_addr_host() { echo "${1%:*}"; }
master_addr_port() { echo "${1##*:}"; }

wait_for_tcp() {
    local host="$1" port="$2" timeout_s="$3" label="$4"
    local elapsed=0 interval=2
    while (( elapsed < timeout_s )); do
        if python3 - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket, sys
try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1): pass
except OSError: raise SystemExit(1)
raise SystemExit(0)
PY
        then return 0; fi
        sleep "$interval"; elapsed=$(( elapsed + interval ))
    done
    log "ERROR: ${label} at ${host}:${port} not reachable within ${timeout_s}s."
    return 1
}

find_packaged_umbp_master() {
    python3 - <<'PY' 2>/dev/null || true
from pathlib import Path
try:
    import mori.umbp, os, mori
    env_path = os.environ.get("UMBP_MASTER_BIN", "")
    if env_path and Path(env_path).is_file(): print(env_path)
    else:
        c = Path(mori.__file__).resolve().parent / "umbp_master"
        if c.is_file(): print(c)
except Exception: pass
PY
}

MASTER_PID=""
kill_master() {
    if [[ -n "${MASTER_PID:-}" ]] && kill -0 "$MASTER_PID" 2>/dev/null; then
        log "Stopping UMBP master (PID $MASTER_PID)..."
        kill -TERM "$MASTER_PID" 2>/dev/null || true
        wait "$MASTER_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$MASTER_PID" 2>/dev/null || true
    fi
    MASTER_PID=""
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

ensure_umbp_master() {
    if [[ -z "$UMBP_MASTER_ADDRESS" ]]; then
        return 0
    fi

    validate_umbp_distributed_config || return 1

    local host port check_host
    host="$(master_addr_host "$UMBP_MASTER_ADDRESS")"
    port="$(master_addr_port "$UMBP_MASTER_ADDRESS")"
    check_host="$host"
    [[ "$check_host" == "0.0.0.0" ]] && check_host="127.0.0.1"

    if wait_for_tcp "$check_host" "$port" 2 "UMBP master"; then
        log "Using existing UMBP master at ${UMBP_MASTER_ADDRESS}."
        return 0
    fi

    if ! bool_is_true "$UMBP_MASTER_AUTO_START"; then
        log "ERROR: UMBP master not reachable at ${UMBP_MASTER_ADDRESS} and auto-start disabled."
        return 1
    fi

    local mori_repo_root mori_build_dir master_bin master_listen master_log
    mori_repo_root="$(cd "${REPO_ROOT}/../mori" 2>/dev/null && pwd || true)"
    mori_build_dir="${MORI_BUILD_DIR:-${mori_repo_root}/build}"
    master_bin="${UMBP_MASTER_BIN:-${mori_build_dir}/src/umbp/umbp_master}"
    if [[ ! -x "$master_bin" ]]; then
        local packaged_master
        packaged_master="$(find_packaged_umbp_master)"
        [[ -n "$packaged_master" ]] && master_bin="$packaged_master"
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

    if ! wait_for_tcp "$check_host" "$port" 30 "UMBP master"; then
        log "ERROR: UMBP master failed to become ready. Check $master_log"
        kill_master
        return 1
    fi

    log "UMBP master is ready at ${UMBP_MASTER_ADDRESS}."
    return 0
}

SERVER_PID=""
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
    # Kill stale spdk_proxy processes and shared memory
    pkill -9 -f spdk_proxy 2>/dev/null || true
    rm -f /dev/shm/umbp_spdk* 2>/dev/null || true
    # Wait for server port to be released
    if [[ "$PD_ROLE" == "prefill" ]]; then
        wait_for_port_free "$PREFILL_PORT"
    else
        wait_for_port_free "$DECODE_PORT"
    fi
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

ROUTER_PID=""
kill_router() {
    if [[ -n "${ROUTER_PID:-}" ]] && kill -0 "$ROUTER_PID" 2>/dev/null; then
        log "Stopping router (PID $ROUTER_PID)..."
        kill -TERM "$ROUTER_PID" 2>/dev/null || true
        wait "$ROUTER_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$ROUTER_PID" 2>/dev/null || true
    fi
    ROUTER_PID=""
}

wait_for_port_free() {
    local port="$1"
    local max_wait=60 elapsed=0
    while (( elapsed < max_wait )); do
        local pids
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
        [[ -z "$pids" ]] && return 0
        if (( elapsed == 0 )); then
            log "Waiting for port $port to be released (PIDs: $pids)..."
        fi
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2; elapsed=$(( elapsed + 2 ))
    done
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log "ERROR: Port $port still occupied after ${max_wait}s (PIDs: $pids)"
        return 1
    fi
}

clean_ssd_dir() {
    for (( i=1; i<=5; i++ )); do
        rm -rf "$UMBP_SSD_DIR" 2>/dev/null && return 0
        log "Retrying SSD dir cleanup ($i/5)..."
        sleep 2
    done
    log "WARNING: Could not fully remove $UMBP_SSD_DIR, proceeding anyway."
}

wait_for_server() {
    local port="$1"
    local host="${2:-localhost}"
    local url="http://${host}:${port}/v1/models"
    local elapsed=0 interval=30
    log "Waiting for server at $url (timeout ${SERVER_READY_TIMEOUT}s)..."
    while (( elapsed < SERVER_READY_TIMEOUT )); do
        if curl -sf "$url" > /dev/null 2>&1; then
            log "Server is ready (took ${elapsed}s)."
            return 0
        fi
        sleep "$interval"; elapsed=$(( elapsed + interval ))
    done
    log "ERROR: Server did not become ready within ${SERVER_READY_TIMEOUT}s."
    return 1
}

run_benchmark() {
    local case_dir="$1" case_tag="$2" bench_port="$3"
    local log_file="${case_dir}/bench.log"
    local metrics_file="${case_dir}/performance_metrics.jsonl"

    log "Starting benchmark (port=${bench_port}, rounds=${NUM_ROUNDS}, clients=${NUM_CLIENTS}, max_parallel=${MAX_PARALLEL}, timeout=${BENCHMARK_TIMEOUT}s)..."
    timeout --signal=TERM --kill-after=30 "$BENCHMARK_TIMEOUT" \
        python "${SCRIPT_DIR}/bench_multiturn.py" \
            --model-path "$MODEL_PATH" \
            --port "$bench_port" \
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

# ---- Build UMBP extra config JSON ---------------------------
build_umbp_extra_config() {
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
        # Optional override for master's PageBitmapAllocator page_size.
        # When set, takes precedence over UMBPStore's auto-probe (Path A).
        [[ -n "${UMBP_DRAM_PAGE_SIZE:-}" ]] && \
            dist_fields+=", \"dram_page_size\": ${UMBP_DRAM_PAGE_SIZE}"
    fi
    # Auto-disable SSD when capacity is zero so UMBPConfig::Validate() does
    # not reject the config (ssd_enabled=true requires ssd_capacity_bytes>0).
    # Note: mori's dual-scheme DistributedClient does not currently use the
    # SSD tier at all (only DRAM is registered with the master), so setting
    # UMBP_SSD_BYTES=0 is the supported way to fully turn SSD off until SSD
    # support lands in dual-scheme.
    local ssd_enabled_json="true"
    if [[ "${UMBP_SSD_BYTES}" -le 0 ]]; then
        ssd_enabled_json="false"
    fi
    local kv_events_fields=""
    if bool_is_true "$ENABLE_KV_EVENTS"; then
        kv_events_fields=", \"kv_events_subscriber\": true"
    fi
    echo "{\"dram_capacity_bytes\": ${UMBP_DRAM_BYTES}, \"ssd_enabled\": ${ssd_enabled_json}, \"ssd_storage_dir\": \"${UMBP_SSD_DIR}\", \"ssd_capacity_bytes\": ${UMBP_SSD_BYTES}, \"auto_promote_on_read\": true, \"eviction_policy\": \"prefix_aware_lru\", \"ssd_durability_mode\": \"${UMBP_SSD_DURABILITY_MODE}\", \"copy_to_ssd_async\": ${UMBP_COPY_TO_SSD_ASYNC}, \"ssd_writer_threads\": ${UMBP_SSD_WRITER_THREADS}${spdk_fields}${dist_fields}${kv_events_fields}}"
}

# ---- Launch server (unified for both roles) -----------------
#
# Uses global variables for all config. Caller sets the role-specific
# host/port and passes the role as $1.
launch_pd_server() {
    local role="$1"
    local host="$2"
    local port="$3"

    # Common args
    local cmd=(
        python -m sglang.launch_server
        --enable-cache-report --enable-metrics
        --model-path "$MODEL_PATH"
        --host "$host"
        --port "$port"
        --tp-size "$TP_SIZE"
        --page-size "$PAGE_SIZE"
        --dp-size "$DP_SIZE"
        --enable-dp-attention
        --moe-a2a-backend "$MOE_A2A_BACKEND"
    )
    if [[ -n "$KV_CACHE_DTYPE" ]]; then
        cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    fi
    if [[ -n "$MEM_FRACTION_STATIC" ]]; then
        cmd+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
    fi

    # Disaggregation args
    cmd+=(
        --disaggregation-mode "$role"
        --disaggregation-transfer-backend "$DISAGG_TRANSFER_BACKEND"
    )
    if [[ "$role" == "prefill" ]]; then
        cmd+=(--disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT")
    fi
    if [[ "$role" == "decode" ]]; then
        cmd+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
    fi

    # Multi-node distributed args
    if [[ -n "$DIST_INIT_ADDR" ]]; then
        cmd+=(--dist-init-addr "$DIST_INIT_ADDR")
    fi
    if (( NNODES > 1 )); then
        cmd+=(--nnodes "$NNODES" --node-rank "$NODE_RANK")
    fi

    # Hierarchical cache / decode offload args
    if bool_is_true "$ENABLE_HICACHE"; then
        if [[ "$role" == "prefill" ]]; then
            # Prefill: full hierarchical cache (HiRadixCache)
            cmd+=(
                --enable-hierarchical-cache
                --hicache-size "$HICACHE_SIZE"
                --hicache-write-policy "$WRITE_POLICY"
                --hicache-mem-layout page_first
            )
        else
            # Decode: offload KV to host/storage via offload manager
            # (--enable-hierarchical-cache conflicts with decode's disable_radix_cache,
            #  so we use --disaggregation-decode-enable-offload-kvcache instead.
            #  The offload manager reads hicache-size/storage-backend from server_args.)
            cmd+=(
                --disaggregation-decode-enable-offload-kvcache
                --hicache-size "$HICACHE_SIZE"
                --hicache-mem-layout page_first
            )
        fi
        if bool_is_true "$ENABLE_UMBP"; then
            local extra_config
            extra_config="$(build_umbp_extra_config)"
            cmd+=(
                --hicache-storage-backend umbp
                --hicache-storage-backend-extra-config "$extra_config"
            )
        fi
    fi

    # Append any extra args forwarded from EXTRA_SERVER_ARGS env (safe under
    # `set -u` even when the array is empty).
    if bool_is_true "$USE_DUMMY_WEIGHTS"; then
        cmd+=(--load-format dummy)
    fi
    cmd+=(${EXTRA_SERVER_ARGS_ARR[@]+"${EXTRA_SERVER_ARGS_ARR[@]}"})

    if bool_is_true "$ENABLE_KV_EVENTS"; then
        cmd+=(--kv-events-config "{\"publisher\": \"${KV_EVENTS_PUBLISHER}\", \"endpoint\": \"${KV_EVENTS_ENDPOINT}\", \"topic\": \"${KV_EVENTS_TOPIC}\"}")
    fi

    "${cmd[@]}"
}

# ---- Determine cache tier label -----------------------------
cache_tier_label() {
    if bool_is_true "$ENABLE_UMBP"; then
        echo "HBM_DRAM_SSD"
    elif bool_is_true "$ENABLE_HICACHE"; then
        echo "HBM_DRAM"
    else
        echo "HBM_Only"
    fi
}

# ---- Main ---------------------------------------------------
trap 'kill_router; kill_server; kill_master; exit 130' INT TERM

CACHE_LABEL="$(cache_tier_label)"
CASE_TAG="pd_${PD_ROLE}_${CACHE_LABEL}"

mkdir -p "$RESULTS_DIR"
SUMMARY_FILE="${RESULTS_DIR}/summary.txt"

log "======================================================"
log "PD Disaggregation Benchmark (${PD_ROLE})"
log "  Model:       $MODEL_PATH"
log "  TP:          $TP_SIZE"
log "  DP:          $DP_SIZE"
log "  MoE A2A:     $MOE_A2A_BACKEND"
log "  KV dtype:    ${KV_CACHE_DTYPE:-auto}"
log "  Disagg mode: $PD_ROLE"
log "  Transfer:    $DISAGG_TRANSFER_BACKEND"
if [[ "$PD_ROLE" == "prefill" ]]; then
    log "  Bootstrap:   $DISAGG_BOOTSTRAP_PORT"
    log "  Bind:        ${PREFILL_HOST}:${PREFILL_PORT}"
else
    log "  Bind:        ${DECODE_HOST}:${DECODE_PORT}"
    log "  Prefill URL: ${PREFILL_URL:-<not set>}"
    log "  Router:      $(bool_is_true "$ENABLE_ROUTER" && echo "enabled (port $ROUTER_PORT)" || echo "disabled")"
    log "  Max running: $MAX_RUNNING_REQUESTS"
fi
log "  Cache tier:  $CACHE_LABEL"
if bool_is_true "$ENABLE_HICACHE"; then
    if [[ "$PD_ROLE" == "prefill" ]]; then
        log "  Cache mode:  hierarchical cache (HiRadixCache)"
    else
        log "  Cache mode:  decode offload (disaggregation-decode-enable-offload-kvcache)"
    fi
    log "  L2 size:     ${HICACHE_SIZE} GB/rank"
fi
if bool_is_true "$ENABLE_UMBP"; then
    log "  L3 DRAM:     $((UMBP_DRAM_BYTES / 1073741824)) GB/rank"
    log "  L3 SSD:      $((UMBP_SSD_BYTES / 1073741824)) GB/rank"
    log "  L3 SSD:      durability=${UMBP_SSD_DURABILITY_MODE}, async_copy=${UMBP_COPY_TO_SSD_ASYNC}, backend=${UMBP_SSD_BACKEND}"
    if [[ "$UMBP_SSD_BACKEND" != "posix" ]]; then
        log "  SPDK:        pci=${UMBP_SPDK_NVME_PCI:-auto}, auto_start=${UMBP_SPDK_PROXY_AUTO_START}"
    fi
    if [[ -n "$UMBP_MASTER_ADDRESS" ]]; then
        log "  UMBP dist:   master=${UMBP_MASTER_ADDRESS} node=${UMBP_NODE_ADDRESS:-auto} io_port=${UMBP_IO_ENGINE_PORT} peer_port=${UMBP_PEER_SERVICE_PORT} auto_master=${UMBP_MASTER_AUTO_START}"
    else
        log "  UMBP dist:   disabled (local-only UMBP)"
    fi
fi
if (( NNODES > 1 )); then
    log "  Dist init:   ${DIST_INIT_ADDR}"
    log "  Nodes:       ${NNODES} (rank ${NODE_RANK})"
fi
if [[ "$PD_ROLE" == "decode" ]]; then
    log "  Rounds:      $NUM_ROUNDS"
    log "  Clients:     $NUM_CLIENTS"
    log "  Max parallel: $MAX_PARALLEL"
    log "  Request len: $REQUEST_LENGTH"
    log "  Sub-q input: $SUB_QUESTION_INPUT_LENGTH"
    log "  Output len:  $OUTPUT_LENGTH"
    log "  Request rate: $REQUEST_RATE"
fi
log "  Results dir: $RESULTS_DIR"
log "======================================================"

{
    echo "PD Disaggregation Benchmark (${PD_ROLE})"
    echo "Started: $(date)"
    echo "Model: $MODEL_PATH  TP: $TP_SIZE  DP: $DP_SIZE  MoE: $MOE_A2A_BACKEND"
    echo "Disagg: mode=${PD_ROLE} transfer=${DISAGG_TRANSFER_BACKEND}"
    echo "Cache: ${CACHE_LABEL}"
    if bool_is_true "$ENABLE_HICACHE"; then
        echo "L2: ${HICACHE_SIZE} GB/rank"
    fi
    if bool_is_true "$ENABLE_UMBP"; then
        echo "L3 DRAM: $((UMBP_DRAM_BYTES / 1073741824)) GB/rank  L3 SSD: $((UMBP_SSD_BYTES / 1073741824)) GB/rank"
    fi
    if [[ "$PD_ROLE" == "decode" ]]; then
        echo "Rounds: $NUM_ROUNDS  Clients: $NUM_CLIENTS  Parallel: $MAX_PARALLEL  Rate: $REQUEST_RATE"
    fi
    echo ""
} > "$SUMMARY_FILE"

# ---- Cleanup before start ----------------------------------
kill_server

if bool_is_true "$ENABLE_UMBP"; then
    log "Cleaning UMBP SSD dir: $UMBP_SSD_DIR"
    clean_ssd_dir
    if ! ensure_umbp_master; then
        log "FATAL: UMBP master is not available."
        echo "FAILED (umbp master unavailable)" >> "$SUMMARY_FILE"
        exit 1
    fi
fi

# ---- Role: PREFILL ------------------------------------------
if [[ "$PD_ROLE" == "prefill" ]]; then
    SERVER_LOG="${RESULTS_DIR}/server_prefill.log"

    log "Launching prefill server..."
    launch_pd_server prefill "$PREFILL_HOST" "$PREFILL_PORT" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    log "Prefill server PID: $SERVER_PID"

    if ! wait_for_server "$PREFILL_PORT"; then
        log "FATAL: Prefill server failed to start. Check $SERVER_LOG"
        echo "FAILED (prefill server did not start)" >> "$SUMMARY_FILE"
        kill_server
        kill_master
        exit 1
    fi

    log "======================================================"
    log "Prefill server is ready at ${PREFILL_HOST}:${PREFILL_PORT}"
    log "Bootstrap port: ${DISAGG_BOOTSTRAP_PORT}"
    log "Waiting for decode node to drive the benchmark..."
    log "Press Ctrl-C to stop."
    log "======================================================"

    echo "Prefill server ready at $(date)" >> "$SUMMARY_FILE"

    # Wait until killed
    wait "$SERVER_PID" 2>/dev/null || true

    log "Prefill server exited."
    echo "Prefill server exited at $(date)" >> "$SUMMARY_FILE"
    kill_master
fi

# ---- Role: DECODE -------------------------------------------
if [[ "$PD_ROLE" == "decode" ]]; then
    CASE_DIR="${RESULTS_DIR}/${CASE_TAG}"
    mkdir -p "$CASE_DIR"
    SERVER_LOG="${CASE_DIR}/server_decode.log"

    # Validate PREFILL_URL
    if [[ -z "$PREFILL_URL" ]]; then
        log "ERROR: PREFILL_URL must be set for decode node (e.g. PREFILL_URL=http://10.0.0.1:30000)"
        exit 1
    fi

    # Extract prefill host/port from URL for TCP check
    PREFILL_CHECK_HOST="$(echo "$PREFILL_URL" | sed -E 's|https?://||; s|:[0-9]+$||; s|/.*||')"
    PREFILL_CHECK_PORT="$(echo "$PREFILL_URL" | sed -E 's|.*:([0-9]+).*|\1|')"

    log "Waiting for prefill server at ${PREFILL_CHECK_HOST}:${PREFILL_CHECK_PORT}..."
    if ! wait_for_tcp "$PREFILL_CHECK_HOST" "$PREFILL_CHECK_PORT" "$PREFILL_WAIT_TIMEOUT" "Prefill server"; then
        log "FATAL: Prefill server not reachable. Start prefill node first."
        echo "FAILED (prefill not reachable)" >> "$SUMMARY_FILE"
        kill_master
        exit 1
    fi
    log "Prefill server is reachable."

    log "Launching decode server..."
    launch_pd_server decode "$DECODE_HOST" "$DECODE_PORT" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    log "Decode server PID: $SERVER_PID"

    if ! wait_for_server "$DECODE_PORT"; then
        log "FATAL: Decode server failed to start. Check $SERVER_LOG"
        echo "FAILED (decode server did not start)" >> "$SUMMARY_FILE"
        kill_server
        kill_master
        exit 1
    fi

    # Determine benchmark target port
    BENCH_PORT="$DECODE_PORT"

    # Launch router if enabled
    if bool_is_true "$ENABLE_ROUTER"; then
        ROUTER_LOG="${CASE_DIR}/router.log"
        log "Launching router at ${ROUTER_HOST}:${ROUTER_PORT}..."
        # Health check tuning: default timeout=5s fails under high-concurrency
        # stress because prefill's uvicorn /health shares its event loop with
        # /generate (which is running 2048-token forwards).  Bump timeout to
        # 120s and failure threshold to 20 so legitimate slow health responses
        # aren't treated as the worker being dead.  Tunable via env.
        python -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill "$PREFILL_URL" "$DISAGG_BOOTSTRAP_PORT" \
            --decode "http://localhost:${DECODE_PORT}" \
            --host "$ROUTER_HOST" \
            --port "$ROUTER_PORT" \
            --health-check-timeout-secs "${ROUTER_HEALTH_TIMEOUT_SECS:-120}" \
            --health-check-interval-secs "${ROUTER_HEALTH_INTERVAL_SECS:-60}" \
            --health-failure-threshold "${ROUTER_HEALTH_FAILURE_THRESHOLD:-20}" \
            > "$ROUTER_LOG" 2>&1 &
        ROUTER_PID=$!
        log "Router PID: $ROUTER_PID"

        # Wait for router to be ready
        if ! wait_for_tcp "localhost" "$ROUTER_PORT" 120 "Router"; then
            log "WARNING: Router failed to start. Running benchmark against decode port directly."
            log "Check $ROUTER_LOG for details."
            kill_router
        else
            log "Router is ready."
            BENCH_PORT="$ROUTER_PORT"
        fi
    fi

    # Run benchmark
    CASE_START=$(date +%s)
    SERVER_CRASHED=false

    run_benchmark "$CASE_DIR" "$CASE_TAG" "$BENCH_PORT" &
    BENCH_PID=$!

    while kill -0 "$BENCH_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "ERROR: Decode server crashed during benchmark! Aborting..."
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
        log "ERROR: Benchmark aborted — decode server crashed (${CASE_ELAPSED}s)."
        echo "${CASE_TAG}: SERVER_CRASH (${CASE_ELAPSED}s)" >> "$SUMMARY_FILE"
    elif (( BENCH_RC == 0 )); then
        log "Benchmark completed in ${CASE_ELAPSED}s."
        echo "${CASE_TAG}: PASSED (${CASE_ELAPSED}s)" >> "$SUMMARY_FILE"
    elif (( BENCH_RC == 124 )); then
        log "WARNING: Benchmark timed out after ${BENCHMARK_TIMEOUT}s (ran ${CASE_ELAPSED}s)."
        echo "${CASE_TAG}: TIMEOUT (${CASE_ELAPSED}s)" >> "$SUMMARY_FILE"
    else
        log "WARNING: Benchmark exited with code ${BENCH_RC} (${CASE_ELAPSED}s)."
        echo "${CASE_TAG}: ERROR rc=${BENCH_RC} (${CASE_ELAPSED}s)" >> "$SUMMARY_FILE"
    fi

    # Cleanup
    kill_router
    kill_server

    if bool_is_true "$ENABLE_UMBP"; then
        log "Cleaning UMBP SSD dir after benchmark."
        clean_ssd_dir
    fi

    {
        echo ""
        echo "Finished: $(date)"
        echo "Total elapsed: ${CASE_ELAPSED}s"
    } >> "$SUMMARY_FILE"

    log "======================================================"
    log "Benchmark finished in ${CASE_ELAPSED}s."
    log "Results: $RESULTS_DIR"
    log "======================================================"
    cat "$SUMMARY_FILE"

    kill_master
fi
