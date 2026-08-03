#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: NODE_LIST=node-a,node-b $0 <arm>" >&2
    echo "Arms: vmm-poc, vmm-access-probe, ipc-poc, aiter-multib, synthetic-2, synthetic-4, synthetic-8, hsa-benchmark, model-smoke, standalone, pd, benchmark" >&2
    exit 2
fi

ARM="$1"
BAD_NODE="smci355-ccs-aus-n08-21"
NODE_LIST="${NODE_LIST:?NODE_LIST must be a fixed comma-separated MI355X node pool}"
SLURM_PARTITION="${SLURM_PARTITION:-Compute-Group01}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-$USER}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
RESULT_ROOT="${RESULT_ROOT:-/home/yanfwang/workspace/dwdp_rocm_slurm_results}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}"
MAX_USED_VRAM_GIB="${MAX_USED_VRAM_GIB:-4}"
MAX_VRAM_SKEW_GIB="${MAX_VRAM_SKEW_GIB:-2}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VRAM_CHECK_SCRIPT="${SCRIPT_DIR}/check_dwdp_rocm_vram.py"

# Keep the known-bad host out of both the candidate pool and Slurm allocation.
IFS=',' read -r -a _nodes <<< "$NODE_LIST"
_safe_nodes=()
for _node in "${_nodes[@]}"; do
    _node="${_node//[[:space:]]/}"
    [[ -z "$_node" || "$_node" == "$BAD_NODE" ]] && continue
    if [[ ! "$_node" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Invalid node name: $_node" >&2
        exit 2
    fi
    _safe_nodes+=("$_node")
done
if [[ "${#_safe_nodes[@]}" -lt 1 ]]; then
    echo "NODE_LIST has no usable nodes after excluding $BAD_NODE" >&2
    exit 2
fi

# Slurm can report a node as idle while stale allocations still occupy VRAM.
# Query memory only (never processes) and submit exclusively to clean nodes.
_vram_clean_nodes=()
for _node in "${_safe_nodes[@]}"; do
    _state="$(sinfo -h -p "$SLURM_PARTITION" -n "$_node" -o "%T" | tr -d '[:space:]')"
    if [[ "$_state" != idle ]]; then
        echo "Skipping $_node because Slurm state is ${_state:-unknown}, not idle" >&2
        continue
    fi
    echo "Checking VRAM on $_node before submission"
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$_node" \
        python3 "$VRAM_CHECK_SCRIPT" \
        --expected-gpus "$EXPECTED_GPU_COUNT" \
        --max-used-gib "$MAX_USED_VRAM_GIB" \
        --max-skew-gib "$MAX_VRAM_SKEW_GIB"; then
        _vram_clean_nodes+=("$_node")
    else
        echo "Skipping $_node because its VRAM preflight failed" >&2
    fi
done
if [[ "${#_vram_clean_nodes[@]}" -lt 1 ]]; then
    echo "No candidate node passed the VRAM preflight; no Slurm job was submitted" >&2
    exit 3
fi
SAFE_NODE_LIST=$(IFS=,; echo "${_vram_clean_nodes[*]}")

case ",${SLURM_EXCLUDE_NODES:-}," in
    *",${BAD_NODE},"*) ;;
    ",,") export SLURM_EXCLUDE_NODES="$BAD_NODE" ;;
    *) export SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES},${BAD_NODE}" ;;
esac
export DWDP_TEST_ARM="$ARM"
export RESULT_ROOT
export EXPECTED_GPU_COUNT
export MAX_USED_VRAM_GIB
export MAX_VRAM_SKEW_GIB

mkdir -p "$RESULT_ROOT"
timestamp=$(date +%Y%m%d-%H%M%S)
log_prefix="${RESULT_ROOT}/dwdp-${ARM}-${timestamp}"

job_id=$(
    sbatch \
        --parsable \
        --exclusive \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-node=8 \
        --account="$SLURM_ACCOUNT" \
        --partition="$SLURM_PARTITION" \
        --time="$TIME_LIMIT" \
        --nodelist="$SAFE_NODE_LIST" \
        --exclude="$SLURM_EXCLUDE_NODES" \
        --output="${log_prefix}-%j.out" \
        --error="${log_prefix}-%j.err" \
        --export=ALL \
        "$(dirname "$0")/dwdp_rocm_job.slurm"
)

job_desc=$(scontrol show job -o "$job_id")
echo "$job_desc"
allocated_nodes=$(squeue --noheader --job "$job_id" --format="%N" | tr -d '[:space:]')
if [[ "$allocated_nodes" == "$BAD_NODE" ]]; then
    scancel "$job_id"
    echo "Refusing allocation on excluded node $BAD_NODE; cancelled job $job_id" >&2
    exit 1
fi

echo "Submitted DWDP arm '$ARM' as Slurm job $job_id"
echo "Candidate nodes: $SAFE_NODE_LIST"
echo "Excluded nodes: $SLURM_EXCLUDE_NODES"
echo "Logs: ${log_prefix}-${job_id}.{out,err}"
