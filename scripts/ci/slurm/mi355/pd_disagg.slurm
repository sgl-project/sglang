#!/bin/bash
# Multi-node SGLang PD-disaggregation Slurm job for AMD MI355 (mooncake + AITER).
#
# Submitted by launch_mi355.sh. Allocates (prefill_nodes + decode_nodes) MI355
# nodes and starts one Docker container per node via a single srun. Inside each
# container, scripts/ci/slurm/mi355/server_pd.sh decides — from the node rank —
# whether to run the prefill server, the decode server, and (on rank 0) the
# router + benchmark client.
#
# The cluster has no pyxis/enroot-srun integration, so containers are launched
# with `docker run` directly (matching the proven mia1 multi-node setup):
# host network/IPC, the full ROCm device set, and the RDMA NICs passed through.
#
# All tunables arrive via the JOB_ENV file written by launch_mi355.sh.

#SBATCH --job-name=mi355-pd-disagg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00

set -uo pipefail

: "${JOB_ENV:?JOB_ENV must be passed via --export}"
# shellcheck disable=SC1090
source "$JOB_ENV"

echo "=== mi355 PD-disagg job $SLURM_JOB_ID on $(hostname) ==="
echo "UTC: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
echo "nodelist: $SLURM_JOB_NODELIST"

NNODES=$((PREFILL_NODES + DECODE_NODES))
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_ARR=($NODES)
if [ "${#NODE_ARR[@]}" -lt "$NNODES" ]; then
    echo "FATAL: allocation has ${#NODE_ARR[@]} nodes, recipe needs $NNODES"
    exit 1
fi
MASTER_NODE="${NODE_ARR[0]}"

# Resolve each node's primary IP (used to wire the router across nodes).
IPS=()
for NODE in "${NODE_ARR[@]:0:$NNODES}"; do
    IP=$(srun --nodes=1 --ntasks=1 --nodelist="$NODE" bash -c 'ip route get 1.1.1.1' \
         | awk '/src/ {print $7; exit}')
    IPS+=("$IP")
done
IPADDRS="$(IFS=,; echo "${IPS[*]}")"
echo "node IPs: $IPADDRS  (master=$MASTER_NODE)"

# Per-node docker privilege detection (group membership varies by node).
export DOCKER_CMD_DETECT='if docker ps &>/dev/null; then DOCKER_CMD=docker; else DOCKER_CMD="sudo docker"; fi'

SANITIZED_USER=$(whoami | tr -c 'a-zA-Z0-9_.-' '_')
CONT_NAME="sgl_pd_${SANITIZED_USER}_${SLURM_JOB_ID}"

# Environment passed into every container. NODE_RANK is the per-task rank.
DOCKER_ENV=(
    -e NODE_RANK='$SLURM_PROCID'
    -e NNODES="$NNODES"
    -e PREFILL_NODES="$PREFILL_NODES"
    -e DECODE_NODES="$DECODE_NODES"
    -e PREFILL_WORKERS="$PREFILL_WORKERS"
    -e DECODE_WORKERS="$DECODE_WORKERS"
    -e IPADDRS="$IPADDRS"
    -e MODEL="$MODEL"
    -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
    -e PREFILL_TP="$PREFILL_TP"
    -e DECODE_TP="$DECODE_TP"
    -e ISL="$ISL"
    -e OSL="$OSL"
    -e RANDOM_RANGE_RATIO="$RANDOM_RANGE_RATIO"
    -e CONCURRENCIES="$CONCURRENCIES"
    -e COMMON_FLAGS="$COMMON_FLAGS"
    -e PREFILL_FLAGS="$PREFILL_FLAGS"
    -e DECODE_FLAGS="$DECODE_FLAGS"
    -e ROUTER_PORT="$ROUTER_PORT"
    -e PREFILL_SERVER_PORT="$PREFILL_SERVER_PORT"
    -e DECODE_SERVER_PORT="$DECODE_SERVER_PORT"
    -e PREFILL_BOOTSTRAP_PORT="$PREFILL_BOOTSTRAP_PORT"
    -e DECODE_BOOTSTRAP_PORT="$DECODE_BOOTSTRAP_PORT"
    -e HF_HUB_CACHE=/models
    -e HF_HUB_OFFLINE=1
    -e RESULTS_DIR=/results
    -e LOGS_DIR=/logs
)

LOGS_DIR="$(dirname "$RESULTS_DIR")/logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Single srun fans out to one task per node; SLURM_PROCID becomes NODE_RANK.
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
     --nodelist="$(IFS=,; echo "${NODE_ARR[*]:0:$NNODES}")" \
     --kill-on-bad-exit=1 --unbuffered \
     bash -lc '
set -uo pipefail
eval "$DOCKER_CMD_DETECT"
echo "[rank $SLURM_PROCID] host=$(hostname) DOCKER_CMD=$DOCKER_CMD"

# Idempotent pre-clean of any stale container with our name.
$DOCKER_CMD rm -f '"$CONT_NAME"' 2>/dev/null || true

exec $DOCKER_CMD run --rm --init --stop-timeout 10 \
    --name '"$CONT_NAME"' \
    --network host --ipc host \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 128G \
    -v /sys:/sys \
    -v '"$MODEL_CACHE_DIR"':/models \
    -v '"$REPO_DIR"':/workspace \
    -v '"$RESULTS_DIR"':/results \
    -v '"$LOGS_DIR"':/logs \
    '"${DOCKER_ENV[*]}"' \
    --entrypoint "" \
    '"$IMAGE"' \
    bash -lc "/workspace/scripts/ci/slurm/mi355/server_pd.sh"
'
RC=$?
echo "=== job $SLURM_JOB_ID finished (rc=$RC) ==="
exit $RC
