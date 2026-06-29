#!/usr/bin/env bash
# setup_container.sh - bring $CONTAINER to running state on each target node.
#
# Default policy (conservative):
#   - if container is running         -> noop (PASS)
#   - if container exists but stopped -> docker start
#   - if container does not exist     -> FAIL with clear message
#
# With SETUP_CREATE=1 a missing container is created via the canonical
# `docker run` recipe (explicit ROCm/RDMA devices + caps, least-privilege).
#
# Required env: PREFILL_NODE  DECODE_NODE  CONTAINER  (+ optional ROUTER_NODE)
# SETUP_CREATE=1 also requires: IMAGE  (sglang+rocm image, e.g.
#   rocm/sgl-dev:v0.5.13.post1-rocm700-mi35x-20260617)
# Optional (create path): SHM_SIZE (default 32g), DATA_DIR (default /data),
#   REMOTE_WORKDIR (created inside container if set).

set -uo pipefail

: "${PREFILL_NODE:?}"
: "${DECODE_NODE:?}"
: "${CONTAINER:?}"
ROUTER_NODE="${ROUTER_NODE:-$PREFILL_NODE}"

SETUP_CREATE="${SETUP_CREATE:-0}"
SHM_SIZE="${SHM_SIZE:-32g}"
DATA_DIR="${DATA_DIR:-/data}"
if [ "$SETUP_CREATE" = "1" ]; then
    : "${IMAGE:?SETUP_CREATE=1 requires IMAGE (sglang+rocm image)}"
fi

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes)
FAILS=0

create_container() {
    local tgt="$1"
    echo "[INFO] $tgt: creating container '${CONTAINER}' from ${IMAGE}"
    if ssh "${SSH_OPTS[@]}" "$tgt" "\
        docker run -d --name ${CONTAINER} \
          --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
          --network=host --ipc=host --shm-size=${SHM_SIZE} \
          --ulimit memlock=-1 --ulimit stack=67108864 \
          --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
          --security-opt seccomp=unconfined \
          --group-add video --group-add render \
          -v ${DATA_DIR}:${DATA_DIR} \
          ${IMAGE} sleep infinity" >/dev/null; then
        if [ -n "${REMOTE_WORKDIR:-}" ]; then
            ssh "${SSH_OPTS[@]}" "$tgt" \
                "docker exec ${CONTAINER} mkdir -p ${REMOTE_WORKDIR}" >/dev/null || true
        fi
        local state
        state=$(ssh "${SSH_OPTS[@]}" "$tgt" "docker inspect -f '{{.State.Status}}' ${CONTAINER}")
        if [ "$state" = "running" ]; then
            echo "[PASS] $tgt: container created and running"
            return 0
        fi
        echo "[FAIL] $tgt: docker run completed but state=$state"
    else
        echo "[FAIL] $tgt: docker run failed"
    fi
    FAILS=$((FAILS+1))
    return 1
}

ensure_running() {
    local tgt="$1"
    local state
    state=$(ssh "${SSH_OPTS[@]}" "$tgt" \
        "docker inspect -f '{{.State.Status}}' ${CONTAINER} 2>/dev/null" || true)

    if [ -z "$state" ]; then
        if [ "$SETUP_CREATE" = "1" ]; then
            create_container "$tgt"
            return $?
        fi
        cat >&2 <<EOF
[FAIL] $tgt: container '${CONTAINER}' does not exist.
       setup_container.sh is conservative by default and will not create it.
       Either:
         (a) launch it manually with the canonical docker run command, OR
         (b) re-run with SETUP_CREATE=1 IMAGE=<sglang+rocm image>
             to auto-create it.
EOF
        FAILS=$((FAILS+1))
        return 1
    fi

    case "$state" in
        running)
            echo "[PASS] $tgt: container '${CONTAINER}' already running"
            ;;
        exited|created|paused)
            echo "[INFO] $tgt: container '${CONTAINER}' state=$state -> docker start"
            if ssh "${SSH_OPTS[@]}" "$tgt" "docker start ${CONTAINER}" >/dev/null; then
                # confirm
                state=$(ssh "${SSH_OPTS[@]}" "$tgt" "docker inspect -f '{{.State.Status}}' ${CONTAINER}")
                if [ "$state" = "running" ]; then
                    echo "[PASS] $tgt: container now running"
                else
                    echo "[FAIL] $tgt: docker start completed but state=$state"
                    FAILS=$((FAILS+1))
                fi
            else
                echo "[FAIL] $tgt: docker start failed"
                FAILS=$((FAILS+1))
            fi
            ;;
        *)
            echo "[FAIL] $tgt: container in unexpected state '$state'"
            FAILS=$((FAILS+1))
            ;;
    esac
}

# De-dup target list (router often == prefill)
declare -A SEEN
for tgt in "$PREFILL_NODE" "$DECODE_NODE" "$ROUTER_NODE"; do
    [ -n "${SEEN[$tgt]:-}" ] && continue
    SEEN[$tgt]=1
    ensure_running "$tgt"
done

if [ "$FAILS" -gt 0 ]; then
    echo "setup_container FAILED on $FAILS node(s)"
    exit 1
fi
echo "setup_container OK"
