#!/usr/bin/env bash
# Body of one launch_amd_ci_mi35x.sh job; runs on the allocated MI35x node.
#
# An 8-GPU job holds the node whole (--exclusive), which ensure_vram_clear.sh
# relies on: it stops every container on the host that holds a GPU. A 1-, 2- or
# 4-GPU job shares the node with others, so it skips that, runs in its own
# container, and passes only the render devices of the GPUs Spur assigned it.
#
# The checkout is copied to node-local scratch first. The CI container runs as
# root and writes into its /sglang-checkout bind mount, which a root-squashed
# NFS home would refuse.

exec 2>&1   # Spur's batch --output captures stdout only
set -euo pipefail

: "${CI_SRC:?}" "${CI_SUITE:?}" "${CI_ROCM_VERSION:?}" "${CI_NODE_SCRATCH:?}" "${CI_EXIT_FILE:?}"
CI_GPUS="${CI_GPUS:-8}"
JOB_ID="${SLURM_JOB_ID:-${SPUR_JOB_ID:-$$}}"
WS="$CI_NODE_SCRATCH/ci-$JOB_ID"

if (( CI_GPUS == 8 )); then
    export CI_CONTAINER_NAME=ci_sglang
else
    export CI_CONTAINER_NAME="ci_sglang_$JOB_ID"
fi

echo "[amd-ci-job] start on $(hostname) job=$JOB_ID suite=$CI_SUITE rocm=$CI_ROCM_VERSION gpus=$CI_GPUS"

finish() {
    local rc=$?
    # Root-owned files from the container can only be removed from inside it.
    docker exec "$CI_CONTAINER_NAME" bash -c 'rm -rf /sglang-checkout/* /sglang-checkout/.[!.]*' >/dev/null 2>&1 || true
    docker rm -f "$CI_CONTAINER_NAME" >/dev/null 2>&1 || true
    rm -rf "$WS" 2>/dev/null || true
    echo "$rc" > "$CI_EXIT_FILE"
    echo "[amd-ci-job] exit $rc"
}
trap finish EXIT

# GPU indices as the scheduler reports them (ROCm order) -> /dev/dri/renderD*.
# ROCm numbers GPUs in KFD topology order, skipping CPU nodes (gpu_id 0).
render_devices() {
    local topo=/sys/class/kfd/kfd/topology/nodes want=",$1," idx=0 node minor
    local out=()
    for node in $(ls "$topo" | sort -n); do
        [[ "$(cat "$topo/$node/gpu_id")" == 0 ]] && continue
        if [[ "$want" == *",$idx,"* ]]; then
            minor=$(awk '$1 == "drm_render_minor" {print $2}' "$topo/$node/properties")
            out+=("/dev/dri/renderD$minor")
        fi
        idx=$((idx + 1))
    done
    echo "${out[*]}"
}

mkdir -p "$WS" "$CI_NODE_SCRATCH/sglang-data"
cp -a "$CI_SRC/." "$WS/"
cd "$WS"

# The scripts read these instead of an ARC pod hostname and /home/runner.
export GITHUB_WORKSPACE="$WS"
export AMD_CI_GPU_ARCH=mi35x
export AMD_CI_CACHE_HOST="$CI_NODE_SCRATCH/sglang-data"

if (( CI_GPUS == 8 )); then
    bash scripts/ci/amd/ensure_vram_clear.sh rocm
else
    gpus="${SPUR_JOB_GPUS:-${ROCR_VISIBLE_DEVICES:-}}"
    AMD_CI_RENDER_DEVICES="$(render_devices "$gpus")"
    read -ra devs <<< "$AMD_CI_RENDER_DEVICES"
    if (( ${#devs[@]} != CI_GPUS )); then
        echo "ERROR: asked for $CI_GPUS GPUs, scheduler assigned '$gpus' -> '${AMD_CI_RENDER_DEVICES}'" >&2
        exit 1
    fi
    export AMD_CI_RENDER_DEVICES
    echo "[amd-ci-job] GPUs $gpus -> $AMD_CI_RENDER_DEVICES, container $CI_CONTAINER_NAME"
fi
bash scripts/ci/amd/amd_ci_start_container.sh --rocm-version "$CI_ROCM_VERSION"
bash scripts/ci/amd/amd_ci_install_dependency.sh
# --env pairs from the launcher, one KEY=VAL per line.
EXEC_ENV_ARGS=()
while IFS= read -r kv; do
    [[ -n "$kv" ]] && EXEC_ENV_ARGS+=(-e "$kv")
done <<< "${CI_EXEC_ENV:-}"
# shellcheck disable=SC2086  # CI_SUITE_ARGS is a list of run_suite.py flags
bash scripts/ci/amd/amd_ci_exec.sh -w /sglang-checkout/test ${EXEC_ENV_ARGS[@]+"${EXEC_ENV_ARGS[@]}"} \
    python3 run_suite.py --hw amd --suite "$CI_SUITE" $CI_SUITE_ARGS
