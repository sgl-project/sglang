#!/usr/bin/env bash
# Body of one launch_amd_ci_mi35x.sh job; runs on the allocated MI35x node.
#
# Runs the same steps as a linux-mi35x-gpu-* job in pr-test-amd.yml. The node
# is held whole (--exclusive), which ensure_vram_clear.sh relies on: it stops
# every container on the host that holds a GPU.
#
# The checkout is copied to node-local scratch first. The CI container runs as
# root and writes into its /sglang-checkout bind mount, which a root-squashed
# NFS home would refuse.

exec 2>&1   # Spur's batch --output captures stdout only
set -euo pipefail

: "${CI_SRC:?}" "${CI_SUITE:?}" "${CI_ROCM_VERSION:?}" "${CI_NODE_SCRATCH:?}" "${CI_EXIT_FILE:?}"
JOB_ID="${SLURM_JOB_ID:-${SPUR_JOB_ID:-$$}}"
WS="$CI_NODE_SCRATCH/ci-$JOB_ID"

echo "[amd-ci-job] start on $(hostname) job=$JOB_ID suite=$CI_SUITE rocm=$CI_ROCM_VERSION"

finish() {
    local rc=$?
    # Root-owned files from the container can only be removed from inside it.
    docker exec ci_sglang bash -c 'rm -rf /sglang-checkout/* /sglang-checkout/.[!.]*' >/dev/null 2>&1 || true
    docker rm -f ci_sglang >/dev/null 2>&1 || true
    rm -rf "$WS" 2>/dev/null || true
    echo "$rc" > "$CI_EXIT_FILE"
    echo "[amd-ci-job] exit $rc"
}
trap finish EXIT

mkdir -p "$WS" "$CI_NODE_SCRATCH/sglang-data"
cp -a "$CI_SRC/." "$WS/"
cd "$WS"

# The scripts read these instead of an ARC pod hostname and /home/runner.
export GITHUB_WORKSPACE="$WS"
export AMD_CI_GPU_ARCH=mi35x
export AMD_CI_CACHE_HOST="$CI_NODE_SCRATCH/sglang-data"

bash scripts/ci/amd/ensure_vram_clear.sh rocm
bash scripts/ci/amd/amd_ci_start_container.sh --rocm-version "$CI_ROCM_VERSION"
bash scripts/ci/amd/amd_ci_install_dependency.sh
# shellcheck disable=SC2086  # CI_SUITE_ARGS is a list of run_suite.py flags
bash scripts/ci/amd/amd_ci_exec.sh -w /sglang-checkout/test \
    python3 run_suite.py --hw amd --suite "$CI_SUITE" $CI_SUITE_ARGS
