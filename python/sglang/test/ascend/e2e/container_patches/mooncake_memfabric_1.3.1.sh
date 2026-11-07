#!/bin/bash
# PATCH_SET: mooncake_memfabric_1.3.1
# PATCH_VERSION: 1.3.1
# TARGET_IMAGE: cann9.1.0-a3-B010
# REMOVE_AFTER: 2026-12-31
# SCOPE: apt,pip
#
# The image does not ship the transfer stack required by
# `--disaggregation-transfer-backend mooncake` with MC_USE_TENT=1:
#   * mooncake_transfer_engine_memfabric 0.3.14 is missing;
#   * memfabric_hybrid is pinned to 1.3.0, which predates the TENT protocol;
#   * ibverbs headers are needed by the mooncake transport at runtime.
#
# Remove this patch set once these wheels are baked into the released image
# (docker/npu.Dockerfile) and that image is used by the NPU workflows.

set -euo pipefail

PATCH_CACHE_DIR="${PATCH_CACHE_DIR:-/tmp/sglang_npu_patch_cache/mooncake_memfabric_1.3.1}"
mkdir -p "${PATCH_CACHE_DIR}"

MEMFABRIC_WHEEL_NAME="memfabric_hybrid-1.3.1-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl"
MEMFABRIC_WHEEL_URL="https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com/memfabric/1.3.1/${MEMFABRIC_WHEEL_NAME}"
MOONCAKE_WHEEL_NAME="mooncake_transfer_engine_memfabric-0.3.14-cp312-cp312-manylinux_2_35_aarch64.whl"
MOONCAKE_WHEEL_URL="https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com/memfabric/1.3.1/${MOONCAKE_WHEEL_NAME}"

fetch_wheel() {
    local url=$1
    local dest=$2
    if [ -f "${dest}" ]; then
        echo "Use cached wheel: ${dest}"
        return
    fi
    echo "Download wheel: ${url}"
    # Unique per pod, so an override that points PATCH_CACHE_DIR at shared
    # storage still cannot interleave writes between pods.
    local tmp="${dest}.part.${HOSTNAME:-$$}"
    rm -f "${tmp}"
    if command -v wget >/dev/null 2>&1; then
        wget -q -O "${tmp}" "${url}"
    else
        curl -sSL -o "${tmp}" "${url}"
    fi
    mv "${tmp}" "${dest}"
}

if dpkg -s libibverbs-dev >/dev/null 2>&1; then
    echo "libibverbs-dev already installed, skip apt."
else
    echo "Install libibverbs-dev"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y libibverbs-dev
fi

fetch_wheel "${MEMFABRIC_WHEEL_URL}" "${PATCH_CACHE_DIR}/${MEMFABRIC_WHEEL_NAME}"
fetch_wheel "${MOONCAKE_WHEEL_URL}" "${PATCH_CACHE_DIR}/${MOONCAKE_WHEEL_NAME}"

python -m pip install --no-cache-dir \
    "${PATCH_CACHE_DIR}/${MEMFABRIC_WHEEL_NAME}" \
    "${PATCH_CACHE_DIR}/${MOONCAKE_WHEEL_NAME}"

# find_spec only locates the package, so this does not depend on LD_LIBRARY_PATH
# being set up yet (that happens later, in the test launcher).
python - <<'PY'
import importlib.util

for name in ("memfabric_hybrid", "mooncake"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing package after patch: {name}")
PY
