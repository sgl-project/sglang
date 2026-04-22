#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

# Source (not bash) so that venv activation, $PIP_CMD, $CU_VERSION, $NVCC_VER, and
# $PIP_INSTALL_SUFFIX all propagate into this shell. Without sourcing, the subshell
# exits and this script would fall back to system Python.
#
# Note: any `exit N` or `set -e` trip inside the sourced script terminates *this*
# script too (bash runs sourced commands in the current shell, so `exit` is not
# caught by `if`/`||`). The real error message appears upstream in the log.
# shellcheck disable=SC1091
source scripts/ci/cuda/ci_install_dependency.sh

# In venv mode, PIP_CMD must be set by the sourced script. If it isn't, the
# source chain is broken and we'd silently fall back to system `pip` below —
# exactly the split-install bug the migration is meant to prevent.
if [ -z "${PIP_CMD:-}" ]; then
    echo "FATAL:PIP_CMD is unset after sourcing ci_install_dependency.sh"
    exit 1
fi

export GDRCOPY_HOME=/usr/src/gdrdrv-2.5.1/
export CUDA_HOME=/usr/local/cuda

GRACE_BLACKWELL=${GRACE_BLACKWELL:-0}
# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

if python3 -c "import deep_ep" >/dev/null 2>&1; then
    echo "deep_ep is already installed or importable. Skipping installation."
    exit 0
fi

# Install system dependencies
# Use fallback logic in case apt fails due to unrelated broken packages on the runner
DEEPEP_SYSTEM_DEPS="curl wget git sudo rdma-core infiniband-diags openssh-server perftest libibumad3 libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils libnl-3-200 libnl-route-3-200 librdmacm1 build-essential cmake"
apt-get install -y --no-install-recommends $DEEPEP_SYSTEM_DEPS || {
    echo "Warning: apt-get install failed, checking if required packages are available..."
    for pkg in $DEEPEP_SYSTEM_DEPS; do
        if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
            echo "ERROR: Required package $pkg is not installed and apt-get failed"
            exit 1
        fi
    done
    echo "All required packages are already installed, continuing..."
}

# Install GDRCopy
rm -rf /opt/gdrcopy && mkdir -p /opt/gdrcopy
cd /opt/gdrcopy
git clone https://github.com/NVIDIA/gdrcopy.git .
git checkout v2.5.1
apt-get update || true  # May fail due to unrelated broken packages
GDRCOPY_DEPS_1="nvidia-dkms-580"
GDRCOPY_DEPS_2="build-essential devscripts debhelper fakeroot pkg-config dkms"
GDRCOPY_DEPS_3="check libsubunit0 libsubunit-dev python3-venv"
for deps_group in "$GDRCOPY_DEPS_1" "$GDRCOPY_DEPS_2" "$GDRCOPY_DEPS_3"; do
    apt-get install -y --no-install-recommends $deps_group || {
        echo "Warning: apt-get install failed for '$deps_group', checking if packages are available..."
        for pkg in $deps_group; do
            if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
                echo "ERROR: Required package $pkg is not installed and apt-get failed"
                exit 1
            fi
        done
        echo "All required packages from '$deps_group' are already installed, continuing..."
    }
done
cd packages
CUDA=/usr/local/cuda ./build-deb-packages.sh
dpkg -i gdrdrv-dkms_*.deb
dpkg -i libgdrapi_*.deb
dpkg -i gdrcopy-tests_*.deb
dpkg -i gdrcopy_*.deb

# Set up library paths based on architecture
LIB_PATH="/usr/lib/$ARCH-linux-gnu"
if [ ! -e "$LIB_PATH/libmlx5.so" ]; then
    ln -s $LIB_PATH/libmlx5.so.1 $LIB_PATH/libmlx5.so
fi
apt-get update || true
apt-get install -y --no-install-recommends libfabric-dev || {
    if ! dpkg -l libfabric-dev 2>/dev/null | grep -q "^ii"; then
        echo "ERROR: Required package libfabric-dev is not installed and apt-get failed"
        exit 1
    fi
    echo "libfabric-dev is already installed, continuing..."
}

# Install DeepEP
DEEPEP_DIR=/root/.cache/deepep
rm -rf ${DEEPEP_DIR}
if [ "$GRACE_BLACKWELL" = "1" ]; then
    # We use Tom's DeepEP fork for GB200 for now, which supports fp4 dispatch.
    GRACE_BLACKWELL_DEEPEP_BRANCH=gb200_blog_part_2
    git clone https://github.com/fzyzcjy/DeepEP.git ${DEEPEP_DIR} && \
    pushd ${DEEPEP_DIR} && \
    git checkout ${GRACE_BLACKWELL_DEEPEP_BRANCH} && \
    sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
    popd
else
    git clone https://github.com/deepseek-ai/DeepEP.git ${DEEPEP_DIR} && \
    pushd ${DEEPEP_DIR} && \
    git checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee && \
    popd
fi

cd ${DEEPEP_DIR}
if [ "$GRACE_BLACKWELL" = "1" ]; then
    # Resolve the toolkit CUDA version. Preference order:
    #   1. $NVCC_VER inherited from the sourced ci_install_dependency.sh
    #      (both scripts agree on the detected value, no re-detection cost).
    #   2. Local `nvcc --version` (authoritative — container toolkit).
    #   3. `nvidia-smi` (host driver; last resort).
    if [ -n "${NVCC_VER:-}" ]; then
        CUDA_VERSION="$NVCC_VER"
    elif command -v nvcc >/dev/null 2>&1; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    else
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | head -n1 | awk '{print $9}' || true)
    fi
    if [ -z "${CUDA_VERSION:-}" ]; then
        echo "FATAL: could not determine CUDA toolkit version (NVCC_VER unset, nvcc missing, nvidia-smi empty)"
        exit 1
    fi
    if [ "$CUDA_VERSION" = "12.8" ]; then
        CHOSEN_TORCH_CUDA_ARCH_LIST='10.0'
    elif awk -v ver="$CUDA_VERSION" 'BEGIN {exit !(ver > 12.8)}'; then
        # CUDA > 12.8 supports sm_103 (Blackwell)
        CHOSEN_TORCH_CUDA_ARCH_LIST='10.0;10.3'
    else
        echo "Unsupported CUDA version for Grace Blackwell: $CUDA_VERSION" && exit 1
    fi && \
    if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py; \
    fi
    TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" ${PIP_CMD:-pip} install --no-build-isolation . ${PIP_INSTALL_SUFFIX:-}
else
    # CUDA 13.0 puts CCCL headers in /usr/local/cuda/include/cccl/ but nvshmem
    # includes them as <cuda/__cccl_config> expecting /usr/local/cuda/include/cuda/.
    # Add the cccl path to setup.py include_dirs so the compiler finds them.
    NVCC_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo "0")
    if [ "$NVCC_MAJOR" = "13" ]; then
        sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME:-/usr/local/cuda}/include/cccl')" setup.py
    fi
    python3 setup.py install
fi
