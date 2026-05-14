#!/bin/bash
set -euxo pipefail

source scripts/ci/cuda/ci_install_dependency.sh

if [ -z "${PIP_CMD:-}" ]; then
    echo "FATAL:PIP_CMD is unset after sourcing ci_install_dependency.sh"
    exit 1
fi

export GDRCOPY_HOME=/usr/src/gdrdrv-2.5.1/
export CUDA_HOME=/usr/local/cuda

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

###############################################################################
# Install FlashMLA
###############################################################################
INSTALL_FLASH_MLA=1
if [ "${FORCE_REBUILD_FLASH_MLA:-0}" = "1" ]; then
    echo "FORCE_REBUILD_FLASH_MLA=1; uninstalling any cached flash_mla before rebuild."
    ${PIP_UNINSTALL_CMD:-pip uninstall -y} flash_mla ${PIP_UNINSTALL_SUFFIX:-} || true
elif python3 -c "import flash_mla" >/dev/null 2>&1; then
    echo "flash_mla is already installed or importable. Skipping installation."
    INSTALL_FLASH_MLA=0
fi

if [ "$INSTALL_FLASH_MLA" = "1" ]; then
    # CUDA 13.0 puts CCCL headers under /usr/local/cuda/include/cccl/cuda but
    # FlashMLA's build expects them at /usr/local/cuda/include/cuda. Symlink so
    # the compiler finds them. Idempotent: skip if the link/dir already exists.
    if [ ! -e /usr/local/cuda/include/cuda ] && [ -d /usr/local/cuda/include/cccl/cuda ]; then
        ln -s /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda
    fi

    FLASH_MLA_DIR=/root/.cache/flash-mla
    rm -rf ${FLASH_MLA_DIR}
    git clone https://github.com/deepseek-ai/FlashMLA.git ${FLASH_MLA_DIR}
    pushd ${FLASH_MLA_DIR}
    git submodule update --init --recursive
    ${PIP_CMD:-pip} install --no-build-isolation -v . ${PIP_INSTALL_SUFFIX:-}
    popd
fi

###############################################################################
# Install DeepEP
###############################################################################
# Default to a forced rebuild so changes to TORCH_CUDA_ARCH_LIST or any other
# build-time input don't silently reuse a cached deep_ep from a prior run.
INSTALL_DEEPEP=1
if [ "${FORCE_REBUILD_DEEPEP:-1}" = "1" ]; then
    echo "FORCE_REBUILD_DEEPEP=1; uninstalling any cached deep_ep before rebuild."
    ${PIP_UNINSTALL_CMD:-pip uninstall -y} deep_ep ${PIP_UNINSTALL_SUFFIX:-} || true
elif python3 -c "import deep_ep" >/dev/null 2>&1; then
    echo "deep_ep is already installed or importable. Skipping installation."
    INSTALL_DEEPEP=0
fi

if [ "$INSTALL_DEEPEP" = "1" ]; then
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
    git clone https://github.com/deepseek-ai/DeepEP.git ${DEEPEP_DIR}
    pushd ${DEEPEP_DIR}
    git checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
    popd

    cd ${DEEPEP_DIR}
    # CUDA 13.0 puts CCCL headers in /usr/local/cuda/include/cccl/ but nvshmem
    # includes them as <cuda/__cccl_config> expecting /usr/local/cuda/include/cuda/.
    # Add the cccl path to setup.py include_dirs so the compiler finds them.
    NVCC_MAJOR=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo "0")
    if [ "$NVCC_MAJOR" = "13" ]; then
        sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME:-/usr/local/cuda}/include/cccl')" setup.py
    fi

    # Build for both Hopper (sm_90) and Blackwell (sm_100) so the same wheel
    # runs on H200 and B200 runners. Mirrors the CUDA-version-keyed list in
    # docker/Dockerfile's DeepEP build stage.
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
        CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0'
    elif awk -v ver="$CUDA_VERSION" 'BEGIN {exit !(ver > 12.8)}'; then
        # CUDA > 12.8 supports sm_103 (Blackwell)
        CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3'
    else
        CHOSEN_TORCH_CUDA_ARCH_LIST='9.0'
    fi
    TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" python3 setup.py install
fi
