#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

bash scripts/ci/ci_install_dependency.sh

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
apt install -y curl wget git sudo rdma-core infiniband-diags openssh-server perftest libibumad3 libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils libnl-3-200 libnl-route-3-200 librdmacm1 build-essential cmake

# Install GDRCopy
rm -rf /opt/gdrcopy && mkdir -p /opt/gdrcopy
cd /opt/gdrcopy
git clone https://github.com/NVIDIA/gdrcopy.git .
git checkout v2.5.1
apt update
apt install -y nvidia-dkms-580
apt install -y build-essential devscripts debhelper fakeroot pkg-config dkms
apt install -y check libsubunit0 libsubunit-dev python3-venv
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
apt-get update && apt-get install -y libfabric-dev

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
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | head -n1 | awk '{print $9}')
    if [ "$CUDA_VERSION" = "12.8" ]; then
        CHOSEN_TORCH_CUDA_ARCH_LIST='10.0'
    elif awk -v ver="$CUDA_VERSION" 'BEGIN {exit !(ver > 12.8)}'; then
        # With cuda > 12.8, the compiler supports 10.3, so we should use
        # CHOSEN_TORCH_CUDA_ARCH_LIST='10.0;10.3'
        #
        # However, our CI machine has a weird setup and nvidia-smi reports wrong CUDA version in the container.
        # The container is actually cuda 12.8, but nvidia-smi reports 13.0, leading to compilation errors. so we
        # drop 10.3.
        CHOSEN_TORCH_CUDA_ARCH_LIST='10.0'
    else
        echo "Unsupported CUDA version for Grace Blackwell: $CUDA_VERSION" && exit 1
    fi && \
    if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py; \
    fi
    TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" pip install --no-build-isolation .
else
    python3 setup.py install
fi
