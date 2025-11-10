#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

bash scripts/ci/ci_install_dependency.sh

export GDRCOPY_HOME=/usr/src/gdrdrv-2.5.1/
export NVSHMEM_DIR=/opt/nvshmem/install
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

GRACE_BLACKWELL=${GRACE_BLACKWELL:-0}
# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# It seems GB200 ci runner preinstalls some wrong version of deep_ep, so we cannot rely on it.
if [ "$GRACE_BLACKWELL" != "1" ]; then
    if python3 -c "import deep_ep" >/dev/null 2>&1; then
        echo "deep_ep is already installed or importable. Skipping installation."
        exit 0
    fi
fi

# Install system dependencies
apt install -y curl wget git sudo libibverbs-dev rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 build-essential cmake

# Install GDRCopy
rm -rf /opt/gdrcopy && mkdir -p /opt/gdrcopy
rm -rf /opt/nvshmem && mkdir -p /opt/nvshmem
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

# Install NVSHMEM
cd /opt/nvshmem
wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.4.5/source/nvshmem_src_cuda12-all-all-3.4.5.tar.gz
tar -xf nvshmem_src_cuda12-all-all-3.4.5.tar.gz
mv nvshmem_src nvshmem && cd nvshmem
if [ "$GRACE_BLACKWELL" = "1" ]; then
    CUDA_ARCH="100;120"
else
    CUDA_ARCH="90"
fi
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/opt/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
cd build
make -j$(nproc) install

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
        CHOSEN_TORCH_CUDA_ARCH_LIST='10.0;10.3'
    else
        echo "Unsupported CUDA version for Grace Blackwell: $CUDA_VERSION" && exit 1
    fi && \
    if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py; \
    fi
    NVSHMEM_DIR=/opt/nvshmem/install TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" pip install --no-build-isolation .
else
    python3 setup.py install
fi

# Verify configuration
echo "=== Verify NVSHMEM ==="
nvshmem-info -a
