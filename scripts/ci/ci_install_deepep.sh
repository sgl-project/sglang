#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

bash scripts/ci/ci_install_dependency.sh

export GDRCOPY_HOME=/usr/src/gdrdrv-2.5.1/
export NVSHMEM_DIR=/opt/nvshmem/install
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

if [ "$ARCH" = "x86_64" ]; then
    # Install Mooncake+EP
    curl -L https://cloud.tsinghua.edu.cn/f/c22ec766545e48bf99e8/?dl=1 -o mooncake_transfer_engine-0.3.6.post1+ep-cp310-cp310-manylinux_2_17_x86_64.manylinux_2_35_x86_64.whl
    UV_SYSTEM_PYTHON=true uv pip install mooncake_transfer_engine-0.3.6.post1+ep-cp310-cp310-manylinux_2_17_x86_64.manylinux_2_35_x86_64.whl
else
    echo "Skipping Mooncake+EP installation for ARM architecture"
fi

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
if [ "GRACE_BLACKWELL" = "1" ]; then
    CUDA_ARCH="90;100;103;120"
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
GRACE_BLACKWELL_DEEPEP_BRANCH=gb200_blog_part_2
CUDA_VERSION=12.9.1
rm -rf ${DEEPEP_DIR}
if [ "GRACE_BLACKWELL" = "1" ]; then
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
case "$CUDA_VERSION" in \
    12.6.1) \
    CHOSEN_TORCH_CUDA_ARCH_LIST='9.0' \
    ;; \
    12.8.1|12.9.1|13.0.1) \
    CHOSEN_TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' \
    ;; \
    *) \
    echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 \
    ;; \
esac && \
if [ "${CUDA_VERSION%%.*}" = "13" ]; then \
    sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py; \
fi
cd ${DEEPEP_DIR}
if [ "GRACE_BLACKWELL" = "1" ]; then
    NVSHMEM_DIR=/opt/nvshmem/install TORCH_CUDA_ARCH_LIST="${CHOSEN_TORCH_CUDA_ARCH_LIST}" pip install --no-build-isolation .
else
    python3 setup.py install
fi

# Verify configuration
echo "=== Verify NVSHMEM ==="
nvshmem-info -a
