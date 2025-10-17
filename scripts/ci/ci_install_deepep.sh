#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

bash scripts/ci/ci_install_dependency.sh

export GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/
export NVSHMEM_DIR=/opt/nvshmem/install
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

# Install Mooncake+EP
curl -L https://cloud.tsinghua.edu.cn/f/c22ec766545e48bf99e8/?dl=1 -o mooncake_transfer_engine-0.3.6.post1+ep-cp310-cp310-manylinux_2_17_x86_64.manylinux_2_35_x86_64.whl
UV_SYSTEM_PYTHON=true uv pip install mooncake_transfer_engine-0.3.6.post1+ep-cp310-cp310-manylinux_2_17_x86_64.manylinux_2_35_x86_64.whl

if python3 -c "import deep_ep" >/dev/null 2>&1; then
    echo "deep_ep is already installed or importable. Skipping installation."
    exit 0
fi

# Install system dependencies
apt install -y curl wget git sudo libibverbs-dev rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 build-essential cmake

# Install GDRCopy
rm -rf /opt/gdrcopy && mkdir -p /opt/gdrcopy
rm -rf /opt/nvshmem && mkdir -p /opt/nvshmem
cd /opt/gdrcopy
git clone https://github.com/NVIDIA/gdrcopy.git .
git checkout v2.4.4
apt update
apt install -y nvidia-dkms-535
apt install -y build-essential devscripts debhelper fakeroot pkg-config dkms
apt install -y check libsubunit0 libsubunit-dev python3-venv
cd packages
CUDA=/usr/local/cuda ./build-deb-packages.sh
dpkg -i gdrdrv-dkms_*.deb
dpkg -i libgdrapi_*.deb
dpkg -i gdrcopy-tests_*.deb
dpkg -i gdrcopy_*.deb

if [ ! -e "/usr/lib/x86_64-linux-gnu/libmlx5.so" ]; then
    ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so
fi
apt-get update && apt-get install -y libfabric-dev

# Install NVSHMEM
cd /opt/nvshmem
wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.3.9/source/nvshmem_src_cuda12-all-all-3.3.9.tar.gz
tar -xf nvshmem_src_cuda12-all-all-3.3.9.tar.gz
mv nvshmem_src nvshmem && cd nvshmem
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/opt/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90
cd build
make -j$(nproc) install

# Install DeepEP
rm -rf /root/.cache/deepep && git clone https://github.com/deepseek-ai/DeepEP.git /root/.cache/deepep && cd /root/.cache/deepep && git checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
cd /root/.cache/deepep && python3 setup.py install

# Verify configuration
echo "=== Verify NVSHMEM ==="
nvshmem-info -a
