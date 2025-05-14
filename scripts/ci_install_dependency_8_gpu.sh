#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

export GDRCOPY_HOME=/usr/src/gdrdrv-2.4.4/
export NVSHMEM_DIR=/opt/nvshmem/install
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

# Clean up existing installations
pip uninstall -y flashinfer flashinfer_python sgl-kernel sglang vllm deepep || true
pip cache purge
rm -rf /root/.cache/flashinfer
if [ -d "lmms-eval" ]; then
    rm -rf lmms-eval
fi
rm -rf /root/.cache/deepep
rm -rf /usr/local/lib/python3.10/dist-packages/flashinfer*
rm -rf /usr/local/lib/python3.10/dist-packages/sgl_kernel*
rm -rf /usr/local/lib/python3.10/dist-packages/deepep*
dpkg -r gdrcopy gdrcopy-tests libgdrapi gdrdrv-dkms || true
rm -rf /opt/gdrcopy
rm -rf /usr/local/lib/libgdrapi*
rm -rf /usr/local/include/gdrapi.h
rm -rf /opt/nvshmem
rm -rf /usr/local/lib/libnvshmem*
rm -rf /usr/local/include/nvshmem*

# Update pip
pip install --upgrade pip

# Install sgl-kernel
pip install sgl-kernel==0.1.2.post1 --no-cache-dir

# Install the main package
pip install -e "python[all]"

# Install additional dependencies
pip install torch_memory_saver
pip install transformers==4.51.0 sentence_transformers accelerate peft pandas datasets timm torchaudio==2.6.0

# For compiling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12

# For lmms_evals evaluating MMMU
git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e lmms-eval/

# Install FlashMLA for attention backend tests
pip install git+https://github.com/deepseek-ai/FlashMLA.git

# Install mooncake-transfer-engine
pip install mooncake-transfer-engine

# Install system dependencies
# apt-get update && apt-get install -y libibverbs-dev infiniband-diags libmlx5-1 rdma-core openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 rdma-core-dev infiniband-diags-dev libibverbs-dev libibverbs-utils librdmacm-dev librdmacm-utils ibverbs-utils rdma-core-utils
apt install curl wget git sudo libibverbs-dev -y
apt install -y rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py

wget https://github.com/Kitware/CMake/releases/download/v3.27.4/cmake-3.27.4-linux-x86_64.sh
chmod +x cmake-3.27.4-linux-x86_64.sh
./cmake-3.27.4-linux-x86_64.sh --skip-license --prefix=/usr/local
rm cmake-3.27.4-linux-x86_64.sh

# Install GDRCopy
mkdir -p /opt/gdrcopy
mkdir -p /opt/nvshmem
cd /opt/gdrcopy
git clone https://github.com/NVIDIA/gdrcopy.git .
git checkout v2.4.4
apt update
apt install -y nvidia-dkms-535
apt install -y build-essential devscripts debhelper fakeroot pkg-config dkms
apt install -y check libsubunit0 libsubunit-dev
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

# Clone DeepEP
git clone https://github.com/deepseek-ai/DeepEP.git /root/.cache/deepep

# Install NVSHMEM
cd /opt/nvshmem
wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
tar -xf nvshmem_src_3.2.5-1.txz
mv nvshmem_src nvshmem
cd nvshmem
git apply /root/.cache/deepep/third-party/nvshmem.patch
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
cd /root/.cache/deepep && python3 setup.py install

# Verify configuration
echo "=== NCCL Configuration ==="
nvidia-smi topo -m
nvidia-smi nvlink -s
echo "=== Verify GDRCOPY ==="
gdrcopy_copybw
echo "=== Verify NVSHMEM ==="
nvshmem-info -a
# /opt/nvshmem/bin/perftest/device/pt-to-pt/shmem_put_bw

# Install hf_xet
pip install huggingface_hub[hf_xet]
