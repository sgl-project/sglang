#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

# Clean up existing installations
pip uninstall -y flashinfer flashinfer_python sgl-kernel sglang vllm || true
pip cache purge
rm -rf /root/.cache/flashinfer
rm -rf /usr/local/lib/python3.10/dist-packages/flashinfer*
rm -rf /usr/local/lib/python3.10/dist-packages/sgl_kernel*

# Update pip
pip install --upgrade pip

# Install sgl-kernel
pip install sgl-kernel==0.0.9.post2 --no-cache-dir

# Install the main package
pip install -e "python[all]"

# Install additional dependencies
pip install torch_memory_saver
pip install transformers==4.51.0 sentence_transformers accelerate peft pandas datasets timm torchaudio

# For compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12

# Install DeepEP dependencies
# Install CMake
wget https://github.com/Kitware/CMake/releases/download/v3.27.4/cmake-3.27.4-linux-x86_64.sh
chmod +x cmake-3.27.4-linux-x86_64.sh
./cmake-3.27.4-linux-x86_64.sh --skip-license --prefix=/usr/local
rm cmake-3.27.4-linux-x86_64.sh

# Install GDRCopy
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
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
cd ../..

# Install IBGDA dependencies
ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so
apt-get install -y libfabric-dev

# Install NVSHMEM
wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.2.5/source/nvshmem_src_3.2.5-1.txz
tar -xf nvshmem_src_3.2.5-1.txz
mv nvshmem_src nvshmem
cd nvshmem
git apply ../DeepEP/third-party/nvshmem.patch
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX=/sgl-workspace/nvshmem/install -DCMAKE_CUDA_ARCHITECTURES=90
cd build
make install -j
cd ../..

# Install DeepEP
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
NVSHMEM_DIR=/sgl-workspace/nvshmem/install python setup.py install
cd ..
