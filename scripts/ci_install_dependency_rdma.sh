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
pip install sgl-kernel==0.1.1 --no-cache-dir

# Install the main package
pip install -e "python[all]"

# Install additional dependencies
pip install torch_memory_saver
pip install transformers==4.51.0 sentence_transformers accelerate peft pandas datasets timm torchaudio==2.6.0
pip install mooncake-transfer-engine

# For compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12

# For lmms_evals evaluating MMMU
git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e lmms-eval/

# Install FlashMLA for attention backend tests
pip install git+https://github.com/deepseek-ai/FlashMLA.git

# Check if sudo is available, and install disaggregation requirement
export DEBIAN_FRONTEND=noninteractive
echo "tzdata tzdata/Areas select Etc" | sudo debconf-set-selections
echo "tzdata tzdata/Zones/Etc select UTC" | sudo debconf-set-selections

if command -v sudo >/dev/null 2>&1; then
    sudo apt update
    sudo apt install curl git sudo libibverbs-dev -y
    sudo apt install rdma-core infiniband-diags openssh-server pciutils perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 -y
fi
