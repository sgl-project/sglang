#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

# Kill existing processes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

# Update pip
pip install --upgrade pip

# Clean up existing installations
pip uninstall -y flashinfer flashinfer_python sgl-kernel sglang vllm || true
pip cache purge || true
rm -rf /root/.cache/flashinfer
rm -rf /usr/local/lib/python3.10/dist-packages/flashinfer*
rm -rf /usr/local/lib/python3.10/dist-packages/sgl_kernel*

# Install the main package
pip install -e "python[dev]"

# Show current packages
pip list

# Install additional dependencies
pip install mooncake-transfer-engine==0.3.2.post1 nvidia-cuda-nvrtc-cu12

# For lmms_evals evaluating MMMU
git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e lmms-eval/

# Install FlashMLA for attention backend tests
# pip install git+https://github.com/deepseek-ai/FlashMLA.git

# Install hf_xet
pip install huggingface_hub[hf_xet]

# Install xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu126 --no-deps --force-reinstall

# Show current packages
pip list
