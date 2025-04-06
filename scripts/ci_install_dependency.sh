#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

# Use repo from environment variables, passed from GitHub Actions
FLASHINFER_REPO="${FLASHINFER_REPO:-https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python}"

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

# Install flashinfer and sgl-kernel
pip install flashinfer_python==0.2.3 --find-links ${FLASHINFER_REPO} --no-cache-dir
pip install sgl-kernel==0.0.8 --no-cache-dir

# Install the main package
pip install -e "python[all]" --find-links ${FLASHINFER_REPO}

# Install additional dependencies
pip install torch_memory_saver
pip install transformers==4.51.0 sentence_transformers accelerate==1.4.0 peft pandas datasets timm torchaudio

# For compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12
