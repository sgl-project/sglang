#!/bin/bash
set -euxo pipefail

# Install the dependency in CI.

# Use repo from environment variable, passed from GitHub Actions
FLASHINFER_REPO="${FLASHINFER_REPO:-https://flashinfer.ai/whl/cu124/torch2.5/flashinfer}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

pip install --upgrade pip
pip uninstall flashinfer -y
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/

# Force reinstall flashinfer and torch_memory_saver
pip install https://github.com/flashinfer-ai/flashinfer-nightly/releases/download/0.2.1%2Bdbb1e4e/flashinfer_python-0.2.1+dbb1e4e.cu124torch2.5-cp310-cp310-linux_x86_64.whl --force-reinstall --no-deps

pip install torch_memory_saver --force-reinstall

pip install transformers==4.45.2 sentence_transformers accelerate peft

# For compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12

# reinstall sgl-kernel
pip install sgl-kernel --force-reinstall --no-deps
