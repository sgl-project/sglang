#!/bin/bash
set -euxo pipefail

# Install the dependency in CI.


# Use repo from environment variable, passed from GitHub Actions
FLASHINFER_REPO="${FLASHINFER_REPO:-https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

pip install --upgrade pip
pip uninstall flashinfer -y
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

rm -rf /root/.cache/flashinfer
# Force reinstall flashinfer and torch_memory_saver
pip install flashinfer_python==0.2.3 --find-links ${FLASHINFER_REPO} --force-reinstall --no-deps

pip install torch_memory_saver --force-reinstall

pip install "git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3" sentence_transformers accelerate==1.4.0 peft pandas datasets

# For compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12

# For DeepSeek-VL2
pip install timm
