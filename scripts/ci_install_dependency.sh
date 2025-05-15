#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

# Update pip
pip install --upgrade pip

# Clean up existing installations
pip uninstall -y flashinfer flashinfer_python sgl-kernel sglang vllm
pip cache purge
rm -rf /root/.cache/flashinfer
rm -rf /usr/local/lib/python3.10/dist-packages/flashinfer*
rm -rf /usr/local/lib/python3.10/dist-packages/sgl_kernel*

# Install the main package
pip install -e "python[all]"

# Install additional dependencies
pip install transformers==4.51.0 timm torchaudio==2.6.0 sentence_transformers accelerate peft pandas datasets mooncake-transfer-engine==0.3.0

# For compiling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12

# For lmms_evals evaluating MMMU
git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e lmms-eval/

# Install FlashMLA for attention backend tests
pip install git+https://github.com/deepseek-ai/FlashMLA.git

# Install hf_xet
pip install huggingface_hub[hf_xet]
