#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

CU_VERSION="cu126"
if [ "$MODE_BLACKWELL" = "1" ]; then
    CU_VERSION="cu129"
fi

# Kill existing processes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

# Update pip
pip install --upgrade pip --break-system-packages

# Clean up existing installations
pip uninstall -y flashinfer flashinfer_python sgl-kernel sglang vllm || true
pip cache purge || true
rm -rf /root/.cache/flashinfer
rm -rf /usr/local/lib/python3.10/dist-packages/flashinfer*
rm -rf /usr/local/lib/python3.10/dist-packages/sgl_kernel*

# Install the main package
pip install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/test/${CU_VERSION} --break-system-packages

# Show current packages
pip list

# Install additional dependencies
pip install mooncake-transfer-engine==0.3.5 nvidia-cuda-nvrtc-cu12 --break-system-packages

# For lmms_evals evaluating MMMU
git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e lmms-eval/ --break-system-packages

# Install FlashMLA for attention backend tests
# pip install git+https://github.com/deepseek-ai/FlashMLA.git --break-system-packages

# Install hf_xet
pip install huggingface_hub[hf_xet] --break-system-packages

# Install xformers
pip install -U xformers --index-url https://download.pytorch.org/whl/${CU_VERSION} --no-deps --force-reinstall --break-system-packages

# To help dumping traces when timeout occurred
pip install py-spy --break-system-packages

# Show current packages
pip list
