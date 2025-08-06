#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

MODE_BLACKWELL=${MODE_BLACKWELL:-0}

CU_VERSION="cu126"
if [ "$MODE_BLACKWELL" = "1" ]; then
    CU_VERSION="cu129"
fi

# Kill existing processes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

if ! command -v git >/dev/null 2>&1; then
    apt update
    apt install -y git
fi

# Update pip
if [ "$MODE_BLACKWELL" != "1" ]; then
    pip install --upgrade pip --break-system-packages
fi

# Clean up existing installations
pip uninstall -y flashinfer flashinfer_python sgl-kernel sglang vllm --break-system-packages || true
pip cache purge || true
rm -rf /root/.cache/flashinfer
# TODO handle other python versions
rm -rf /usr/local/lib/python3.10/dist-packages/flashinfer*
rm -rf /usr/local/lib/python3.10/dist-packages/sgl_kernel*

# Install the main package
pip install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/test/${CU_VERSION} --break-system-packages

# Show current packages
pip list

# Install additional dependencies
pip install mooncake-transfer-engine==0.3.5 nvidia-cuda-nvrtc-cu12 --break-system-packages

if [ "$MODE_BLACKWELL" != "1" ]; then
    # For lmms_evals evaluating MMMU
    git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    pip install -e lmms-eval/ --break-system-packages
fi

# Install FlashMLA for attention backend tests
# pip install git+https://github.com/deepseek-ai/FlashMLA.git --break-system-packages

# Install hf_xet
pip install huggingface_hub[hf_xet] --break-system-packages

if [ "$MODE_BLACKWELL" != "1" ]; then
    # Install xformers
    pip install -U xformers --index-url https://download.pytorch.org/whl/${CU_VERSION} --no-deps --force-reinstall --break-system-packages
fi

# To help dumping traces when timeout occurred
pip install py-spy --break-system-packages

# Show current packages
pip list
