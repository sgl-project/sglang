#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

IS_BLACKWELL=${IS_BLACKWELL:-0}

CU_VERSION="cu126"
if [ "$IS_BLACKWELL" = "1" ]; then
    CU_VERSION="cu129"
fi

# Kill existing processes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

# Install apt packages
apt install -y git libnuma-dev

# Install uv
pip install --upgrade pip
pip install uv
export UV_SYSTEM=true

# Clean up existing installations
uv pip uninstall flashinfer_python sgl-kernel sglang vllm || true

# Install the main package
uv pip install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION}

if [ "$IS_BLACKWELL" = "1" ]; then
    # TODO auto determine sgl-kernel version
    SGL_KERNEL_VERSION=0.3.2
    uv pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}-cp39-abi3-manylinux2014_x86_64.whl --force-reinstall
fi

# Show current packages
uv pip list

# Install additional dependencies
uv pip install mooncake-transfer-engine==0.3.5 nvidia-cuda-nvrtc-cu12 py-spy huggingface_hub[hf_xet]

if [ "$IS_BLACKWELL" != "1" ]; then
    # For lmms_evals evaluating MMMU
    git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    uv pip install -e lmms-eval/

    # Install xformers
    uv pip install -U xformers --index-url https://download.pytorch.org/whl/${CU_VERSION} --no-deps
fi

# Install FlashMLA for attention backend tests
# pip install git+https://github.com/deepseek-ai/FlashMLA.git

# Show current packages
uv pip list
