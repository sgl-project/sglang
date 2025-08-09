#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

IS_BLACKWELL=${IS_BLACKWELL:-0}

if [ "$IS_BLACKWELL" = "1" ]; then
    CU_VERSION="cu129"
else
    CU_VERSION="cu126"
fi

# Kill existing processes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

# Install apt packages
apt install -y git libnuma-dev

# Install uv
if [ "$IS_BLACKWELL" = "1" ]; then
    # The blackwell CI runner has some issues with pip and uv,
    # so we can only use pip with `--break-system-packages`
    pip install --upgrade pip --break-system-packages
    PIP_CMD="pip --break-system-packages"
else
    pip install --upgrade pip
    pip install uv
    export UV_SYSTEM_PYTHON=true
    PIP_CMD="uv pip" # uv pip is not supported on blackwell CI runner
fi

# Clean up existing installations
$PIP_CMD uninstall flashinfer_python sgl-kernel sglang vllm || true

# Install the main package
$PIP_CMD install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION}  --index-strategy unsafe-best-match

if [ "$IS_BLACKWELL" = "1" ]; then
    # TODO auto determine sgl-kernel version
    SGL_KERNEL_VERSION=0.3.2
    uv pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}-cp39-abi3-manylinux2014_x86_64.whl --force-reinstall
fi

# Show current packages
$PIP_CMD list

# Install additional dependencies
$PIP_CMD install mooncake-transfer-engine==0.3.5 nvidia-cuda-nvrtc-cu12 py-spy huggingface_hub[hf_xet]

if [ "$IS_BLACKWELL" != "1" ]; then
    # For lmms_evals evaluating MMMU
    git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    $PIP_CMD install -e lmms-eval/

    # Install xformers
    $PIP_CMD install -U xformers --index-url https://download.pytorch.org/whl/${CU_VERSION} --no-deps
fi

# Install FlashMLA for attention backend tests
# $PIP_CMD install git+https://github.com/deepseek-ai/FlashMLA.git

# Show current packages
$PIP_CMD list
