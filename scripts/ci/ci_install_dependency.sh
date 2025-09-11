#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

IS_BLACKWELL=${IS_BLACKWELL:-0}

if [ "$IS_BLACKWELL" = "1" ]; then
    CU_VERSION="cu129"
else
    CU_VERSION="cu126"
fi

# Clear torch compilation cache
python3 -c 'import os, shutil, tempfile, getpass; cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "torchinductor_" + getpass.getuser()); shutil.rmtree(cache_dir, ignore_errors=True)'

# Kill existing processes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/../killall_sglang.sh"

# Install apt packages
apt install -y git libnuma-dev

# Install uv
if [ "$IS_BLACKWELL" = "1" ]; then
    # The blackwell CI runner has some issues with pip and uv,
    # so we can only use pip with `--break-system-packages`
    PIP_CMD="pip"
    PIP_INSTALL_SUFFIX="--break-system-packages"

    # Clean up existing installations
    $PIP_CMD uninstall -y flashinfer_python sgl-kernel sglang vllm $PIP_INSTALL_SUFFIX || true
else
    # In normal cases, we use uv, which is much faster than pip.
    pip install --upgrade pip
    pip install uv
    export UV_SYSTEM_PYTHON=true

    PIP_CMD="uv pip"
    PIP_INSTALL_SUFFIX="--index-strategy unsafe-best-match"

    # Clean up existing installations
    $PIP_CMD uninstall flashinfer_python sgl-kernel sglang vllm || true
fi

# Install the main package
$PIP_CMD install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION} $PIP_INSTALL_SUFFIX

# Install router for pd-disagg test
SGLANG_ROUTER_BUILD_NO_RUST=1 $PIP_CMD install -e "sgl-router" $PIP_INSTALL_SUFFIX


SGL_KERNEL_VERSION=0.3.9.post2
if [ "$IS_BLACKWELL" = "1" ]; then
    # TODO auto determine sgl-kernel version
    $PIP_CMD install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu128-cp310-abi3-manylinux2014_x86_64.whl --force-reinstall $PIP_INSTALL_SUFFIX
else
    $PIP_CMD install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu124-cp310-abi3-manylinux2014_x86_64.whl --force-reinstall $PIP_INSTALL_SUFFIX
fi

# Show current packages
$PIP_CMD list

# Install additional dependencies
$PIP_CMD install mooncake-transfer-engine==0.3.5 nvidia-cuda-nvrtc-cu12 py-spy huggingface_hub[hf_xet] $PIP_INSTALL_SUFFIX

if [ "$IS_BLACKWELL" != "1" ]; then
    # For lmms_evals evaluating MMMU
    git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    $PIP_CMD install -e lmms-eval/ $PIP_INSTALL_SUFFIX

    # Install xformers
    $PIP_CMD install xformers --index-url https://download.pytorch.org/whl/${CU_VERSION} --no-deps $PIP_INSTALL_SUFFIX
fi

# Install FlashMLA for attention backend tests
# $PIP_CMD install git+https://github.com/deepseek-ai/FlashMLA.git $PIP_INSTALL_SUFFIX

# Show current packages
$PIP_CMD list

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
