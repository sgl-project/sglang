#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

IS_BLACKWELL=${IS_BLACKWELL:-0}
RUN_DEEPSEEK_V32=${RUN_DEEPSEEK_V32:-0}
CU_VERSION="cu128"

# Kill existing processes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/../killall_sglang.sh"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Clear torch compilation cache
python3 -c 'import os, shutil, tempfile, getpass; cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "torchinductor_" + getpass.getuser()); shutil.rmtree(cache_dir, ignore_errors=True)'

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
$PIP_CMD install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION} $PIP_INSTALL_SUFFIX --force-reinstall

# Install router for pd-disagg test
SGLANG_ROUTER_BUILD_NO_RUST=1 $PIP_CMD install -e "sgl-router" $PIP_INSTALL_SUFFIX

# Install sgl-kernel
SGL_KERNEL_VERSION_FROM_KERNEL=$(grep -Po '(?<=^version = ")[^"]*' sgl-kernel/pyproject.toml)
SGL_KERNEL_VERSION_FROM_SRT=$(grep -Po -m1 '(?<=sgl-kernel==)[0-9A-Za-z\.\-]+' python/pyproject.toml)
echo "SGL_KERNEL_VERSION_FROM_KERNEL=${SGL_KERNEL_VERSION_FROM_KERNEL} SGL_KERNEL_VERSION_FROM_SRT=${SGL_KERNEL_VERSION_FROM_SRT}"

if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ]; then
    ls -alh sgl-kernel/dist
    $PIP_CMD install sgl-kernel/dist/sgl_kernel-${SGL_KERNEL_VERSION_FROM_KERNEL}-cp310-abi3-manylinux2014_x86_64.whl --force-reinstall $PIP_INSTALL_SUFFIX
else
    $PIP_CMD install sgl-kernel==${SGL_KERNEL_VERSION_FROM_SRT} --force-reinstall $PIP_INSTALL_SUFFIX
fi

# Show current packages
$PIP_CMD list

# Install additional dependencies
$PIP_CMD install mooncake-transfer-engine==0.3.6.post1 nvidia-cuda-nvrtc-cu12 py-spy huggingface_hub[hf_xet] $PIP_INSTALL_SUFFIX

if [ "$IS_BLACKWELL" != "1" ]; then
    # For lmms_evals evaluating MMMU
    git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    $PIP_CMD install -e lmms-eval/ $PIP_INSTALL_SUFFIX

    # Install xformers
    $PIP_CMD install xformers --index-url https://download.pytorch.org/whl/${CU_VERSION} --no-deps $PIP_INSTALL_SUFFIX
fi

# Install dependencies for deepseek-v3.2
if [ "$RUN_DEEPSEEK_V32" = "1" ]; then
    # Install flashmla
    FLASHMLA_COMMIT="1408756a88e52a25196b759eaf8db89d2b51b5a1"
    FLASH_MLA_DISABLE_SM100="0"
    if [ "$IS_BLACKWELL" != "1" ]; then
        FLASH_MLA_DISABLE_SM100="1"
    fi
    git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
    cd flash-mla
    git checkout ${FLASHMLA_COMMIT}
    git submodule update --init --recursive
    FLASH_MLA_DISABLE_SM100=${FLASH_MLA_DISABLE_SM100} $PIP_CMD install -v . $PIP_INSTALL_SUFFIX --no-build-isolation
    cd ..

    # Install fast-hadamard-transform
    FAST_HADAMARD_TRANSFORM_COMMIT="7fd811c2b47f63b0b08d2582619f939e14dad77c"
    git clone https://github.com/Dao-AILab/fast-hadamard-transform
    cd fast-hadamard-transform
    git checkout ${FAST_HADAMARD_TRANSFORM_COMMIT}
    $PIP_CMD install . $PIP_INSTALL_SUFFIX --no-build-isolation
    cd ..
fi

# Show current packages
$PIP_CMD list
python3 -c "import torch; print(torch.version.cuda)"
