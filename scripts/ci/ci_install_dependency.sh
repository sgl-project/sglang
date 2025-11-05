#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

IS_BLACKWELL=${IS_BLACKWELL:-0}
CU_VERSION="cu129"

# Detect system architecture
ARCH=$(uname -m)
echo "Detected architecture: ${ARCH}"

if [ "$CU_VERSION" = "cu130" ]; then
    NVRTC_SPEC="nvidia-cuda-nvrtc"
else
    NVRTC_SPEC="nvidia-cuda-nvrtc-cu12"
fi

# Kill existing processes
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/../killall_sglang.sh"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Clear torch compilation cache
python3 -c 'import os, shutil, tempfile, getpass; cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "torchinductor_" + getpass.getuser()); shutil.rmtree(cache_dir, ignore_errors=True)'
rm -rf /root/.cache/flashinfer
pip3 uninstall flashinfer-python flashinfer-cubin flashinfer-jit-cache || true

# Install apt packages
apt install -y git libnuma-dev libssl-dev pkg-config

# Check if protoc of correct architecture is already installed
if command -v protoc >/dev/null 2>&1; then
    if protoc --version >/dev/null 2>&1; then
        echo "protoc already installed: $(protoc --version)"
    else
        echo "protoc found but not runnable, reinstalling..."
        INSTALL_PROTOC=1
    fi
else
    INSTALL_PROTOC=1
fi

# Install protoc for router build (gRPC protobuf compilation)
if [ "${INSTALL_PROTOC:-0}" = "1" ]; then
    echo "Installing protoc..."
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        apt-get update
        apt-get install -y wget unzip gcc g++ perl make
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS
        yum update -y
        yum install -y wget unzip gcc gcc-c++ perl-core make
    fi

    cd /tmp
    # Determine protoc architecture
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        PROTOC_ARCH="aarch_64"
    else
        PROTOC_ARCH="x86_64"
    fi
    PROTOC_ZIP="protoc-32.0-linux-${PROTOC_ARCH}.zip"
    wget https://github.com/protocolbuffers/protobuf/releases/download/v32.0/${PROTOC_ZIP}
    unzip -o ${PROTOC_ZIP} -d /usr/local
    rm ${PROTOC_ZIP}
    protoc --version
    cd -
else
    echo "protoc already installed: $(protoc --version)"
fi

# Install uv
if [ "$IS_BLACKWELL" = "1" ]; then
    # The blackwell CI runner has some issues with pip and uv,
    # so we can only use pip with `--break-system-packages`
    PIP_CMD="pip"
    PIP_INSTALL_SUFFIX="--break-system-packages"
    $PIP_CMD install --upgrade pip

    # Clean up existing installations
    $PIP_CMD uninstall -y sgl-kernel sglang $PIP_INSTALL_SUFFIX || true
else
    # In normal cases, we use uv, which is much faster than pip.
    pip install --upgrade pip
    pip install uv
    export UV_SYSTEM_PYTHON=true

    PIP_CMD="uv pip"
    PIP_INSTALL_SUFFIX="--index-strategy unsafe-best-match --prerelease allow"

    # Clean up existing installations
    $PIP_CMD uninstall sgl-kernel sglang || true
fi

# Install the main package
$PIP_CMD install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION} $PIP_INSTALL_SUFFIX

# Install router for pd-disagg test
$PIP_CMD install sglang-router $PIP_INSTALL_SUFFIX

# Install sgl-kernel
SGL_KERNEL_VERSION_FROM_KERNEL=$(grep -Po '(?<=^version = ")[^"]*' sgl-kernel/pyproject.toml)
SGL_KERNEL_VERSION_FROM_SRT=$(grep -Po -m1 '(?<=sgl-kernel==)[0-9A-Za-z\.\-]+' python/pyproject.toml)
echo "SGL_KERNEL_VERSION_FROM_KERNEL=${SGL_KERNEL_VERSION_FROM_KERNEL} SGL_KERNEL_VERSION_FROM_SRT=${SGL_KERNEL_VERSION_FROM_SRT}"

if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ]; then
    ls -alh sgl-kernel/dist
    # Determine wheel architecture
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        WHEEL_ARCH="aarch64"
    else
        WHEEL_ARCH="x86_64"
    fi
    $PIP_CMD install sgl-kernel/dist/sgl_kernel-${SGL_KERNEL_VERSION_FROM_KERNEL}-cp310-abi3-manylinux2014_${WHEEL_ARCH}.whl --force-reinstall $PIP_INSTALL_SUFFIX
else
    $PIP_CMD install sgl-kernel==${SGL_KERNEL_VERSION_FROM_SRT} --force-reinstall $PIP_INSTALL_SUFFIX
fi

# Show current packages
$PIP_CMD list

$PIP_CMD install mooncake-transfer-engine==0.3.7.post2 "${NVRTC_SPEC}" py-spy scipy huggingface_hub[hf_xet] $PIP_INSTALL_SUFFIX

if [ "$IS_BLACKWELL" != "1" ]; then
    # For lmms_evals evaluating MMMU
    git clone --branch v0.5 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    $PIP_CMD install -e lmms-eval/ $PIP_INSTALL_SUFFIX

    # Install xformers
    $PIP_CMD install xformers --index-url https://download.pytorch.org/whl/${CU_VERSION} --no-deps $PIP_INSTALL_SUFFIX
fi

# Show current packages
$PIP_CMD list
python3 -c "import torch; print(torch.version.cuda)"
