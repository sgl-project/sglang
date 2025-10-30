#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

IS_BLACKWELL=${IS_BLACKWELL:-0}
CU_VERSION="cu129"

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

# Install apt packages
apt install -y git libnuma-dev

# Install protoc for router build (gRPC protobuf compilation)
if ! command -v protoc &> /dev/null; then
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
    wget https://github.com/protocolbuffers/protobuf/releases/download/v32.0/protoc-32.0-linux-x86_64.zip
    unzip protoc-32.0-linux-x86_64.zip -d /usr/local
    rm protoc-32.0-linux-x86_64.zip
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

    # Clean up existing installations
    $PIP_CMD uninstall -y flashinfer_python sgl-kernel sglang vllm $PIP_INSTALL_SUFFIX || true

    # Install the main package
    $PIP_CMD install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION} $PIP_INSTALL_SUFFIX --force-reinstall
else
    # In normal cases, we use uv, which is much faster than pip.
    pip install --upgrade pip
    pip install uv
    export UV_SYSTEM_PYTHON=true

    PIP_CMD="uv pip"
    PIP_INSTALL_SUFFIX="--index-strategy unsafe-best-match"

    # Clean up existing installations
    $PIP_CMD uninstall flashinfer_python sgl-kernel sglang vllm || true

    # Install the main package without deps
    $PIP_CMD install -e "python[dev]" --no-deps $PIP_INSTALL_SUFFIX --force-reinstall

    # Install flashinfer-python 0.4.1 dependency that requires prerelease (This should be removed when flashinfer fixes this issue)
    $PIP_CMD install flashinfer-python==0.4.1 --prerelease=allow $PIP_INSTALL_SUFFIX

    # Install the main package
    $PIP_CMD install -e "python[dev]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION} $PIP_INSTALL_SUFFIX --upgrade
fi

# Install OpenSSL development libraries for router build
apt install -y libssl-dev pkg-config

# Install router for pd-disagg test
$PIP_CMD install -e "sgl-router" $PIP_INSTALL_SUFFIX

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

$PIP_CMD install mooncake-transfer-engine==0.3.6.post1 "${NVRTC_SPEC}" py-spy scipy huggingface_hub[hf_xet] $PIP_INSTALL_SUFFIX

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
