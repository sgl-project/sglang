#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

# Set up environment variables
IS_BLACKWELL=${IS_BLACKWELL:-0}
CU_VERSION="cu129"
FLASHINFER_VERSION=0.6.2
OPTIONAL_DEPS="${1:-}"

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
bash "${SCRIPT_DIR}/../../killall_sglang.sh"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Resolve a Python binary for mixed CI environments where python3/python may be missing from PATH.
resolve_python_bin() {
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        echo "python"
        return 0
    fi

    # Fall back to pip/pip3 shebang interpreter when python executables are not exported.
    for pip_name in pip pip3; do
        if command -v "$pip_name" >/dev/null 2>&1; then
            local pip_bin shebang pip_python
            pip_bin="$(command -v "$pip_name")"
            shebang="$(head -n 1 "$pip_bin" 2>/dev/null || true)"
            if [[ "$shebang" == "#!"* ]]; then
                pip_python="${shebang#\#!}"
                if [ -x "$pip_python" ]; then
                    echo "$pip_python"
                    return 0
                fi
            fi
        fi
    done

    return 1
}

# Run pip in environments where only a subset of pip/python binaries are exported.
run_pip() {
    if command -v pip >/dev/null 2>&1; then
        pip "$@"
        return $?
    fi
    if command -v pip3 >/dev/null 2>&1; then
        pip3 "$@"
        return $?
    fi

    local py_bin="${PYTHON_BIN:-}"
    if [ -z "$py_bin" ]; then
        py_bin="$(resolve_python_bin || true)"
    fi
    if [ -n "$py_bin" ]; then
        "$py_bin" -m pip "$@"
        return $?
    fi

    echo "ERROR: pip is not available on PATH and no Python interpreter could be resolved."
    return 1
}

PYTHON_BIN="$(resolve_python_bin || true)"

# Clear torch compilation cache (best effort).
if [ -n "$PYTHON_BIN" ]; then
    "$PYTHON_BIN" -c 'import os, shutil, tempfile, getpass; cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "torchinductor_" + getpass.getuser()); shutil.rmtree(cache_dir, ignore_errors=True)'
else
    echo "Warning: Python interpreter not found; skipping torch compilation cache cleanup."
fi

# Install apt packages
# Use --no-install-recommends and ignore errors from unrelated broken packages on the runner
# The NVIDIA driver packages may have broken dependencies that are unrelated to these packages
apt-get install -y --no-install-recommends git libnuma-dev libssl-dev pkg-config libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils || {
    echo "Warning: apt-get install failed, checking if required packages are available..."
    # Verify the packages we need are actually installed
    for pkg in git libnuma-dev libssl-dev pkg-config libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils; do
        if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
            echo "ERROR: Required package $pkg is not installed and apt-get failed"
            exit 1
        fi
    done
    echo "All required packages are already installed, continuing..."
}

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
    # TODO: move this to a separate script
    echo "Installing protoc..."
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        apt-get update || true  # May fail due to unrelated broken packages
        apt-get install -y --no-install-recommends wget unzip gcc g++ perl make || {
            echo "Warning: apt-get install failed, checking if required packages are available..."
            for pkg in wget unzip gcc g++ perl make; do
                if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
                    echo "ERROR: Required package $pkg is not installed and apt-get failed"
                    exit 1
                fi
            done
            echo "All required packages are already installed, continuing..."
        }
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
run_pip install --upgrade pip

if [ "$IS_BLACKWELL" = "1" ]; then
    # The blackwell CI runner has some issues with pip and uv,
    # so we can only use pip with `--break-system-packages`
    PIP_CMD="run_pip"
    PIP_INSTALL_SUFFIX="--break-system-packages"
    PIP_UNINSTALL_CMD="run_pip uninstall -y"
    PIP_UNINSTALL_SUFFIX="--break-system-packages"
else
    # In normal cases, we use uv, which is much faster than pip.
    run_pip install uv
    export UV_SYSTEM_PYTHON=true

    PIP_CMD="uv pip"
    PIP_INSTALL_SUFFIX="--index-strategy unsafe-best-match --prerelease allow"
    PIP_UNINSTALL_CMD="uv pip uninstall"
    PIP_UNINSTALL_SUFFIX=""
fi

# Clean up existing installations
$PIP_UNINSTALL_CMD sgl-kernel sglang $PIP_UNINSTALL_SUFFIX || true
$PIP_UNINSTALL_CMD flashinfer-python flashinfer-cubin flashinfer-jit-cache $PIP_UNINSTALL_SUFFIX || true
$PIP_UNINSTALL_CMD opencv-python opencv-python-headless $PIP_UNINSTALL_SUFFIX || true

# Install the main package
EXTRAS="dev"
if [ -n "$OPTIONAL_DEPS" ]; then
    EXTRAS="dev,${OPTIONAL_DEPS}"
fi
echo "Installing python extras: [${EXTRAS}]"

$PIP_CMD install -e "python[${EXTRAS}]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION} $PIP_INSTALL_SUFFIX

# Install router for pd-disagg test
$PIP_CMD install sglang-router $PIP_INSTALL_SUFFIX

# Remove flash_attn folder to avoid conflicts (best effort).
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(resolve_python_bin || true)"
fi

if [ -n "$PYTHON_BIN" ]; then
    PYTHON_LIB_PATH=$("$PYTHON_BIN" -c "import site; print(site.getsitepackages()[0])")
    FLASH_ATTN_PATH="${PYTHON_LIB_PATH}/flash_attn"

    if [ -d "$FLASH_ATTN_PATH" ]; then
        echo "Directory $FLASH_ATTN_PATH exists. Removing..."
        rm -rf "$FLASH_ATTN_PATH"
    else
        echo "Directory $FLASH_ATTN_PATH does not exist."
    fi
else
    echo "Warning: Python interpreter not found; skipping flash_attn cleanup."
fi

# Install sgl-kernel
SGL_KERNEL_VERSION_FROM_KERNEL=$(grep -Po '(?<=^version = ")[^"]*' sgl-kernel/pyproject.toml)
SGL_KERNEL_VERSION_FROM_SRT=$(grep -Po -m1 '(?<=sgl-kernel==)[0-9A-Za-z\.\-]+' python/pyproject.toml)
echo "SGL_KERNEL_VERSION_FROM_KERNEL=${SGL_KERNEL_VERSION_FROM_KERNEL} SGL_KERNEL_VERSION_FROM_SRT=${SGL_KERNEL_VERSION_FROM_SRT}"

if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ] && [ -d "sgl-kernel/dist" ]; then
    ls -alh sgl-kernel/dist
    # Determine wheel architecture
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        WHEEL_ARCH="aarch64"
    else
        WHEEL_ARCH="x86_64"
    fi
    $PIP_CMD install sgl-kernel/dist/sgl_kernel-${SGL_KERNEL_VERSION_FROM_KERNEL}-cp310-abi3-manylinux2014_${WHEEL_ARCH}.whl --force-reinstall $PIP_INSTALL_SUFFIX
elif [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ] && [ ! -d "sgl-kernel/dist" ]; then
    # CUSTOM_BUILD_SGL_KERNEL was set but artifacts not available (e.g., stage rerun without wheel build)
    # Fail instead of falling back to PyPI - we need to test the built kernel, not PyPI version
    echo "ERROR: CUSTOM_BUILD_SGL_KERNEL=true but sgl-kernel/dist not found."
    echo "This usually happens when rerunning a stage without the sgl-kernel-build-wheels job."
    echo "Please re-run the full workflow using /tag-and-rerun-ci to rebuild the kernel."
    exit 1
else
    # On Blackwell machines, skip reinstall if correct version already installed to avoid race conditions
    if [ "$IS_BLACKWELL" = "1" ]; then
        INSTALLED_SGL_KERNEL=$(run_pip show sgl-kernel 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
        if [ "$INSTALLED_SGL_KERNEL" = "$SGL_KERNEL_VERSION_FROM_SRT" ]; then
            echo "sgl-kernel==${SGL_KERNEL_VERSION_FROM_SRT} already installed, skipping reinstall"
        else
            echo "Installing sgl-kernel==${SGL_KERNEL_VERSION_FROM_SRT} (current: ${INSTALLED_SGL_KERNEL:-none})"
            $PIP_CMD install sgl-kernel==${SGL_KERNEL_VERSION_FROM_SRT} $PIP_INSTALL_SUFFIX
        fi
    else
        $PIP_CMD install sgl-kernel==${SGL_KERNEL_VERSION_FROM_SRT} --force-reinstall $PIP_INSTALL_SUFFIX
    fi
fi

# Show current packages
$PIP_CMD list

# Install other python dependencies
$PIP_CMD install mooncake-transfer-engine==0.3.9 "${NVRTC_SPEC}" py-spy scipy huggingface_hub[hf_xet] pytest $PIP_INSTALL_SUFFIX

if [ "$IS_BLACKWELL" != "1" ]; then
    # For lmms_evals evaluating MMMU
    git clone --branch v0.5 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    $PIP_CMD install -e lmms-eval/ $PIP_INSTALL_SUFFIX
fi

# DeepEP depends on nvshmem 3.4.5
# On Blackwell machines, skip reinstall if correct version already installed to avoid race conditions
if [ "$IS_BLACKWELL" = "1" ]; then
    INSTALLED_NVSHMEM=$(run_pip show nvidia-nvshmem-cu12 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
    if [ "$INSTALLED_NVSHMEM" = "3.4.5" ]; then
        echo "nvidia-nvshmem-cu12==3.4.5 already installed, skipping reinstall"
    else
        $PIP_CMD install nvidia-nvshmem-cu12==3.4.5 $PIP_INSTALL_SUFFIX
    fi
else
    $PIP_CMD install nvidia-nvshmem-cu12==3.4.5 --force-reinstall $PIP_INSTALL_SUFFIX
fi

# Cudnn with version less than 9.16.0.29 will cause performance regression on Conv3D kernel
# On Blackwell machines, skip reinstall if correct version already installed to avoid race conditions
if [ "$IS_BLACKWELL" = "1" ]; then
    INSTALLED_CUDNN=$(run_pip show nvidia-cudnn-cu12 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
    if [ "$INSTALLED_CUDNN" = "9.16.0.29" ]; then
        echo "nvidia-cudnn-cu12==9.16.0.29 already installed, skipping reinstall"
    else
        $PIP_CMD install nvidia-cudnn-cu12==9.16.0.29 $PIP_INSTALL_SUFFIX
    fi
else
    $PIP_CMD install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall $PIP_INSTALL_SUFFIX
fi
$PIP_CMD uninstall xformers || true

# Install flashinfer-jit-cache with caching and retry logic (flashinfer.ai can have transient DNS issues)
# Cache directory for flashinfer wheels (persists across CI runs on self-hosted runners)
FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer-wheels"
mkdir -p "${FLASHINFER_CACHE_DIR}"

# Clean up old versions to avoid cache bloat
find "${FLASHINFER_CACHE_DIR}" -name "flashinfer_jit_cache-*.whl" ! -name "flashinfer_jit_cache-${FLASHINFER_VERSION}*" -type f -delete 2>/dev/null || true

FLASHINFER_WHEEL_PATTERN="flashinfer_jit_cache-${FLASHINFER_VERSION}*.whl"
CACHED_WHEEL=$(find "${FLASHINFER_CACHE_DIR}" -name "${FLASHINFER_WHEEL_PATTERN}" -type f 2>/dev/null | head -n 1)

FLASHINFER_INSTALLED=false

# Try to install from cache first
if [ -n "$CACHED_WHEEL" ] && [ -f "$CACHED_WHEEL" ]; then
    echo "Found cached flashinfer wheel: $CACHED_WHEEL"
    if $PIP_CMD install "$CACHED_WHEEL" $PIP_INSTALL_SUFFIX; then
        FLASHINFER_INSTALLED=true
        echo "Successfully installed flashinfer-jit-cache from cache"
    else
        echo "Failed to install from cache, will try downloading..."
        rm -f "$CACHED_WHEEL"
    fi
fi

# If not installed from cache, download with retry logic
if [ "$FLASHINFER_INSTALLED" = false ]; then
    for i in {1..5}; do
        # Download wheel to cache directory (use pip directly as uv pip doesn't support download)
        if run_pip download flashinfer-jit-cache==${FLASHINFER_VERSION} \
            --index-url https://flashinfer.ai/whl/${CU_VERSION} \
            -d "${FLASHINFER_CACHE_DIR}"; then

            CACHED_WHEEL=$(find "${FLASHINFER_CACHE_DIR}" -name "${FLASHINFER_WHEEL_PATTERN}" -type f 2>/dev/null | head -n 1)
            if [ -n "$CACHED_WHEEL" ] && [ -f "$CACHED_WHEEL" ]; then
                if $PIP_CMD install "$CACHED_WHEEL" $PIP_INSTALL_SUFFIX; then
                    FLASHINFER_INSTALLED=true
                    echo "Successfully downloaded and installed flashinfer-jit-cache"
                    break
                fi
            else
                echo "Warning: Download succeeded but wheel file not found"
            fi
        fi
        echo "Attempt $i to download flashinfer-jit-cache failed, retrying in 10 seconds..."
        sleep 10
    done
fi

if [ "$FLASHINFER_INSTALLED" = false ]; then
    echo "ERROR: Failed to install flashinfer-jit-cache after 5 attempts"
    exit 1
fi

# Show current packages
$PIP_CMD list
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(resolve_python_bin || true)"
fi
if [ -n "$PYTHON_BIN" ]; then
    "$PYTHON_BIN" -c "import torch; print(torch.version.cuda)"
else
    echo "Warning: Python interpreter not found; skipping torch CUDA version print."
fi

# Prepare the CI runner (cleanup HuggingFace cache, etc.)
bash "${SCRIPT_DIR}/prepare_runner.sh"
