#!/bin/bash
# Install the dependency in CI.
set -euxo pipefail

# Set up environment variables
IS_BLACKWELL=${IS_BLACKWELL:-0}
CU_VERSION="cu129"
FLASHINFER_VERSION=0.5.3
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
bash "${SCRIPT_DIR}/../killall_sglang.sh"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Clear torch compilation cache
python3 -c 'import os, shutil, tempfile, getpass; cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "torchinductor_" + getpass.getuser()); shutil.rmtree(cache_dir, ignore_errors=True)'

# Fix corrupted apt cache if present (common on self-hosted runners)
rm -f /var/cache/apt/*.bin 2>/dev/null || true

# Install apt packages
apt install -y git libnuma-dev libssl-dev pkg-config libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils

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
pip install --upgrade pip

if [ "$IS_BLACKWELL" = "1" ]; then
    # The blackwell CI runner has some issues with pip and uv,
    # so we can only use pip with `--break-system-packages`
    PIP_CMD="pip"
    PIP_INSTALL_SUFFIX="--break-system-packages"
    PIP_UNINSTALL_CMD="pip uninstall -y"
    PIP_UNINSTALL_SUFFIX="--break-system-packages"
else
    # In normal cases, we use uv, which is much faster than pip.
    pip install uv
    export UV_SYSTEM_PYTHON=true

    PIP_CMD="uv pip"
    PIP_INSTALL_SUFFIX="--index-strategy unsafe-best-match --prerelease allow"
    PIP_UNINSTALL_CMD="uv pip uninstall"
    PIP_UNINSTALL_SUFFIX=""
fi

# Clean up corrupted packages on Blackwell (race conditions can leave packages in broken state)
if [ "$IS_BLACKWELL" = "1" ]; then
    PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

    # Check for corrupted sgl-kernel (missing METADATA or missing package files)
    SGL_KERNEL_CORRUPTED=false
    if ls "${PYTHON_SITE_PACKAGES}"/sgl_kernel-*.dist-info 1>/dev/null 2>&1; then
        for dist_info in "${PYTHON_SITE_PACKAGES}"/sgl_kernel-*.dist-info; do
            if [ ! -f "${dist_info}/METADATA" ] || [ ! -f "${dist_info}/RECORD" ]; then
                SGL_KERNEL_CORRUPTED=true
                break
            fi
        done
    fi
    # Also check if package directory exists but is incomplete
    if [ -d "${PYTHON_SITE_PACKAGES}/sgl_kernel" ]; then
        if [ ! -f "${PYTHON_SITE_PACKAGES}/sgl_kernel/__init__.py" ]; then
            SGL_KERNEL_CORRUPTED=true
        fi
    fi
    if [ "$SGL_KERNEL_CORRUPTED" = true ]; then
        echo "Detected corrupted sgl-kernel installation, cleaning up..."
        rm -rf "${PYTHON_SITE_PACKAGES}"/sgl_kernel*
        rm -rf "${PYTHON_SITE_PACKAGES}"/sgl-kernel*
    fi

    # Check for corrupted nvidia packages
    for pkg in nvidia_nvshmem_cu12 nvidia_cudnn_cu12; do
        if ls "${PYTHON_SITE_PACKAGES}"/${pkg}-*.dist-info 1>/dev/null 2>&1; then
            for dist_info in "${PYTHON_SITE_PACKAGES}"/${pkg}-*.dist-info; do
                if [ ! -f "${dist_info}/METADATA" ]; then
                    echo "Detected corrupted ${pkg} installation, cleaning up..."
                    rm -rf "${PYTHON_SITE_PACKAGES}"/${pkg}*
                    break
                fi
            done
        fi
    done
fi

# Clean up existing installations (skip for Blackwell - dependencies are pre-installed)
if [ "$IS_BLACKWELL" != "1" ]; then
    $PIP_UNINSTALL_CMD sgl-kernel sglang $PIP_UNINSTALL_SUFFIX || true
    $PIP_UNINSTALL_CMD flashinfer-python flashinfer-cubin flashinfer-jit-cache $PIP_UNINSTALL_SUFFIX || true
    $PIP_UNINSTALL_CMD opencv-python opencv-python-headless $PIP_UNINSTALL_SUFFIX || true
fi

# Install the main package
EXTRAS="dev"
if [ -n "$OPTIONAL_DEPS" ]; then
    EXTRAS="dev,${OPTIONAL_DEPS}"
fi
echo "Installing python extras: [${EXTRAS}]"

if [ "$IS_BLACKWELL" = "1" ]; then
    # For Blackwell, use --no-deps and --no-build-isolation to avoid race conditions
    # Dependencies are pre-installed on the machines
    $PIP_CMD install -e "python" --no-deps --no-build-isolation $PIP_INSTALL_SUFFIX
else
    $PIP_CMD install -e "python[${EXTRAS}]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION} $PIP_INSTALL_SUFFIX
fi

# Install router for pd-disagg test (skip for Blackwell if already installed)
if [ "$IS_BLACKWELL" != "1" ]; then
    $PIP_CMD install sglang-router $PIP_INSTALL_SUFFIX
fi

# Remove flash_attn folder to avoid conflicts
PYTHON_LIB_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")
FLASH_ATTN_PATH="${PYTHON_LIB_PATH}/flash_attn"

if [ -d "$FLASH_ATTN_PATH" ]; then
    echo "Directory $FLASH_ATTN_PATH exists. Removing..."
    rm -rf "$FLASH_ATTN_PATH"
else
    echo "Directory $FLASH_ATTN_PATH does not exist."
fi

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
    # On Blackwell machines, skip reinstall if correct version already installed to avoid race conditions
    if [ "$IS_BLACKWELL" = "1" ]; then
        INSTALLED_SGL_KERNEL=$(pip show sgl-kernel 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
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

# Install other python dependencies (skip for Blackwell - pre-installed)
if [ "$IS_BLACKWELL" != "1" ]; then
    $PIP_CMD install mooncake-transfer-engine==0.3.8.post1 "${NVRTC_SPEC}" py-spy scipy huggingface_hub[hf_xet] pytest $PIP_INSTALL_SUFFIX

    # For lmms_evals evaluating MMMU
    git clone --branch v0.5 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    $PIP_CMD install -e lmms-eval/ $PIP_INSTALL_SUFFIX

    # DeepEP depends on nvshmem 3.4.5
    $PIP_CMD install nvidia-nvshmem-cu12==3.4.5 --force-reinstall $PIP_INSTALL_SUFFIX

    # Cudnn with version less than 9.16.0.29 will cause performance regression on Conv3D kernel
    $PIP_CMD install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall $PIP_INSTALL_SUFFIX

    $PIP_CMD uninstall xformers || true
fi

# Install flashinfer-jit-cache (skip for Blackwell - pre-installed)
if [ "$IS_BLACKWELL" != "1" ]; then
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
            if pip download flashinfer-jit-cache==${FLASHINFER_VERSION} \
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
fi

# Show current packages
$PIP_CMD list
python3 -c "import torch; print(torch.version.cuda)"

# Prepare the CI runner (cleanup HuggingFace cache, etc.)
bash "${SCRIPT_DIR}/prepare_runner.sh"
