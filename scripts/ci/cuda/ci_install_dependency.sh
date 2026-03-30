#!/bin/bash
# Install the dependency in CI.
#
# Structure (see section banners below):
# - Configuration & timing
# - Host / runner detection (arch, Blackwell, pip vs uv)
# - Kill existing processes
# - Install apt packages
# - Python package site hygiene & install protoc
# - Pip / uv toolchain & stale package cleanup
# - Uninstall Flashinfer
# - Install main package
# - Install sglang-kernel
# - Install sglang-router
# - Download flashinfer artifacts
# - Install extra dependency
# - Fix other dependencies
# - Prepare runner
# - Verify imports
set -euxo pipefail

# ------------------------------------------------------------------------------
# Configuration & timing
# ------------------------------------------------------------------------------
# Set up environment variables
CU_VERSION="cu129"
OPTIONAL_DEPS="${1:-}"

SECONDS=0
_CI_MARK_PREV=${SECONDS}

mark_step_done() {
    local label=$1
    local now=${SECONDS}
    local step=$((now - _CI_MARK_PREV))
    printf '\n[STEP DONE] %s,  step: %ss,  total: %ss,  date: %s\n' \
        "${label}" "${step}" "${now}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    _CI_MARK_PREV=${now}
}

mark_step_done "Configuration"

# ------------------------------------------------------------------------------
# Host / runner detection (CPU arch, Blackwell, USE_UV)
# ------------------------------------------------------------------------------
# Detect CPU architecture (x86_64 or aarch64)
ARCH=$(uname -m)
echo "Detected architecture: ${ARCH}"

# Detect GPU architecture (blackwell or not)
if [ "${IS_BLACKWELL+set}" = set ]; then
    case "$IS_BLACKWELL" in 1 | true | yes) IS_BLACKWELL=1 ;; *) IS_BLACKWELL=0 ;; esac
    echo "IS_BLACKWELL=${IS_BLACKWELL} (manually set via environment)"
else
    IS_BLACKWELL=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        while IFS= read -r cap; do
            major="${cap%%.*}"
            if [ "${major:-0}" -ge 10 ] 2>/dev/null; then
                IS_BLACKWELL=1
                break
            fi
        done <<< "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
    fi
    echo "IS_BLACKWELL=${IS_BLACKWELL} (auto-detected via nvidia-smi)"
fi

# Whether to use pip or uv to install dependencies
if [ "${USE_UV+set}" != set ]; then
    if [ "$IS_BLACKWELL" = "1" ]; then
        # Our current b200 runners have some issues with uv, so we default to pip
        # It is a runner specific issue, not a GPU architecture issue.
        USE_UV=false
    else
        USE_UV=true
    fi
fi
case "$(printf '%s' "$USE_UV" | tr '[:upper:]' '[:lower:]')" in 1 | true | yes) USE_UV=1 ;; *) USE_UV=0 ;; esac
echo "USE_UV=${USE_UV}"

mark_step_done "Host / runner detection"

# ------------------------------------------------------------------------------
# Kill existing processes
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
python3 "${REPO_ROOT}/python/sglang/cli/killall.py"
KILLALL_EXIT=$?
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

if [ $KILLALL_EXIT -ne 0 ]; then
    echo "ERROR: killall.py detected uncleanable GPU memory. Aborting CI."
    exit 1
fi

mark_step_done "Kill existing processes"

# ------------------------------------------------------------------------------
# Install apt packages
# ------------------------------------------------------------------------------
# Install apt packages (including python3/pip which may be missing on some runners)
# Use --no-install-recommends and ignore errors from unrelated broken packages on the runner
# The NVIDIA driver packages may have broken dependencies that are unrelated to these packages
# Run apt-get update first to refresh package index (stale index causes 404 on security.ubuntu.com)
apt-get update || true
CI_APT_PACKAGES=(
    python3 python3-pip python3-venv python3-dev git libnuma-dev libssl-dev pkg-config
    libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
)
apt-get install -y --no-install-recommends "${CI_APT_PACKAGES[@]}" || {
    echo "Warning: apt-get install failed, checking if required packages are available..."
    for pkg in "${CI_APT_PACKAGES[@]}"; do
        if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
            echo "ERROR: Required package $pkg is not installed and apt-get failed"
            exit 1
        fi
    done
    echo "All required packages are already installed, continuing..."
}

mark_step_done "Install apt packages"

# ------------------------------------------------------------------------------
# Python package site hygiene & install protoc
# ------------------------------------------------------------------------------
# Clear torch compilation cache
python3 -c 'import os, shutil, tempfile, getpass; cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "torchinductor_" + getpass.getuser()); shutil.rmtree(cache_dir, ignore_errors=True)'

# Remove broken dist-info directories (missing METADATA per PEP 376)
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE_PACKAGES" ]; then
    { set +x; } 2>/dev/null
    find "$SITE_PACKAGES" -maxdepth 1 -name "*.dist-info" -type d | while read -r d; do
        if [ ! -f "$d/METADATA" ]; then
            echo "Removing broken dist-info: $d"
            rm -rf "$d"
        fi
    done
    set -x
fi

# Install protoc
bash "${SCRIPT_DIR}/../utils/install_protoc.sh"

mark_step_done "Python package site hygiene & install protoc"

# ------------------------------------------------------------------------------
# Pip / uv toolchain & stale package cleanup
# ------------------------------------------------------------------------------
# Install pip and uv (use python3 -m pip for robustness since some runners only have pip3)
python3 -m pip install --upgrade pip

if [ "$USE_UV" = "0" ]; then
    PIP_CMD="pip"
    PIP_INSTALL_SUFFIX="--break-system-packages"
    PIP_UNINSTALL_CMD="pip uninstall -y"
    PIP_UNINSTALL_SUFFIX="--break-system-packages"
else
    pip install uv
    export UV_SYSTEM_PYTHON=true

    PIP_CMD="uv pip"
    PIP_INSTALL_SUFFIX="--index-strategy unsafe-best-match --prerelease allow"
    PIP_UNINSTALL_CMD="uv pip uninstall"
    PIP_UNINSTALL_SUFFIX=""
fi

# Clean up existing installations
$PIP_UNINSTALL_CMD sgl-kernel sglang-kernel sglang sgl-fa4 flash-attn-4 $PIP_UNINSTALL_SUFFIX || true

mark_step_done "Pip / uv toolchain & stale package cleanup"

# ------------------------------------------------------------------------------
# Uninstall Flashinfer
# ------------------------------------------------------------------------------
# Keep flashinfer packages installed if version matches to avoid re-downloading:
# - flashinfer-cubin: 150+ MB, plus extra cubins from ci_download_flashinfer_cubin.sh
# - flashinfer-jit-cache: 1.2+ GB, by far the largest download in CI
FLASHINFER_PYTHON_REQUIRED=$(grep -Po -m1 '(?<=flashinfer_python==)[0-9A-Za-z\.\-]+' python/pyproject.toml || echo "")
FLASHINFER_CUBIN_REQUIRED=$(grep -Po -m1 '(?<=flashinfer_cubin==)[0-9A-Za-z\.\-]+' python/pyproject.toml || echo "")
FLASHINFER_CUBIN_INSTALLED=$(pip show flashinfer-cubin 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
FLASHINFER_JIT_INSTALLED=$(pip show flashinfer-jit-cache 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//' || echo "")

UNINSTALL_CUBIN=true
UNINSTALL_JIT_CACHE=true

if [ "$FLASHINFER_CUBIN_INSTALLED" = "$FLASHINFER_CUBIN_REQUIRED" ] && [ -n "$FLASHINFER_CUBIN_REQUIRED" ]; then
    echo "flashinfer-cubin==${FLASHINFER_CUBIN_REQUIRED} already installed, keeping it"
    UNINSTALL_CUBIN=false
else
    echo "flashinfer-cubin version mismatch (installed: ${FLASHINFER_CUBIN_INSTALLED:-none}, required: ${FLASHINFER_CUBIN_REQUIRED}), reinstalling"
fi

if [ "$FLASHINFER_JIT_INSTALLED" = "$FLASHINFER_PYTHON_REQUIRED" ] && [ -n "$FLASHINFER_PYTHON_REQUIRED" ]; then
    echo "flashinfer-jit-cache==${FLASHINFER_PYTHON_REQUIRED} already installed, keeping it"
    UNINSTALL_JIT_CACHE=false
else
    echo "flashinfer-jit-cache version mismatch (installed: ${FLASHINFER_JIT_INSTALLED:-none}, required: ${FLASHINFER_PYTHON_REQUIRED}), will reinstall"
fi

# Build uninstall list based on what needs updating
FLASHINFER_UNINSTALL="flashinfer-python"
[ "$UNINSTALL_CUBIN" = true ] && FLASHINFER_UNINSTALL="$FLASHINFER_UNINSTALL flashinfer-cubin"
[ "$UNINSTALL_JIT_CACHE" = true ] && FLASHINFER_UNINSTALL="$FLASHINFER_UNINSTALL flashinfer-jit-cache"
$PIP_UNINSTALL_CMD $FLASHINFER_UNINSTALL $PIP_UNINSTALL_SUFFIX || true
$PIP_UNINSTALL_CMD opencv-python opencv-python-headless $PIP_UNINSTALL_SUFFIX || true

mark_step_done "Uninstall Flashinfer"

# ------------------------------------------------------------------------------
# Install main package
# ------------------------------------------------------------------------------
# Install the main package
EXTRAS="dev"
if [ -n "$OPTIONAL_DEPS" ]; then
    EXTRAS="dev,${OPTIONAL_DEPS}"
fi
echo "Installing python extras: [${EXTRAS}]"
$PIP_CMD install -e "python[${EXTRAS}]" --extra-index-url https://download.pytorch.org/whl/${CU_VERSION} $PIP_INSTALL_SUFFIX

mark_step_done "Install main package"

# ------------------------------------------------------------------------------
# Install sglang-kernel
# ------------------------------------------------------------------------------
# Install sgl-kernel
SGL_KERNEL_VERSION_FROM_KERNEL=$(grep -Po '(?<=^version = ")[^"]*' sgl-kernel/pyproject.toml)
SGL_KERNEL_VERSION_FROM_SRT=$(grep -Po -m1 '(?<=sglang-kernel==)[0-9A-Za-z\.\-]+' python/pyproject.toml)
echo "SGL_KERNEL_VERSION_FROM_KERNEL=${SGL_KERNEL_VERSION_FROM_KERNEL} SGL_KERNEL_VERSION_FROM_SRT=${SGL_KERNEL_VERSION_FROM_SRT}"

if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ] && [ -d "sgl-kernel/dist" ]; then
    ls -alh sgl-kernel/dist
    # Determine wheel architecture
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        WHEEL_ARCH="aarch64"
    else
        WHEEL_ARCH="x86_64"
    fi
    $PIP_CMD install sgl-kernel/dist/sglang_kernel-${SGL_KERNEL_VERSION_FROM_KERNEL}-cp310-abi3-manylinux2014_${WHEEL_ARCH}.whl --force-reinstall $PIP_INSTALL_SUFFIX
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
        INSTALLED_SGL_KERNEL=$(pip show sglang-kernel 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
        if [ "$INSTALLED_SGL_KERNEL" = "$SGL_KERNEL_VERSION_FROM_SRT" ]; then
            echo "sglang-kernel==${SGL_KERNEL_VERSION_FROM_SRT} already installed, skipping reinstall"
        else
            echo "Installing sglang-kernel==${SGL_KERNEL_VERSION_FROM_SRT} (current: ${INSTALLED_SGL_KERNEL:-none})"
            $PIP_CMD install sglang-kernel==${SGL_KERNEL_VERSION_FROM_SRT} $PIP_INSTALL_SUFFIX
        fi
    else
        $PIP_CMD install sglang-kernel==${SGL_KERNEL_VERSION_FROM_SRT} --force-reinstall $PIP_INSTALL_SUFFIX
    fi
fi

mark_step_done "Install sglang-kernel"

# ------------------------------------------------------------------------------
# Install sglang-router
# ------------------------------------------------------------------------------
# Install router for pd-disagg test
$PIP_CMD install sglang-router $PIP_INSTALL_SUFFIX

# Show current packages
$PIP_CMD list

mark_step_done "Install sglang-router"

# ------------------------------------------------------------------------------
# Download flashinfer artifacts
# ------------------------------------------------------------------------------
# Download flashinfer jit cache
UNINSTALL_JIT_CACHE="$UNINSTALL_JIT_CACHE" \
    FLASHINFER_PYTHON_REQUIRED="$FLASHINFER_PYTHON_REQUIRED" \
    CU_VERSION="$CU_VERSION" \
    PIP_CMD="$PIP_CMD" \
    PIP_INSTALL_SUFFIX="$PIP_INSTALL_SUFFIX" \
    bash "${SCRIPT_DIR}/ci_download_flashinfer_jit_cache.sh"
# Download flashinfer cubins
bash "${SCRIPT_DIR}/ci_download_flashinfer_cubin.sh"

mark_step_done "Download flashinfer artifacts"

# ------------------------------------------------------------------------------
# Install extra dependency
# ------------------------------------------------------------------------------
# Install other python dependencies
if [ "$CU_VERSION" = "cu130" ]; then
    NVRTC_SPEC="nvidia-cuda-nvrtc"
else
    NVRTC_SPEC="nvidia-cuda-nvrtc-cu12"
fi
$PIP_CMD install mooncake-transfer-engine==0.3.9 "${NVRTC_SPEC}" py-spy scipy huggingface_hub[hf_xet] pytest $PIP_INSTALL_SUFFIX

# Install other test dependencies
if [ "$IS_BLACKWELL" != "1" ]; then
    # For lmms_evals evaluating MMMU
    git clone --branch v0.5 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    $PIP_CMD install -e lmms-eval/ $PIP_INSTALL_SUFFIX
fi
$PIP_CMD uninstall xformers || true

mark_step_done "Install extra dependency"

# ------------------------------------------------------------------------------
# Fix other dependencies
# ------------------------------------------------------------------------------
# Fix CUDA version mismatch between torch and torchaudio.
# PyPI's torch 2.9.1 bundles cu128 but torchaudio from pytorch.org/cu129 uses cu129.
# This mismatch causes torchaudio's C extension to fail loading, producing:
#   "partially initialized module 'torchaudio' has no attribute 'lib'"
# We cannot replace torch with cu129 (breaks sgl_kernel ABI), so instead we reinstall
# torchaudio/torchvision from an index matching torch's CUDA version.
TORCH_CUDA_VER=$(python3 -c "import torch; v=torch.version.cuda; parts=v.split('.'); print(f'cu{parts[0]}{parts[1]}')")
echo "Detected torch CUDA version: ${TORCH_CUDA_VER}"
if [ "${TORCH_CUDA_VER}" != "${CU_VERSION}" ]; then
    # Pin versions to match what was installed by pyproject.toml (strip +cuXYZ suffix)
    TORCHAUDIO_VER=$(pip show torchaudio 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//')
    TORCHVISION_VER=$(pip show torchvision 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//')
    echo "Reinstalling torchaudio==${TORCHAUDIO_VER} torchvision==${TORCHVISION_VER} from ${TORCH_CUDA_VER} index to match torch..."
    $PIP_CMD install "torchaudio==${TORCHAUDIO_VER}" "torchvision==${TORCHVISION_VER}" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_VER}" --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
fi

# Fix dependencies: DeepEP depends on nvshmem 3.4.5 — skip reinstall when already correct (avoids pip races / wasted work)
INSTALLED_NVSHMEM=$(pip show nvidia-nvshmem-cu12 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [ "$INSTALLED_NVSHMEM" = "3.4.5" ]; then
    echo "nvidia-nvshmem-cu12==3.4.5 already installed, skipping reinstall"
else
    $PIP_CMD install nvidia-nvshmem-cu12==3.4.5 $PIP_INSTALL_SUFFIX
fi

# Fix dependencies: Cudnn with version less than 9.16.0.29 will cause performance regression on Conv3D kernel
INSTALLED_CUDNN=$(pip show nvidia-cudnn-cu12 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [ "$INSTALLED_CUDNN" = "9.16.0.29" ]; then
    echo "nvidia-cudnn-cu12==9.16.0.29 already installed, skipping reinstall"
else
    $PIP_CMD install nvidia-cudnn-cu12==9.16.0.29 $PIP_INSTALL_SUFFIX
fi

mark_step_done "Fix other dependencies"

# Force reinstall nvidia-cutlass-dsl to ensure the .pth file exists.
# The Docker image ships nvidia-cutlass-dsl-libs-base 4.3.5; upgrading to 4.4.2
# can delete the .pth file without reliably recreating it (pip race condition).
$PIP_CMD install "nvidia-cutlass-dsl>=4.4.1" "nvidia-cutlass-dsl-libs-base>=4.4.1" --no-deps --force-reinstall $PIP_INSTALL_SUFFIX || true

# ------------------------------------------------------------------------------
# Prepare runner
# ------------------------------------------------------------------------------
# Prepare the CI runner (cleanup HuggingFace cache, etc.)
bash "${SCRIPT_DIR}/prepare_runner.sh"

mark_step_done "Prepare runner"

# ------------------------------------------------------------------------------
# Verify imports
# ------------------------------------------------------------------------------
# Show current packages
$PIP_CMD list
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import cutlass; import cutlass.cute;"

mark_step_done "Verify imports"
