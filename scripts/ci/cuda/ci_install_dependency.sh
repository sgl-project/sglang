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
#
# CU_VERSION controls:
#   - PyTorch index URL (pytorch.org/whl/${CU_VERSION})
#   - FlashInfer JIT cache index (flashinfer.ai/whl/${CU_VERSION})
#   - nvrtc variant selection (cu12 vs cu13)

CU_VERSION="${CU_VERSION:-cu130}"
CU_STRIP="${CU_VERSION#cu}"
CU_MAJOR="${CU_STRIP:0:2}"

# Nvidia package versions we override (torch pins older versions).
# Used both as pip constraints during install and for post-install verification.
NVIDIA_CUDNN_VERSION="9.16.0.29"
NVIDIA_NVSHMEM_VERSION="3.4.5"
OPTIONAL_DEPS="${1:-}"

# Whether to create a uv venv. Default false; set USE_VENV=false to install
# directly into system Python (useful for runners where uv venv misbehaves).
USE_VENV="${USE_VENV:-0}"
echo "USE_VENV=${USE_VENV}"

# uv must be available on system Python (to create the venv, or to run
# `uv pip install --system` when venv mode is disabled). Install if missing.
python3 -m pip install --upgrade pip
if ! command -v uv >/dev/null 2>&1; then
    pip install uv
fi

SYS_PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [ "$USE_VENV" = "1" ]; then
    # Per-job unique path. Include $$ (shell PID) so concurrent/back-to-back jobs
    # on the same runner never target the same directory even if GITHUB_JOB
    # doesn't differentiate matrix partitions.
    UV_VENV="/tmp/sglang-ci-${GITHUB_RUN_ID:-norun}-${GITHUB_JOB:-nojob}-$$"
    # --seed installs pip/setuptools into the venv so bare `pip` calls in
    # cache_nvidia_wheels.sh and the human-eval setup resolve to the venv's
    # pip (rather than silently falling back to system Python).
    uv venv "$UV_VENV" --python "python${SYS_PYTHON_VER}" --seed
    # shellcheck disable=SC1091
    source "$UV_VENV/bin/activate"
    # Assert activation actually took effect. A misconfigured activate script
    # would otherwise leave us silently running against system Python.
    [ "${VIRTUAL_ENV:-}" = "$UV_VENV" ] || { echo "FATAL: venv activation did not set VIRTUAL_ENV correctly"; exit 1; }
    [ "$(command -v python3)" = "$UV_VENV/bin/python3" ] || { echo "FATAL: python3 still resolves outside venv (got $(command -v python3))"; exit 1; }

    # Propagate to subsequent workflow steps. GITHUB_ENV/GITHUB_PATH only
    # affect *later* steps, never the current one.
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "VIRTUAL_ENV=$UV_VENV" >> "$GITHUB_ENV"
        echo "SGLANG_CI_VENV_PATH=$UV_VENV" >> "$GITHUB_ENV"
        # Set BASH_ENV early so subsequent steps auto-source the venv's env script.
        # LD_LIBRARY_PATH is written to this file later (after packages are installed)
        # and gets picked up even if GITHUB_ENV becomes unavailable at that point.
        echo "BASH_ENV=$UV_VENV/env.sh" >> "$GITHUB_ENV"
        touch "$UV_VENV/env.sh"
    fi
    if [ -n "${GITHUB_PATH:-}" ]; then
        echo "$UV_VENV/bin" >> "$GITHUB_PATH"
    fi
else
    echo "USE_VENV=0: skipping uv venv creation, installing into system Python"
    UV_VENV=""
fi

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

# Install Rust toolchain (needed by crates built via setuptools-rust, e.g. the
# native gRPC extension bundled into the sglang wheel).
bash "${SCRIPT_DIR}/../utils/install_rustup.sh"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

mark_step_done "Python package site hygiene & install protoc + rust"

# ------------------------------------------------------------------------------
# Pip / uv toolchain & stale package cleanup
# ------------------------------------------------------------------------------
# Install pip and uv (use python3 -m pip for robustness since some runners only have pip3).
# In venv mode this upgrades the venv's pip (the bootstrap block near the top
# already upgraded system pip before `uv venv`).
python3 -m pip install --upgrade pip

# uv is already installed on system Python (above).
# - Venv mode: the venv is active and `uv pip` targets it automatically.
# - Non-venv mode: UV_SYSTEM_PYTHON=1 makes `uv pip` operate on system Python
#   (otherwise uv refuses to run outside a venv).
if [ "$USE_VENV" != "1" ]; then
    export UV_SYSTEM_PYTHON=1
fi

export UV_LINK_MODE=copy
PIP_CMD="uv pip"
PIP_INSTALL_SUFFIX="--index-strategy unsafe-best-match --prerelease allow"
PIP_UNINSTALL_CMD="uv pip uninstall"
PIP_UNINSTALL_SUFFIX=""


# Clean up existing installations
$PIP_UNINSTALL_CMD sgl-kernel sglang-kernel sglang sgl-fa4 flash-attn-4 $PIP_UNINSTALL_SUFFIX || true

mark_step_done "Pip / uv toolchain & stale package cleanup"

# ------------------------------------------------------------------------------
# Uninstall Flashinfer
# ------------------------------------------------------------------------------
# Keep flashinfer packages installed if version matches to avoid re-downloading:
# - flashinfer-cubin: 150+ MB
# - flashinfer-jit-cache: 1.2+ GB, by far the largest download in CI
FLASHINFER_PYTHON_REQUIRED=$(grep -Po -m1 '(?<=flashinfer_python==)[0-9A-Za-z\.\-]+' python/pyproject.toml || echo "")
FLASHINFER_CUBIN_REQUIRED=$(grep -Po -m1 '(?<=flashinfer_cubin==)[0-9A-Za-z\.\-]+' python/pyproject.toml || echo "")
FLASHINFER_CUBIN_INSTALLED=$(pip show flashinfer-cubin 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
FLASHINFER_JIT_INSTALLED=$(pip show flashinfer-jit-cache 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//' || echo "")
FLASHINFER_JIT_CU_VERSION=$(pip show flashinfer-jit-cache 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed -n 's/.*+//p' || echo "")

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

if [ "$UNINSTALL_JIT_CACHE" = false ] && [ "$FLASHINFER_JIT_CU_VERSION" != "$CU_VERSION" ]; then
    echo "flashinfer-jit-cache CUDA version mismatch (installed: ${FLASHINFER_JIT_CU_VERSION:-none}, required: ${CU_VERSION}), will reinstall"
    UNINSTALL_JIT_CACHE=true
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
EXTRAS="dev,runai,tracing"
if [ -n "$OPTIONAL_DEPS" ]; then
    EXTRAS="dev,runai,tracing,${OPTIONAL_DEPS}"
fi
echo "Installing python extras: [${EXTRAS}]"
# source "${SCRIPT_DIR}/cache_nvidia_wheels.sh"
$PIP_CMD install -e "python[${EXTRAS}]" $PIP_INSTALL_SUFFIX

mark_step_done "Install main package"

# ------------------------------------------------------------------------------
# Install torch/sglang-kernel
# ------------------------------------------------------------------------------
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
    # Wheel may have +cuXYZ suffix (e.g. sglang_kernel-0.4.0+cu130-...) depending on CUDA version
    KERNEL_WHL=$(ls sgl-kernel/dist/sglang_kernel-${SGL_KERNEL_VERSION_FROM_KERNEL}*-cp310-abi3-manylinux2014_${WHEEL_ARCH}.whl 2>/dev/null | head -1)
    if [ -z "$KERNEL_WHL" ]; then
        echo "ERROR: No matching sgl-kernel wheel found in sgl-kernel/dist/ for version ${SGL_KERNEL_VERSION_FROM_KERNEL} arch ${WHEEL_ARCH}"
        ls -alh sgl-kernel/dist/
        exit 1
    fi
    echo "Installing sgl-kernel wheel: $KERNEL_WHL"
    $PIP_CMD install "$KERNEL_WHL" --force-reinstall $PIP_INSTALL_SUFFIX
else
    if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ] && [ ! -d "sgl-kernel/dist" ]; then
        # CUSTOM_BUILD_SGL_KERNEL was set but artifacts not available (e.g., stage rerun without wheel build)
        # Fail instead of falling back to PyPI - we need to test the built kernel, not PyPI version
        echo "ERROR: CUSTOM_BUILD_SGL_KERNEL=true but sgl-kernel/dist not found."
        echo "This usually happens when rerunning a stage without the sgl-kernel-build-wheels job."
        echo "Please re-run the full workflow using /tag-and-rerun-ci to rebuild the kernel."
        exit 1
    fi
fi

# Now we are running torch with cuda13 in CI environment, so the torch packages will be reinstalled if they are still at CU129 version
# TODO: Remove this part after torch has been upgraded to 2.11, where cu13 is enabled by default
TORCH_CUDA_VER=$(python3 -c "import torch; v=torch.version.cuda; parts=v.split('.'); print(f'cu{parts[0]}{parts[1]}')")
echo "Detected torch CUDA version: ${TORCH_CUDA_VER}"
if [ "${TORCH_CUDA_VER}" != "${CU_VERSION}" ]; then
    TORCH_VER=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//')
    TORCHAUDIO_VER=$(pip show torchaudio 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//')
    TORCHVISION_VER=$(pip show torchvision 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//')
    echo "Reinstalling torch==${TORCH_VER} torchaudio==${TORCHAUDIO_VER} torchvision==${TORCHVISION_VER} from ${CU_VERSION} index to match torch..."
    $PIP_CMD install "torch==${TORCH_VER}" "torchaudio==${TORCHAUDIO_VER}" "torchvision==${TORCHVISION_VER}" --index-url "https://download.pytorch.org/whl/${CU_VERSION}" --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
fi

# sglang-kernel wheels carry a +cuXYZ local version tag (e.g. 0.4.1+cu130).
# If it doesn't match CU_VERSION, reinstall from the matching index.
SGL_KERNEL_FULL_VER=$(pip show sglang-kernel 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
SGL_KERNEL_CUDA_VER=$(printf '%s' "$SGL_KERNEL_FULL_VER" | sed -n 's/.*+//p')
echo "Detected sglang-kernel version: ${SGL_KERNEL_FULL_VER} (CUDA tag: ${SGL_KERNEL_CUDA_VER:-none})"
if [ -n "$SGL_KERNEL_CUDA_VER" ] && [ "$SGL_KERNEL_CUDA_VER" != "$CU_VERSION" ]; then
    SGL_KERNEL_VER="${SGL_KERNEL_FULL_VER%+*}"
    echo "Reinstalling sglang-kernel==${SGL_KERNEL_VER} from ${CU_VERSION} index to match torch..."
    if [ "$CU_MAJOR" = "13" ]; then
        $PIP_CMD install "sglang-kernel==${SGL_KERNEL_VER}" --index-url "https://docs.sglang.ai/whl/${CU_VERSION}/" --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
    else
        $PIP_CMD install "sglang-kernel==${SGL_KERNEL_VER}" --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
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

mark_step_done "Download flashinfer artifacts"

# ------------------------------------------------------------------------------
# Stabilize FlashInfer JIT cache paths
# ------------------------------------------------------------------------------
# FlashInfer JIT writes build.ninja with hardcoded -isystem paths pointing to the
# venv's flashinfer/data/ and tvm_ffi/include/. With per-job venvs each job gets
# a unique /tmp/sglang-ci-<run>-<job>-<pid>/ path, but the JIT cache is shared
# on the host mount. When the next job's venv has a different path and the old one
# is cleaned up, ninja fails because source files no longer exist at the cached path.
#
# Fix (two parts):
# 1. Clear only STALE cached_ops (build.ninja referencing non-existent venv paths).
#    Do NOT clear all cached_ops — they contain compiled .so files that take 10-20 min
#    to recompile. Only remove entries where the source paths no longer exist.
# 2. Copy source files to a stable host-mounted path and symlink each venv's
#    copy there. build.ninja then references the stable path across all jobs.
#
# Part 1: Clear stale cached_ops (keep valid compiled kernels)
if [ "$USE_VENV" = "1" ]; then
    STABLE_FI_DIR="${HOME}/.cache/flashinfer/_stable_src"
    if [ -d "${HOME}/.cache/flashinfer" ]; then
        STALE_COUNT=0
        while IFS= read -r ninja_file; do
            # Check for stale venv paths (/tmp/sglang-ci-*) or old stable path (flashinfer-src)
            STALE_PATH=$(grep -o '/tmp/sglang-ci-[^ ]*\|flashinfer-src' "$ninja_file" 2>/dev/null | head -1 || true)
            if [ -n "$STALE_PATH" ]; then
                if echo "$STALE_PATH" | grep -q "flashinfer-src" || [ ! -d "$STALE_PATH" ]; then
                    rm -rf "$(dirname "$ninja_file")"
                    STALE_COUNT=$((STALE_COUNT + 1))
                fi
            fi
        done < <(find "${HOME}/.cache/flashinfer" -name "build.ninja" -type f 2>/dev/null)
        echo "Cleaned $STALE_COUNT stale FlashInfer cached_ops (kept valid ones)"
    fi

    # Part 2: Stabilize paths (STABLE_FI_DIR set above in Part 1)
    FI_DATA=$(python3 -c "import flashinfer, os; print(os.path.join(os.path.dirname(flashinfer.__file__), 'data'))")
    TVM_INC=$(python3 -c "import tvm_ffi, os; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'include'))")

    FI_VERSION="${FLASHINFER_PYTHON_REQUIRED}"
    if [ ! -d "$STABLE_FI_DIR/flashinfer-data" ] || [ "$(cat "$STABLE_FI_DIR/.version" 2>/dev/null)" != "$FI_VERSION" ]; then
        rm -rf "$STABLE_FI_DIR"
        mkdir -p "$STABLE_FI_DIR"
        cp -a "$FI_DATA" "$STABLE_FI_DIR/flashinfer-data"
        cp -a "$TVM_INC" "$STABLE_FI_DIR/tvm-ffi-include"
        echo "$FI_VERSION" > "$STABLE_FI_DIR/.version"
        echo "Copied flashinfer source files to stable path: $STABLE_FI_DIR (version=$FI_VERSION)"
    else
        echo "Stable flashinfer source path up to date (version=$FI_VERSION)"
    fi

    rm -rf "$FI_DATA"
    ln -s "$STABLE_FI_DIR/flashinfer-data" "$FI_DATA"
    TVM_INC_PARENT=$(dirname "$TVM_INC")
    rm -rf "$TVM_INC_PARENT/include"
    ln -s "$STABLE_FI_DIR/tvm-ffi-include" "$TVM_INC_PARENT/include"
    echo "Symlinked venv flashinfer/tvm_ffi -> $STABLE_FI_DIR"

    mark_step_done "Stabilize FlashInfer JIT cache paths"
fi


# ------------------------------------------------------------------------------
# Install extra dependency
# ------------------------------------------------------------------------------
# Install other python dependencies.
# Match on CUDA major version so future minor bumps (cu131, etc.) don't fall
# through to the wrong branch. Prefer NVCC_VER (set in the venv path); otherwise
# parse the first two digits of CU_VERSION (pytorch convention is cu{major}{minor}
# with a single-digit minor, e.g. cu126, cu129, cu130).
if [ "$CU_MAJOR" = "13" ]; then
    MOONCAKE_PKG="mooncake-transfer-engine-cuda13==0.3.10.post2"
    EXTRA_NVIDIA_SPECS="nvidia-cuda-nvrtc"
else
    MOONCAKE_PKG="mooncake-transfer-engine==0.3.10.post2"
    EXTRA_NVIDIA_SPECS="nvidia-cuda-nvrtc-cu12"
fi
$PIP_CMD install ${MOONCAKE_PKG} ${EXTRA_NVIDIA_SPECS} py-spy scipy huggingface_hub[hf_xet] pytest $PIP_INSTALL_SUFFIX

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

# Pick cu12 vs cu13 variants of nvshmem / cudnn based on CU_VERSION
if [ "$CU_MAJOR" = "13" ]; then
    NVSHMEM_PKG="nvidia-nvshmem-cu13"
    CUDNN_PKG="nvidia-cudnn-cu13"
else
    NVSHMEM_PKG="nvidia-nvshmem-cu12"
    CUDNN_PKG="nvidia-cudnn-cu12"
fi

# Fix dependencies: DeepEP depends on nvshmem 3.4.5 — skip reinstall when already correct (avoids pip races / wasted work)
INSTALLED_NVSHMEM=$(pip show ${NVSHMEM_PKG} 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [ "$INSTALLED_NVSHMEM" = "$NVIDIA_NVSHMEM_VERSION" ]; then
    echo "${NVSHMEM_PKG}==${NVIDIA_NVSHMEM_VERSION} already installed, skipping reinstall"
else
    $PIP_CMD install ${NVSHMEM_PKG}==${NVIDIA_NVSHMEM_VERSION} $PIP_INSTALL_SUFFIX
fi

# Fix dependencies: Cudnn with version less than 9.16.0.29 will cause performance regression on Conv3D kernel
INSTALLED_CUDNN=$(pip show ${CUDNN_PKG} 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
if [ "$INSTALLED_CUDNN" = "$NVIDIA_CUDNN_VERSION" ]; then
    echo "${CUDNN_PKG}==${NVIDIA_CUDNN_VERSION} already installed, skipping reinstall"
else
    $PIP_CMD install ${CUDNN_PKG}==${NVIDIA_CUDNN_VERSION} $PIP_INSTALL_SUFFIX
fi

mark_step_done "Fix other dependencies"

# Download kernels from kernels community
kernels download python || true
kernels lock python || true
# Ensure target is a directory — on fresh containers or after a previous buggy
# `mv` that created a FILE at this path, mkdir -p would fail silently.
[ -e "${HOME}/.cache/sglang" ] && [ ! -d "${HOME}/.cache/sglang" ] && rm -f "${HOME}/.cache/sglang"
mkdir -p "${HOME}/.cache/sglang/"
mv python/kernels.lock "${HOME}/.cache/sglang/" || true

# Install human-eval. This script is sourced from ci_install_deepep.sh, so a
# bare `cd human-eval` would leave the caller stuck in that directory for the
# rest of its execution. The subshell keeps the cd local to the pip install.
$PIP_CMD install "setuptools==70.0.0" $PIP_INSTALL_SUFFIX
[ -d human-eval ] || git clone https://github.com/merrymercy/human-eval.git
(
    cd human-eval
    $PIP_CMD install -e . --no-build-isolation $PIP_INSTALL_SUFFIX)

# ------------------------------------------------------------------------------
# Prepare runner
# ------------------------------------------------------------------------------
# Prepare the CI runner (cleanup HuggingFace cache, etc.)
bash "${SCRIPT_DIR}/prepare_runner.sh"

mark_step_done "Prepare runner"

# ------------------------------------------------------------------------------
# LD_LIBRARY_PATH discovery
# ------------------------------------------------------------------------------
# NVIDIA pip packages (cublas, cudnn, nccl, nvrtc, ...) and torch ship .so files
# under site-packages. In venv mode these are NOT on the default LD_LIBRARY_PATH,
# so dlopen('libcublas.so.12') from torch would fail. Prepend them here.
# In non-venv mode, system site-packages may also need this if the runner's
# default ld config doesn't cover the NVIDIA pip layout.
SITE_PACKAGES=$(python3 -c "import site, sys; print(site.getsitepackages()[0])")
# Glob matches NVIDIA pip-package layout:
# site-packages/nvidia/<component>/lib/lib*.so. If NVIDIA restructures
# packaging, this may need updating.
NVIDIA_LIBS=$(find "$SITE_PACKAGES" -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\n' ':')
TORCH_LIB="$SITE_PACKAGES/torch/lib"
VENV_LD="${NVIDIA_LIBS}${TORCH_LIB}"
export LD_LIBRARY_PATH="${VENV_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Write LD_LIBRARY_PATH to the venv's env.sh (always succeeds — local file)
# so subsequent steps auto-source it via BASH_ENV. In non-venv mode, skip the
# env.sh write and rely on GITHUB_ENV propagation.
if [ "$USE_VENV" = "1" ] && [ -n "$UV_VENV" ]; then
    echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"" >> "$UV_VENV/env.sh"
fi
# Also try GITHUB_ENV (may fail if runner temp file was cleaned up during long installs).
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV" || echo "WARNING: GITHUB_ENV write failed; LD_LIBRARY_PATH will be set via BASH_ENV instead"
fi
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"


# ------------------------------------------------------------------------------
# Verify imports
# ------------------------------------------------------------------------------
# Show current packages
$PIP_CMD list
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import cutlass; import cutlass.cute;"
