#!/bin/bash
# Install dependencies for CUDA CI jobs.
#
# CU_VERSION (default: cu130) controls PyTorch index URL, FlashInfer JIT cache
# index, and nvrtc variant selection.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

configure_environment() {
    # CU_VERSION controls PyTorch index URL, FlashInfer JIT cache index, and
    # nvrtc variant selection (cu12 vs cu13).
    CU_VERSION="${CU_VERSION:-cu130}"
    CU_STRIP="${CU_VERSION#cu}"
    CU_MAJOR="${CU_STRIP:0:2}"

    OPTIONAL_DEPS="${1:-}"

    # Whether to create a uv venv (set USE_VENV=1). Default: 0.
    USE_VENV="${USE_VENV:-0}"
    echo "USE_VENV=${USE_VENV}"

    python3 -m pip install --upgrade pip
    if ! command -v uv >/dev/null 2>&1; then
        pip install uv
    fi

    SYS_PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

    if [ "$USE_VENV" = "1" ]; then
        UV_VENV="/tmp/sglang-ci-${GITHUB_RUN_ID:-norun}-${GITHUB_JOB:-nojob}-$$"
        uv venv "$UV_VENV" --python "python${SYS_PYTHON_VER}" --seed
        # shellcheck disable=SC1091
        source "$UV_VENV/bin/activate"
        [ "${VIRTUAL_ENV:-}" = "$UV_VENV" ] || { echo "FATAL: venv activation did not set VIRTUAL_ENV correctly"; exit 1; }
        [ "$(command -v python3)" = "$UV_VENV/bin/python3" ] || { echo "FATAL: python3 still resolves outside venv (got $(command -v python3))"; exit 1; }

        if [ -n "${GITHUB_ENV:-}" ]; then
            echo "VIRTUAL_ENV=$UV_VENV" >> "$GITHUB_ENV"
            echo "SGLANG_CI_VENV_PATH=$UV_VENV" >> "$GITHUB_ENV"
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

    mark_step_done "${FUNCNAME[0]}"
}

detect_host() {
    ARCH=$(uname -m)
    echo "Detected architecture: ${ARCH}"

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

    if [ "${USE_UV+set}" != set ]; then
        if [ "$IS_BLACKWELL" = "1" ]; then
            USE_UV=false
        else
            USE_UV=true
        fi
    fi
    case "$(printf '%s' "$USE_UV" | tr '[:upper:]' '[:lower:]')" in 1 | true | yes) USE_UV=1 ;; *) USE_UV=0 ;; esac
    echo "USE_UV=${USE_UV}"

    mark_step_done "${FUNCNAME[0]}"
}

kill_existing_processes() {
    python3 "${REPO_ROOT}/python/sglang/cli/killall.py"
    KILLALL_EXIT=$?
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

    if [ $KILLALL_EXIT -ne 0 ]; then
        echo "ERROR: killall.py detected uncleanable GPU memory. Aborting CI."
        exit 1
    fi

    mark_step_done "${FUNCNAME[0]}"
}

install_apt_packages() {
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

    mark_step_done "${FUNCNAME[0]}"
}

clean_site_packages() {
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

    # Install protoc + Rust toolchain (needed by setuptools-rust, e.g. the native gRPC extension)
    bash "${SCRIPT_DIR}/../utils/install_rust_protoc.sh"
    export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

    mark_step_done "${FUNCNAME[0]}"
}

setup_pip_toolchain() {
    python3 -m pip install --upgrade pip

    if [ "$USE_VENV" != "1" ]; then
        export UV_SYSTEM_PYTHON=1
    fi

    export UV_LINK_MODE=copy
    PIP_CMD="uv pip"
    PIP_INSTALL_SUFFIX="--index-strategy unsafe-best-match"
    PIP_UNINSTALL_CMD="uv pip uninstall"
    PIP_UNINSTALL_SUFFIX=""

    $PIP_UNINSTALL_CMD sgl-kernel sglang-kernel sglang sgl-fa4 flash-attn-4 $PIP_UNINSTALL_SUFFIX || true

    mark_step_done "${FUNCNAME[0]}"
}

uninstall_stale_flashinfer() {
    # Keep flashinfer packages if version matches to avoid re-downloading:
    # - flashinfer-cubin: 150+ MB
    # - flashinfer-jit-cache: 1.2+ GB
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

    FLASHINFER_UNINSTALL="flashinfer-python"
    [ "$UNINSTALL_CUBIN" = true ] && FLASHINFER_UNINSTALL="$FLASHINFER_UNINSTALL flashinfer-cubin"
    [ "$UNINSTALL_JIT_CACHE" = true ] && FLASHINFER_UNINSTALL="$FLASHINFER_UNINSTALL flashinfer-jit-cache"
    $PIP_UNINSTALL_CMD $FLASHINFER_UNINSTALL $PIP_UNINSTALL_SUFFIX || true
    $PIP_UNINSTALL_CMD opencv-python opencv-python-headless $PIP_UNINSTALL_SUFFIX || true

    mark_step_done "${FUNCNAME[0]}"
}

install_sglang() {
    EXTRAS="dev,runai,tracing"
    if [ -n "$OPTIONAL_DEPS" ]; then
        EXTRAS="dev,runai,tracing,${OPTIONAL_DEPS}"
    fi
    echo "Installing python extras: [${EXTRAS}]"
    $PIP_CMD install -e "python[${EXTRAS}]" $PIP_INSTALL_SUFFIX

    # Defensive: some runners ended up with nvidia-cusparselt-cu13 metadata
    # present but libcusparseLt.so.0 missing on disk, breaking any torch import.
    # If the file is missing, force-reinstall the wheel before downstream steps.
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    if [ ! -f "$SITE_PACKAGES/nvidia/cusparselt/lib/libcusparseLt.so.0" ] \
       && pip show nvidia-cusparselt-cu13 >/dev/null 2>&1; then
        echo "WARNING: nvidia-cusparselt-cu13 metadata present but libcusparseLt.so.0 missing — reinstalling"
        $PIP_CMD install --reinstall nvidia-cusparselt-cu13 $PIP_INSTALL_SUFFIX
    fi

    mark_step_done "${FUNCNAME[0]}"
}

install_sglang_kernel() {
    SGL_KERNEL_VERSION_FROM_KERNEL=$(grep -Po '(?<=^version = ")[^"]*' sgl-kernel/pyproject.toml)
    SGL_KERNEL_VERSION_FROM_SRT=$(grep -Po -m1 '(?<=sglang-kernel==)[0-9A-Za-z\.\-]+' python/pyproject.toml)
    echo "SGL_KERNEL_VERSION_FROM_KERNEL=${SGL_KERNEL_VERSION_FROM_KERNEL} SGL_KERNEL_VERSION_FROM_SRT=${SGL_KERNEL_VERSION_FROM_SRT}"

    if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ] && [ -d "sgl-kernel/dist" ]; then
        ls -alh sgl-kernel/dist
        if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
            WHEEL_ARCH="aarch64"
        else
            WHEEL_ARCH="x86_64"
        fi
        KERNEL_WHL=$(ls sgl-kernel/dist/sglang_kernel-${SGL_KERNEL_VERSION_FROM_KERNEL}+${CU_VERSION}-cp310-abi3-manylinux2014_${WHEEL_ARCH}.whl 2>/dev/null | head -1 || true)
        if [ -z "$KERNEL_WHL" ]; then
            echo "ERROR: No matching sgl-kernel wheel found in sgl-kernel/dist/ for version ${SGL_KERNEL_VERSION_FROM_KERNEL} arch ${WHEEL_ARCH} cuda ${CU_VERSION}"
            ls -alh sgl-kernel/dist/
            exit 1
        fi
        echo "Installing sgl-kernel wheel: $KERNEL_WHL"
        $PIP_CMD install "$KERNEL_WHL" --force-reinstall $PIP_INSTALL_SUFFIX
    else
        if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ] && [ ! -d "sgl-kernel/dist" ]; then
            echo "ERROR: CUSTOM_BUILD_SGL_KERNEL=true but sgl-kernel/dist not found."
            echo "This usually happens when rerunning a stage without the sgl-kernel-build-wheels job."
            echo "Please re-run the full workflow using /tag-and-rerun-ci to rebuild the kernel."
            exit 1
        fi
    fi

    # Reinstall torch with matching CUDA version if needed
    # TODO: Remove after torch 2.11 where cu13 is enabled by default
    TORCH_CUDA_VER=$(python3 -c "import torch; v=torch.version.cuda; parts=v.split('.'); print(f'cu{parts[0]}{parts[1]}')")
    echo "Detected torch CUDA version: ${TORCH_CUDA_VER}"
    TORCHAUDIO_CUDA_VER=$(pip show torchaudio 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed -n 's/.*+\(cu[0-9][0-9]*\)$/\1/p' || true)
    TORCHVISION_CUDA_VER=$(pip show torchvision 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed -n 's/.*+\(cu[0-9][0-9]*\)$/\1/p' || true)
    REINSTALL_TORCH=false
    if [ "${TORCH_CUDA_VER}" != "${CU_VERSION}" ]; then
        REINSTALL_TORCH=true
    else
        for cuda_ver in "${TORCHAUDIO_CUDA_VER}" "${TORCHVISION_CUDA_VER}"; do
            if [ -n "${cuda_ver}" ] && [ "${cuda_ver}" != "${CU_VERSION}" ]; then
                REINSTALL_TORCH=true
                break
            fi
        done
    fi
    if [ "${REINSTALL_TORCH}" = true ]; then
        TORCH_VER=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//')
        TORCHAUDIO_VER=$(pip show torchaudio 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//')
        TORCHVISION_VER=$(pip show torchvision 2>/dev/null | grep "^Version:" | awk '{print $2}' | sed 's/+.*//')
        echo "Reinstalling torch==${TORCH_VER} torchaudio==${TORCHAUDIO_VER} torchvision==${TORCHVISION_VER} from ${CU_VERSION} index to match torch..."
        $PIP_CMD install "torch==${TORCH_VER}" "torchaudio==${TORCHAUDIO_VER}" "torchvision==${TORCHVISION_VER}" --index-url "https://download.pytorch.org/whl/${CU_VERSION}" --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
    fi

    if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" != "true" ]; then
        # install_sglang above pulls sglang-kernel from PyPI, whose default wheel
        # tracks one CUDA version (currently cu130). Force-reinstall from the
        # CU_VERSION-matched sglang wheel index so runners on a different CUDA
        # (e.g. h20 / cu129) get a wheel linked against the right libnvrtc.
        $PIP_CMD install "sglang-kernel==${SGL_KERNEL_VERSION_FROM_SRT}" --index-url "https://docs.sglang.ai/whl/${CU_VERSION}/" --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
    else
        echo "CUSTOM_BUILD_SGL_KERNEL=true: keeping freshly built sgl-kernel wheel."
    fi
    SGL_DEEP_GEMM_VERSION=$(grep -Po -m1 '(?<=sgl-deep-gemm==)[0-9A-Za-z\.\-]+' python/pyproject.toml)
    if [ "$CU_MAJOR" = "13" ]; then
        $PIP_CMD install "sgl-deep-gemm==${SGL_DEEP_GEMM_VERSION}" --force-reinstall $PIP_INSTALL_SUFFIX
    else
        $PIP_CMD install "https://github.com/sgl-project/whl/releases/download/v${SGL_DEEP_GEMM_VERSION}/sgl_deep_gemm-${SGL_DEEP_GEMM_VERSION}+cu129-py3-none-manylinux2014_$(uname -m).whl" --force-reinstall $PIP_INSTALL_SUFFIX
    fi

    mark_step_done "${FUNCNAME[0]}"
}

install_sglang_router() {
    $PIP_CMD install sglang-router $PIP_INSTALL_SUFFIX
    $PIP_CMD list

    mark_step_done "${FUNCNAME[0]}"
}

download_flashinfer_cache() {
    UNINSTALL_JIT_CACHE="$UNINSTALL_JIT_CACHE" \
        FLASHINFER_PYTHON_REQUIRED="$FLASHINFER_PYTHON_REQUIRED" \
        CU_VERSION="$CU_VERSION" \
        PIP_CMD="$PIP_CMD" \
        PIP_INSTALL_SUFFIX="$PIP_INSTALL_SUFFIX" \
        bash "${SCRIPT_DIR}/ci_download_flashinfer_jit_cache.sh"

    mark_step_done "${FUNCNAME[0]}"
}

stabilize_flashinfer_jit_paths() {
    # In venv mode, FlashInfer JIT writes build.ninja with hardcoded -isystem
    # paths. Per-job venvs get unique paths, but the JIT cache is shared on the
    # host mount. Fix by symlinking venv copies to a stable host-mounted path.
    if [ "$USE_VENV" != "1" ]; then
        return
    fi

    STABLE_FI_DIR="${HOME}/.cache/flashinfer/_stable_src"

    # Clear stale cached_ops (keep valid compiled kernels)
    if [ -d "${HOME}/.cache/flashinfer" ]; then
        STALE_COUNT=0
        while IFS= read -r ninja_file; do
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

    # Copy source files to stable path and symlink venv copies there
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

    mark_step_done "${FUNCNAME[0]}"
}

install_extra_deps() {
    if [ "$CU_MAJOR" = "13" ]; then
        MOONCAKE_PKG="mooncake-transfer-engine-cuda13==0.3.10.post2"
        MOONCAKE_STALE_PKG="mooncake-transfer-engine"
        EXTRA_NVIDIA_SPECS="nvidia-cuda-nvrtc"
    else
        MOONCAKE_PKG="mooncake-transfer-engine==0.3.10.post2"
        MOONCAKE_STALE_PKG="mooncake-transfer-engine-cuda13"
        EXTRA_NVIDIA_SPECS="nvidia-cuda-nvrtc-cu12"
    fi
    # Both variants own the same mooncake/ package files and bin/ scripts
    # (mooncake_master, etc.). Uninstalling the stale variant deletes shared
    # files that the live variant's RECORD still references, so we force a
    # reinstall to restore them — pip would otherwise see "already satisfied"
    # and skip.
    if pip show ${MOONCAKE_STALE_PKG} >/dev/null 2>&1; then
        $PIP_UNINSTALL_CMD ${MOONCAKE_STALE_PKG} $PIP_UNINSTALL_SUFFIX || true
        $PIP_CMD install ${MOONCAKE_PKG} --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
    fi
    $PIP_CMD install ${MOONCAKE_PKG} ${EXTRA_NVIDIA_SPECS} py-spy scipy huggingface_hub[hf_xet] pytest $PIP_INSTALL_SUFFIX

    # Best-effort NIXL install for decode-radix disaggregation coverage.
    $PIP_CMD install nixl $PIP_INSTALL_SUFFIX || echo "Warning: nixl install failed; continuing without nixl"

    if [ "$IS_BLACKWELL" != "1" ]; then
        git clone --branch v0.5 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
        $PIP_CMD install -e lmms-eval/ $PIP_INSTALL_SUFFIX
    fi
    $PIP_CMD uninstall xformers || true

    mark_step_done "${FUNCNAME[0]}"
}

install_test_tools() {
    # Download kernels from kernels community
    kernels download python || true
    kernels lock python || true
    [ -e "${HOME}/.cache/sglang" ] && [ ! -d "${HOME}/.cache/sglang" ] && rm -f "${HOME}/.cache/sglang"
    mkdir -p "${HOME}/.cache/sglang/"
    mv python/kernels.lock "${HOME}/.cache/sglang/" || true

    # Install human-eval (subshell keeps cd local)
    $PIP_CMD install "setuptools==70.0.0" $PIP_INSTALL_SUFFIX
    [ -d human-eval ] || git clone https://github.com/merrymercy/human-eval.git
    (
        cd human-eval
        $PIP_CMD install -e . --no-build-isolation $PIP_INSTALL_SUFFIX
    )

    mark_step_done "${FUNCNAME[0]}"
}

prepare_runner() {
    bash "${SCRIPT_DIR}/prepare_runner.sh"

    mark_step_done "${FUNCNAME[0]}"
}

setup_ld_library_path() {
    # NVIDIA pip packages and torch ship .so files under site-packages that are
    # not on the default LD_LIBRARY_PATH.
    SITE_PACKAGES=$(python3 -c "import site, sys; print(site.getsitepackages()[0])")
    NVIDIA_LIBS=$(find "$SITE_PACKAGES" -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\n' ':')
    TORCH_LIB="$SITE_PACKAGES/torch/lib"
    VENV_LD="${NVIDIA_LIBS}${TORCH_LIB}"
    export LD_LIBRARY_PATH="${VENV_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    if [ "$USE_VENV" = "1" ] && [ -n "$UV_VENV" ]; then
        echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"" >> "$UV_VENV/env.sh"
    fi
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$GITHUB_ENV" || echo "WARNING: GITHUB_ENV write failed; LD_LIBRARY_PATH will be set via BASH_ENV instead"
    fi
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

    mark_step_done "${FUNCNAME[0]}"
}

verify_imports() {
    $PIP_CMD list
    python3 -c "import torch; print(torch.version.cuda)"
    python3 -c "import cutlass; import cutlass.cute;"

    mark_step_done "${FUNCNAME[0]}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    configure_environment "$@"
    detect_host
    kill_existing_processes
    install_apt_packages
    clean_site_packages
    setup_pip_toolchain
    uninstall_stale_flashinfer
    install_sglang
    install_sglang_kernel
    install_sglang_router
    download_flashinfer_cache
    stabilize_flashinfer_jit_paths
    install_extra_deps
    install_test_tools
    prepare_runner
    setup_ld_library_path
    verify_imports
}

main "$@"
