#!/bin/bash
# Install dependencies for CUDA CI jobs.
#
# CU_VERSION (default: cu130) controls PyTorch index URL, FlashInfer JIT cache
# index, and nvrtc variant selection.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# shellcheck source=scripts/ci/utils/git_clone_with_retry.sh
source "${SCRIPT_DIR}/../utils/git_clone_with_retry.sh"

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
            # Self-heal: see install_rustup.sh for context on missing _runner_file_commands/.
            mkdir -p "$(dirname "$GITHUB_ENV")" 2>/dev/null || true
            echo "VIRTUAL_ENV=$UV_VENV" >> "$GITHUB_ENV" || true
            echo "SGLANG_CI_VENV_PATH=$UV_VENV" >> "$GITHUB_ENV" || true
            echo "BASH_ENV=$UV_VENV/env.sh" >> "$GITHUB_ENV" || true
            touch "$UV_VENV/env.sh"
        fi
        if [ -n "${GITHUB_PATH:-}" ]; then
            mkdir -p "$(dirname "$GITHUB_PATH")" 2>/dev/null || true
            echo "$UV_VENV/bin" >> "$GITHUB_PATH" || true
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

cleanup_stale_shm() {
    # Reclaim /dev/shm segments leaked by SIGKILLed processes from earlier
    # jobs; leaked segments accumulate until the tmpfs fills and scheduler
    # init dies with SIGBUS. Runs right after killall so every dead creator's
    # segments are reclaimable. The module is dependency-free and runnable by
    # path, so this works before sglang is installed.
    SGLANG_IS_IN_CI=true python3 "${REPO_ROOT}/python/sglang/srt/utils/stale_shm_cleanup.py" || true

    mark_step_done "${FUNCNAME[0]}"
}

install_apt_packages() {
    CI_APT_PACKAGES=(
        python3 python3-pip python3-venv python3-dev git libnuma-dev libssl-dev pkg-config
        build-essential cmake rdma-core infiniband-diags perftest libibumad3
        libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils
        libfabric-dev libnl-3-200 libnl-route-3-200 librdmacm1
        ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
    )

    # The images bake these in, so the usual run pays apt-get update's round
    # trips to install nothing. Skipping it costs no currency either: apt-get
    # install only ever considers the packages named above, and a passing run
    # leaves 100+ others un-upgraded - the image is what pins these versions.
    local pkg
    local -a MISSING_APT_PACKAGES=()
    for pkg in "${CI_APT_PACKAGES[@]}"; do
        dpkg -l "$pkg" 2>/dev/null | grep -q "^ii" || MISSING_APT_PACKAGES+=("$pkg")
    done

    if [ ${#MISSING_APT_PACKAGES[@]} -eq 0 ]; then
        echo "All required apt packages are already installed, skipping apt-get"
    else
        echo "Installing missing apt packages: ${MISSING_APT_PACKAGES[*]}"
        apt-get update || true
        apt-get install -y --no-install-recommends "${MISSING_APT_PACKAGES[@]}" || {
            echo "ERROR: apt-get failed to install: ${MISSING_APT_PACKAGES[*]}"
            exit 1
        }
    fi

    mark_step_done "${FUNCNAME[0]}"
}

install_gdrcopy() {
    # DeepEP tests only run on 4+ GPU hosts. Keep GDRCopy in the shared CUDA
    # bootstrap while avoiding a DKMS/package build on the 1- and 2-GPU jobs.
    local gpu_count=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count=$(
            (nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true) |
                awk 'NF {count++} END {print count + 0}'
        )
    fi
    if [ "${gpu_count}" -lt 4 ]; then
        echo "Skipping GDRCopy install on ${gpu_count}-GPU runner"
        mark_step_done "${FUNCNAME[0]}"
        return
    fi

    if ldconfig -p 2>/dev/null | grep 'libgdrapi\.so' >/dev/null; then
        echo "GDRCopy userspace library is already installed"
        mark_step_done "${FUNCNAME[0]}"
        return
    fi

    local gdrcopy_root=/opt/gdrcopy
    local gdrcopy_version=2.5.1
    local -a gdrcopy_packages=(
        nvidia-dkms-580 devscripts debhelper fakeroot dkms
        check libsubunit0 libsubunit-dev python3-venv
    )

    apt-get update || true
    apt-get install -y --no-install-recommends "${gdrcopy_packages[@]}" || {
        echo "Warning: apt-get failed while installing GDRCopy build dependencies; checking installed packages"
        local package
        for package in "${gdrcopy_packages[@]}"; do
            if ! dpkg -l "${package}" 2>/dev/null | grep -q '^ii'; then
                echo "ERROR: Required GDRCopy package ${package} is unavailable"
                exit 1
            fi
        done
    }

    git_clone_with_retry https://github.com/NVIDIA/gdrcopy.git "${gdrcopy_root}" "--branch v${gdrcopy_version}"
    (
        cd "${gdrcopy_root}/packages"
        CUDA=/usr/local/cuda ./build-deb-packages.sh
        dpkg -i gdrdrv-dkms_*.deb
        dpkg -i libgdrapi_*.deb
        dpkg -i gdrcopy-tests_*.deb
        dpkg -i gdrcopy_*.deb
    )

    local lib_path="/usr/lib/${ARCH}-linux-gnu"
    if [ ! -e "${lib_path}/libmlx5.so" ] && [ -e "${lib_path}/libmlx5.so.1" ]; then
        ln -s "${lib_path}/libmlx5.so.1" "${lib_path}/libmlx5.so"
    fi
    ldconfig

    mark_step_done "${FUNCNAME[0]}"
}

clean_site_packages() {
    # The torch compilation cache is deliberately NOT wiped here: entries are
    # content-hash addressed so stale ones are never reused, and hosts packing
    # several runners share one cache mount - a wipe unlinks files a concurrent
    # job is compiling against.

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

    # An orphaned sglang/ shadows the checkout: `uv pip uninstall` deletes only
    # what RECORD lists, so __pycache__ keeps the directory alive without
    # __init__.py, PathFinder claims it as a namespace portion, and the editable
    # install's _EditableFinder sits at the END of sys.meta_path - submodule
    # imports fail. No install leaves this directory without __init__.py.
    if [ -d "$SITE_PACKAGES/sglang" ] && [ ! -f "$SITE_PACKAGES/sglang/__init__.py" ]; then
        echo "Removing orphaned sglang skeleton that would shadow the checkout: $SITE_PACKAGES/sglang"
        find "$SITE_PACKAGES/sglang" -maxdepth 2 | head -20
        rm -rf "$SITE_PACKAGES/sglang"
    fi

    # Install protoc + Rust toolchain (needed by setuptools-rust, e.g. the native gRPC extension)
    bash "${SCRIPT_DIR}/../utils/install_rust_protoc.sh"
    export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

    # Same-step counterpart of the PATH export above: install_rustup.sh exports
    # RUSTUP_TOOLCHAIN too, but as a child process only reaches later steps.
    # rust-toolchain.toml does not cover it either - setuptools-rust runs cargo
    # from python/, outside the pin's cwd scope - so without this an image's own
    # older rustc builds the crates and fails their MSRV.
    RUST_PINNED_CHANNEL=$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' "${REPO_ROOT}/rust/rust-toolchain.toml" 2>/dev/null || true)
    if [ -n "${RUST_PINNED_CHANNEL}" ]; then
        export RUSTUP_TOOLCHAIN="${RUST_PINNED_CHANNEL}"
        echo "Using pinned Rust toolchain ${RUST_PINNED_CHANNEL} for this install"
    fi

    mark_step_done "${FUNCNAME[0]}"
}

setup_cargo_cache() {
    if [ "${SGLANG_BUILD_RUST_EXTS:-}" = "none" ]; then
        echo "Using prebuilt Rust extensions; skipping Cargo target setup"
        mark_step_done "${FUNCNAME[0]}"
        return
    fi

    # actions/checkout's `git clean -ffdx` deletes the gitignored in-repo
    # rust/target, so every job recompiles the whole dependency graph. Move the
    # target dir out of the tree: setuptools-rust has no target-dir option of its
    # own and defers to CARGO_TARGET_DIR, which uv passes to the build backend.
    export CARGO_TARGET_DIR="${HOME}/.cache/sglang-cargo-target"
    local cargo_target_lock="${HOME}/.cache/sglang-cargo-target.lock"
    mkdir -p "${HOME}/.cache"
    exec 9>"${cargo_target_lock}"
    echo "Waiting for exclusive cargo target lock: ${cargo_target_lock}"
    flock --exclusive 9
    CARGO_TARGET_LOCK_HELD=1
    echo "Acquired cargo target lock"
    mkdir -p "${CARGO_TARGET_DIR}"

    # Same disk-pressure guard as the uv cache in ci_cleanup_venv.sh (which
    # carries the ENOSPC story). cargo cannot prune partially, so drop the whole
    # tree and pay one cold build.
    local used
    used="$(df --output=pcent "${CARGO_TARGET_DIR}" 2>/dev/null | tr -dc '0-9')"
    if [ "${used:-0}" -ge 85 ]; then
        echo "cargo target dir filesystem at ${used}%; dropping ${CARGO_TARGET_DIR}"
        rm -rf "${CARGO_TARGET_DIR}"
        mkdir -p "${CARGO_TARGET_DIR}"
    fi

    mark_step_done "${FUNCNAME[0]}"
}

release_cargo_cache_lock() {
    if [ "${CARGO_TARGET_LOCK_HELD:-0}" = "1" ]; then
        flock --unlock 9
        exec 9>&-
        CARGO_TARGET_LOCK_HELD=0
        echo "Released cargo target lock"
    fi
}

setup_pip_toolchain() {
    if [ "$USE_VENV" = "1" ]; then
        # The bootstrap upgrade hit system pip; this upgrades the venv's own.
        python3 -m pip install --upgrade pip
    fi

    if [ "$USE_VENV" != "1" ]; then
        export UV_SYSTEM_PYTHON=1
    fi

    export UV_LINK_MODE=copy
    PIP_CMD="uv pip"
    PIP_INSTALL_SUFFIX="--index-strategy unsafe-best-match"
    PIP_UNINSTALL_CMD="uv pip uninstall"
    PIP_UNINSTALL_SUFFIX=""

    # Remove both the legacy source distribution and the SGLang wheel before
    # resolving the pyproject pin. They own the same deep_ep module files, so
    # leaving either installed can make pip preserve a mixed installation.
    $PIP_UNINSTALL_CMD deep-ep sgl-deep-ep $PIP_UNINSTALL_SUFFIX || true

    # sglang-kernel stays: install_sglang_kernel version-gates and reinstalls it.
    $PIP_UNINSTALL_CMD sgl-kernel sglang sgl-fa4 flash-attn-4 $PIP_UNINSTALL_SUFFIX || true

    mark_step_done "${FUNCNAME[0]}"
}

remove_stale_cuda12_nvidia_wheels() {
    local package_name spec
    local -a INSTALLED_NVIDIA_WHEELS=()
    local -a NVIDIA_WHEELS_TO_RESTORE=()
    local -a STALE_CUDA12_NVIDIA_WHEELS=()

    if [ "$CU_MAJOR" != "13" ]; then
        mark_step_done "${FUNCNAME[0]}"
        return
    fi

    mapfile -t INSTALLED_NVIDIA_WHEELS < <(
        python3 -m pip list --format=freeze | sed -n '/^nvidia-.*==/p'
    )
    for spec in "${INSTALLED_NVIDIA_WHEELS[@]}"; do
        package_name="${spec%%==*}"
        case "$package_name" in
            *-cu12) STALE_CUDA12_NVIDIA_WHEELS+=("$package_name") ;;
            *) NVIDIA_WHEELS_TO_RESTORE+=("$spec") ;;
        esac
    done

    if [ ${#STALE_CUDA12_NVIDIA_WHEELS[@]} -eq 0 ]; then
        echo "No stale CUDA 12 NVIDIA wheels found for ${CU_VERSION} job"
        mark_step_done "${FUNCNAME[0]}"
        return
    fi

    echo "Removing stale CUDA 12 NVIDIA wheels from ${CU_VERSION} job: ${STALE_CUDA12_NVIDIA_WHEELS[*]}"
    $PIP_UNINSTALL_CMD "${STALE_CUDA12_NVIDIA_WHEELS[@]}" $PIP_UNINSTALL_SUFFIX

    # CUDA 12 and CUDA 13 wheels can own the same nvidia/* paths. Uninstalling
    # the stale variant deletes those shared files even though the remaining
    # wheel metadata still says they are installed. Restore every remaining
    # NVIDIA wheel at its already-installed version to make the transition
    # atomic and avoid package-specific payload checks.
    if [ ${#NVIDIA_WHEELS_TO_RESTORE[@]} -gt 0 ]; then
        echo "Restoring NVIDIA wheels after CUDA 12 cleanup: ${NVIDIA_WHEELS_TO_RESTORE[*]}"
        $PIP_CMD install --force-reinstall --no-deps "${NVIDIA_WHEELS_TO_RESTORE[@]}" $PIP_INSTALL_SUFFIX
    fi

    mark_step_done "${FUNCNAME[0]}"
}

uninstall_stale_flashinfer() {
    # Keep flashinfer packages if version matches to avoid re-downloading:
    # - flashinfer-cubin: 150+ MB
    # - flashinfer-jit-cache: 1.2+ GB
    FLASHINFER_PYTHON_REQUIRED=$(grep -Po -m1 'flashinfer_python(\[[^]]+\])?==\K[0-9A-Za-z\.\-]+' python/pyproject.toml || echo "")
    # flashinfer-cubin is no longer a pyproject dependency (installed explicitly below), tracks the same version as flashinfer_python
    FLASHINFER_CUBIN_REQUIRED="$FLASHINFER_PYTHON_REQUIRED"
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

install_pytorch_stack() {
    PYTORCH_SPECS=()
    for package in torch torchaudio torchvision torchcodec; do
        spec=$(grep -Po -m1 "\"${package}([<>=!~ ;][^\"]*)?\"" python/pyproject.toml | tr -d '"' || true)
        if [ -n "$spec" ]; then
            PYTORCH_SPECS+=("$spec")
        fi
    done

    $PIP_CMD install \
        "${PYTORCH_SPECS[@]}" \
        --index-url "https://download.pytorch.org/whl/${CU_VERSION}"

    mark_step_done "${FUNCNAME[0]}"
}

install_cuda12_deepep_wheel() {
    if [ "$CU_MAJOR" = "13" ]; then
        echo "CUDA 13 uses the public sgl-deep-ep wheel declared in python/pyproject.toml"
        mark_step_done "${FUNCNAME[0]}"
        return
    fi

    local version
    version=$(grep -Po -m1 '"sgl-deep-ep==\K[^"]+' python/pyproject.toml || true)
    if [ -z "$version" ]; then
        echo "ERROR: python/pyproject.toml must pin sgl-deep-ep"
        exit 1
    fi

    # CUDA 12 wheels intentionally live only on the SGLang wheel index. Their
    # local version satisfies the public-version pyproject pin, so the later
    # editable SGLang install keeps this CUDA-matched wheel.
    $PIP_CMD install "sgl-deep-ep==${version}+${CU_VERSION}" \
        --index-url "https://docs.sglang.ai/whl/${CU_VERSION}/" \
        --force-reinstall --no-deps $PIP_INSTALL_SUFFIX

    mark_step_done "${FUNCNAME[0]}"
}

require_prebuilt_rust_exts() {
    # Stages whose download succeeded set this to none. Runs before
    # setup_pip_toolchain uninstalls sglang, so clearing it here still reaches
    # install_sglang below - setup.py reads it from the environment at build time.
    if [ "${SGLANG_BUILD_RUST_EXTS:-}" != "none" ]; then
        mark_step_done "${FUNCNAME[0]}"
        return
    fi

    # Exact EXT_SUFFIX rather than an _*.so glob: no crate sets abi3, so a module
    # built for another minor version satisfies the glob while the import system
    # ignores it, leaving is_rust_server_built() false and the Rust-server tests
    # silently skipped. Stages have no setup-python, so the interpreter is whatever
    # the image ships, and the pools are not on one version (h20 is 3.12 while
    # h100 is 3.10) - a mismatch is drift to route around, not a failure.
    local suffix
    suffix=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
    local missing=()
    local module
    for module in server grpc multimodal; do
        [ -f "python/sglang/srt/rust_extensions/_${module}${suffix}" ] || missing+=("${module}")
    done
    [ -f "python/sglang/srt/mem_cache/rust_tree_core/mem_cache${suffix}" ] \
        || missing+=("mem_cache")
    [ -f "python/sglang/srt/mem_cache/rust_tree_core/mem_cache_inspection${suffix}" ] \
        || missing+=("mem_cache_inspection")
    if [ ${#missing[@]} -gt 0 ]; then
        echo "::warning::no prebuilt Rust extension ${suffix} for: ${missing[*]}; building from source"
        ls -l python/sglang/srt/rust_extensions/_*.so 2>/dev/null || echo "(no extension modules at all)"
        ls -l python/sglang/srt/mem_cache/rust_tree_core/mem_cache*.so 2>/dev/null || true
        export SGLANG_BUILD_RUST_EXTS=
        export SGLANG_RUST_BUILD_MODE=auto
        if [ -n "${GITHUB_ENV:-}" ]; then
            echo "SGLANG_RUST_BUILD_MODE=auto" >> "${GITHUB_ENV}"
        fi
        mark_step_done "${FUNCNAME[0]}"
        return
    fi
    echo "Using prebuilt Rust extension modules; skipping the cargo build."

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

install_nccl() {
    if [ "$CU_MAJOR" = "13" ]; then
        # PyTorch pins 2.29.7, so this override must run after every command
        # that resolves Python dependencies (including lmms-eval).
        $PIP_CMD install "nvidia-nccl-cu13==2.30.7" \
            --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
    else
        echo "CUDA ${CU_MAJOR} does not require the NCCL Gin wheel"
    fi

    mark_step_done "${FUNCNAME[0]}"
}

# Trust an installed wheel only if the version matches and every RECORD file is
# on disk (dist-info can survive a partial install - cf. the cusparselt guard).
# reject-local refuses wheels installed from a local file: a kernel-PR job
# installs its own build under the SAME +cuXXX version string, and only file://
# provenance (PEP 610; index installs record none) tells them apart.
installed_wheel_ok() {
    WHEEL_DIST="$1" WHEEL_WANTED="$2" WHEEL_REJECT_LOCAL="${3:-}" python3 - <<'EOF'
import importlib.metadata as md
import os
import sys

name = os.environ["WHEEL_DIST"]
wanted = os.environ["WHEEL_WANTED"]
try:
    dist = md.distribution(name)
except md.PackageNotFoundError:
    print(f"{name} not installed (required: {wanted}); installing")
    sys.exit(1)
if dist.version != wanted:
    print(f"{name} mismatch (installed: {dist.version}, required: {wanted}); reinstalling")
    sys.exit(1)
if os.environ["WHEEL_REJECT_LOCAL"] and "file://" in (dist.read_text("direct_url.json") or ""):
    print(f"{name} came from a locally built wheel; reinstalling from the index")
    sys.exit(1)
if dist.files is None:
    print(f"{name} has no RECORD to verify; reinstalling")
    sys.exit(1)
missing = [str(f) for f in dist.files if not dist.locate_file(f).exists()]
if missing:
    print(f"{name} is missing {len(missing)} installed files (e.g. {missing[0]}); reinstalling")
    sys.exit(1)
EOF
}

install_sglang_kernel() {
    SGL_KERNEL_VERSION_FROM_KERNEL=$(grep -Po '(?<=^version = ")[^"]*' python/sglang/kernels/aot/pyproject.toml)
    SGL_KERNEL_VERSION_FROM_SRT=$(grep -Po -m1 '(?<=sglang-kernel==)[0-9A-Za-z\.\-]+' python/pyproject.toml)
    echo "SGL_KERNEL_VERSION_FROM_KERNEL=${SGL_KERNEL_VERSION_FROM_KERNEL} SGL_KERNEL_VERSION_FROM_SRT=${SGL_KERNEL_VERSION_FROM_SRT}"

    if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ] && [ -d "python/sglang/kernels/aot/dist" ]; then
        ls -alh python/sglang/kernels/aot/dist
        if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
            WHEEL_ARCH="aarch64"
        else
            WHEEL_ARCH="x86_64"
        fi
        KERNEL_WHL=$(ls python/sglang/kernels/aot/dist/sglang_kernel-${SGL_KERNEL_VERSION_FROM_KERNEL}+${CU_VERSION}-cp310-abi3-manylinux2014_${WHEEL_ARCH}.whl 2>/dev/null | head -1 || true)
        if [ -z "$KERNEL_WHL" ]; then
            echo "ERROR: No matching sgl-kernel wheel found in python/sglang/kernels/aot/dist/ for version ${SGL_KERNEL_VERSION_FROM_KERNEL} arch ${WHEEL_ARCH} cuda ${CU_VERSION}"
            ls -alh python/sglang/kernels/aot/dist/
            exit 1
        fi
        echo "Installing sgl-kernel wheel: $KERNEL_WHL"
        $PIP_CMD install "$KERNEL_WHL" --force-reinstall $PIP_INSTALL_SUFFIX
    else
        if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" = "true" ] && [ ! -d "python/sglang/kernels/aot/dist" ]; then
            echo "ERROR: CUSTOM_BUILD_SGL_KERNEL=true but python/sglang/kernels/aot/dist not found."
            echo "This usually happens when rerunning a stage without the sgl-kernel-build-wheels job."
            echo "Please re-run the full workflow using /tag-and-rerun-ci to rebuild the kernel."
            exit 1
        fi
    fi

    if [ "${CUSTOM_BUILD_SGL_KERNEL:-}" != "true" ]; then
        # The PyPI default wheel tracks one CUDA version (currently cu130); other
        # runners (e.g. h20 / cu129) need the +${CU_VERSION}-tagged wheel from the
        # sglang index, linked against the right libnvrtc.
        SGL_KERNEL_WANTED="${SGL_KERNEL_VERSION_FROM_SRT}+${CU_VERSION}"
        if installed_wheel_ok sglang-kernel "${SGL_KERNEL_WANTED}" reject-local; then
            echo "sglang-kernel==${SGL_KERNEL_WANTED} already installed, keeping it"
        else
            $PIP_CMD install "sglang-kernel==${SGL_KERNEL_VERSION_FROM_SRT}" --index-url "https://docs.sglang.ai/whl/${CU_VERSION}/" --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
        fi
    else
        echo "CUSTOM_BUILD_SGL_KERNEL=true: keeping freshly built sgl-kernel wheel."
    fi
    SGL_DEEP_GEMM_VERSION=$(grep -Po -m1 '(?<=sgl-deep-gemm==)[0-9A-Za-z\.\-]+' python/pyproject.toml)
    if [ "$CU_MAJOR" = "13" ]; then
        SGL_DEEP_GEMM_WANTED="${SGL_DEEP_GEMM_VERSION}"
    else
        SGL_DEEP_GEMM_WANTED="${SGL_DEEP_GEMM_VERSION}+cu129"
    fi
    # No reject-local: nothing builds sgl-deep-gemm locally.
    if installed_wheel_ok sgl-deep-gemm "${SGL_DEEP_GEMM_WANTED}"; then
        echo "sgl-deep-gemm==${SGL_DEEP_GEMM_WANTED} already installed, keeping it"
    elif [ "$CU_MAJOR" = "13" ]; then
        $PIP_CMD install "sgl-deep-gemm==${SGL_DEEP_GEMM_VERSION}" --force-reinstall $PIP_INSTALL_SUFFIX
    else
        $PIP_CMD install "https://github.com/sgl-project/whl/releases/download/v${SGL_DEEP_GEMM_VERSION}/sgl_deep_gemm-${SGL_DEEP_GEMM_VERSION}+cu129-py3-none-manylinux2014_$(uname -m).whl" --force-reinstall $PIP_INSTALL_SUFFIX
    fi

    mark_step_done "${FUNCNAME[0]}"
}

install_sglang_router() {
    $PIP_CMD install sglang-router $PIP_INSTALL_SUFFIX

    mark_step_done "${FUNCNAME[0]}"
}

install_flashinfer_cubin() {
    if [ "$UNINSTALL_CUBIN" = false ]; then
        echo "flashinfer-cubin==${FLASHINFER_CUBIN_REQUIRED} already installed, skipping install"
    else
        # flashinfer-cubin is CUDA-version-agnostic, unlike jit-cache, so its index-url has no cu${CU_VERSION} suffix
        $PIP_CMD install "flashinfer-cubin==${FLASHINFER_CUBIN_REQUIRED}" --index-url https://flashinfer.ai/whl $PIP_INSTALL_SUFFIX
    fi

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
    MOONCAKE_VERSION="0.3.13"
    NIXL_VERSION="1.3.0"
    if [ "$CU_MAJOR" = "13" ]; then
        MOONCAKE_PKG="mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
        MOONCAKE_STALE_PKG="mooncake-transfer-engine"
        NIXL_BIN_NAME="nixl-cu13"
        EXTRA_NVIDIA_SPECS="nvidia-cuda-nvrtc"
    else
        MOONCAKE_PKG="mooncake-transfer-engine==${MOONCAKE_VERSION}"
        MOONCAKE_STALE_PKG="mooncake-transfer-engine-cuda13"
        NIXL_BIN_NAME="nixl-cu12"
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

    NIXL_INSTALLED=$(pip show nixl 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
    NIXL_BIN_INSTALLED=$(pip show "${NIXL_BIN_NAME}" 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
    if [ "$NIXL_INSTALLED" = "$NIXL_VERSION" ] && [ "$NIXL_BIN_INSTALLED" = "$NIXL_VERSION" ]; then
        echo "nixl==${NIXL_VERSION} and ${NIXL_BIN_NAME}==${NIXL_VERSION} already installed, keeping them"
    else
        echo "nixl mismatch (meta: ${NIXL_INSTALLED:-none}, ${NIXL_BIN_NAME}: ${NIXL_BIN_INSTALLED:-none}, required: ${NIXL_VERSION}); installing"
        # Meta stub owns the nixl import path; install only the CUDA binary for
        # this runner's torch CUDA major. --no-deps avoids pulling the other CUDA
        # variant; leave any other variant already on the runner image untouched.
        $PIP_CMD install "nixl==${NIXL_VERSION}" "${NIXL_BIN_NAME}==${NIXL_VERSION}" \
            --no-deps --force-reinstall $PIP_INSTALL_SUFFIX
    fi

    if [ "$IS_BLACKWELL" != "1" ]; then
        $PIP_CMD install "lmms_eval==0.5.0" $PIP_INSTALL_SUFFIX
        # lmms_eval 0.5.0 pulls antlr4-python3-runtime==4.7.2, clobbering the
        # 4.9.3 that sgl-eval's latex2sympy2_extended needs (4.7.2 ImportError
        # at sgl-eval import). Pin it back so the nightly sgl-eval path works.
        $PIP_CMD install "antlr4-python3-runtime==4.9.3" --force-reinstall --no-deps $PIP_INSTALL_SUFFIX
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

    mark_step_done "${FUNCNAME[0]}"
}

prepare_runner() {
    bash "${SCRIPT_DIR}/prepare_runner.sh"

    mark_step_done "${FUNCNAME[0]}"
}

setup_ld_library_path() {
    # NVIDIA pip packages and torch ship .so files under site-packages that are
    # not on the default LD_LIBRARY_PATH; lib/ always nests under nvidia/.
    SITE_PACKAGES=$(python3 -c "import site, sys; print(site.getsitepackages()[0])")
    NVIDIA_LIBS=$( (find "$SITE_PACKAGES/nvidia" -type d -name lib 2>/dev/null || true) | tr '\n' ':')
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

    # One process; torch/cutlass do not import sglang, so the find_spec check
    # still runs ahead of any sglang import.
    SGLANG_EXPECTED_INIT="${REPO_ROOT}/python/sglang/__init__.py" python3 -c '
import ctypes
import importlib.metadata
import os
import sys

if sys.argv[1] == "13":
    if importlib.metadata.version("nvidia-nccl-cu13") != "2.30.7":
        raise SystemExit("nvidia-nccl-cu13 was changed after the final CI override")
    nccl = ctypes.CDLL("libnccl.so.2")
    nccl_version = ctypes.c_int()
    status = nccl.ncclGetVersion(ctypes.byref(nccl_version))
    if status != 0 or nccl_version.value != 23007:
        raise SystemExit(
            f"expected NCCL runtime 2.30.7, got status={status}, "
            f"raw_version={nccl_version.value}"
        )
    print("NCCL package and runtime versions are 2.30.7")

import torch
print(torch.version.cuda)
import deep_ep
print(f"deep_ep loads from {deep_ep.__file__}")
import cutlass
import cutlass.cute

# A shadowed sglang still imports, so without this the failure only surfaces
# as a missing submodule during the test step. find_spec, not import: the
# finders alone answer this without importing sglang.
import importlib.util
want = os.environ["SGLANG_EXPECTED_INIT"]
spec = importlib.util.find_spec("sglang")
if spec is None:
    raise SystemExit("sglang is not importable at all after install")
if spec.origin != want:
    raise SystemExit(
        f"sglang resolves to origin={spec.origin} "
        f"(search={list(spec.submodule_search_locations or [])}), expected {want}; "
        "something in site-packages is shadowing the checkout"
    )
print(f"sglang resolves to {spec.origin}")

# Import, not find_spec: the finders locate an extension without dlopening it,
# so a .so that cannot load passes find_spec and only fails inside some suite.
import importlib
for mod in ("server", "grpc", "multimodal"):
    name = f"sglang.srt.rust_extensions._{mod}"
    try:
        importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"{name} is present but does not load: {exc!r}")
    print(f"{name} loads")
' "$CU_MAJOR"

    mark_step_done "${FUNCNAME[0]}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    configure_environment "$@"
    detect_host
    kill_existing_processes
    cleanup_stale_shm
    install_apt_packages
    install_gdrcopy
    clean_site_packages
    require_prebuilt_rust_exts
    setup_pip_toolchain
    remove_stale_cuda12_nvidia_wheels
    uninstall_stale_flashinfer
    install_pytorch_stack
    install_cuda12_deepep_wheel
    setup_cargo_cache
    install_sglang
    release_cargo_cache_lock
    # Diffusion B200 CI imports torch inside install_sglang_kernel after removing
    # stale CUDA 12 NVIDIA wheels, so opt into one early LD_LIBRARY_PATH refresh.
    if [ "${SGLANG_CI_EARLY_LD_LIBRARY_PATH:-0}" = "1" ]; then
        setup_ld_library_path
    fi
    install_sglang_kernel
    install_sglang_router
    install_flashinfer_cubin
    download_flashinfer_cache
    stabilize_flashinfer_jit_paths
    install_extra_deps
    install_test_tools
    install_nccl
    prepare_runner
    setup_ld_library_path
    verify_imports
}

main "$@"
