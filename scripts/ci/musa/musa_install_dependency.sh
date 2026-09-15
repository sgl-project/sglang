#!/bin/bash
# shellcheck disable=SC2034  # OPTIONAL_DEPS is retained for CLI compatibility.
set -euo pipefail

# Parse command line arguments
OPTIONAL_DEPS=""
SKIP_SGLANG_BUILD=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-sglang-build) SKIP_SGLANG_BUILD="1"; shift;;
    -h|--help)
      echo "Usage: $0 [OPTIONS] [OPTIONAL_DEPS]"
      echo "Options:"
      echo "  --skip-sglang-build         Don't build checkout sglang, use what was shipped with the image"
      exit 0
      ;;
    *)
      OPTIONAL_DEPS="$1"
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_HELPER="${SCRIPT_DIR}/musa_python_stack.py"
# Keep installed packages isolated in PYTHONUSERBASE, but let pip reuse its
# content-addressed wheel cache across jobs. Disabling the cache made every
# MUSA lane cold-download hundreds of megabytes and exceed the install timeout.
PIP_INSTALL=(python3 -m pip install)
readonly MUSA_TRITON_VERSION="3.2.0"
# Current MUSA CI uses the CPython 3.10 x86_64 wheel. A Python upgrade must
# update this digest together with the pinned artifact.
readonly MUSA_TRITON_SHA256="65b15d42fac24a2eca4c0c9f0ac68c8bd7cbe6bcc9f619c3483fb4f323391303"
readonly MUSA_TRITON_INDEX_URL="https://dl.mthreads.com/repo/api/pypi/pypi/simple"
readonly MUSA_TORCHADA_VERSION="0.1.82"
readonly MUSA_TORCHADA_SHA256="472663da083ef23502f08429618a36e5e6e9b2447cf72fff2568787c815b5903"
readonly MUSA_TORCHADA_INDEX_URL="https://pypi.org/simple"
readonly MUSA_SETUPTOOLS_SPEC="setuptools<82"
MUSA_CI_SCRATCH="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/sglang-musa-ci.XXXXXX")"
MUSA_CI_ISOLATED_USERBASE=""

if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    MUSA_CI_ISOLATED_USERBASE="1"
    PYTHONUSERBASE="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/sglang-musa-python.XXXXXX")"
    export PYTHONUSERBASE
    if [ -n "${GITHUB_ENV:-}" ]; then
        echo "PYTHONUSERBASE=${PYTHONUSERBASE}" >> "$GITHUB_ENV"
    fi
    if [ -n "${GITHUB_PATH:-}" ]; then
        echo "${PYTHONUSERBASE}/bin" >> "$GITHUB_PATH"
    fi
    echo "Using task-local Python user base: ${PYTHONUSERBASE}"
fi

# torchada has an unpinned Torch dependency. Installing it from the public
# index before the MUSA wheel bundle can pull CUDA Torch and CUDA Triton.
"${PIP_INSTALL[@]}" --upgrade pip "$MUSA_SETUPTOOLS_SPEC" --user

echo "Checking stale torchada extension locks..."
active_torchada_builds="$(
    pgrep -af '(^|[[:space:]/])(mcc|ninja)([[:space:]]|$)|torchada_cpp_ops' 2>/dev/null \
        | awk -v self="$$" '$1 != self'
)" || true
if [ -n "$active_torchada_builds" ]; then
    echo "$active_torchada_builds"
    echo "::error::Active torchada extension build detected; refusing to remove lock files"
    exit 1
fi
torch_extensions_dir="${HOME}/.cache/torch_extensions"
if [ -d "$torch_extensions_dir" ]; then
    find "$torch_extensions_dir" \
        -path '*/torchada_cpp_ops/lock' \
        -type f \
        -print \
        -delete
fi

WHL_DIR="${WHL_DIR:-/sglang-checkout/whl}"
MUSA_TRITON_WHEEL=""
MUSA_TRITON_TASK_LOCAL=""
MUSA_TORCHADA_WHEEL=""

wheel_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

find_bundled_musa_triton() {
    local candidate
    local -a candidates
    if [ ! -d "$WHL_DIR" ]; then
        return
    fi
    mapfile -t candidates < <(
        compgen -G "${WHL_DIR}/triton-${MUSA_TRITON_VERSION}-*.whl" || true
    )
    for candidate in "${candidates[@]}"; do
        if [ "$(wheel_sha256 "$candidate")" = "$MUSA_TRITON_SHA256" ]; then
            MUSA_TRITON_WHEEL="$candidate"
            return
        fi
    done
}

download_musa_triton() {
    local download_dir="${MUSA_CI_SCRATCH}/musa-triton"
    local -a candidates
    mkdir -p "$download_dir"
    python3 -m pip --isolated download \
        --dest "$download_dir" \
        --index-url "$MUSA_TRITON_INDEX_URL" \
        --no-deps \
        --only-binary=:all: \
        "triton==${MUSA_TRITON_VERSION}"
    mapfile -t candidates < <(find "$download_dir" -maxdepth 1 -type f -name '*.whl')
    if [ "${#candidates[@]}" -ne 1 ]; then
        echo "::error::Expected one MUSA Triton wheel, found ${#candidates[@]}"
        exit 1
    fi
    MUSA_TRITON_WHEEL="${candidates[0]}"
}

find_bundled_torchada() {
    local candidate
    local -a candidates
    if [ ! -d "$WHL_DIR" ]; then
        return
    fi
    mapfile -t candidates < <(
        compgen -G "${WHL_DIR}/torchada-${MUSA_TORCHADA_VERSION}-*.whl" || true
    )
    for candidate in "${candidates[@]}"; do
        if [ "$(wheel_sha256 "$candidate")" = "$MUSA_TORCHADA_SHA256" ]; then
            MUSA_TORCHADA_WHEEL="$candidate"
            return
        fi
    done
}

download_torchada() {
    local download_dir="${MUSA_CI_SCRATCH}/torchada"
    local -a candidates
    mkdir -p "$download_dir"
    python3 -m pip --isolated download \
        --dest "$download_dir" \
        --index-url "$MUSA_TORCHADA_INDEX_URL" \
        --no-deps \
        --only-binary=:all: \
        "torchada==${MUSA_TORCHADA_VERSION}"
    mapfile -t candidates < <(find "$download_dir" -maxdepth 1 -type f -name '*.whl')
    if [ "${#candidates[@]}" -ne 1 ]; then
        echo "::error::Expected one torchada wheel, found ${#candidates[@]}"
        exit 1
    fi
    MUSA_TORCHADA_WHEEL="${candidates[0]}"
}

find_bundled_musa_triton
if [ -z "$MUSA_TRITON_WHEEL" ]; then
    if python3 "$STACK_HELPER" verify \
        --expected-triton-version "$MUSA_TRITON_VERSION" \
        --triton-only; then
        echo "Reusing the installed MUSA Triton"
    else
        # Existing runner bundles do not contain Triton yet. Keep a hash-pinned
        # vendor-index fallback until the wheel is shipped in /sglang-checkout/whl.
        download_musa_triton
    fi
fi
if [ -n "$MUSA_TRITON_WHEEL" ]; then
    if [ "$(wheel_sha256 "$MUSA_TRITON_WHEEL")" != "$MUSA_TRITON_SHA256" ]; then
        echo "::error::MUSA Triton wheel SHA256 does not match the pinned artifact"
        exit 1
    fi
    MUSA_TRITON_TASK_LOCAL="1"
    echo "Using MUSA Triton wheel: ${MUSA_TRITON_WHEEL}"
fi

# torchada declares an unpinned dependency on Torch. Install the exact wheel
# with --no-deps so a fresh user site cannot resolve public CUDA Torch/Triton.
find_bundled_torchada
if [ -z "$MUSA_TORCHADA_WHEEL" ]; then
    download_torchada
fi
if [ "$(wheel_sha256 "$MUSA_TORCHADA_WHEEL")" != "$MUSA_TORCHADA_SHA256" ]; then
    echo "::error::torchada wheel SHA256 does not match the pinned artifact"
    exit 1
fi
echo "Using torchada wheel: ${MUSA_TORCHADA_WHEEL}"

VENDOR_WHEELS=("$MUSA_TORCHADA_WHEEL")
if [ -n "$MUSA_TRITON_WHEEL" ]; then
    VENDOR_WHEELS+=("$MUSA_TRITON_WHEEL")
fi
if [ -d "$WHL_DIR" ] && compgen -G "${WHL_DIR}"/*.whl > /dev/null; then
    for whl in "${WHL_DIR}"/*.whl; do
        case "$(basename "$whl")" in
          triton-*.whl|torchada-*.whl) continue;;
        esac
        VENDOR_WHEELS+=("$whl")
    done
fi

if [ -z "$MUSA_CI_ISOLATED_USERBASE" ]; then
    echo "Uninstall old packages based on wheel METADATA..."
    PKGS=$(
      for whl in "${VENDOR_WHEELS[@]}"; do
        meta_file=$(zipinfo -1 "$whl" | awk '/\.dist-info\/METADATA$/ {print; exit}')
        [ -n "$meta_file" ] || continue
        unzip -p "$whl" "$meta_file" 2>/dev/null | sed -n 's/^Name: //p' | head -n1
      done | sort -u
    )
    for pkg in $PKGS; do
      echo "Uninstalling $pkg"
      python3 -m pip uninstall -y "$pkg" || true
    done
fi

if [ "${#VENDOR_WHEELS[@]}" -gt 0 ]; then
    echo "Installing vendor wheels without dependency resolution..."
    if [ -n "$MUSA_CI_ISOLATED_USERBASE" ]; then
        "${PIP_INSTALL[@]}" --ignore-installed --no-deps --user "${VENDOR_WHEELS[@]}"
    else
        "${PIP_INSTALL[@]}" --no-deps --user "${VENDOR_WHEELS[@]}"
    fi
fi

MUSA_CONSTRAINTS="${MUSA_CI_SCRATCH}/constraints.txt"
python3 "$STACK_HELPER" constraints --output "$MUSA_CONSTRAINTS"
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "MUSA_CONSTRAINTS=${MUSA_CONSTRAINTS}" >> "$GITHUB_ENV"
fi
python3 "$STACK_HELPER" verify \
    --expected-triton-version "$MUSA_TRITON_VERSION" \
    ${MUSA_TRITON_TASK_LOCAL:+--require-user-site}

if [ -n "$SKIP_SGLANG_BUILD" ]; then
    echo "Didn't build checkout SGLang"
    exit 0
else
    if [ -z "$MUSA_CI_ISOLATED_USERBASE" ]; then
        python3 -m pip uninstall sgl-kernel -y || true
        python3 -m pip uninstall sglang -y || true
    fi
    # Clear Python cache to ensure latest code is used (works for any env: venv, system, conda)
    REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
    find "$REPO_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    find "$REPO_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    rm -f "${REPO_ROOT}/python/pyproject.toml" && mv "${REPO_ROOT}/python/pyproject_other.toml" "${REPO_ROOT}/python/pyproject.toml"

    # setuptools-rust builds the sglang-mm extension
    # (sglang.srt.rust_extensions._multimodal)
    # declared in pyproject_other.toml, so a Rust toolchain must be present like
    # on the CUDA/AMD CI paths. Idempotent; installs per-user under $HOME/.cargo.
    # Export PATH here because the pip install below runs in this same shell
    # (install_rustup.sh's own export/GITHUB_PATH only reach subsequent steps).
    bash "${REPO_ROOT}/scripts/ci/utils/install_rustup.sh"
    export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

    cd "${REPO_ROOT}" && "${PIP_INSTALL[@]}" \
        --constraint "$MUSA_CONSTRAINTS" \
        -v \
        -e "python[dev_musa]" \
        --user

    cd "${REPO_ROOT}/python/sglang/kernels/aot"
    rm -f pyproject.toml && mv pyproject_musa.toml pyproject.toml && MTGPU_TARGET=mp_31 python3 setup_musa.py install --user
    python3 "$STACK_HELPER" verify \
        --expected-triton-version "$MUSA_TRITON_VERSION" \
        --require-driver \
        --require-resolved-dependencies \
        ${MUSA_TRITON_TASK_LOCAL:+--require-user-site}
fi
