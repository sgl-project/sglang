#!/bin/bash
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

PIP_INSTALL="python3 -m pip install --no-cache-dir"
${PIP_INSTALL} --upgrade pip setuptools torchada --user

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

WHL_DIR="/sglang-checkout/whl"
if [ -d "$WHL_DIR" ] && compgen -G "${WHL_DIR}"/*.whl > /dev/null; then
    echo "Uninstall old packages based on wheel METADATA..."
    PKGS=$(
      for whl in "${WHL_DIR}"/*.whl; do
        meta_file=$(zipinfo -1 "$whl" | awk '/\.dist-info\/METADATA$/ {print; exit}')
        [ -n "$meta_file" ] || continue
        unzip -p "$whl" "$meta_file" 2>/dev/null | sed -n 's/^Name: //p' | head -n1
      done | sort -u
    )
    for pkg in $PKGS; do
      echo "Uninstalling $pkg"
      pip uninstall -y "$pkg" || true
    done
    echo "Installing wheel files without dependency resolution..."
    ${PIP_INSTALL} "${WHL_DIR}"/*.whl --user
fi

if [ -n "$SKIP_SGLANG_BUILD" ]; then
    echo "Didn't build checkout SGLang"
    exit 0
else
    pip uninstall sgl-kernel -y || true
    pip uninstall sglang -y || true
    # Clear Python cache to ensure latest code is used (works for any env: venv, system, conda)
    REPO_ROOT="${GITHUB_WORKSPACE:-$(pwd)}"
    find "$REPO_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    find "$REPO_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    rm -f "${REPO_ROOT}/python/pyproject.toml" && mv "${REPO_ROOT}/python/pyproject_other.toml" "${REPO_ROOT}/python/pyproject.toml"

    # setuptools-rust builds the sglang-mm extension (sglang.srt.multimodal._core)
    # declared in pyproject_other.toml, so a Rust toolchain must be present like
    # on the CUDA/AMD CI paths. Idempotent; installs per-user under $HOME/.cargo.
    # Export PATH here because the pip install below runs in this same shell
    # (install_rustup.sh's own export/GITHUB_PATH only reach subsequent steps).
    bash "${REPO_ROOT}/scripts/ci/utils/install_rustup.sh"
    export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

    cd "${REPO_ROOT}" && ${PIP_INSTALL} -v -e "python[dev_musa]" --user

    cd "${REPO_ROOT}/sgl-kernel"
    rm -f pyproject.toml && mv pyproject_musa.toml pyproject.toml && MTGPU_TARGET=mp_31 python3 setup_musa.py install --user
    echo "$HOME/.local/bin" >> "$GITHUB_PATH"
fi
