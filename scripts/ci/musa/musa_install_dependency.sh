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
    cd "${REPO_ROOT}" && ${PIP_INSTALL} -v -e "python[dev_musa]" --user

    cd "${REPO_ROOT}/sgl-kernel"
    rm -f pyproject.toml && mv pyproject_musa.toml pyproject.toml && MTGPU_TARGET=mp_31 python3 setup_musa.py install --user
    echo "$HOME/.local/bin" >> "$GITHUB_PATH"
fi
