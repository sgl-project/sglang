#!/bin/bash
set -euxo pipefail

source scripts/ci/cuda/ci_install_dependency.sh

if [ -z "${PIP_CMD:-}" ]; then
    echo "FATAL:PIP_CMD is unset after sourcing ci_install_dependency.sh"
    exit 1
fi

export CUDA_HOME=/usr/local/cuda

if [ "${FORCE_REBUILD_FLASH_MLA:-0}" = "1" ]; then
    echo "FORCE_REBUILD_FLASH_MLA=1; uninstalling any cached flash_mla before rebuild."
    ${PIP_UNINSTALL_CMD:-pip uninstall -y} flash_mla ${PIP_UNINSTALL_SUFFIX:-} || true
elif python3 -c "import flash_mla" >/dev/null 2>&1; then
    echo "flash_mla is already installed or importable. Skipping installation."
    exit 0
fi

# CUDA 13.0 puts CCCL headers under /usr/local/cuda/include/cccl/cuda but
# FlashMLA's build expects them at /usr/local/cuda/include/cuda. Symlink so
# the compiler finds them. Idempotent: skip if the link/dir already exists.
if [ ! -e /usr/local/cuda/include/cuda ] && [ -d /usr/local/cuda/include/cccl/cuda ]; then
    ln -s /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda
fi

# Install FlashMLA
FLASH_MLA_DIR=/root/.cache/flash-mla
rm -rf ${FLASH_MLA_DIR}
git clone https://github.com/deepseek-ai/FlashMLA.git ${FLASH_MLA_DIR}
pushd ${FLASH_MLA_DIR}
git submodule update --init --recursive
${PIP_CMD:-pip} install --no-build-isolation -v . ${PIP_INSTALL_SUFFIX:-}
popd
