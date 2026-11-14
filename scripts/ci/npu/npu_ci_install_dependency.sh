#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
UV_PIP_INSTALL="uv pip install "
DEVICE_TYPE=$1

CANN_VERSION="${CANN_VERSION:-9.1.0}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.10.0}"
SGLANG_KERNEL_NPU_TAG="${SGLANG_KERNEL_NPU_TAG:-2026.11.0.post5}"
PYARROW_VERSION="${PYARROW_VERSION:-23.0.1}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-${CANN_VERSION}}"
export ASCEND_HOME_PATH
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/lib:${ASCEND_HOME_PATH}/$(arch)-linux/devlib/device:/usr/local/Ascend/driver/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"

GITHUB_PROXY_URL="${GITHUB_PROXY_URL:-}"

apt update -y && apt install -y \
    unzip \
    build-essential \
    cmake \
    wget \
    curl \
    net-tools \
    zlib1g-dev \
    lld \
    clang \
    locales \
    ccache \
    ffmpeg \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    ca-certificates \
    libgl1 \
    libglib2.0-0
update-ca-certificates
${PIP_INSTALL} --upgrade pip
${PIP_INSTALL} uv
export UV_NO_CACHE=true
export UV_SYSTEM_PYTHON=true
export UV_INDEX_STRATEGY=unsafe-best-match

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../utils/install_rustup.sh"
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:${PATH}"

${UV_PIP_INSTALL} pybind11 pyyaml decorator scipy attrs psutil

${PIP_INSTALL} memfabric-hybrid==1.3.0
mfcli kernel install
${PIP_INSTALL} memcache-hybrid==1.3.0

### Install memfabric-zbal
if [ "${DEVICE_TYPE}" = "950" ]; then
    ${PIP_INSTALL} memfabric-zbal==1.2.21004.post1
else
    ${PIP_INSTALL} memfabric-zbal==1.1.3
fi

### Install SGLang Model Gateway
${PIP_INSTALL} sglang-router

${UV_PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==0.25.0 torchaudio==2.10.0 --index-url ${TORCH_CACHE_URL:="https://download.pytorch.org/whl/cpu"} --extra-index-url ${PYPI_CACHE_URL:="https://pypi.org/simple/"}
${PIP_INSTALL} torch-npu==2.10.0.post6 --extra-index-url https://ascend.devcloud.huaweicloud.com/pypi/simple/

case "$(arch)" in
aarch64)
    ${PIP_INSTALL} "${GITHUB_PROXY_URL}https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
    ;;
x86_64)
    ${PIP_INSTALL} "${GITHUB_PROXY_URL}https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    ;;
*)
    echo "Unsupported architecture: $(arch)" >&2
    exit 1
    ;;
esac

mkdir -p sgl-kernel-npu
(
    cd sgl-kernel-npu
    wget "${GITHUB_PROXY_URL}https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch${PYTORCH_VERSION}-py312-cann${CANN_VERSION}-${DEVICE_TYPE}-$(arch).zip"
    unzip ./*.zip

    ${UV_PIP_INSTALL} --no-deps ./sgl_kernel_npu*.whl
    ${UV_PIP_INSTALL} \
        ./deep_ep*.whl \
        ./attentions*.whl \
        ./torch_memory_saver*.whl

    cd "$(python3 -m pip show deep-ep | grep -E '^Location:' | awk '{print $2}')"
    ln -sf deep_ep/deep_ep_cpp*.so
)
rm -rf sgl-kernel-npu

rm -f python/pyproject.toml
mv python/pyproject_npu.toml python/pyproject.toml
${UV_PIP_INSTALL} -v -e "python[dev_npu]" "pyarrow==${PYARROW_VERSION}"
