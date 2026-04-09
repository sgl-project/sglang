#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
UV_PIP_INSTALL="uv pip install "
DEVICE_TYPE=$1
OPTIONAL_DEPS="${2:-}"


# Install the required dependencies in CI.
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

# Pin wheel to 0.45.1, REF: https://github.com/pypa/wheel/issues/662
${UV_PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil


### Install MemFabric
${UV_PIP_INSTALL} memfabric-hybrid==1.0.5


### Install PyTorch and PTA
if [ -n "$OPTIONAL_DEPS" ]; then
    PYTORCH_VERSION="2.10.0"
    TORCHVISION_VERSION="0.25.0"
    ${UV_PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_CACHE_URL:="https://download.pytorch.org/whl/cpu"} --extra-index-url ${PYPI_CACHE_URL:="https://pypi.org/simple/"}
    PTA_URL="https://gitcode.com/Ascend/pytorch/releases/download/7.3.0.alpha002/torch_npu-2.10.0rc2-cp311-cp311-manylinux_2_28_aarch64.whl"
    # GitCode does not allow UV downloads.
    ${PIP_INSTALL} ${PTA_URL}
else
    PYTORCH_VERSION="2.8.0"
    TORCHVISION_VERSION="0.23.0"
    ${UV_PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_CACHE_URL:="https://download.pytorch.org/whl/cpu"} --extra-index-url ${PYPI_CACHE_URL:="https://pypi.org/simple/"}
    PTA_URL="https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl"
    ${PIP_INSTALL} ${PTA_URL}
fi


### Install Triton-Ascend
${UV_PIP_INSTALL} triton-ascend


### Install sgl-kernel-npu
SGLANG_KERNEL_NPU_TAG="2026.03.10.rc1"
mkdir sgl-kernel-npu
(cd sgl-kernel-npu && wget "${GITHUB_PROXY_URL:=""}https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann8.5.0-${DEVICE_TYPE}-$(arch).zip" \
&& unzip ./sgl-kernel-npu-${SGLANG_KERNEL_NPU_TAG}-torch2.8.0-py311-cann8.5.0-${DEVICE_TYPE}-$(arch).zip \
&& ${UV_PIP_INSTALL} ./deep_ep*.whl ./sgl_kernel_npu*.whl \
&& (cd "$(python3 -m pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so))


### Install SGLang
rm -rf python/pyproject.toml && mv python/pyproject_npu.toml python/pyproject.toml
${UV_PIP_INSTALL} -v -e "python[dev_npu]"
