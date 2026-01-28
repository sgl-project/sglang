#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
DEVICE_TYPE=$1


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
    ca-certificates \
    libgl1 \
    libglib2.0-0
update-ca-certificates
${PIP_INSTALL} --upgrade pip
# Pin wheel to 0.45.1, REF: https://github.com/pypa/wheel/issues/662
${PIP_INSTALL} wheel==0.45.1 pybind11 pyyaml decorator scipy attrs psutil


### Install MemFabric
${PIP_INSTALL} memfabric-hybrid==1.0.5


### Install PyTorch and PTA
PYTORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu
PTA_URL="https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.8.0/torch_npu-2.8.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl"
${PIP_INSTALL} ${PTA_URL}


### Install Triton-Ascend
${PIP_INSTALL} triton-ascend


### Install sgl-kernel-npu
SGLANG_KERNEL_NPU_TAG="2026.01.21"
mkdir sgl-kernel-npu
(cd sgl-kernel-npu && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGLANG_KERNEL_NPU_TAG}/sgl-kernel-npu_${SGLANG_KERNEL_NPU_TAG}_8.5.0_${DEVICE_TYPE}.zip \
&& unzip sgl-kernel-npu_${SGLANG_KERNEL_NPU_TAG}_8.5.0_${DEVICE_TYPE}.zip \
&& ${PIP_INSTALL} output/deep_ep*.whl output/sgl_kernel_npu*.whl \
&& (cd "$(python3 -m pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so))


### Install SGLang
rm -rf python/pyproject.toml && mv python/pyproject_npu.toml python/pyproject.toml
${PIP_INSTALL} -v -e "python[dev_npu]"
