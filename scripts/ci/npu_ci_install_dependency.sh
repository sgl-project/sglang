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
    unzip
update-ca-certificates
${PIP_INSTALL} --upgrade pip
# Pin wheel to 0.45.1, REF: https://github.com/pypa/wheel/issues/662
${PIP_INSTALL} wheel==0.45.1 pybind11


### Install MemFabric
${PIP_INSTALL} memfabric-hybrid==1.0.0


### Install PyTorch and PTA
PYTORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu

PTA_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/torch_npu/torch_npu-2.8.0.post2.dev20251224-cp311-cp311-manylinux_2_28_aarch64.whl"
${PIP_INSTALL} ${PTA_URL}


### Install Triton-Ascend
TRITON_ASCEND_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/triton_ascend/triton_ascend-3.2.0.dev2025112116-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
${PIP_INSTALL} ${TRITON_ASCEND_URL}


### Install BiSheng
BISHENG_NAME="Ascend-BiSheng-toolkit_aarch64_20251121.run"
BISHENG_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/triton_ascend/${BISHENG_NAME}"
wget -O "${BISHENG_NAME}" "${BISHENG_URL}" && chmod a+x "${BISHENG_NAME}" && "./${BISHENG_NAME}" --install && rm "${BISHENG_NAME}"


### Install sgl-kernel-npu
SGLANG_KERNEL_NPU_TAG="2026.01.07"
mkdir sgl-kernel-npu
(cd sgl-kernel-npu && wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/${SGL_KERNEL_NPU_TAG}/sgl-kernel-npu_${SGL_KERNEL_NPU_TAG}_8.3.rc2_910b.zip \
&& unzip sgl-kernel-npu_${SGL_KERNEL_NPU_TAG}_8.3.rc2_910b.zip \
&& ${PIP_INSTALL} output/deep_ep*.whl output/sgl_kernel_npu*.whl \
&& (cd "$(python3 -m pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -sf deep_ep/deep_ep_cpp*.so))


### Install CustomOps (TODO: to be removed once merged into sgl-kernel-npu)
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/CANN-custom_ops-8.3.0.1-$DEVICE_TYPE-linux.aarch64.run
chmod a+x ./CANN-custom_ops-8.3.0.1-$DEVICE_TYPE-linux.aarch64.run
./CANN-custom_ops-8.3.0.1-$DEVICE_TYPE-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/custom_ops-2.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl
pip install ./custom_ops-2.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl

### Install SGLang
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
${PIP_INSTALL} -v -e "python[srt_npu]"

### Other dependencies
${PIP_INSTALL} tabulate

