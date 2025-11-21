#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"
DEVICE_TYPE=$1


# Install the required dependencies in CI.
apt update -y && apt install -y \
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
    ca-certificates
update-ca-certificates
${PIP_INSTALL} --upgrade pip
# Pin wheel to 0.45.1, REF: https://github.com/pypa/wheel/issues/662
${PIP_INSTALL} wheel==0.45.1


### Install MemFabric
${PIP_INSTALL} mf-adapter==1.0.0


### Install PyTorch and PTA
PYTORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
${PIP_INSTALL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu

PTA_VERSION="v7.2.0-pytorch${PYTORCH_VERSION}"
PTA_NAME="torch_npu-${PYTORCH_VERSION}-cp311-cp311-manylinux_2_28_aarch64.whl"
PTA_URL="https://gitcode.com/Ascend/pytorch/releases/download/${PTA_VERSION}/${PTA_NAME}"
wget -O "${PTA_NAME}" "${PTA_URL}" && ${PIP_INSTALL} "./${PTA_NAME}"


### Install Triton-Ascend
TRITON_ASCEND_NAME="triton_ascend-3.2.0+gitb0ea0850-cp311-cp311-linux_aarch64.whl"
TRITON_ASCEND_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/triton_ascend-3.2.0%2Bgitb0ea0850-cp311-cp311-linux_aarch64.whl"
${PIP_INSTALL} attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11
wget -O "${TRITON_ASCEND_NAME}" "${TRITON_ASCEND_URL}" && ${PIP_INSTALL} "./${TRITON_ASCEND_NAME}"


### Install BiSheng
BISHENG_NAME="Ascend-BiSheng-toolkit_aarch64.run"
BISHENG_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/sglang/${BISHENG_NAME}"
wget -O "${BISHENG_NAME}" "${BISHENG_URL}" && chmod a+x "${BISHENG_NAME}" && "./${BISHENG_NAME}" --install && rm "${BISHENG_NAME}"


### Install sgl-kernel-npu
SGL_KERNEL_NPU_TAG="20251120"
git clone --depth 1 https://github.com/sgl-project/sgl-kernel-npu.git --branch ${SGL_KERNEL_NPU_TAG}
(cd sgl-kernel-npu && bash ./build.sh && ${PIP_INSTALL} output/deep_ep*.whl output/sgl_kernel_npu*.whl && cd "$(python3 -m pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so)


### Install CustomOps (TODO: to be removed once merged into sgl-kernel-npu)
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run
chmod a+x ./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run
./CANN-custom_ops-8.2.0.0-$DEVICE_TYPE-linux.aarch64.run --quiet --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
wget https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com/ops/custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl
pip install ./custom_ops-1.0.$DEVICE_TYPE-cp311-cp311-linux_aarch64.whl

### Install SGLang
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
${PIP_INSTALL} -v -e "python[srt_npu]"
