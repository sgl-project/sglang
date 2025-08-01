#!/bin/bash
set -euo pipefail

CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
PIP_INSTALL="pip install --no-cache-dir"


# Update apt & pip sources
sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
pip config set global.index-url http://${CACHING_URL}/pypi/simple
pip config set global.trusted-host ${CACHING_URL}


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
python3 -m ${PIP_INSTALL} --upgrade pip


### Download MemFabricV2
MF_WHL_NAME="mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/sglang/${MF_WHL_NAME}"
wget "${MEMFABRIC_URL}" && ${PIP_INSTALL} "./${MF_WHL_NAME}"


### Install vLLM
VLLM_TAG=v0.8.5
git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG
(cd vllm && VLLM_TARGET_DEVICE="empty" ${PIP_INSTALL} -v -e .)


### Install PyTorch and PTA
PYTORCH_VERSION=2.6.0
TORCHVISION_VERSION=0.21.0
PTA_VERSION=2.6.0
${PIP_INSTALL} torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu
${PIP_INSTALL} torch_npu==$PTA_VERSION


### Install Triton-Ascend
TRITON_ASCEND_VERSION=3.2.0rc2
${PIP_INSTALL} attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11
${PIP_INSTALL} triton-ascend==$TRITON_ASCEND_VERSION


### Install SGLang
${PIP_INSTALL} -v -e "python[srt_npu]"


### Modify PyTorch TODO: to be removed later
TORCH_LOCATION=$(pip show torch | grep Location | awk -F' ' '{print $2}')
sed -i 's/from triton.runtime.autotuner import OutOfResources/from triton.runtime.errors import OutOfResources/' "${TORCH_LOCATION}/torch/_inductor/runtime/triton_heuristics.py"
