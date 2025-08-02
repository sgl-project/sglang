#!/bin/bash
set -euo pipefail

# Install the required dependencies from cache
sed -Ei 's@(ports|archive).ubuntu.com@cache-service.nginx-pypi-cache.svc.cluster.local:8081@g' /etc/apt/sources.list
apt update -y
apt install -y build-essential cmake python3-pip python3-dev wget net-tools zlib1g-dev lld clang software-properties-common curl

# Setup pip cache
pip config set global.index-url http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple
pip config set global.trusted-host cache-service.nginx-pypi-cache.svc.cluster.local
python3 -m pip install --upgrade pip
pip uninstall sgl-kernel -y || true


### Download MemFabricV2
MF_WHL_NAME="mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/sglang/${MF_WHL_NAME}"
wget "${MEMFABRIC_URL}" && pip install "./${MF_WHL_NAME}"


### Install vLLM
VLLM_TAG=v0.8.5
git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG
(cd vllm && VLLM_TARGET_DEVICE="empty" pip install -v -e .)


### Install PyTorch and PTA
PYTORCH_VERSION=2.6.0
TORCHVISION_VERSION=0.21.0
PTA_VERSION=2.6.0rc1
pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==$PTA_VERSION


### Install Triton-Ascend
TRITON_ASCEND_VERSION=3.2.0rc2
pip install attrs==24.2.0 numpy==1.26.4 scipy==1.13.1 decorator==5.1.1 psutil==6.0.0 pytest==8.3.2 pytest-xdist==3.6.1 pyyaml pybind11
pip install triton-ascend==$TRITON_ASCEND_VERSION


pip install -e "python[srt_npu]"


### Modify PyTorch TODO: to be removed later
TORCH_LOCATION=$(python3 -c 'import torch; print(torch.__path__[0])')
sed -i 's/from triton.runtime.autotuner import OutOfResources/from triton.runtime.errors import OutOfResources/' "${TORCH_LOCATION}/_inductor/runtime/triton_heuristics.py"
