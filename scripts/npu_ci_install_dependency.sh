#!/bin/bash
# Install the required dependencies in CI.
sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
apt update -y
apt install -y build-essential cmake python3-pip python3-dev wget net-tools zlib1g-dev lld clang software-properties-common


pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple
python3 -m pip install --upgrade pip
pip uninstall sgl-kernel -y || true


### Download MemFabricV2
MEMFABRIC_URL="https://sglang-ascend.obs.cn-east-3.myhuaweicloud.com:443/sglang/mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl"
wget "${MEMFABRIC_URL}" && pip install ./mf_adapter-1.0.0-cp311-cp311-linux_aarch64.whl


### Install vLLM
VLLM_TAG=v0.8.5
git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_TAG && \
    cd vllm && VLLM_TARGET_DEVICE="empty" pip install -v -e . && cd ..


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
TORCH_LOCATION=$(pip show torch | grep Location | awk -F' ' '{print $2}')
sed -i 's/from triton.runtime.autotuner import OutOfResources/from triton.runtime.errors import OutOfResources/' "${TORCH_LOCATION}/torch/_inductor/runtime/triton_heuristics.py"



# official PPA comes with ffmpeg 2.8, which lacks tons of features, we use ffmpeg 4.0 here
# add-apt-repository -y ppa:jonathonf/ffmpeg-4 # for ubuntu20.04 official PPA is already version 4.2, you may skip this step
# apt-get update -y
# apt-get install -y build-essential python3-dev python3-setuptools make cmake
# apt-get install -y ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
# # build decord
# git clone --recursive https://github.com/dmlc/decord
# cd decord
# mkdir build && cd build
# cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
# make
# cd ../python
# sed -i 's/maintainer_email.*/&\n    dependency_links=[\n        "https:\/\/mirrors.tuna.tsinghua.edu.cn\/pypi\/web\/simple"\n    ],/g' setup.py
# python3 setup.py install --user
# cd ../..
# pwd
# note: make sure you have cmake 3.8 or later, you can install from cmake official website if it's too old



# pip install -e "python[all_npu]"

# pip install modelscope
# pip install pytest
