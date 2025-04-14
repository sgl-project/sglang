#!/bin/bash

set -ex

# 压测工具
pip3 uninstall -y libra-openai-benchmark
pip3 install -i https://mirrors.tencent.com/pypi/simple/ --trusted-host mirrors.tencent.com libra-openai-benchmark

# 下载luban工具
pip3 install lubanml --index-url=https://mirrors.tencent.com/pypi/simple --extra-index-url=https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple


# output those env export to ~/.bashrc
echo "export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113" >> ~/.bashrc
echo "export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113" >> ~/.bashrc
echo "export http_proxy=http://hk-mmhttpproxy.woa.com:11113" >> ~/.bashrc
echo "export https_proxy=http://hk-mmhttpproxy.woa.com:11113" >> ~/.bashrc
echo "export NO_PROXY=127.0.0.1,0.0.0.0" >> ~/.bashrc


# create a alias proxy_on for those four commands in ~/.bashrc
echo "alias proxy_on='export HTTP_PROXY=http://hk-mmhttpproxy.woa.com:11113; export HTTPS_PROXY=http://hk-mmhttpproxy.woa.com:11113; export http_proxy=http://hk-mmhttpproxy.woa.com:11113; export https_proxy=http://hk-mmhttpproxy.woa.com:11113;'" >> ~/.bashrc
# alias proxy off
echo "alias proxy_off='unset HTTP_PROXY; unset HTTPS_PROXY; unset http_proxy; unset https_proxy'" >> ~/.bashrc
# echo "export PYTHONPATH=/sgl-workspace/:/sgl-workspace/sglang/:/sgl-workspace/sglang/sgl-kernel:$PYTHONPATH" >> ~/.bashrc


source ~/.bashrc


proxy_on

pip3 uninstall -y sgl_kernel && pip3 install sgl_kernel==0.0.5.post3
pip3 uninstall -y flashinfer-python && pip3 install flashinfer-python==0.2.3 -i https://flashinfer.ai/whl/cu124/torch2.5
pip3 install "nvidia-nccl-cu12==2.25.1" --no-deps


# python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# python3 -m pip install intel-extension-for-pytorch
# # python3 -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
# python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# python3 -m pip install intel-extension-for-pytorch==v2.5.0+cpu oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

apt update
apt install -y net-tools tmux iputils-ping htop nvtop

# apt install -y --no-install-recommends gnupg
# echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
# apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# apt update
# apt install nsight-systems-cli

# put those envs in ~/.bashrc

echo "export NCCL_IB_GID_INDEX=3" >> ~/.bashrc
echo "export NCCL_IB_SL=3" >> ~/.bashrc
echo "export NCCL_CHECK_DISABLE=1" >> ~/.bashrc
echo "export NCCL_P2P_DISABLE=0" >> ~/.bashrc
echo "export NCCL_IB_DISABLE=0" >> ~/.bashrc
echo "export NCCL_LL_THRESHOLD=16384" >> ~/.bashrc
echo "export NCCL_IB_CUDA_SUPPORT=1" >> ~/.bashrc
echo "export NCCL_SOCKET_IFNAME=bond1" >> ~/.bashrc
echo "export UCX_NET_DEVICES=bond1" >> ~/.bashrc
echo "export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6" >> ~/.bashrc
echo "export NCCL_COLLNET_ENABLE=0" >> ~/.bashrc
echo "export SHARP_COLL_ENABLE_SAT=0" >> ~/.bashrc
echo "export NCCL_NET_GDR_LEVEL=2" >> ~/.bashrc
echo "export NCCL_IB_QPS_PER_CONNECTION=4" >> ~/.bashrc
echo "export NCCL_IB_TC=160" >> ~/.bashrc
echo "export NCCL_PXN_DISABLE=0" >> ~/.bashrc

source ~/.bashrc


proxy_off
# 拉取模型
# 点击右上角头像获取token：https://lubanml.woa.com/#/
python3 -c "import os; os.environ['LubanUsername'] = 'yongtongwu'; os.environ['LubanUserToken'] = 'QTF0MFlxaHZNZlhOOVR2ZEhyeUxvNVR3dW9VZDMwSUhqbFZndXBWNVFqVT0='; os.environ['LubanCachePath'] = '/home/qspace/upload/luban_cache'; from lubanml.api.common import get_file_from_luban; ret = get_file_from_luban('luban:llm_deepseek_r1:model_path'); print(ret)"

# python3 -c "import os; os.environ['LubanUsername'] = 'yongtongwu'; os.environ['LubanUserToken'] = 'QTF0MFlxaHZNZlhOOVR2ZEhyeUxvNVR3dW9VZDMwSUhqbFZndXBWNVFqVT0='; os.environ['LubanCachePath'] = '/home/qspace/upload/luban_cache'; from lubanml.api.common import get_file_from_luban; ret = get_file_from_luban('luban:llm_deepseek_v3:model_path'); print(ret)"


# python3 -c "import os; os.environ['LubanUsername'] = 'yongtongwu'; os.environ['LubanUserToken'] = 'QTF0MFlxaHZNZlhOOVR2ZEhyeUxvNVR3dW9VZDMwSUhqbFZndXBWNVFqVT0='; os.environ['LubanCachePath'] = '/home/qspace/upload/luban_cache'; from lubanml.api.common import get_file_from_luban; ret = get_file_from_luban('luban:llm_deepseek_r1_distill_qwen_1_5b:model_path'); print(ret)"


proxy_on
# update r1 tokenizer.
# replace /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1/tokenizer_config.json
# with https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/tokenizer_config.json
rm -rf /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1/tokenizer_config.json
wget -O /home/qspace/upload/luban_cache/model/luban-llm_deepseek_r1-model_path/DeepSeek-R1/tokenizer_config.json https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/tokenizer_config.json


# python3 -c "import os; os.environ['LubanUsername'] = 'yongtongwu'; os.environ['LubanUserToken'] = 'QTF0MFlxaHZNZlhOOVR2ZEhyeUxvNVR3dW9VZDMwSUhqbFZndXBWNVFqVT0='; os.environ['LubanCachePath'] = '/home/qspace/upload/luban_cache'; from lubanml.api.common import get_file_from_luban; ret = get_file_from_luban('luban:llm_deepseek_r1_distill_qwen_1_5b:model_path'); print(ret)"