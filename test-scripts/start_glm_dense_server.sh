#!/bin/bash

# 1号机 http://10.52.101.26:30000 - V2 baseline
# 2号机 http://10.52.100.149:30000 - V3 baseline
# 3号机 http://10.52.101.14:30000  -  dense baseline tp8 tool parser
# 4号机 http://10.52.100.141:30000 -  dense baseline
# 5号机 http://10.52.98.151:30000 - V3 swe tool parser
# 6号机 http://10.52.106.215:30000 - dense swe

# pkill -9 -f "sglang.launch_server"

# tail -f /tmp/glm_clean_server.log

source .venv/bin/activate
export PYTHONPATH=/root/paddlejob/inference-public/denghaodong/code/sglang/python
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export FLASHINFER_USE_CUDA_NORM=1

# nohup \
# python3 -m sglang.launch_server \
#     --model-path /root/paddlejob/inference-public/denghaodong/code/model/GLM_v2 \
#     --host 0.0.0.0 \
#     --port 30000 \
#     --served-model-name EB \
#     --tensor-parallel-size 8 \
#     --context-length 131072 \
#     --max-running-requests 64 \
#     --quantization fp8 \
#     --reasoning-parser glm45 \
#     --tool-call-parser qwen \
#     > /tmp/glm_dense_tool_call_parser_server3.log 2>&1 &

nohup \
python3 -m sglang.launch_server \
    --model-path /root/paddlejob/inference-public/denghaodong/code/model/GLM_v2 \
    --host 0.0.0.0 \
    --port 30000 \
    --tensor-parallel-size 8 \
    --context-length 131072 \
    --max-running-requests 64 \
    --quantization fp8 \
    --reasoning-parser glm45 \
    --tool-call-parser qwen \
    > /tmp/glm_server_v3_qwen_tool.log 2>&1 &