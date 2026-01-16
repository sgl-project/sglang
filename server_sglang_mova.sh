#!/usr/bin/env bash
set -euo pipefail

# 1) 环境
export PYTHONPATH=/inspire/ssd/project/video-generation/public/openveo3/sglang/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 2) 模型路径
MODEL_PATH="/inspire/ssd/project/video-generation/public/dhyu/Wan/mossVG/checkpoints/A14B-360p-wsd-105000"

# 3) 服务端口
HOST=0.0.0.0
PORT=30000

# 4) 启动 Server
python -m sglang.multimodal_gen.runtime.entrypoints.cli.main serve \
    --backend sglang \
    --pipeline-class-name MoVA \
    --model-path "${MODEL_PATH}" \
    --num-gpus 8 \
    --ulysses-degree 4 \
    --ring-degree 2 \
    --sp-degree 8 \
    --host ${HOST} \
    --port ${PORT}
