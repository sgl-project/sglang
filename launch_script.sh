#!/bin/bash

export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_SPEC_V2=1
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
SPEC_MODEL=lmsys/sglang-EAGLE-LLaMA3-Instruct-8B
PORT=23333
command="""
python -m sglang.launch_server \
    --dtype float16 \
    --model-path $MODEL \
    --attention-backend triton \
    --decode-log-interval 1 \
    --cuda-graph-bs $(seq -s ' ' 1 64) \
    --enable-beta-spec \
    --speculative-algorithm EAGLE \
    --speculative-draft-model $SPEC_MODEL \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --host 127.0.0.1 \
    --port $PORT \
    --load-format dummy \
    --max-running-requests 1
"""
echo $command
$command
