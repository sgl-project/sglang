export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/home/wzy/qwen3-30b-a3b

#MODEL_PATH=/home/weights/Qwen3-30B-A3B

export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600


export PYTHONPATH=/home/wzy/sgl-sglang/python:$PYTHONPATH
#export HCCL_BUFFSIZE=400
#export HCCL_SOCKET_IFNAME=lo
#export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"

python -m sglang.launch_server --model-path $MODEL_PATH \
    --host 127.0.0.1 --port 6990 --trust-remote-code --nnodes 1 --node-rank 0  \
    --attention-backend ascend --device npu \
    --max-running-requests 8 \
    --disable-radix-cache \
    --chunked-prefill-size -1 --max-prefill-tokens 35000  \
    --tp-size 8 --mem-fraction-static 0.5 --cuda-graph-bs 1 2 3 4 5 6 7 8 --dtype bfloat16 \
    --base-gpu-id 0 --disable-cuda-graph 
