model=/models/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic/
TP=8
EP=1

# for profiling
# export SGLANG_TORCH_PROFILER_DIR=./profile_log
# export SGLANG_PROFILE_WITH_STACK=1
# export SGLANG_PROFILE_RECORD_SHAPES=1

echo "launching ${model}"
echo "TP=${TP}"
echo "EP=${EP}"

python3 -m sglang.launch_server \
    --model-path ${model} \
    --host localhost \
    --port 9000 \
    --tp-size ${TP} \
    --ep-size ${EP} \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --max-prefill-tokens 32768 \
    --cuda-graph-max-bs 128 \
    --max-running-requests 128 \
    --mm-attention-backend aiter_attn \
    2>&1 | tee log.server.log &
