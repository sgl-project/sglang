model=/data/models/Deepseek-R1-FP8-Dynamic
TP=8
EP=1

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
    --mem-fraction-static 0.9 \
    --disable-radix-cache \
    --max-prefill-tokens 32768 \
    --cuda-graph-max-bs 128 \
    2>&1 | tee log.server.log &
