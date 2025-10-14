model=/data/pretrained-models/DeepSeek-R1-MXFP4-Preview
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
    --chunked-prefill-size 196608 \
    --mem-fraction-static 0.9 \
    --disable-radix-cache \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    --cuda-graph-max-bs 128 \
    2>&1 | tee log.server.log &
