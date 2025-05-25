MODEL=meta-llama/Llama-3.1-8B-Instruct
BACKEND=${BACKEND:-sglang}
INPUT=${INPUT:-8000}
OUTPUT=${OUTPUT:-500}
PORT=${PORT:-8000}

python3 -m sglang.bench_serving \
    --backend $BACKEND \
    --dataset-name random \
    --num-prompts 500 \
    --random-input $INPUT \
    --random-output $OUTPUT \
    --random-range-ratio 1 \
    --port $PORT \
    --dataset-name "random" \
    --model $MODEL \
    --pd-separated \
    --request-rate 5
