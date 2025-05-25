MODEL=meta-llama/Llama-3.1-8B-Instruct
DEVICES="mlx5_1,mlx5_2,mlx5_3,mlx5_4"
PAGE_SIZE=32
ATTENTION=fa3
TRANSFER=mooncake
HOST=0.0.0.0
PDLB=http://localhost:8000

python -m sglang.launch_server \
    --model-path $MODEL \
    --disaggregation-mode prefill \
    --host $HOST \
    --port 30000 \
    --page-size $PAGE_SIZE \
    --disaggregation-ib-device $DEVICES \
    --disable-radix-cache \
    --disaggregation-transfer-backend $TRANSFER \
    --attention-backend $ATTENTION \
    --pdlb-url $PDLB \
    --tp-size 2 &

python -m sglang.launch_server \
    --model-path $MODEL \
    --disaggregation-mode decode \
    --host $HOST \
    --port 40000 \
    --page-size $PAGE_SIZE \
    --base-gpu-id 2 \
    --disaggregation-ib-device $DEVICES \
    --disable-radix-cache \
    --disaggregation-transfer-backend $TRANSFER \
    --attention-backend $ATTENTION \
    --pdlb-url $PDLB \
    --tp-size 2
