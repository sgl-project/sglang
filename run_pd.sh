MODEL=meta-llama/Llama-3.1-8B-Instruct
DEVICES="mlx5_101,mlx5_102,mlx5_103"
PAGE_SIZE=64
ATTENTION=fa3
TRANSFER=mooncake

# python3 -m sglang.srt.disaggregation.mini_lb --prefill http://localhost:30000 --decode http://localhost:40000 --host 0.0.0.0 --port 8000 &

python3 -m sgl_pdlb.launch_lb --prefill http://localhost:30000 --decode http://localhost:40000 --host 0.0.0.0 --port 8000 &

sleep 3

python -m sglang.launch_server \
    --model-path $MODEL \
    --disaggregation-mode prefill \
    --port 30000 \
    --page-size $PAGE_SIZE \
    --host localhost \
    --disaggregation-ib-device $DEVICES \
    --disable-radix-cache \
    --disaggregation-transfer-backend $TRANSFER \
    --attention-backend $ATTENTION \
    --tp-size 2 &

sleep 10

python -m sglang.launch_server \
    --model-path $MODEL \
    --disaggregation-mode decode \
    --port 40000 \
    --page-size $PAGE_SIZE \
    --base-gpu-id 2 \
    --disaggregation-ib-device $DEVICES \
    --disable-radix-cache \
    --disaggregation-transfer-backend $TRANSFER \
    --attention-backend $ATTENTION \
    --tp-size 2

sleep 10
