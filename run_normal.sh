MODEL=/models/hub/Meta-Llama-3.1-8B-Instruct/
PAGE_SIZE=1
ATTENTION=fa3

python -m sglang.launch_server \
    --model-path $MODEL \
    --port 33000 \
    --page-size $PAGE_SIZE \
    --host localhost \
    --attention-backend $ATTENTION \
    --disable-radix-cache \
    --max-total-tokens 80000 \
    --base-gpu-id 4 \
    --tp-size 2
