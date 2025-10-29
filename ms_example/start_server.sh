python3 -m sglang.launch_server \
    --model-path /home/ckpt/Qwen3-8B \
    --host 0.0.0.0 \
    --port 37654 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --tp-size 1 \
    --dp-size 1 \
    --mem-fraction-static 0.8
    #--dist-init-addr 127.0.0.1:29500 --nnodes 2 --node-rank 0 \
