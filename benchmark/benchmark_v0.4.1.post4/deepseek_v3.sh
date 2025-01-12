# Docker single-node command: (FP8 version) * PROVISIONAL *
: '
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v /mnt/co-research/shared-models:/root/.cache/huggingface \
    --name sglang_singlenodeFP8 \
    -it \
    -rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --quantization fp8 --kv-cache-dtype fp8_e5m2 --trust-remote-code --host 0.0.0.0 --port 40000 --enable-dp-attention
'

# Docker multi-node command: (BF16 version) * PROVISIONAL *
# Node0: * PROVISIONAL *
: '
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v /mnt/co-research/shared-models:/root/.cache/huggingface \
    --name sglang_multinode0 \
    -it \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 ----dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 40000

'

# Node1: * PROVISIONAL *
: '
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v /mnt/co-research/shared-models:/root/.cache/huggingface \
    --name sglang_multinode1 \
    -it \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 ----dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 1 --trust-remote-code --host 0.0.0.0 --port 40000

'

# Docker basic client command: * PROVISIONAL *
: '
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v /mnt/co-research/shared-models:/root/.cache/huggingface \
    --name sglang_bnchmrk_client \
    -it \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 512 --random-range-ratio 1 --num-prompts 1 --host 0.0.0.0 --port 40000
'

# 8xH200/2x8xH200 FP8/BF16
# Online
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 300 --request-rate 1 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_FP8_online_output.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 600 --request-rate 2 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_FP8_online_output.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 1200 --request-rate 4 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_FP8_online_output.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 2400 --request-rate 8 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_FP8_online_output.jsonl
