## Benchmark for SGLang v0.4.1.post4 - DeepSeek v3 on Different H200 configurations 

We research the capabilites of two configurations of H200 NVIDIA GPUs:
- Single-node 8xH200 (BF16/FP8)

For the benchmarking, we choose as baseline parameters:

- `--random-range-ratio 1` 
- `--request-rate 1 `
- `--random-input 1024` 
- `--random-output 1024`

Complete results and logs for benchmarks are in https://github.com/datacrunch-research/h200-benchmarks

## DeepSeek V3 on 8xH200 (single-node) 

### BF16


### FP8


## Environment

To guarantee benchmarking results reproducibility we execute all the experiments with the latest available SGLang Docker image. Build benchmarking environment running the following commands:

```bash
$docker pull lmsysorg/sglang:dev

$docker run -it -d --shm-size 32g --gpus all --net host \
--env "HF_TOKEN=$HF_TOKEN" \
-v <models_dir>:/root/.cache/huggingface \
--ipc=host --name sglang_dev lmsysorg/sglang:latest bash

$docker exec -it /bin/bash sglang_dev
```

## Notes

Keep in mind the diferences in the commands for optimization techniques due to memory constrains.

## Online benchmarks

## DeepSeek V3 on 8xH200 (single-node) 

### BF16

```bash
# launch server
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --enable-torch-compile --enable-dp-attention --mem-fraction-static 0.8 --disable-cuda-graph


# bench serving
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 300 --request-rate 1 --random-input 1024 --random-output 1024  --output-file deepseek_v3_8xh200_BF16_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 600 --request-rate 2 --random-input 1024 --random-output 1024 --output-file deepseek_v3_8xh200_BF16_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 1200 --request-rate 4 --random-input 1024 --random-output 1024 --output-file deepseek_v3_8xh200_BF16_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 2400 --request-rate 8 --random-input 1024 --random-output 1024 --output-file deepseek_v3_8xh200_BF16_online_output.jsonl

```

### FP8

```bash
# launch server
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 
--quantization fp8 --kv-cache-dtype fp8_e5m2 --trust-remote-code --enable-dp-attention


# bench serving
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 300 --request-rate 1 --random-input 1024 --random-output 1024  --output-file deepseek_v3_8xh200_FP8_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 600 --request-rate 2 --random-input 1024 --random-output 1024 --output-file deepseek_v3_8xh200_FP8_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 1200 --request-rate 4 --random-input 1024 --random-output 1024 --output-file deepseek_v3_8xh200_FP8_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 2400 --request-rate 8 --random-input 1024 --random-output 1024 --output-file deepseek_v3_8xh200_FP8_online_output.jsonl
```
## Deepseek V3 on 2x8xH200 (multi-node)

### BF16

```bash
# launch server
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 ----dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 40000 --enable-torch-compile --mem-fraction-static 0.8 --disable-cuda-graph

python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 ----dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 1 --trust-remote-code --host 0.0.0.0 --port 40000 --enable-torch-compile --mem-fraction-static 0.8 --disable-cuda-graph


# bench serving
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 300 --request-rate 1 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_BF16_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 600 --request-rate 2 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_BF16_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 1200 --request-rate 4 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_BF16_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 2400 --request-rate 8 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_BF16_online_output.jsonl
```

### FP8

```bash
# launch server
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 ----dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 40000 --enable-torch-compile --quantization fp8 --kv-cache-dtype fp8_e5m2 --disable-cuda-graph

python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 ----dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 1 --trust-remote-code --host 0.0.0.0 --port 40000 --enable-torch-compile --quantization fp8 --kv-cache-dtype fp8_e5m2 --disable-cuda-graph


# bench serving
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 300 --request-rate 1 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_FP8_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 600 --request-rate 2 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_FP8_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 1200 --request-rate 4 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_FP8_online_output.jsonl

python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-range-ratio 1 --num-prompt 2400 --request-rate 8 --random-input 1024 --random-output 1024 --host 0.0.0.0 --port 40000 --output-file deepseek_v3_2x8xh200_FP8_online_output.jsonl
```

#### Note: Detach mode
```
nohup python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 
--quantization fp8 --kv-cache-dtype fp8_e5m2 --trust-remote-code --enable-dp-attention --host 0.0.0.0 --port 40000 &> singlenode_fp8.log &
```

```
nohup deepseek_v3.sh &> deepseek_v3_fp8_8xh200_log_output.txt
```