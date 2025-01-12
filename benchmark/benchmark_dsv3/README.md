## Benchmark for SGLang v0.4.1 - DeepSeek v3 on Different H200 configurations 

We research the capabilites of two configurations of H200 NVIDIA GPUs:
- Single-node 8xH200 (BF16/FP8)
- Multi-node 2x8xH200 (BF16/FP8)
  - using Infiniband (400Gbps) with `nccl=2.21.5`

For the benchmarking, we choose as baseline parameters:

- `--random-range-ratio 1` 
- `--request-rate 1 `
- `--random-input 1024` 
- `--random-output 1024`

Complete results and logs for benchmarks are in [https://github.com/datacrunch-research/h200-benchmarks](https://github.com/datacrunch-research/h200-benchmarks/commit/700675be3e55a62925f9c1a80f0b68ecf724ec13)

## DeepSeek V3 on 8xH200 (single-node) 

### BF16

| RPS  | Num Prompts | Median E2E Latency (ms) | Median TTFT (ms) | Median TPOT (ms) | Median ITL (ms) | Output token throughput (tok/s) |
| ---- | ----------- | ----------------------- | ---------------- | ---------------- | --------------- | ------------------------------- |
| 1    | 300         | 214,924.09              | 587.15           | 209.48           | 159.64          | 639.99                          |
| 2    | 600         | 235,524.70              | 598.77           | 229.30           | 162.99          | 1313.74                         |
| 4    | 1200        | 324,438.44              | 766.70           | 316.35           | 237.99          | 2378.26                         |
| 8    | 2400        | 686,261.57              | 1191.74          | 516.67           | 255.96          | 2249.03                         |

### FP8

| RPS  | Num Prompts | Median E2E Latency (ms) | Median TTFT (ms) | Median TPOT (ms) | Median ITL (ms) | Output token throughput (tok/s) |
| ---- | ----------- | ----------------------- | ---------------- | ---------------- | --------------- | ------------------------------- |
| 1    | 300         | 147,735.43              | 563.41           | 143.71           | 101.78          | 773.15                          |
| 2    | 600         | 234,757.13              | 684.33           | 228.78           | 149.46          | 1401.77                         |
| 4    | 1200        | 376,040.67              | 865.26           | 366.48           | 287.95          | 2214.76                         |
| 8    | 2400        | 692,710.83              | 1358.77          | 675.95           | 515.18          | 2864.31                         |

## DeepSeek V3 on 2x8xH200 (multi-node) 

### BF16

| RPS  | Num Prompts | Median E2E Latency (ms) | Median TTFT (ms) | Median TPOT (ms) | Median ITL (ms) | Output token throughput (tok/s) |
| ---- | ----------- | ----------------------- | ---------------- | ---------------- | --------------- | ------------------------------- |
| 1    | 300         | 971,353.97              | 53,189.54        | 843.03           | 638.68          | 275.06                          |
| 2    | 600         | 2,010,951.23            | 313,373.93       | 1622.07          | 1192.37         | 256.50                          |
| 4    | 1200        | 3,881,082.65            | 774,460.73       | 1645.51          | 1178.42         | 255.45                          |
| 8    | 2400        | 6,819,185.61            | 4,072,706.72     | 2239.22          | 1205.60         | 250.08                          |

### FP8

| RPS  | Num Prompts | Median E2E Latency (ms) | Median TTFT (ms) | Median TPOT (ms) | Median ITL (ms) | Output token throughput (tok/s) |
| ---- | ----------- | ----------------------- | ---------------- | ---------------- | --------------- | ------------------------------- |
| 1    | 300         | 985,610.62              | 56,824.07        | 862.84           | 662.33          | 271.60                          |
| 2    | 600         | 1,975,371.99            | 305,318.37       | 1632.35          | 1219.14         | 288.41                          |
| 4    | 1200        | 3,901,390.30            | 767,082.14       | 3023.99          | 2189.83         | 269.19                          |
| 8    | 2400        | 7,374,173.14            | 1,680,440.41     | 2974.87          | 2007.02         | 276.74                          |

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
