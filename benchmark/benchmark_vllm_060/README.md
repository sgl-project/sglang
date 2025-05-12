## How to reproduce the benchmark results for SGLang v0.3.0 compared to vLLM v0.6.0

In short, with multi step enabled, in online scenarios that we benchmarked, the Median TTFT of vLLM is **3 times** that of SGLang, and the Median ITL is **10 times** that of SGLang. Lower Median TTFT and ITL are better. vLLM's multi-step optimization did not improve throughput while ensuring lower Median TTFT and ITL. Also, under maximum throughput benchmark, if vLLM does not set gpu util to 0.95 separately and uses the default configuration instead, its maximum throughput is **lower** than that of SGLang.

## Online benchmark results

### Llama 3.1 8B Instruct 1 x A100 80G

| RPS  | Num prompts | Engine | Median E2E Latency | Median TTFT | Median TPOT | Median ITL |
|------|-------------|--------|--------------------|-------------|-------------|------------|
| 4    | 1200        | SGLang | 1564.17            | **31.98**   | 13.17       | **11.93**  |
| 4    | 1200        | vLLM   | 1691.97            | **100.48**  | 14.14       | **129.32** |
| 8    | 2400        | SGLang | 2175.02            | **35.68**   | 17.85       | **14.41**  |
| 8    | 2400        | vLLM   | 2137.16            | **120.39**  | 17.09       | **158.63** |

### Llama 3.1 70B Insruct 4 x H100 80G

| RPS  | Num Prompts | Engine | Median E2E Latency | Median TTFT | Median TPOT | Median ITL |
|------|-------------|--------|--------------------|-------------|-------------|------------|
| 4    | 1200        | SGLang | 3005.24            | **53.94**   | 25.03       | **21.67**  |
| 4    | 1200        | vLLM   | 2915.60            | **179.15**  | 23.58       | **231.23** |
| 8    | 2400        | SGLang | 4064.98            | **58.11**   | 33.07       | **24.45**  |
| 8    | 2400        | vLLM   | 3752.38            | **207.12**  | 29.15       | **275.32** |

## Offline benchmark results

### Llama 3.1 8B Instruct 1 x A100 80G

| RPS  | Num Prompts | Engine | Request throughput | Output token throughput |
|------|-------------|--------|--------------------|-------------------------|
| inf  | 5000        | SGLang | 22.03              | **4281.51**             |
| inf  | 5000        | vLLM   | 21.27              | **4132.37**             |

### Llama 3.1 70B Insruct 4 x H100 80G

| RPS  | Num Prompts | Engine | Request throughput | Output token throughput |
|------|-------------|--------|--------------------|-------------------------|
| inf  | 5000        | SGLang | 19.84              | **3856.01**             |
| inf  | 5000        | vLLM   | 19.04              | **3700.64**             |

## Installation

```bash
# install sglang v0.3.0
pip install --upgrade pip
pip install "sglang[all]"==0.3.0
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# install vllm v0.6.0
pip install vllm==0.6.0
```

## Notes

We referred to the reproduction method in https://github.com/vllm-project/vllm/issues/8176, and added the `--num-scheduler-steps 10` parameter when starting the vLLM server. The `gpu_memory_utilization` of vLLM is by default 0.9 at both TP 1 and TP 4, while SGLang's `mem_frac` is 0.88 at TP 1 and 0.85 at TP 4, so we manually set it to 0.88 at TP 4.

## Online benchmarks

```bash
# Llama 3.1 8B Instruct on 1 x A100
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile --disable-radix-cache
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --num-scheduler-steps 10 --max_model_len 4096

# Llama 3.1 70B Instruct on 4 x H100
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --disable-radix-cache --tp 4
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-70B-Instruct --disable-log-requests --num-scheduler-steps 10 --tensor 4 --max_model_len 4096

# bench serving
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 1200 --request-rate 4
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 2400 --request-rate 8
python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --num-prompts 1200 --request-rate 4
python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --num-prompts 2400 --request-rate 8
```

## Offline benchmarks

```bash
# Llama 3.1 8B Instruct on 1 x A100
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile --disable-radix-cache
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --num-scheduler-steps 10 --max_model_len 4096

# Llama 3.1 70B Instruct on 4 x H100
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --disable-radix-cache --tp 4 --mem-frac 0.88
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-70B-Instruct --disable-log-requests --num-scheduler-steps 10 --tensor 4 --max_model_len 4096

# bench serving
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 5000
python3 -m sglang.bench_serving --backend vllm --dataset-name sharegpt --num-prompts 5000
```
