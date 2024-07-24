# How to reproduce the benchmark results of SGLang

## Prerequisite

### Install the latest SGLang

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip setuptools wheel
pip install -e "python[all]"

pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/
```

### Set up ulimit and HF_TOKEN

```bash
ulimit -n 65535
# Change the token to a real and usable one, with access permissions for the Llama 3 models.
export HF_TOKEN=hf_token
```

### Launch the server

```bash
# Meta-Llama-3.1-8B-Instruct
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --enable-torch-compile --disable-radix-cache

# Meta-Llama-3.1-70B-Instruct
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-70B-Instruct --disable-radix-cache --tp 8

# Meta-Llama-3-70B-Instruct-FP8
python -m sglang.launch_server --model-path neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-radix-cache --tp 8
```

## Benchmark

### Hardware Requirements

- 8B models: Single NVIDIA A100 80GB GPU
- 70B models: 8 x NVIDIA A100 80GB GPUs with Tensor Parallelism (TP) 8
- 70B FP8 models: 8 x NVIDIA H100 GPUs with Tensor Parallelism (TP) 8

Please ensure you have the appropriate hardware before running the benchmarks.

#### Offline benchmark

```bash
# Random dataset, Input [512, 1024], Output [512, 1024], num prompts 3k
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5 --output-file sglang_offline_benchmark.jsonl
		
# Random dataset, Input [2048, 4096], Output [512, 1024], num prompts 3k      							
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 4096 --random-output 1024 --random-range-ratio 0.5 --output-file sglang_offline_benchmark.jsonl

# Random dataset, Input [512, 1024], Output [256, 512], num prompts 3k							
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 512 --random-range-ratio 0.5 --output-file sglang_offline_benchmark.jsonl

# Random dataset, Input [2048, 4096], Output [256, 512], num prompts 3k
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 3000 --random-input 4096 --random-output 512 --random-range-ratio 0.5 --output-file sglang_offline_benchmark.jsonl

# ShareGPT dataset, num prompts 3k
python3 -m sglang.bench_serving --backend sglang --num-prompts 3000 --output-file sglang_offline_benchmark.jsonl

# get output token throughput
cat sglang_offline_benchmark.jsonl | cut -d':' -f12 | cut -d',' -f1
```

#### Online benchmark

```bash
# Random dataset, Input [512, 4096], Output [128, 1024], request rate 1, num prompts 300
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 300 --request-rate 1 --output-file sglang_online_benchmark.jsonl

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 2, num prompts 600
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 600 --request-rate 2 --output-file sglang_online_benchmark.jsonl

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 4, num prompts 1200
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 1200 --request-rate 4 --output-file sglang_online_benchmark.jsonl

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 8, num prompts 2400
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 2400 --request-rate 8 --output-file sglang_online_benchmark.jsonl

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 16, num prompts 3200
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 3200 --request-rate 16 --output-file sglang_online_benchmark.jsonl

# get median e2e latency
cat sglang_online_benchmark.jsonl | cut -d':' -f9 | cut -d',' -f1
```

## Other

We tried using vLLM 0.5.3.post1, but it often crashes under high loads, and it seems to have similar or worse performance compared to vLLM 0.5.2 from our partial benchmarking, so we are using the older version, vLLM 0.5.2.

Preparation for TensorRT LLM can refer to https://github.com/sgl-project/tensorrt-demo. Specifically, we used a batch size of 512, a max input length of 8192, and a max number of tokens of 8192. The instance count for preprocessing and postprocessing in Triton Server is 16.

```bash
# vLLM
pip install vllm==0.5.2

# Meta-Llama-3-8B-Instruct
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --disable-log-requests

# meta-llama/Meta-Llama-3-70B-Instruct
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --disable-log-requests --tensor 8

# neuralmagic/Meta-Llama-3-70B-Instruct-FP8
python -m vllm.entrypoints.openai.api_server --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-log-requests --tensor 8
```

```bash
wget https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/bench_serving.py
```

```bash
# vLLM Offline

# Random dataset, Input [512, 1024], Output [512, 1024], num prompts 3k
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5 --output-file vllm_offline_benchmark.jsonl

# Random dataset, Input [2048, 4096], Output [512, 1024], num prompts 3k
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 3000 --random-input 4096 --random-output 1024 --random-range-ratio 0.5 --output-file vllm_offline_benchmark.jsonl

# Random dataset, Input [512, 1024], Output [256, 512], num prompts 3k
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 512 --random-range-ratio 0.5 --output-file vllm_offline_benchmark.jsonl

# Random dataset, Input [2048, 4096], Output [256, 512], num prompts 3k
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 3000 --random-input 4096 --random-output 512 --random-range-ratio 0.5 --output-file vllm_offline_benchmark.jsonl

# ShareGPT dataset, num prompts 3k
python3 bench_serving.py --backend vllm --num-prompts 3000 --output-file vllm_offline_benchmark.jsonl

# get output token throughput
cat vllm_offline_benchmark.jsonl | cut -d':' -f12 | cut -d',' -f1
```

```bash
# vLLM Online

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 1, num prompts 300
python3 bench_serving.py --backend vllm --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 300 --request-rate 1 --output-file vllm_online_benchmark.jsonl

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 2, num prompts 600
python3 bench_serving.py --backend vllm --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 600 --request-rate 2 --output-file vllm_online_benchmark.jsonl

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 4, num prompts 1200
python3 bench_serving.py --backend vllm --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 1200 --request-rate 4 --output-file vllm_online_benchmark.jsonl

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 8, num prompts 2400
python3 bench_serving.py --backend vllm --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 2400 --request-rate 8 --output-file vllm_online_benchmark.jsonl

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 16, num prompts 3200
python3 bench_serving.py --backend vllm --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 3200 --request-rate 16 --output-file vllm_online_benchmark.jsonl

# get median e2e latency
cat vllm_online_benchmark.jsonl | cut -d':' -f9 | cut -d',' -f1
```

```bash
# TensorRT LLM Offline 8B

# Random dataset, Input [512, 1024], Output [512, 1024], num prompts 3k
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5 --output-file trt_offline_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# Random dataset, Input [2048, 4096], Output [512, 1024], num prompts 3k
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 3000 --random-input 4096 --random-output 1024 --random-range-ratio 0.5 --output-file trt_offline_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# Random dataset, Input [512, 1024], Output [256, 512], num prompts 3k
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 512 --random-range-ratio 0.5 --output-file trt_offline_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# Random dataset, Input [2048, 4096], Output [256, 512], num prompts 3k
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 3000 --random-input 4096 --random-output 512 --random-range-ratio 0.5 --output-file trt_offline_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# ShareGPT dataset, num prompts 3k
python3 bench_serving.py --backend trt --num-prompts 3000 --output-file trt_offline_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# get output token throughput
cat trt_offline_benchmark_8b.jsonl | cut -d':' -f12 | cut -d',' -f1
```

```bash
# TensorRT LLM Online 8B

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 1, num prompts 300
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 300 --request-rate 1 --output-file trt_online_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 2, num prompts 600
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 600 --request-rate 2 --output-file trt_online_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 4, num prompts 1200
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 1200 --request-rate 4 --output-file trt_online_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 8, num prompts 2400
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 2400 --request-rate 8 --output-file trt_online_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 16, num prompts 3200
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 3200 --request-rate 16 --output-file trt_online_benchmark_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct

# get median e2e latency
cat trt_online_benchmark_8b.jsonl | cut -d':' -f9 | cut -d',' -f1
```

```bash
# TensorRT LLM Offline 70B

# Random dataset, Input [512, 1024], Output [512, 1024], num prompts 3k
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 1024 --random-range-ratio 0.5 --output-file trt_offline_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# Random dataset, Input [2048, 4096], Output [512, 1024], num prompts 3k
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 3000 --random-input 4096 --random-output 1024 --random-range-ratio 0.5 --output-file trt_offline_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# Random dataset, Input [512, 1024], Output [256, 512], num prompts 3k
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 3000 --random-input 1024 --random-output 512 --random-range-ratio 0.5 --output-file trt_offline_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# Random dataset, Input [2048, 4096], Output [256, 512], num prompts 3k
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 3000 --random-input 4096 --random-output 512 --random-range-ratio 0.5 --output-file trt_offline_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# ShareGPT dataset, num prompts 3k
python3 bench_serving.py --backend trt --num-prompts 3000 --output-file trt_offline_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# get output token throughput
cat trt_offline_benchmark_70b.jsonl | cut -d':' -f12 | cut -d',' -f1
```

```bash
# TensorRT LLM Online 70B

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 1, num prompts 300
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 300 --request-rate 1 --output-file trt_online_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 2, num prompts 600
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 600 --request-rate 2 --output-file trt_online_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 4, num prompts 1200
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 1200 --request-rate 4 --output-file trt_online_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 8, num prompts 2400
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 2400 --request-rate 8 --output-file trt_online_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# Random dataset, Input [512, 4096], Output [128, 1024], request rate 16, num prompts 3200
python3 bench_serving.py --backend trt --dataset-name random --random-input 4096 --random-output 1024 --random-range-ratio 0.125 --num-prompts 3200 --request-rate 16 --output-file trt_online_benchmark_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct

# get median e2e latency
cat trt_online_benchmark_70b.jsonl | cut -d':' -f9 | cut -d',' -f1
```
