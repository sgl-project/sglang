# How to reproduce the benchmark results of SGLang

## Prerequisite

### Install the latest SGLang

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.2.7

pip install --upgrade pip
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
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --enable-torch-compile --disable-radix-cache

# Meta-Llama-3.1-70B-Instruct
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --disable-radix-cache --tp 8

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
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 4000 --random-input 1024 --random-output 1024 --output-file offline.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 5000 --random-input 1024 --random-output 512 --output-file offline.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 2048 --output-file offline.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 2000 --random-input 4096 --random-output 1024 --output-file offline.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 6000 --random-input 256 --random-output 512 --output-file offline.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 3000 --output-file offline.jsonl
cat offline.jsonl | cut -d':' -f12 | cut -d',' -f1
```

#### Online benchmark

```bash
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 300 --request-rate 1 --output-file online.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 600 --request-rate 2 --output-file online.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 1200 --request-rate 4 --output-file online.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 2400 --request-rate 8 --output-file online.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 3200 --request-rate 16 --output-file online.jsonl
cat online.jsonl | cut -d':' -f9 | cut -d',' -f1
```

## Other

We tried using vLLM 0.5.3.post1, but it often crashes under high loads, and it seems to have similar or worse performance compared to vLLM 0.5.2 from our partial benchmarking, so we are using the older version, vLLM 0.5.2.

Preparation for TensorRT LLM can refer to https://github.com/sgl-project/tensorrt-demo. Specifically, we used a batch size of 512, a max input length of 8192, and a max number of tokens of 8192. The instance count for preprocessing and postprocessing in Triton Server is 16.

```bash
# vLLM
pip install vllm==0.5.2
pip install jsonschema==4.21.1

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

python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 4000 --random-input 1024 --random-output 1024 --output-file offline_vllm.jsonl
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 5000 --random-input 1024 --random-output 512 --output-file offline_vllm.jsonl
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 2048 --output-file offline_vllm.jsonl
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 2000 --random-input 4096 --random-output 1024 --output-file offline_vllm.jsonl
python3 bench_serving.py --backend vllm --dataset-name random --num-prompts 6000 --random-input 256 --random-output 512 --output-file offline_vllm.jsonl
python3 bench_serving.py --backend vllm --dataset-name sharegpt --num-prompts 3000 --output-file offline_vllm.jsonl
cat offline_vllm.jsonl | cut -d':' -f12 | cut -d',' -f1
```

```bash
# vLLM Online

python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 300 --request-rate 1 --output-file online_vllm.jsonl
python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 600 --request-rate 2 --output-file online_vllm.jsonl
python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 1200 --request-rate 4 --output-file online_vllm.jsonl
python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 2400 --request-rate 8 --output-file online_vllm.jsonl
python3 bench_serving.py --backend vllm --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 3200 --request-rate 16 --output-file online_vllm.jsonl
cat online_vllm.jsonl | cut -d':' -f9 | cut -d',' -f1
```

```bash
# TensorRT LLM Offline 8B

python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --num-prompts 4000 --random-input 1024 --random-output 1024 --output-file offline_trt_8b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --num-prompts 5000 --random-input 1024 --random-output 512 --output-file offline_trt_8b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 2048 --output-file offline_trt_8b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --num-prompts 2000 --random-input 4096 --random-output 1024 --output-file offline_trt_8b.jsonl
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 6000 --random-input 256 --random-output 512 --output-file offline_trt_8b.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name sharegpt --num-prompts 3000 --output-file offline_trt_8b.jsonl
cat offline_trt_8b.jsonl | cut -d':' -f12 | cut -d',' -f1
```

```bash
# TensorRT LLM Online 8B

python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 300 --request-rate 1 --output-file online_trt_8b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 600 --request-rate 2 --output-file online_trt_8b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 1200 --request-rate 4 --output-file online_trt_8b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 2400 --request-rate 8 --output-file online_trt_8b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-8B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 3200 --request-rate 16 --output-file online_trt_8b.jsonl
cat online_trt_8b.jsonl | cut -d':' -f9 | cut -d',' -f1
```

```bash
# TensorRT LLM Offline 70B

python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name random --num-prompts 4000 --random-input 1024 --random-output 1024 --output-file offline_trt_70b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name random --num-prompts 5000 --random-input 1024 --random-output 512 --output-file offline_trt_70b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name random --num-prompts 1000 --random-input 4096 --random-output 2048 --output-file offline_trt_70b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name random --num-prompts 2000 --random-input 4096 --random-output 1024 --output-file offline_trt_70b.jsonl
python3 bench_serving.py --backend trt --dataset-name random --num-prompts 6000 --random-input 256 --random-output 512 --output-file offline_trt_70b.jsonl --model meta-llama/Meta-Llama-3-70B-Instruct
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name sharegpt --num-prompts 3000 --output-file offline_trt_70b.jsonl
cat offline_trt_70b.jsonl | cut -d':' -f12 | cut -d',' -f1
```

```bash
# TensorRT LLM Online 70B

python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 300 --request-rate 1 --output-file online_trt_70b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 600 --request-rate 2 --output-file online_trt_70b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 1200 --request-rate 4 --output-file online_trt_70b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 2400 --request-rate 8 --output-file online_trt_70b.jsonl
python3 bench_serving.py --backend trt --model meta-llama/Meta-Llama-3-70B-Instruct --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 3200 --request-rate 16 --output-file online_trt_70b.jsonl
cat online_trt_70b.jsonl | cut -d':' -f9 | cut -d',' -f1
```
