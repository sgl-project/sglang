# 

## Deployment Guide for SGLang DeepSeekR1 FP8 and NVFP4

## Introduction

This deployment guide provides step-by-step instructions for running the DeepSeek R1 model using SGLang, specifically configured for a single-node NVIDIA B200 system, with plans to extend support to NVL72 platforms. It delivers installation guides, setup steps, and custom installation procedures for both SGLang and the required FlashInfer components.  
The guide covers the requirements for running DeepSeek R1, including blockscale FP8 and FP4 datatypes for Blackwell.   
Additional highlights:

* The deployment starts with obtaining model weights, then prepares the software—ensuring custom FP8/FP4 quantization and blockscale datatype support are included.  
* Instructions detail how to configure SGLang runtime for DeepSeek R1, including hardware (B200) tuning and optional extension to NVL72 multi-node infrastructures as support becomes available.  
* Finally, it walks through server launch, inference validation, and best practices for production operations, ensuring you have a reliable, high-throughput setup for advanced language model inference.

## Access & Licensing

To use DeepSeekR1, you must first agree to DeepSeek’s Community License ([https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/main/LICENSE](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/main/LICENSE)). NVIDIA’s quantized versions (FP8 and FP4) are built on top of the base model and are available for research and commercial use under the same license.

## Models

* FP4 (4-bit quantized): [nvidia/DeepSeek-R1-0528-FP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4)  
* FP8 (8-bit quantized): [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)

## Prerequisites

OS: Linux  
Drivers: CUDA Driver 575 or above  
GPU: Blackwell architecture  
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

## Building Docker Image

Build a Docker image with SGLang and all dependencies using the official SGLang base image as a starting point. The provided example is for x86 (NVIDIA B200);

```shell
# This is x86 container. We will modify this to multiplatform blackwell in the next iteration
FROM lmsysorg/sglang:b200-cu129

WORKDIR /workspace

# Install latest pip and necessary development tools
RUN apt-get update && apt-get install -y git python3-dev build-essential ninja-build && \
    pip install --upgrade pip

# Install lm-eval harness (latest version from commit or main branch)
RUN pip install --no-build-isolation "lm-eval[api] @ git+https://github.com/EleutherAI/lm-evaluation-harness@4f8195f"

# Clone SGLang and FlashInfer sources
RUN git clone https://github.com/sgl-project/sglang.git
 /workspace/sglang/sglang-src -b #HASH && \
    git clone --recursive https://github.com/FlashInfer-ai/FlashInfer.git -b v0.2.12 /workspace/flashinfer 


# Build/install SGLang from source
RUN cd /workspace/sglang/sglang-src && \
    pip install --break-system-packages -e python

# Build/install FlashInfer from source, with AOT kernels for Blackwell
RUN cd /workspace/flashinfer && \
    pip install ninja && \
    export TORCH_CUDA_ARCH_LIST="9.0a 9.0 10.0 10.0a" && \
    python -m pip install --no-build-isolation -e . -v 

# Install any additional dependencies for your workload
RUN python3 -m pip install --no-cache-dir \
    "nvidia-cudnn-cu12>=9.11" \
    "nvidia-cudnn-frontend>=1.13"

RUN pip install --break-system-packages httpx openai

ENV PYTHONPATH=/workspace/sglang/sglang-src:/workspace/flashinfer:${PYTHONPATH}
```

### Running the Docker Container

Once built, use this script to start a container with full GPU access and an expanded shared memory segment (important for multi-GPU workloads):

```shell
#!/bin/bash

IMAGE_NAME="sglang"

docker build --pull --no-cache -t "$IMAGE_NAME" .

docker run \
  --network=host \ 
  --gpus=all \
  --shm-size=256gb \
  -ti --rm "$IMAGE_NAME" bash
```

## Downloading the Quantized Model Weights

### Downloading Inside Docker

To fetch the weights directly within your Docker container, use these commands:

For FP4:

```shell
MODEL_PATH=/workspace/model
mkdir -p $MODEL_PATH
cd $MODEL_PATH
git clone https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4
```

For FP8:

```shell
MODEL_PATH=/workspace/model
mkdir -p $MODEL_PATH
cd $MODEL_PATH
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-05284
```

## Launch the SGLang Server and Client

SGLang follows a client–server architecture for efficient large language model (LLM) serving. To use it, you need to run two separate processes: one for the server and one for the client.

### Server Process 

Below is an example command to launch the SGLang server with the FP4 DSR1 model. The explanation of each flag is shown in the “Configs and Parameters” section.

launch\_server.sh

```shell
python3 -m sglang.launch_server \
--model-path nvidia/DeepSeek-R1-0528-FP4 \
--trust-remote-code 
--quantization modelopt_fp4 \
--tp 8 
--enable-flashinfer-cutlass-moe 

# or if you have downloaded the model: 

# python3 -m sglang.launch_server \
# --model-path nvidia/DeepSeek-R1-0528-FP4 \
# --trust-remote-code 
# --quantization modelopt_fp4 \
# --tp 8 
# --enable-flashinfer-cutlass-moe 
```

After the server is set up, the client can now send prompt requests to the server and receive results.

**Note on Quantization Choice:**  
For Hopper, FP8 offers the best performance for most workloads. For Blackwell, NVFP4 provides additional memory savings and throughput gains, but may require tuning to maintain accuracy on certain tasks.

## Configuration Profiles: High Throughput vs. Low Latency

SGLang supports tuning for different deployment needs. For best results, you should select configuration parameters according to your workload—whether you prioritize maximizing throughput (processing many concurrent requests) or minimizing response latency (faster lower number of requests).

### FP8, High Throughput Configuration

Use these settings to maximize the number of concurrent requests and model utilization across multiple GPUs.

**Server:**

```shell
SGL_ENABLE_JIT_DEEPGEMM=0 SGLANG_CUTLASS_MOE=1 \
python3 -m sglang.launch_server \
  --tokenizer-path /workspace/model/DeepSeek-R1-0528 \
  --trust-remote-code \
  --enable-dp-attention \
  --disable-radix-cache \
  --max-running-requests 3072 \
  --chunked-prefill-size 32768 \
  --mem-fraction-static 0.89 \
  --max-prefill-tokens 32768 \
  --model-path /workspace/model/DeepSeek-R1-0528 \
  --tensor-parallel-size 8 \
  --data-parallel-size 8 \
  --attention-backend trtllm-mla \
  --kv-cache-dtype fp8_e4m3
```

**Client (Benchmarking):**

```
python3 -m sglang.bench_serving \
  --model deepseek-ai/DeepSeek-R1-0528 \
  --dataset-name random \
  --backend sglang-oai \
  --random-range-ratio 1 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --max-concurrency 3072 \
  --num-prompts 6148

```

*This setup is ideal for batch processing or when serving many users at once.*

### FP8, Low Latency Configuration (TEP8)

Choose these parameters to reduce the time it takes to get a response for a single or very few requests, e.g., for demo or conversational agents requiring snappy replies.

**Server:**

```shell
SGL_ENABLE_JIT_DEEPGEMM=0 \
python3 -m sglang.launch_server \
  --model-path /workspace/model/DeepSeek-R1-0528/ \
  --trust-remote-code \
  --tp 8 \
  --enable-ep-moe \
  --enable-flashinfer-trtllm-moe

```

**Client (Benchmarking):**

```shell
python3 -m sglang.bench_serving \
  --model /workspace/model/DeepSeek-R1-0528 \
  --dataset-name random \
  --backend sglang-oai \
  --random-range-ratio 1 \
  --random-input-len 1024 \
  --random-output-len 8192 \
  --max-concurrency 1 \
  --num-prompts 10

```

**Sample Output**  
Below is an example of the output produced with the configuration

```shell
# Server Initialization

[2025-08-01 21:05:23 TP0 EP0] Registering 6273 cuda graph addresses
[2025-08-01 21:05:23 TP2 EP2] Registering 6273 cuda graph addresses
[2025-08-01 21:05:23 TP5 EP5] Registering 6273 cuda graph addresses
[2025-08-01 21:05:23 TP4 EP4] Registering 6273 cuda graph addresses
[2025-08-01 21:05:23 TP6 EP6] Registering 6273 cuda graph addresses
[2025-08-01 21:05:23 TP7 EP7] Registering 6273 cuda graph addresses
[2025-08-01 21:05:23 TP3 EP3] Registering 6273 cuda graph addresses
[2025-08-01 21:05:23 TP1 EP1] Registering 6273 cuda graph addresses
[2025-08-01 21:05:24 TP0 EP0] Capture cuda graph end. Time elapsed: 26.07 s. mem usage=6.37 GB. avail mem=22.72 GB.
[2025-08-01 21:05:24 TP0 EP0] max_total_num_tokens=980344, chunked_prefill_size=16384, max_prefill_tokens=16384, max_running_requests=3063, context_len=163840, available_gpu_mem=22.72 GB
[2025-08-01 21:05:24] INFO:     Started server process [16140]
[2025-08-01 21:05:24] INFO:     Waiting for application startup.
[2025-08-01 21:05:24] INFO:     Application startup complete.
[2025-08-01 21:05:24] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
[2025-08-01 21:05:25] INFO:     127.0.0.1:48312 - "GET /get_model_info HTTP/1.1" 200 OK
[2025-08-01 21:05:25 TP0 EP0] Prefill batch. #new-seq: 1, #new-token: 7, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, 
[2025-08-01 21:05:25 TP0 EP0] Using configuration from /workspace/sglang/sglang-src/python/sglang/srt/layers/quantization/configs/N=4096,K=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json for W8A8 Block FP8 kernel.
[2025-08-01 21:05:28] INFO:     127.0.0.1:48316 - "POST /generate HTTP/1.1" 200 OK
[2025-08-01 21:05:28] The server is fired up and ready to roll!


#Client -- you would see results like below
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 1         
Successful requests:                     10        
Benchmark duration (s):                  1287.25   
Total input tokens:                      10240     
Total generated tokens:                  ---     
Total generated tokens (retokenized):    ---     
Request throughput (req/s):              ---     
Input token throughput (tok/s):          ---      
Output token throughput (tok/s):         ---     
Total token throughput (tok/s):          ---     
Concurrency:                             ---      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   --- 
Median E2E Latency (ms):                 --- 
---------------Time to First Token----------------
Mean TTFT (ms):                          ---    
Median TTFT (ms):                        ---    
P99 TTFT (ms):                           ---   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           ---     
Median ITL (ms):                         ---     
P95 ITL (ms):                            ---     
P99 ITL (ms):                            ---     
Max ITL (ms):                            ---     
==================================================

```

## Key Performance Metrics

| `Median Time to First Token (ms)` | The typical time elapsed from when a request is sent until the first output token is generated in milliseconds. |
| :---- | :---- |
| `Median ITL  (ms)` | The typical time delay between the completion of one token and the completion of the next in milliseconds. |
| `Median E2E Latency (ms)` | The typical total time from when a request is submitted until the final token of the response is received in milliseconds.   |
| `Total token throughput (tok/s)` | The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens in tokens/second.  |

### FP4 Serve Command

```shell
python3 -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-0528-FP4 \
  --trust-remote-code \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --disable-radix-cache \
  --enable-ep-moe \
  --moe-dense-tp-size 1 \
  --max-running-requests 8192 \
  --mem-fraction-static 0.85 \
  --cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
  --kv-cache-dtype fp8_e4m3 \
  --quantization modelopt_fp4 \
  --attention-backend trtllm_mla \
  --stream-interval 10 \
  --enable-flashinfer-cutlass-moe \
  --tensor-parallel-size=8 \
  --data-parallel-size=8

```

### Testing Accuracy

When the server is still running, we can run accuracy command:

```shell
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --num-questions 1316 --parallel 1316 
```

### Testing Performance

To benchmark the performance, you can use the “`sglang.bench_serving`” command

Run\_performance.sh

```shell
python3 -m sglang.bench_serving \
--backend sglang \
--model nvidia/DeepSeek-R1-0528-FP4 \
--num-prompts 512 \
--dataset-name random \
--random-input-len 1000 \
--random-output-len 1000 \
--random-range-ratio 1 \
--max-concurrency 512  \--warmup-request 512 \--save-result --result-filename vllm_benchmark_serving_results.json

```

### Server Flag Descriptions

| `--tokenizer-path` | The path of the tokenizer. |
| :---- | :---- |
|   `--trust-remote-code` | Whether or not to allow for custom models defined on the Hub in their own modeling files. |
| `--enable-dp-attention`  | Whether to you data parallel scheme over the attention layer |
| `--disable-radix-cache` | Disable RadixAttention for prefix caching. |
| `--max-running-requests` | The maximum number of running requests |
| `--chunked-prefill-size` | The maximum number of tokens in a chunk for the chunked prefill. Setting this to \-1 means disabling chunked prefill  |
| `--mem-fraction-static`  | The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.  |
| `--max-prefill-tokens`  | The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model’s maximum context length. |
| `--model-path` | The path of the model weights. This can be a local folder or a Hugging Face repo ID. |
| `--tensor-parallel-size` | Tensor parallel size |
| `--data-parallel-size` | Data parallel size for used on for the attention layer on DSR1 |
| `--attention-backend` | Which backend to use for the attention layer |

### Client Flag Description

| `--model` | The fp4 or the fp8 DSR1 model |
| :---- | :---- |
| `--num-prompts` | Total number of prompts to process |
| `--dataset-name` | Which dataset to use for benchmarking. We use a “random” dataset here. |
| `--random-input-len` | Specifies the average input sequence length. |
| `--random-output-len` | Specifies the average output sequence length. |
| `--max-concurrency` | Maximum number of in-flight requests. We recommend matching this with the “--max-num-seqs” flag used to launch the server. |
| `--save-result --result-filename` | Output location for the performance benchmarking result.  |


