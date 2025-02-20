# DeepSeek V3 Support

The SGLang and DeepSeek teams collaborated to get DeepSeek V3 FP8 running on NVIDIA and AMD GPUs **from day one**. SGLang also supports [MLA optimization](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) and [DP attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models), making SGLang one of the best open-source LLM engines for running DeepSeek models. SGLang is the inference engine recommended by the official [DeepSeek team](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended).

Special thanks to Meituan's Search & Recommend Platform Team and Baseten's Model Performance Team for implementing the model, and DataCrunch for providing GPU resources.

For optimizations made on the DeepSeek series models regarding SGLang, please refer to [DeepSeek Model Optimizations in SGLang](https://docs.sglang.ai/references/deepseek.html).

## Hardware Recommendation

- 8 x NVIDIA H200 GPUs

If you do not have GPUs with large enough memory, please try multi-node tensor parallelism. There is an example serving with [2 H20 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208) below.

For running on AMD MI300X, use this as a reference. [Running DeepSeek-R1 on a single NDv5 MI300X VM](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726)

## Installation & Launch

If you encounter errors when starting the server, ensure the weights have finished downloading. It's recommended to download them beforehand or restart multiple times until all weights are downloaded.

### Using Docker (Recommended)

```bash
# Pull latest image
# https://hub.docker.com/r/lmsysorg/sglang/tags
docker pull lmsysorg/sglang:latest

# Launch
docker run --gpus all --shm-size 32g -p 30000:30000 -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host --network=host --privileged lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
```

If you are using RDMA, please note that:

1. `--network host` and `--privileged` are required by RDMA. If you don't need RDMA, you can remove them.
2. You may need to set `NCCL_IB_GID_INDEX` if you are using RoCE, for example: `export NCCL_IB_GID_INDEX=3`.

Add [performance optimization options](#performance-optimization-options) as needed.

### Using pip

```bash
# Installation
pip install "sglang[all]>=0.4.3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# Launch
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

Add [performance optimization options](#performance-optimization-options) as needed.

<a id="option_args"></a>

### Performance Optimization Options

[MLA optimizations](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) are enabled by default. Here are some optional optimizations can be enabled as needed.

- [Data Parallelism Attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models): For high QPS scenarios, add the `--enable-dp-attention` argument to boost throughput.
- [Torch.compile Optimization](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#torchcompile-latency-optimizations): Add `--enable-torch-compile` argument to enable it. This will take some time while server starts. The maximum batch size for torch.compile optimization can be controlled with `--torch-compile-max-bs`. It's recommended to set it between `1` and `8`. (e.g., `--torch-compile-max-bs 8`)

### Example: Sending requests with OpenAI API

```python3
import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

# Chat completion
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(response)
```

### Example: Serving with two H20\*8 nodes

For example, there are two H20 nodes, each with 8 GPUs. The first node's IP is `10.0.0.1`, and the second node's IP is `10.0.0.2`. Please **use the first node's IP** for both commands.

If the command fails, try setting the `GLOO_SOCKET_IFNAME` parameter. For more information, see [Common Environment Variables](https://pytorch.org/docs/stable/distributed.html#common-environment-variables).

If the multi nodes support NVIDIA InfiniBand and encounter hanging issues during startup, consider adding the parameter `export NCCL_IB_GID_INDEX=3`. For more information, see [this](https://github.com/sgl-project/sglang/issues/3516#issuecomment-2668493307).

```bash
# node 1
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 0 --trust-remote-code

# node 2
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 1 --trust-remote-code
```

If you have two H100 nodes, the usage is similar to the aforementioned H20.

> **Note that the launch command here does not enable Data Parallelism Attention or `torch.compile` Optimization**. For optimal performance, please refer to the command options in [Performance Optimization Options](#option_args).

### Example: Serving with two H200\*8 nodes and docker

There are two H200 nodes, each with 8 GPUs. The first node's IP is `192.168.114.10`, and the second node's IP is `192.168.114.11`. Configure the endpoint to expose it to another Docker container using `--host 0.0.0.0` and `--port 40000`, and set up communications with `--dist-init-addr 192.168.114.10:20000`.
A single H200 with 8 devices can run DeepSeek V3, the dual H200 setup is just to demonstrate multi-node usage.

```bash
# node 1
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name sglang_multinode1 \
    -it \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 40000
```

```bash
# node 2
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name sglang_multinode2 \
    -it \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 192.168.114.10:20000 --nnodes 2 --node-rank 1 --trust-remote-code --host 0.0.0.0 --port 40000
```

To ensure functionality, we include a test from a client Docker container.

```bash
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name sglang_multinode_client \
    -it \
    --rm \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.bench_serving --backend sglang --dataset-name random --random-input 1 --random-output 512 --random-range-ratio 1 --num-prompts 1 --host 0.0.0.0 --port 40000 --output-file "deepseekv3_multinode.jsonl"
```

> **Note that the launch command here does not enable Data Parallelism Attention or `torch.compile` Optimization**. For optimal performance, please refer to the command options in [Performance Optimization Options](#option_args).

### Example: Serving with four A100\*8 nodes

To serve DeepSeek-V3 with A100 GPUs, we need to convert the [FP8 model checkpoints](https://huggingface.co/deepseek-ai/DeepSeek-V3) to BF16 with [script](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py) mentioned [here](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py) first.

Since the BF16 model is over 1.3 TB, we need to prepare four A100 nodes, each with 8 80GB GPUs. Assume the first node's IP is `10.0.0.1`, and the converted model path is `/path/to/DeepSeek-V3-BF16`, we can have following commands to launch the server.

```bash
# node 1
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 --dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 30000

# node 2
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 --dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 1 --trust-remote-code

# node 3
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 --dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 2 --trust-remote-code

# node 4
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 --dist-init-addr 10.0.0.1:5000 --nnodes 4 --node-rank 3 --trust-remote-code
```

> **Note that the launch command here does not enable Data Parallelism Attention or `torch.compile` Optimization**. For optimal performance, please refer to the command options in [Performance Optimization Options](#option_args).

Then we can benchmark the accuracy and latency by accessing the first node's exposed port with the following example commands.

```bash
# bench accuracy
python3 benchmark/gsm8k/bench_sglang.py --num-questions 1319 --host http://10.0.0.1 --port 30000

# bench latency
python3 -m sglang.bench_one_batch_server --model None --base-url http://10.0.0.1:30000 --batch-size 1 --input-len 128 --output-len 128
```

### Example: Serving on any cloud or Kubernetes with SkyPilot

SkyPilot helps find cheapest available GPUs across any cloud or existing Kubernetes clusters and launch distributed serving with a single command. See details [here](https://github.com/skypilot-org/skypilot/tree/master/llm/deepseek-r1).

To serve on multiple nodes:

```bash
git clone https://github.com/skypilot-org/skypilot.git
# Serve on 2 H100/H200x8 nodes
sky launch -c r1 llm/deepseek-r1/deepseek-r1-671B.yaml --retry-until-up
# Serve on 4 A100x8 nodes
sky launch -c r1 llm/deepseek-r1/deepseek-r1-671B-A100.yaml --retry-until-up
```

#### Troubleshooting

If you encounter the following error with fp16/bf16 checkpoint:

```bash
ValueError: Weight output_partition_size = 576 is not divisible by weight quantization block_n = 128.
```

edit your `config.json` and remove the `quantization_config` block. For example:

```json
"quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128]
},
```

Removing this block typically resolves the error. For more details, see the discussion in [sgl-project/sglang#3491](https://github.com/sgl-project/sglang/issues/3491#issuecomment-2650779851).

## DeepSeek V3 Optimization Plan

https://github.com/sgl-project/sglang/issues/2591
