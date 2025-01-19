# DeepSeek V3 Support

The SGLang and DeepSeek teams collaborated to get DeepSeek V3 FP8 running on NVIDIA and AMD GPUs **from day one**. SGLang also supports [MLA optimization](https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-mla-throughput-optimizations) and [DP attention](https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models), making SGLang one of the best open-source LLM engines for running DeepSeek models. SGLang is the inference engine recommended by the official [DeepSeek team](https://github.com/deepseek-ai/DeepSeek-V3/tree/main?tab=readme-ov-file#62-inference-with-sglang-recommended).

Special thanks to Meituan's Search & Recommend Platform Team and Baseten's Model Performance Team for implementing the model, and DataCrunch for providing GPU resources.

For optimizations made on the DeepSeek series models regarding SGLang, please refer to [DeepSeek Model Optimizations in SGLang](https://docs.sglang.ai/references/deepseek.html).

## Hardware Recommendation
- 8 x NVIDIA H200 GPUs

If you do not have GPUs with large enough memory, please try multi-node tensor parallelism. There is an example serving with [2 H20 nodes](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208) below.

## Installation & Launch

If you encounter errors when starting the server, ensure the weights have finished downloading. It's recommended to download them beforehand or restart multiple times until all weights are downloaded.

### Using Docker (Recommended)
```bash
# Pull latest image
# https://hub.docker.com/r/lmsysorg/sglang/tags
docker pull lmsysorg/sglang:latest

# Launch
docker run --gpus all --shm-size 32g -p 30000:30000 -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code --port 30000
```

For high QPS scenarios, add the `--enable-dp-attention` argument to boost throughput.

### Using pip
```bash
# Installation
pip install "sglang[all]>=0.4.1.post5" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer

# Launch
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

For high QPS scenarios, add the `--enable-dp-attention` argument to boost throughput.

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

### Example: Serving with two H20*8 nodes
For example, there are two H20 nodes, each with 8 GPUs. The first node's IP is `10.0.0.1`, and the second node's IP is `10.0.0.2`. Please **use the first node's IP** for both commands.

If the command fails, try setting the `GLOO_SOCKET_IFNAME` parameter. For more information, see [Common Environment Variables](https://pytorch.org/docs/stable/distributed.html#common-environment-variables).

```bash
# node 1
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 0 --trust-remote-code

# node 2
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --dist-init-addr 10.0.0.1:5000 --nnodes 2 --node-rank 1 --trust-remote-code
```

If you have two H100 nodes, the usage is similar to the aforementioned H20.

### Example: Serving with two H200*8 nodes and docker
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

## DeepSeek V3 Optimization Plan

https://github.com/sgl-project/sglang/issues/2591
