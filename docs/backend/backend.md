# Backend: SGLang Runtime (SRT)
The SGLang Runtime (SRT) is an efficient serving engine.

## Quick Start
Launch a server
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

Send a request
```
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 16,
      "temperature": 0
    }
  }'
```

Learn more about the argument specification, streaming, and multi-modal support [here](../references/sampling_params.md).

## OpenAI Compatible API
In addition, the server supports OpenAI-compatible APIs.

```python
import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

# Text completion
response = client.completions.create(
	model="default",
	prompt="The capital of France is",
	temperature=0,
	max_tokens=32,
)
print(response)

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

# Text embedding
response = client.embeddings.create(
    model="default",
    input="How are you today",
)
print(response)
```

It supports streaming, vision, and almost all features of the Chat/Completions/Models/Batch endpoints specified by the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/).

## Additional Server Arguments
- To enable multi-GPU tensor parallelism, add `--tp 2`. If it reports the error "peer access is not supported between these two devices", add `--enable-p2p-check` to the server launch command.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2
```
- To enable multi-GPU data parallelism, add `--dp 2`. Data parallelism is better for throughput if there is enough memory. It can also be used together with tensor parallelism. The following command uses 4 GPUs in total.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --dp 2 --tp 2
```
- If you see out-of-memory errors during serving, try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`. The default value is `0.9`.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --mem-fraction-static 0.7
```
- See [hyperparameter tuning](../references/hyperparameter_tuning.md) on tuning hyperparameters for better performance.
- If you see out-of-memory errors during prefill for long prompts, try to set a smaller chunked prefill size.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --chunked-prefill-size 4096
```
- To enable torch.compile acceleration, add `--enable-torch-compile`. It accelerates small models on small batch sizes. This does not work for FP8 currently.
- To enable torchao quantization, add `--torchao-config int4wo-128`. It supports other [quantization strategies (INT8/FP8)](https://github.com/sgl-project/sglang/blob/v0.3.6/python/sglang/srt/server_args.py#L671) as well.
- To enable fp8 weight quantization, add `--quantization fp8` on a fp16 checkpoint or directly load a fp8 checkpoint without specifying any arguments.
- To enable fp8 kv cache quantization, add `--kv-cache-dtype fp8_e5m2`.
- If the model does not have a chat template in the Hugging Face tokenizer, you can specify a [custom chat template](../references/custom_chat_template.md).

- To run tensor parallelism on multiple nodes, add `--nnodes 2`. If you have two nodes with two GPUs on each node and want to run TP=4, let `sgl-dev-0` be the hostname of the first node and `50000` be an available port, you can use the following commands. If you meet deadlock, please try to add `--disable-cuda-graph`
```
# Node 0
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --nccl-init sgl-dev-0:50000 --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --nccl-init sgl-dev-0:50000 --nnodes 2 --node-rank 1
```

## Engine Without HTTP Server

We also provide an inference engine **without a HTTP server**. For example,

```python
import sglang as sgl

def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = {"temperature": 0.8, "top_p": 0.95}
    llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if __name__ == "__main__":
    main()
```

This can be used for offline batch inference and building custom servers.
You can view the full example [here](https://github.com/sgl-project/sglang/tree/main/examples/runtime/engine).

## Use Models From ModelScope
<details>
<summary>More</summary>

To use a model from [ModelScope](https://www.modelscope.cn), set the environment variable SGLANG_USE_MODELSCOPE.
```
export SGLANG_USE_MODELSCOPE=true
```
Launch [Qwen2-7B-Instruct](https://www.modelscope.cn/models/qwen/qwen2-7b-instruct) Server
```
SGLANG_USE_MODELSCOPE=true python -m sglang.launch_server --model-path qwen/Qwen2-7B-Instruct --port 30000
```

Or start it by docker.
```bash
docker run --gpus all \
    -p 30000:30000 \
    -v ~/.cache/modelscope:/root/.cache/modelscope \
    --env "SGLANG_USE_MODELSCOPE=true" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```

</details>

## Example: Run Llama 3.1 405B
<details>
<summary>More</summary>

```bash
# Run 405B (fp8) on a single node
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 --tp 8

# Run 405B (fp16) on two nodes
## on the first node, replace the `172.16.4.52:20000` with your own first node ip address and port
GLOO_SOCKET_IFNAME=eth0 python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --nccl-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 0 --disable-cuda-graph

## on the first node, replace the `172.16.4.52:20000` with your own first node ip address and port
GLOO_SOCKET_IFNAME=eth0 python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-405B-Instruct --tp 16 --nccl-init-addr 172.16.4.52:20000 --nnodes 2 --node-rank 1 --disable-cuda-graph
```

</details>
