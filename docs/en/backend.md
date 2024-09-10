## Backend: SGLang Runtime (SRT)
The SGLang Runtime (SRT) is an efficient serving engine.

### Quick Start
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
Learn more about the argument format [here](docs/en/sampling_params.md).

### OpenAI Compatible API
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

It supports streaming, vision, and most features of the Chat/Completions/Models/Batch endpoints specified by the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/).

### Additional Server Arguments
- Add `--tp 2` to enable multi-GPU tensor parallelism. If it reports the error "peer access is not supported between these two devices", add `--enable-p2p-check` to the server launch command.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000 --tp 2
```
- Add `--dp 2` to enable multi-GPU data parallelism. Data parallelism is better for throughput if there is enough memory. It can also be used together with tensor parallelism. The following command uses 4 GPUs in total.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000 --dp 2 --tp 2
```
- If you see out-of-memory errors during serving, try to reduce the memory usage of the KV cache pool by setting a smaller value of `--mem-fraction-static`. The default value is `0.9`.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000 --mem-fraction-static 0.7
```
- See [hyperparameter_tuning.md](docs/en/hyperparameter_tuning.md) on tuning hyperparameters for better performance.
- If you see out-of-memory errors during prefill for long prompts, try to set a smaller chunked prefill size.
```
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000 --chunked-prefill-size 4096
```
- To enable torch.compile support, you can add `--enable-torch-compile`. It accelerates small models on small batch sizes.
- To enable fp8 weight quantization, you can add `--quantization fp8` on a fp16 checkpoint or directly load a fp8 checkpoint without specifying any arguments.
- To enable fp8 kv cache quanzation, you can add `--kv-cache-dtype fp8_e5m2`.
- If the model does not have a template in the Hugging Face tokenizer, you can specify a [custom chat template](docs/en/custom_chat_template.md).
- Add `--nnodes 2` to run tensor parallelism on multiple nodes. If you have two nodes with two GPUs on each node and want to run TP=4, let `sgl-dev-0` be the hostname of the first node and `50000` be an available port.
```
# Node 0
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --nccl-init sgl-dev-0:50000 --nnodes 2 --node-rank 0

# Node 1
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 4 --nccl-init sgl-dev-0:50000 --nnodes 2 --node-rank 1
```
 
### Supported Models

**Generative Models**
- Llama / Llama 2 / Llama 3 / Llama 3.1
- Mistral / Mixtral / Mistral NeMo
- Gemma / Gemma 2
- Qwen / Qwen 2 / Qwen 2 MoE
- DeepSeek / DeepSeek 2
- [LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)
  - `python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-7b-ov --port=30000 --chat-template=chatml-llava`
  - `python3 -m sglang.launch_server --model-path lmms-lab/llava-onevision-qwen2-72b-ov --port=30000 --tp-size=8 --chat-template=chatml-llava`
  - Query the server with the [OpenAI Vision API](https://platform.openai.com/docs/guides/vision). See examples at [test/srt/test_vision_openai_server.py](test/srt/test_vision_openai_server.py)
- LLaVA 1.5 / 1.6 / NeXT
  - `python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --port=30000 --tp-size=1 --chat-template=llava_llama_3`
  - `python -m sglang.launch_server --model-path lmms-lab/llava-next-72b --port=30000 --tp-size=8 --chat-template=chatml-llava`
  - Query the server with the [OpenAI Vision API](https://platform.openai.com/docs/guides/vision). See examples at [test/srt/test_vision_openai_server.py](test/srt/test_vision_openai_server.py)
- Yi-VL
- StableLM
- Command-R
- DBRX
- Grok
- ChatGLM
- InternLM 2
- Exaone 3

**Embedding Models**

- e5-mistral
- gte-Qwen2
  - `python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-7B-instruct --is-embedding`

Instructions for supporting a new model are [here](https://github.com/sgl-project/sglang/blob/main/docs/en/model_support.md).

#### Use Models From ModelScope
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
  
</details>

#### Run Llama 3.1 405B
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

### Benchmark Performance

- Benchmark a single static batch by running the following command without launching a server. The arguments are the same as for `launch_server.py`.
  Note that this is not a dynamic batching server, so it may run out of memory for a batch size that a real server can handle.
  A real server truncates the prefill into several batches, while this unit test does not. For accurate large batch testing, please use `sglang.bench_serving` instead.
  ```
  python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 32 --input-len 256 --output-len 32
  ```
- Benchmark online serving. Launch a server first and run the following command.
  ```
  python3 -m sglang.bench_serving --backend sglang --num-prompt 10
  ```