# Embedding Models

SGLang provides robust support for embedding models by integrating efficient serving mechanisms with its flexible programming interface. This integration allows for streamlined handling of embedding tasks, facilitating faster and more accurate retrieval and semantic search operations. SGLang's architecture enables better resource utilization and reduced latency in embedding model deployment.

```{important}
Embedding models are executed with `--is-embedding` flag and some may require `--trust-remote-code`
```

## Quick Start

### Launch Server

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-4B \
  --is-embedding \
  --host 0.0.0.0 \
  --port 30000
```

### Client Request

```python
import requests

url = "http://127.0.0.1:30000"

payload = {
    "model": "Qwen/Qwen3-Embedding-4B",
    "input": "What is the capital of France?",
    "encoding_format": "float"
}

response = requests.post(url + "/v1/embeddings", json=payload).json()
print("Embedding:", response["data"][0]["embedding"])
```

### Smoke Benchmark

Use `bench_serving` with the `sglang-embedding` backend to benchmark the
OpenAI-compatible `/v1/embeddings` endpoint. The benchmark reports successful
requests, benchmark duration, total input tokens, request throughput, input
token throughput, peak concurrent requests, concurrency, and end-to-end latency
percentiles. Embedding requests do not generate output tokens, so generation
throughput metrics are omitted.

Start an embedding server:

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-0.6B \
  --is-embedding \
  --host 0.0.0.0 \
  --port 30000
```

Run a synthetic random-text smoke benchmark against `/v1/embeddings`. The
`random-ids` dataset avoids downloading ShareGPT data:

```shell
python3 -m sglang.bench_serving \
  --backend sglang-embedding \
  --dataset-name random-ids \
  --model Qwen/Qwen3-Embedding-0.6B \
  --num-prompts 100 \
  --random-input-len 128 \
  --random-output-len 0 \
  --request-rate inf \
  --max-concurrency 16 \
  --host 127.0.0.1 \
  --port 30000
```

You can also use the small usage wrapper to validate or print the same commands
without loading a model:

```shell
python3 -m sglang.bench_embedding_serving --dry-run
```

To run the benchmark through the wrapper against an already running server:

```shell
python3 -m sglang.bench_embedding_serving \
  --run-benchmark \
  --num-prompts 100 \
  --random-input-len 128 \
  --max-concurrency 16
```

To let the wrapper launch and stop the server around the benchmark, add
`--launch-server`. Extra server arguments can be passed after `--`:

```shell
python3 -m sglang.bench_embedding_serving \
  --run-benchmark \
  --launch-server \
  --model-path Qwen/Qwen3-Embedding-0.6B \
  --num-prompts 100 \
  -- --trust-remote-code
```



## Multimodal Embedding Example

For multimodal models like GME that support both text and images:

```shell
python3 -m sglang.launch_server \
  --model-path Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
  --is-embedding \
  --chat-template gme-qwen2-vl \
  --host 0.0.0.0 \
  --port 30000
```

```python
import requests

url = "http://127.0.0.1:30000"

text_input = "Represent this image in embedding space."
image_path = "https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/023.jpg"

payload = {
    "model": "gme-qwen2-vl",
    "input": [
        {
            "text": text_input
        },
        {
            "image": image_path
        }
    ],
}

response = requests.post(url + "/v1/embeddings", json=payload).json()

print("Embeddings:", [x.get("embedding") for x in response.get("data", [])])
```

## Matryoshka Embedding Example

[Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) or [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) is a technique used in training embedding models. It allows user to trade off between performance and cost.

### 1. Launch a Matryoshka‑capable model

If the model config already includes `matryoshka_dimensions` or `is_matryoshka` then no override is needed. Otherwise, you can use `--json-model-override-args` as below:

```shell
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-Embedding-0.6B \
    --is-embedding \
    --host 0.0.0.0 \
    --port 30000 \
    --json-model-override-args '{"matryoshka_dimensions": [128, 256, 512, 1024, 1536]}'
```

1. Setting `"is_matryoshka": true` allows truncating to any dimension. Otherwise, the server will validate that the specified dimension in the request is one of `matryoshka_dimensions`.
2. Omitting `dimensions` in a request returns the full vector.

### 2. Make requests with different output dimensions

```python
import requests

url = "http://127.0.0.1:30000"

# Request a truncated (Matryoshka) embedding by specifying a supported dimension.
payload = {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Explain diffusion models simply.",
    "dimensions": 512  # change to 128 / 1024 / omit for full size
}

response = requests.post(url + "/v1/embeddings", json=payload).json()
print("Embedding:", response["data"][0]["embedding"])
```


## Supported Models

| Model Family                               | Example Model                          | Chat Template | Description                                                                 |
| ------------------------------------------ | -------------------------------------- | ------------- | --------------------------------------------------------------------------- |
| **E5 (Llama/Mistral based)**              | `intfloat/e5-mistral-7b-instruct`     | N/A           | High-quality text embeddings based on Mistral/Llama architectures          |
| **GTE-Qwen2**                             | `Alibaba-NLP/gte-Qwen2-7B-instruct`   | N/A           | Alibaba's text embedding model with multilingual support                   |
| **Qwen3-Embedding**                       | `Qwen/Qwen3-Embedding-4B`             | N/A           | Latest Qwen3-based text embedding model for semantic representation        |
| **BGE**                                    | `BAAI/bge-large-en-v1.5`              | N/A           | BAAI's text embeddings (requires `attention-backend` triton/torch_native)  |
| **GME (Multimodal)**                      | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`| `gme-qwen2-vl`| Multimodal embedding for text and image cross-modal tasks                  |
| **CLIP**                                   | `openai/clip-vit-large-patch14-336`   | N/A           | OpenAI's CLIP for image and text embeddings                                |
