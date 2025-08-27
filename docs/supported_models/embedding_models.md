<<<<<<< HEAD
# Embedding Models

SGLang provides robust support for embedding models by integrating efficient serving mechanisms with its flexible programming interface. This integration allows for streamlined handling of embedding tasks, facilitating faster and more accurate retrieval and semantic search operations. SGLang's architecture enables better resource utilization and reduced latency in embedding model deployment.

```{important}
They are executed with `--is-embedding` and some may require `--trust-remote-code`
```

## Example Launch Command

```shell
python3 -m sglang.launch_server \
  --model-path Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
  --is-embedding \
  --host 0.0.0.0 \
  --chat-template gme-qwen2-vl \
  --port 30000
```
## Example Client Request
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


## Supported models

| Model Family (Embedding)                        | Example HuggingFace Identifier                | Chat Template | Description                                                                                                                          |
|-------------------------------------------------|-----------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------|
| **Llama/Mistral based (E5EmbeddingModel)**      | `intfloat/e5-mistral-7b-instruct`             | N/A           | Mistral/Llama-based embedding model fine‑tuned for high‑quality text embeddings (top‑ranked on the MTEB benchmark).                   |
| **GTE (QwenEmbeddingModel)**                    | `Alibaba-NLP/gte-Qwen2-7B-instruct`           | N/A           | Alibaba’s general text embedding model (7B), achieving state‑of‑the‑art multilingual performance in English and Chinese.             |
| **GME (MultimodalEmbedModel)**                  | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`        | `gme-qwen2-vl`          | Multimodal embedding model (2B) based on Qwen2‑VL, encoding image + text into a unified vector space for cross‑modal retrieval.      |
| **CLIP (CLIPEmbeddingModel)**                   | `openai/clip-vit-large-patch14-336`           | N/A           | OpenAI’s CLIP model (ViT‑L/14) for embedding images (and text) into a joint latent space; widely used for image similarity search.   |
| **BGE (BgeEmbeddingModel)**                     | `BAAI/bge-large-en-v1.5`                        | N/A                 | Currently only support `attention-backend`   `triton` and `torch_native`. BAAI's BGE embedding models optimized for retrieval and reranking tasks. |
=======
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

## Supported Models

| Model Family                               | Example Model                          | Chat Template | Description                                                                 |
| ------------------------------------------ | -------------------------------------- | ------------- | --------------------------------------------------------------------------- |
| **E5 (Llama/Mistral based)**              | `intfloat/e5-mistral-7b-instruct`     | N/A           | High-quality text embeddings based on Mistral/Llama architectures          |
| **GTE-Qwen2**                             | `Alibaba-NLP/gte-Qwen2-7B-instruct`   | N/A           | Alibaba's text embedding model with multilingual support                   |
| **Qwen3-Embedding**                       | `Qwen/Qwen3-Embedding-4B`             | N/A           | Latest Qwen3-based text embedding model for semantic representation        |
| **BGE**                                    | `BAAI/bge-large-en-v1.5`              | N/A           | BAAI's text embeddings (requires `attention-backend` triton/torch_native)  |
| **GME (Multimodal)**                      | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`| `gme-qwen2-vl`| Multimodal embedding for text and image cross-modal tasks                  |
| **CLIP**                                   | `openai/clip-vit-large-patch14-336`   | N/A           | OpenAI's CLIP for image and text embeddings                                |
>>>>>>> origin/main
