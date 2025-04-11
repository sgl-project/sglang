# Embedding Models

SGLang provides robust support for embedding models by integrating efficient serving mechanisms with its flexible programming interface. This integration allows for streamlined handling of embedding tasks, facilitating faster and more accurate retrieval and semantic search operations. SGLang's architecture enables better resource utilization and reduced latency in embedding model deployment.

```{important}
They are executed with `--is-embedding` and some may require `--trust-remote-code` and/or `--chat-template`
```

## Example launch Command

```shell
python3 -m sglang.launch_server \
  --model-path Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \  # example HF/local path
  --is-embedding \
  --host 0.0.0.0 \
  --chat-template gme-qwen2-vl \                     # set chat template
  --port 30000 \
```

## Supporting Matrixs

| Model Family (Embedding)                        | Example HuggingFace Identifier                | Chat Template | Description                                                                                                                          |
|-------------------------------------------------|-----------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------|
| **Llama/Mistral based (E5EmbeddingModel)**      | `intfloat/e5-mistral-7b-instruct`             | N/A           | Mistral/Llama-based embedding model fine‑tuned for high‑quality text embeddings (top‑ranked on the MTEB benchmark).                   |
| **GTE (QwenEmbeddingModel)**                    | `Alibaba-NLP/gte-Qwen2-7B-instruct`           | N/A           | Alibaba’s general text embedding model (7B), achieving state‑of‑the‑art multilingual performance in English and Chinese.             |
| **GME (MultimodalEmbedModel)**                  | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`        | `gme-qwen2-vl`          | Multimodal embedding model (2B) based on Qwen2‑VL, encoding image + text into a unified vector space for cross‑modal retrieval.      |
| **CLIP (CLIPEmbeddingModel)**                   | `openai/clip-vit-large-patch14-336`           | N/A           | OpenAI’s CLIP model (ViT‑L/14) for embedding images (and text) into a joint latent space; widely used for image similarity search.   |
