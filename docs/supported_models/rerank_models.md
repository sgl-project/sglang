# Rerank Models

SGLang offers comprehensive support for rerank models by incorporating optimized serving frameworks with a flexible programming interface. This setup enables efficient processing of cross-encoder reranking tasks, improving the accuracy and relevance of search result ordering. SGLang’s design ensures high throughput and low latency during reranker model deployment, making it ideal for semantic-based result refinement in large-scale retrieval systems.

```{important}
They are executed with `--is-embedding` and some may require `--trust-remote-code`
```

## Example Launch Command

```shell
python3 -m sglang.launch_server \
  --model-path BAAI/bge-reranker-v2-m3 \
  --host 0.0.0.0 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --attention-backend triton \
  --is-embedding \
  --port 30000
```

## Example Client Request

```python
import requests

url = "http://127.0.0.1:30000/v1/rerank"

payload = {
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "what is panda?",
    "documents": [
        "hi",
        "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
    ]
}

response = requests.post(url, json=payload)
response_json = response.json()

for item in response_json:
    print(f"Score: {item['score']:.2f} - Document: '{item['document']}'")
```

## Supported rerank models

| Model Family (Rerank)                          | Example HuggingFace Identifier       | Chat Template | Description                                                                                                                      |
|------------------------------------------------|--------------------------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------|
| **BGE-Reranker (BgeRerankModel)**              | `BAAI/bge-reranker-v2-m3`            | N/A           | Currently only support `attention-backend`   `triton` and `torch_native`.  high-performance cross-encoder reranker model from BAAI. Suitable for reranking search results based on semantic relevance.   |
