# Rerank Models

SGLang offers comprehensive support for rerank models by incorporating optimized serving frameworks with a flexible programming interface. This setup enables efficient processing of cross-encoder reranking tasks, improving the accuracy and relevance of search result ordering. SGLang’s design ensures high throughput and low latency during reranker model deployment, making it ideal for semantic-based result refinement in large-scale retrieval systems.

```{important}
Rerank models in SGLang fall into two categories:

- **Cross-encoder rerank models**: run with `--is-embedding` (embedding runner).
- **Decoder-only rerank models (e.g. Qwen3-Reranker)**: run **without** `--is-embedding` and use next-token logprob scoring (yes/no).

Some models may require `--trust-remote-code`.
```

## Cross-Encoder Rerank (embedding runner)

### Launch Command

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
    ],
    "top_n": 1,
    "return_documents": True
}

response = requests.post(url, json=payload)
response_json = response.json()

for item in response_json:
    if item.get("document"):
        print(f"Score: {item['score']:.2f} - Document: '{item['document']}'")
    else:
        print(f"Score: {item['score']:.2f} - Index: {item['index']}")
```

**Request Parameters:**

- `query` (required): The query text to rank documents against
- `documents` (required): List of documents to be ranked
- `model` (required): Model to use for reranking
- `top_n` (optional): Maximum number of documents to return. Defaults to returning all documents. If specified value is greater than the total number of documents, all documents will be returned.
- `return_documents` (optional): Whether to return documents in the response. Defaults to `false`. Only included when set to `true`.

## Qwen3-Reranker (decoder-only yes/no rerank)

### Launch Command

```shell
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Reranker-0.6B \
  --served-model-name Qwen3-Reranker-0.6B \
  --trust-remote-code \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 8001 \
  --chat-template examples/chat_template/qwen3_reranker.jinja
```

```{note}
Qwen3-Reranker uses decoder-only logprob scoring (yes/no). Do NOT launch it with `--is-embedding`.
```

### Client Request (supports optional instruct, top_n, and return_documents)

```shell
curl -X POST http://127.0.0.1:8001/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Reranker-0.6B",
    "query": "法国首都是哪里？",
    "documents": [
      "法国的首都是巴黎。",
      "德国的首都是柏林。",
      "香蕉是黄色的水果。"
    ],
    "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
    "top_n": 2,
    "return_documents": true
  }'
```

**Request Parameters:**

- `query` (required): The query text to rank documents against
- `documents` (required): List of documents to be ranked
- `model` (required): Model to use for reranking
- `instruct` (optional): Instruction text for the reranker
- `top_n` (optional): Maximum number of documents to return. Defaults to returning all documents. If specified value is greater than the total number of documents, all documents will be returned.
- `return_documents` (optional): Whether to return documents in the response. Defaults to `false`. Only included when set to `true`.

### Response Format

`/v1/rerank` returns a list of objects (sorted by descending score):

- `score`: float, higher means more relevant
- `document`: the original document string (only included when `return_documents` is `true`)
- `index`: the original index in the input `documents`
- `meta_info`: optional debug/usage info (may be present for some models)

The number of returned results is controlled by the `top_n` parameter. If `top_n` is not specified or is greater than the total number of documents, all documents are returned.

Example (with `return_documents: true`):

```json
[
  {"score": 0.99, "document": "法国的首都是巴黎。", "index": 0},
  {"score": 0.01, "document": "德国的首都是柏林。", "index": 1},
  {"score": 0.00, "document": "香蕉是黄色的水果。", "index": 2}
]
```

Example (with `return_documents: false`):

```json
[
  {"score": 0.99, "index": 0},
  {"score": 0.01, "index": 1},
  {"score": 0.00, "index": 2}
]
```

Example (with `top_n: 2`):

```json
[
  {"score": 0.99, "document": "法国的首都是巴黎。", "index": 0},
  {"score": 0.01, "document": "德国的首都是柏林。", "index": 1}
]
```

### Common Pitfalls

- If you launch Qwen3-Reranker with `--is-embedding`, `/v1/rerank` cannot compute yes/no logprob scores. Relaunch **without** `--is-embedding`.
- If you see a validation error like “score should be a valid number” and the backend returned a list, upgrade to a version that coerces `embedding[0]` into `score` for rerank responses.

## Supported rerank models

| Model Family (Rerank)                          | Example HuggingFace Identifier       | Chat Template | Description                                                                                                                      |
|------------------------------------------------|--------------------------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------|
| **BGE-Reranker (BgeRerankModel)**              | `BAAI/bge-reranker-v2-m3`            | N/A           | Currently only support `attention-backend`   `triton` and `torch_native`.  high-performance cross-encoder reranker model from BAAI. Suitable for reranking search results based on semantic relevance.   |
| **Qwen3-Reranker (decoder-only yes/no)**       | `Qwen/Qwen3-Reranker-8B`             | `examples/chat_template/qwen3_reranker.jinja` | Decoder-only reranker using next-token logprob scoring for labels (yes/no). Launch **without** `--is-embedding`. |
