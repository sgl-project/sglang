# Server Quick Start

This guide will help you quickly set up and run the SGLang server, and send test requests for both generative and embedding models.

## 1. Launch the Server

### For Generative Models

Run the following command to start the SGLang server with a generative model:

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000
```

This command launches a server using the Meta-Llama-3-8B-Instruct model on port 30000.

### For Embedding Models

To launch an embedding model server:

```bash
python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-7B-instruct --is-embedding --port 30000
```

This command launches a server using the gte-Qwen2-7B-instruct model on port 30000.

## 2. Send Test Requests

Once the server is running, you can send test requests using curl.

### For Generative Models

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is a LLM?"
      }
    ]
  }'
```

### For Embedding Models

```bash
curl http://localhost:30000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Once upon a time"
  }'
```

## 3. Using OpenAI Compatible API

SGLang supports OpenAI-compatible APIs. Here's a Python example:

```python
import openai

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
# You should assign an api_key even if you didn't specify this argument when initializing the server.
# However, we strongly suggest setting an API key when serving.

# Chat completion example
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

For more advanced usage, including multi-GPU setups, custom models, and additional features, please refer to the full documentation in the [SGLang GitHub repository](https://github.com/sgl-project/sglang).