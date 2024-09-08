# Server Quick Start

This guide will help you quickly set up and run the SGLang server, and send test requests for both generative and embedding models.

## 1. Launch the Server

### For Generative Models

Run the following command to start the SGLang server with a generative model:

```bash
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
--port 30000 --api-key sk-None-hello-world --host 0.0.0.0
```

This command launches a server using the Llama-2-7b-chat-hf model on port 30000 with an authentication key `sk-None-hello-world` and host `0.0.0.0` to enable external access.

### For Embedding Models

To launch an embedding model server:

```bash
python -m sglang.launch_server --model-path Alibaba-NLP/gte-Qwen2-7B-instruct \
--port 30000 --api-key sk-None-hello-world --host 0.0.0.0 --is-embedding
```

This command launches a server using the gte-Qwen2-7B-instruct model on port 30000. Note that `--is-embedding` is required when you are serving an embedding model.

## 2. Send Test Requests

Once the server is running, you can send test requests using curl.

Note that for the `model` arguments, if you do not set `served_model_name` when you launch the server, it should be the same as the `model_path`. Otherwise, it’s the `served_model_name`.

### For Generative Models

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-None-hello-world" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
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
  -H "Authorization: Bearer sk-None-hello-world" \
  -d '{
    "model": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "input": "Once upon a time"
  }'
```

## 3. Using OpenAI Compatible API

SGLang supports OpenAI-compatible APIs. Here's the Python examples:

### Generative Models

```python
import openai

# You should always assign an api_key even if you didn't specify this argument when initializing the server.
# However, we strongly suggest setting an API key when serving.

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="sk-None-hello-world")

# Chat completion example

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)
print(response)
```

#### Embedding Models

```python
import openai

# You should always assign an api_key even if you didn't specify this argument when initializing the server.
# However, we strongly suggest setting an API key when serving.

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key="sk-None-hello-world")

# Text embedding example

response = client.embeddings.create(
    model="Alibaba-NLP/gte-Qwen2-7B-instruct",
    input="How are you today",
)
print(response)
```

For more advanced usage, including multi-GPU setups, custom models, and additional features, please refer to the full documentation in the [SGLang GitHub repository](https://github.com/sgl-project/sglang).