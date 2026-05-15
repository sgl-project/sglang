# MiniMax M2.5/M2.1/M2 Usage

[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5), [MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1), and [MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) are advanced large language models created by [MiniMax](https://www.minimax.io/).

The MiniMax-M2 series redefines efficiency for agents. These compact, fast, and cost-effective MoE models (230 billion total parameters with 10 billion active parameters) are built for elite performance in coding and agentic tasks, all while maintaining powerful general intelligence. With just 10 billion activated parameters, the MiniMax-M2 series provides sophisticated, end-to-end tool use performance expected from today's leading models, but in a streamlined form factor that makes deployment and scaling easier than ever.

## Supported Models

This guide applies to the following models. You only need to update the model name during deployment. The following examples use **MiniMax-M2**:

- [MiniMaxAI/MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)
- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)

## System Requirements

The following are recommended configurations; actual requirements should be adjusted based on your use case:

- 4x 96GB GPUs: Supported context length of up to 400K tokens.
- 8x 144GB GPUs: Supported context length of up to 3M tokens.

## Deployment with Python

4-GPU deployment command:

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 4 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

8-GPU deployment command:

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2 \
    --tp-size 8 \
    --ep-size 8 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

### AMD GPUs (MI300X/MI325X/MI355X)

8-GPU deployment command:

```bash
SGLANG_USE_AITER=1 python -m sglang.launch_server \
    --model-path MiniMaxAI/MiniMax-M2.5 \
    --tp-size 8 \
    --ep-size 8 \
    --attention-backend aiter \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85
```

## Testing Deployment

After startup, you can test the SGLang OpenAI-compatible API with the following command:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMaxAI/MiniMax-M2",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who won the world series in 2020?"}]}
        ]
    }'
```

## Using MiniMax API with SGLang Frontend

In addition to self-hosted deployment, you can use MiniMax models through the [MiniMax API](https://platform.minimax.io/) with SGLang's frontend language via the built-in `MiniMax` backend.

### Setup

Set the `MINIMAX_API_KEY` environment variable:

```bash
export MINIMAX_API_KEY="your-api-key"
```

### Usage

```python
import sglang as sgl

backend = sgl.MiniMax("MiniMax-M2.5")
sgl.set_default_backend(backend)

@sgl.function
def chat(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer"))

state = chat.run(question="What is MiniMax?")
print(state["answer"])
```

### Available Models

| Model | Description |
|-------|-------------|
| `MiniMax-M2.5` | Default. Peak performance, ultimate value. 204K context window. |
| `MiniMax-M2.5-highspeed` | Same performance, faster and more agile. 204K context window. |

### Configuration

```python
# Custom base URL (e.g., for users in mainland China)
backend = sgl.MiniMax(
    "MiniMax-M2.5",
    api_key="your-api-key",
    base_url="https://api.minimaxi.com/v1",
)
```

For more details on the MiniMax API, see the [MiniMax Platform Documentation](https://platform.minimax.io/docs/api-reference/text-openai-api).
