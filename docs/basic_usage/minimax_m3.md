# MiniMax M3 Usage

[MiniMax-M3](https://huggingface.co/MiniMaxAI/Minimax-M3-preview) is a sparse
MoE model that ships with MXFP8 weights.

## Deployment with Python

8-GPU deployment command:

```bash
python -m sglang.launch_server \
    --model-path MiniMaxAI/Minimax-M3-preview \
    --tp-size 8 \
    --chunked-prefill-size 8192 \
    --tool-call-parser minimax-m3 \
    --reasoning-parser minimax-m3 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.80
```

On AMD gfx950 systems, SGLang auto-detects the MXFP8 checkpoint, selects the
Triton MiniMax-M3 MoE path, uses the packaged tuned MXFP8 MoE configs, and
enables AITER fused all-reduce for single-node tensor parallel deployments.

## Testing Deployment

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMaxAI/Minimax-M3-preview",
        "messages": [
            {
                "role": "user",
                "content": "Solve: If a train travels 45 miles in 30 minutes, what is its speed in miles per hour?"
            }
        ],
        "max_tokens": 256,
        "temperature": 0
    }'
```
