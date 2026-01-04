# SGLang Attention Explorer

A web-based visualization tool for exploring attention patterns in LLM inference.

## Features

- Real-time attention visualization during text generation
- Token-level attention mapping (which tokens attend to which)
- Support for thinking/reasoning models (Qwen3-Next, etc.)
- Interactive token selection with bidirectional attention display

## Usage

1. Start SGLang server with attention token capture enabled:

```bash
python -m sglang.launch_server \
    --model-path your-model \
    --return-attention-tokens \
    --disable-cuda-graph
```

Note: `--disable-cuda-graph` is required for attention capture.

2. Serve the explorer:

```bash
cd examples/attention_explorer
python -m http.server 8081
```

3. Open http://localhost:8081/explorer.html in your browser

## Configuration

Edit the `CONFIG` object in `explorer.html` to change:
- `apiBase`: SGLang server URL (default: `http://localhost:8000`)
- `modelId`: Model name for API calls

## API Integration

The explorer uses the OpenAI-compatible API with these extensions:
- `return_attention_tokens: true` - Enable attention capture
- `top_k_attention: 5` - Number of top attention positions to return

Response includes `attention_tokens` array with:
- `token_positions`: Indices of attended tokens
- `attention_scores`: Attention weights (sum to ~1.0)
- `layer_id`: Which attention layer was captured
