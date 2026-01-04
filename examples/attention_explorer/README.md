# SGLang Attention Explorer

A web-based visualization tool for exploring attention patterns in LLM inference.

## Features

- Real-time attention visualization during text generation
- Token-level attention mapping (which tokens attend to which)
- Support for thinking/reasoning models (Qwen3-Next, etc.)
- Interactive token selection with bidirectional attention display
- True probability calculation for accurate attention weights

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

## Server Configuration

### Required Flags
- `--return-attention-tokens`: Enable attention token capture
- `--disable-cuda-graph`: Required for attention capture (incompatible with CUDA graphs)

### Optional Flags
- `--attention-tokens-top-k N`: Number of top attention positions to return (default: 5)
- `--attention-tokens-max N`: Maximum tokens to record per request, 0=unlimited (default: 4096)
- `--attention-tokens-stride N`: Record every Nth token, 1=all (default: 1)
- `--attention-backend triton`: Use triton backend (required for attention capture)

## Client Configuration

Edit the `CONFIG` object in `explorer.html` to change:
- `apiBase`: SGLang server URL (default: `http://localhost:8000`)
- `modelId`: Model name for API calls

## API Integration

The explorer uses the OpenAI-compatible API with these extensions:
- `return_attention_tokens: true` - Enable attention capture
- `top_k_attention: 5` - Number of top attention positions to return

### Response Schema

Each entry in `attention_tokens` array contains:
- `token_positions`: Indices of attended tokens (list of integers)
- `attention_scores`: Attention weights softmax-normalized over top-k only (list of floats)
- `layer_id`: Which attention layer was captured (integer)
- `topk_logits`: Raw attention logits before softmax (list of floats)
- `logsumexp_all`: Logsumexp normalizer over all positions (float)

### Computing True Probabilities

The `attention_scores` field is softmax-normalized over only the top-k positions, which sums to 1.0 but doesn't represent the true probability mass. For true probabilities:

```javascript
// True probability = exp(logit - logsumexp_all)
const trueProbs = topk_logits.map(logit => Math.exp(logit - logsumexp_all));
// Sum of trueProbs <= 1.0 (accounts for probability mass in non-top-k positions)
```

This is useful for understanding what fraction of attention is captured by the top-k tokens.

## Tensor Parallelism (TP) Behavior

### Current Limitations
- Attention capture requires `--attention-backend triton` which is used by default
- When using TP > 1, attention is captured on TP rank 0 only
- The captured attention represents the full attention pattern averaged across all heads

### Recommended Setup
For best results, use single GPU or TP=1 configuration when visualizing attention:

```bash
python -m sglang.launch_server \
    --model-path your-model \
    --return-attention-tokens \
    --disable-cuda-graph \
    --tp 1
```

### Memory Efficiency
The attention capture uses a memory-efficient chunked algorithm:
- For sequences under ~2K tokens: Direct PyTorch computation
- For longer sequences: Chunked Triton kernel with O(batch x heads x num_chunks) memory

For a 1M token context, this uses ~125KB vs ~256MB for a full attention matrix.
