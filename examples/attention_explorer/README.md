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
- `--attention-tokens-window N`: Context window for capture, 0=all tokens (default: 0). For very long contexts (1M+), set this to limit which tokens are considered (e.g., 8192 for last 8K tokens only)
- `--attention-capture-layers MODE`: Which layers to capture. Options:
  - `last` (default): Only the last layer
  - `auto`: Automatically select ~4 layers spread across depth [L/4, L/2, 3L/4, L-1]
  - Comma-separated indices: e.g., `0,10,20,30` for specific layers
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
- `logsumexp_candidates`: Exact logsumexp normalizer over ALL attention scores (float)
- `decode_step`: Which decode step this entry corresponds to (integer)

### Computing True Probabilities

The `attention_scores` field is softmax-normalized over only the top-k positions, which sums to 1.0 but doesn't represent the true probability mass. For **true probabilities**:

```javascript
// True probability = exp(logit - logsumexp_candidates)
// logsumexp_candidates is computed over ALL tokens (exact, not approximate)
const trueProbs = topk_logits.map(logit => Math.exp(logit - logsumexp_candidates));
// Sum of trueProbs represents the fraction of attention captured by top-k
// (typically 70-95% of total attention mass)
```

This is useful for:
- Understanding what fraction of attention is captured by the top-k tokens
- Computing stable influence rankings (true_prob is stable across different k values)
- Hub detection (tokens with high cumulative true probability across decode steps)

## Tensor Parallelism (TP) Behavior

### How TP Affects Attention Capture

When using tensor parallelism (TP > 1), attention heads are distributed across GPUs:
- **TP rank 0 only**: Attention is captured only from the heads on TP rank 0
- **Partial view**: With TP=2, you see ~50% of heads; with TP=4, ~25% of heads
- **Head distribution**: Different heads may focus on different aspects (syntax vs semantics)

This means TP > 1 gives you a partial, potentially biased view of the model's attention patterns. The captured attention represents only the heads assigned to rank 0, not an average across all heads.

### Recommended Setup

For interpretability work, use single GPU or TP=1 to see all attention heads:

```bash
python -m sglang.launch_server \
    --model-path your-model \
    --return-attention-tokens \
    --disable-cuda-graph \
    --tp 1
```

### When TP > 1 Is Acceptable

- **Quick exploration**: Rank 0 heads often provide useful signal
- **Large models**: When TP=1 is not possible due to memory constraints
- **Comparative analysis**: Comparing attention patterns across prompts (relative differences are still informative)

Note: Results with TP > 1 should be interpreted with caution for detailed interpretability work.

### Memory Efficiency
The attention capture uses a memory-efficient chunked algorithm:
- For sequences under ~2K tokens: Direct PyTorch computation
- For longer sequences: Chunked Triton kernel with O(batch x heads x num_chunks) memory

For a 1M token context, this uses ~125KB vs ~256MB for a full attention matrix.
