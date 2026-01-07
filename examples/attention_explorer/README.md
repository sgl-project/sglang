# SGLang Attention Explorer

A comprehensive interpretability dashboard for exploring attention patterns, behavioral zones, and routing decisions in LLM inference.

## Features

- **Real-time attention visualization** during text generation
- **Token Lens** with hover/pin interactions and sink token handling
- **Fingerprint-based zone classification** (syntax, semantic, long-range, etc.)
- **Manifold discovery** with persistent clustering and UMAP projection
- **Actionable Router** with sampling recommendations
- **Cross-run comparison** to diff attention patterns between sessions
- **Think segmentation** for reasoning models with collapsible sections
- **Export/Import** traces as JSONL for replay and analysis
- **Privacy controls** with system prompt masking and guardrails

## Quick Start with React UI

### 1. Start SGLang Server

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --return-attention-tokens \
  --attention-tokens-top-k 32
```

### 2. Start the React UI

```bash
cd examples/attention_explorer/ui
npm install
npm run dev
```

Open http://localhost:3001 in your browser.

### 3. Explore Attention Patterns

- **Chat View**: Interactive chat with real-time attention streaming
- **Inspect View**: Layer-by-layer attention breakdown
- **Manifold View**: 2D projection of behavioral clusters
- **Router View**: Zone detection with sampling recommendations
- **Compare View**: Side-by-side session comparison

## UI Views

### Chat View
The primary interface for interactive exploration:
- Real-time token streaming with attention overlays
- Click tokens to pin them in the Token Lens drawer
- Hover to see attention links
- Think section collapsing for reasoning models
- Segment timeline showing think vs output phases

### Token Lens Drawer
Slides in from right on token hover/click:
- **Links Tab**: Top-k attended positions with scores
- **Signal Tab**: Fingerprint metrics, distance histogram, zone classification
- **MoE Tab**: Expert routing information (for MoE models)

### Inspect View
Detailed attention analysis:
- Layer selector for multi-layer inspection
- Top-k attended positions with scores
- Segment filtering (think vs output)
- Virtualized token list for long sequences

### Manifold View
Behavioral clustering and discovery:
- 2D UMAP projection of fingerprint space
- Cluster visualization with zone labels
- Scope selector (session/recent/saved/all)
- Click points to load saved sessions
- Session details panel with metrics

### Router View
Dynamic zone detection and recommendations:
- Real-time zone classification with confidence
- Evidence meters showing classification signals
- Recommended settings (temperature, top_p, capture mode)
- "Apply to Next Run" persistence via localStorage

### Compare View
Cross-session analysis:
- Select two saved traces for side-by-side comparison
- Overall similarity score (cosine + zone match + entropy)
- Key differences in human-readable format
- Metrics diff with colored bars
- Distance histogram overlay
- PCA interpretation of fingerprint differences

## Legacy HTML Explorer

For quick debugging without npm:

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
- `--attention-sketch-mode`: Return per-layer summary sketches instead of raw edges (bandwidth efficient for long outputs)
- `--attention-fingerprint-mode`: Production mode - compute 20D feature vector on GPU (64 bytes vs 200KB/step)
- `--attention-fingerprint-max-steps N`: Early exit for fingerprint mode after N decode steps (default: 256)
- `--attention-sidecar-url URL`: ZMQ URL for streaming fingerprints to clustering sidecar
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

### Sketch Mode

For very long outputs (86k+ tokens), sketch mode provides bandwidth-efficient summaries:

```javascript
// Enable sketch mode via API
extra_body: {
  return_attention_tokens: true,
  attention_sketch_mode: true  // Or use server flag --attention-sketch-mode
}
```

Sketch mode response format (per decode step):
```javascript
{
  "mode": "sketch",
  "layer_sketches": {
    "31": {  // Layer ID
      "top_hubs": [[42, 0.15], [128, 0.12], ...],  // Top 32 positions by attention mass
      "dist_hist": [0.05, 0.12, 0.23, ...],        // 16-bin log-distance histogram
      "entropy": 3.45,                              // Attention entropy estimate
      "mass_captured": 0.82                         // Fraction of total attention in top-k
    }
  },
  "decode_step": 100
}
```

Distance histogram bins:
- Bin 0: Distance 0-1 (immediate context)
- Bin 1: Distance 2-3
- Bin 2: Distance 4-7
- ...
- Bin 15: Distance 32768+ (very long range)

This is ~500 bytes per layer vs ~200KB for raw edges, enabling efficient streaming for long generations.

### Fingerprint Mode (Production)

For production use at 100+ req/s, fingerprint mode compresses attention patterns into a tiny **20D feature vector** computed entirely on GPU, eliminating the CPU export bottleneck.

```bash
python -m sglang.launch_server \
    --model-path your-model \
    --return-attention-tokens \
    --attention-fingerprint-mode \
    --attention-fingerprint-max-steps 256 \
    --disable-cuda-graph
```

#### Why Fingerprint Mode?

| Mode | Output Size | CPU Sync | Use Case |
|------|-------------|----------|----------|
| **Raw (Debug)** | ~200KB/step | Yes | Visualization, debugging |
| **Sketch** | ~500B/layer | Yes | Long output analysis |
| **Fingerprint** | 80 bytes | **No** | Production routing, 100+ req/s |

Fingerprint mode streams a 20D vector instead of raw indices, enabling real-time manifold discovery without impacting throughput.

#### Response Format

```javascript
{
  "fingerprint": [0.36, 0.64, 0.0, 0.40, ...],  // 20D feature vector
  "manifold": "semantic_bridge",                 // Classified pattern
  "step": 42,
  "think_phase": "output"
}
```

The 20D fingerprint contains:
- `[0]` local_mass: Attention to immediate context (bins 0-2, offset < 8)
- `[1]` mid_mass: Mid-range attention (bins 3-7, offset 8-255)
- `[2]` long_mass: Long-range retrieval (bins 8+, offset 256+)
- `[3]` entropy: Attention concentration (0=focused, 1=diffuse)
- `[4:20]` histogram: 16-bin log-distance distribution

#### Manifold Zones

The `manifold` field classifies attention patterns into interpretable zones:

| Manifold Zone | Signal | Typical Layer | Pattern |
|--------------|--------|---------------|---------|
| `syntax_floor` | High local_mass (>0.6) | Early layers | Grammar, BPE, local syntax |
| `semantic_bridge` | High mid_mass (>0.4) | Middle layers | Reasoning, entity tracking |
| `long_range` | High long_mass (>0.3) | Late layers | Retrieval, instruction following |
| `diffuse` | High entropy (>0.7) | Any | No clear pattern |

#### Early Exit Optimization

The attention manifold typically stabilizes within the first few hundred tokens. Use `--attention-fingerprint-max-steps` to stop capture early and save ~90% of compute:

```bash
--attention-fingerprint-max-steps 256  # Default: stop after 256 decode steps
--attention-fingerprint-max-steps 0    # Unlimited (capture all steps)
```

This is critical for long generations where you only need the manifold signature, not per-token tracking.

### Think Segmentation (Reasoning Models)

For reasoning models (Qwen3, DeepSeek-R1, etc.) that use `<think>...</think>` tags, attention data is automatically segmented by phase:

```bash
# Enable reasoning parser for think segmentation
python -m sglang.launch_server \
    --model-path your-model \
    --return-attention-tokens \
    --disable-cuda-graph \
    --reasoning-parser qwen3  # or deepseek-r1, etc.
```

Each attention entry includes a `think_phase` field:
- `"think"`: Token generated inside `<think>...</think>` block
- `"output"`: Token generated after `</think>` (the actual response)

Example response:
```javascript
{
  "token_positions": [42, 128, 256],
  "attention_scores": [0.15, 0.12, 0.08],
  "decode_step": 100,
  "think_phase": "think"  // or "output"
}
```

This enables:
- Separate analysis of thinking vs output attention patterns
- Filtering attention data by phase for visualization
- Comparing reasoning strategies across different outputs

Supported reasoning parsers: `qwen3`, `qwen3-thinking`, `deepseek-r1`, `deepseek-v3`, `kimi`, `glm45`, `step3`, `minimax`

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

## Clustering Sidecar

For production use, the attention fingerprints can be streamed to a clustering sidecar for manifold discovery and sampling optimization.

### Architecture

```
SGLang Scheduler  --fingerprints-->  RAPIDS Sidecar  --centroids-->  Proxy Router
     (ZMQ PUSH)        (ZMQ PULL)           |
     |<----------- manifold hints ----------|
```

### Starting the Sidecar

```bash
# Install dependencies
pip install pyzmq hdbscan scikit-learn
# OR for GPU acceleration: pip install cuml-cu12

# Start sidecar with ZMQ receiver
cd examples/attention_explorer
python rapids_sidecar.py --zmq-bind tcp://*:9001 --port 9000

# Start SGLang with fingerprint streaming
python -m sglang.launch_server \
    --model-path your-model \
    --return-attention-tokens \
    --attention-fingerprint-mode \
    --attention-sidecar-url tcp://localhost:9001 \
    --disable-cuda-graph
```

### Clustering Modes

**Batch Mode (default)**: Periodically re-clusters buffer using HDBSCAN
```bash
python rapids_sidecar.py --zmq-bind tcp://*:9001 --recluster-interval 60
```

**Online Mode**: Real-time micro-cluster updates (no batching)
```bash
python rapids_sidecar.py --zmq-bind tcp://*:9001 --online --online-threshold 2.0
```

### Sidecar API

Query centroids and statistics via HTTP:
```bash
# Get current cluster centroids
curl http://localhost:9000/centroids

# Get sidecar stats
curl http://localhost:9000/stats

# Predict cluster for a fingerprint
curl -X POST http://localhost:9000/predict -d '{"vector": [0.7, 0.5, ...]}'
```

### Cluster Traits

The sidecar interprets fingerprint patterns into semantic traits:
- `syntax_floor` / `local_attention`: High attention to immediate context
- `semantic_bridge` / `retrieval_heavy`: Mid-range attention (reasoning)
- `long_range` / `context_aware`: Long-distance attention
- `focused` / `diffuse`: Attention concentration
- `periodic`: Regular attention patterns (code, repetition)

These traits inform sampling hints (temperature, top_p) for optimized generation.

## Zone Classification

Attention patterns are classified into behavioral zones based on fingerprint metrics:

| Zone | Signal | Description |
|------|--------|-------------|
| `syntax_floor` | local_mass > 0.5 | Local syntax/grammar processing (immediate neighbors) |
| `semantic_bridge` | mid_mass > 0.5 | Paragraph-level semantic retrieval |
| `long_range` | long_mass > 0.3 | Document-level dependencies |
| `structure_ripple` | fft_low > 0.6 | Periodic patterns (code, lists, structured text) |
| `diffuse` | high entropy | Exploratory/uncertain attention |

Distance bands for mass calculation:
- **Local**: 0-32 tokens (syntax/immediate context)
- **Mid**: 33-256 tokens (semantic/paragraph level)
- **Long**: 257+ tokens (document-level/cross-context)

## PCA Interpretation

The 20D fingerprint vector is interpreted via pre-computed PCA loadings:

| PC | Name | Variance | Interpretation |
|----|------|----------|----------------|
| PC1 | Local vs Long-Range | 35% | Positive = local syntax, Negative = long-range retrieval |
| PC2 | Focused vs Diffuse | 25% | Positive = high confidence, Negative = exploratory |
| PC3 | Semantic Bridge | 15% | Positive = paragraph-level reasoning |
| PC4 | Structure Ripple | 10% | Positive = periodic patterns (code, lists) |

The UI provides natural language explanations based on PCA projections:
- "Strong local syntax processing (immediate neighbors)"
- "Moderate semantic context retrieval (paragraph-level reasoning)"
- "Weak long-range information retrieval"

## Capabilities Endpoint

Query server capabilities before making requests:

```bash
curl http://localhost:30000/v1/attention/capabilities
```

Response:
```json
{
  "supported": true,
  "modes": ["raw", "sketch", "fingerprint"],
  "max_top_k": 64,
  "max_layers": 32,
  "features": {
    "include_prompt_attention": true,
    "head_filtering": true,
    "moe_routing": true,
    "logit_lens": false
  },
  "guardrails": {
    "max_concurrent_capture": 10,
    "requires_api_key": false,
    "system_prompt_masked": false
  }
}
```

## Privacy & Guardrails

### System Prompt Masking

Prevent system prompt structure leakage via attention patterns:

```bash
python -m sglang.launch_server \
  --model-path your-model \
  --return-attention-tokens \
  --attention-mask-system-prompt
```

Or per-request:
```python
extra_body={
    "return_attention_tokens": True,
    "attention_mask_prefix": 100  # Mask first 100 tokens
}
```

### Multi-Tenant Guardrails

For shared deployments:

| Argument | Description |
|----------|-------------|
| `--attention-api-key KEY` | Require API key for attention endpoints |
| `--attention-cors-origins` | Allowed CORS origins (default: `["*"]`) |
| `--attention-max-concurrent-capture N` | Limit concurrent capture sessions |
| `--attention-disable-in-production` | Auto-disable if not in debug mode |

### Head Filtering

Capture attention from specific heads only:

```python
extra_body={
    "return_attention_tokens": True,
    "attention_capture_head_ids": [0, 1, 2, 3]  # Only these heads
}
```

Useful for mechanistic interpretability to isolate specific head behaviors.

## Export/Import Traces

### Export

Click "Export" in the UI top bar to download current session as JSONL:

```jsonl
{"record_type":"header","version":"1.0","trace_id":"trace-123","model":"Qwen2.5-7B"}
{"record_type":"message","id":"user-1","role":"user","content":"Hello"}
{"record_type":"token","index":0,"text":"Hi","segmentId":"assistant-1-main"}
{"record_type":"step","tokenIndex":0,"attention":{...},"fingerprint":{...}}
{"record_type":"segment","id":"assistant-1-think","type":"assistant_think",...}
{"record_type":"metrics","avgEntropy":0.45,"dominantZone":"semantic_bridge",...}
```

### Import

Click "Import" to load a previously exported trace for replay, comparison, or analysis.

## Discovery Pipeline

For building persistent manifold visualizations over time:

### 1. Database Schema

The discovery pipeline uses SQLite with tables for:
- `fingerprints`: 20D vectors with request metadata
- `sessions`: Chat session summaries
- `discovery_runs`: PCA/UMAP/clustering job metadata
- `clusters`: Cluster centroids and zone assignments

### 2. Run Discovery Job

```bash
python examples/attention_explorer/discovery/run_discovery.py \
  --db ./attention_data.db \
  --output-dir ./discovery_artifacts \
  --min-samples 100
```

Pipeline stages:
1. PCA dimensionality reduction (20D → intermediate)
2. UMAP projection to 2D
3. HDBSCAN clustering
4. Zone assignment based on cluster centroids
5. Artifact export (Parquet + model files)

### 3. Sidecar Integration

```bash
python -m sglang.srt.attention_sidecar \
  --db ./attention_data.db \
  --discovery-dir ./discovery_artifacts \
  --auto-reload-interval 300
```

Endpoints:
- `POST /classify`: Classify a fingerprint against artifacts
- `GET /discovery/status`: Current artifact status
- `POST /discovery/reload`: Reload artifacts from disk
- `POST /storage/flush`: Flush buffered fingerprints to SQLite

## Overhead Estimates

| Mode | Per-Step Size | Notes |
|------|---------------|-------|
| Raw (k=16) | ~200 bytes | Scales with top_k |
| Raw (k=64) | ~800 bytes | |
| Sketch (1 layer) | ~500 bytes | Scales with layers |
| Sketch (32 layers) | ~16 KB | |
| Fingerprint | ~64 bytes | Fixed size |

The Overhead HUD in the UI shows real-time metrics:
- Token count and tokens/second
- Attention data bytes received
- Current capture mode

Control overhead with:
- `--attention-chunk-size`: Trade memory vs latency (default: 2048)
- `--attention-capture-stride`: Skip steps (e.g., 2 = every other step)
- `--attention-fingerprint-max-steps`: Early exit for fingerprint mode

## Troubleshooting

### Attention data not showing
1. Verify server started with `--return-attention-tokens`
2. Check capabilities: `curl http://localhost:30000/v1/attention/capabilities`
3. Ensure `return_attention_tokens: true` in request
4. Check browser console for SSE parsing errors

### UI shows "Disconnected"
1. Check server is running on expected port
2. Verify CORS settings if UI on different origin
3. Check browser console for WebSocket/SSE errors

### High memory usage
1. Reduce `--attention-tokens-top-k`
2. Increase `--attention-capture-stride`
3. Use fingerprint mode instead of raw
4. Enable `--attention-fingerprint-max-steps`

### Manifold view empty
1. Save some traces first (manifold needs data)
2. If using sidecar, ensure discovery artifacts exist
3. Check sidecar is running with correct `--discovery-dir`

### Compare view has no sessions
1. Save current trace before comparing
2. Complete at least one generation (need attention data)
3. Check that sessions have fingerprint data

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SGLang Server                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Model Runner│──│ Attention   │──│ SSE Stream          │ │
│  │             │  │ Capture     │  │ (attention_tokens)  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Attention Explorer UI                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐    │
│  │ Chat   │ │Inspect │ │Manifold│ │ Router │ │Compare │    │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘    │
│                         │                                   │
│                   TraceSession                              │
│              (canonical data model)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (optional)
┌─────────────────────────────────────────────────────────────┐
│                     Sidecar Service                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ SQLite      │  │ Discovery   │  │ Classification      │ │
│  │ Storage     │  │ Artifacts   │  │ Endpoints           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## License

Apache 2.0 - Same as SGLang
