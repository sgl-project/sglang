# Attention Token Capture - Local Test Documentation

**Date**: 2026-01-04
**Branch**: feature/attention-token-visualization

## Summary

This document captures the testing and fixes performed for the attention token capture feature in SGLang.

## Features Tested

### 1. Attention Token Capture API
- Captures top-k attention scores during inference
- Returns token positions and attention weights for each decode step
- Multi-layer capture support with configurable layer selection

### 2. Server Arguments
```bash
--return-attention-tokens        # Enable attention token capture
--attention-capture-layers auto  # Automatic layer selection (L/4, L/2, 3L/4, L-1)
--attention-tokens-top-k 10      # Number of top attention tokens to capture
--attention-tokens-max 4096      # Maximum tokens to capture
--attention-tokens-stride 1      # Capture every Nth step
```

### 3. API Request
```json
{
  "model": "...",
  "messages": [...],
  "return_attention_tokens": true
}
```

## Fixes Applied

### Fix 1: CUDA Graph Compatibility
**Files modified:**
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`

**Issue:** Attention capture requires dynamic computation that cannot be replayed from CUDA graphs. Requests with `capture_attention_tokens=True` were being processed through CUDA graph replay, which skipped the actual attention capture logic.

**Solution:** Added check in `can_run()` method to skip CUDA graphs when attention capture is enabled:
```python
# Attention capture requires dynamic computation that cannot be replayed
is_attention_capture_disabled = not getattr(
    forward_batch, "capture_attention_tokens", False
)
```

### Fix 2: Sidecar Feedback Subscriber
**File:** `python/sglang/srt/managers/scheduler_output_processor_mixin.py`

**Issue:** Scheduler was using `bind()` instead of `connect()` for feedback channel, causing "Address already in use" errors.

**Solution:** Changed `bind()` to `connect()` - sidecar binds, scheduler connects:
```python
self._feedback_subscriber.connect(feedback_url)  # Connect to sidecar's PUSH socket
```

### Fix 3: Fingerprint Publisher Initialization
**File:** `python/sglang/srt/managers/scheduler_output_processor_mixin.py`

**Issue:** Premature `hasattr` check prevented fingerprint publisher from initializing.

**Solution:** Removed the `hasattr` check - `_stream_fingerprint_to_sidecar()` handles lazy init internally.

### Fix 4: ZMQ Socket Shutdown Hangs
**File:** `python/sglang/srt/managers/scheduler_output_processor_mixin.py`

**Issue:** ZMQ sockets without `LINGER=0` could hang during server shutdown.

**Solution:** Added `LINGER=0` to both fingerprint publisher and feedback subscriber:
```python
self._fingerprint_publisher.setsockopt(zmq.LINGER, 0)  # Don't hang on shutdown
self._feedback_subscriber.setsockopt(zmq.LINGER, 0)    # Don't hang on shutdown
```

### Fix 5: Attention Bias Scaling Guard
**File:** `python/sglang/srt/layers/attention/triton_backend.py`

**Issue:** PyTorch fallback for attention bias has nested loops that don't scale to 1M context.

**Solution:** Added max sequence length guard (32K) with warning:
```python
_ATTENTION_BIAS_MAX_SEQ_LEN = 32768

if max_seq_len > _ATTENTION_BIAS_MAX_SEQ_LEN:
    logger.warning(
        f"Attention bias requested but seq_len={max_seq_len} exceeds "
        f"max={_ATTENTION_BIAS_MAX_SEQ_LEN}. Skipping bias (PyTorch "
        f"fallback too slow). Use router biases for long-context steering."
    )
    use_bias_fallback = False
```

### Fix 6: FlashInfer tile_tokens_dim
**File:** `python/sglang/srt/layers/quantization/fp8.py`

**Issue:** `tile_tokens_dim=None` parameter was being passed to `trtllm_fp8_block_scale_moe` which doesn't accept it.

**Solution:** Removed the unsupported parameter.

## Test Results

### Successful Tests (Qwen2.5-1.5B-Instruct)

**Server command:**
```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --tp 1 \
    --context-length 4096 \
    --port 30000 \
    --return-attention-tokens \
    --attention-capture-layers auto \
    --attention-backend triton
```

**Test request:**
```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
    "max_tokens": 50,
    "return_attention_tokens": true
  }'
```

**Response includes:**
```json
{
  "attention_tokens": [
    {
      "layers": {
        "7": {
          "token_positions": [0, 37, 38, ...],
          "attention_scores": [0.5078125, 0.1455, ...],
          "topk_logits": [9.875, 8.625, ...],
          "logsumexp_candidates": 10.625,
          "topk_mass": 0.9296875
        },
        "14": {...},
        "21": {...},
        "27": {...}
      },
      "decode_step": 1,
      "think_phase": "output"
    },
    ...
  ]
}
```

### Successful Test: Qwen3-Next-80B-A3B-Thinking-FP8

**Server command:**
```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_ENABLE_JIT_DEEPGEMM=0 \
python3 -m sglang.launch_server \
    --model Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
    --attention-backend triton \
    --moe-runner-backend triton \
    --context-length 4096 \
    --port 30000 \
    --kv-cache-dtype bfloat16 \
    --return-attention-tokens \
    --attention-capture-layers auto \
    --tp 1
```

**Result:** Successfully captured decode steps with layer 47 attention scores.

**Note on hybrid attention:** Qwen3-Next uses hybrid attention:
- Full attention layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47
- Linear attention layers: all others (0-2, 4-6, 8-10, etc.)
- "auto" mode calculates [12, 24, 36, 47] but only layer 47 has full attention
- To capture more layers, explicitly specify: `--attention-capture-layers 3,7,11,15,19,23,27,31,35,39,43,47`

## Sidecar Integration Test

**Date**: 2026-01-04

### Setup

**Start sidecar first:**
```bash
python3 -u scripts/mock_attention_sidecar.py --fingerprint-port 9001 --feedback-port 9002
```

**Start server with sidecar:**
```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_ENABLE_JIT_DEEPGEMM=0 \
python3 -m sglang.launch_server \
    --model Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
    --attention-backend triton \
    --moe-runner-backend triton \
    --context-length 4096 \
    --port 30000 \
    --kv-cache-dtype bfloat16 \
    --return-attention-tokens \
    --attention-capture-layers auto \
    --attention-sidecar-url tcp://localhost:9001 \
    --attention-fingerprint-mode \
    --tp 1
```

### Results (2026-01-04 22:46)

Server logs show successful connections:
```
Connected fingerprint publisher to tcp://localhost:9001
Connected to sidecar feedback on tcp://localhost:9002
```

Sidecar receives fingerprints and sends feedback:
```
[Sidecar] Listening for fingerprints on port 9001
[Sidecar] Feedback channel ready on port 9002
[Sidecar] Ready. Press Ctrl+C to stop.

[Fingerprint] rid=98db2ac6... manifold=semantic_bridge vector_len=20
[Fingerprint] rid=98db2ac6... manifold=semantic_bridge vector_len=20
[Fingerprint] rid=98db2ac6... manifold=semantic_bridge vector_len=20
[Feedback] -> rid=98db2ac6... zone=exploration
[Fingerprint] rid=98db2ac6... manifold=semantic_bridge vector_len=20
...
[Feedback] -> rid=98db2ac6... zone=steering
```

**Test Summary:**
| Component | Status | Details |
|-----------|--------|---------|
| Fingerprint publisher | ✅ | Connected to tcp://localhost:9001 |
| Feedback subscriber | ✅ | Connected to tcp://localhost:9002 |
| Fingerprint streaming | ✅ | 30 fingerprints sent, 20-dim vectors |
| Feedback loop | ✅ | 2 steering commands received |
| ZMQ LINGER=0 | ✅ | Server shutdown clean |

Fingerprint mode returns semantic fingerprints (20-dim vectors) instead of raw attention scores:
```json
{
  "fingerprint": [0.503, 0.497, 0.0, 0.534, 0.093, ...],
  "manifold": "semantic_bridge",
  "step": 1,
  "think_phase": "output"
}
```

**Note:** In fingerprint mode (`--attention-fingerprint-mode`), attention data is streamed to the sidecar and not returned in the API response.

## Known Issues

1. **FlashInfer SM100 (Blackwell)**: Some flashinfer MoE kernels may not work on Blackwell.
   **Workaround:** Use `--moe-runner-backend triton` and `--attention-backend triton`

2. **Hybrid Linear Attention**: Models using `HybridLinearAttnBackend` (like Qwen3-Next) route full attention layers to Triton backend. Attention capture works for these full attention layers.

3. **Attention Bias at Long Context**: PyTorch fallback for attention bias doesn't scale beyond 32K tokens. Use router biases for long-context steering instead.

## Architecture Notes

### Attention Capture Flow
1. Request with `return_attention_tokens=true` → `GenerateReqInput`
2. `GenerateReqInput` → `TokenizedGenerateReqInput` (tokenizer manager)
3. `TokenizedGenerateReqInput` → `Req` (scheduler)
4. `Req` → `ScheduleBatch` → `ModelWorkerBatch`
5. `ModelWorkerBatch` → `ForwardBatch` (sets `capture_attention_tokens=True`)
6. `ForwardBatch` → Triton backend captures attention scores
7. Captured scores returned in `LogitsProcessorOutput`
8. Output processor stores scores in `req.attention_tokens`

### Layer Selection ("auto" mode)
For a model with L layers, "auto" selects:
- L/4 (early layer)
- L/2 (middle layer)
- 3L/4 (late layer)
- L-1 (last layer)

Example for 28-layer Qwen2.5-1.5B: layers 7, 14, 21, 27

### Sidecar Communication
- **Fingerprint channel**: PUSH (server) → PULL (sidecar) on port 9001
- **Feedback channel**: PUSH (sidecar) → PULL (server) on port 9002
- Both use ZMQ with `LINGER=0` for clean shutdown

## Commits

1. `28e7ecb18` - [interpretability] Add top-k attention token capture for visualization
2. `e6560ea9f` - [interpretability] Fix CUDA graph and sidecar compatibility for attention capture
3. `5c9bbcec1` - [interpretability] Add safeguards for attention bias and ZMQ sockets

## Files Changed

### Core Implementation
- `python/sglang/srt/layers/attention/triton_backend.py` - Attention score capture
- `python/sglang/srt/model_executor/forward_batch_info.py` - ForwardBatch configuration
- `python/sglang/srt/managers/schedule_batch.py` - Batch processing
- `python/sglang/srt/managers/scheduler_output_processor_mixin.py` - Output processing, sidecar integration

### CUDA Graph Fixes
- `python/sglang/srt/model_executor/cuda_graph_runner.py`
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`

### API Layer
- `python/sglang/srt/entrypoints/openai/protocol.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/entrypoints/openai/serving_completions.py`
- `python/sglang/srt/managers/io_struct.py`

### Tests
- `test/srt/test_attention_tokens.py`
- `test/srt/test_topk_attention.py`
- `test/srt/test_attention_moe_integration.py`
