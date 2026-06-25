# Fix: Async Generator Reader Lock Leak Causing `pause_generation("abort")` to Hang

## Problem

Running `test/registered/rl/test_update_weights_from_tensor.py` hangs indefinitely.
The hang occurs in `TestServerUpdateWeightsFromTensorNonBlocking.test_update_weights`,
which calls the HTTP `pause_generation` endpoint with `mode="abort"`.

## Root Cause

`pause_generation("abort")` polls `model_update_lock.is_locked()` in a loop, waiting
for the RWLock reader count to reach 0. However, the reader count was permanently > 0
because multiple HTTP endpoint handlers leaked reader locks.

The leak pattern: `generate_request()` returns an async generator that acquires the
`model_update_lock` reader lock. Many endpoints consumed only the first result via
`.__anext__()` but never called `.aclose()` on the generator. In Python, unclosed async
generators are not cleaned up until the event loop shuts down, so the reader lock
remained held indefinitely.

### Affected call sites (10 files, ~14 call sites)

| File | Endpoint / Function |
|------|-------------------|
| `http_server.py` | `/generate` (non-streaming), `/encode`, `/classify` |
| `http_server.py` | `/health` / `/health_generate` (task cancellation interrupted cleanup) |
| `warmup.py` | Server warmup request |
| `grpc_bridge.py` | gRPC generate (non-streaming) and embed |
| `openai/serving_chat.py` | Non-streaming chat completions |
| `openai/serving_completions.py` | Non-streaming completions |
| `openai/serving_embedding.py` | Embedding requests |
| `openai/serving_classify.py` | Classification requests |
| `openai/serving_rerank.py` | Reranking requests (2 sites) |
| `openai/serving_transcription.py` | Transcription requests |
| `ollama/serving.py` | Ollama chat and generate |

## Fix

For every `generate_request().__anext__()` call, wrap it with proper cleanup:

```python
# Before (leaks reader lock):
ret = await tokenizer_manager.generate_request(obj, request).__anext__()

# After (properly releases reader lock):
gen = tokenizer_manager.generate_request(obj, request)
try:
    ret = await gen.__anext__()
finally:
    await gen.aclose()
```

For the health check, additionally replaced `task.cancel()` (which could interrupt
the generator's lock-release cleanup mid-flight) with `asyncio.shield(task)` to
ensure the reader lock is always released.

## Verification

All 5 tests in `test_update_weights_from_tensor.py` pass (ran in ~250s):

- `TestUpdateWeightsFromTensor.test_update_weights_from_tensor`
- `TestUpdateWeightsFromTensor.test_update_weights_from_tensor_load_format_direct`
- `TestUpdateWeightsFromTensor.test_update_weights_from_tensor_load_format_custom`
- `TestUpdateWeightsFromTensor.test_update_weights_from_tensor_load_format_flattened_bucket`
- `TestServerUpdateWeightsFromTensorNonBlocking.test_update_weights`

## Debugging Notes

The root cause was identified by adding temporary tracing to `RWLock.acquire_reader()`
and `RWLock.release_reader()` which showed:
- `readers=2` with health generation enabled (one from `/health_generate`, one from warmup)
- `readers=1` with health generation disabled (from warmup only)
- No matching `release_reader` calls — confirming the async generators were never closed
