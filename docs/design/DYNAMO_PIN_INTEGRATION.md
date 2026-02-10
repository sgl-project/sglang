# Dynamo PIN Integration Design

## Overview

This document describes the integration between SGLang's HiCache PIN block pinning
and Dynamo's router/worker architecture, enabling router-initiated cache control
across a distributed inference deployment.

## Architecture

```
External Caller (agent framework, orchestrator)
    |
    |  Dynamo service mesh (TCP/NATS)
    v
+------------------+
|  Standalone       |  Endpoints:
|  KV Router        |  - generate (existing)
|                   |  - best_worker_id (existing)
|  + pin_prefix     |  - cache_control (NEW)
|  + unpin_prefix   |  - pin_prefix (NEW)
|  + cache_control  |  - unpin_prefix (NEW)
+--------+---------+
         |
         |  client.direct(command, worker_id)
         |  via worker "cache_control" endpoint
         v
+------------------+     +------------------+
| Worker 0         |     | Worker 1         |
|                  |     |                  |
| SGLang Engine    |     | SGLang Engine    |
| + HiRadixCache   |     | + HiRadixCache   |
| + pin_blocks()   |     | + pin_blocks()   |
| + unpin_blocks() |     | + unpin_blocks() |
|                  |     |                  |
| Endpoints:       |     | Endpoints:       |
| - generate       |     | - generate       |
| - cache_control  |     | - cache_control  |
+------------------+     +------------------+
```

## Components Modified

### Worker Side (Dynamo SGLang component)

**handler_base.py** - Added `cache_control` async generator to `BaseWorkerHandler`:
- Dispatches to existing `pin_blocks`, `unpin_blocks`, `evict_descendants` methods
- Served as Dynamo service mesh endpoint (not just engine route)
- Callable by router via `client.direct(command, worker_id)`

**main.py** - Added `cache_control` endpoint serving:
- Runs alongside `generate` endpoint in `asyncio.gather()`
- Available in both `init()` (decode/aggregated) and `init_prefill()`

### Router Side (Standalone KV Router)

**router/__main__.py** - Added three new capabilities:

1. **`cache_control` endpoint** - Forward cache control commands to workers
   - Accepts `worker_id` for targeted commands
   - Broadcasts to all workers when `worker_id` is omitted

2. **`pin_prefix` endpoint** - High-level token-based pinning
   - Takes `token_ids`, computes block hashes via `compute_block_hash_for_seq_py`
   - Uses KV router to find best-matching worker
   - Pins blocks on that worker

3. **`unpin_prefix` endpoint** - High-level token-based unpinning
   - Same flow as `pin_prefix` but unpins

## Communication Paths

### Path 1: Direct worker cache control (via engine routes)

For single-worker testing or direct worker targeting:

```
curl -X POST http://worker:DYN_SYSTEM_PORT/engine/pin_blocks \
  -H "Content-Type: application/json" \
  -d '{"block_hashes": [123, 456]}'
```

### Path 2: Router-mediated cache control (via service mesh)

For multi-worker deployments, the router forwards commands:

```python
# Via Dynamo service mesh client
router_endpoint = runtime.namespace("dynamo").component("router").endpoint("cache_control")
client = await router_endpoint.client()

# Pin on specific worker
async for result in await client.random({
    "action": "pin_blocks",
    "block_hashes": [123, 456],
    "worker_id": 42,
}):
    print(result)
```

### Path 3: Token-based pin_prefix (via service mesh)

Highest-level API -- caller provides tokens, router handles everything:

```python
router_endpoint = runtime.namespace("dynamo").component("router").endpoint("pin_prefix")
client = await router_endpoint.client()

# Pin a system prompt by token IDs
async for result in await client.random({
    "token_ids": system_prompt_tokens,
}):
    print(result)
    # result includes: worker_id, dp_rank, overlap_blocks, pinned_count
```

## Request/Response Formats

### cache_control

Request:
```json
{
    "action": "pin_blocks",
    "block_hashes": [15310707395893867146, 15292316782987903195],
    "worker_id": 42
}
```

Response:
```json
{
    "status": "ok",
    "pinned_count": 2,
    "message": "Pinned 2/2 blocks",
    "worker_id": 42
}
```

### pin_prefix

Request:
```json
{
    "token_ids": [1, 2, 3, ..., 1024]
}
```

Response:
```json
{
    "status": "ok",
    "pinned_count": 8,
    "message": "Pinned 8/8 blocks",
    "worker_id": 42,
    "dp_rank": 0,
    "overlap_blocks": 8,
    "total_blocks": 8
}
```

## Prerequisites

- SGLang workers must have `--enable-hierarchical-cache` for PIN to work
- KV events should be enabled (`--enable-kv-cache-events`) for router prefix matching
- Workers must be started with the Dynamo SGLang component (not standalone SGLang)

## Block Hash Compatibility

Block hashes are computed consistently between:
- **Dynamo router**: `compute_block_hash_for_seq_py(tokens, block_size)` (Rust)
- **SGLang KV events**: Block hashes in `KvCacheEvent.Stored` events
- **SGLang HiRadixCache**: `block_hash_index` keyed by int64 hashes

The hashes are unsigned 64-bit integers. The KV event flow ensures the router's
indexer has the same hashes as the worker's `block_hash_index`.

## Next Steps

1. **Frontend HTTP endpoints**: Add `/cache/pin_blocks`, `/cache/pin_prefix` to the
   Rust frontend HTTP service (following `clear_kv_blocks.rs` pattern)
2. **Automatic system prompt pinning**: Router detects repeated prefixes from KV events
   and auto-pins them above a frequency threshold
3. **EVICT support**: Wire up `evict_descendants` for cache pruning
4. **Metrics**: Track pin/unpin operations, pinned block counts, load-back latency
