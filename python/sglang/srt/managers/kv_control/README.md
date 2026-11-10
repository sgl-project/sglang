# Programmable KV cache

Native KV control implements [RFC #45189](https://github.com/sgl-project/sglang/issues/45189).
It is disabled by default. Enabling it allows explicit cache control commands.

Native storage reads accept optional `cache_limits` settings:
`max_storage_read_inflight_tokens`, `max_speculative_storage_read_tokens`,
`demand_host_reserve_tokens`, and `storage_demand_priority`. Token budgets default
to zero (disabled) and must be page aligned; priority defaults to false. The
total-read gate counts ordinary read occupancy before admitting native work;
ordinary inference retains its existing recovery path. The speculative budget
counts native in-flight physical spans, not completed cache residency. A real
demand joining that job changes its classification without duplicating I/O.
Demand preference allows one background turn after eight demand admissions and
preserves GPU FIFO and same-salt serialization. The host reserve is an admission
floor, not a new allocation. A scope larger than its enabled budget is deferred.
Read cost and capacity observations are available through the existing capacity
query and `loads.memory.storage.prefetch_stats`; byte counters describe anchor
KV API payload, and wire bytes remain unknown.

## Configuration

```bash
python -m sglang.launch_server --model-path MODEL_PATH \
  --kv-control-config '{"namespace":"model-weights-v1","engine_id":"worker-a"}'
```

The executable profile requires one TP/PP/CP/DP owner, the built-in unified
cache, and ordinary text generation. Priority control supports FULL, SWA and
Mamba components through the Python tree backend. Copy and move require FULL
attention, persistent-host HiCache, and `--disable-overlap-schedule`. Enable
HiCache with `--enable-hierarchical-cache`; file-backed transfers additionally
require `--hicache-storage-backend file`.

Remote engines, buffer-only/direct-linker transfers, speculative decoding, PD
disaggregation, LoRA, multimodal and streaming-session control are unsupported.
Capability discovery reports the active profile; an unsupported operation does
not imply successful publication or cache reuse.

## Control API

Use `GET /v1/kv/capabilities` for discovery and `POST /v1/kv/control` for commands.
The Python engine exposes `kv_control(command)` and `async_kv_control(command)`.
An inference request can attach the existing `KvHintsEnvelope` through
`kv_hints`; other consumers' actions remain available to those consumers.

Capability discovery advertises `read_batch` version `1.0` with `max_queries: 8`.
Its `commands` list contains `capabilities`, `query`, `residency`, `claim`, and
`events`. Submit `{"kind":"read_batch","context":CONTEXT,"version":"1.0",
"queries":[QUERY,...]}` to execute 1–8 typed read queries in order during one
owner turn. Every child carrying a context must exactly match the batch context,
including caller, binding and owner epoch. Writes and nested batches are rejected
before execution. A `capabilities` child has no context; the other four require it.
The reply data is `{"version":"1.0","results":[DATA,...]}`,
using each command's existing data format. Any child error returns the usual
error for the whole batch; preceding reads may already have progressed state.
This is not a global atomic snapshot: `events` can advance owner state, and
clients must retain the existing watermark and invalidation checks. Existing
transport response byte limits apply to the entire reply. Clients must negotiate
this capability; older Engines reject `read_batch` and require single queries.

Control commands use `engine_id`, `engine_epoch` and `namespace` from discovery,
a caller-supplied `caller_id`, and a versioned envelope:

```json
{
  "kind": "submit",
  "context": {
    "engine_id": "worker-a",
    "engine_epoch": "EPOCH_FROM_DISCOVERY",
    "namespace": "model-weights-v1",
    "caller_id": "router"
  },
  "envelope": {
    "protocol_version": "0.1",
    "message_id": "retain-prefix-1",
    "actions": [{
      "action_id": "keep",
      "action_type": "sglang.set_priority",
      "action_version": "1.0",
      "payload": {
        "target": {"kind": "tokens", "namespace": "model-weights-v1", "token_ids": [1, 2, 3, 4]},
        "location": {"tier": "gpu"},
        "claim_id": "prefix-1",
        "revision": 1,
        "value": 90,
        "ttl_seconds": 30
      }
    }]
  }
}
```

| Action | Behavior |
| --- | --- |
| `sglang.set_priority` | Apply or clear a signed, expiring priority on existing copies. |
| `sglang.copy_to` | Publish a destination copy while keeping the source. |
| `sglang.move_to` | Publish the destination, then reclaim eligible source state. |
| `sglang.associate` | Associate an append-only continuation with a logical task. |
| `sglang.finish_task` | Remove task associations and abandon its pending interests. |
| `sglang.signal` | Reports an unconsumed Engine signal; use Router lifecycle events to request actions. |

Targets identify token prefixes, logical content descriptors, or tracked session
continuations. The attached-request target `"request_cached"` binds only after
successful completion and cache checkpointing. Cache salt and model/weight
namespace must match the request that produced the KV.

Priorities are soft: active request/transfer references remain protected, and
memory pressure can evict any unreferenced copy. Explicit priorities take
precedence over session eviction preferences. Expiry, copy
eviction and cache generation changes end a claim; cache hits do not renew it.
There is no public pin operation.

Use `query`, `ack` and `cancel` commands with the returned operation references.
Message replay is idempotent within bounded retention; changed payloads conflict.
Cancellation drains outstanding worker access before releasing resources.
`deferred` is a terminal resource refusal and requires a new message ID to retry.
Task cleanup does not delete shared KV bytes. Weight updates fence inference and
rotate the lookup namespace before parameter mutation; stale contexts are rejected.
