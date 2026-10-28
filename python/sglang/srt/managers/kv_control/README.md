# Programmable KV cache

Native KV control implements [RFC #45189](https://github.com/sgl-project/sglang/issues/45189).
It is disabled by default. Enabling it executes explicit commands; policies run
only when listed in the configuration.

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

Optional local policies can be composed in configuration order:

```json
{
  "namespace": "model-weights-v1",
  "policies": [
    {
      "name": "retain",
      "implementation": "retention",
      "options": {"priority": 50, "ttl_seconds": 30}
    },
    {
      "name": "capacity",
      "implementation": "quota",
      "options": {"borrow_idle_capacity": true}
    },
    {
      "name": "tools",
      "implementation": "tool_wait",
      "options": {"tier": "host", "prefetch_on_admission": true}
    }
  ]
}
```

Retention consumes `importance`/`resume` signals and submits expiring copy
priorities. Quota provides borrowable session shares with bounded admission
waiting; native allocation checks still decide whether a request fits.
Tool-wait coordination considers memory pressure, pause estimates and measured
transfer cost before moving a prefix. A restored prefix retains an internal
reference until native request admission takes ownership. An unconfigured
signal returns `unconsumed`.

## Control API

Use `GET /v1/kv/capabilities` for discovery and `POST /v1/kv/control` for commands.
The Python engine exposes `kv_control(command)` and `async_kv_control(command)`.
An inference request can attach the existing `KvHintsEnvelope` through
`kv_hints`; other consumers' actions remain available to those consumers.

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
| `sglang.signal` | Offer importance, tool-wait, resume or resume-prediction information to configured policies. |

Targets identify token prefixes, logical content descriptors, or tracked session
continuations. The attached-request target `"request_cached"` binds only after
successful completion and cache checkpointing. Cache salt and model/weight
namespace must match the request that produced the KV.

Priorities are soft: active request/transfer references remain protected, and
memory pressure can evict any unreferenced copy. Explicit priorities take
precedence over local policies and session eviction preferences. Expiry, copy
eviction and cache generation changes end a claim; cache hits do not renew it.
There is no public pin operation.

Use `query`, `ack` and `cancel` commands with the returned operation references.
Message replay is idempotent within bounded retention; changed payloads conflict.
Cancellation drains outstanding worker access before releasing resources.
`deferred` is a terminal resource refusal and requires a new message ID to retry.
Task cleanup does not delete shared KV bytes. Weight updates fence inference and
rotate the lookup namespace before parameter mutation; stale contexts are rejected.
