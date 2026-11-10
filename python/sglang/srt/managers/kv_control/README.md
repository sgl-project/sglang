# Programmable KV cache

Native KV control is disabled by default. It lets a caller use three versioned
actions:

| Action | Effect |
| --- | --- |
| `sglang.pin` | Acquire, renew, or release finite eviction protection for existing host KV. |
| `sglang.prefetch` | Prepare a supported destination while keeping the source. |
| `sglang.demote` | Preserve data at a lower tier, then reclaim the eligible source range. |

The [Router orchestrator](../../../../../sgl-router/KV_ORCHESTRATION.md) uses
these actions for application task and tool events. The executing Engine owns
the allocation, transfer, and reference checks. Query actual target coverage
and protection state before treating a receipt as proof of readiness.

## Configuration

```bash
python -m sglang.launch_server --model-path MODEL_PATH \
  --enable-hierarchical-cache --disable-overlap-schedule \
  --hicache-storage-backend mooncake \
  --kv-control-config '{"namespace":"model-weights-v1","engine_id":"worker-a"}'
```

Configure the Mooncake backend separately. The file backend is also useful for
local tests. Transfers and PIN require a built-in FULL-attention cache with
persistent host HiCache. The serving profile has one TP/PP/CP/DP owner and
ordinary text requests. Discovery reports which actions and directions the
current backend can execute.

## Control API

Use `GET /v1/kv/capabilities` for discovery and `POST /v1/kv/control` for commands.
The Python Engine exposes `kv_control(command)` and `async_kv_control(command)`.
Use the Engine ID, epoch and namespace returned by discovery in every command:

```json
{
  "kind": "submit",
  "context": {
    "engine_id": "worker-a",
    "engine_epoch": "epoch-from-discovery",
    "namespace": "model-weights-v1",
    "caller_id": "trusted-controller"
  },
  "envelope": {
    "protocol_version": "0.1",
    "message_id": "keep-prefix-1",
    "actions": [{
      "action_id": "keep",
      "action_type": "sglang.pin",
      "action_version": "1.0",
      "payload": {
        "target": {"kind": "tokens", "namespace": "model-weights-v1", "token_ids": [1, 2, 3, 4]},
        "location": {"tier": "host"},
        "mode": "acquire",
        "claim_id": "prefix-1",
        "revision": 1,
        "ttl_seconds": 30,
        "required_protection": "eviction_exclusion"
      }
    }]
  }
}
```

Use a target that matches the worker's page size, layout and cache salt. A PIN
requires an already published host copy; it does not copy missing data. An
attached PIN targeting `"request_cached"` binds only after successful request
completion and cache checkpointing.

Use `query` and `claim` to inspect the result. `cancel` drains worker access before
releasing resources. Acknowledge with `ack` once the operation is quiescent. An
identical message can be replayed within the retention window; a changed payload
with the same identity is rejected. Capacity refusal preserves the inference
request's normal recovery path. DEMOTE can confirm preservation and report
that a source range is blocked by a live GPU copy, reference, or PIN.
