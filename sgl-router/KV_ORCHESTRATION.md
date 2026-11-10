# Session KV cache orchestration

The optional Router orchestrator lets an application prepare KV for a later
request in the same task or branch. It uses the existing worker registry and
cache index. Engines execute commands through HiCache and remain responsible for
capacity, residency, and source reclamation. Normal inference remains available
when control is disabled, a worker is unavailable, or preparation is incomplete.

Enable Engine KV control with `--kv-control-config` and hierarchical caching.
Use a shared Mooncake namespace for storage transfers. Configure the Router with
`--kv-orchestration-config` and a protected copy of
[kv-orchestration.example.json](kv-orchestration.example.json). Supply its
`control_bearer` as a Bearer token on control routes. Use the same model
namespace, layout, page size, and compatible storage configuration on source and
destination Engines.

| Action | Initial use |
| --- | --- |
| `sglang.pin` | Set or release a TTL and/or eviction priority for a complete host prefix. |
| `sglang.prefetch` | Prepare a storage prefix in host memory. The Engine reports the actually available range. |
| `sglang.demote` | Preserve a host prefix in storage and reclaim the eligible host range. The result separates these outcomes. |

Applications can submit these actions directly with `POST /v1/kv/control`. A
command carries a task identity, an idempotent message ID, and a target. Targets
can select a session, branch, observed request, or compatible prefix. Session
and branch selectors freeze their current scope when the command is accepted.
Use Engine capabilities to check which transfer directions the worker supports.

For example, after registering a task and observing its branch, pin its host
prefix on the worker that owns it:

```json
{
  "kind": "submit",
  "task": {"session_id": "session-1", "incarnation": "task-1"},
  "message_id": "keep-branch",
  "action": "sglang.pin",
  "target": {"kind": "branch", "branch_id": "main"},
  "worker_url": "http://worker-a:30000",
  "location": {"tier": "host"},
  "mode": "acquire",
  "claim_id": "keep",
  "revision": 1,
  "ttl_seconds": 3,
  "required_protection": "eviction_exclusion"
}
```

Send lifecycle events to `POST /v1/kv/session/events`:

1. Send `task_started` and observe the actual rendered input via
   `request_observed` or the inference request's `kv_context`. This establishes
   logical association; it does not assert physical residency.
2. On `tool_wait`, supply the intended source worker and `pressure: true`
   to request host-to-storage DEMOTE. Use PIN to protect a hot prefix in L2.
3. Send `prepare` or `resume` when the application wants storage-to-host
   PREFETCH to start. The event queues work without a predicted wait or timer.
   Send inference at its normal time even if the prefix is still being read.
4. Send `task_finished` or `task_cancelled` to release the task's interests.
   An ordinary inference completion does not end the task.

Use a new monotonic event revision and an event ID for each lifecycle transition.
`query`, `cancel`, and `ack` use the same task and message ID as the original
intent. Cancel detaches only that intent; shared physical work can continue for
other interests. Release a PIN before acknowledging its intent and wait for
quiescence before discarding an operation's receipt.

For request-triggered prefetch, send `kv_context` on a generation request along
with `x-kv-control-authorization: Bearer <control_bearer>`. The Router consumes
this header locally; inference can use a separate `Authorization` credential.
The existing routing policy picks a worker. The Router then offers the
page-aligned reusable input prefix to that worker's control queue and forwards
inference without waiting for preparation. The hint can share a compatible
in-flight HiCache restore. If tokens or queue capacity are unavailable, the
request takes its normal recovery path. A control receipt by itself does not
prove subsequent inference reused the prefix.

See the [Engine control README](../python/sglang/srt/managers/kv_control/README.md)
for configuration and command envelopes.
