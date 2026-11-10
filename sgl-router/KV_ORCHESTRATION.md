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
| `sglang.pin` | Acquire, renew, or release finite eviction protection for a complete host prefix. |
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
2. On `tool_wait`, supply the intended source worker, `pressure: true` and
   `expected_wait_ms` to request host-to-storage DEMOTE. If the application
   wants to protect a hot prefix in L2 instead, it can explicitly submit PIN.
3. Send `resume_expected` with the expected wait and desired destination worker
   for storage-to-host PREFETCH. `tool_done` can advance pending preparation
   for the matching scope. Send inference normally if the prefix is not ready.
4. Send `task_finished` or `task_cancelled` to release the task's interests.
   An ordinary inference completion does not end the task.

Use a new monotonic event revision and an event ID for each lifecycle transition.
`query`, `cancel`, and `ack` use the same task and message ID as the original
intent. Cancel detaches only that intent; shared physical work can continue for
other interests. Release a PIN before acknowledging its intent and wait for
quiescence before discarding an operation's receipt.

Routing can use recently confirmed host coverage as an additional cache signal.
The existing routing policy still decides the inference worker. Discovery,
Engine epoch, and scoped cache feedback fence that signal; a missing worker or
an observation gap invalidates it. A control receipt by itself does not prove
the subsequent inference request reused the prefix.

See the [Engine control README](../python/sglang/srt/managers/kv_control/README.md)
for configuration and command envelopes.
