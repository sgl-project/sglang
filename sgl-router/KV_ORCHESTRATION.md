# Optional KV orchestration

Enable with `--kv-orchestration-config FILE`. The JSON example provides explicit
resource limits and an initial transfer-cost estimate; it is a local experiment
configuration. Omission leaves existing inference/routing behavior unchanged.
The configured caller owns this Router's task/control domain. Lifecycle and
control endpoints require `Authorization: Bearer <control_bearer>`; caller IDs
in request bodies cannot select another caller. Worker HTTP credentials continue
to use `--worker-api-key`.

The checked build used task-local Rust 1.99.1; the installed system Rust 1.75 cannot build the locked upstream dependencies. Replace the local toolchain/cache paths below when using another environment.

Start against the local Qwen Engine pair (legacy or `--chat-routing reorg`):

```sh
export PATH=/home/youwei/.cache/sglang-rfc-tools/cargo/bin:$PATH
export CARGO_HOME=/home/youwei/.cache/sglang-rfc-tools/cargo
export RUSTUP_HOME=/home/youwei/.cache/sglang-rfc-tools/rustup
export CARGO_TARGET_DIR=/home/youwei/.cache/sglang-rfc-tools/target-router
export CARGO_HTTP_CAINFO=/etc/ssl/certs/ca-certificates.crt
export CARGO_BUILD_JOBS=8
cargo build --locked --manifest-path sgl-router/Cargo.toml -p sgl-router
"$CARGO_TARGET_DIR/debug/sgl-router" \
  --host 127.0.0.1 --port 12910 \
  --model-id /home/youwei/bzh/model/Qwen/Qwen3-8B \
  --tokenizer-path /home/youwei/bzh/model/Qwen/Qwen3-8B/tokenizer.json \
  --worker-urls http://127.0.0.1:51801 http://127.0.0.1:51803 \
  --policy cache_aware --cache-prefix-provider radix_tree \
  --kv-orchestration-config sgl-router/kv-orchestration.example.json
```

Use this Python sequence with `requests`. It creates a trial-specific raw salt,
warms an actual request, binds its rendered/input tokens to a Router branch,
demotes through A's last routed owner, and schedules preparation on B. It never
waits on preparation inside a formal inference call. The separate demonstration
query loop is an experiment harness; production resumption can arrive at any time.
The request-observed signal reports an input already sent to A, without another
inference or a placement assumption. A `worker_url` names only a
preparation/control owner; normal Router policy still makes the formal pick.

```python
import json, time, uuid, requests
from transformers import AutoTokenizer
base = "http://127.0.0.1:12910"
a, b = "http://127.0.0.1:51801", "http://127.0.0.1:51803"
headers = {"Authorization": "Bearer local-p6-control"}
task = {"session_id": "trial-" + uuid.uuid4().hex, "incarnation": "task-1"}
salt = "raw-" + uuid.uuid4().hex
ids = AutoTokenizer.from_pretrained("/home/youwei/bzh/model/Qwen/Qwen3-8B").encode(
    "A repeated input for the preparation experiment. " * 300)[:2048]
ids = ids[:len(ids) // 64 * 64]
def ctl(body):
    response = requests.post(base + "/v1/kv/control", json=body, headers=headers, timeout=60)
    response.raise_for_status()
    result = response.json()
    assert result["success"], result
    return result["data"]
def event(name, revision, event_id, **fields):
    return ctl(dict(kind="session_event", event=name, task=task,
                    revision=revision, event_id=event_id, **fields))
event("task_started", 0, "start")
# Warm A without Router-only metadata; this provides the source KV for DEMOTE.
requests.post(a + "/generate", json={"input_ids": ids, "cache_salt": salt,
    "sampling_params": {"max_new_tokens": 1, "temperature": 0}}, timeout=60).raise_for_status()
# Report the exact actual input already rendered/sent to A. This creates only
# a logical association; physical residency remains unknown until a scope query.
event("request_observed", 1, "observed-r1", branch_id="main", branch_revision=1,
      request_id="r1", request_incarnation="binding-1", input_ids=ids,
      cache_salt=salt, worker_url=a)
kv = dict(**task, branch_id="main", revision=1, branch_revision=1,
          request_id="r1", request_incarnation="binding-1")
# Demote on A explicitly; the background control does not change inference affinity/load.
event("tool_wait", 2, "wait", target={"kind": "session"}, worker_url=a,
      pressure=True, expected_wait_ms=3000)
for _ in range(100):
    view = ctl(dict(kind="query", task=task, message_id="wait"))
    if view["quiescent"]: break
    time.sleep(.05)
# Source storage backend/namespace and destination Engine epoch come from live caps.
resume = event("resume_expected", 3, "resume", target={"kind": "session"},
               worker_url=b, expected_wait_ms=3000)
print("resume decision", json.dumps(resume))
# In an experiment, query independently while the application/tool does its work.
view = resume
if resume.get("state") != "on_demand_fallback":
    for _ in range(100):
        view = ctl(dict(kind="query", task=task, message_id="resume"))
        if any(op["ready_scope"] for op in view["operations"]): break
        time.sleep(.05)
print("queried preparation", json.dumps(view))
kv.update(revision=4, branch_revision=2, request_id="r2", request_incarnation="binding-2")
formal = requests.post(base + "/generate", json={"input_ids": ids, "cache_salt": salt,
    "kv_context": kv, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}, timeout=60)
formal.raise_for_status()
print("formal selected worker", formal.headers.get("x-sgl-worker-url"))
print(event("task_finished", 5, "finish"))
```

If the resume response is `on_demand_fallback`, no preparation intent was admitted;
skip its query loop and send formal inference normally. `tool_done` uses the same
frozen target and admits preparation only when the estimated cost fits
`available_ms` (or configured `max_resume_prefetch_ms`). `resume_expected` schedules
at expected wait minus estimated transfer cost and margin. Full-scope, uncontended
ready observations update an EMA; formal dispatches suppress cost learning for
that interval. The estimate is a policy input, not a reservation or latency promise.

Controls use `kind=submit`, `task`, `message_id`, `action`, `target`, and action
fields. `target` may be `session`, `branch` (`branch_id`, optional `revision`),
`request` (`request_id`, `request_incarnation`), or a P1 `prefix`. Session selectors
freeze the current branch set/revision. Responses list known/missing/material gaps
and admitted child operations; `allow_partial` defaults false. `worker_url` is
optional and must be a current healthy registered Engine. Non-PIN transfers use
`source` and `destination`. PIN uses `location`, `mode`, `claim_id`, `revision`,
`ttl_seconds` and `required_protection`. Caller/task/prefix derive Engine-local
claim identities; returned Engine IDs remain queryable in results. No `owner_task`
is sent, so independent controls require no Engine-local task index.

Use `kind=query|cancel|ack`, the same task and message ID, for an intent. Exact
replays keep child IDs; conflicting payloads fail. New `attempt` integers permit
explicit retry after a terminal/expired exact job. Cancellation releases only that
interest; the physical child is retained until Engine quiescence. `ack` releases transfer interest and only forgets quiescent work. Cancel or
explicitly release a PIN intent before acknowledging; acknowledgement cannot
implicitly release an active PIN. Ordinary request completion leaves the task
alive. Explicit `request_finished` can drop a request binding. `task_finished` and
`task_cancelled` default `cascade=false`; explicit cascade cleans descendants.
`task_started` accepts an optional parent task. Application signal revisions share
the session's monotone revision domain; exact event retries have bounded receipts.

Rendered chat and `/generate input_ids` share the HTTP hook, including legacy and
reorg routing. `kv_context` is stripped before Engine deserialization. Legacy
session-only labels lack a branch/incarnation and stay unknown. LoRA/multimodal or
unrenderable inputs without compatible concrete tokens also stay unknown.
Association admission is best effort on a bounded background queue; control
resolution exposes missing metadata rather than guessing a latest branch.

Location results are scoped Engine queries, not operation-result leases. Pending
routing stakes never create confirmed locations. A failed query, epoch/generation
change or unhealthy worker removes preparation evidence. Normal routing consumes
only a bounded recent advisory view (at most two configured poll periods); it
retains final admission/affinity/load decisions. When Engine discovery advertises
feedback `events` version 1.0, the Router polls its journal and fences the view
with its independent native cache watermark. The operation evidence sequence is never
compared to a cache stream version. Results include worker URL, owner epoch,
query age, and cache watermark when supplied; formal HTTP responses include
`x-sgl-worker-url`. Storage-only preservation/PIN is not Engine host readiness.

For L2, live Engine capabilities must advertise PIN. Query results include the
current `kind=claim` result; a PIN can be active with an empty prefetch-ready scope.
For L3, configure the single backend owner below. Engine `storage_targets` supplies
actual K/V object keys; short event hashes cannot select backend keys.

```json
{
  "admin_url": "http://127.0.0.1:52092",
  "domain": {
    "cluster": "local-mooncake",
    "tenant_id": "default",
    "model_namespace": "qwen3-8b-bf16-v1",
    "storage_namespace": "sglang.storage.v1:737b851e80e9b107816bf784708951c630ce594e2a3adaab26fd9a16a5aefdf3",
    "layout_id": "python:full:p64:w0"
  },
  "owner_epoch": "unique-deployment-incarnation",
  "fresh_domain_assertion": null,
  "private_missing_probe_key": "private-owned-confirmed-absent-probe-key",
  "max_ttl_ms": 60000,
  "max_objects": 4096,
  "max_objects_per_claim": 4096,
  "max_claims_per_object": 128,
  "max_inflight_batches": 8,
  "batch_size": 256
}
```

Install that object as `backend_owner` in the config. `storage_namespace` must
match current Engine caps; the example records this local pair's inspected value.
A null `fresh_domain_assertion` starts in UnknownDomain and cannot advertise
positive protection. Use a nonempty assertion only after externally establishing
single ownership and absence of old writer/RPC/claim obligations; create a unique
owner epoch and an owned absent probe key. Fresh startup probes both admin APIs.
TTL bounds must match the deployed server. A Router restart does not prove a fresh
domain. Unsupported or unknown protection leaves inference available.

An L3 PIN command is:

```json
{"kind":"submit","task":{"session_id":"s","incarnation":"task-1"},"message_id":"keep-storage","action":"sglang.pin","target":{"kind":"branch","branch_id":"main"},"location":{"tier":"storage"},"mode":"acquire","claim_id":"keep","revision":1,"ttl_seconds":30,"required_protection":"backend_soft_pin"}
```

Renew/release reuse task, claim ID and scope with a newer claim revision.
Protection status is returned under `claims`; real backend receipts determine
active/partial/unknown. Task cleanup releases only its claims and preserves other
tasks sharing the same physical objects. `tool_wait` optionally accepts
`retention={tier,ttl_seconds,required_protection}` to create a bounded retention
intent; pressure-free signals skip DEMOTE.

Persistent collections, queue, token bytes, transport responses, claims and
operation replay are bounded. A full active budget refuses new control work;
inference continues. Acknowledge drained intents to reclaim operation history.
Lifecycle replay/task tombstones intentionally refuse new state at their limits;
they cannot be silently discarded to accept stale incarnations. This is one
Router owner in memory; durable/multi-Router ownership and publisher/indexer replay
are separate from the polling recovery described below.


## Engine feedback polling (P7c)

Polling is enabled by trusted Engine capabilities `feedback.query=events`,
`version=1.0`, and `recovery=query_known_scopes`. No publisher socket or global
placement mirror is created. Each registered owner has a bounded cursor and one
native cache watermark. A background tick refreshes/polls one current healthy
owner, and scoped residency/claim queries have feedback polls around them. Each
page requests at most `min(queue_capacity,256)` events and obeys
`max_engine_bytes`. Owner entries are bounded by `max_workers`; exhausted metadata
refuses new feedback-dependent readiness rather than discarding an active fence.

The Router binds journal and native cache source IDs/incarnations from discovery.
A page cannot rebind an owner just by arriving later. It validates Engine epoch,
namespace, caller, cursor bounds and ordered event sequences. Caller filtering can
skip sequence numbers, so skipped numbers alone do not imply a gap. Retention
loss/explicit gaps require current queries of known scopes, operations and claims.
An empty page never certifies a complete Engine snapshot. Old recorder history
inside a current journal is consumed conservatively as unknown and cannot change
the discovered source or install a location.

Native cache `generation/version` comparisons apply only within the same cache
source ID/incarnation. They are independent of journal sequence, operation
`public_result.state_version`, outcome sequence, arrival order and wall clock.
A newer native mutation invalidates older preparation evidence even when its
journal page has no new caller-visible events. The Router polls after a residency
query as well: a delayed response older than that watermark is rejected. A query
crossing a gap/invalidation/source failure is also rejected even if its native
version happens to be equal. Current matching queries recover only their requested
scopes; no global completeness state is created. Late operation events remain
historical outcomes and cannot assert current residency.

Journal source changes, capability loss, owner/cache generation changes, polling
failures and explicit invalidations clear affected advisory/claim observations.
An Engine that never advertised feedback retains legacy scoped-query fallback.
Once advertised, disappearing capabilities cannot silently downgrade it to
unfenced legacy confirmation; feedback discovery/polling must recover first.
Both chat routing paths consume the same fenced publication and retain normal
policy/admission control.

Storage attachment invalidation suppresses positive L3 claim views while keeping
claim identities, interests and cleanup available. A cached backend-owner receipt
cannot prove that an invalidated storage attachment is still usable. Those old
claim views remain unknown pending external object/domain reconciliation; polling
the same cached receipt cannot restore positive protection. Other owner-local L2
claim snapshots remain independent of storage-only invalidation. A new authoritative
claim query is needed after owner-local protection changes; journal claim payloads
are not installed as current protection.

Authenticated diagnostic inspection uses:

```json
{"kind":"feedback"}
```

`POST /v1/kv/control` returns owner cursors/native watermarks/errors,
`rejected_scope_queries`, last recovery cause, and bounded association/request-use
facts. Retention uses at most `max_operations` diagnostic records and
`max_engine_bytes` aggregate encoded bytes; older diagnostics can be dropped, with
an explicit dropped count. Only bounded identity, accounting and provenance fields
are retained; committed token material is excluded. The response always says
`coverage=requested_scopes_only`, `recovery=query_known_scopes`, and
`global_snapshot_complete=false`.

These are diagnostics, with `task_authority=none`. Missing provenance stays
`unknown`; even supplied branch/session labels do not create or resurrect a task,
claim or branch, infer a latest branch, or extend P5 associations. The journal HTTP
API filters to the configured caller plus owner-wide invalidations. Ordinary
unhinted Engine requests use caller `request`, so a Router polling as its configured
control caller does **not** automatically receive their `request_use` events.
Privileged subscribers or an explicitly authorized `request` poller could observe
those facts separately; this adapter does neither. P6's concrete rendered/input-token
association remains the ordinary Router association path. No automatic association
completion from unhinted request-use feedback is claimed.
