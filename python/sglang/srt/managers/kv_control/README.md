# Programmable KV control and local cache priorities

This package implements native control contracts and logical lifecycle.
`KVControl` is disabled by default. `--kv-control-config` explicitly opts the
scheduler into the service and copy-priority profile; no controller/claims are
allocated on the disabled path. The service executes explicit commands; there
are no configured policies.
With no executor, cache primitives return `unsupported`. The optional
`UnifiedCacheExecutor` now applies copy-scoped priorities to real
`UnifiedRadixCache` copies and inspects their current residency. The HTTP/Engine
control APIs and attached request lifecycle use that executor. A persistent-host
HiCache adapter also executes FULL-prefix COPY/MOVE on its actual data path.
Verification includes CPU pools, real CUDA byte checks and an OPT dummy-weight
server smoke test with loaded-prefix reuse.

The design follows the accepted
[SGLang RFC](../../../../../docs/programmable-kvc/RFC-programmable-kv-cache.md)
and [design decisions](../../../../../docs/programmable-kvc/design-differences.md).
The vLLM `kv_hints` contracts, registry, index and dispatcher and the October
18–20 reports were design references. This implementation has no vLLM imports
or copied source. In particular, it preserves per-action interpretation instead
of adopting the reference's envelope-wide native/backend consumer switch.

Use the existing `KvHintsEnvelope` / `KvHintAction`. The native consumer handles
`sglang.*`, protocol `0.1`, action version `1.0`. Other action names are returned
unchanged in `Submission.unhandled_actions`; their payloads are never inspected,
fingerprinted, retained or executed here. The integration owner forwards those
actions to their individual consumers. It must not also forward consumed native
actions to another interpreter. Existing generic envelope transport is unchanged.
An opaque consumer remains responsible for its own validation and replay.

| Native action | Typed payload in `contracts.py` | Meaning |
| --- | --- | --- |
| `sglang.set_priority` | `SetPriority(target, location, claim_id, revision, value, ttl_seconds)` | Signed soft priority on currently resolved copies; null value clears the caller's claim |
| `sglang.copy_to` | `CopyTo(target, src, dst)` | Request destination publication while preserving source |
| `sglang.move_to` | `MoveTo(target, src, dst)` | Request destination publication, then eligible source reclamation |
| `sglang.associate` | `Associate(session_id, continuation_id, expected_revision, target, task_parent?, parent_prefix?)` | Create or append an exact logical continuation |
| `sglang.finish_task` | `FinishTask(session_id, expected_revision)` | Remove a task tree's logical associations and abandon its pending operation interests |
| `sglang.signal` | `Signal(target, signal, importance?, resume_after_seconds?)` | Record an `unconsumed` outcome without resolving history or collecting observations |

All structs use keyword arguments. Payloads forbid unknown fields. Signal names
are `tool_wait`, `resume`, `importance`, and `resume_prediction`. A tool wait or
request completion does not end a task. Locations are objects with `tier` equal
to `gpu`, `host`, or `storage`, plus optional `engine_id` and `backend`. Omitting
the engine means the addressed engine. A different engine ID is unsupported:
that engine must approve its own resources. A backend name never establishes an
executable capability. There are no public physical addresses or Pin semantics.

Priority values are signed 32-bit integers. Revisions and token coordinates are
nonnegative integers at most `2**53 - 1`; booleans and float-valued integers do
not satisfy integer fields. TTL and resume estimates must be positive, finite,
and at most 365 days. Setting a claim requires TTL; clearing requires a higher
revision and no TTL. The executor must scope claim IDs by `context.caller_id`,
copy incarnation, namespace, epoch, location and coverage. Equal/older revisions
must not renew claims, equal conflicting revisions must fail before mutation,
and different claims keep independent expiry. TTL starts at effectiveness;
hits/splits cannot renew it and reconstructed copies cannot inherit it. The
cache-owned executor implements these rules for the supported configurations
below. Equal/older revisions return `revision_unchanged` on affected outcomes;
they acknowledge revision processing without claiming a renewed lifetime.

Targets use tagged objects:

| `kind` | Fields | Resolution |
| --- | --- | --- |
| `tokens` | `namespace`, `token_ids` | Exact token prefix from position zero; physical component/layout information remains unknown |
| `content` | `namespace`, `token_count`, `components`, `spans` | Explicit logical descriptors, including gaps in known history |
| `session` | `namespace`, `session_id`, optional `continuation_id`, `expected_revision` | One explicit continuation or all tracked continuations in that session |

Token and content targets also carry `cache_salt`, `extra_key`, `layout_id`, and
`is_multimodal`. Their defaults describe unsalted text in the addressed cache's
layout. The logical index preserves and compares these fields during append,
branch inheritance and reverse lookup. A session target uses its frozen content's
compatibility metadata. The cache executor supplies its exact `layout_id` through
capability discovery; an omitted layout uses that configuration. Explicit extra
keys (including LoRA) and multimodal targets are unsupported. They never fall
through to a base-model lookup. The cache's configured namespace is an owner
assertion of model/weight compatibility; serving integration must supply it and
derive attached-request metadata from the actual request.

All three accept `start_token=0`, optional exclusive `end_token`, and
`allow_clip=False`. A `ContentSpan` has `content_id`, `component`, `start_token`,
`end_token`, and `dependency_start=0`. It also carries `coverage_kind` (`tokens`,
`window`, or `checkpoint`) and optional `window_size`. Component names identify the namespace's
layout; a name such as `mamba` does not advertise executable Mamba support.
Checkpoint descriptors are nonzero points `[end, end]`. Window descriptors
identify actual token spans in a trailing window and require its positive size;
the resolver does not invent missing per-token Mamba state or require historical
SWA data outside that window. Descriptors are indivisible and nonoverlapping
within a component. Resolution
separately reports fully included spans, intersecting unaligned spans, missing
component ranges, and earlier dependency ranges. It never rounds outward.
An exact action intersecting an indivisible descriptor is unsupported. Explicit
clipping allows the executor to act on included descriptors and report partial
coverage. Prefix/window/checkpoint feasibility and dependency materialization
remain cache-owned. Overlapping checkpoint aliases require a later descriptor
extension; this core rejects them rather than inventing their meaning.

`"request_cached"` is the fourth target spelling, a string. It requires a unique
request incarnation in `ControlContext.binding_id`. The successful-completion
callback supplies authoritative committed content, including any supported
generated suffix. No checkpoint or preemption callback binds or executes it.
Abort/error cancels queued actions. Attached SET_PRIORITY and GPU-source
COPY/MOVE also wait for success; their historical targets are already frozen at
submission. Cold-source historical COPY executes immediately. All historical
targets in a message resolve before any sibling mutates the index. Action order
does not express a transaction or COPY-to-priority dependency.

The logical index accepts full-prefix associations only. `expected_revision=0`
creates a continuation; updates require its current revision. Token sequences
must extend the previous sequence exactly. Descriptors can add history but cannot
replace existing spans, shrink the boundary, or change components. Changing
content requires a new continuation. A `ParentPrefix` names the parent's session,
continuation, expected revision and exact end boundary; a child declares a full
prefix whose inherited portion must match. Parent extension never expands the
captured branch prefix. `task_parent: TaskReference` separately specifies task
ancestry. Task ancestry is immutable and never inferred from token sharing.

The index generates fresh continuation/task incarnations and monotonic revisions.
Descendant changes advance ancestor session revisions, so stale task cleanup
cannot erase newly attached branches. Raw `index.finish_task` returns
`LogicalChange` with the removed `TaskIdentity` values. Raw index pruning and
task cleanup only mutate logical state; immutable snapshots remain valid and
existing operation interests are unchanged. No index operation deletes cache
bytes, Store objects, or active request/transfer references.

The native `sglang.finish_task` action additionally abandons pending operations
addressed through those removed task/continuation incarnations, including child
tasks and snapshots prepared by siblings in the same envelope. Matching uses
`TaskIdentity` from each frozen target, so a reopened session name cannot select
an older incarnation's interests. Queued operations become `cancelled` before
dispatch. Started operations request executor cancellation and remain active
until their obligations are quiescent. The finish action's completed result
reports logical cleanup and cancellation initiation; affected started operations
retain their own pending results. Other tasks selecting identical content keep
their interests. Bare token/content targets and `request_cached` do not infer
task ownership from shared content. Stale cleanup revisions are rejected and
replaying a completed finish does not repeat cleanup. The core frees no buffers.
Units 2/4 still use the removed identities to clean up policy-owned claims and
policy work. Arbitrary explicit claim IDs do not imply task ownership.

Correlation is scoped by caller, request binding and message ID inside a fixed
engine epoch/namespace. Native action IDs must be unique within that message.
Native JSON is snapshotted before acceptance, so later input mutation cannot
change recorded intent or a queued execution. A changed replay conflicts; an
identical replay returns retained outcomes without executing again. Malformed
payloads and unsupported native actions have individual outcomes. Invalid
wrappers, non-JSON native values, duplicate native IDs or serialization budgets
reject the native submission before side effects. Independent opaque payloads
are outside these checks.

`Limits` bounds native action count/bytes/JSON nodes, active work, detailed
records, replay messages, sessions, continuations, indexed spans/tokens, resolved
scope and query pages. Active, record and replay slots are checked together
before acceptance. There is no eviction of active work. The replay slot is
reserved up front, so ACK never needs an additional slot. A message's retention
deadline starts when all native actions become terminal. ACK releases details
while preserving fingerprint, operation IDs, statuses and error summaries until
that original deadline; repeated queries/ACK/replays do not renew it. An ACK
receipt preserves failed/partial statuses. `replay_expires_at` uses the supplied
monotonic clock, not wall time. After expiry, callers must use unique IDs rather
than assume indefinite exactly-once execution.

`queued` denotes waiting without physical reservations. `deferred` is a terminal
resource refusal, retried deliberately under a new message ID. Cancellation
requires the full `OperationRef`, including the expected operation incarnation.
Queued cancellation ends immediately. Dispatched cancellation remains
`cancelling` until the executor certifies quiescence. Exceptions do not certify
quiescence; the core retains the record, sends cancellation during progress, and
waits for explicit retirement. It preserves the originating executor failure
through subsequent cleanup errors and progress reports. A quiescent cancellation
receipt after an executor exception produces `failed`, with the original error
code/reason; intentional cancellation without an executor failure produces
`cancelled`. Previously observed completed stages and target outcomes survive
sparse drain receipts, and additional observed progress is merged within the
existing result bounds. ACK and replay preserve the failed status and cause.
Terminal results are immutable and ACK cannot retire active work. Cancellation
of one operation does not cancel another operation selecting the same content;
physical load coalescing and buffer retirement belong to the future transfer
owner.

Unit 2's integration surface is:

| Interface | Required integration use |
| --- | --- |
| `KVControl(engine_id, namespace, enabled=False, executor=None, limits=Limits(), clock=time.monotonic)` | Construct on the scheduler owner; startup configuration remains outside unit 1 |
| `core.context(caller_id, binding_id=None)` | Bind trusted caller identity and unique fan-out/request incarnation |
| `core.submit(envelope, context) -> Submission` | Reuse both attached and independent control ingress; route unhandled actions separately |
| `core.request_event(context, event, committed=None) -> tuple[OperationResult, ...]` | Report success/abort/error/preempt/checkpoint; success must run before releasing request references |
| `Executor.start(ExecutionRequest) -> ExecutionUpdate` | Validate capability and current physical state/capacity, acquire real holds, apply or dispatch approved work |
| `Executor.poll(OperationRef) -> ExecutionUpdate \| None` | Publish cumulative progress; all calls execute on the owner thread |
| `Executor.cancel(OperationRef) -> ExecutionUpdate \| None` | Cancel one caller's interest and drain its obligations without releasing other users' holds |
| `core.progress(limit=64)` | Progress bounded batches, including when no inference request is active |
| `core.cancel(ref)` | Validate incarnation before cancellation |
| `core.registry.get(ref)`, `.query(context, message_id)`, `.acknowledge(context, message_id)`, `.capacity()` | Implement operation APIs and metadata occupancy reporting |
| `core.index.associate(payload)`, `.resolve(target)`, `.get(session_id, continuation_id)` | Feed authoritative logical content observations and take immutable snapshots |
| `core.index.query(session_id=None, cursor=None, limit=256)` | Paginate with epoch, namespace, generation and filter checks; restart after mutation |
| `core.index.members(span)`, `.token_members(prefix)` | Query exact descriptor or token-prefix sharing; token lookup scans the bounded logical index only when requested |
| `core.index.finish_task(session_id, expected_revision=...)`, `.prune(...)` | Raw logical cleanup only; use the native finish action for operation-interest cleanup, and propagate removed task identities to future policy claim cleanup |
| `core.reset(namespace)` | Rotate logical epoch only after real cache/reset/weight-update draining; refuses active operations |

An `ExecutionRequest` contains the operation reference, immutable parsed primitive
and `ResolvedTarget`. `TargetOutcome.branch_index` identifies a branch in that
frozen scope. Outcomes report component/range/content and actual destination.
`ExecutionUpdate.completed_steps` and each target's steps are cumulative;
successful COPY requires `destination_ready`, successful MOVE additionally
requires `source_reclaimed` in that order, and priority requires
`priority_applied` or `priority_cleared`. The core checks declared coverage,
branches, stages and destination before accepting complete results. For token
targets, the executor must discover and validate all required components. The
executor alone can certify DMA completion, actual publication and quiescence.
Cancelling a logical interest never authorizes freeing a buffer still accessed
by a worker or another waiter. Serving integration must continue progress until
those obligations end, even after a task's finish action completes.

Multi-rank publication, linker transfers, policies and event feeds remain
pending. File-backed storage transfers are described below. CPU tests validate in-process HTTP/IPC and cache/request behavior; GPU
tests cover the explicitly described one-owner FULL profile. No serving
performance or model-quality claim follows from these smoke tests.

## Serving entry points

Launch with an explicit model/weight compatibility namespace:

```bash
python -m sglang.launch_server --model-path MODEL \
  --kv-control-config '{"namespace":"model-weights-v1","engine_id":"worker-a"}'
```

The current executable profile requires TP=PP=CP=DP=1 on one node, the enabled
built-in unified cache and ordinary text generation. Speculative decoding, PD
disaggregation and alternate caches are rejected at startup. Persistent-host
HiCache requires FULL-only state and `--disable-overlap-schedule`; buffer-only
and direct-linker control paths remain unsupported. File-backed native storage
transfers are available for one complete token prefix. Python backend
fallback is deliberate.
LoRA, multimodal and StreamingSession primitive execution returns unsupported;
ordinary inference and opaque action consumers retain their existing behavior.
The serving CPU/CUDA profile supports guarded weight transitions. The callable
owner-assertion mode without namespaced ordinary keys continues to refuse weight
mutation. Discovery reports the current update state and committed version.

GET `/v1/kv/capabilities` returns `{success, data}` with `engine_id`,
`engine_epoch`, `namespace`, executable actions/components/layout and the selected
topology. POST `/v1/kv/control` accepts a strict tagged command:

| `kind` | Required fields beyond `kind` |
| --- | --- |
| `capabilities` | none |
| `submit` | `context`, `envelope` (the existing KV hints envelope) |
| `query`, `ack` | `context`, `message_id` |
| `lookup` | `context`, `message_id`, optional bounded `limit` |
| `cancel` | exact operation `ref` returned in a submission/query |
| `residency` | `context`, logical `target`, optional `limit` and `location` (default GPU) |
| `associations` | `context`, optional `session_id`, `cursor`, `limit` |
| `capacity` | `context` |

`context` contains the discovered engine/epoch/namespace and a `caller_id`.
Independent submission must omit `binding_id`. Example with Python `requests`:

```python
import requests

base = "http://localhost:30000"
owner = requests.get(base + "/v1/kv/capabilities").json()["data"]
context = {k: owner[k] for k in ("engine_id", "engine_epoch", "namespace")}
context["caller_id"] = "router-a"
reply = requests.post(base + "/v1/kv/control", json={
    "kind": "submit", "context": context, "envelope": kv_hints,
}).json()
```

The routes use the server's `ADMIN_OPTIONAL` authentication policy, matching
cache flush. Caller/session IDs are correlation and declaration identities, not
credentials. Top-level errors carry `{success: false, error: {code, reason}}`;
HTTP maps stale ownership/conflicts to 409, capacity to 429, missing to 404 and
unsupported service to 503. Accepted envelopes have individual action outcomes;
an HTTP 200 receipt does not imply each action completed successfully. Foreign
independent actions are returned unhandled for their owner.

Python Engine exposes `kv_capabilities()`, `kv_control(command)` and matching
`async_kv_capabilities()` / `async_kv_control(command)`. They use typed IPC through
the tokenizer's correlated communicator, with no synthesized generation request.
RPC cancellation does not cancel accepted work: query or explicitly cancel it.
Late responses from a cancelled waiter cannot satisfy a later control call.

Native `/generate` and supported OpenAI chat/completion requests accept
`kv_hints`. Each sample/request incarnation receives an independent binding even
when the original envelope or `rid` is shared. Discover attached results with
`kind: lookup`, `message_id` and `context.caller_id: request`, omitting the
binding. Each returned result contains its exact context/reference for query,
ACK or cancellation. Lookup is bounded and reports overflow instead of silently
dropping branches; narrow subsequent queries to a returned binding.

The native consumer removes only consumed actions from a fresh envelope for
that request. Successful completion binds `request_cached` to actual committed
tokens, including a committed generated suffix and the request's effective salt,
after checkpoint and before releasing cache references. Ordinary checkpoints and
preemption do not complete it. Queued aborts are cancelled even without a KV row.
Malformed native siblings have individual rejected results. Whole-envelope or
registry-capacity rejection aborts the generation with a controlled request error.
Idle processing progresses independent work. Active work prevents destructive
flush; successful cache flush rotates the epoch and invalidates old contexts.

Opt into cache-owned priority execution when constructing the real cache:

```python
from sglang.srt.managers.kv_control import KVControl
from sglang.srt.mem_cache.unified_cache.priority_claims import CacheControlConfig
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

# params is the owner's existing CacheInitParams with real pools/allocator.
params.kv_control_config = CacheControlConfig(namespace="model-v1/layout-v1")
cache = UnifiedRadixCache(params)
control = KVControl("worker-a", "model-v1/layout-v1", enabled=True)
control.executor = cache.create_kv_control_executor(
    engine_id="worker-a", engine_epoch=control.registry.engine_epoch
)
```

`CacheInitParams.kv_control_config` is keyword-only and defaults to `None`.
Disabled caches allocate no claim table. Python TreeCore owns the optional table;
the existing central backend resolver deliberately falls back from Rust to
Python when the option is supplied, before attempting to load Rust. Custom
backends/components and a disabled cache reject this opt-in. Its eviction profile
supports the `lru` and `priority` configuration names and orders device victims
by explicit signed claims, with default zero and LRU ties. Legacy session bias
and `node.priority` do not override that profile. Session associations remain
available; session preference can become an explicit policy input in a later
unit. Host-copy ordering retains its existing contract.

FULL leaf eviction, SWA/Mamba component eviction and the Mamba path-state cap
consult live claim TTLs. Where an eviction would cascade across components, its
ranking includes the maximum valid explicit claim among the affected copies.
Locks, structural eligibility and host backup/retirement paths still belong to
the cache. Priority never changes correctness reference counts or allocates KV
slots. The opt-in selector builds a candidate heap once per eviction walk and
refreshes newly exposed parents. Explicit edits or crossing a live TTL deadline
rebuild the ordering; ordinary victim removal does not rescan the cache. CPU
regressions bound key evaluations for FULL/SWA/Mamba and check expiry during a
walk. Serving overhead and GPU performance have not been measured.

The side table stores `(caller_id, claim_id)` revisions independently for each
copy incarnation. Clears and expired claims retain revision tombstones until
copy invalidation. Eviction, component tombstoning/recovery, FULL replacement
during SWA recovery and reset end the old claims. Splits propagate only the
selected physical fragments with the original revision and expiry; Mamba state
remains on its checkpoint fragment. No fields were added to ordinary tree nodes
or `ComponentData`.

`CacheControlConfig` bounds `max_copy_units=65536`, `max_claims_per_copy=16`,
`max_query_nodes=4096` and `max_result_targets=4096`. A claimed token segment
reserves enough metadata units for all its possible page-sized fragments; a
Mamba checkpoint reserves one unit. This bounds split growth without rejecting
ordinary tree splits or dropping valid claims. Higher revisions replace entries;
there is no expiry heap. Conflict and capacity validation cover the whole
selected scope before splitting or changing any claim. Result planning also
honors the controller's `ExecutionRequest.max_result_targets` budget.

| Current capability | Supported scope |
| --- | --- |
| Executor operation | Synchronous `sglang.set_priority`, local device tier (`gpu` in the protocol, backed by CPU pools in these tests) |
| Components | Without HiCache: built-in FULL, FULL+SWA, FULL+Mamba and FULL+SWA+Mamba; persistent-host control currently requires FULL |
| Addressing | Tokens, frozen session/continuation targets, authoritative `request_cached` completion content, and descriptors returned by residency inspection |
| FULL coverage | Page-aligned token ranges, including boundaries inside radix nodes |
| SWA coverage | Actual page-aligned trailing-window spans, with the physical window size reported; explicit window/token descriptors also resolve current copies |
| Mamba coverage | An existing sequence-end checkpoint, reported as a point; absent intermediate checkpoints produce missing/partial results |
| Ingress/ownership | HTTP/Engine or callable core, typed IPC, one scheduler/cache owner and epoch |
| Unsupported priority scope | Host/storage priority, priority on pending transfers, TP/PP/CP/sharded execution, speculative/bigram/draft/extra-buffer layouts, LoRA/extra-key and multimodal targeting, custom components/backends |

Tokens are matched through actual namespace-aware `RadixKey` traversal without
touching recency or creating cache hits. Prefix descriptor IDs hash the compatible
logical prefix and component; they contain no physical indices. Descriptor
lookup is bounded and remains valid across radix splits. Exact unaligned token
ranges are unsupported. With `allow_clip=True`, only complete pages are selected
and omitted edges are reported; an empty aligned scope is unsupported. Missing
components produce partial/missing results. No control response promises lasting
residency or reconstructs missing cache data.

The cache executor exposes:

```python
executor.capabilities()  # executable actions/components, layout_id and generation
snapshot = executor.residency(control.index.resolve(target), limit=256)
```

`ResidencySnapshot.items` contains bounded `ResidencyEntry` values with logical
descriptors, coverage kind/window size, effective priority and current lock count.
`missing` reports unrepresented/unresident portions. These reads acquire no
cache ownership. `cache.tree_core.priority_claims.capacity()` reports actual
side-table occupancy and reserved metadata units.

## Model namespaces and actual keys

The scheduler namespaces every ordinary request key when control is enabled,
including requests without hints. Actual tree/store `cache_salt` is the SHA256
of compact UTF-8 JSON `["sglang.kv.weight.v1", namespace, original_salt_or_null]`.
Discovery reports the current namespace and encoding. Public token/content
targets continue to carry the original logical salt; the executor applies the
same mapping for lookup and descriptor hashes. Passing a physical encoded salt
as a logical salt does not bypass that mapping.

Preparation is idempotent for a request and captures its original logical salt.
`request_cached` completion uses that original salt, so binding does not encode
it twice. Preparation occurs outside the request-constructor extension point;
forks returning a `Req` subclass still receive namespace and hint processing.
Flush rebind preserves both the key mapper and scheduler capacity approval.

The private owner namespace-rebind seam requires model idleness, no outstanding
operations/transfers, no StreamingSession-owned KV and no correctness references.
It rotates epoch, logical records and declarations. Old physical copies may
remain under old encoded keys, but new requests/control cannot match them; their
actual storage root hashes differ. This seam does not mutate model parameters.
Weight mutation uses this seam before touching parameters and at publication of
a successfully labelled model generation.

The low-level callable service defaults to owner-asserted compatibility without
rewriting manually constructed cache keys. The serving scheduler explicitly
enables `namespace_keys` and supplies its idle check. A low-level owner opting
into namespaced keys must prepare requests before insertion and supply an idle
check before invoking namespace rebind.

## Parameter update lifecycle

Disk/IPC updates use a standalone fence. Tensor/distributed updates use the
existing `begin_weight_update()` / `end_weight_update()` session. The owner
requires drained inference, control, transfers, StreamingSession KV and
correctness references before changing lookup generations or packing parameters.
During an open update, new generation/embedding requests are aborted directly
with 503 before construction or forwarding; independent KV submissions return
busy. Read-only discovery remains available.

Parameter work must synchronize before requested cache flush and publication.
With `flush_cache=False`, old physical cache entries can remain under old
namespace keys and cannot satisfy new ordinary or independent lookup. A supplied
new `weight_version` derives a compatible generation from the configured base
namespace; a known unchanged label or mixed labels in a session are refused.
Unlabelled updates use fresh generations. Metadata version assertions and the
debug checker's `reset_tensors` action use the same isolation boundary.

```python
engine.begin_weight_update(selector="target")
engine.update_weights_from_tensor(
    named_tensors, weight_version="step-2", selector="target", flush_cache=False,
)
engine.end_weight_update()
owner = engine.kv_capabilities()["data"]  # obtain the new context
```

Runner failures/rollback retain a fresh isolated lookup generation; a failed
chunk is not converted to successful publication by a later `end`. Unexpected
parameter exceptions or synchronization failures keep inference fenced. Final
cache-flush failure can be retried through `end` without repeating the parameter
mutation. Existing runners retain ownership of parameter loading/rollback; this
layer does not assert atomic replacement of every numerical model parameter.
Memory release/resume refuses an open weight fence, and offloaded resources must
be resumed before parameter mutation.

Cache reset clears claims and changes the cache generation. Existing executors
reject subsequent work. After the owner safely resets the cache and calls
`control.reset(namespace)`, it must construct a new executor with the fresh
control epoch. Rebinding a live generation to a different owner/epoch, or reusing
an old epoch after reset, is rejected. A weight/layout namespace change requires
a correspondingly configured cache, not relabeling existing copies. Serving
integration must supply rank-consistent identity or an explicit single-owner
architecture and address DP replicas separately; distributed execution is not
advertised by this adapter.

Native `finish_task` continues to abandon pending operation interests. Completed
explicit copy claims retain their caller-controlled lifetime, so task cleanup
cannot revoke another caller's declaration or delete shared bytes. Policy-owned
claim ownership and cleanup remain a unit 2/4 integration obligation.

## Persistent-host movement

Enable the supported serving profile with the ordinary HiCache options:

```bash
python -m sglang.launch_server --model-path MODEL \
  --kv-control-config '{"namespace":"model-weights-v1","engine_id":"worker-a"}' \
  --enable-hierarchical-cache --hicache-write-policy write_back \
  --hicache-host-memory-mode cache --disable-overlap-schedule
```

This adapter supports COPY GPU→host, COPY host→GPU, and MOVE GPU→host.
Discovery lists these directions only when the actual cache/controller supports
them. Transfers address whole page-aligned prefixes at existing radix boundaries
using tokens, token-backed session snapshots or `request_cached`. Other ranges,
descriptor-only transfers, hybrid state, buffer/draft/sidecar paths, host→GPU
MOVE and direct-linker publication are unsupported in this stage. Storage transfers support a single complete token prefix.

Approval counts each selected physical fragment once, including shared branch
prefixes. It protects already-resident members and checks the complete reusable
scope before queuing any allocation/transfer. The scheduler supplies a fresh
allocator prefill budget with its existing running-decode growth estimate;
subsequent branch loads recheck current capacity and allocations. That estimate
is not an additional memory reservation carried into request admission. The
callable cache API without a scheduler uses allocator capacity; its owner remains
responsible for admission headroom.

Temporary source/capacity waits remain `queued`, hold no buffers and can be
cancelled. A known scope larger than the entire destination pool returns terminal
`deferred`, so an impossible scope cannot leave the engine indefinitely active.
`max_query_nodes` bounds worst-case prefix pages across branches, preserving room
to inspect completion even if ordinary requests split every page during DMA.
Per-caller result budgets cover partial destination runs and source retention
before execution. Adjacent equivalent results are merged into logical ranges.

Compatible interests share the transfer and allocator work, with separate
contexts and results. Cancelling one interest does not release the job's holds.
The last cancellation drains queued/active copies to a real completion fence
before retiring its buffers. Partial submission failures retain their first
cause, observed published ranges and failed status through cancellation,
ACK/replay and a subsequent independent retry.

COPY completes after the required HiCache fences and publication. MOVE then
releases only eligible device residency; request references or external branches
produce `partial` with `destination_ready` and `source_busy` for retained ranges.
Successful reclamation additionally reports `source_reclaimed`. Host inspection
uses `location: {"tier": "host"}` and identifies that location per entry. It does
not treat pending DMA indices as ready data. Fresh GPU copies have no inherited
priority claims.

Native H2D uses HiCache's existing allocation/commit/load machinery. Its installed
indices are not ready until the fence. While such a native load is pending,
fresh prefill admission waits; running decode continues under the supported
non-overlap scheduler. Idle control progress drives the fences and removes the
barrier on completion. The normal request-driven load threshold stays unchanged;
an explicit approved prefix can load below that heuristic threshold.

Run from the repository root:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest test/registered/unit/managers/test_kv_control.py -q
```

The real-cache CPU proof on this host uses a nonexistent device index because
the existing full test utilities require a nonempty `CUDA_VISIBLE_DEVICES`:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=999 .venv/bin/python -m pytest \
  test/registered/unit/managers/test_kv_control.py \
  test/registered/unit/mem_cache/test_kv_control_cache.py \
  test/registered/unit/mem_cache/test_session_unified_radix_cache.py -q
```

The cache suite asserts `torch.cuda.device_count() == 0`, constructs real CPU
FULL/SWA/Mamba pools and checks actual victims, pool balances and lock counts.
Initial CPU runs used PyTorch 2.13.0+cu132 and gguf 0.19.0. The target virtualenv
was later updated to the pinned PyTorch 2.14.1+cu130, torchvision 0.29.1 and
sglang-kernel 0.4.10 for GPU 1 validation; the reference environment is unchanged.
The core/cache/service/communicator suites passed again on that runtime (114
tests). This does not validate the complete production GPU stack. Existing joint-allocation tests
were run with `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`: the unmodified
default Rust route cannot build with this host's Cargo 1.75 (edition 2024 is
required). The opt-in Rust-to-Python fallback itself is covered by the CPU suite.

`test_kv_control_service.py` adds in-process HTTP/typed IPC, real committed request
release, fan-out discovery, session reconstruction and queued-abort coverage.
The pure control suite has no runtime dependencies; the cache/service suites use
the repository's CPU pool fixtures and need no model or backend service.

The manually invoked real-device proof is:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  test/manual/entrypoints/programmable_kv_gpu_smoke.py
```

It verifies actual CUDA KV bytes, victims, request locks, soft-priority reclaim,
copy-lifetime/replay and allocator balance on one A100. The corresponding real
HiCache proof is `test/manual/entrypoints/programmable_kv_hicache_smoke.py`; it
checks actual host/CUDA K/V bytes, MOVE reclamation, COPY restoration and final
references/pool balances using the direct I/O backend.

The reusable server proof is:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  test/manual/entrypoints/programmable_kv_server_smoke.py \
  --model-path LOCAL_OPT_CONFIG --load-format dummy --hicache
```

It launches an owned server, completes generation with attached priority,
discovers the result, independently demotes the committed prefix, verifies its
GPU absence and host presence, restores it, and observes cache hits on a repeated
generation with equal output tokens. It also checks fresh-copy priority, flush
epoch rotation and stale-reference rejection, retaining logs/results. Dummy
weights validate the interface and execution path. A local tiny-Llama checkpoint
attempt still fails before cache initialization at the existing DualGemm
`cutlass.cute` import in this environment; that unsuccessful path is retained in
the validation record. Neither fixture establishes model quality or throughput.


## File storage transfers

With persistent FULL HiCache and the actual `file` storage backend, discovery
adds COPY GPU→storage, COPY host→storage and MOVE GPU→storage. The optional
`dst.backend` must be `file` or omitted. A storage write requires one complete
page-aligned token prefix at an existing radix boundary, with at least two
result slots for a possible short write. Multi-branch cold transfers and other
storage backends return unsupported. Ordinary storage prefetch and backups
retain their existing behavior.

GPU-source writes first use the existing host copy path with whole-scope
capacity approval. Explicit GPU→host COPY does not enqueue an additional cold
write; a separate storage action owns that side effect. Identical storage jobs
share staging and one write, while each caller can cancel independently. The
last cancellation drains both DMA and storage worker access before releasing
source/host holds. No public pin or persistent retention guarantee is added.

The worker writes the immutable complete hash chain and verifies its contiguous
existence before returning the final ACK. The owner checks the acknowledged
token count against the required prefix; enqueue, an ACK alone, or successful
individual batches cannot certify destination readiness. A short write reports
actual verified prefix coverage and preserves the GPU source. MOVE releases
its GPU hold only after full required storage publication, then uses normal
eviction eligibility; another request's reference produces PARTIAL/source_busy.
The intermediate persistent-host copy may remain after a GPU→storage MOVE.
Cold pages remain independently evictable by the storage backend after this
completion boundary. Cancellation never deletes shared cold data.

Only active native writes retain storage ACK facts, bounded by active control
limits. Ordinary ACKs do not accumulate native history. Source paths can split
during writes without changing the captured worker buffers or logical coverage.
Runtime storage attach/detach refuses active native operations; idle changes
refresh adapter capabilities. Serving namespaces apply to the actual hash
chain, including ordinary requests and parameter generations.

On Linux/NVIDIA CUDA, with one visible GPU and TP=PP=CP=DP=1, the tiny manual
probe uses real pools and a temporary file backend without model downloads or
credentials:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 python \
  test/manual/entrypoints/programmable_kv_hicache_smoke.py --storage
```

The A100 FP16 direct-I/O fixture verifies nonzero file bytes, waits for storage
completion before source reclaim, evicts both local tiers, poisons free GPU
rows and restores from the file backend, then checks locks/jobs/allocator
balance. It does not measure serving throughput or model quality.


### Exact restoration from file storage

COPY storage→host and storage→GPU use real asynchronous HiCache lookup,
hit-sized host allocation, file reads and host insertion. A token target may
have no remaining physical radix nodes. Existing resident endpoints still need
to be radix boundaries; general ranges and multi-branch cold restoration are
unsupported. A destination already published for the complete scope satisfies
COPY without another cold read. Missing tails are queried under the actual
namespace and layout hash chain; a generation change cannot load old pages.

Native reads override the ordinary prefetch size threshold for their own
operation only. They do not use ordinary best-effort/timeout termination. An
entire missing host scope must fit before lookup, and GPU destinations also
need the owner's current copy budget. Capacity waits retain no buffers or
correctness references. Different cold scopes serialize and re-evaluate existing
host matches to avoid charging shared prefixes twice. Identical destination
loads coalesce with independent interests. Actual hit allocation refuses a
partial capacity allocation if headroom changed while the lookup was pending.

Storage lookup completion, partial progress, cancellation flags and installed
GPU indices do not certify final readiness. Query-only rejection/miss retires
when the query result is consumed. Submitted reads wait for completed_req ACK,
which follows all worker access. The existing cache insertion publishes actual
completed coverage. A native callback holds it before ordinary admissions can
evict it; GPU restoration then uses the existing host transfer path and prefill
barrier until its real fence. These internal holds end at operation completion;
COPY alone does not hold a prefix until a future request is admitted.

Cancellation of the last reader drains disk/DMA access and unused scratch-page
releases. Failed ACK causes survive late cancellation and acknowledgement.
Partial file reads report actual published destination coverage and do not
start a GPU load for an incomplete host scope. Native accounting/retry entries
are removed when the job retires.


The server smoke probe accepts `--storage` (implies `--hicache`) with the same
Linux/CUDA, one-GPU topology and model-path/load-format arguments described
above. It stores an actual generated prefix in files, flushes physical GPU/host
state without clearing storage, discovers the new epoch, restores independently
through `/v1/kv/control`, then verifies subsequent generation reuses that KV.
The OPT-125m dummy-weight probe reports three cached input tokens and identical
output IDs. Real nonzero low-level KV byte checks complement that serving
interface proof.
