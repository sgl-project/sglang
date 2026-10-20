# Programmable KV control core (unit 1)

This package implements CPU-only native control contracts and logical lifecycle.
`KVControl` is disabled by default. Nothing imports it from serving entry points,
and importing it installs no scheduler hooks or process-global state. Enabling
this core executes explicit logical commands; there are no configured policies.
With no executor, cache primitives return `unsupported`. There is no cache
allocation, priority application, transfer, residency database, or GPU work here.

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
hits/splits cannot renew it and reconstructed copies cannot inherit it. These
copy-lifetime rules are obligations for unit 2's executor, not simulated claim
state in this package.

Targets use tagged objects:

| `kind` | Fields | Resolution |
| --- | --- | --- |
| `tokens` | `namespace`, `token_ids` | Exact token prefix from position zero; physical component/layout information remains unknown |
| `content` | `namespace`, `token_count`, `components`, `spans` | Explicit logical descriptors, including gaps in known history |
| `session` | `namespace`, `session_id`, optional `continuation_id`, `expected_revision` | One explicit continuation or all tracked continuations in that session |

All three accept `start_token=0`, optional exclusive `end_token`, and
`allow_clip=False`. A `ContentSpan` has `content_id`, `component`, `start_token`,
`end_token`, and `dependency_start=0`. Component names identify the namespace's
layout; a name such as `mamba` does not advertise executable Mamba support.
Descriptors are indivisible and nonoverlapping within a component. Resolution
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

Unit 2 still supplies serving configuration, IPC/API entry points, request
ownership, capability discovery, actual copy-scoped bounded claims and eviction
integration. Unit 3 supplies transfers, allocator/hold accounting, shared-load
joining, whole-scope preparation and all-rank publication. Policies and event
feeds are later units. No serving, reuse, performance or GPU validation is claimed
by the CPU suite.

Run from the repository root:

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest test/registered/unit/managers/test_kv_control.py -q
```

Tests use the repository's registered pure-pytest CPU pattern and real core state.
They do not require `CustomTestCase`, import stubs, a model, or backend services.
