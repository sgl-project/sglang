# Engine selection: target design

This document describes the target architecture for engine selection. It defines
responsibilities and behavior; the interface and configuration examples are
sketches, not a specification of the current API or CLI. Implementation status is
listed at the end.

## Principles

1. **Order compatible buckets by token length first.** `BucketResolver` returns
   all matching buckets, smallest capacity first, without inspecting workers or policies.
2. **The bucket owns plain versus PD engine selection.** `Bucket::pick_engines`
   calls its one plain group or both prefill and decode groups, returning a
   complete selection. Both PD engines come from that same bucket.
3. **An engine group owns membership and its policy.** `EngineGroup::pick`
   filters live workers by model, health, stage, and membership, invokes the
   policy, and validates that its result belongs to the exact candidate set.
4. **A policy owns selection, fallback, admission, and its state dependencies.**
   Construction injects the load, KV, or affinity handles it needs. Request-time
   arguments contain facts and candidates. A policy cannot choose another bucket
   or cross a PD role boundary.
5. **The handler owns bucket fallback and dispatch.** It calls `pick_engines`
   on each bucket and dispatches only after a complete selection succeeds.
   Missing candidates or admission rejection advance to the next bucket, where
   all required engines are selected again. Invalid signals or policy results stop routing.

An engine is represented by `Worker`. Registry role labels remain authoritative
when filtering candidates. Groups reference worker IDs rather than owning live
workers. Group policy instances are reused across requests.

### Data ownership

| Type | Owns |
| --- | --- |
| `BucketResolver` | A model's bucket collection, length filtering, and ordering |
| `Bucket` | ID, length constraints, rank, groups, and complete plain/PD engine selection |
| `EngineGroup` | Engine membership, attached policy, and engine selection |
| `WorkerRegistry` | Live workers, model membership, health, and role |
| Request handler | Request preparation, ordered bucket attempts, HTTP errors, and dispatch |

`worker_ids: None` means every healthy engine serving the requested model and
role. An explicit empty set means no engines. `EngineGroup::new(policy)` creates
a catch-all membership group. `BucketGroups::Pd` requires both groups, making
partial or mixed plain/PD bucket configurations unrepresentable.

## 1. Code organization

```text
src/
  buckets.rs                    Bucket resolution and engine-group selection
  policies/
    mod.rs                      Policy contract and construction
    admission.rs                Per-engine acceptance checks
    cache_aware.rs               Cache selection and local/remote adapter
    session_aware.rs             Session selection and fallback
    sticky.rs                   Routing-key selection and fallback
    least_load.rs               Least-load selection
    power_of_two.rs              Pair sampling and stage-aware comparison
    random.rs                   Random selection
    round_robin.rs              Rotation
  state/
    mod.rs                      Shared exports
    kv_events/                  Local index, subscriptions, hashing, wire format
    load_monitor/               Engine reports and router-local request accounting
    affinity_store.rs           Assignments, expiry, and atomic updates
  server/
    app_context.rs              Shared service lifecycle and policy wiring
    routes/chat.rs              Request preparation, PD coordination, dispatch
```

During migration, `buckets_reorg.rs` and `policies_reorg/` implement this design
beside the live `policies/` path. The layout above is the target after switchover.
Shared state already lives directly under `src/state/`.

Dependencies flow from engine groups to policies, and from policies and admission
to shared state. State does not depend on bucket ordering or concrete policy
strategies. Small shared helpers are sufficient; no generic score-composition,
tier executor, or separate selection framework is required.

## 2. Responsibilities and request flow

`BucketResolver::resolve(input_tokens, expected_peak_tokens)` returns an ordered
list of compatible bucket references (possibly empty), or an invalid-signal error.
It does not receive a stage or a load view, resolve live engines, or invoke policies.
The handler iterates this list until a bucket supplies the complete engine selection.

`Bucket::pick_engines(workers, request)` accepts a `BucketRequest`
of prepared routing facts, invokes the required groups, and returns `BucketPick`
(one plain pick or a complete P/D pair). Failures retain their stage. This API
has no HTTP headers, `AppContext`, or forwarding dependency.

`EngineGroup::pick(workers, request)` resolves healthy workers for the request's
model and stage, intersects them with its membership, sorts them by stable ID,
and invokes its attached policy. It rejects foreign results, including a newly
allocated worker with the same ID as a candidate.

```text
chat_completions (reorg configured): prepare tokens and expected peak
  |
  v
BucketResolver::resolve: ordered length-compatible buckets
  |
  v
For each bucket: bucket.pick_engines(...)
  |
  +-- BucketGroups::Plain
  |     plain.pick() -> one plain engine
  |
  +-- BucketGroups::Pd
        prefill.pick() -> P engine
        decode.pick()  -> D engine from the same bucket
  |
  +-- empty group / admission rejection -> try next bucket (repeat all picks)
  +-- invalid signal / configuration / foreign pick -> return error
  |
  v
Complete selection -> forward_chat_request: acquire guards, attach PD bootstrap, forward response
```

The handler extracts token facts and header keys once into `BucketRequest`.
The bucket creates a stage-specific `PickRequest` for each group call, supplying
its own ID and the role associated with that group. Input length, expected peak,
token IDs, and session/routing keys pass through. Policies obtain observations
from their own shared-state handles; buckets and handlers do not provide load,
KV, or affinity services on each call. PD uses separate group policies, but never
independently resolves a decode bucket. Decode selection
failure discards that tentative prefill choice and advances to the next bucket
on missing candidates or admission rejection. No forwarding guards are acquired
and no prefill request is sent until both picks in one bucket succeed.
PD compatibility constraints beyond model and role remain follow-up work.

There is one endpoint: `POST /v1/chat/completions`. `AppContext::chat_routing`
chooses its implementation:

- `ChatRouting::Legacy` (default) uses the existing policies and bucket selector.
- `ChatRouting::Reorg(HashMap<ModelId, BucketResolver>)` uses the new bucket and
  policy interfaces, with explicit model-specific resolvers.

Callers set this field before building the router. A missing model in the reorg
map returns 404, without falling back to legacy routing. This PR adds the
programmatic configuration switch; CLI/configuration factory construction and
the remaining production policies remain follow-ups. Power-of-two is implemented
for explicit attachments; the default serving path remains legacy.

Both implementations reuse request preparation (including sampling validation
and tokenization), forwarding, streaming, middleware, and the 32 MiB body limit.
The reorg implementation requests tokenization for length matching, retaining
the existing body-size estimate when tokenization is unavailable.

## 3. Bucket resolution

1. Validate that a known expected peak is at least the input length.
2. Keep buckets whose inclusive input-token range contains the input length.
3. Check the bucket context capacity against input plus requested output when
   known, or against input length when the output budget is unknown.
4. Sort by ascending input capacity (the lesser of the input upper bound and
   context capacity). Unbounded capacities sort last. Break ties by ascending
   bucket rank, then ID, and return the entire ordered list.
5. The handler calls each bucket's `pick_engines` until one supplies its complete
   selection. A failed PD attempt never contributes an engine to a later pair.

The handler checks addition overflow when computing the expected peak.
An empty bucket list becomes a 400 `NoMatchingBucket` response. After exhausting
the list, accumulated admission rejection details produce a selection failure
(503); if there were no admission rejections, the last unavailable stage produces
a stage-specific 503. Invalid policy signals/configuration or out-of-candidate
picks stop the pass immediately with an internal error. Successful engine
selection ends the pass; forwarding errors do not restart bucket iteration.

Token ranges and rank belong to the bucket, not its engine groups. For a PD
bucket, both groups share this one request-length decision. Policy fallback on
a cache/affinity miss stays within that group's candidates. There is no second
pass with relaxed admission and no post-policy substitution.

SLO ordering, global session modes, and sticky policies are follow-ups. Their
integration must preserve bucket-first selection and the same-bucket PD rule.
Cross-bucket affinity probing is not part of this interface. Session/routing
keys still pass through `PickRequest` for policies operating inside the selected
group; unsupported legacy modes need explicit migration decisions before the
standard serving path switches.

## 4. Policy and admission contracts

### Policy

All policies implement one asynchronous, object-safe `Policy::pick` interface.
A boxed future supports the remote prefix indexer; local policies can return an
immediately ready result.

```rust
pub trait Policy: Send + Sync + std::fmt::Debug {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>>;
}
```

Policies own their required state handles and read observations locally. Nested
fallback uses `pick_fallback(engines, request)`, which calls the fallback's `pick`;
the fallback reads its own state. There is no shared observation context or cache.
Buckets and HTTP handlers supply only candidates and request facts.

`Pick` identifies one engine and a selection reason for metrics and tracing.
`PickRequest` carries model, stage, selected bucket ID, input and optional
expected peak counts, optional token IDs, and session/routing keys. It contains
no HTTP body, bucket resolver, state handles, snapshots, or backend configuration.

| Outcome | Meaning |
| --- | --- |
| `Pick` | The chosen engine belongs to the supplied set and passed admission |
| `NoMatchingBucket` | No bucket supports the requested length |
| `NoCandidates` | No eligible member or policy selection miss |
| `NoAdmissibleEngine` | A policy exhausted its candidates; includes rejection reasons |
| `AdmissionRejected` | The chosen engine failed after-selection admission |
| `InvalidSignal` / `InvalidConfiguration` | Invalid policy input or configuration |
| `OutsideCandidates` | Policy returned an engine outside its exact candidate set |

### Admission

Admission evaluates acceptance. It does not rank engines, choose replacements,
change buckets, or mutate affinity.

An admission policy is a set of per-engine caps, `AdmissionLimits`. Each cap
is optional; an unset cap is not checked, and the default admits everything.
A cap admits while the engine's current metric is below it. Request size is
not part of admission: buckets already select by input length and context
capacity, and admission only observes load without reserving it.

| Limit | Engine metric |
| --- | --- |
| `max_running_requests` | Reported running requests |
| `max_waiting_requests` | Reported waiting requests |
| `max_kv_tokens` | Reported total KV tokens |
| `max_pending_prefill_tokens` | Reported waiting uncached tokens |
| `max_inflight_requests` | Router-local in-flight requests |

```json
{"max_running_requests": 64, "max_kv_tokens": 1048576, "max_inflight_requests": 64}
```

Limits are absolute caps; they do not default to capacities reported by the
engine. Unknown fields are rejected during deserialization.

The policy reads the selected engine's `EngineMetrics` from the load snapshot
it already captured for selection plus the live in-flight counter and calls
`EngineAdmission::check(engine, metrics)`, which returns `Allow`,
`Reject(limit name)`, or an error. Reported
metrics are `None` without a fresh, complete report, never zero, and such
limits fail open; the in-flight count is always known. Policies attach the
checker as `Arc<dyn EngineAdmission>` and decide where checking belongs in
their selection algorithm; there is no placement setting or filtering wrapper.

Power-of-two first selects an engine, then calls admission exactly once on that
engine. A rejection returns `AdmissionRejected` to the bucket loop; it does not
resample, choose the other sampled engine, or run a policy fallback. No candidates
returns `NoCandidates` without invoking admission. A single candidate is selected
directly; otherwise two distinct candidates are sampled uniformly, and the one
with lower stage pressure wins. A complete tie keeps the first sampled engine.

Power-of-two reuses the existing pure pressure-comparison functions. Plain and
prefill stages compare estimated prefill queue time when both reports provide it,
then waiting uncached tokens, waiting requests, and running requests. Decode
compares waiting requests, running requests, KV usage fraction, then used KV tokens.
Reported-pressure ties use router-local active requests. If either sampled engine
lacks a fresh, complete native report with valid capacity, both are compared by
router-local active requests instead. Basic reports from older publishers are
still passed to admission when fresh, but do not supply native pressure metrics.

Power-of-two retains the selected engine's load record from selection and
passes it to admission without another snapshot. A single candidate still has
its load read for admission, even though selection needs no comparison.
Neither the bucket nor HTTP handler supplies observations. Synchronous checks
do not fetch telemetry over the network themselves.

`AdmissionLimits::default()` is the default for new explicit policy attachments.
It leaves health, role, membership, and policy preferences in force. Migrated
configurations must retain their existing capacity and configured budget checks;
see compatibility below.

The cache policy's `worker_queue_limit` is a **soft preference**;
`max_waiting_requests` is a hard rejection. Saturation handling can reconsider
a queued engine, but cannot bypass attached hard admission.

Admission checks observe capacity; they do not reserve it. Concurrent requests
may pass against the same observation. Strict reservations would require a
separate mechanism.

## 5. Concrete policies

| Policy | Selection behavior |
| --- | --- |
| `RoundRobinPolicy` | Rotate over candidates using a cursor owned by this policy instance |
| `RandomPolicy` | Choose uniformly from candidates |
| `PowerOfTwoPolicy` | Sample two distinct candidates when possible and choose the lower-pressure engine using the stage's load comparison |
| `LeastLoadPolicy` (`load_based`) | Choose the least loaded engine; preserve tie-breaking, telemetry fallback, and recent-dispatch correction |
| `SessionAwarePolicy` | Reuse an admitted session binding; use power-of-two for new or keyless sessions |
| `StickyPolicy` | Reuse an admitted routing-key binding; use the configured fallback for new or missing keys |
| `CacheAwarePolicy` | Prefer a usable prefix under cache and pressure rules; use a load-based fallback on a miss |

Session assignments are scoped by model, bucket ID, stage, and session key.
`SessionAwarePolicy::new(store, engine_load)` receives shared state; the caller
owns the store's idle timeout and eviction task. Missing or empty session keys
use power-of-two without creating assignments. A new or out-of-group binding
uses power-of-two with `AdmissionLimits::default()`, then the session policy
checks its selected engine before binding. A concurrent live assignment wins,
but is checked before returning it; rejection ends that attempt without
rewriting the binding or retrying another engine. Existing bindings are reused
regardless of pressure when admitted. Session policies can be attached
independently to each role. Programmatic reorg callers configure
`model.affinity.session_id_header` for HTTP header extraction; this does not
enable legacy global modes or backup escape.

Session and sticky policies do not create assignments for missing keys. A
binding outside the candidates cannot win. A missing binding may invoke policy
fallback within the group; hard admission rejection remains an error.

Sticky fallback supports `round_robin`, `random`, `power_of_two`, and `load_based`,
with round-robin as the default. Nested fallbacks use
`AdmissionLimits::default()`; the owning policy explicitly checks the engine
returned by its fallback.

### Cache-aware behavior

The architecture preserves the cache algorithm within the supplied candidate
set. Its responsibilities are:

1. Look up prefix ownership through the local index or remote indexer.
2. Apply minimum matched-token and optional ratio thresholds.
3. Bound candidates using prefix/pressure ordering and the configured minimum,
   ratio, and maximum worker counts.
4. Apply the soft queue gate and saturation rules, and call admission explicitly
   as required by the cache policy's candidate-selection algorithm.
5. Choose among usable prefix holders using uncached work, the switch margin,
   and the pressure guard.
6. On a miss, run the load fallback, preferring engines admitted by the soft
   queue gate when available.

Candidate limits and saturation observations use only the selected bucket's
role-group candidates.
Saturation pinning must still pass hard admission.

The target load fallback supports power-of-k sampling through
`--min-load-choices`, default 2. When k covers the group, choose the exact minimum.
Preserve queue-tier preference and avoid sorting with a pairwise pressure
comparator that does not define a total ordering.

Memoize the prefix lookup once per request, including remote I/O. Each policy
restricts those matches to its own candidates. A memoized lookup does not imply
that another bucket or stage's cache selection and admission have already run.
Any optimization that skips those steps must establish that the previous result
applies to the current group.

## 6. Shared state and construction

Application wiring starts shared services once. Policy construction validates
configuration and passes the required handles to each policy and admission
implementation. For example, `PowerOfTwoPolicy::new(Arc<EngineReportedLoadTable>)`
retains the application's shared load table. KV-aware and affinity-aware policies
receive their corresponding shared handles when implemented. Policies with no
state dependency require none. Policy instances do not create duplicate
subscriptions, polling loops, indexes, or remote-client concurrency limits.

Requirements come from all configured policies, their admission checks, and
nested fallbacks. This includes tokenization, affinity-header extraction, load
observations, and dispatch timestamps. A role-group override must receive its
required settings even when the model's default uses a different policy.

### Load state

`state/load_monitor/` owns engine reports and existing router-local request
accounting. Power-of-two owns an `Arc<EngineReportedLoadTable>` and captures a snapshot
locally for each nonempty selection attempt. Its `pick` method selects the engine,
then passes that engine's borrowed load record directly to admission.
No snapshot or observation is added to `Pick`, `PickRequest`,
or the bucket interface, and no shared observation context is threaded through
policies or fallbacks.

The existing snapshot reader preserves rank aggregation, freshness, and capacity
fields. Missing, stale, or rank-incomplete reports yield `None`, not zero load.
A new pick reads current state. Admission reuses the selected observation even if
reports change after selection; it neither recaptures nor reserves capacity.
Fallback policies read their own state and do not share snapshots with callers.

Snapshot capture still scans the full table; an engine-scoped reader can be added
if profiling justifies it. Power-of-two reuses the legacy prefill/decode pressure
comparisons, including router-local fallback. Concrete load-aware admission remains
in #40271. Further shared load interpretation and correction for dispatches since
the report remain follow-ups; these must preserve source, freshness, and available
measurements without adding another
independent in-flight counter. Load and cache observations are not an atomic global
snapshot. Preserve request-guard cleanup.

### Cache state

`state/kv_events/` owns local subscriptions, event application, hashing, and the
radix-tree index. The tree stores prefix ownership and storage tiers, not KV
tensors. Eviction, invalidation, and worker removal update this shared state.

A small `CacheSource` adapter beside the cache policy normalizes local and remote
lookups into common prefix results. Preserve model/cache namespaces, required
rank information, and hash/block-size handling. Convert block counts to token
counts only with a valid conversion.

A confirmed miss and an unavailable backend both allow cache fallback, but
remain distinguishable in diagnostics. Invalid queries and configuration errors
remain explicit errors. Remote indexing and load-only deployments do not need
a duplicate local cache tree.

### Affinity state

`AffinityStore` owns scoped assignments, idle expiry, worker invalidation, and
atomic updates. Policies decide when to reuse or replace an assignment.

Concurrent first assignments must converge on an effective binding that is
still a candidate and passes admission. If a concurrent binding returns a
different engine, the policy must check that engine before returning it.
Reconcile conflicts with bounded retry.

Create or replace a binding only after admission succeeds. A binding records
preferred placement, not successful execution, so it may remain if later PD
selection or dispatch fails. It must not increment dispatch accounting.

## 7. Configuration and compatibility

This conceptual example shows the target attachment model. It is not copyable
current CLI/JSON syntax; existing bucket field names need not change.

```yaml
buckets:
  - id: short-context
    rank: 10
    limits: {min: 0, max: 4096}
    max_context_tokens: 8192
    groups:
      pd:
        prefill:
          worker_ids: [P1, P2]
          policy:
            type: cache_aware
            admission: {max_running_requests: 64, max_kv_tokens: 1048576}
        decode:
          worker_ids: [D1, D2]
          policy:
            type: power_of_two
            admission: {max_running_requests: 64, max_kv_tokens: 1048576}

  - id: long-context
    rank: 20
    limits: {min: 0, max: 131072}
    max_context_tokens: 131072
    groups:
      pd:
        prefill:
          worker_ids: [P3, P4]
          policy:
            type: cache_aware
            admission: {max_running_requests: 64, max_kv_tokens: 1048576}
        decode:
          worker_ids: [D3, D4]
          policy:
            type: power_of_two
            admission: {max_running_requests: 64, max_kv_tokens: 1048576}
```

A request with 4k input tokens and a 16k expected peak cannot fit the short
bucket's context capacity. It selects the long bucket and both of its P/D groups.
With a known peak of 8k or less, the same input selects both groups of the short bucket.

The planned factory validates unique nonempty bucket IDs, token ranges, and
role-compatible membership. Each bucket is either plain or PD. The selected
bucket invokes its required groups through `pick_engines`. The existing worker registry
still rejects mixed plain and PD engines within one model; this PR preserves
that constraint. The engine group's model and stage filters apply on every pick.

Legacy `BucketSpec` represents a single role-specific membership set. Migration
must explicitly associate prefill and decode specs into complete PD buckets;
never infer those associations from matching rank or similar names. Translation
of role-specific ranges/ranks into bucket-level constraints needs explicit
validation and is deferred with the configuration factory.

Retained settings keep their meanings, defaults, units, and validation unless a
change is listed below. Policy-specific tuning applies to role groups using that
policy. Reject unsupported settings and incompatible combinations at startup;
do not accept and ignore them.

### Retained behavior

- Keep `load_based` as the CLI name for `LeastLoadPolicy`.
- Preserve explicit engine membership and context constraints. Bucket-level
  ranges and ordering replace independent per-stage selection.
  Restore existing SLO behavior in the separate SLO PR before serving switchover;
  legacy routing continues to support SLOs during this skeleton-only phase.
- Preserve cache-provider selection, endpoint validation, query timeout and
  concurrency limits, and unavailable-backend fallback.
- Preserve cache thresholds and tuning: the 1,024-token default minimum hit,
  optional ratio gate, candidate bounds, switch margin, pressure guard, soft
  queue limit, and saturation floor.
- Preserve session and sticky headers, idle timeouts, eviction cadence, and the
  four sticky fallback choices. Global modes need a bucket-first migration design.
- Map `--filter overloaded` and `--max-in-flight` to `max_inflight_requests`;
  the existing router-local counter remains the source.
- Preserve configured capacity, pending-prefill, and in-flight checks, including
  their missing-report behavior. Power-of-two applies admission to its selected
  engine; other policies explicitly place checks in their selection logic.
  Other paths use `AdmissionLimits::default()` unless a check is configured.
  Never silently discard a configured budget.

Listener and shutdown configuration, discovery, worker health and circuit
breakers, tokenizer loading, request timeouts, sampling overrides, and logging
remain outside the policy redesign. Preserve their existing behavior and
validation, including dispatch-time breaker probes and request cancellation.

### Deliberate changes

| Behavior | Target |
| --- | --- |
| Power-of-two admission | Check only the chosen engine; rejection advances to the next bucket |
| Round-robin cursor | One cursor per role-group policy instance |
| Capacity exhaustion | Try the next compatible bucket; return accumulated rejection details if all fail |
| Primary/backup proposals and post-policy substitution | Removed; each policy returns one engine |
| Session affinity | Reuse admitted bindings; remove primary/backup pressure escape |
| Omitted `--affinity-mode` | Admitted-binding reuse replaces the former soft-mode default |
| Pressure-guard tuning | Applies to cache-aware selection; reject session-only use |
| Policy attachment | Explicit role-group policy overrides the applicable model/stage default |

Reject these dropped options explicitly:

- `--policy fused_score` and `--policy score_policy`, including `--fuse` terms
  and weights.
- `--decode-policy legacy_host_affinity`.
- `--stable-pair`.
- `--affinity-mode soft`; only strict admitted-binding reuse remains.
- `--filter prefix_cache` and `--prefix-cache-min-share`. The removed prefix-share
  filter is not equivalent to the cache-aware minimum-hit gate.

## 8. Dispatch and failure handling

A successful pick means admission passed against the observed state. Health and
capacity can change before dispatch. The handler owns network operations, retry
rules, and accounting for the engines actually dispatched to.

Selection fallback advances through the ordered compatible buckets before any
network dispatch. Every PD attempt picks both engines from that bucket. Transport
retry integration remains separate work; a forwarding failure does not resume
the bucket loop or silently replace a successful policy pick.

For PD, acquire and release accounting for the actual stages and clean up
partial setup on failure. Policy selection does not own the PD request lifetime.
Selection metrics must not imply that dispatch or execution succeeded.

## Implementation status

This PR adds the side-by-side interfaces in `src/buckets_reorg.rs` and
`src/policies_reorg/`, plus a configurable bucket-first implementation behind
`chat_completions`. The live `src/policies/` path remains the default.

Implemented here:

- `BucketResolver::resolve` returns all length-compatible buckets in capacity/rank/ID order.
- `Bucket::pick_engines` owns plain/PD orchestration and stage-specific policy
  requests; `BucketRequest` carries prepared facts and `BucketPick` retains picks.
- `Bucket` owns input limits, context capacity, rank, and plain-or-PD groups.
- `EngineGroup::pick` owns live candidate filtering, policy invocation, and
  exact candidate validation, without cross-bucket fallback.
- `Policy::pick`, within-group fallback interface, per-engine `EngineAdmission::check`,
  and `AdmissionLimits` over running, waiting, KV, pending-prefill and in-flight
  metrics. Power-of-two samples two distinct engines, compares stage pressure,
  and checks its selected engine with no replacement on rejection.
- Policy-owned load dependency and local observations. Power-of-two passes the
  selected engine's load record directly to admission, without another snapshot.
  `PickRequest`, `Pick`, and bucket APIs carry no load observations.
- The reorg chat implementation iterates resolved buckets, calls `pick_engines`,
  advances on empty candidates/admission rejection, and
  dispatches only after one complete selection. Exhaustion retains admission reasons.
- `AppContext::chat_routing` configures legacy versus reorg routing on the same
  endpoint and carries the reorg model-resolver map.
- `CacheAwarePolicy` reads local radix-tree or remote indexer prefixes, intersects
  exact worker URLs with the current group, applies hit thresholds and candidate
  bounds, and preserves the soft queue gate, saturation pin and pressure guard.
- `PrefixMemo` shares lookup results (including misses and unavailable backends)
  across bucket attempts for one prepared request. Entries are keyed by the shared
  `Arc<CacheSource>` so different index namespaces remain independent. Each pick
  reruns its own candidate filtering and admission after obtaining a fresh snapshot.
- Cache selection checks bounded candidates explicitly; hard rejection cannot
  become a cold fallback or bypass admission through saturation pinning. A miss
  defaults to power-of-two within the group's soft queue tier, then the cache
  policy checks its fallback winner. Cache policies require plain/prefill groups.
- `SessionAwarePolicy` reuses admitted model/bucket/role-scoped bindings from a
  shared `AffinityStore`, falling back to power-of-two for new or keyless sessions.
  Assignments follow admission; concurrent binding winners are rechecked.
  Rejection preserves existing bindings and advances to the next bucket.
  The caller owns expiry and sweeper lifecycle. A binding may remain after a
  later PD group fails, because it records placement rather than dispatch.

Follow-up work includes bucket SLO ordering, remaining selection policies,
and production configuration.

Not yet implemented in the reorg path:

- Other concrete selection policies.
- SLO estimates, targets, and bucket preference ordering.
- CLI/configuration parsing, validation, and model-specific construction.
  The YAML above is illustrative; reorg resolvers are installed in code.
- Global session modes and sticky routing-key affinity.
- Power-of-k cache-miss fallback configuration and cache decision metrics.
- Shared load interpretation, dispatch correction, and policy-specific
  dispatch-timestamp requirements.
- PD compatibility filtering, retry integration, and legacy-route switchover.

The preceding policy sections describe target behavior for those follow-ups;
they do not claim those capabilities are present in this PR.
