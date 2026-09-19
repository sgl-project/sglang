# Engine selection: target design

This document describes the target architecture for engine selection. It defines
responsibilities and behavior; the interface and configuration examples are
sketches, not a specification of the current API or CLI. Implementation status is
listed at the end.

## Principles

1. **Choose one bucket by token length first.** `BucketResolver` selects the
   smallest compatible bucket without inspecting worker availability or policies.
2. **The selected bucket determines plain versus PD serving.** `BucketGroups`
   contains either one plain `EngineGroup`, or both prefill and decode groups.
   Both PD engines come from that same selected bucket.
3. **An engine group owns membership and its policy.** `EngineGroup::pick`
   filters live workers by model, health, stage, and membership, invokes the
   policy, and validates that its result belongs to the exact candidate set.
4. **A policy owns selection, fallback, and admission within its candidates.**
   It cannot choose another bucket or cross a PD role boundary.
5. **The handler coordinates stages and dispatch.** A plain request needs one
   pick; a PD request needs both picks before either engine is dispatched.
   Missing candidates and admission failures return errors without changing buckets.

An engine is represented by `Worker`. Registry role labels remain authoritative
when filtering candidates. Groups reference worker IDs rather than owning live
workers. Group policy instances are reused across requests.

### Data ownership

| Type | Owns |
| --- | --- |
| `BucketResolver` | A model's bucket collection and length-based selection |
| `Bucket` | ID, input-token limits, context capacity, tie-break rank, and plain or PD groups |
| `EngineGroup` | Engine membership, attached policy, and engine selection |
| `WorkerRegistry` | Live workers, model membership, health, and role |
| Request handler | Prepared request facts, group calls for the selected mode, and dispatch |

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
    admission.rs                Acceptance checks and placement
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
    load_view.rs                Lazy load snapshot for a selection pass
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

`BucketResolver::resolve(input_tokens, expected_peak_tokens)` returns one bucket
reference or `NoMatchingBucket`. It does not receive a stage or a load view and
does not resolve live engines or invoke policies.

`EngineGroup::pick(workers, request)` resolves healthy workers for the request's
model and stage, intersects them with its membership, sorts them by stable ID,
and invokes its attached policy. It rejects foreign results, including a newly
allocated worker with the same ID as a candidate.

```text
chat_completions (reorg configured): prepare tokens and expected peak
  |
  v
BucketResolver::resolve: choose best length-compatible bucket
  |
  +-- BucketGroups::Plain
  |     plain.pick() -> one plain engine
  |
  +-- BucketGroups::Pd
        prefill.pick() -> P engine
        decode.pick()  -> D engine from the same bucket
  |
  v
forward_chat_request: acquire guards, attach PD bootstrap, forward response
```

The handler creates a fresh `LoadView` and stage-specific `PickRequest` for each
group call. Each request carries the selected bucket ID, input length, optional
expected peak, token IDs, and session/routing keys. PD uses separate group
policies, but never independently resolves a decode bucket. Decode selection
failure happens before forwarding acquires accounting guards or sends prefill.
PD compatibility constraints beyond model and role remain follow-up work.

There is one endpoint: `POST /v1/chat/completions`. `AppContext::chat_routing`
chooses its implementation:

- `ChatRouting::Legacy` (default) uses the existing policies and bucket selector.
- `ChatRouting::Reorg(HashMap<ModelId, BucketResolver>)` uses the new bucket and
  policy interfaces, with explicit model-specific resolvers.

Callers set this field before building the router. A missing model in the reorg
map returns 404, without falling back to legacy routing. This PR adds the
programmatic configuration switch; CLI/configuration factory construction and
concrete production policies remain follow-ups. No default attaches the
power-of-two placeholder to serving.

Both implementations reuse request preparation (including sampling validation
and tokenization), forwarding, streaming, middleware, and the 32 MiB body limit.
The reorg implementation requests tokenization for length matching, retaining
the existing body-size estimate when tokenization is unavailable.

## 3. Bucket resolution

1. Validate that a known expected peak is at least the input length.
2. Keep buckets whose inclusive input-token range contains the input length.
3. Check the bucket context capacity against input plus requested output when
   known, or against input length when the output budget is unknown.
4. Choose the smallest compatible input capacity (the lesser of the input upper
   bound and context capacity). Unbounded capacities sort last. Break ties by
   ascending bucket rank, then ID.
5. Inspect that bucket's `BucketGroups` and select its required engine(s).

The handler checks addition overflow when computing the expected peak.
`NoMatchingBucket` becomes a 400 response; missing group engines become a
stage-specific 503. Admission rejection remains a selection failure (503), and
invalid policy signals/configuration or out-of-candidate picks are internal
errors. No error causes automatic selection from another bucket.

Token ranges and rank belong to the bucket, not its engine groups. For a PD
bucket, both groups share this one request-length decision. Policy fallback on
a cache/affinity miss stays within that group's candidates. There is no second
pass with relaxed admission and no post-policy substitution.

SLO ordering, cache lookup, and session/sticky policies are follow-ups. Their
integration must preserve bucket-first selection and the same-bucket PD rule.
Cross-bucket affinity probing is not part of this interface. Session/routing
keys still pass through `PickRequest` for policies operating inside the selected
group; unsupported legacy modes need explicit migration decisions before the
standard serving path switches.

## 4. Policy and admission contracts

### Policy

All policies implement one asynchronous, object-safe interface. A boxed future
supports the remote prefix indexer; local policies can return an immediately
ready result.

```rust
pub trait Policy: Send + Sync + std::fmt::Debug {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>>;
}
```

`Pick` identifies one engine and a selection reason for metrics and tracing.
`PickRequest` carries model, stage, selected bucket ID, input and optional
expected peak counts, optional token IDs, session/routing keys, and a lazy load
view. It contains no HTTP body, bucket resolver, or backend configuration.

| Outcome | Meaning |
| --- | --- |
| `Pick` | The chosen engine belongs to the supplied set and passed admission |
| `NoMatchingBucket` | No bucket supports the requested length |
| `NoCandidates` | No eligible member or policy selection miss |
| `NoAdmissibleEngine` | Before-selection checks rejected all candidates |
| `AdmissionRejected` | The chosen engine failed after-selection admission |
| `InvalidSignal` / `InvalidConfiguration` | Invalid policy input or configuration |
| `OutsideCandidates` | Policy returned an engine outside its exact candidate set |

### Admission

Admission evaluates acceptance. It does not rank engines, choose replacements,
change buckets, or mutate affinity.

| Check | Acceptance rule |
| --- | --- |
| `AllowAll` | Add no acceptance constraint |
| `CapacityAdmission` | Projected running requests and KV tokens fit reported capacity |
| `PendingPrefillAdmission` | Waiting uncached tokens plus incoming uncached work fit the budget |
| `InFlightLimitAdmission` | Router-local in-flight requests are below the limit |
| `QueueLimitAdmission` | Engine-reported waiting requests are below the limit |
| `AllOfAdmission` | Every attached check allows the request |

One `EngineAdmission` interface supports checking an engine or a list of engines
against the same prepared context. Batch results preserve input order and carry
one decision per engine. Filtering cannot add candidates. Invalid inputs are
errors, distinct from an explicit acceptance rejection.

Each policy has an admission attachment and a placement:

| Placement | Execution | On rejection |
| --- | --- | --- |
| `BeforeSelection` | Prepare signals, filter candidates through admission, then select | Return `NoAdmissibleEngine` if none survive |
| `AfterSelection` | Prepare signals, select an engine, then check it | Return `AdmissionRejected`; do not silently try another engine |

Both placements return `NoCandidates` for an empty original set. If a cache
policy prefers A but only B passes admission, before-selection can choose B;
after-selection rejects A and the handler returns the selection failure.

Prepare the signals needed by admission before checking. A pending-prefill check
uses per-engine uncached work when a prefix is known, and full input otherwise.
Decode capacity uses the expected peak sequence length when available, including
on a cache hit. Admission and ranking share one load view for the selection pass.
Synchronous checks do not fetch telemetry themselves.

`AllowAll` is the default for new explicit policy attachments. It leaves health,
role, membership, and policy preferences in force. Migrated configurations must
retain their existing capacity and configured budget checks; see compatibility
below. Each check defines its missing-data behavior. Unknown load is not zero;
the existing capacity and pending-prefill checks allow requests without a fresh,
complete native report.

The cache policy's `worker_queue_limit` is a **soft preference**, not
`QueueLimitAdmission`. Saturation handling can reconsider a queued engine, but
cannot bypass attached hard admission.

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

Session and sticky policies do not create assignments for missing keys. A
binding outside the candidates cannot win. A missing binding may invoke policy
fallback within the group; hard admission rejection remains an error.

Sticky fallback supports `round_robin`, `random`, `power_of_two`, and `load_based`,
with round-robin as the default. Nested fallbacks use `AllowAll`; the owning
policy applies hard admission once at its configured placement.

### Cache-aware behavior

The architecture preserves the cache algorithm within the supplied candidate
set. Its responsibilities are:

1. Look up prefix ownership through the local index or remote indexer.
2. Apply minimum matched-token and optional ratio thresholds.
3. Bound candidates using prefix/pressure ordering and the configured minimum,
   ratio, and maximum worker counts.
4. Apply the soft queue gate and saturation rules, together with hard admission
   at its configured placement.
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
configuration and passes the required handles to each policy. Policy instances
do not create duplicate subscriptions, polling loops, indexes, or remote-client
concurrency limits.

Requirements come from all configured policies, their admission checks, and
nested fallbacks. This includes tokenization, affinity-header extraction, load
observations, and dispatch timestamps. A role-group override must receive its
required settings even when the model's default uses a different policy.

### Load state

`state/load_monitor/` owns engine reports and existing router-local request
accounting. `state/load_view.rs` provides a lazy wrapper over `EngineLoadTable`:
the first `snapshot()` call captures reports, and subsequent calls reuse them.
The skeleton does not yet implement shared load interpretation, local fallback,
or correction for dispatches since the report. Those follow-ups must preserve
source, freshness, and available measurements without adding another independent
in-flight counter.

The request handler creates a fresh `LoadView` for each stage's selection pass
and lends it through `PickRequest`. Reuse it across admission, selection, and
fallback in that pass; never store it on a bucket, group, or long-lived policy.
A retry needs a new view. The type itself does not enforce this lifecycle.
No snapshot is collected if no consumer reads it. Load and cache observations
are not an atomic global snapshot. Preserve report freshness, rank aggregation,
and request-guard cleanup.

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
still a candidate and passes admission. Before-selection requires membership in
the admitted subset; after-selection must check a different engine returned by
a concurrent binding. Reconcile conflicts with bounded retry.

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
            admission: {type: capacity, placement: before_selection}
        decode:
          worker_ids: [D1, D2]
          policy:
            type: power_of_two
            admission: {type: capacity, placement: before_selection}

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
            admission: {type: capacity, placement: before_selection}
        decode:
          worker_ids: [D3, D4]
          policy:
            type: power_of_two
            admission: {type: capacity, placement: after_selection}
```

A request with 4k input tokens and a 16k expected peak cannot fit the short
bucket's context capacity. It selects the long bucket and both of its P/D groups.
With a known peak of 8k or less, the same input selects both groups of the short bucket.

The planned factory validates unique nonempty bucket IDs, token ranges, and
role-compatible membership. Each bucket is either plain or PD. The selected
bucket determines which groups the handler invokes. The existing worker registry
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
- Translate `--filter overloaded` and `--max-in-flight` into
  `InFlightLimitAdmission`, composed with other checks through `AllOfAdmission`.
- When migrating existing configuration, attach before-selection capacity
  checks to plain/prefill power-of-two, session-aware, cache-aware, and decode
  power-of-two. Retain applicable pending-prefill and in-flight checks, including
  their missing-report behavior. Other paths use `AllowAll` unless a check is
  configured. Never silently discard a configured budget.

Listener and shutdown configuration, discovery, worker health and circuit
breakers, tokenizer loading, request timeouts, sampling overrides, and logging
remain outside the policy redesign. Preserve their existing behavior and
validation, including dispatch-time breaker probes and request cancellation.

### Deliberate changes

| Behavior | Target |
| --- | --- |
| Round-robin cursor | One cursor per role-group policy instance |
| Capacity exhaustion | Return the selected group's error; do not switch buckets |
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

A retry rebuilds candidates within the selected bucket, excludes failed engines
as required, and invokes the attached policy. It does not silently replace a cache winner
with a different engine after selection.

For PD, acquire and release accounting for the actual stages and clean up
partial setup on failure. Policy selection does not own the PD request lifetime.
Selection metrics must not imply that dispatch or execution succeeded.

## Implementation status

This PR adds the side-by-side interfaces in `src/buckets_reorg.rs` and
`src/policies_reorg/`, plus a configurable bucket-first implementation behind
`chat_completions`. The live `src/policies/` path remains the default.

Implemented here:

- `BucketResolver::resolve` chooses one bucket by length, then capacity/rank/ID.
- `Bucket` owns input limits, context capacity, rank, and plain-or-PD groups.
- `EngineGroup::pick` owns live candidate filtering, policy invocation, and
  exact candidate validation, without cross-bucket fallback.
- `Policy::pick`, within-group fallback interface, admission placement, and `AllowAll`.
- Lazy report capture through one `LoadView` per group selection pass.
- The reorg chat implementation prepares request facts, resolves one bucket,
  calls one plain group or both PD groups, maps errors, and reuses the forwarder.
- `AppContext::chat_routing` configures legacy versus reorg routing on the same
  endpoint and carries the reorg model-resolver map.

Follow-up order: power-of-two and admission (#40271), then bucket SLO ordering
in a separate PR, followed by remaining policies and production configuration.

Not yet implemented in the reorg path:

- Concrete power-of-two selection (its body is still a placeholder), other
  policies, and capacity/in-flight admission checks.
- SLO estimates, targets, and bucket preference ordering.
- CLI/configuration parsing, validation, and model-specific construction.
  The YAML above is illustrative; reorg resolvers are installed in code.
- Session modes, prefix memoization, and cache-aware selection.
- Shared load interpretation, dispatch correction, and policy-specific
  dispatch-timestamp requirements.
- PD compatibility filtering, retry integration, and legacy-route switchover.

The preceding policy sections describe target behavior for those follow-ups;
they do not claim those capabilities are present in this PR.
