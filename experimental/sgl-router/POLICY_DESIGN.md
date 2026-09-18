# Engine selection: target design

This document describes the target architecture for engine selection. It defines
responsibilities and behavior; the interface and configuration examples are
sketches, not a specification of the current API or CLI. Implementation status is
listed at the end.

The central idea is simple: **the resolver decides which engines a request may
use, a bucket's policy chooses one of them, and admission decides whether that
engine may accept the request.**

An engine is represented by `Worker` in the code. A bucket is a configured group
of engines with membership rules, request limits, and an attached policy.
Prefill/decode disaggregation (PD) runs selection separately for each stage.

## 1. Responsibilities and request flow

| Component | Owns | Returns |
| --- | --- | --- |
| Bucket resolver | Model, role, health, bucket membership, size limits, SLO ordering, and fallback between groups | One selected engine for a stage |
| Policy | Selection and fallback within its supplied candidates; application of its attached admission | One admitted engine, or a selection error |
| Admission | Acceptance checks, such as capacity or in-flight limits | Allow or reject, with a reason |
| Shared state | Load reports, local request accounting, cache ownership, and affinity assignments | Observations and atomic assignment updates |
| Request handler | Request preparation, coordination of PD stages, dispatch, and request cleanup | The HTTP response |

```text
Prepare request: model, tokens, output budget, SLO targets, affinity keys
  |
  v
Resolve the healthy pool for this model and stage
  |
  v
Order candidate groups: optional affinity group, then compatible size buckets
  |
  v
For each group, call its policy with that group's candidates
  |
  +-- policy selects an engine and applies admission --> selected engine
  |
  +-- empty group or admission rejection -------------> next permitted group
  |
  +-- invalid signal or configuration ----------------> error
  |
  v
For PD, repeat for decode; then dispatch and track the actual requests
```

A plain request selects one engine. A PD request selects a prefill engine and a
decode engine. If decode compatibility depends on the prefill choice, the handler
resolves that compatibility before selecting decode.

Each bucket's policy is constructed once and reused across requests. Buckets may
use different policies, including different policies for prefill and decode.
Without explicit buckets, each model/stage pool acts as one implicit bucket.
Model-level policy settings supply the defaults; a bucket can override its
policy. Decode keeps its own default, power-of-two.

The policy must return an engine from the exact candidate set it received. It
cannot add an engine, change buckets, or cross a PD role boundary. The caller
validates this contract before dispatch. There is no later substitution of a
policy's chosen engine.

## 2. Bucket resolution

The resolver builds an ordered list of **resolved engine groups**. Each group
contains current candidates, an attached policy, and the scope and mode for this
pick. A group can represent a configured bucket, an implicit bucket, or the
special affinity group described below.

### Size and SLO ordering

For one model and stage, resolution proceeds as follows:

1. Obtain the healthy, role-compatible pool.
2. Add the affinity group first when enabled.
3. Keep size buckets whose request and context limits fit.
4. Sort those buckets by ascending `rank`, then bucket ID.
5. Apply the configured service-level objective (SLO) preference, preserving
   rank order within each preference group.
6. Intersect bucket membership with the pool and skip empty groups.

| Stage | Size compatibility | SLO match |
| --- | --- | --- |
| Plain / prefill | Input length fits the extend-token range and context limit | `ttft_p95_at_capacity_ms <= requested_ttft_ms` |
| Decode | Input plus requested output fits the sequence range and context limit | `tps_p05_at_capacity >= requested_tokens_per_second` |

When output length is unknown, decode can use only a catch-all bucket without
sequence bounds; its context limit is checked against input length. If no decode
buckets are configured, decode uses its implicit stage pool. Existing `prefill`
bucket configuration also applies to plain requests.

A missing request SLO means all buckets match. A supplied SLO with no bucket
estimate means that bucket does not match. These estimates establish preference;
they do not guarantee observed performance.

| SLO preference | Order |
| --- | --- |
| `disabled` | Rank, then ID |
| `slo_first` | Matching buckets, then nonmatching buckets |
| `best_effort` | Nonmatching buckets, then matching buckets |

Nonmatching buckets remain fallback candidates under `slo_first`. The resolver
does not reorder buckets using live load or cache hits.

### Affinity before size buckets

A useful cache prefix or an existing session binding may live outside the
request's size bucket. For example, a 6k-token request may share a 5k-token prefix
with an engine in a bucket whose extend-token limit is 4k.

The resolver supports this with an optional **affinity group** before the size
buckets. This gives a policy a broader candidate set explicitly, while preserving
the rule that it can only choose from the candidates it receives.

| Property | Affinity group behavior |
| --- | --- |
| Candidates | Healthy engines in the stage pool, filtered by each engine's own bucket context limit and applicable SLO rule |
| Size ranges | Extend-token ranges do not exclude an existing affinity holder |
| SLO filtering | Under prefill `slo_first`, the engine's own bucket must meet the TTFT target |
| Unbucketed engines | Remain eligible within the healthy stage pool |
| Policy | The stage's default affinity-capable policy |
| Pick mode | `HitRequired`: return an admitted affinity hit; never run a fallback or create a binding |

A miss advances to size buckets. A hit rejected by admission follows the
resolver's admission-rejection setting, which allows advancing by default.

The default is to enable affinity-first for `cache_aware`, `sticky`, and
`session_aware` in a global session mode. Other policies do not use this group.
Affinity-first is a per-stage resolver setting, not a bucket policy's permission
to look outside its candidates.

For the 6k-token example:

| Situation | Result |
| --- | --- |
| The short-bucket engine has the prefix and passes context, SLO, and admission checks | Select it from the affinity group |
| No engine has a usable prefix | Try the compatible size buckets and their policy fallbacks |
| The prefix holder fails admission | Try the compatible size buckets if rejection fallback is enabled |
| The holder fails its own context limit or applicable SLO rule | Exclude it from the affinity group |

### Session scope

Affinity keys distinguish model, stage, affinity kind, scope, and key value.
Session and routing-key assignments cannot collide with each other.

| Session mode | Lookup and binding behavior |
| --- | --- |
| `bucket` (default) | No session affinity group; bindings belong to the selected bucket |
| `global-rebind` | Probe the global binding first; a size bucket may create or replace it when the probe cannot be used |
| `global-preserve` | Probe the global binding first; if a binding exists but cannot be used, disable lookup and binding in later groups for this request |

A new session can establish a binding in either global mode. Global keys remain
global when selection reaches a size bucket. Sticky routing uses global pins so
an existing routing key can be reused across request sizes.

### The selection loop

The resolver makes one ordered pass:

```text
for group in ordered_groups(request):
    result = await group.policy.pick(group.engines, request_for(group))

    match result:
        selected engine:
            validate membership and return it
        NoCandidates:
            continue
        NoAdmissibleEngine or AdmissionRejected:
            retain rejection details
            advance if configured; otherwise return the error
        other error:
            return the error

return exhaustion with the retained failure details
```

The resolver preserves enough information to distinguish an empty pool from
admission exhaustion. Admission rejection advances by default; an explicit
resolver option can disable that behavior.

Inside an ordinary bucket, a cache miss or missing affinity binding is handled
by the policy's fallback. It does not by itself move the request to another
bucket. The affinity group's `HitRequired` mode is the exception: a miss there
means the resolver should continue.

There is no second pass with relaxed capacity, and no automatic backup selection
after a policy returns a rejection.

## 3. Policy and admission contracts

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
`PickRequest` carries request facts: model and stage, bucket and affinity scope,
pick mode, input and expected peak token counts, optional token IDs and affinity
keys, a shared load view, and a per-request prefix memo. It also allows the
resolver to disable affinity lookup and binding for global-preserve fallback.
It does not contain an HTTP body, a bucket resolver, or backend configuration.

| Outcome | Meaning |
| --- | --- |
| `Pick` | The chosen engine belongs to the supplied set and passed admission |
| `NoCandidates` | The set is empty, or a `HitRequired` probe has no usable affinity hit |
| `NoAdmissibleEngine` | Before-selection checks rejected all available candidates; includes reasons |
| `AdmissionRejected` | The chosen engine failed after-selection admission; includes its reason |
| `InvalidSignal` | A required input or signal is invalid; propagate the error |

An affinity probe must distinguish a missing binding from an existing binding
that was excluded or rejected. Global-preserve needs that distinction to decide
whether later groups may bind. Soft cache gates can report a miss; hard admission
rejection remains a rejection.

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
after-selection rejects A and lets the resolver apply its fallback rule.

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

## 4. Concrete policies

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
binding outside the candidates cannot win. A rejected binding may be preserved
while normal-mode selection falls back within the bucket. `HitRequired` never
runs those fallbacks or creates a binding.

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

Candidate limits and saturation observations use the current group's domain:
the stage pool for the affinity group, and the bucket for an ordinary group.
Saturation pinning must still pass hard admission.

The target load fallback supports power-of-k sampling through
`--min-load-choices`, default 2. When k covers the pool, choose the exact minimum.
Preserve queue-tier preference and avoid sorting with a pairwise pressure
comparator that does not define a total ordering.

Memoize the prefix lookup once per request, including remote I/O. Each policy
restricts those matches to its own candidates. A memoized lookup does not imply
that another bucket or stage's cache selection and admission have already run.
Any optimization that skips those steps must establish that the previous result
applies to the current group.

## 5. Shared state and construction

Application wiring starts shared services once. Policy construction validates
configuration and passes the required handles to each policy. Policy instances
do not create duplicate subscriptions, polling loops, indexes, or remote-client
concurrency limits.

Requirements come from all configured policies, their admission checks, and
nested fallbacks. This includes tokenization, affinity-header extraction, load
observations, and dispatch timestamps. A bucket override must receive its
required settings even when the model's default uses a different policy.

### Load state

`state/engine_load/` combines engine reports with existing router-local request
accounting. A request-scoped `LoadView` preserves source, freshness, and available
measurements, including the fallback to local load and correction for dispatches
since the report. Do not add a second independent in-flight counter.

Capture the load view lazily and reuse it within a stage's selection pass.
Prefill and decode may capture separate views. Load and cache observations are
not an atomic global snapshot. Preserve report freshness, rank aggregation, and
request-guard cleanup.

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

## 6. Configuration and compatibility

This conceptual example shows the target attachment model. It is not copyable
current CLI/JSON syntax; existing bucket field names need not change.

```yaml
buckets:
  - id: prefill-short
    stage: prefill
    rank: 10
    worker_ids: [P1, P2]
    max_extend_tokens: 4096
    policy:
      type: cache_aware
      admission:
        type: capacity
        placement: before_selection

  - id: prefill-long
    stage: prefill
    rank: 20
    worker_ids: [P3, P4]
    min_extend_tokens: 4097
    policy:
      type: session_aware
      admission:
        type: capacity
        placement: before_selection

  - id: decode-default
    stage: decode
    rank: 10
    worker_ids: [D1, D2]
    policy:
      type: power_of_two
      admission:
        type: capacity
        placement: after_selection
```

Retained settings keep their meanings, defaults, units, and validation unless a
change is listed below. Policy-specific tuning applies to buckets using that
policy. Reject unsupported settings and incompatible combinations at startup;
do not accept and ignore them.

### Retained behavior

- Keep `load_based` as the CLI name for `LeastLoadPolicy`.
- Preserve bucket membership, ranges, context limits, rank/ID ordering, SLO
  estimates, and the SLO preferences described above.
- Preserve cache-provider selection, endpoint validation, query timeout and
  concurrency limits, and unavailable-backend fallback.
- Preserve cache thresholds and tuning: the 1,024-token default minimum hit,
  optional ratio gate, candidate bounds, switch margin, pressure guard, soft
  queue limit, and saturation floor.
- Preserve session and sticky headers, idle timeouts, eviction cadence, global
  session modes, and the four sticky fallback choices.
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
| Round-robin cursor | One cursor per bucket policy instance |
| Capacity exhaustion | One ordered pass; return exhaustion when no permitted group admits the request |
| Primary/backup proposals and post-policy substitution | Removed; each policy returns one engine |
| Session affinity | Reuse admitted bindings; remove primary/backup pressure escape |
| Omitted `--affinity-mode` | Admitted-binding reuse replaces the former soft-mode default |
| Pressure-guard tuning | Applies to cache-aware selection; reject session-only use |
| Policy attachment | Explicit per-bucket policy overrides the applicable model/stage default |

Reject these dropped options explicitly:

- `--policy fused_score` and `--policy score_policy`, including `--fuse` terms
  and weights.
- `--decode-policy legacy_host_affinity`.
- `--stable-pair`.
- `--affinity-mode soft`; only strict admitted-binding reuse remains.
- `--filter prefix_cache` and `--prefix-cache-min-share`. The removed prefix-share
  filter is not equivalent to the cache-aware minimum-hit gate.

## 7. Dispatch and failure handling

A successful pick means admission passed against the observed state. Health and
capacity can change before dispatch. The handler owns network operations, retry
rules, and accounting for the engines actually dispatched to.

A retry rebuilds the permitted candidates, excludes failed engines as required,
and invokes the attached policy. It does not silently replace a cache winner
with a different engine after selection.

For PD, acquire and release accounting for the actual stages and clean up
partial setup on failure. Policy selection does not own the PD request lifetime.
Selection metrics must not imply that dispatch or execution succeeded.

## 8. Code organization

```text
src/
  buckets.rs                    Group resolution, ordering, and selection loop
  policies/
    mod.rs                      Policy contract and construction
    admission.rs                Acceptance checks and placement
    pools.rs                    Model and PD pool resolution
    cache_aware.rs              Cache selection and local/remote adapter
    session_aware.rs             Session selection and fallback
    sticky.rs                   Routing-key selection and fallback
    least_load.rs               Least-load selection
    power_of_two.rs             Pair sampling and stage-aware comparison
    random.rs                   Random selection
    round_robin.rs              Rotation
    state/
      mod.rs                    Shared exports and prefix result types
      kv_events/                Local index, subscriptions, hashing, wire format
      engine_load/
        reports.rs              Engine reports and freshness
        inflight.rs             Local request accounting
        view.rs                 Shared load interpretation
      affinity_store.rs         Assignments, expiry, and atomic updates
  server/
    app_context.rs              Shared service lifecycle and policy wiring
    routes/chat.rs              Request preparation, PD coordination, dispatch
```

Dependencies flow from the resolver to policies, and from policies and admission
to shared state. State does not depend on bucket ordering or concrete policy
strategies. Small shared helpers are sufficient; no generic score-composition,
tier executor, or separate selection framework is required.

## Implementation status

The PR series introduces the policy interface, shared state, per-bucket policy
selection, and the resolver. It removes the legacy selection ladder and moves
the new engine into the layout above.

The target configuration is broader than the implemented surface: per-stage
`affinity_first`, disabling fallback on admission rejection, per-bucket admission
attachments/placement, and `--min-load-choices` are not yet configurable. The
current implementation derives admission from policy kind and uses a
power-of-two cache fallback. These limitations do not redefine the target
contracts in this document.
