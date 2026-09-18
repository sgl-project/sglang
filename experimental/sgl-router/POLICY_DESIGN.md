# Engine selection policies

Status: proposed design. This document describes a refactor; the interfaces and configuration examples below are sketches, not implemented APIs.

## Principles

1. Resolve PD and bucketing first to obtain a set of candidate engines.
2. **Each bucket has an attached policy.** The selected bucket determines both the candidate set and the policy used to select an engine.
3. A policy reads the signals it needs—cache, load, or affinity—runs its own selection logic, and implements `pick()`.
4. A policy must return an engine from the supplied candidate set. It cannot select another bucket, cross a PD role boundary, or add candidates.
5. Each engine-selection policy has an attached `EngineAdmission`. The same admission interface checks one engine or a list, either before selection or after selection. The default is `AllowAll`.

Note on the `AllowAll` default: it applies to newly written bucket policy configuration only. Today the capacity and pending-prefill checks run implicitly for the power-of-two, session-aware, cache-aware, and decode power-of-two paths whenever fresh engine data exists. Migrating an existing configuration must construct the equivalent explicit admission so that those checks are not silently lost; see the admission paragraph in "Configuration migration".

`Policy` selects one engine from a supplied set; `EngineAdmission` decides whether engines may accept the request. There is no `PDPolicy`, `BucketPolicy`, generic `TieredPolicy`, or separate picker framework.

In code, an engine continues to be represented by the existing `Worker` type. Renaming `Worker` is outside this refactor.

## Request flow and policy attachment

```text
Request
  |
  v
Resolve model, PD role, and bucket
  |
  +-- Prefill bucket: [P1, P2, P3], policy = CacheAwarePolicy
  |     |
  |     +-- policy.pick(candidates, request) --> P2
  |
  +-- Decode bucket: [D1, D2], policy = PowerOfTwoPolicy
        |
        +-- policy.pick(candidates, request) --> D1
```

For a plain deployment, the same flow selects one plain bucket and one engine. For PD, the caller coordinates prefill and decode selection. If decode compatibility depends on the selected prefill engine, it resolves decode candidates afterward.

Multiple prefill buckets may attach different policies, as may multiple decode buckets. A bucket's policy is constructed at configuration load and reused across requests; it is not constructed per request.

When explicit buckets are disabled, each model/role pool behaves as one implicit bucket with an attached policy. Existing model-level policy configuration can supply the default for these implicit buckets. An explicit bucket policy overrides that default. Preserve the configured decode policy when supported; reject a dropped policy rather than silently substituting another. The existing decode default remains power-of-two.

Conceptual configuration:

```yaml
buckets:
  - id: prefill-short
    stage: prefill
    engines: [P1, P2]
    policy:
      type: cache_aware
      admission:
        type: allow_all
        placement: before_selection

  - id: prefill-long
    stage: prefill
    engines: [P3, P4]
    policy:
      type: session_aware
      on_new_session: power_of_two

  - id: decode-default
    stage: decode
    engines: [D1, D2]
    policy:
      type: power_of_two
      admission:
        type: capacity
        placement: after_selection
```

The exact configuration syntax should follow the existing configuration types. The example illustrates ownership and attachment, not a new parser requirement.

### Candidate-set contract

PD/bucket resolution owns model compatibility, role, health, membership, request-size compatibility, and configured bucket ordering. A request receives a snapshot of the currently eligible engines, not an unfiltered static membership list.

Cache and affinity preference apply only within this set. Policies do not probe other buckets for a better match. Global cache and session affinity are provided by the resolver, not by the policy: when affinity-first resolution is enabled, the resolver hands the policy a stage-wide affinity group before any size bucket (see "Affinity-first resolution"). A cache holder or session assignment that is in neither the affinity group nor the selected bucket cannot win.

If the bucket resolver supports fallback buckets, it owns that fallback order and the conditions for advancing. A cache miss inside a size bucket is not a reason to change buckets: the current bucket's policy handles it. A miss inside the affinity group is reported as `NoCandidates` and advances to the size buckets. An empty or unavailable bucket, no admitted candidates, or rejection of the selected engine may trigger the resolver's explicitly configured fallback. Configuration and invalid-signal errors are propagated, not silently treated as bucket exhaustion.

`AllowAll` adds no acceptance constraints to the concrete policy's selection logic. It does not define cache preference or disable that policy's configured queue and pressure behavior. Before-selection admission can remove a cache holder; after-selection admission can reject the selected engine. Structural compatibility and health checks remain in effect. Reimplement existing cache-aware selection inside `CacheAwarePolicy`, including its configured queue and pressure behavior within the supplied candidates. Do not inherit an outer capacity-relaxation ladder or post-return substitution from the old selection path.

### Ordered bucket resolution, including SLO buckets

`BucketResolver` in `buckets.rs` exposes `ordered_buckets(request)` for one model and routing stage. It reuses the existing PD pool resolution and bucket compatibility checks, and returns ordered `ResolvedEngineGroup` values. Each value includes the bucket's attached policy and its current candidate engines. The name refers to ordering buckets, not selecting engines inside them.

Resolution proceeds as follows:

1. Select the model's healthy plain, prefill, or decode pool.
2. If affinity-first resolution is enabled for this stage, emit the affinity group first (see below).
3. Keep configured buckets for that stage whose request-size and context limits fit. For prefill, use input length; for decode, use expected peak sequence length when available. Preserve the explicit unknown-output-length behavior for decode catch-all buckets.
4. Sort compatible buckets by ascending `rank`, then bucket ID.
5. Apply the configured SLO preference, preserving rank order within each preference group.
6. Intersect each bucket's membership with the stage's available engines and skip empty groups.

SLO matching uses configured bucket performance estimates and request targets:

| Stage | Bucket matches the request SLO when |
| --- | --- |
| Prefill | `ttft_p95_at_capacity_ms <= requested_ttft_ms` |
| Decode | `tps_p05_at_capacity >= requested_tokens_per_second` |

When the request has no SLO for that stage, all compatible buckets count as matching. When a target is supplied but the bucket lacks the corresponding estimate, it does not match. These estimates determine preference, not a guarantee of observed performance.

| SLO preference | Ordered buckets |
| --- | --- |
| `disabled` | Rank and ID order only. |
| `slo_first` | Matching buckets first, then nonmatching buckets. |
| `best_effort` | Preserve the current implementation's ordering: nonmatching buckets first, then matching buckets. |

Under `slo_first`, a nonmatching bucket remains a fallback if earlier matching buckets cannot produce an admitted engine. The resolver does not rerank buckets using live engine load or cache hits. With no explicit bucketing, it returns the implicit stage bucket. Preserve the existing global decode pool fallback when no decode buckets are configured.

### Affinity-first resolution

Cache locality and session assignment are properties of an engine, not of the request's size. Resolving size buckets first would discard a prefix holder or a bound session engine that lives in another bucket. The current code avoids this by running a global cache rung before bucket domains (`selection.rs`, `cache_winner`) and a global session probe for the global session modes; PR #38814 measured a hit-rate collapse from 72% to near zero when cache selection was bucket-scoped. This design keeps that behavior as a resolver-owned group rather than a policy-owned escape.

When `affinity_first` is enabled for a stage, `ordered_buckets()` emits one synthesized `ResolvedEngineGroup` before the size buckets:

- **Membership** is the whole healthy stage pool, filtered per engine by that engine's own bucket rules that do not depend on request size: `max_context_tokens`, and TTFT eligibility under `slo_first`. Unbucketed engines are always eligible. This is the existing `prepare_prefill_cache_candidate` and `prefill_affinity_domain` logic applied to the pool. The extend-token ranges are not applied, because the point of the group is to let a holder outside the size range win.
- **Policy** is the stage's model-level policy instance, the same object the implicit or size buckets use. Only affinity-capable policies participate: `CacheAwarePolicy`, `SessionAwarePolicy`, and `StickyPolicy`. For any other model policy the group is not emitted.
- **Mode** is `HitRequired`, carried on `PickRequest`. In this mode a policy returns its affinity winner if one exists and is admitted; otherwise it returns `NoCandidates` and does not run its fallback or create a binding.
- **Scope** is `Global` for affinity keys and for cache candidate limits and saturation observations. Inside a size bucket the scope is the bucket, as today under `session-affinity-mode bucket`.

Nothing new is stored. The group is a per-request view over the same pool, built the same way size buckets are built. The policy still returns an engine from the set it was given, and the loop still contains no cache or session logic.

```text
fn ordered_buckets(request):
    pool = healthy stage pool
    groups = []
    if stage.affinity_first and model_policy.supports_affinity():
        engines = [e in pool where bucket_of(e) is None
                   or structurally_compatible(bucket_of(e), request)]
        groups.push(group(id="affinity", scope=Global, mode=HitRequired,
                          policy=model_policy, engines=engines))
    groups.extend(size buckets, ordered as in steps 3–6)
    return groups
```

Loop outcomes for the affinity group map onto the existing rules: a selected engine wins; `NoCandidates` (miss, no binding, or every holder failed the policy's soft gates) advances to the size buckets; `NoAdmissibleEngine` or `AdmissionRejected` follows the configured bucket-fallback option, which by default also advances. This reproduces the current ladder: global cache winner, then size buckets; an admission-rejected global winner falls back to its size bucket.

Worked example, prefill stage. `P1, P2` are in `prefill-short` (`max_extend_tokens 4096`), `P3, P4` in `prefill-long` (`min_extend_tokens 4097`), `P5` is unbucketed.

| Request | Affinity group | Size buckets | Result |
| --- | --- | --- | --- |
| 6k tokens, 5k-token prefix on P2 | `[P1..P5]`, P2 is an admitted hit | not reached | P2, although P2 is outside `prefill-long` |
| 6k tokens, no prefix anywhere | `NoCandidates` | `prefill-long` `[P3, P4]`, min-load fallback | P3 or P4 |
| 6k tokens, prefix on P2, P2 fails capacity admission | `NoAdmissibleEngine`, advance | `prefill-long` `[P3, P4]` | P3 or P4 |
| 2k tokens, prefix on P3, request TTFT SLO not met by `prefill-long` under `slo_first` | P3 filtered out by its own bucket's SLO rule, `NoCandidates` | `prefill-short` `[P1, P2]` | P1 or P2 |

The prefix lookup is memoized per request so that a miss in the affinity group does not query the remote indexer a second time from the size bucket's policy. The memo is keyed by the request, not by the group; the policy restricts the memoized matches to its supplied candidates.

Global session modes map onto the group directly. `global-rebind`: the group looks up the global key and returns an admitted in-set binding; the size bucket's policy binds on a miss. `global-preserve`: same lookup; when the group found a binding that could not be used (filtered or rejected), the size bucket's policy runs with lookup and binding disabled so the global assignment is preserved. `bucket`: the group is not emitted for session affinity, and lookup and binding are bucket-scoped.

Defaults: `affinity_first` is on for a stage whose model policy is `cache_aware`, `sticky`, or `session_aware` with a global session mode, and off otherwise. This preserves current behavior for existing deployments. It is a resolver option, not a bucket field.

`BucketResolver::pick()` in `buckets.rs` walks this order and invokes each bucket's own policy:

```text
# request here is scoped to one model and stage.
for bucket in resolver.ordered_buckets(request):
    candidates = bucket.engines
    if candidates is empty:
        continue

    result = await bucket.policy.pick(candidates, request)

    match result:
        selected engine:
            return selected engine
        NoCandidates:
            continue
        NoAdmissibleEngine or AdmissionRejected:
            if bucket_fallback_on_admission_rejection:
                continue
            return result.error
        other error:
            return result.error

return NoSelectableBucket(attempt_reasons)
```

Here `bucket.policy` is shorthand for the resolved group's `bucket.policy`; the loop in `buckets.rs` supplies the current bucket ID, scope, and pick mode in `PickRequest`. The affinity group, when present, is simply the first entry of `ordered_buckets`. The loop retains enough failure information to distinguish an empty pool from admission exhaustion. An implementation may lazily resolve groups, but must preserve this order.

For the initial design, admission rejection advances to the next bucket by default; an explicit resolver option can disable this. With before-selection admission, it advances only when no candidate is admitted. With after-selection admission, it advances when the chosen engine is rejected, even if another engine in that bucket could pass. It does not silently retry that bucket with weaker admission or a different engine.

This replaces the current two-pass admission/capacity-relaxation ladder with one ordered pass. Inside a size bucket, cache misses, missing session bindings, and busy engines accepted by admission are handled by the bucket's policy and do not advance the loop. Inside the affinity group they surface as `NoCandidates` and do advance it. For PD, the coordinator runs this process for prefill and decode, resolving any prefill-dependent decode compatibility before the decode pass.

## Code layout

The outer layer is `buckets.rs`. Concrete policies, their admission checks, and the state they read live under `policies/`. Use `state/` because these components own mutable information, including affinity assignments, rather than only emitting signals.

```text
src/
  buckets.rs                    Defines buckets and their attached policies;
                                resolves PD pools, orders buckets by SLO/rank,
                                calls policy.pick(), and handles bucket fallback

  policies/
    mod.rs                      Policy, PickRequest, PickError, dependencies,
                                and build_policy(config, dependencies)
    admission.rs                EngineAdmission, single/batch checks, placement
    cache_aware.rs              CacheAwarePolicy; local/remote prefix lookup adapter
    session_aware.rs            SessionAwarePolicy
    sticky.rs                   StickyPolicy
    least_load.rs               LeastLoadPolicy and shared least-load helper
    power_of_two.rs             PowerOfTwoPolicy
    random.rs                   RandomPolicy
    round_robin.rs              RoundRobinPolicy

    state/
      mod.rs                    Shared state exports and prefix query/result types
      kv_events/                KV subscriptions and router-local radix-tree index
        mod.rs                  KvEventIndex exports and local prefix lookup API
        index.rs                Index lifecycle and event application
        tree.rs                 Existing HashTree
        subscriber.rs           Existing KV/load event subscriptions
        hash.rs                 Existing block hashing
        ...                     Existing wire, discovery, and metrics helpers
      engine_load/
        mod.rs                  EngineLoadMonitor and request-scoped LoadView
        reports.rs              Engine-reported telemetry storage and freshness
        inflight.rs             Router-local request tracking and dispatch correction
      affinity_store.rs         Scoped assignments, expiry, and atomic updates

  workers/                      Existing Worker, discovery, registry, and health
    ...

  server/
    app_context.rs              Shared policy state and configured bucket resolver
    routes/chat.rs              Prepare request; select each stage through buckets;
                                coordinate PD execution and dispatch accounting
    ...
```

`buckets.rs` owns both bucket resolution and the ordered selection loop. There is no separate `engine_selection.rs`. The HTTP handler calls this layer once for a plain request or for each required PD stage; it owns network dispatch and the prefill/decode execution lifecycle.

`policies/mod.rs` owns the small construction function; a separate `factory.rs` is unnecessary initially. Construction matches configuration to concrete types, validates dependencies, and injects shared handles. It does not own subscriptions or request selection logic.

`policies/state/kv_events/` already provides both event ingestion and local cache indexing. Keep that name and implementation, move it under `state/`, and expose prefix lookup through the same subsystem. The remote KV indexer is a separate optional backend: a small adapter next to `CacheAwarePolicy` normalizes local and remote results. No additional cache or prefix directory is required.

`policies/state/engine_load/` groups engine reports and router-local in-flight tracking under one public load view. Keeping `reports.rs` and `inflight.rs` separate avoids merging two substantial implementations. Existing worker load guards remain the dispatch-facing handles to this accounting.

The existing subscription infrastructure may continue to ingest both KV events and engine-load reports, feeding the shared engine-load state. Load-only deployments must be able to receive reports without maintaining a local cache tree. Start shared services once from application wiring and update them on worker lifecycle events; policy instances only hold handles.

The dependency direction is:

```text
buckets -> policies -> state
                    -> admission -> state
```

Shared state and admission checks do not depend on bucket ordering or concrete policy strategies. The local/remote adapter translates backend results into common prefix result types exported by `state/mod.rs`.

## Core interfaces

### Bucket and resolved engine group

The long-lived bucket owns configuration and an attached policy. Its resolved group contains the current per-request candidates.

```rust
pub struct Bucket {
    pub id: BucketId,
    pub model: ModelId,
    pub stage: RoutingStage, // Plain, Prefill, Decode
    pub policy: Arc<dyn Policy>,
    // Membership and existing compatibility/bucket rules.
}

pub struct ResolvedEngineGroup {
    pub bucket: Arc<Bucket>,
    pub engines: Vec<Arc<Worker>>,
}
```

`BucketResolver` in `buckets.rs` incorporates the existing PD pool resolution and bucket compatibility logic. Its `ordered_buckets(request)` builds the ordered groups, and its asynchronous `pick(request)` runs the loop and calls `group.bucket.policy.pick(...)`, returning one engine for the requested stage. The HTTP handler coordinates the prefill/decode pair. Neither bucket ordering nor the outer loop contains cache ranking or session lookup logic. Bucket configuration owns membership, rank, and SLO estimates; resolver configuration owns stage-specific SLO preference and whether admission rejection permits moving to the next bucket.

### Policy

Use one object-safe interface. A boxed future accommodates the existing remote prefix indexer without blocking an executor thread or requiring a separate asynchronous preparation framework. Policies that only read local state return immediately from their future.

```rust
use futures::future::BoxFuture;

pub trait Policy: Send + Sync + std::fmt::Debug {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Arc<Worker>, PickError>>;
}

pub struct PickRequest<'a> {
    pub model: &'a ModelId,
    pub stage: RoutingStage,
    pub bucket_id: &'a BucketId,
    pub scope: AffinityScope,   // Global for the affinity group, Bucket otherwise
    pub mode: PickMode,         // Normal, or HitRequired in the affinity group
    pub tokens: Option<&'a [u32]>,
    pub session_id: Option<&'a str>,
    pub routing_key: Option<&'a str>,
    pub prefix: Option<&'a PrefixMemo>, // per-request memoized prefix lookup
}

pub enum PickMode {
    /// Run the policy's full logic, including its fallback.
    Normal,
    /// Return an admitted affinity winner or `NoCandidates`; never fall back or bind.
    HitRequired,
}

pub enum PickError {
    NoCandidates,
    NoAdmissibleEngine(Vec<EngineRejection>),
    AdmissionRejected(EngineRejection),
    InvalidSignal(String),
}
```

These signatures are design sketches; supporting IDs and error details should reuse existing types where possible. `PickRequest` contains request facts and an opaque scope identity, not the bucket selector, PD resolver, HTTP body, or telemetry backend configuration.

The return value is the selected engine. Record selection reasons through existing metrics/tracing rather than introducing primary/backup proposals or cache-specific result variants.

### Policy construction and dependencies

```rust
pub struct PolicyDependencies {
    pub load: Arc<EngineLoadMonitor>,
    pub local_cache: Option<Arc<KvEventIndex>>,
    pub remote_cache: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
    pub affinity: Arc<AffinityStore>,
}

pub fn build_policy(
    config: &PolicyConfig,
    deps: &PolicyDependencies,
) -> Result<Arc<dyn Policy>, ConfigError>;

pub struct LeastLoadPolicy {
    load: Arc<EngineLoadMonitor>,
    admission: AdmissionConfig,
}

pub struct CacheAwarePolicy {
    cache: CacheSource, // Local event index or remote prefix indexer.
    load: Arc<EngineLoadMonitor>,
    admission: AdmissionConfig,
}

pub struct SessionAwarePolicy {
    assignments: Arc<AffinityStore>,
    fallback: Arc<dyn Policy>,
    admission: AdmissionConfig,
}
```

`build_policy()` in `policies/mod.rs` passes only the dependencies needed by each concrete policy and validates configuration up front. It constructs the attached admission policy and placement, defaulting to `AllowAll`. Initially, affinity fallbacks can be restricted to the simple placement policies to avoid recursive configuration. Such nested fallbacks use `AllowAll`: the owning session/sticky policy applies admission once at its configured placement. Random and round-robin selection need no telemetry, although their attached admission policy may require it.

“Subscribing to signals” means holding a shared service handle and reading its current view. Policies do not spawn duplicate engine subscriptions, polling loops, or cache indexes. Services start once and follow worker lifecycle events.

### EngineAdmission: one engine or a list

Keep the interface and initial implementations in `policies/admission.rs`. It evaluates acceptance only; it does not rank engines, select replacements, mutate affinity, or choose another bucket.

```rust
pub enum AdmissionPlacement {
    BeforeSelection,
    AfterSelection,
}

pub struct AdmissionConfig {
    pub policy: Arc<dyn EngineAdmission>,
    pub placement: AdmissionPlacement,
}

pub enum AdmissionDecision {
    Allow,
    Reject(AdmissionReason),
}

pub struct EngineRejection {
    pub engine_id: WorkerId,
    pub reason: AdmissionReason,
}

pub struct AdmissionContext<'a> {
    pub request: &'a PickRequest<'a>,
    pub input_tokens: u64,
    pub expected_peak_sequence_tokens: Option<u64>,
    pub load: Option<&'a LoadView>,
    pub prefix: Option<&'a PrefixLookup>,
}

pub trait EngineAdmission: Send + Sync + std::fmt::Debug {
    fn check(
        &self,
        engine: &Worker,
        ctx: &AdmissionContext<'_>,
    ) -> Result<AdmissionDecision, AdmissionError>;

    fn check_many(
        &self,
        engines: &[Arc<Worker>],
        ctx: &AdmissionContext<'_>,
    ) -> Result<Vec<AdmissionDecision>, AdmissionError> {
        engines.iter().map(|engine| self.check(engine, ctx)).collect()
    }
}
```

`check()` handles a single engine; `check_many()` handles a candidate list through the same interface. The batch result contains exactly one decision per input engine, in input order, including rejection reasons. A batch override may optimize evaluation but must preserve the result of independent checks against the same context. Empty input returns an empty list. Filtering preserves candidate order and cannot add engines.

Signal preparation happens before admission when needed. For example, a pending-prefill budget based on uncached tokens requires prefix results even in `BeforeSelection` mode. Reuse one prepared load view for admission and load comparison within the pick; synchronous checks do not fetch telemetry themselves. Request-size values come from existing request preparation, including when exact routing tokens are unavailable.

Admission errors (such as invalid required inputs) are separate from explicit rejections. Each concrete admission policy defines its missing/stale-data behavior; unknown load must not silently become zero. A migrated check should preserve its documented missing-data behavior unless configuration explicitly changes it.

Initial implementations:

| Implementation | Acceptance rule |
| --- | --- |
| `AllowAll` | Always allow; default attachment. |
| `QueueLimitAdmission` | Engine-reported waiting requests are below the configured limit. |
| `CapacityAdmission` | Projected running requests and KV tokens fit reported capacity. |
| `PendingPrefillAdmission` | Waiting uncached tokens plus incoming prefill work fit the configured budget. |
| `InFlightLimitAdmission` | Router-local in-flight requests are below the configured limit. |

If multiple checks are needed, an `AllOfAdmission` implementation can require all checks to allow, using the same context. It is still one attached admission policy. A rejecting decision carries a concrete reason; none of these checks introduces an implicit capacity-relaxation fallback.

`QueueLimitAdmission` is a hard acceptance check and is not a direct replacement for the existing cache-aware `worker_queue_limit`. That option participates in a soft queue gate, diversion, and saturation behavior owned by `CacheAwarePolicy`. Such policy preferences may reconsider a queued engine, but cannot override rejection by the attached admission.

### Admission placement and control flow

Concrete policies implement their own selection logic and apply their attachment at the configured point. Small helpers can share batch filtering and error construction without introducing a new selection-policy wrapper or base class.

```text
BeforeSelection:
    prepare required signals
    decisions = admission.check_many(candidates, context)
    admitted = candidates whose decision is Allow
    if admitted is empty:
        return NoAdmissibleEngine(rejections)
    selected = run this policy's selection logic on admitted
    return selected

AfterSelection:
    prepare required signals
    selected = run this policy's selection logic on candidates
    decision = admission.check(selected, context)
    if decision is Reject:
        return AdmissionRejected(selected, reason)
    return selected
```

Both placements return `NoCandidates` immediately for an empty original set. Admission errors become explicit pick errors rather than rejection decisions. After-selection rejection does not automatically try a backup or another engine within the same bucket. The caller can advance to another bucket only through its configured bucket-fallback behavior. Retrying within the bucket would be an additional, separately specified behavior.

Placement affects results. Suppose the cache-aware policy would prefer A under its configured selection rules, but A fails admission and B passes. Before-selection admission removes A and lets the policy select among the admitted engines. After-selection admission selects A and then returns a rejection. `AllowAll` adds no rejection in either placement.

Admission checks do not reserve capacity. Concurrent requests may pass against the same observation; strict enforcement would require an explicit reservation mechanism. Retain existing dispatch accounting without presenting snapshot checks as atomic capacity guarantees.

## Shared components

### EngineLoadMonitor and LoadView

`policies/state/engine_load/` owns engine-reported measurements and router-local in-flight tracking. `reports.rs` stores reported observations; `inflight.rs` tracks local request lifetimes. `mod.rs` exposes their combined view:

```text
EngineLoadMonitor.snapshot(candidates, now) -> LoadView
LoadView.get(engine) -> LoadObservation
LoadView.compare(a, b) -> Ordering
```

A load observation retains its source, freshness, and available measurements. Unknown load is distinct from a measured zero. Preserve existing fallback to router-local accounting and corrections for recent dispatches not yet reflected in engine reports; put this interpretation in one place.

Each pick captures one load view when needed and uses it consistently. Prefill and decode can capture separate views. There is no claim that load and cache observations form an atomic global snapshot.

Reuse the existing request guards and accounting lifecycle. Do not add a second independent in-flight counter. Local accounting reduces concurrent selection herding but does not provide distributed reservations across routers.

### Local KV-event index and remote prefix lookup

`policies/state/kv_events/` owns the local cache index. Its `KvEventIndex` manages subscribers and an event pump that applies stored/removed/cleared-block events to the shared `HashTree`. It also clears state when workers are removed. The tree stores prefix hashes, engine/rank ownership, and storage tiers, not actual KV tensors.

Move the current `RadixTreePrefixProvider` lookup logic into this subsystem so it exposes a local `match_prefix(query, candidates)` operation. The existing tree, hashing, wire format, and event lifecycle remain reusable.

`cache_aware.rs` contains a small `CacheSource` adapter:

```rust
enum CacheSource {
    Local(Arc<KvEventIndex>),
    Remote(Arc<dyn sgl_kv_indexer::PrefixIndex>),
}
```

It exposes one asynchronous operation to `CacheAwarePolicy`, immediately returning local results or awaiting the remote indexer:

```text
match_prefix(query, candidates) -> Result<PrefixLookup, PrefixError>

PrefixLookup:
  Available(matches)    # Empty means a confirmed miss.
  Unavailable(reason)   # Timeout, unavailable backend, or unusable index state.

PrefixMatch:
  engine identity
  matched prefix length, with explicit units
```

Common prefix query/result types live in `policies/state/mod.rs`, so admission can consume prepared prefix information without depending on `CacheAwarePolicy`. Hashing helpers stay in `state/kv_events/hash.rs` and are reused by the remote adapter. Backend choice, response normalization, and mapping addresses to current candidate identities belong to the adapter; neither the bucket loop nor the HTTP route performs backend-specific lookup work.

Reuse the existing external indexer contract and local hash-tree behavior. Preserve model/cache namespace and rank information where needed; a reported hit must refer to cache usable by the engine being selected. Do not reinterpret block counts as exact token counts without a valid conversion.

Policies receive matches restricted to their supplied candidates. Local cache maintenance owns worker removal, eviction events, and index invalidation. Unavailable and empty results both permit the initial cache-aware fallback, but remain distinct in metrics. Malformed queries or configuration errors can retain explicit error behavior. The remote backend can be used without maintaining a duplicate local tree.

### AffinityStore

`policies/state/affinity_store.rs` owns assignment storage, expiry, and concurrency:

```text
lookup(key) -> optional assignment
touch(key, expected_version)
bind_if_absent(key, engine) -> effective assignment
replace_if_version(key, expected_version, engine) -> update result
invalidate_engine(engine)
```

Keys include model, stage, scope, affinity kind (session or routing key), and key value. The scope is `Global` when the policy runs inside the affinity group and the selected bucket otherwise. The global session modes are preserved through affinity-first resolution; see that section and the migration table.

The store does not select engines. Session and sticky policies decide whether to reuse, bind, or replace an assignment. Separate policy instances can share storage safely through scoped keys.

On a concurrent first assignment, use the effective binding returned by the atomic operation only if it is still a candidate and satisfies the configured admission. In before-selection mode it must belong to the admitted subset; in after-selection mode a different effective engine must be checked before return. On conflicts or stale bindings, reconcile with bounded retry; never return an out-of-set or admission-rejected assignment.

An assignment records preferred placement, not successful execution. Create or replace a binding during `pick()` only after the proposed engine passes admission; a rejected proposal must not create or rewrite a binding. An existing binding may be preserved on rejection. A healthy assignment may remain if later PD selection or dispatch fails. It must not increment dispatch load or claim successful execution. This avoids adding a policy commit protocol solely for affinity; worker invalidation and expiry handle obsolete assignments.

## Concrete policy behavior

The table describes policy responsibilities. Every policy applies its attached admission before or after its selection logic as configured. Cache-aware algorithm redesign is outside this document: reimplement the existing policy in the new code, subject to the candidate-set and admission boundaries and the explicit migration changes below.

| Policy | Signals/state | Initial behavior |
| --- | --- | --- |
| `LeastLoadPolicy` | Load | The existing `load_based` algorithm: select the least-loaded candidate with the current ordering, tie-breaker, incomplete-telemetry fallback, and recent-dispatch correction. It is not the min-load fallback used inside `CacheAwarePolicy`; that fallback keeps its power-of-k sampling (`--min-load-choices`). |
| `CacheAwarePolicy` | Cache and load | Reimplement existing cache-aware candidate construction, ranking, configured queue/pressure behavior, and fallback inside `pick()`, restricted to the supplied candidates. |
| `SessionAwarePolicy` | Affinity store and fallback policy | Reuse an in-set session assignment; otherwise select through fallback and establish/reconcile a binding. |
| `StickyPolicy` | Affinity store and fallback policy | Same pattern using the configured routing key and sticky assignment rules. |
| `PowerOfTwoPolicy` | Load | Sample two distinct candidates when possible and compare their load. |
| `RandomPolicy` | RNG | Select a candidate randomly. |
| `RoundRobinPolicy` | Per-instance cursor | Rotate over candidates; cursor belongs to the bucket's policy instance. |

For missing session/routing keys, affinity policies invoke their fallback without creating an assignment. A previous assignment excluded from the candidate set does not win. Under bucket-scoped behavior, an obsolete binding can be replaced. In `HitRequired` mode, `CacheAwarePolicy` returns `NoCandidates` when no candidate survives its minimum-hit gate, queue gate, and admission, and the affinity policies return `NoCandidates` when no admitted in-set binding exists; neither runs its fallback nor creates a binding in that mode.

The cache-aware implementation reference is the current `src/policies/cache_aware.rs`, the cache-specific resolution and queue/pressure helpers in `src/policies/admission.rs`, and the cache fallback path in `src/policies/selection.rs`, together with their behavioral tests. Preserve the existing minimum-hit thresholds, bounded candidate construction, prefix/uncached-work ranking, switching margin, pressure guard, queue gate, saturation behavior, and the power-of-k min-load fallback (`--min-load-choices`, default 2, tier-aware under the queue gate). Move their policy-owned parts behind `pick()` rather than retaining an outer cache-specific selector. This document does not introduce replacement cache ranking or miss-selection rules.

Compatibility is scoped to the engines allowed by the new flow. Candidate limits and fleet saturation observations use the current group's candidate domain: the stage pool inside the affinity group, matching today's global cache rung, and the bucket inside a size bucket. Before-selection admission further restricts the engines eligible to win. Capacity and pending-prefill acceptance belong to the attached admission and may not be relaxed by a cache fallback. The old global cache probe is preserved as the affinity group; the cross-bucket capacity-relaxation pass is explicitly removed. The migration table identifies these boundary changes separately from retained cache tuning.

Concrete policies may reuse load-comparison, prefix-preparation, and admission helpers without introducing another selector trait or generic tier executor.

## Configuration migration: keep / change / drop

This is the proposed migration contract, not a claim that the current parser implements it. The inventory covers every explicit CLI option in `src/config/cli.rs`, all fields of the bucket JSON in `src/config/types.rs`, and the nested sampling configuration in `src/config/sampling.rs`. Internal configuration field names are included where they differ from the CLI. Moving a field to a new owner alone does not change its classification.

- **Keep:** retain the option's meaning, defaults, units, and validation. The common candidate-set and admission boundaries still apply.
- **Change:** retain a configuration path for the capability, but explicitly change its scope, behavior, or representation as stated.
- **Drop:** omit the capability from the initial refactor. Reject explicit use at configuration load with an actionable error; never accept and ignore it or silently select another policy.

Preserve the CLI spelling `load_based` even if its Rust implementation is named `LeastLoadPolicy`. Model-level policy settings become defaults for the applicable buckets; an explicit bucket policy overrides them. Retained policy tuning is inherited only by buckets using that policy. Validate dependency requirements and incompatible combinations at construction time. Proposed drops below are scope decisions for this design, not changes to the existing executable.

The new attachment default is `AllowAll`, but migrating legacy configuration must not silently erase existing acceptance checks. Construct explicit before-selection `CapacityAdmission` for the current top-level prefill/plain power-of-two, session-aware, and cache-aware paths, and for the supported decode power-of-two path. Add configured pending-prefill and in-flight checks through `AllOfAdmission`. Other paths retain `AllowAll` unless an applicable check is configured. This preserves the checks, not the old algorithm for choosing a backup or relaxing capacity; exhausted admission now follows the resolver's single-pass fallback contract. Missing native capacity data retains the current fail-open behavior.

| Existing option / configuration field | Disposition | Migration behavior / owner |
| --- | --- | --- |
| `--host`, `--port` (`server.host`, `server.port`) | Keep | HTTP listener configuration remains outside policies. |
| `--shutdown-drain-secs`, `--termination-grace-secs` | Keep | Preserve shutdown timing and its startup validation/advisory. |
| `--model-id` (`model.id`) | Keep | Model identity scopes resolution, cache queries, and affinity. |
| `--tokenizer-path` (`model.tokenizer_path`) | Keep | Preserve tokenizer loading, model-ID fallback, and ingress token preparation/forwarding. |
| `--policy round_robin` | Change | Preserve rotation and the current default policy; cursor becomes local to each bucket's policy instance. |
| `--policy random` | Keep | Uniform random choice within supplied candidates. |
| `--policy power_of_two` | Change | Preserve distinct-pair sampling and prefill pressure comparison. Admission precedes selection for migrated configuration; remove outer proposal/backup resolution and capacity relaxation. |
| `--policy load_based` | Keep | Implement as `LeastLoadPolicy`; preserve current load ordering, tie-breaking, incomplete-telemetry fallback, and recent-dispatch correction. |
| `--policy cache_aware` | Change | Reimplement existing cache selection and fallback, with retained tuning below. Global cache winners are preserved through the affinity group; inside a size bucket, lookup results and winners are restricted to that bucket. Move hard acceptance to explicit admission. No replacement cache algorithm is specified here. |
| `--policy session_aware` | Change | Preserve session lookup, expiry, keyless/new-session power-of-two fallback, and admitted assignment. Use scoped atomic binding (global in the affinity group, bucket otherwise); remove the shared primary/backup pressure-escape path. |
| `--policy sticky` | Change | Preserve routing-key pins, expiry, and configured fallback. Pins are looked up in the affinity group with global scope and reconciled atomically on concurrent initial assignment. |
| `--policy fused_score`, `--policy score_policy` (`model.fused`) | Drop | No generic score-composition framework in the initial refactor; require an explicitly selected supported policy. |
| `--decode-policy power_of_two` (`model.decode_policy`) | Change | Remains the decode default. Use the common policy interface with existing decode pressure comparison and explicit capacity admission; remove the capacity-relaxation pass. Do not inherit the prefill model policy as the decode default. |
| `--decode-policy legacy_host_affinity` | Drop | Omit the same-host preference/load-tolerance algorithm initially; reject this value. Structural PD compatibility remains the coordinator's responsibility. |
| `--bucket-config` (`model.bucket_config`) | Change | Retain JSON loading and validation; extend bucket configuration with policy attachment and resolver options, including per-stage `affinity_first`. The YAML above is illustrative, not a parser migration. |
| `bucket_config.buckets` | Change | Each configured bucket gains a long-lived policy; omitted policy uses the applicable model/stage default. No explicit buckets means one implicit bucket per model/stage. |
| `buckets[].id` | Keep | Stable bucket identity, rank tie-breaker, and affinity scope. |
| `buckets[].stage` (`prefill`, `decode`) | Change | Preserve existing stage values and role isolation; define plain-stage attachment explicitly in the new schema. Existing prefill bucket rules also serve plain requests today and must retain that mapping during migration. |
| `buckets[].rank` | Keep | Ascending priority, then bucket ID, within each SLO preference group. |
| `buckets[].worker_ids` | Keep | Preserve configured membership, intersected with eligible model/stage workers. The conceptual `engines` example does not require renaming this key. |
| `buckets[].min_extend_tokens`, `buckets[].max_extend_tokens` | Keep | Resolve prefill size buckets using input length. As today, an affinity winner found in the affinity group may sit outside these ranges; `max_context_tokens` and SLO rules still apply to it. |
| `buckets[].min_sequence_tokens`, `buckets[].max_sequence_tokens` | Keep | Decode compatibility uses expected peak sequence length; unknown output length can use only catch-all decode buckets. |
| `buckets[].max_context_tokens` | Keep | Preserve stage-appropriate context compatibility and the unknown-output-length rule. |
| `buckets[].ttft_p95_at_capacity_ms`, `buckets[].tps_p05_at_capacity` | Keep | Configured estimates feed SLO preference, not live-load scoring or performance guarantees. |
| `bucket_config.ttft_slo_policy`, `bucket_config.tps_slo_policy` (`disabled`, `slo_first`, `best_effort`) | Keep | Preserve ordered-domain rank/SLO rules, including nonmatching-first `best_effort`, and the current `slo_first` TTFT filter on affinity-group members; nonmatching `slo_first` buckets remain fallback candidates. |
| `buckets[].max_pending_prefill_tokens` | Change | Represent as attached `PendingPrefillAdmission`, composed with other hard checks. Preserve uncached-work accounting for known cache matches and full-input accounting otherwise, with current missing-report behavior. Make enforcement explicit for the bucket's policy; never discard a configured budget during migration. |
| `--cache-prefix-provider` (`cache_aware.prefix_provider`: `radix_tree`, `indexer`) | Keep | Preserve local/remote backend selection, existing default inference, and dependency validation. Policy wiring owns the adapter. |
| `--kv-indexer-endpoint` (`cache_aware.kv_indexer_endpoint.url`) | Keep | Shared remote client endpoint and existing scheme/provider validation. |
| `--kv-indexer-query-timeout-ms` (`query_timeout_ms`) | Keep | Preserve timeout and unavailable-backend fallback behavior. |
| `--kv-indexer-query-max-inflight` (`query_max_inflight`) | Keep | Preserve the router-wide query concurrency limit; do not multiply it by the number of bucket policy instances. |
| `--cache-affinity-min-matched-tokens` (`affinity.cache_affinity_min_matched_tokens`) | Keep | Preserve the minimum-hit gate and current default of 1,024 tokens; do not replace it with any-positive-hit eligibility. |
| `--cache-affinity-min-match-ratio` (`affinity.cache_affinity_min_match_ratio`) | Keep | Preserve the optional ratio gate and its validation. |
| `--cache-candidate-min-workers` (`affinity.cache_candidate_min_workers`) | Keep | Preserve the bounded-candidate formula and default minimum, computed over the current group's candidate domain: the pool in the affinity group, as today. |
| `--cache-candidate-ratio` (`affinity.cache_candidate_ratio`) | Keep | Preserve the configured fraction and validation; its denominator is the current group's candidate domain. |
| `--cache-candidate-max-workers` (`affinity.cache_candidate_max_workers`) | Keep | Preserve the candidate upper bound and prefix/pressure ordering used for truncation. |
| `--cache-switch-margin-tokens` (`affinity.cache_switch_margin_tokens`) | Keep | Preserve the uncached-work margin within which the cache pressure guard may alter ranking. |
| `--disable-pressure-guard` (`affinity.pressure_guard`, inverted flag) | Change | Retain the cache-aware guard and its enabled default inside the concrete policy. Session-aware pressure escape is dropped; reject explicit session-only use of this tuning. |
| `--pressure-abs-threshold-tokens` (`affinity.pressure_abs_threshold_tokens`) | Change | Retain cache-aware token-gap semantics/default; reject tuning this for the simplified session policy. |
| `--pressure-abs-threshold-ms` (`affinity.pressure_abs_threshold_ms`) | Change | Retain cache-aware queue-time semantics and comparability requirements; reject session-only use. |
| `--pressure-rel-threshold` (`affinity.pressure_rel_threshold`) | Change | Retain cache-aware relative-pressure semantics/default; reject session-only use. |
| `--worker-queue-limit` (`affinity.worker_queue_limit`) | Keep | Preserve the cache policy's soft queue gate, missing-sample fail-open behavior, and queue-aware fallback. Diversion and saturation observations use the current group's domain, the pool in the affinity group as today. Do not translate it into hard `QueueLimitAdmission`. |
| `--saturation-queue-floor` (`affinity.saturation_queue_floor`) | Keep | Preserve configured saturation pinning and validation against the queue limit over the current group's domain. A pin may not bypass attached hard admission. |
| `--min-load-choices` (`affinity.min_load_choices`) | Change | Preserve power-of-k sampling for the cache-aware min-load fallback: default 2, `k >= pool` returns the exact minimum, tier-aware under `--worker-queue-limit`, and the no-sort constraint on the pairwise pressure comparator. The sample is drawn from the bucket's candidate domain. Requires `--policy cache_aware` today; threading it to the standalone `power_of_two` and affinity fallbacks remains a follow-up. |
| `--session-id-header` (`affinity.session_id_header`) | Keep | Ingress extracts the configured key before calling the bucket policy. |
| `--session-idle-secs` (`affinity.session_idle_secs`) | Keep | Preserve session assignment idle timeout in shared affinity storage. |
| `--session-eviction-interval-secs` (`affinity.session_eviction_interval_secs`) | Keep | Preserve cleanup cadence while centralizing service lifecycle. |
| `--stable-pair` (`affinity.stable_pair`) | Drop | The initial session policy has no deterministic primary/backup pair. Reject explicit enablement. |
| `--affinity-mode` (`affinity.mode`: `strict`, `soft`) | Change | `strict` maps to reuse of an admitted in-set binding. `soft` pressure escape is dropped and explicit `soft` is rejected. The former omitted-option default was `soft`; the new session behavior is admitted-binding reuse, so this is a documented default change. |
| `--session-affinity-mode bucket` (`affinity.session_affinity_mode`) | Keep | Remains the default; no session affinity group is emitted, and lookup and binding stay within the selected bucket. |
| `--session-affinity-mode global-preserve`, `global-rebind` | Keep | Enable the session affinity group with global scope. `global-rebind` lets the size bucket bind on a miss; `global-preserve` disables lookup and binding in the size bucket when the group found a binding it could not use. See "Affinity-first resolution". |
| `--routing-key-header` (`sticky.header_name`) | Keep | Preserve configurable ingress extraction and the default header. |
| `--sticky-fallback-policy` (`sticky.fallback_policy`: `round_robin`, `random`, `power_of_two`, `load_based`) | Keep | Preserve all four choices and the round-robin default. Nested fallback uses `AllowAll`; the owning sticky policy applies any attached admission once. |
| `--sticky-idle-secs` (`sticky.idle_secs`) | Keep | Preserve sticky assignment idle timeout. |
| `--sticky-eviction-interval-secs` (`sticky.eviction_interval_secs`) | Keep | Preserve cleanup cadence while centralizing service lifecycle. |
| `--fuse` (`fused[].kind`, `fused[].weight`; `random`, `load_based`, `prefix_cache` terms) | Drop | Reject score terms/weights together with the dropped composition policies. The standalone random/load policies remain supported. |
| `--filter overloaded` (`eligibility.filters`) | Change | Translate to before-selection `InFlightLimitAdmission`; combine hard checks with `AllOfAdmission` instead of `Pipeline`. Preserve rejection when no candidate passes. |
| `--max-in-flight` (`eligibility.max_in_flight`) | Keep | Preserve the strict router-local in-flight ceiling and positive-limit validation for the translated admission. |
| `--filter prefix_cache` (`eligibility.filters`) | Drop | Omit the separate prefix-share eligibility filter and its abstain-on-empty behavior; it is not equivalent to a cache-aware minimum-hit threshold. |
| `--prefix-cache-min-share` (`eligibility.min_prefix_share`) | Drop | Reject with the dropped prefix-share filter; do not silently map it to cache-affinity tuning. |
| `--cb-threshold`, `--cb-cool-down-secs` (`circuit_breaker.threshold`, `cool_down_secs`) | Keep | Preserve breaker configuration, non-mutating candidate health checks, and dispatch-time probe handling. |
| `--worker-urls` (`discovery.StaticUrls.urls`) | Keep | Preserve static registration and worker introspection. |
| `--service-discovery`, `--service-discovery-namespace` (`discovery.K8s.namespace`) | Keep | Preserve discovery backend selection and namespace behavior. |
| `--selector` (`discovery.K8s.mode.Plain.label_selector`) | Keep | Preserve plain discovery and selector validation. |
| `--prefill-selector`, `--decode-selector` (`discovery.K8s.mode.PdDisaggregation`) | Keep | Preserve PD discovery selectors, their mutual requirements, and introspected worker roles. |
| `--request-timeout-secs` (`proxy.request_timeout_secs`) | Keep | Upstream request lifetime remains owned by dispatch/proxy code. |
| `--stale-request-timeout-secs` (`active_load.stale_request_timeout_secs`) | Keep | Preserve stale-request cleanup, cancellation, and guard accounting. |
| `--override-sampling-params` (`sampling_overrides.params`) | Keep | Preserve all fields: `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `n`; exact values and inclusive `min`/`max` bands, validation, and omission behavior remain at ingress. |
| `--sampling-param-conflict` (`sampling_overrides.conflict`: `reject`, `allow`) | Keep | Preserve request rejection/forwarding behavior before engine selection, including range restrictions. |
| `--log-level`, `--log-format` (`observability.log_level`, `log_format`: `text`, `json`) | Keep | Preserve logging configuration; adapt selection reasons to the new ownership without reporting selection as successful dispatch. |
| Generated `--help` / `--version` | Keep | Preserve CLI metadata/help; update help text for changed and dropped routing options. |

Some important behaviors have no dedicated configuration option. The existing global cache probe becomes the resolver's affinity group, enabled by default for the policies that use it today. The two-pass capacity-relaxation ladder and post-policy engine substitution are dropped; they must not survive as hidden compatibility defaults. Preserve worker lifecycle cleanup, load-report freshness/rank aggregation, hash/block-size handling, and request-guard cleanup when moving shared services. A cache-aware behavior test should continue to exercise the same ranking or fallback unless its assertion specifically depends on a boundary change listed above; update those cases to assert the new boundary explicitly.

## Selection versus dispatch

After a successful `pick()`, the configured admission has passed. The caller validates the result belongs to the supplied candidates and performs existing dispatch accounting. A policy result is a proposal to dispatch to that engine, not a guarantee that engine health or capacity cannot change immediately afterward.

Network/health failures use existing retry rules. A retry constructs the permitted candidate set again, excludes failed engines as appropriate, and invokes that bucket's attached policy. There is no later load-based substitution that silently changes a cache hit into a miss.

For PD, the caller acquires/releases request accounting for the actual dispatched stages and cleans up partial setup on failure. A policy is responsible for one engine choice, not the lifetime of the PD request.
