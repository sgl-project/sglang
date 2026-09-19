# Engine selection: target design

This document describes the target architecture for engine selection. It defines
responsibilities and behavior; the interface and configuration examples are
sketches, not a specification of the current API or CLI. Implementation status is
listed at the end.

## Principles

1. **Resolve model, PD role, and buckets first.** Resolution produces ordered
   candidate groups of eligible engines.
2. **Each bucket contains role groups, each with its own policy.** The bucket
   defines shared service constraints. Its plain, prefill, and decode groups
   define engine membership, role-specific limits, rank, and selection policy.
3. **A policy owns its selection logic.** It reads the signals it needs—cache,
   load, or affinity—and implements `pick()`, including its within-bucket fallback.
4. **A policy can only return an engine from its supplied candidates.** It cannot
   select another bucket, cross a PD role boundary, or add candidates.
5. **Each policy has attached admission checks.** Inside `pick()`, admission
   either filters candidates before selection or checks the chosen engine after
   selection. A successful pick always returns an admitted engine. New explicit
   attachments default to `AllowAll`; migrated configurations retain their
   existing acceptance checks.
6. **The selection loop owns fallback between groups.** It tries their policies
   in order and returns one engine for the stage. It never relaxes admission or
   substitutes another engine after a successful pick.

An engine is represented by `Worker`. A bucket describes a service class, such
as a context capacity. It contains optional plain, prefill, and
decode groups. All three use the same `EngineGroup` type; the containing field
identifies the role. Each group has its own policy instance and engine IDs.

```text
Model's bucket configuration
  Bucket: short-context
    shared context limit
    prefill: EngineGroup { engine IDs, token range, rank, policy }
    decode:  EngineGroup { engine IDs, token range, rank, policy }
  Bucket: long-context
    shared context limit
    prefill: EngineGroup { engine IDs, token range, rank, policy }
    decode:  EngineGroup { engine IDs, token range, rank, policy }
```

A plain deployment uses `plain` groups. A PD deployment uses `prefill` and
`decode` groups. A bucket may provide just one role. There is no persistent
`Pool` or `Pools` container: a stage's eligible engines are a filtered view of
`WorkerRegistry`. The registry owns live workers; groups reference their IDs.
Engine role labels remain authoritative when filtering candidates.

PD runs selection separately for each stage. Prefill and decode may select
different buckets. Nesting both groups under a bucket does not require them to
be selected together, paired engine by engine, or retried together.

### Data ownership

| Type | Owns |
| --- | --- |
| `Bucket` | ID, shared context limit, optional plain/prefill/decode groups |
| `EngineGroup` | Engine membership, token range, rank, and attached policy |
| `WorkerRegistry` | Live `Worker` objects, model membership, and access to engine health and role |
| `BucketResolver` | Bucket collection and rejection fallback setting |

`worker_ids: None` means every healthy engine serving the requested model and
role. An explicit empty set means no engines. `EngineGroup::new(policy)` creates
an unbounded catch-all group. Without explicit bucket configuration, construction
creates a `default` bucket containing catch-all groups for the deployment's roles.
Missing role groups are skipped; the resolver does not synthesize a replacement.
Model-specific construction supplies the appropriate bucket collection; the
request's model still filters live workers on every selection.

## 1. Code organization

```text
src/
  buckets.rs                    Group resolution, ordering, and selection loop
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

Dependencies flow from the resolver to policies, and from policies and admission
to shared state. State does not depend on bucket ordering or concrete policy
strategies. Small shared helpers are sufficient; no generic score-composition,
tier executor, or separate selection framework is required.

## 2. Responsibilities and request flow

| Component | Owns | Returns |
| --- | --- | --- |
| Bucket matching | Requested role, context/token limits, and rank/ID ordering | Matching bucket references |
| Selection loop | Filtering live engines by model, health, role, and membership; invoking policies and applying fallback | One selected engine for a stage, or an error |
| Policy | Selection and fallback within its supplied candidates; application of its attached admission | One admitted engine, or a selection error |
| Admission | Acceptance checks, such as capacity or in-flight limits | Allow or reject, with a reason |
| Shared state | Load reports, local request accounting, cache ownership, and affinity assignments | Observations and atomic assignment updates |
| Request handler | Request preparation, coordination of PD stages, dispatch, and request cleanup | The HTTP response |

`BucketResolver::matching_buckets(request)` returns bucket references whose
role and length limits fit, ordered by rank and ID. It does not resolve live
engines or invoke policies. A matching bucket can have no available engines.

`BucketResolver::pick(request)` resolves healthy engines for the requested model
and role, intersects them with each matching bucket's group membership, skips
empty groups, and invokes the attached policy. Both methods use `PickRequest`;
there is no additional request wrapper or resolved-group data structure.

```text
Prepare request: model, tokens, output budget, affinity keys
  |
  v
Match buckets by role and length, then order by rank and ID
  |
  v
Resolve healthy engines for this model and stage
  |
  v
For each bucket, resolve its role group's engines and call its policy
  |
  +-- policy.pick() ----------------------------------> admitted engine
  |     BeforeSelection: filter by admission, then choose
  |     AfterSelection:  choose, then check admission
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

Each role group's policy is constructed once and reused across requests.
Prefill and decode groups in the same bucket may use different policies, such
as cache-aware for prefill and power-of-two for decode. Model/stage settings
supply defaults; each group can override its policy. Decode keeps its own
default, power-of-two. Policy state, such as a round-robin cursor, belongs to
that group policy instance.

The policy must return an engine from the exact candidate set it received. It
cannot add an engine, change buckets, or cross a PD role boundary. The caller
validates this contract before dispatch. There is no later substitution of a
policy's chosen engine.

## 3. Bucket resolution

### Length matching

For one model and stage, routing proceeds as follows:

1. Keep buckets with a group for the requested role.
2. Check the group's token range and the parent bucket's context limit.
3. Sort matching buckets by the group's ascending rank, then bucket ID.
4. In `pick`, obtain healthy engines serving the model and matching the role.
5. Intersect each group's membership with those engines; skip empty groups.
6. Invoke that group's attached policy, returning the first admitted engine.

| Stage | Size compatibility |
| --- | --- |
| Plain / prefill | Input length fits the group's token range and bucket context limit |
| Decode | Input plus requested output fits the group's token range and bucket context limit |

When output length is unknown, decode can use only a group without sequence
bounds; its parent bucket's context limit is checked against input length.
Plain/prefill context checks use input length; decode uses expected peak length
when known. Token ranges are group-specific, so P and D can have different
ranges and ranks even under the same bucket.

The skeleton preserves the existing input-length check for plain/prefill. The
roadmap's `uncached_prefill_tokens` signal is not implemented by this range
check; cache-informed routing remains separate work. Changing that signal must
retain full-context checks and establish how per-engine cache observations
produce a group decision.

Legacy configuration with no decode buckets is translated to a catch-all
decode group. Existing prefill bucket configuration maps to plain groups for a
plain deployment. Explicitly absent groups in the new model remain absent.

### SLO ordering (separate follow-up)

The skeleton supports length matching only. After the power-of-two PR (#40271),
a separate PR adds optional TTFT and token-throughput estimates and request
targets, with per-stage `disabled`, `slo_first`, and `best_effort` ordering.
SLO preference orders length-compatible buckets; it never relaxes context or
token limits. The skeleton has no SLO fields, request wrapper, or ordering enum.

### Affinity before size buckets (follow-up)

A useful cache prefix or an existing session binding may live outside the
request's size bucket. For example, a 6k-token request may share a 5k-token prefix
with an engine in a bucket whose extend-token limit is 4k.

The resolver supports this with an optional **affinity group** before the size
buckets. This gives a policy a broader candidate set explicitly, while preserving
the rule that it can only choose from the candidates it receives.

| Property | Affinity group behavior |
| --- | --- |
| Candidates | Healthy engines serving this model and stage, filtered by each engine's own bucket context limit and, after SLO support lands, the applicable SLO rule |
| Size ranges | Extend-token ranges do not exclude an existing affinity holder |
| SLO filtering (after the SLO PR) | Under prefill `slo_first`, the engine's own bucket must meet the TTFT target |
| Unbucketed engines | Remain eligible among healthy engines of the model and stage |
| Policy | The stage's default affinity-capable policy |
| Pick mode | `HitRequired`: return an admitted affinity hit; never run a fallback or create a binding |

A miss advances to size buckets. A hit rejected by admission follows the
resolver's admission-rejection setting, which allows advancing by default.

The default is to enable affinity-first for `cache_aware`, `sticky`, and
`session_aware` in a global session mode. Other policies do not use this group.
Affinity-first is a per-stage resolver setting, not a group policy's permission
to look outside its candidates. This probe is not implemented by the skeleton.

For the 6k-token example:

| Situation | Result |
| --- | --- |
| The short-bucket engine has the prefix and passes context, admission, and any enabled SLO checks | Select it from the affinity group |
| No engine has a usable prefix | Try the compatible size buckets and their policy fallbacks |
| The prefix holder fails admission | Try the compatible size buckets if rejection fallback is enabled |
| The holder fails its own context limit or an enabled SLO rule | Exclude it from the affinity group |

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
for bucket in matching_buckets(request):
    group = bucket.group(request.stage)
    engines = healthy_members(group, request.model, request.stage)
    if engines is empty:
        continue
    result = await group.policy.pick(engines, request_for(bucket))

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

The resolver preserves enough information to distinguish missing candidates from
admission exhaustion. Admission rejection advances by default; an explicit
resolver option can disable that behavior.

Inside an ordinary bucket, a cache miss or missing affinity binding is handled
by the policy's fallback. It does not by itself move the request to another
bucket. The affinity group's `HitRequired` mode is the exception: a miss there
means the resolver should continue.

There is no second pass with relaxed capacity, and no automatic backup selection
after a policy returns a rejection.

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
all eligible stage engines for the affinity probe, and the role group otherwise.
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
    max_context_tokens: 8192
    prefill:
      rank: 10
      worker_ids: [P1, P2]
      limits: {min: 0, max: 4096}
      policy:
        type: cache_aware
        admission: {type: capacity, placement: before_selection}
    decode:
      rank: 20
      worker_ids: [D1, D2]
      limits: {max: 8192}
      policy:
        type: power_of_two
        admission: {type: capacity, placement: after_selection}

  - id: long-context
    max_context_tokens: 131072
    prefill:
      rank: 20
      worker_ids: [P3, P4]
      limits: {min: 4097}
      policy:
        type: session_aware
        admission: {type: capacity, placement: before_selection}
    decode:
      rank: 10
      worker_ids: [D3, D4]
      limits: {min: 8193}
      policy:
        type: power_of_two
        admission: {type: capacity, placement: after_selection}
```

A request with 4k input tokens and a 16k expected peak can select the first
bucket's prefill group and the second bucket's decode group. A bucket containing
only one role simply omits the other group fields.

Construction validates unique nonempty bucket IDs, token ranges, role-compatible
membership, and deployment mode per model. A plain model must not mix with PD
engines. Missing roles produce no candidates at runtime; they do not fall back
to a different role. These configuration checks belong in the planned factory,
not in another persistent container hierarchy.

Legacy `BucketSpec` represents a single role-specific membership set. Preserve
each legacy spec as a bucket with one populated role group, retaining its ID,
limits, rank, estimates, and applicable policy defaults. There is no existing
parent-bucket association: combining old P and D specs into one shared bucket
requires explicit configuration, never inference from rank or similar names.

Retained settings keep their meanings, defaults, units, and validation unless a
change is listed below. Policy-specific tuning applies to role groups using that
policy. Reject unsupported settings and incompatible combinations at startup;
do not accept and ignore them.

### Retained behavior

- Keep `load_based` as the CLI name for `LeastLoadPolicy`.
- Preserve bucket membership, ranges, context limits, and rank/ID ordering.
  Restore existing SLO behavior in the separate SLO PR before serving switchover;
  legacy routing continues to support SLOs during this skeleton-only phase.
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
| Round-robin cursor | One cursor per role-group policy instance |
| Capacity exhaustion | One ordered pass; return exhaustion when no permitted group admits the request |
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

A retry rebuilds the permitted candidates, excludes failed engines as required,
and invokes the attached policy. It does not silently replace a cache winner
with a different engine after selection.

For PD, acquire and release accounting for the actual stages and clean up
partial setup on failure. Policy selection does not own the PD request lifetime.
Selection metrics must not imply that dispatch or execution succeeded.

## Implementation status

This PR adds the side-by-side skeleton in `src/buckets_reorg.rs` and
`src/policies_reorg/`. The live `src/policies/` path still serves traffic.

Implemented here:

- `Bucket` with a shared context limit and optional role groups.
- `EngineGroup` with membership, token range, rank, and attached policy.
- `matching_buckets` for role/length matching and rank/ID ordering.
- `pick` for model/health/role/membership filtering and one-pass selection with
  exact candidate validation.
- `Policy::pick`, fallback interface, admission placement, and `AllowAll`.
- Lazy report capture through `LoadView`.

Follow-up order: power-of-two and admission (#40271), then bucket SLO ordering
in a separate PR, followed by the remaining policy and wiring work.

Not yet implemented or wired in this skeleton:

- SLO estimates, request targets, and per-stage preference ordering.
- Concrete power-of-two selection (its body is still a placeholder), other
  policies, and capacity/in-flight admission checks.
- Configuration parsing, validation, model-specific construction, and default
  group synthesis for the new bucket format. The YAML above is illustrative.
- Affinity probes, session modes, prefix memoization, and cache-aware selection.
- Shared load interpretation and dispatch correction.
- Request-handler wiring, PD compatibility filtering, retry integration, and
  removal of the legacy path.

The preceding sections describe the target behavior for those follow-ups; they
do not claim those capabilities are present in this PR.
