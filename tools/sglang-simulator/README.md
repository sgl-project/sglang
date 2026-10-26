# SGLang Simulator

SGLang Simulator reuses SGLang's scheduler and cache implementation while
replacing model forward execution with a latency predictor. It supports
timestamped trace replay, synthetic workloads, hierarchical cache simulation,
and serving-compatible metrics without loading model weights.

See the [SGLang Simulator advanced-feature guide](../../docs/docs/advanced_features/sglang_simulator.mdx)
for the user-facing setup and serving workflow.

## Compatibility

SGLang Simulator tracks the current SGLang `main` branch and maintains compatibility
with recent SGLang releases. The current integration is validated with `v0.5.16`,
`v0.5.17`, `v0.5.18`, and `main`. Compatibility code uses API and capability
checks instead of branching on version numbers.

The native `_build_dsv4_kv_pool` factory, when available, retains its original
dimensions, layout checks and real CPU buffers. Use a small explicit token
capacity for metadata checks. This path does not qualify older packed factories
without that seam, virtualize payload storage or establish large-capacity model
support.

### DSPark integration boundary

The DSPark adapter provides CPU protocols, a native runtime and worker/scheduler
hooks; it does not yet qualify a V4.1 serving run. Existing ordinary-forward predictors
and launch validation keep their current support bounds.

`simulation.sglang.dspark` separates immutable before-work (`DSparkStep`),
complete sequential worker timing (`DSparkStepTiming`) and independently modeled
output progress (`DSparkProgress`). `DSparkSession` checks prior injected or
committed KV ranges, queries cost before progress, and commits history only after
both results validate. Target prefix length cannot initialize draft history.
Each phase cost is already coordinated across the modeled participants; the
simulator adds no TP or layer multiplier. Missing preamble, proposal, target,
accept/commit/publication or tail costs fail before progress advances.

The before-work contract represents TP2/ADP1, TP4/ADP1 and TP2/ADP2. All actual
attention-DP partitions, including idle groups, must be supplied explicitly.
This representation does not implement a DP load balancer or establish measured
support for these layouts. Heterogeneous prefill/verify phases require a separate
source-specific coordination adapter.

Repeated immutable before-work inputs reuse a bounded cache of successful
canonical JSON validation and integer recipe budgets. The cache retains no
parsed mutable objects, preserves the complete original serialized strings, and
does not bypass per-request budget or before-state checks. Its size limits only
retention; larger valid inputs still use the original validation path.

The integration seam is an explicitly configured predictor with
`supports_dspark = True` and a `dspark_runtime` object:

- `forward_batch_generation(worker=..., batch=..., on_publish=...)` returns the
  native SGLang result with a `DSparkStepResult` attached under
  `_sglang_simulator_dspark_step`.
- `note_request_finished(rid=..., natural_stop=...)` releases request state.
- `clear_cache_pool()` clears the adapter's causal state when native caches reset.
- The runtime owns source-correct prefill/KV injection, native result and next
  draft-input construction, logical allocation, publication and cleanup. It must
  preserve full sampling recipes and exact source/runtime profiles when calling
  the existing InferCast UMD/PerfDB paths.

The decode session accepts separate
`cost_model.predict_step(before_step)` and
`progress_model.predict_progress(before_step)` objects. There is no built-in
latency, acceptance fallback, fixed speedup or use of the current measured
acceptance. The scheduler consumes the attached staged timing once, accounts for
predictor wall time, and preserves coordinated idle time. It does not issue an
ordinary one-token DECODE query for that speculative step.

`DSparkNativeRuntime` reads native CPU chunk/prefix/committed-length mirrors and
all fields from the SDK-owned normalized sampling inventory. Prefill logically
injects only the executed range: a target cache hit supplies no missing draft KV.
Intermediate chunks emit no token. Decode obtains independently modeled progress,
builds native `GenerationBatchResult`/`DFlashDraftInputV2`, publishes new lengths,
then commits the causal ledger. Result/publication failures leave that ledger
unchanged. The marked next input uses SGLang's page-aligned reservation sizing,
public allocator and request-table writes on CPU; unmarked DFlash inputs keep
their native path. Synthetic token content follows the ordinary simulator
contract and does not determine progress.

The complete causal KV range stays explicit. Proposal reads project a bounded
128-token window from that range; two successive commits can extend the range
to 130/132 while the source query still reads 128. Unknown gaps and 127-token
history fail. Persistent proposal-buffer capacity is tracked from successful
calls, including the initial allocation gap, and is not reset by a KV-only flush.

`time_predictor.dspark_umd.UMDDSparkCostModel` evaluates prepared source-owned
graphs through InferCast's existing composition executor. Its built-in proposal
preparation and capture resolution use verified FIDB profile assets. Additional
phase preparers must cover their complete worker boundary; a target composition
with uncovered call spans cannot become a complete target cost. Missing phases
fail before progress is sampled. The default runtime execution resolver uses
this same cost adapter rather than a second graph/cost planner.

Native golden regions default to `cost_mode="SILICON"`. Explicit `HYBRID`
additionally requires a `native_analytic_model` `{path, sha256}` asset whose
versioned SDK parameters bind the same execution profile, TP layout and canonical
model configuration. Exact measurements and supported interpolation take priority;
only typed data misses may use the SDK's source-partitioned compute, bandwidth,
launch and communication model. These costs retain `sol` provenance and calibrated
anchor/assumption metadata. Invalid identity, unsupported execution and tampered
data remain failures. Resource initialization, source-bound batching and progress
contracts are unchanged; the exact initial-buffer witness cannot use theory.

Set `native_cost_evidence_path` to a fresh run output to stream every evaluated
phase's complete SDK query evidence as canonical UTF-8 JSONL. The phase reference
hashes the exact line bytes, including the newline. Metrics retain the model,
cost totals by source/phase, theoretical fraction and ledger size/SHA-256. A line
also retains `request_ids_by_attention_dp` in attention-DP rank and native request
order, including empty idle groups. These outer request IDs support forward-to-request
attribution; they do not change SDK query or PerfDB identities, and their presence
does not establish a historical request-body or token-ID join. Each line records a
predicted query, not a committed native forward; a partial run or output
failure must not be reported as a completed simulation. Theory never creates or
qualifies PerfDB measurement rows and does not establish native timing accuracy.

The library-only `UMDDSparkCostModel.for_native_joint` selects one complete
TP2/attention-DP2 worker cost. It accepts a typed `NativeDPProgressPolicy`, the
immutable canonical `runtime_profile_json`, `profiles` keyed by `EXTEND` and
`DECODE`, and optional matching `analytic_models` with the SDK joint policy and
calibration `none` in explicit HYBRID mode. These are the existing SDK profile and
model types; the two phases retain distinct identities. Each typed step group
must carry the same existing `query.to_runtime(source_policy)` object in its
canonical `execution_json`. The adapter binds the full ordered DP geometry,
normalized recipes, source identity and runtime, while the SDK query retains
each DP's buffer, read extent, planner bound and full capture keys.

This route evaluates one `dsv41_native_forward_module` UMD leaf and returns one
exclusive `whole_worker` timing. It never sums ADP1 estimates or splits the joint
worker into fabricated region costs. Exact silicon takes priority; only the
SDK's typed numeric miss can reach its explicit joint analytic model. Source,
profile and invalid-row errors remain failures. Optional JSONL evidence retains
the complete SDK query and ordered request IDs; metrics count the evaluated
whole-worker query once. The test fixture `dspark_joint_declared.json` contains
declared source facts and analytic parameters, with no measurements or timing
qualification.

Callers must supply source-resolved joint inputs; this cost adapter does not
project native before-state or authorize a runtime. The explicit joint source
entry below supplies the separate runtime custody and compatibility boundary.
This library contract does not establish C64 serving, native timing or accuracy.

`time_predictor.dspark_progress.EmpiricalDSparkProgress` reads a hash-pinned
artifact and sibling observation/native/causal-evidence files. It verifies raw
native frames with the shared SDK participant join, checks full recipes and
leader commit labels, and retains peers' native null request records. The model
fits conditional count distributions from training trajectories only, keyed by
explicit prefix/history bins and the full recipe digest. Distinct heldout
trajectories provide visible mean/distribution errors. Native observation IDs
and trajectories cannot overlap splits; unsupported recipes/layouts/conditions
have no fallback. Local batch sizes come from the joined original participant
metadata. Each queried cell and local batch size must occur in both training
and heldout evidence. This is an observed-support gate; the per-cell marginal
distribution is not refitted or claimed to be batch-conditioned. Deterministic
sampling reads only before-work and preserves the native verify commit count,
including commits crossing the remaining output
budget. SGLang's native scheduler owns visible output and finish clipping; KV
advances by the full native commit. Neither loading nor heldout support is an accuracy
admission; causal-history evidence still requires its separate source review.

`conditional_commit_counts(before_step)` exposes immutable per-request native
commit/frequency pairs from the training distribution without drawing an outcome.
It applies the same runtime, recipe and heldout-support checks as prediction.
A source-owned measured phase preparer can use these frequencies when its
outcome-dependent cost composition is justified independently. They are marginal
request distributions: multiplying them into a joint batch law is not supported.
Reading them neither admits a phase cost nor supplies current native acceptance;
missing measurements and complete-boundary checks still fail before sampling.

The explicit `progress_kind="source_golden351"` instead selects the pinned
DSpark match-expected 3.51 source law. It requires an exact nonnegative integer
`model_seed`, no empirical `progress_artifact`, and an SDK-verified native-region
policy whose golden law and stable material identity match. Older SDKs or natural
profiles fail closed. The SDK owns source/material verification and the separate
golden marginal-cost contract; there is no caller-supplied approval flag. The
default remains empirical and retains its existing artifact and identity.

Golden progress represents TP2/TP4 logical B1-B64, ADP1/PP1, with full
static draft Q5/verify Q6 work and length-only output. SHA256 of the model identity
and DECODE forward ID, modulo100, selects commit4 for residues0–50 and commit3
otherwise. Identity includes the explicit seed, complete runtime and verified
golden material. This is an idempotent modeled realization of the 49/51 law,
not replay of native Torch CPU RNG history. Prefill draws nothing; the existing
middle-chunk0/final-bonus1 path remains. Costs resolve before progress, without
shrinking Q or scaling latency by3.51. Full committed lengths advance KV even at
the final OSL boundary; native scheduling clips visible output separately.
Each batch repeats the one event draw across all requests in their original order;
it never draws independently per RID or clips their committed lengths to a
remaining output budget. Every request must satisfy the length-only domain.
The typed SDK source policy and ordinary cost reader still gate ordered batch
work, complete sampling, actual captures and independently qualified direct
whole-batch costs. B1 measurements cannot supply missing batched work.
The logical domain does not replace the actual pool, sampler or padded capture
inventory. B65+ and ADP2 remain unsupported by this ADP1 path. Root client
concurrency does not bound native B, and diagnostic request caps or memory
fractions do not establish equivalence to the reference serving recipe.
Complete newly qualified golden costs are still required for an actual run.

The optional `native_source_work_policy="native_region_golden351_decode_work_v1"`
is an experimental B1 DECODE cost lookup option with `source_golden351`. It passes
the existing typed SDK source policy through runtime-only UMD adapter arguments;
the SDK still verifies helper-file inventory, full capture/planner/buffer/source
identity and qualified original costs. Full request recipes and row identities
remain intact. Exact and ordinary in-domain prefix lookups retain their existing
route. A source-work estimate requires a bounded SDK query record matching the
requested region/profile, selected policy and original bracketing anchors; an
ordinary empirical source string is insufficient. Missing proof fails before
progress or history changes. The selected policy and query proof enter the phase
evidence reference. Omitted options retain existing identities and behavior.
EXTEND and B>1 use their existing exact lookup paths. This plumbing supplies neither
missing costs nor actual heldout validation, complete AgentX coverage or an
accuracy claim; operational activation requires those independent results.
Source-law implementation and CPU tests do not establish native timing or
Simulator/Pareto accuracy. Actual completion/overlap validation and complete
C1 AgentX coverage require independent evidence beyond these CPU contracts.

The source-only ADP2 library seam is separate from this factory. Its
`SourceGoldenDPProgress` requires the SDK's `NativeDPProgressPolicy` and an
explicit canonical cost-runtime context. Each active attention-TP singleton
gets one local event hash shared by its ordered requests; idle groups draw
nothing. Independent per-DP hashes are an explicit model assumption, not a
replay or independence claim for native CPU generators. Each request retains
its own visible OSL clipping and full native commit.

The separate `capture_native_dp_peer_step` native adapter can exchange complete
CPU before-work groups through a real two-rank `GroupCoordinator` CPU collective.
It checks the native scheduler counter and model FID independently, retains each
DP's own buffer state and exact ordered requests, and requires explicit native
idle participation. The returned `DSparkStep` is unresolved: the adapter calls
no cost/progress provider, publishes no result, changes no request history and
advances no virtual clock. This entry point does not supply a joint HYBRID cost
model or complete native peer publication, scheduling or whole-workload support.

The explicit library route `DSparkNativeRuntime.select_native_dp_peer_batch`
freezes each real scheduler's local reference selection after that exchange.
The caller retains the native overlap-decision/result-processing order, then
calls `forward_native_dp_peer` with the same native TP group, local attention-DP
rank/counter and a source-owned `joint_execution_resolver`. The resolver receives
the complete worker-before step, both selected-local CPU reference views and
both local execution candidates. It must return one agreed full step changing
execution JSON only. This is where the existing named joint whole-forward cost
adapter can receive its exact SDK query/runtime binding. The retained-native
`native_joint_group_execution` witness reader is not a projector for future
dynamic simulator work. `NativeJointExecutionResolver` instead derives prospective
coordinates from the local reference views, original request metadata and the
complete physical graph inventory. It never invents a completed native frame.

The existing predictor owner's internal `_native_joint_runtime` assembles this
typed resolver, both phase-specific profiles and uncalibrated HYBRID models,
the source progress policy and the existing whole-forward UMD adapter. It requires
the canonical model digest and an observed reference CPU-mirror boolean; missing
mirror evidence stays unknown and fails before opening a cost ledger. The caller
must independently retain the runtime/capture/allocated-source custody: the
acceptance policy does not attest that physical context. The explicit source
entry below retains the separate original-asset attestor and topology checks.

With this explicit resolver attached, the existing reference driver's `selected`
entry joins the actual native TP group after native MLP synchronization and idle
construction. The existing worker forward entry uses that same group, scheduler
counter and source context for the peer transaction. Rank-local validation errors
rendezvous before costs. Both-empty polling binds the local reference owner
without inventing a forward; paired drained reset needs no fabricated native FID.
The overlap loop may bind its scheduler before that first poll. The first actual
attention-DP rank seals a pristine local ledger independently of the scheduler
owner; subsequent polls and resets cannot change either established owner.
Retirements synchronize at the next selection, outside local result callbacks.

Rank 0 alone predicts the global cost and progress. Each owner binds that shared
result to its own reference transaction and builds its native result. Both local
builds and reference candidates must succeed before either publication callback;
only successful publication on both ranks permits replicated Session/FID commit.
Failures before publication preserve Session history/FID. A publication or commit
failure is sticky terminal; external effects are not represented as rolled back.
Each active decode's persistent draft buffer grows from its actual local native
sequence vector. Idle participation does not allocate or borrow a warm peer's
buffer. Finished local references queue global history retirement until the
explicit `synchronize_native_dp_peer_lifecycle` collective applies the same
drained retirement/reset to both Sessions. Reset preserves the native worker's
persistent draft buffer and forward sequence.

This route advances no clock. The DPC child has a spawn-safe early bootstrap
entry point; that alone does not implement a common clock or authorize ADP2.
CPU transaction controls use the existing runtime/reference owners with a
labeled transport double. Actual native communication and publication still require their own
bounded native controls.

The existing `DSparkStepTiming` also recognizes one exclusive `whole_worker`
cost for a complete serialized TP2/ADP2 two-group step. It cannot be combined
with region phases, used for ADP1, multiplied by DP, or obtained by summing two
ADP1 calls. The existing session queries this one global-FID cost before progress
and commits all causal histories once. A cost provider still owns full joint
geometry, idle participation, per-DP buffers/planners/captures, source labeling
and evidence. Cold buffers start independently; a warm peer supplies no state.
The source policy validates neither the caller's cost-runtime JSON nor latency.
This seam adds no performance row, theoretical formula or qualification and does
not itself enable the native factory, actual peer scheduler routing or mixed
phases.
A complete ADP2 C64 workflow remains a separate validation scope.

Actual multi-DP scheduling/heterogeneous coordination, complete measured worker
costs (including allocation), independently reviewed progress data and target
source resolution remain required for a complete V4.1 AgentX run. The native runtime
supports homogeneous TP2/TP4 ADP1 and the explicitly source-bound TP2/ADP2 route
below. CPU fixtures and native-type checks must not be reported as
actual Simulator accuracy or hardware performance data.

Select the minimal JSON factory with `predictor.name="infercast_dspark"`. It
assembles the existing UMD cost adapter, independent empirical progress model,
session and native CPU runtime for modeled TP2/ADP1/PP1, or TP4/ADP1/PP1 with
verified native golden regions. It binds the local
checkpoint and native text configuration to the SDK's revision-pinned source
configuration, and requires full model and installed-provider commits plus
hash-pinned runtime and progress assets. It opens the selected FIDB database and
records the stack digest; this does not certify its manifest or complete table
coverage. Asset paths resolve from the configuration directory. The
legacy `proposal_profile_id` selection prepares only the existing proposal
module. The explicit native-region selection below prepares all four prefill
and five decode regions through the SDK and installs the source event driver
described below. Construction does not establish measured coverage or accuracy.

The before-work resolver receives the original native CPU batch and authoritative
proposal-buffer capacity/device state before costs are queried. The optional
`capture_simulator_source_facts` helper freezes ordered original-input lengths,
budgets, request-state flags and CPU mirrors. These remain simulator observations;
they do not establish the reference path's mirror availability, target graph bound
or separate verify-plan argument. With a verified native policy, the resolver
passes the source driver's separate `reference_view` and matching ordered request
metadata to SDK `project_native_segment_query`. It retains the returned canonical
module variant unchanged. Native-region phase graphs bind the exact region, ordered work, full sampling,
buffer state, nominal attention read extent, execution profile and actual runtime identity, including a complete
phase attestation for native proposal, through SDK `prepare_native_dspark_phase`.
Native and legacy proposal selectors reject conflicting cost grains.

Native-region schema2 lets the SDK derive nominal cached read extent from the
verified non-ring HIP SWA128 branch and ordered prefix lengths. It does not use
the simulator's valid-written-KV ledger to choose that cost coordinate. Semantic
validity, acceptance and future progress retain their existing state ownership;
the separate legacy proposal selector keeps its saturated-history requirement.
Schema1 region history keys remain distinct and are never relabeled as schema2.
The immutable reference ledger and known CPU mirror vectors retain all ordered
requests through B64 for TP2/TP4 ADP1. Page allocation, publication and FIFO
result retirement still use each request's own geometry and output budget.
This bookkeeping supplies neither measured batch costs nor ADP2 coordination.
The phase binder also transports the SDK's schema3 graph-work queries unchanged.
They retain schema2 nominal read semantics and add explicit padded or larger
capture work; they do not acquire a legacy written-history field. The SDK still
owns capture selection, source policy and cost admission. Binding a query does
not provide missing measured rows.
Native phase preparation requires the SDK source policy, typed step and query
to agree on TP size; the independent graph binder also checks query against step.
Both paths retain a single ADP1/PP1 group with ordered logical B1-B64. The factory
requires configured TP, verified source policy and
progress topology to agree. TP4 uses the same source-law commit3/4 event; it adds
no TP latency scaling. Explicit TP4 B1 source-work reuse follows the SDK's same
versioned policy, requiring all eight target/draft FILE/context inventories,
full capture/planner/buffer/source equality and strict P brackets. The bounded
query proof and original row identities remain mandatory. This does not establish
cross-OSL accuracy or full AgentX coverage. B>1 source-work reuse and ADP2 remain closed.

For a HuggingFace snapshot whose `config.json` links into the cache's blob
directory, set `model_config_root` to the explicit containing checkpoint/cache
root. Both the requested checkpoint and resolved config must remain inside it.
Without this option, the checkpoint directory is the bound. Binding preserves
the requested and resolved paths, checks bounded regular-file bytes against the
SDK source, and rejects link or target changes during native config binding.
It does not rewrite or materialize the cache. This root path also resolves from
the simulation configuration directory.

```json
{
  "name": "infercast_dspark",
  "model_id": "deepseek-ai/DeepSeek-V4.1-Flash",
  "model_revision": "5254448103b6254fc646dacbca1cbbf20e9216c1",
  "provider_revision": "<installed InferCast full commit>",
  "model_config_root": "../checkpoint-cache",
  "systems_root": "../systems",
  "runtime_profile": {"path": "runtime.json", "sha256": "<file SHA-256>"},
  "progress_artifact": {"path": "progress.json", "sha256": "<file SHA-256>"},
  "proposal_profile_id": "<existing proposal profile SHA-256>",
  "output_token_id": 1
}
```

For the native-region path, replace `proposal_profile_id` with both fields:

```json
{
  "native_segment_profile_id": "<existing native region profile SHA-256>",
  "native_capture_profile": {
    "path": "decode-capture-profile.json",
    "sha256": "<file SHA-256>"
  }
}
```

The capture artifact is the existing SDK `DecodeExecutionProfile.to_dict()`
document, parsed with its strict serializer and supplied explicitly; capture
buckets cannot reconstruct its compatibility fields. The selected slice's
existing kernel identities provide profile/measurement references. Legacy
profile1 metadata retains its original source-inventory path. Profile2 requires
measurement2 and each measurement's explicit source cohort. The SDK independently
validates every cohort's startup, capture, driver, proposal and policy-window
receipts. The configured capture document must exactly match each cohort's
canonical capture profile, and all verified prospective policies must be equal.
Partial epochs cannot be combined to establish one policy. Runtime bytes must
exactly match the policy's canonical physical-source document and the
independently bound progress artifact.

With explicit `cost_mode: "HYBRID"` and `native_analytic_model`, an optional
`native_source_binding` can instead select this configuration's original source
cohort without requiring numerical rows for that profile:

```json
{
  "native_source_binding": {
    "root": "native-source-evidence",
    "profile": {"path": "_meta/dsv41-native-segment-profile-<digest>.json", "sha256": "<digest>"},
    "cohort": {"path": "_meta/dsv41-native-segment-cohort-<digest>.json", "sha256": "<digest>"},
    "stack": {"path": "_meta/dsv41-native-segment-stack-<digest>.json", "sha256": "<digest>"}
  }
}
```

The root resolves from the simulation configuration directory and must be a
resolved directory without links. These are the existing SDK content-addressed
descriptors and profile2/cohort formats, read with the SDK's bounded regular-file
reader and verified SHA256 values. `NativeSegmentSourcePolicy.from_cohort`
validates each rank/role's source, startup, graph, proposal and allocated-runtime
facts. The explicit stack descriptor selects the cohort's full original stack;
the profile's separate stable-stack artifact is also verified. Their stable
stack identities must both match the numerical database. Configured runtime,
capture and topology remain exact. This source-only
binding grants no cost qualification. Existing qualified rows remain separately
validated, and missing numerical work follows the explicit analytic SOL path.
Omitting the option retains the existing finalized kernel-identity loader and
does not add a source-binding field to predictor identity.

Measurement `native` windows are separately checked against the slice manifest
and their SHA256; they need not equal the cohort's policy windows. Measurement
proposal paths must match that cohort, with one original proposal receipt per
SDK-verified TP rank. Factory source validation admits no cost:
the existing SDK cost reader still validates original measurement frames, cells
and independent timing qualification before progress can be queried.

Native projection represents TP2 B1-B8 and TP4 B1, ADP1/PP1. Its seven logical
preparers select nine mode-specific regions; costs still use the ordinary UMD
executor and finalized silicon PerfDB. Missing rows remain data gaps.

The virtual proposal buffer starts cold by default. To start a request after
benchmark warmup, explicitly provide `native_initial_draft_buffer` as an existing
`{"path": "warm-before-work-query.json", "sha256": "<file SHA-256>"}` asset.
Its contents must be one complete SDK `NativeSegmentQuery.to_shape()` for an
observed warm `decode_preamble` cell in the same native source/profile/topology
and configured full capture layout. The SDK must resolve it as exact `silicon`;
interpolation, missing cells and another profile are rejected. Only its verified
before-work buffer capacity/device state seeds the modeled resource; its latency
is unused and request/progress history remains empty. Metrics retain the warm
witness identity. This describes an observed warm service state, not automatic
allocation by CUDA graph capture or a physical CPU buffer. Later allocation,
growth and transaction failure rules are unchanged. Cold allocation costs remain
required when the option is absent. The witness does not establish equivalence
to the reference benchmark's initial state or qualify a whole workflow point.

When `native_source_binding` is explicit, the same warm-query asset can instead
be verified against that cohort's original policy windows. The SDK joins the
complete rank set and original forward, preserves the full native input, and
checks before-proposal buffer capacity/device/allocation together with actual
graph and planner facts. At least one complete matching original is required.
This route records `native_source_cohort` / `observed_before_work` provenance and
its forward IDs; it reads no numerical latency row and grants no silicon timing
qualification. The default witness path above still requires exact silicon.

This is the predictor object inside the ordinary simulation configuration;
replace digest/commit placeholders with verified values. Configure the modeled
scheduler/backend and platform normally. There is no JSON callback, latency
override, fallback table or live acceptance option. Generic forward prediction
is rejected: the scheduler consumes the result attached by the native DSPark
worker. Use explicit `--skip-server-warmup`, no custom warmups, `/health` for
readiness with `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`, and a bounded
page-aligned native KV capacity for the initial run;
default warmup and `/health_generate` submit separate unsupported recipes.
The optional InferCast `memory_budget` does not support this path. Coordinated
`OFFLINE` execution requires the native reference profile and a driver implementing
the existing Ready/Grant and timing-metadata protocol; an ordinary HTTP workload
continues to require `BLOCKING`.
Synthetic token content requires `ignore_eos=true` and no token, string or regex
stop rules. Unsupported stop semantics fail before cost, progress or publication.

The progress artifact has `schema_version: 1`, definition
`dspark_recipe_prefix_history_empirical_commit_v1`, `source_pin`,
`runtime_profile_sha256`, `seed`, `prefix_edges`, `history_edges`, `training` and
`validation`. Each split lists `{path, sha256}` descriptors for sibling
`progress-observations-<sha256>.json` files. Each observation document contains
`schema_version`, `source_pin`, `runtime_profile_sha256`, `participants`,
`phase_policy`, `native`, `causal_history_evidence` and `observations`.
Native/causal descriptors retain complete sibling JSON bytes, bounded by the SDK's
`MAX_NATIVE_WINDOW_BYTES` policy. Progress artifacts and observation documents
retain their separate 64 MiB bounds. Each observation
has `native_key`, `forward_ct`, `rid`, `trajectory_id`, `prefix_len`,
`draft_history_tokens` (the source read window, at most 128), full
`sampling_params`, `verify_width: 6` and `commit_len`. Native record identity is
recomputed independently of the authored label; request IDs and trajectories
cannot cross training/validation splits. An observation's declared causal
history still requires separate source-lineage review. No field sets an
automatic timing or accuracy admission.

For additional UMD boundaries, pass source-owned `phase_preparers` to
`UMDDSparkCostModel`. Each returns `(ModelDescription, point, graph_phase)` using
the existing family graph and operation registry. Besides the built-in proposal,
the description must declare `extra.complete_dspark_worker_phase` equal to its
phase. This is a coverage contract for reviewed implementation, not a user
override: it cannot remove `unmodeled_target_call_units` or introduce constant
latencies. Complete worker queries require every phase before querying progress.
An explicit native source policy installs its own complete phase preparers and
rejects additional custom preparers.

## Requirements

- A compatible SGLang checkout. The simulator uses the SGLang source from the
  same monorepo checkout.
- A local model directory containing model configuration files. Tokenizer files
  are also required unless tokenizer initialization is disabled.
- Predictor data for AIConfigurator, ML, or replay mode.

Use an official SGLang image matching the checkout when validating GPU and
runtime compatibility.

## Installation

From the SGLang repository:

```bash
pip install -e tools/sglang-simulator
```

The simulator does not install or pin a second `sglang` package. Run it from a
checkout whose `python/sglang` package is available on `PYTHONPATH`, or from a
matching official SGLang image.

AIConfigurator is optional. Install it separately when using the
`aiconfigurator` predictor or automatic device-memory profiling. Replay, including
its KNN fallback, can use an explicit `max_total_tokens` without importing AIC.
The `aic` extra pins AIConfigurator to the exact release
validated with the simulator so upstream API changes cannot silently alter an
installation. In a clean virtual environment, install the extra with:

```bash
pip install -e "tools/sglang-simulator[aic]"
```

In an existing SGLang image, install the same pin without dependency resolution
to avoid replacing its NumPy/CUDA stack:

```bash
pip install --no-deps "aiconfigurator==0.10.0"
```

Upgrade this pin only after rerunning the AIC predictor and compatibility tests.

## Quick start

The maintained tests define the supported first-version scope:

- [`test/test_simulation_sglang_runner.py`](test/test_simulation_sglang_runner.py):
  direct Python use of the repository-level
  [`SGLangBenchmarkRunner`](../../benchmark/simulator/bench_runner.py);
- [`test/test_simulation_sglang_serving.py`](test/test_simulation_sglang_serving.py):
  server plus benchmark-client use through the HTTP serving path with AIC, ML,
  and replay predictors and ShareGPT or timestamped traffic;
- [`test/test_simulation_offline_blocking.py`](test/test_simulation_offline_blocking.py):
  equivalent logical results in `OFFLINE` and `BLOCKING` modes;
- [`test/test_simulation_cache_hit_ratio.py`](test/test_simulation_cache_hit_ratio.py):
  reusable-prefix accounting and cache-tier hit metrics across repeated runs.

From `tools/sglang-simulator`:

```bash
python3 -m pytest -q test/test_simulation_sglang_runner.py
python3 -m pytest -q test/test_simulation_sglang_serving.py
```

The direct `SGLangBenchmarkRunner` entry point defaults both CPU bootstrap
flags to `1` before importing SGLang, matching the serving benchmark adapter.
Explicit environment values are preserved. The compatibility shim still
requires both flags to equal `1`; native quantization validation remains active.

Read these tests as the minimal maintained examples for constructing a dataset,
running a benchmark, starting a simulator server, sending programmatic, ShareGPT,
or timestamped traffic, comparing execution modes, and collecting request,
latency, throughput, and prefix-cache metrics.

## Serving mode

### Closed-loop workloads

In a closed-loop workload, request completion or a tool timer determines the next
request. The **workload driver** owns these dependencies, its dataset parser,
concurrency, warmup, and stopping conditions. The **simulator** owns model request
scheduling, cache state, and predicted execution time. No dataset schema or agent
framework is required by the server.

Clients can use either existing execution mode:

| | `BLOCKING`: real time | `OFFLINE` with coordinator: fast-forward |
|---|---|---|
| Client | An ordinary HTTP workload driver. | A driver using the coordinator protocol. |
| Timing | Predicted batch latency is spent sleeping; client timers use real time. | Client and server advance virtual time together. |
| Metrics | Client measurements include HTTP, tokenization, and scheduling overhead. | The driver uses simulator timestamps. |

Both paths use the same scheduler, cache, and latency predictor. The model output
is synthetic: the driver must supply response history and workload dependencies
when replaying a trace. These runs do not evaluate generated answers or agents
choosing tools from model output. Results depend on the predictor data;
the bundled replay table is an example, not a hardware performance calibration.

In the explicit CPU simulator, packed DeepSeek KV/index payloads use metadata-only
storage with their native shape and dtype. Integer allocation, request/window state
and logical cache capacities remain real CPU state. This allows observed accelerator
cache capacities without allocating unused model-value tensors on the host; it does
not change native CUDA execution or estimate the capacity itself.

#### Real-time HTTP clients

Start a small example server from the repository root:

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=BLOCKING
unset SGLANG_SIMULATOR_COORDINATOR

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path tools/sglang-simulator/test/assets/qwen3-8b \
  --tokenizer-path tools/sglang-simulator/examples/assets/tokenizer \
  --chat-template chatml \
  --sim-config-path tools/sglang-simulator/examples/sim_configs/replay.json \
  --max-total-tokens 8192 \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --port 30000
```

Point an HTTP driver at `http://127.0.0.1:30000`. It can use `/generate` or the
OpenAI-compatible API and issue dependent requests after responses and tool
waits. No coordinator integration is needed. Requests can run concurrently, but
a five-second tool wait takes five real seconds. For actual traces, replace the
example model configuration and tokenizer and set cache capacity for the workload.
Compare client wall-time metrics separately from simulator logical-time metrics.

#### Fast-forward clients

Restart the server with the same launch command and these environment settings:

```bash
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SGLANG_SIMULATOR_COORDINATOR=tcp://127.0.0.1:30001
```

Run the two-request example:

```bash
python3 tools/sglang-simulator/examples/closed_loop.py --tool-delay 5
```

The tool delay advances virtual time without sleeping. For parallel branches
and a join, restart the server and run
`python3 tools/sglang-simulator/examples/fork_join.py --tool-delay 5`.
Both examples use `/generate`; see [the example guide](examples/README.md) to
adapt the event loop to another dataset. OpenAI HTTP/SSE requests use the same
protocol by including `rid` in the body. One server handles one run; restarting
gives the next run a clean clock and cache.

Plain `OFFLINE` cannot account for a client's future requests or tool waits.
Use `BLOCKING` for clients without coordinator support. Coordination itself
requires `OFFLINE`.

#### Driver contract

The protocol uses conservative event synchronization: the server advances only
when the driver has accounted for all work at the current virtual time. The
transport is JSON over a loopback ZeroMQ REQ/REP connection, separate from HTTP.
[`AsyncCoordinatorClient`](src/sglang_simulator/simulation/client.py) implements
the exchange; it does not execute workloads.

The adapter interface is `AsyncCoordinatorClient` and the `Ready`, `Submit`,
`Cancel`, `Grant`, and `Terminal` message types in
[`simulation.coordinator`](src/sglang_simulator/simulation/coordinator.py).
The server-side `Coordinator`, scheduler bridge, and native request objects are
implementation details; workload adapters should not depend on them.

Protocol version 1 has two messages:

| Message | Fields |
|---|---|
| `Ready` | Stable `run_id`, sequential `round` starting at 0, ordered `operations` (`{"type":"submit","rid":"..."}` or `{"type":"cancel","rid":"..."}`), optional `next_wakeup_ns`, and `finished`. A cancel may include `reason`. |
| `Grant` | Matching `run_id` and `round`, `now_ns`, `terminals` (`rid`, `kind`, `at_ns`, optional `reason`), and `status` (`CONTINUE`, `FINISHED`, or `ERROR` with `error`). Terminal kinds are `COMPLETED`, `CANCELLED`, and `ERROR`. |

Both carry `protocol_version: 1`; only one `Ready` may be outstanding. All times
are integer nanoseconds relative to the run. The earliest pending client timer
is `next_wakeup_ns`; omit it when no timer is pending. Each round follows these rules:

1. Process current completions and timers, including work their callbacks create.
   Start HTTP requests concurrently with `Ready`; its submits form the input
   barrier, so the server waits for every active submitted payload before advancing.
2. Read `Grant` and consume HTTP results for its completed requests before
   releasing dependent work. A timer-only grant needs no HTTP result; other
   requests may remain pending across rounds. Do not wait for all pending HTTP
   requests before sending the next `Ready`, or release dependencies on HTTP
   arrival alone.
3. When stopping, submit required cancels and set `finished`. Continue consuming
   terminals and sending finished rounds until `FINISHED` confirms native cleanup.

Use unique, equal-length request IDs within a run: SGLang's native cancellation
matches ID prefixes. The HTTP pool must admit every request in a round without
waiting for earlier responses; the examples use a dedicated unlimited aiohttp
pool and control workload concurrency explicitly. Protocol watchdogs use real
time to detect failures. They do not advance simulated time, and a timed-out
exchange must not be retried on the same run.

The initial scope is one local driver and scheduler, text generation with
ordinary radix cache, and completion-triggered workflows. HiCache, PD, explicit
sessions, LoRA, and asynchronous grammar are rejected. Speculation is limited to
the explicit source-bound native DSPark profile: its original overlap loop,
FutureMap and reference ledger retain native event order. Every result advances
virtual time by its own forward duration, and already launched native tails drain
before FINISHED even after all requests become terminal. Discarded tails add no
visible output-token event.
Native profile start/stop administration is allowed before the first driver Ready
and after FINISHED with all native results drained. It is rejected during an
active coordinated run so resetting accounting cannot discard pending work.
Coordinated runs preserve native `--prefill-decode-interval` countdowns when the
runtime exposes the validated cadence helper and schedule-pass surfaces. A
cadence-only pass with pending native work consumes no modeled forward time:
it neither advances the clock nor grants a future client timer. Unsupported
native cadence surfaces fail before coordinator binding. Other compatibility
guards, including other speculative algorithms and DP attention, remain unchanged.

Phase deadlines are driver-owned virtual timers; native wall-time waiting/running
timeouts must be disabled. HTTP/SSE transport and predictor execution time do not
contribute to virtual CPU overhead. Unmodeled frontend/CPU costs are zero.
Predicted batch durations must be positive at nanosecond resolution; a zero-cost
fallback is rejected rather than allowing an endless workload at one timestamp.

Startup generation warmup is skipped. The driver may perform workload/cache
warmup, then start its measurement window without resetting cache state. Warmup
requests and time must be excluded from benchmark results.
Administrative resets and profiling requests are rejected. Harness-specific
phase scheduling and metric aggregation remain in that harness's adapter.

#### Output and metrics

For fixed-length trace replay, request the desired output length with
`ignore_eos: true`. Prompt lengths depend on the supplied tokenizer and chat
template; use actual server token and cache counts for metrics. Trace-declared
lengths and synthetic output text do not establish cache reuse by themselves.

Coordinated `/generate` responses include a timing snapshot under
`meta_info.sglang_simulator`. Streaming chat (`/v1/chat/completions`, `n=1`)
emits the same snapshot as an additional SSE event before each native content output:

```text
event: sglang_simulator
data: {"version":1,"rid":"request0","arrival_ns":0,"first_token_ns":10000000,"last_token_ns":30000000,"commit_ns":30000000,"terminal_ns":30000000,"prompt_tokens":7,"completion_tokens":3,"cached_tokens":0}
```

All times are integer nanoseconds relative to the coordinated run. Arrival is
the `Submit` time; first/last token times are native visible-token commits after
the modeled batch finishes. `commit_ns` is the current result's commit time.
`terminal_ns` is null until native completion; first/last are null until a token
is committed. A result with no new tokens leaves first/last unchanged. Counts
come from native prompt, visible output and cache state; cache reporting is
enabled automatically for coordinated runs.

Snapshots use constant space per request and preserve first/last times when
streaming chunks are coalesced or emitted less frequently. They do not contain
per-token timestamps, so they support TTFT, completion latency and mean TPOT,
not per-token ITL percentiles. The initial empty role chunk can precede the first
snapshot. For streaming chat metrics, request `stream_options: {"include_usage": true}`
and validate the terminal snapshot against final usage before consuming `[DONE]`.
Non-streaming `/generate` returns its snapshot and counts in the JSON response.
Cancellation before a token commit retains the native abort response.

The ordinary HTTP path is covered by `test_simulation_blocking_agentic.py`;
`test_simulation_blocking_pacing.py` checks cache-load response timing.
Focused coordinator tests are `test_simulation_coordinator.py`,
`test_simulation_timing_metadata.py`, `test_simulation_coordinated_bridge.py`, and
`test_simulation_coordinated_serving.py`. Run test files in separate processes,
as simulator hooks cache configuration at import time.

### Timestamped serving workloads

Choose a fresh output directory and export it in the server terminal before
starting the server:

```bash
export SGLANG_USE_CPU_ENGINE=1
export CUDA_VISIBLE_DEVICES=""
export SGLANG_SIMULATOR_OUTPUT_MODE=OFFLINE
export SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving-001
test ! -e "$SIMULATOR_OUTPUT_DIR"
export SGLANG_SIMULATOR_OUTPUT_DIR="$SIMULATOR_OUTPUT_DIR"

python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path /absolute/path/to/model \
  --sim-config-path /absolute/path/to/simulator.json \
  --port 30000
```

In the benchmark terminal, export the same output directory before sending
timestamped traffic with the simulator-aware benchmark adapter:

```bash
cd /path/to/sglang
export SIMULATOR_OUTPUT_DIR=/tmp/sglang-simulator-serving-001
export SGLANG_SIMULATOR_OUTPUT_DIR="$SIMULATOR_OUTPUT_DIR"

python3 benchmark/simulator/bench_serving.py \
  --simulator-mode offline \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /absolute/path/to/model \
  --dataset-name autobench \
  --dataset-path /absolute/path/to/trace.jsonl \
  --use-trace-timestamps \
  --num-prompts 100 \
  --profile \
  --output-file "$SIMULATOR_OUTPUT_DIR/benchmark.json"
```

The server and benchmark are separate processes, so exporting
`SGLANG_SIMULATOR_OUTPUT_DIR` in the server terminal does not configure the
benchmark terminal. The benchmark adapter reads `metrics.json` from this path
after profiling and uses those server-side logical-time metrics for its serving
table and output file. If the benchmark points at another directory, it may show
unrelated stale metrics or client wall-clock values. Use the same fresh path in
both terminals for every run.

The simulator always runs the SGLang runtime with `tp_size=ep_size=dp_size=pp_size=1`
and both attention/decode context-parallel sizes set to `1`. Parallel CLI options
accepted by SGLang are therefore ignored by this simulator entry point. This keeps
simulator-only CPU work single-process; it does not change the modeled deployment.
Set the real deployment topology under `scheduler` in `--sim-config-path`. That
topology drives predictor and cache-resource modeling without launching physical
parallel workers.

Other server options are normal SGLang command-line arguments. For direct Python
integration, see
[`test_simulation_sglang_runner.py`](test/test_simulation_sglang_runner.py); for the
process/HTTP path, see
[`test_simulation_sglang_serving.py`](test/test_simulation_sglang_serving.py).

## Simulation modes

| Mode | Behavior |
|---|---|
| `OFFLINE` | Advances the simulator's logical clock without sleeping. |
| `BLOCKING` | Sleeps for predicted forward and visible L2-to-L1 load latency. |

Use server-side simulator metrics for comparisons. Client wall-clock duration is
not the simulated timeline in `OFFLINE` mode. When using the benchmark adapter,
make sure its `SGLANG_SIMULATOR_OUTPUT_DIR` matches the server's output directory
so the printed table and `benchmark.json` are sourced from the current run's
`metrics.json`.

## Configuration

A simulator configuration has three sections:

```json
{
  "platform": {
    "accelerator": {"name": "h20_sxm"},
    "disk_read_bandwidth_gb": 8,
    "disk_write_bandwidth_gb": 8,
    "memory_read_bandwidth_gb": 64,
    "memory_write_bandwidth_gb": 64,
    "num_device_per_node": 1
  },
  "predictor": {
    "name": "replay",
    "database_path": "/absolute/path/to/replay_table.json"
  },
  "scheduler": {
    "tp_size": 4,
    "ep_size": 4,
    "dp_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "cp_style": "none",
    "data_type": "BF16",
    "kv_cache_data_type": "BF16",
    "backend_name": "sglang"
  }
}
```

- `platform` describes the simulated accelerator and storage bandwidth.
- `predictor` selects forward-latency prediction.
- `scheduler` describes the real target deployment topology and backend metadata.
  `tp_size`, `ep_size`, `dp_size`, `pp_size`, and `cp_size` are modeled values;
  they do not launch physical workers. For AIConfigurator, `tp_size` is converted
  to attention TP after removing modeled DP and CP, while `cp_size` is passed as
  AIConfigurator context parallelism. `cp_style` uses the AIConfigurator values
  such as `none`, `allgather`, `ulysses`, or `ring`. Decode-only `dcp_size` has no
  separate AIConfigurator field and is not modeled yet.

### Prefix-cache accuracy

Prefix-cache hit accuracy is highly sensitive to `max_total_tokens`. It controls
the simulated device KV-cache capacity and participates in hierarchical host-cache
sizing, so a mismatch changes eviction timing and device, host, and storage hit
attribution. For deployment-faithful results, copy `max_total_num_tokens=N` from
the real SGLang server startup log and launch the simulator with
`--max-total-tokens N`. Avoid relying on a separately estimated capacity when
comparing the simulator with production traces.

Supported predictors:

| Predictor | Purpose |
|---|---|
| `aiconfigurator` | Operator and module performance-database estimation. |
| `ml` | A trained sklearn-compatible 18-feature latency model. |
| `replay` | Exact or nearest-neighbor batch-composition replay. |

Relative predictor paths are resolved from the simulator configuration location.
Environment variables in paths use `${NAME}` syntax.

Predictors opting into `requires_prepared_batch` receive ordered EXTEND/MIXED
prefix and query lengths, or DECODE past-KV lengths, captured from native
`ForwardBatch` CPU mirrors at model-forward entry. The immutable snapshot travels
with that forward's logits output, so later batch mutations cannot change the
prediction input. Missing, inconsistent, or non-CPU mirrors fail explicitly.

## Workload formats

The Autobench trace format uses timestamps in milliseconds:

```json
{"prompt":[1,2,3],"prompt_len":3,"output_len":1,"timestamp":200}
```

Random and ShareGPT workloads are also supported by the runner API and serving
benchmark paths.

## Validation

Run the CPU compatibility and unit tests from the repository root:

```bash
pip install -e tools/sglang-simulator
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_runner.py
python3 -m pytest -q tools/sglang-simulator/test/test_simulation_sglang_serving.py
```

Run the two files as separate pytest commands because the runner test installs
process-global simulator hooks and state.

The native DSPark CPU type/allocation check is
`tools/sglang-simulator/test/test_dspark_native_results.py`. In the ROCm image
`lmsysorg/sglang-rocm@sha256:b28fe77ae3d1c6022da1b579687a5f5ecec08269081301a4887d2093cfd34ae9`,
AITER imports can require an explicit target architecture even when no GPU
devices are mounted. For a known MI355X target, use `GPU_ARCHS=gfx950` and
`AITER_JIT_DIR=/sgl-workspace/aiter/aiter/jit` to select its packaged read-only
JIT directory. Set `USER` and `LOGNAME` to the intended account name if the
container UID has no passwd entry. Keep all GPU device mounts absent, enable
both `SGLANG_SIMULATOR_BOOTSTRAP=1` and `SGLANG_USE_CPU_ENGINE=1`, and check
`torch.cuda.is_initialized()` remains false before and after the smoke. These
are explicit import settings for the declared target; they do not detect or
establish available GPU hardware. The check exercises real native result types,
CPU request-table buffers and the public allocation seam with protocol fixture
costs; it does not validate measured latency or a full Simulator run.
The joint Scheduler/FutureMap/reference-driver case additionally requires
InferCast's native sampling contract and skips when that optional SDK is absent;
the primitive native result/allocation case remains independent of the SDK.

CPU DSPark argument compatibility is installed only with both explicit bootstrap
flags above. It preserves `device=cpu` and delegates non-CPU calls to the native
handler. The default path requires physical single-rank execution; the explicit
joint source entry below instead retains two native CPU workers. Both require
physical graphs disabled, static draft Q5 / target Q6, an explicit draft path
and full draft revision commit, and positive explicit `max_running_requests`.
Target TP topology remains part of the cost/runtime profile. Unsupported ragged,
adaptive, DP-prefill-coordinated and constrained paths fail before simulator
declarations.
This removes native accelerator-only argument rejection; complete cost/progress
assets and predictor launch wiring remain necessary for a full run.
The native-region predictor selection enables the original overlap scheduler;
other selections retain disabled overlap. The CLI config path is established
before native argument resolution, so CLI and environment config selection use
the same rule.
Native `ServerArgs.from_cli_args()` only constructs the raw record. The simulator
launcher explicitly calls `resolve_once()` before launch; the native engine's
later call is idempotent. Resolution declarations preserve the raw argument
values. Argument smokes must perform resolution before reading the resolved view.
Scheduler selection composes the cadence observer with the reference adapter:
the original native selector runs once inside the cadence boundary, then the
reference adapter observes its returned plan. The native signature, cadence
counter checks and original selection exceptions are preserved.

For offline DSPark timing, the native adapter records its entire CPU replacement
worker call as simulation overhead. The scheduler excludes that wall time from
its CPU contribution while retaining native scheduler work outside the worker.
Before-work capture, database lookup, synthetic result construction, publication
and history bookkeeping therefore do not add host-dependent emulator time to a
complete modeled worker duration.

The ordinary scheduler path binds DSPark shape and timing accounting to the exact
returned result. Offline accounting excludes replacement-worker wall time once
in the host interval containing its launches, including multiple launches before
result processing. Pending results must drain before profiling resets accounting.
After native result processing, its pending-result decrement completes before
the coordinator publishes any post-result terminal or timer Grant.
Coordinator delegation retains its existing clock contract. The native-region
driver uses the original overlap loop; other predictors retain their existing
loop behavior. SGLang Simulator owns scheduling, batching, cache/lifecycle behavior
and virtual time; InferCast supplies cost queries and performance interpolation.

`simulation.sglang.dspark_reference.DSparkReferenceLedger` provides an explicit,
immutable reference CPU bookkeeping seam for homogeneous TP2/TP4 attention-DP1
or TP2 attention-DP2/PP1, local B1-B64, Q6 work. It keeps CPU-processed committed
lengths, persistent reserved lengths, published prefixes, processed output counts
and prefill completion separate. Decode preparation calls native
`page_aligned_decode_alloc_lens` with the original double-buffer reserve; it does
not copy the synchronous simulator's CPU mirror or implement another allocation
formula.

A source adapter must supply the actual event order: prepare the current batch,
optionally process a previous result at a synchronization boundary, publish the
current modeled result, then process the previous FIFO result in the ordinary
overlap path. The FIFO can temporarily contain two results. Preparation freezes
its CPU seed and reserved vector, so later settlement cannot retroactively alter
the planner input. Known absent future indices retain that seed; resolving a
published relay uses the explicitly known backend CPU-mirror policy. Unknown
policy or missing publication rejects rather than becoming a null mirror.

For the TP2/attention-DP2 library path, construct each ledger with its actual
`local_attention_dp_rank` (default 0 preserves attention-DP1). The full global
step retains both ordered groups and true topology. Allocation, CPU vectors,
publication, FIFO settlement, skips and deferred retirement affect only that
ledger's selected group. An explicitly idle group has an empty CPU view and no
local allocation or relay; it still owns its global forward's pending result.
The resolver receives only that selected group and its own before-buffer state,
and the adapter preserves the peer's execution data. After the existing peer
exchange, `transaction.with_resolved_executions(agreed_step)` accepts execution
JSON changes only; all requests, runtime, topology, FID, local CPU vectors and
prior ownership remain frozen. It does not project execution, perform the exchange
or attest its source. Both owners must agree on the complete resolved global step
before the one cost/progress prediction. Call
`transaction.bind_result(shared_result)` before building each local native result:
it checks the complete resolved step and binds the receiving rank, while retaining
the shared timing/progress objects. Publishing a result bound to another rank
fails. This library seam does not coordinate native peers, publish FutureMaps,
advance clocks or open the public attention-DP2 factory/runtime guards.

An internal TP2/attention-DP2 clock bridge places the existing external
Ready/Grant Coordinator in the native data-parallel controller. Both native
schedulers report through their actual CPU TP group; only the leader exchanges
local IPC with that controller. Native routing retains the original dispatcher,
explicit routed-DP requests, native load snapshots and budget algorithm, with
its 20ms refresh cadence evaluated in virtual time. A dispatch generation is
released only after both schedulers acknowledge actual receipt. Bootstrap
administration reaches both native handlers before waiting for the first Ready,
so a client can query server state before starting coordinated admission. One complete
joint whole-worker result advances time once, and both native callbacks and
accounting consumption precede global terminal delivery. Profile boundaries
require both drained owners, combine request statistics using the existing
formulas and write once. The explicit joint source entry below supplies the
separate compatibility and asset validation boundary. CPU transport-double
controls alone do not establish native scheduler execution or C64 support.

All operations return a new ledger. Keep the original until cost lookup,
publication and the surrounding transaction succeed. Only typed modeled
`DSparkStepResult` objects supply pending progress. Settlement preserves the full
native commit count, including output-budget overshoot; finished/retracted skips
are explicit. Drain pending ownership before forgetting a request or clearing
cache state. Prefill allocation sets committed/reserved lengths before forward,
and prefill publication adds no sampled bonus to the KV prefix. Chunk completion
is carried by the existing request's `emits_output` flag.

The native-region factory installs `dspark_reference_driver.DSparkReferenceDriver`.
Scheduler hooks retain the native `event_loop_overlap`, `get_next_batch_to_run`,
`is_disable_overlap_for_batch`, `run_batch` and `process_batch_result` methods.
The driver freezes ordered selection/allocation before the native synchronization
decision, then captures worker-before output counts after optional previous-result
processing. Ordinary previous results settle at the actual post-publication FIFO
processing event. A launch can therefore precede the finish of an earlier result.
The driver has no membership selection algorithm. CPU request/cache allocation
and FutureMap publication remain native operations.

Physical reference backend policy remains a required source gate. A present relay
requires `reference_needs_cpu_seq_lens` on the SDK-verified native source policy.
The current SDK does not expose that verified field, so decode relay activation
fails explicitly with `reference_backend_cpu_mirror_policy`. CPU `torch_native`
FutureMap policy validates only the simulator's own mirror; it cannot supply the
physical HIP backend OR/TBO decision. Original observed null mirrors are not a
global policy proof. No JSON option or guessed boolean bypasses this gate.
Prefill/absent-relay preparation can still reach the existing SDK cost path.
Predictor metrics report driver binding, published/pending counts, this missing
source fact, and terminal failure. These are execution diagnostics, not performance
qualification or accuracy evidence.

The opt-in `dspark_reference_adapter.DSparkReferenceAdapter` consumes those
explicit events and defers request retirement. A source finish or retraction
marks the request inactive; an already prepared or published result retains its
ownership until processed. Later results for that retired request are explicitly
skipped, and the adapter forgets its metadata only after all owners drain. This
covers a finish that occurs while the next reference batch is already in flight.
For a natural finish, process that finishing result before issuing retirement;
retirement before processing represents an existing finish/retraction/abort.
It does not equate G's finish callback with that source event: native prefill
finish and retraction do not uniformly pass through the worker callback. An idle
event must process the actual pending FIFO; reset requires an already drained
ledger. The production driver uses these same adapter transitions. It observes
finished state around native result processing for both prefill and decode,
including prefill finishes that have no draft-worker callback. Never-admitted
queued cancellation owns no reference row. Retraction or retained-output re-entry
fails explicitly until a source-backed rebootstrap transition exists.
The native `_release_aborted_request` hook also retires an admitted queued chunk
when no later result exists to carry its cleanup event.

The same module's pure `prepare_reference_step` bridges a provisional typed step
to a source-aware execution resolver. Callers explicitly provide page size,
relay/backend policy, whether to process the previous result before forward,
and any previously known skips. Preparation freezes allocation before optional
previous-result processing. The resolver receives the existing group, runtime,
native batch and buffer arguments plus an immutable `reference_view`. Only
`execution_json` changes in the final step; ordered request facts, including
worker-before output counts, remain the caller's facts. Buffer metadata remains
authoritative. The original native batch is not rewritten.

The returned `ReferenceStepTransaction.complete(result, current_ledger=...)`
requires the owner's current ledger to be the original base object, publishes
the exact final step's modeled result, then retires only the previously frozen
FIFO head when appropriate. With no previous result, the current result remains
pending. It accepts no new synchronization or skip decisions from current
acceptance. Compute this candidate before external publication/session commit
and install it only after those operations succeed; the pure helper does not
perform or roll back external side effects. The native factory's existing cost
model serves as this resolver. The production driver additionally passes its
frozen `allocation_ledger`, computes the completion candidate before publication,
and leaves ordinary previous settlement to the actual native event by calling
`complete(..., process_previous_after=False)`. An external callback failure is
terminal and recorded; its effects cannot be rolled back or silently retried.
The standalone pure bridge remains available for explicit source-event tests:

```python
transaction = prepare_reference_step(
    reference_ledger,
    unresolved_step,
    page_size=256,
    future_indices_ready=source_future_indices_ready,
    needs_cpu_seq_lens=source_needs_cpu_seq_lens,
    process_previous_before=source_sync_previous,
    skipped_previous_ids=source_skipped_previous_ids,
    execution_resolver=predictor.dspark_runtime.session.cost_model.execution_for,
    native_batch=source_worker_before_batch,
    buffer_capacity=before_capacity,
    buffer_device_matches=before_device_matches,
)
prediction = predictor.dspark_runtime.session.predict(transaction.step)
```

The driver must supply reference-order request membership, abort/embed/multimodal
flags, relay/backend policy and causal history before current progress. Request
IDs, budgets and complete normalized sampling must match the step. The SDK gets
reference CPU vectors, including known absence, separately from synchronous
simulator mirrors. EXTEND's SDK decode-only arguments are None; the outer step
still retains its real persistent buffer state. Unknown inputs reject explicitly.
The ordinary non-native runtime callback retains its existing behavior. Native
workers require their matching source selection and overlap decision; directly
calling a native worker without those events fails. Costs still precede progress,
and result construction/stale-owner failures install no current session/reference
success. Idle preserves history; a drained cache reset clears both histories while
preserving proposal capacity and monotonic forward IDs. Reference event emulation
outside the worker is excluded once from offline host timing, alongside the full
replacement-worker wall interval. No scheduler or cost lookup is duplicated.
Cancellation after publication or native response processing makes the driver
terminal and propagates unchanged. A successful profile reset clears prior wall
credits while preserving runtime history/buffer ownership; failed saves preserve
those credits. An already-launched successor still incurs its cost when its
finished request is skipped, but adds no response/ITL event without new output.

CPU regressions include the native overlap loop body loaded from this checkout,
installed scheduler/worker hooks, cached chunks/final bonus, dynamic membership,
both settlement orders, retirement/cancellation, callback/stale failures and
missing real SDK costs before progress. Protocol costs and source-policy fixtures
are explicitly synthetic test inputs. The next native smoke must use a verified
physical backend policy, native SGLang/torch types, exact runtime/progress/cost
assets, and the unchanged AgentX 1.0.7 client. It must preserve warmup handoff,
DAG/credit completion and native response pacing. No complete 13/priority-10
configuration or 393-trace Pareto result is established by these CPU checks.

The optional `test_native_overlap_scheduler_future_map_and_reference_driver_drain_together`
smoke uses installed SGLang `Scheduler`, `FutureMap`, `ScheduleBatch`, DSPark worker
and result types. It runs the production hooks and original overlap loop,
selection/decode preparation, launch and native result processor for one prefill,
consecutive decodes, length finish and an already-launched successor. It checks
publication before previous-result processing, relay event waits, terminal
reference/history cleanup and exactly-once accounting. Model initialization,
initial prefill admission, memory resources, telemetry and CPU stream/events are
explicit fixtures; costs, progress and reference mirror policy are synthetic.
The native scheduler's grammar barrier is accepted on an explicitly grammar-free
batch, matching the native worker's conditional use; grammar progress still
rejects. This smoke does not establish physical mirror policy, silicon timing,
AgentX completion or coverage of every synchronization/dynamic-membership path.

In the matching pinned SGLang/PyTorch CPU environment, with Simulator and the
matching InferCast SDK source mounted at `/work/sglang` and `/work/infercast`, run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/work/sglang/tools/sglang-simulator/src:/work/sglang/tools/sglang-simulator/test:/work/infercast/src:/work/infercast \
python3 -B -m pytest -q -ra -p no:cacheprovider \
  /work/sglang/tools/sglang-simulator/test/test_dspark_native_results.py
```

The subprocess enables the CPU bootstrap itself. Missing torch/SGLang explicitly
skips these native checks; a skip is not a successful native smoke. In the fixed
ROCm image's no-GPU CPU probe, `USER`/`LOGNAME` and its existing read-only AITER
JIT cache must be available. `GPU_ARCHS=gfx950` selects that image's AITER build
target; it does not observe hardware or establish a physical backend policy.

Run repository checks before submitting:

```bash
git ls-files -z tools/sglang-simulator | \
  xargs -0 env SKIP=no-commit-to-branch pre-commit run --files
```

Runtime changes should also be validated in a matching official SGLang image
with both `OFFLINE` and `BLOCKING` modes. Predictor changes should report
step-level error, and scheduler or cache changes should compare request latency,
throughput, and prefix-cache reuse against measured traces.

## Optional committed-output ledger

For coordinated OFFLINE runs, set
`SGLANG_SIMULATOR_TOKEN_EVENTS=/absolute/path/to/token_events.jsonl` to retain
native visible output-token commits. The parent directory must exist and the
file must be new. With the variable unset, the recorder is disabled and does no
file I/O. Recording does not change virtual batch costs or scheduling.

The JSONL stream starts with
`{"event":"start","version":1,"clock":"virtual_ns"}`. Each output-producing
batch writes an `output_commit` containing integer `at_ns` and `commits` entries
with `rid`, zero-based `output_index`, and positive incremental `token_count`.
Counts come from native visible output progression, including the first prefill
token and multi-token commits. Empty/chunk-only/aborted results add no tokens.
Previously committed tokens remain in the stream when their request is later
cancelled. This measures output, not speculative or cancelled computation whose
tokens were discarded before becoming visible.

Only successful coordinator completion writes a `finish` footer containing
`run_id`, integer `at_ns`, `status: "FINISHED"`, and `total_output_tokens`.
Interrupted or failed streams have no success footer and must not be treated as
complete results. Writes are flushed per batch; a write failure aborts the
coordinated run. Existing files are never overwritten.

The ledger contains no workload phases. A caller supplies exact integer virtual
phase boundaries and request ownership separately. For fixed-duration throughput,
sum the selected profiling requests' commits in `[start_ns, start_ns + duration_ns)`
and divide by that configured duration, retaining output from requests later
cancelled. Exclude warmup by request ownership as well as time: warmup completion
and profiling start can share a timestamp. Report grace-period output and successful
request latency cohorts separately. An early-ended measurement is incomplete and
must not be presented as a full-duration throughput point. Legacy request statistics'
requested `output_length` are not a substitute for this ledger.

The AgentX reference output-token throughput counts output from successful
profiling requests over their observed earliest-start/latest-end span. It uses a
different output cohort and denominator; report it separately from fixed-window
committed throughput.
# Explicit joint source construction

The existing `infercast_dspark` provider accepts `native_joint_source_binding`
only with `cost_mode: HYBRID`, `progress_kind: source_golden351` and a modeled
TP2/attention-DP2/PP1 topology. This explicit selection enables the existing
two-worker CPU entry and paired clock; it does not establish C64 workflow
completion, native timing qualification or accuracy.

Early compatibility reads only that selection intent. It retains physical CPU
TP2/native-DP1/attention-DP2 (singleton attention), PP1/CP1/EP1 on one node,
original overlap scheduling and disabled physical CUDA graphs. For this explicit
mi355x joint domain, native model-feature validation uses a function-local HIP
feature view so the source's explicit decoder SWA bounded replay remains enabled
on the CPU simulator. The native validator code and other restrictions are
retained; the physical runtime platform and raw/resolved cache flags are unchanged.
Full source attestation still belongs to the provider constructor. CPU DSPark
resolution additionally requires DP LM head, native attention-DP local control
broadcast, built-in TP MoE (`none`), static draft Q5/target Q6, no two-batch overlap
and no DP speculative prefill coordination. The native prefill/idle versus decode serialization, routing and
request-pool division remain unchanged: do not divide `max_running_requests`
again. The modeled source-law sampler does not enable native simulated
acceptance. Coordinator use retains OFFLINE output, loopback TCP, virtual
deadlines, disabled LoRA/HiCache/disaggregation and the original drained profile
boundaries. Other predictor selections retain the original single-worker
overrides and ADP1 validation.

The binding has exactly `root`, `source_runtime`, `startup`, `proposal`, `windows`,
`captures`, `allocated`, `profiles`, `container` and `supplemental_files`. Under
the resolved root, all originals are byte-preserved regular `_meta/<basename>`
assets addressed by standard `{path, sha256}` descriptors. Keep allocated
basenames referenced by proposal receipts. Counts are four startup roles, two
proposal paths, two windows, four graph inventories and two allocated pools.
`profiles` maps `EXTEND` and `DECODE` to their own standard profile descriptors;
the separate `native_analytic_model` maps the same phases to explicit joint,
uncalibrated model assets. Regional/proposal selectors and warm-witness options
cannot be combined with this joint binding. Each live DP buffer starts cold and
subsequently follows the actual worker's source-owned state.

Construction reuses the SDK startup, allocated-pool and capture validators and
reconstructs the complete normalized `runtime_profile` from those originals.
The retained container configuration supplies finite otherwise-unobserved
environment coordinates; missing configured values are declared defaults, not
observed process absence. Supplemental source FILEs establish conditional source
semantics, not loaded alias equivalence. Missing mirror observations remain
unknown and fail. Both source phase profiles/models remain independent of numeric
row qualification; only typed numeric misses may use the existing joint SOL path.
One whole-worker price represents the coupled forward, with no invented regional
prices or ADP1 cost summation.

Only this explicit domain maps native `dp_size: 1, attn_dp_size: 2, tp_size: 2`
to modeled attention-DP2. Native process counts and raw arguments are unchanged;
the model still has two chips, shared MLP TP2 and singleton attention TP1. Other
predictors and the default ADP1 provider retain their existing mapping.
