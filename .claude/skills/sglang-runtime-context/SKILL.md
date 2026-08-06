---
name: sglang-runtime-context
description: How SGLang's runtime configuration and process-global state are organized (RuntimeContext tiers, publish + namespace config bags, the pristine ServerArgs seed, override entry points, resource/stream/buffer leases, per-forward flags), the CI guardrails that enforce the design, and the idioms for developing and testing against it. Load this before touching server_args, model overrides, module-level state, or per-forward state in sglang.
---

# SGLang runtime-context architecture

One container owns process-static runtime state: `sglang.srt.runtime_context.RuntimeContext`
(a process singleton reached via `get_context()`). Everything below is a tier on it.

| Tier | Accessor | Holds | Lifecycle |
|------|----------|-------|-----------|
| raw config seed | `get_server_args()` | the published **pristine** `ServerArgs` (resolved-at-startup record; kept for debugging, dumps, per-runner fork copies) | published at process entry; re-publish is **last-publish-wins** (in-process tokenizer build, multi-Engine) and re-projects the bags; read-only |
| resolved config | `get_exec()` `get_memory()` `get_schedule()` `get_model()` `get_spec()` `get_serving()` `get_observability()` `get_disagg()` `get_lora()` `get_mm()` `get_device()` | namespace **config bags** — the single source of truth for resolved config; leaves are real attributes (dynamo-traceable) | projected from `server_args` at `publish`; mutated only via `get_context().override` |
| runtime flags | `get_flags()` | state that is *not* a pure function of config: `capture` (cuda-graph lifecycle), `moe` (ACTIVE backends, swappable), `dp` (DP-attention runtime flags) | materialized at subsystem init; groups offer `override()` for tests |
| resources | `get_resources()`, `get_stream(name)`, `get_buffer(name, factory)` | process-level handles: graph pools, EPLB state, EP dispatcher state, named side streams, workspace buffers | lazy; cleared by `reset_context()` |
| per-forward | `get_forward()` | forward-scoped flags (multi-stream switch, MoE output buffer, attn-TP inputs, extend-in-batch) | contextvar-backed; `scoped(**kw)` restores on exit; new threads see defaults |
| parallel | `get_parallel()` | **dual**: live topology (tp/pp/moe/attn sizes, ranks, groups — `@property`, read-through) *plus* parallel config-bag leaves via `__getattr__` | live: after dist init; config leaves: after publish |

`reset_context()` (unit-test teardown) drops the published config and installs fresh
flags/resources/forward tiers.

## Config: publish + namespace bags

**`ServerArgs` is a pristine seed. Business code never reads it for decisions —
resolved configuration lives in the namespace bags.**

- Every publishing process entry calls `publish(server_args, role=...)`
  (`run_scheduler_process`, the Ray `SchedulerActor`, the DP controller, tokenizer,
  encoder, weight-cache daemon, launcher, ...). The one deliberate exception is the
  detokenizer: its processes never publish and read only the raw config handed to
  their constructors — code that can run detokenizer-side must not use the
  namespace accessors. `publish` snapshots the resolved field
  values into the config bags; the accessors (`get_exec()` etc.) fail closed before it
  runs. `role` records which process type published, and keys per-role namespace
  enforcement: `SGLANG_ROLE_NAMESPACES=record` audits which namespaces each role's
  process actually reads (per-pair persisted via `SGLANG_ROLE_NAMESPACES_OUT`;
  reads inside torch.compile-traced code are NOT observed — audit with
  compilation disabled before restricting a role), and
  `=enforce` fails closed on bag reads outside the role's `ROLE_NAMESPACE_SETS` entry
  (`None` = full tree; only audited roles are restricted).
- Bag membership is metadata on the dataclass: every `ServerArgs` field carries
  `NS("path")` (e.g. `NS("exec.moe")`); coverage is linted two-way
  (`test_server_args_namespaces.py`, `test_runtime_context_config_bags.py`).
- **Reading config**: `get_<ns>()[.sub].field` — e.g.
  `get_exec().moe.moe_a2a_backend`, `get_schedule().max_running_requests`. Bag leaves
  are plain instance attributes, safe inside `torch.compile`-traced code.
- **Mutating config after publish**: the ONLY entry point is
  `get_context().override(source, **fields)`. It writes the bag leaves in place
  (namespace readers see the new value) and records provenance in the overrides log.
  There is **no write-through** to the `ServerArgs` instance — it stays pristine.
  There is no in-place mutation entry on the instance at all: it is read-only after
  resolution.
- **Late launcher-stage resolution (pre-publish)**: a few rules cannot run inside
  `__post_init__` — LoRA normalization, and the auto-parser detection that needs a
  tokenizer/chat-template load. They are resolution, not mutation, and they write
  **in place** via `arg_groups.overrides.declare_late_resolution(server_args,
  source, **fields)`, which refuses the published instance. In place is the point:
  every holder of that object must see the resolved value — the HTTP server, the
  multi-tokenizer workers it is serialized for, the schedulers it forks. Returning a
  variant here is a bug: the launcher rebinds its local and everyone else keeps the
  unresolved object.
- **A config another runner / worker / process is built from**: `server_args.derive(
  source, **fields)` returns a variant (an encode worker's `base_gpu_id`/`tp_size`).
  The receiver — and any bags projected from it — are untouched; resolution does
  **not** re-run, so do not reach for it to "re-resolve" a config.
- **Per-runner values inside one process are constructor arguments, not a variant.**
  The draft worker's `context_length`, load format and attention backend travel as
  arguments to `TpModelWorker` / `ModelRunner` and live on the runner
  (`ModelRunner.draft_attention_backend`, `kv_cache_dtype_str`, …), because target
  and draft coexist and the process-wide bags can only describe one of them.

**Why a bag override cannot stand in for the last two.** The bags are projected at
publish *from the instance's fields*, so anything the runtime must read has to be on
the instance before publish — an override afterwards puts instance and bags back out
of agreement, and whole-object readers (`ModelConfig.from_server_args`,
`build_load_config`, `MMEncoder`'s own `self.server_args.X`) never see it. And bags do
not cross a process boundary: a child publishes from the object it receives and
re-projects its own bags, so a parent-side override is lost. Values that feed
construction before any bag exists (group init reads `server_args.tp_size`) have no
bag to override at all.

- **Nested publishes**: a construction step that must publish a private copy wraps
  itself in `get_context().preserve_config()` — the enclosing lifecycle, including its
  post-publish overrides, is value-snapshotted and reinstated on exit. The draft build
  no longer needs it: per-runner values are constructor arguments now.

### Reads that legitimately stay on a `ServerArgs` instance

- **Per-runner (fork) fields** — fields the draft-worker deepcopy rewrites
  (`attention_backend`, `prefill/decode_attention_backend`,
  `speculative_draft_attention_backend`, `skip_tokenizer_init`, `context_length`,
  `load_format`, `json_model_override_args`, `kv_cache_dtype`): each runner's copy is
  authoritative for that runner, so runner code reads `self.server_args.X`, and
  *resolved* per-runner values live as runner attributes
  (`model_runner.kv_cache_dtype_str` is the pattern — threaded to consumers as
  constructor args, never backfilled onto shared objects).
- **Per-instance boundaries** — the tokenizer-manager family, everything under
  `entrypoints/`, and the tokenizer-process multimodal processors read
  `self.server_args`: several `Engine`s can share one process, and the process-global
  bags are last-publish-wins across engines. `base_gpu_id` also differs per worker
  (the encode-server DP workers each specialize their own copy), so no process-global
  value can stand in for it — `BaseMultimodalProcessor._fast_image_processor_device`
  is the shape to copy.
- **Whole-object passes** (`f(server_args)` handing the instance along) keep the
  supplied-instance contract; don't rewrite the parameter reads to bag reads unless the
  field is runtime-mutated (see the elastic-EP `ep_size` case in
  `eplb/expert_location.py`).

### `get_parallel()`: config leaves vs live topology

Config leaves (`nccl_port`, `enable_dp_attention`, `dp_size`, `ep_size`,
`dwdp_size`, ...) resolve through the parallel bag; live topology (`tp_size`,
`attn_tp_group`, ranks) are `@property` and **win on name collisions**. Five topology
sizes are live-shadowed (`tp/pp/dcp/attn_cp/moe_dp_size`): a config-intent read of
those must stay on `server_args.X` — the live property always wins on the accessor.
Fail-loud is narrower: before dist init, any live size/group read raises; after it,
only the DCP group is optional (`_DCP` exists only when `dcp_size > 1`; attn-CP and
moe-DP always install, as size-1 aliases if unused). `ParallelContext.__getattr__` is deliberately dynamo-traceable (no
`object.__getattribute__`); gate helpers like `enable_moe_dense_fully_dp()` run inside
compiled model forwards (`test_parallel_config_leaves_trace_under_torch_compile` pins
this).

### Mid-resolution reads (inside the pipeline only)

Resolution itself still runs in `__post_init__`: handlers and hooks read the
in-flight state through `resolved_view(server_args)` / `self._resolved()`, fields are
read-only during resolution, and declarations materialize once at the very end of
`__post_init__` (gate order, last writer wins) — *then* `publish` snapshots the
resolved values into the bags. `resolved_view` is pipeline-internal
(`server_args.py` / `arg_groups/`, plus helpers the pipeline itself invokes
mid-resolution, e.g. `adaptive_spec_params`); do not introduce new
out-of-pipeline call sites.

### Adding a model-specific config adjustment

Never assign `server_args` fields from model code. Declare instead
(`sglang/srt/arg_groups/overrides.py`):

- Constant per-arch values → `MODEL_OVERRIDES["MyArchForCausalLM"] = {...}`.
- Derived values → `@register_model_override("MyArchForCausalLM")` returning a dict; the
  callable receives *pristine* `server_args` + `hf_config` and must not write.
- Normalization that must see earlier declarations → a post-process pass invoked via
  `run_post_process_pass` at its slot (reads a view, returns a declaration dict).
- Values only knowable at weight-load time → `declare_load_time_override(source, {...})`
  — validates the whitelist, then routes through `get_context().override` (**bag-only**;
  the declaration lands on the published bags, not on any `ServerArgs` instance).
  Scope caveat for draft models: only a draft build that publishes a private copy
  under `preserve_config` discards its declarations with the scope. Draft loads
  that skip publish share the process bags, so their declarations land
  process-wide — declares reachable from a draft load must be draft-safe (guard
  or same-value).

Declarable fields form a whitelist: `Arg(..., resolvable=True)` in the `ServerArgs`
dataclass. A declaration against a non-whitelisted field fails at its slot.

### Load-time vs resolution-time (critical)

`__post_init__` runs in the launcher process before any model/platform import. Logic that
consults an **extensible registry** (e.g. out-of-tree platforms registering attention
backends in `init_backend()`, which runs at `model_runner` import) must stay at load time
(ModelRunner init), writing through `get_context().override()`. Before moving any
load-time logic into resolution, verify everything it reads is already complete at
construction time.

## Runtime flags (`get_flags()`)

For state that init-time code *derives* and runtime code reads — parsed enums, platform
probes, swappable ACTIVE values. Not for config mirrors (read the bag leaf instead).

- Groups are typed dataclasses on `Flags` (`capture` / `moe` / `dp`): typo-safe writes,
  transactional test-only `override(**kw)` context manager.
- `flags.moe` is materialized by `initialize_moe_config(server_args)` at scheduler init;
  accessors (`get_moe_a2a_backend` etc.) are thin shims with lazy defaults. The speculative
  contexts (`speculative_moe_backend_context`) swap the ACTIVE leaves around draft forwards.
- `flags.dp` is materialized by `initialize_dp_attention`; `is_dp_attention_enabled()` is a
  shim over `flags.dp.enabled`.
- Adding a leaf: declare the dataclass field with a default equal to the pre-init behavior,
  materialize it at the owning subsystem's init, keep any public accessor as a shim.

## Resources (`get_resources()`)

Named slots + two keyed-lazy registries:

- `get_stream(name)` — get-or-create a named CUDA side stream; `set_stream(name, stream)`
  installs explicitly. **Name leases by subsystem ROLE**: all model alternate streams share
  `"alt"`; the offloader's copy stream is `"offload"`; DP-TBO comm is `"dp_tbo_comm"`; LoRA
  side stream is `"lora_side"`. Two call sites may share a name only if their work belongs
  on one stream — sharing across roles serializes intended overlap.
- `get_buffer(name, factory)` — get-or-create a named persistent buffer. Grow-only or
  per-device semantics manage their `resources.buffers` entries directly (see tokenspeed /
  SM120 split / Marlin workspace). Buffer names are per-backend today; do not silently
  share.
- Singletons with manager semantics (EP dispatcher buffers, EPLB recorder/metadata, graph
  memory pool) keep their owning accessors/classes as facades; only the *state* lives in a
  resources entry. Preserve exact semantics in the shim: lazy defaults (the EPLB recorder
  defaults to a Noop instance, not None), publish-once asserts, event-reuse contracts.
- Stream/buffer creation is a driver call — it must happen outside cuda-graph capture;
  keep lease points at init/warmup time.

## Per-forward flags (`get_forward()`)

Contextvar-backed; a new thread sees the defaults; `scoped(**kw)` is the regular write path
(transactional, restores on exit and on exception); `set(name, value)` exists for legacy
sticky setters (`is_extend_in_batch` is intentionally sticky within a thread). Use this
tier for anything set-per-forward and read-within-forward. Before adding cross-thread
state here, prove the readers' thread affinity: contextvars do NOT propagate to already-
running or newly spawned threads. Note TBO ("two-batch overlap") interleaves ubatches on
ONE thread — do not design for TBO threads that don't exist.

## Testing idioms

- **Force a code path by overriding causes, not effects**: compose
  `get_context().override_server_args(**fields)` (publishes a fresh dummy-boundary
  `ServerArgs` carrying the overrides AND projects the bags — `with`-scoped, or
  `install()`/`restore()` + `addCleanup` for fixture-lifetime use) +
  `get_<ns>().override(...)` (scoped override of one bag's own leaves) +
  `get_parallel().override(...)` (live topology) + `get_flags().<group>.override(...)` +
  `get_forward().scoped(...)`. All are scoped and transactional. Tests control execution
  through the context — do not hand-build and publish config objects.
- **Never monkeypatch import bindings** (`module.get_x = lambda: ...`) and never fake a
  config source with a `SimpleNamespace` stand-in: production reads the published bags,
  so a faked accessor silently stops intercepting after any reader migration. Publish
  for real (`override_server_args(...)`), then adjust bag leaves with the scoped bag
  `override` where the constructed `ServerArgs` cannot carry the value (e.g.
  `get_device().override(device="meta")`).
- Mocked runners/managers still need the **per-runner instance attributes** the code
  under test reads (`kv_cache_dtype_str`, `server_args` for whole-object passes) — set
  them explicitly on the mock; `MagicMock(spec=...)` raises on attributes that only
  exist post-`__init__`, which is the fastest way to find a missed stub.
- `reset_context()` in teardown when a test publishes outside a scoped override.
- `ServerArgs(model_path="dummy")` early-returns `__post_init__` (no materialization, no
  strict guard) — fine for lightweight fixtures.
- **Run changed test files per-file** (own process), the way CI does: a monolithic local
  pytest run lets a context published by an earlier file mask a missing-publish bug in a
  later one.

## Guardrails (these fail CI; what to do when they fire)

1. **Strict mutation guard** (always on): bare `server_args.x = ...` after resolution
   raises unconditionally in `ServerArgs.__setattr__` — this *is* the guarantee that
   no writer can desync the bags, so there is no writer ratchet any more. Change
   resolved config with `get_context().override`, build a per-runner config with
   `server_args.derive`. Projected bags are sealed the same way (leaf assignment
   raises).
2. **Mutation ratchet** (`test_server_args_mutation_ratchet.py`, exact pin 0 over the whole
   package minus the pipeline / multimodal_gen): textual scan for assignment forms. Never
   raise the baseline.
3. **Derive contract** (`test_server_args_derive.py`): deriving leaves the receiver
   intact, a published config still refuses assignment, and deriving does not publish.
   Rerouting a writer to the bags means flipping **all its readers in the same commit**
   (no transitional dual-write).
4. **Legacy-accessor ratchet** (`test_legacy_global_ratchet.py`): `get_global_server_args`
   call sites must not grow — new code uses `runtime_context.get_server_args()` (and
   business decisions should read the bags).
5. **Module-state ratchet** (`test_module_state_ratchet.py`): `global` statements in the
   flag-owning layers are pinned by name. A new module-level runtime global belongs on a
   flags group / resources slot instead; migrating a pinned survivor must shrink the pin.
6. **Namespace coverage** (`test_server_args_namespaces.py`,
   `test_runtime_context_config_bags.py`): every `ServerArgs` field carries `NS(...)`
   metadata and the projected bags must cover the fields exactly (two-way).

Never module-skip a test "until the migration settles" — seed the context instead
(the deferral ratchet that once pinned this is retired; the rule stands).

## Hard-won pitfalls (check these before/while refactoring)

- **Moving code drops first-line guards**: early returns (`if self.is_draft_worker: return`)
  are the easiest thing to lose when relocating a method body. Every draft is built
  under a preserved publish of its own config: the scheduler makes the copy with
  `draft_server_args_copy()` (seeded from the resolved config, so load-time overrides
  carry) and publishes it around the worker factory, and `build_draft_tp_worker()`
  nests the same shape for dflash / dspark. The publish ends when construction does —
  anything the draft reads later (`alloc_memory_pool`, `init_attention_backends`,
  cuda-graph capture) is back on the target's bags.
- **Registry-completeness timing**: a gate that consults an extensible list is only correct
  after the registrars ran (platform `init_backend()` at module import). See "load-time vs
  resolution-time".
- **Late function-scope imports shadow module names** for the WHOLE function
  (UnboundLocalError at earlier lines). Audit moves with AST, not grep.
- **Lease names are per-role**, not per-API-shape (the offloader-vs-"alt" lesson).
- **Storage matrix for state read inside torch.compile-traced model code**
  (piecewise cuda graph compiles the whole model forward): contextvars are
  untraceable (hard error); dict-slot values are guarded per value — for a
  per-forward int that is one recompile per distinct size, straight into the
  recompile limit; **class/instance attributes are the only compile-friendly
  form** (attribute-source ints get automatic-dynamic after the first size
  change). Bools (≤2 values) are tolerable in any form — see
  `ForwardFlags._GRAPH_VISIBLE`. Config-bag leaves are real instance attributes for
  exactly this reason, and `ParallelContext.__getattr__` must stay free of
  `object.__getattribute__` (dynamo graph-breaks on it). Before moving such state,
  prove its readers sit outside compile coverage; a piecewise-prefill boot of a small
  model is the fast check (recompile storms show as `torch._dynamo hit
  config.recompile_limit` during the compile pass).
- **Engine-booting e2e tests are the only coverage for launcher-path code**; a child crash
  kills the process tree and pytest dies silently — run with `PYTHONUNBUFFERED=1` and read
  child logs.
- CI arms `SGLANG_ENABLE_ASYNC_ASSERT=1` (device-side `torch._assert_async` probes, e.g.
  KV-cache OOB): a fired device assert kills the tree with no Python traceback, and the
  same bug is *silent corruption* locally with the flag off. Arm it when reproducing CI
  crashes.
- CI startup logs print the full `server_args=ServerArgs(...)`; diffing that dump between
  runs is the fastest config-divergence check.

## Where to read the code

Key source files: `python/sglang/srt/runtime_context.py` (the container, every tier,
`publish`, `_ConfigBag`, `preserve_config`, `override_server_args`),
`python/sglang/srt/arg_groups/overrides.py` (override registry, passes,
`declare_load_time_override`), `python/sglang/srt/server_args.py` (`NS` metadata,
`Arg(..., resolvable=True)`, `__setattr__` strict guard), and the guardrail tests under
`test/registered/unit/` (`test_server_args_mutation_ratchet.py`,
`test_server_args_writer_ratchet.py`, `test_legacy_global_ratchet.py`,
`test_module_state_ratchet.py`, `test_server_args_namespaces.py`,
`test_runtime_context.py` — the last one doubles
as executable documentation of every tier's semantics).
