---
name: sglang-runtime-context
description: How SGLang's runtime configuration and process-global state are organized (RuntimeContext tiers, resolve-at-end ServerArgs, override entry point, resource/stream/buffer leases, per-forward flags), the CI guardrails that enforce the design, and the idioms for developing and testing against it. Load this before touching server_args, model overrides, module-level state, or per-forward state in sglang.
---

# SGLang runtime-context architecture

One container owns process-static runtime state: `sglang.srt.runtime_context.RuntimeContext`
(a process singleton reached via `get_context()`). Everything below is a tier on it.

| Tier | Accessor | Holds | Lifecycle |
|------|----------|-------|-----------|
| config | `get_server_args()` | the process-wide **resolved** `ServerArgs` | resolved once in `__post_init__`; audited mutations only |
| runtime flags | `get_flags()` | state that is *not* a pure function of config: `capture` (cuda-graph lifecycle), `moe` (ACTIVE backends, swappable), `dp` (DP-attention runtime flags) | materialized at subsystem init; groups offer `override()` for tests |
| resources | `get_resources()`, `get_stream(name)`, `get_buffer(name, factory)` | process-level handles: graph pools, EPLB state, EP dispatcher state, named side streams, workspace buffers | lazy; cleared by `reset_context()` |
| per-forward | `get_forward()` | forward-scoped flags (multi-stream switch, MoE output buffer, attn-TP inputs, extend-in-batch) | contextvar-backed; `scoped(**kw)` restores on exit; new threads see defaults |
| parallel | `get_parallel()` | read-through wrapper over topology getters (tp/pp/moe/attn sizes, ranks, groups) | stateless; `override()` for tests |

`reset_context()` (unit-test teardown) drops the published server_args and installs fresh
flags/resources/forward tiers.

## Config: the resolve-at-end contract

**After `ServerArgs.__post_init__` returns, the fields ARE the resolved configuration.**
Model overrides and normalization passes *declare* values during resolution (into a
provenance stash); `materialize_declarations()` applies them once at the very end of
`__post_init__` (gate order, last writer wins). Consequences:

- **Reading config**: read fields directly, in any process, at any time after construction.
  For global access use `runtime_context.get_server_args()` (the blessed accessor —
  `get_global_server_args()` is a legacy shim over the same slot and its call-site count is
  ratcheted; do not add new ones).
- **Mutating config after resolution**: the ONLY entry point is
  `ServerArgs.override(source, **fields)`. It records provenance (`_runtime_mutations`),
  keeps whitelisted resolvable fields consistent with the declaration stash (so a republish
  resolves the same values), and bypasses the strict guard. Bare `server_args.x = ...`
  after resolution **raises** under `SGLANG_STRICT_CONFIG_MUTATION=1`, which the test
  harness (`sglang.test.test_utils`) turns on by default.
- **Mid-resolution code** (inside `server_args.py` / `arg_groups/` only): fields are
  read-only during resolution; handlers and hooks read the in-flight state through
  `resolved_view(server_args)` / `self._resolved()`. This is pipeline-internal — never use
  `resolved_view` outside the pipeline. (One sanctioned exception: helpers that the pipeline
  itself invokes mid-resolution, e.g. `adaptive_spec_params`.)

### Adding a model-specific config adjustment

Never assign `server_args` fields from model code. Declare instead
(`sglang/srt/arg_groups/overrides.py`):

- Constant per-arch values → `MODEL_OVERRIDES["MyArchForCausalLM"] = {...}`.
- Derived values → `@register_model_override("MyArchForCausalLM")` returning a dict; the
  callable receives *pristine* `server_args` + `hf_config` and must not write.
- Normalization that must see earlier declarations → a post-process pass invoked via
  `run_post_process_pass` at its slot (reads a view, returns a declaration dict).
- Values only knowable at weight-load time → `declare_load_time_override(source, {...})`
  (validates the whitelist, routes through `override()`).

Declarable fields form a whitelist: `Arg(..., resolvable=True)` in the `ServerArgs`
dataclass. A declaration against a non-whitelisted field fails at its slot.

### Load-time vs resolution-time (critical)

`__post_init__` runs in the launcher process before any model/platform import. Logic that
consults an **extensible registry** (e.g. out-of-tree platforms registering attention
backends in `init_backend()`, which runs at `model_runner` import) must stay at load time
(ModelRunner init), writing through `override()`. Before moving any load-time logic into
resolution, verify everything it reads is already complete at construction time.

## Runtime flags (`get_flags()`)

For state that init-time code *derives* and runtime code reads — parsed enums, platform
probes, swappable ACTIVE values. Not for config mirrors (those died with resolve-at-end:
read the field).

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
  `get_context().override_server_args(**fields)` (config tier: publishes a fresh
  dummy-boundary `ServerArgs` carrying the overrides — `with`-scoped, or
  `install()`/`restore()` for fixture-lifetime use) + `get_parallel().override(...)` +
  `get_flags().<group>.override(...)` + `get_forward().scoped(...)` +
  `get_resources().override(...)`. All are scoped and transactional. Tests control
  execution through the context — do not hand-build and publish config objects.
  Note `override_server_args` is itself transitional (to be deprecated): it exists
  while production still branches on raw `server_args` fields at runtime; prefer the
  finer-grained tier overrides wherever they already cover the path you need.
- **Never monkeypatch import bindings** (`module.get_x = lambda: ...`): production code may
  read a different accessor over the same slot and your patch silently stops intercepting.
  Publish/inject for real: `get_context().override_server_args(...)` for config;
  `monkeypatch.setattr(get_resources(), "slot", fake)` for resources.
- Fixtures standing in for `ServerArgs` need an `override` method if the code under test
  mutates config (`_fake_server_args`-style: SimpleNamespace + write-through override).
  MagicMock swallows `override()` calls silently — prefer SimpleNamespace so misses raise.
- `reset_context()` in teardown; `_IsolatedServerArgs`-style save/restore when a test
  publishes.
- `ServerArgs(model_path="dummy")` early-returns `__post_init__` (no materialization, no
  strict guard) — fine for lightweight fixtures.

## Guardrails (these fail CI; what to do when they fire)

1. **Strict mutation guard** (`SGLANG_STRICT_CONFIG_MUTATION=1`, default-on in tests):
   bare `server_args.x = ...` after resolution raises. Fix: route through
   `override(source, ...)`, or move genuinely config-decidable logic into resolution.
2. **Mutation ratchet** (`test_server_args_mutation_ratchet.py`, exact pin 0 over the whole
   package minus the pipeline / multimodal_gen / the mock-fixture factory): textual scan
   for assignment forms. Never raise the baseline.
3. **Legacy-accessor ratchet** (`test_legacy_global_ratchet.py`): `get_global_server_args`
   call sites must not grow — new code uses `runtime_context.get_server_args()`.
4. **Module-state ratchet** (`test_module_state_ratchet.py`): `global` statements in the
   flag-owning layers are pinned by name. A new module-level runtime global belongs on a
   flags group / resources slot instead; migrating a pinned survivor must shrink the pin.

## Hard-won pitfalls (check these before/while refactoring)

- **Moving code drops first-line guards**: early returns (`if self.is_draft_worker: return`)
  are the easiest thing to lose when relocating a method body. Draft workers share the
  target's `server_args` object — a draft-side write poisons the target.
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
  `ForwardFlags._GRAPH_VISIBLE`. Before moving such state, prove its readers
  sit outside compile coverage; a piecewise-prefill boot of a small model is
  the fast check (recompile storms show as `torch._dynamo hit
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

Key source files: `python/sglang/srt/runtime_context.py` (the container and every tier),
`python/sglang/srt/arg_groups/overrides.py` (override registry, passes,
`declare_load_time_override`, `resolved_view`), `python/sglang/srt/server_args.py`
(`override`, `__setattr__`, `materialize_declarations` call,
`_handle_model_capability_adjustments`), and the guardrail tests under
`test/registered/unit/` (`test_server_args_mutation_ratchet.py`,
`test_legacy_global_ratchet.py`, `test_module_state_ratchet.py`,
`test_runtime_context.py` — the last one doubles as executable documentation of
every tier's semantics).
