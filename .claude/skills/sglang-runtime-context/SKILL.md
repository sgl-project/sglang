---
name: sglang-runtime-context
description: How SGLang's runtime configuration and process-global state are organized (RuntimeContext tiers, publish + namespace config bags, the pristine ServerArgs seed, override entry points, resource/stream/buffer leases, per-forward flags), the CI guardrails that enforce the design, and the idioms for developing and testing against it. Load this before touching server_args, model overrides, module-level state, or per-forward state in sglang.
---

# SGLang runtime-context architecture

One container owns process-static runtime state: `sglang.srt.runtime_context.RuntimeContext`
(a process singleton reached via `get_context()`). Everything below is a tier on it.

| Tier | Accessor | Holds | Lifecycle |
|------|----------|-------|-----------|
| raw config seed | `get_server_args()` | the published `ServerArgs` — the startup record, for debugging, dumps and provenance. **Business code does not read fields off it**: the read ratchet pins that at zero, and "Reading config: the seed is off limits" below says what to read instead, which forms the ratchet sees, and what is outside it by construction (a runtime-computed name; a whole-object hand-off) | published at process entry; re-publish is **last-publish-wins** (the tokenizer publish in the launcher process; sequential engine rebuild in one process, e.g. unit tests) and re-projects the bags; read-only |
| resolved config | `get_exec()` `get_memory()` `get_schedule()` `get_model()` `get_spec()` `get_serving()` `get_observability()` `get_disagg()` `get_lora()` `get_mm()` `get_device()` | namespace **config bags** — the single source of truth for resolved config; leaves are real attributes (dynamo-traceable). Each is a **module function of no arguments**, and a module binds the name once: `manager.get_disagg()`, `self.get_disagg = get_disagg`, or a same-named import next to the bag one (`from model_loader import get_model`) all import fine and fail only when that path runs. `ruff --select F811` catches the import collision; `RuntimeContext` has no bag-named member and no `__getattr__`, so the member-call shapes are an `AttributeError` at call time — give it a delegating `__getattr__` and they go silent instead | projected at `publish` from the declarations over `server_args`' raw fields; mutated only via `get_context().override` |
| runtime flags | `get_flags()` | state that is *not* a pure function of config: `capture` (cuda-graph lifecycle), `moe` (ACTIVE backends, swappable), `dp` (DP-attention runtime flags) | materialized at subsystem init; groups offer `override()` for tests |
| resources | `get_resources()`, `get_stream(name)`, `get_buffer(name, factory)` | process-level handles: graph pools, EPLB state, EP dispatcher state, named side streams, workspace buffers | lazy; cleared by `reset_context()` |
| per-forward | `get_forward()` | forward-scoped flags (multi-stream switch, MoE output buffer, attn-TP inputs, extend-in-batch) | contextvar-backed; `scoped(**kw)` restores on exit; new threads see defaults |
| parallel | `get_parallel()` | **dual, spelled**: bare names are the live topology (tp/pp/moe/attn sizes, ranks, groups — `@property`, read-through); `get_parallel().config.<leaf>` is the parallel config bag | live: after dist init; `config`: after publish |

`reset_context()` (unit-test teardown) drops the published config and installs fresh
flags/resources/forward tiers.

## Config: publish + namespace bags

**`ServerArgs` holds the raw input and nothing else. Resolution writes no field:
it declares, and the declarations are what the namespace bags are projected from.
Business code never reads the record for a decision: a field read there answers
with what the operator typed, not with what resolution decided.**

- Every publishing process entry calls `publish(server_args, role=...)`
  (`run_scheduler_process`, the Ray `SchedulerActor`, the DP controller, tokenizer,
  detokenizer, encoder, weight-cache daemon, the multi-tokenizer worker, the
  spawned encoder TP/DP workers, the benchmark work functions, ...); constructors
  do not publish — `ModelRunner`, `TokenizerManager` and `MMEncoder` call
  `assert_published` and fail loudly if an entry forgot. The roles are enumerated once,
  as the keys of `ROLE_NAMESPACE_SETS` — there is no `launcher` role, the launch
  path publishes as `tokenizer`. The remaining non-publisher is
  `run_multi_detokenizer_router_process`: it *is* handed a `ServerArgs`, and uses
  it only for `configure_logger(server_args)` today, so it has nothing to publish
  for — a bag read added under that entry needs a `publish` at the entry first.
  `publish` projects the config bags from the declarations over the record's raw
  fields; the accessors (`get_exec()` etc.) fail closed before it
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
- **Reading a leaf when the caller holds the field *name*** (a readback endpoint,
  a control-plane handler): `get_context().config_leaf(name)` — the read side of
  `override`. It resolves the flat name through the same `NS` map the write side
  uses and raises on a name that is not a config leaf. Code that knows its field
  when it is written reads the bag leaf directly; `config_leaf` is for
  name-driven code, not a way around the seed ratchet.
- **Post-startup control-plane changes** — a weight update, a HiCache mirror
  attach, a parser resolved from the chat template — go through
  `TokenizerManager.record_config_updates(source, **fields)`, a named wrapper
  over `get_context().override`. One process keeps one log: the request dumps
  ship `get_context().overrides_log()`, and `config_value(name)` /
  `resolved_config_dict(base)` answer from the bags. The exposure ratchet
  resolves the wrapper, so a field recorded through it joins the post-publish
  override surface exactly like a direct `override` and needs the same ordering
  judgment against any supplied-instance read of it
  (`test_supplied_instance_exposure_ratchet.py`).
- **`model_path` and `served_model_name` are answered off the manager.** Both are
  `NS` leaves and `override` accepts them, but the tokenizer-side weight reload
  records only `load_format` and writes the two path fields as `TokenizerManager`
  attributes (`_MANAGER_OWNED_FIELDS`); `config_value` and `resolved_config_dict`
  overlay them on top of the bags. Bags do not cross a process boundary (above),
  so recording those two in the tokenizer process would leave every other
  process's bag on the old path while the log claimed a process-wide change. The
  scheduler rewrites its own copy where the reload happens —
  `ModelRunner.update_model_fields` overrides `model_path` / `load_format` for
  the target runner.
- **Late launcher-stage resolution (pre-publish)**: a few rules cannot run inside
  `__post_init__` — LoRA normalization, and the auto-parser detection that needs a
  tokenizer/chat-template load. They are resolution, not mutation, and they
  **declare** via `arg_groups.overrides.declare_late_resolution(server_args,
  source, **fields)`, which refuses the published instance. The declaration lands
  in the stash on that very object, so every holder of it carries the decision —
  the HTTP server, the multi-tokenizer workers it is serialized for, the
  schedulers it forks — and each of them publishes bags projected from it. The
  fields stay the operator's input; `resolution_result(sa, field)` and the bags
  are what answer for the decision. Returning a variant here is a bug: the
  launcher rebinds its local and everyone else keeps the unresolved object.
- **A value another runner / worker owns is a constructor argument, not a config
  copy.** The draft worker's `context_length`, load format and attention backend
  travel as arguments to `TpModelWorker` / `ModelRunner` and live on the runner
  (`ModelRunner.draft_attention_backend`, `kv_cache_dtype_str`, …); the encoder
  DP worker's device is `MMEncoder(gpu_id=...)`. There is no `ServerArgs.derive`
  any more — a config object is never copied-and-edited; test doubles that need a
  modified copy use `sglang.test.test_utils.server_args_variant`.

**Why a bag override cannot stand in for late resolution or per-runner
construction.** The bags are projected at
publish *from the declarations over the instance's raw fields*, so anything the
runtime must read has to be declared before publish — an override afterwards puts instance and bags back out
of agreement, and whole-object readers (`ModelConfig.from_server_args`,
`build_load_config`, `MMEncoder`'s own `self.server_args.X`) never see it. And bags do
not cross a process boundary: a child publishes from the object it receives and
re-projects its own bags, so a parent-side override is lost. Values that feed
construction before any bag exists (group init reads `server_args.tp_size`) have no
bag to override at all.


### Reads that legitimately stay on a `ServerArgs` instance

- **Per-runner values** — there is no per-runner `ServerArgs` any more. The
  draft-worker config copy is gone: every worker (`TpModelWorker`, the draft
  workers in `speculative/`) is handed the *same* instance the process published,
  so a bag leaf is the decision and `self.server_args.X` is the operator's
  input — a
  post-publish `override` moves only the bag, which is exactly why a field that
  is process-wide config (`attention_backend`, `skip_tokenizer_init`,
  `kv_cache_dtype`) reads from the bags like any other, and why a residual
  instance read on this path is stale the moment someone overrides that leaf.
  What is genuinely per-runner travels two ways, neither of them a config
  instance: **constructor arguments** (`ModelRunner(draft_attention_backend=...)`,
  `MMEncoder(gpu_id=...)`) and **runner attributes holding the resolved value**
  (`model_runner.kv_cache_dtype_str`, `prefill_attention_backend_str`,
  `num_fused_shared_experts`, `linear_attn_backends`) — threaded to consumers as
  arguments, never backfilled onto a shared object. A per-runner choice also stays
  *out* of the bags: recording it there is how a second runner inherits the first
  one's answer, which is exactly the bug `linear_attn_backends` replaced. The one sanctioned bend in that rule is
  *scoped*: `ModelRunner._load_format_scope` exposes the draft's
  `--speculative-draft-load-format` through `get_model().override(load_format=...)`
  for exactly the duration of the draft build, because model construction
  reads that bag leaf — the override restores on exit, so nothing outlives
  the scope. When there is a runner in hand, read its
  stamp; that is a different rule from "read the instance".
- **Per-instance boundaries** — the tokenizer-manager family, everything under
  `entrypoints/`, and the tokenizer-process multimodal processors read the bags.
  The old justification for keeping them on `self.server_args` ("several
  `Engine`s can share one process, bags are last-publish-wins across them") is
  **retracted** — owner ruling (2026-08-15): a process holds at most one live
  config at a time (concurrent multi-Engine is unsupported; sequential rebuild
  stays legal, unit tests rely on it). Nothing in those files reads the instance
  any more -- the exposure ratchet's pin set is empty, so the next such read is a
  new entry that has to argue for itself. What
  genuinely stays per-instance is what differs per *worker* within one engine:
  `base_gpu_id` travels as a constructor argument (`MMEncoder(gpu_id=...)`;
  `BaseMultimodalProcessor._fast_image_processor_device` is the shape to copy).
- **Whole-object passes** (`f(server_args)` handing the instance along) keep the
  supplied-instance contract; don't rewrite the parameter reads unless the
  field is runtime-mutated (see the elastic-EP `ep_size` case in
  `eplb/expert_location.py`) — **or the field is one that resolution fills in
  and the callee runs in a process that has published.** That second case is a
  decision, not a style question: the record carries the user's raw input, so a
  resolution-filled field read off it inside a runner-owned constructor answers
  with the pre-resolution value instead of the effective one. The answer is not
  automatically a bag read: pick where the value should come from — usually the
  `get_*()` bag, sometimes a runner stamp or a constructor argument (the per-mode
  attention pair and the encode-server `gpu_id` above are both this). The per-instance
  boundaries above are **not** exempt from this unless-clause (the multi-Engine
  exemption is retracted); each one gets its own disposition.
  `test_supplied_instance_exposure_ratchet.py`
  pins that set (empty today) — three spellings of the read: `server_args.field`,
  literal-name `getattr(server_args, "field", default)`, and the parked form
  (`self.x = server_args` in a method that takes the parameter, read as
  `self.x.field` anywhere in the class) — and fails on a new one, so the
  disposition gets picked when the read is written. Two shapes stay parameter-form on purpose: a helper the
  *resolution pipeline* calls with a `resolved_view` (its parameter happens to be
  named `server_args`), and a factory whose contract is "build X from the record
  you are handed" (`create_kt_config_from_server_args`, `DllmConfig.from_server_args`).

### Four ways a config sweep breaks something no test runs

Each of these shipped in a review round and cost a real defect; each now has a
guard, named here so the next sweep checks the same four things by hand first.

1. **The other implementations of an interface.** Dropping a parameter means
   auditing implementers, not just callers: `CustomSpecAlgo` is the plugin
   base for speculative algorithms, and the dispatch calls it with the
   built-in's argument list. Nothing in the tree implements it, so only a
   plugin user hits the `TypeError`.
   Guard: `test_plugin_hook_signatures.py`.
2. **Publish order inside a process entry, not per file.** A file containing a
   `publish` says nothing about whether a given read runs before it. Spawned
   workers (`MMEncoder` for encoder DP/TP, the Ray scheduler actor) start with
   an empty context, so a bag read above the publish raises only there.
   Guard: `test_publish_precedes_bag_reads.py`.
3. **The role namespace a process publishes under.** `ROLE_NAMESPACE_SETS`
   narrows what each role may read; the DP controller is audited for `exec`
   alone. A helper that reaches for another namespace passes every default-mode
   test and aborts startup under `SGLANG_ROLE_NAMESPACES=enforce`. Prefer
   answering from the caller's own namespaces over widening the set.
4. **Sibling surfaces of a readback.** Changing what one entry point reports
   means enumerating the others: HTTP, gRPC and in-process `Engine` each have
   their own server-info and model-info, and each passes its own tests while
   its users lose the field.
   Guard: `test_effective_state_surfaces.py`.

A fifth, from the same rounds: the accessor **name itself**. Called as an
object member (`manager.get_disagg()`), or shadowed by a same-named import
(`from model_loader import get_model` next to the model bag, where the later
import silently wins and the loader call gets a zero-argument bag), it imports
fine and fails only when that path runs. The invariant is one line: the name
means the process-wide bag, takes no arguments, and is bound once per module.
`ruff --select F811` catches the import collision; the member-call shapes are an
`AttributeError` at call time only because `RuntimeContext` has no bag-named
member and no `__getattr__` -- a delegating `__getattr__` would make them
silent, and that is when this needs a guard again rather than a rule.

Write these guards over a **derived** set, never a hand-kept list: an entry
naming a function that no longer exists, or a field list missing the one field
nobody migrated, passes green forever. Both happened here -- a `_ENTRY_POINTS`
row for a method the Ray actor does not have, and an effective-field set
without `load_format` -- and both were invisible because the assertion had
slack (`>= len(...) - 1`) or compared key names instead of value sources.

### `get_parallel()`: live topology bare, configuration under `config`

**Bare is the live group, `config` is what was configured.** `get_parallel().tp_size`
and its size / rank / group siblings are `@property` read-through over the canonical
getters; `get_parallel().config.<leaf>` reads the published `parallel` bag
(`nccl_port`, `enable_dp_attention`, `dp_size`, `ep_size`, `dwdp_size`, ... and the
five sizes that also have a live property). A bare read of a config-only leaf raises
an `AttributeError` naming the `.config` spelling — the tier is never guessed from
whether a property happens to exist.

The two tiers are **not** two spellings of one number. Live diverges from configured
wherever elastic EP scales the world away from the launch shape, and wherever
`initialize_model_parallel` aliases `_MOE_DP` to `_ATTN_CP` (`attn_cp_size >
moe_dp_size`), which makes a live comparison of that pair degenerate. The five
live-shadowed sizes (`tp/pp/dcp/attn_cp/moe_dp_size`) are where the choice matters,
and every business read of `get_parallel().config.<one of them>` is registered with
its reason in `_CONFIGURED_SIZE_CALL_SITES` (`test_global_config_read_ratchet.py`).
DCP has a third shape: the live `get_parallel().attn_dcp_size` / `.dcp_enabled`
answer the *effective* topology (`1` / `False` with no group installed), never the
requested size — `.config.dcp_size` is the requested one.

A process-global seed field-read of one of these sizes
(`get_server_args().tp_size`, or an alias of it) is a read-ratchet failure. A
`server_args` the object was *handed* is a different thing and not a ratchet
matter — see "Reads that legitimately stay on a ServerArgs instance".
Fail-loud is narrower: before dist init, a live size/group read raises — except
the DCP pair, which degrades instead (`dcp_enabled` → `False`,
`attn_dcp_size` → `1` when no group is installed;
`test_attn_dcp_defaults_when_group_is_uninitialized` pins this). After init,
only the DCP group is optional (`_DCP` exists only when `dcp_size > 1`; attn-CP and
moe-DP always install, as size-1 aliases if unused). The `config` hop is
deliberately dynamo-traceable (a plain property over a slot, no
`object.__getattribute__`); gate helpers like `enable_moe_dense_fully_dp()` run inside
compiled model forwards (`test_parallel_config_leaves_trace_under_torch_compile` pins
this).

A third surface carries the same names: `ParallelState` (`self.ps` / `mr.ps`), the
frozen per-process snapshot built once in `Scheduler.__init__` from these configured
sizes plus this process's ranks, and handed down (draft runners included). Prefer it
where an object was handed one; it is not a global accessor.

### Reading config: the seed is off limits

`get_server_args().field` in business code is a ratchet failure. Read:

- **a resolved leaf** → its namespace bag (`get_exec().moe.moe_runner_backend`,
  `get_schedule().chunked_prefill_size`, …). Bag-backed reads — a leaf directly, or
  a bag-derived accessor below, including the `get_parallel().config` hop — are
  what see post-publish overrides. Only the
  instance-derived accessors (the ones with no leaf to read) answer from the
  startup record and therefore do not.
- **a leaf the caller names at runtime** (a readback reporting a list of fields)
  → `get_context().config_leaf(name)`; it resolves the name through `NS` and
  raises on a non-leaf. A call site that knows its field reads the bag leaf.
- **the live topology** → `get_parallel()` (bare names).
- **a value derived from published leaves** → an accessor in `runtime_context` that
  derives it *from the bags*: `mamba_extra_buffer_enabled()` /
  `mamba_extra_buffer_lazy_enabled()` read `get_memory()` and `get_exec()`, so
  they see post-publish overrides. Prefer this shape whenever the inputs are
  leaves; the same-named `ServerArgs` members are the pre-publish equivalents the
  resolution pipeline uses, and wrapping one of those instead would quietly cost
  you override visibility. `is_ep_joiner()` / `is_ep_scale_joiner()` are the same
  shape over `exec.moe.ep_join_mode`, `attention_backends()` derives the
  `(prefill, decode)` pair from the three `exec.kernel` leaves, and
  `max_speculative_num_draft_tokens()` / `cutedsl_moe_max_num_tokens()` derive
  theirs from `spec` / `schedule` / `exec.graph`.
- **a value only the instance can compute** → the named accessor in
  `runtime_context`, which is the one module allowed to read the slot:
  `mamba_cache_chunk_size()`, `uses_mla_backend()`, `process_model_config()`.
  These have no leaf to read — they combine several fields, the HF config, or a
  property with no bag of its own. A new derived member gets an accessor here
  rather than call sites reaching for the record, and only when the bag-derived
  shape above cannot express it.
- **what was *configured*, where the bare name is the live value**
  → `get_parallel().config.{tp,pp,moe_dp,attn_cp,dcp}_size`. It reads the parallel
  bag's own leaf, so it answers with the resolved configuration and follows a
  post-publish override. The DCP live pair (`get_parallel().attn_dcp_size` /
  `.dcp_enabled`) is a different question again: it answers the effective topology
  (`1` / `False` when no group is installed), never the requested size, and it does
  not *need* dist init to answer. Every (file, size) pair is registered
  with its reason in `test_global_config_read_ratchet.py`
  (`_CONFIGURED_SIZE_CALL_SITES`), and that test fails if the code and the list
  disagree — a new file, or a new size in a listed file, has to be added — so a new
  site needs both an answer the live property cannot give and an entry saying what
  it is.
- **this runner's resolved value** → the runner
  (`prefill_attention_backend_str`, `kv_cache_dtype_str`,
  `draft_attention_backend`, `num_fused_shared_experts` on the model).

`self.server_args.field` is still right for handed per-instance config (see
"Reads that legitimately stay on a ServerArgs instance" above for the full set —
per-instance boundaries and whole-object passes; there are no per-runner config
copies to read any more). The allow-list is `GrammarManager` and `MMEncoder`;
what sits beside it is residue, not a family — and not for one single reason:

- the tokenizer-manager family and `entrypoints/` **read the bags**; what is
  left of them in the exposure ratchet is a handful of individually-dispositioned
  pairs, not a family awaiting conversion. Read the ratchet for the current set
  rather than assuming a directory is off-limits;
- `GrammarManager` is a handed instance for its residual `self.server_args`
  reads, but backend selection is **not** on the instance any more:
  `create_grammar_backend` reads `get_exec().kernel.grammar_backend`, and
  `__init__` calls that factory whenever `skip_tokenizer_init` is false. In
  production the scheduler process has published; a test that constructs one
  without publishing has to keep patching the factory (or publish itself);
- `MMEncoder` publishes the very instance it is handed (`publish(server_args,
  role="encoder")`) and takes its per-worker device as a separate `gpu_id`
  argument. Its `self.server_args` reads are on this list as a construction-path
  convention, and the residual is real: they answer with the raw input, so a leaf
  resolution decided and a post-publish `override` both pass them by.

Their tests are not one story: a `GrammarManager` built standalone turns the
factory's bag read into "config namespace not published" unless the test patches
it or publishes, while `MMEncoder` publishes in its own `__init__` and so needs
no such arrangement.

**Test doubles publish, they do not inject.** A stand-in that carries
`server_args=SimpleNamespace(field=...)` stops working the moment production reads
the bag; seed the value with `override_server_args`, which publishes only once it is
entered or installed — the bare call just builds the override:

```python
override = get_context().override_server_args(field=...)
override.install()
self.addCleanup(override.restore)      # or: with get_context().override_server_args(...):
```

Five separate test files learned this the hard way during the sweep.

The rule is about a double standing in for **config**: a `SimpleNamespace` that
pretends to be `server_args`. Prefer the context override even where a
single-accessor stub would work — `override_server_args(...)` composed with the
scoped bag / `get_parallel()` overrides expresses the *cause* (the configuration)
rather than pinning one helper's answer, and it keeps working when a reader
migrates between the accessor and the leaf. The sweep converted the last two
accessor stubs to exactly that shape (`test_attention_patching.py` publishes the
non-lazy strategy; `test_kimi_k3_vision.py` publishes `tp_size` and forces the
live topology through `get_parallel().override`), so no test stubs an accessor
today. Stubbing one *named accessor* remains a last resort for a case that
isolates one branch of one helper where no published config can reach it —
if you do it, say so in the test.

### Mid-resolution reads (inside the pipeline only)

Resolution runs in `__post_init__` and **writes nothing onto the record**: a
handler declares (`self._declare` / `declare_resolution`), the declaration goes
into the stash, and the fields keep what the caller passed. So a mid-resolution
read of a field answers with the *raw input* — every reader in the pipeline goes
through a view instead:

- `resolving_view(server_args)` / `self._resolved()` — the live view (walks the
  stash per read). This is what handlers and hooks bind, conventionally as
  `cfg = resolving_view(self)` at the top of the handler.
- `resolved_view(server_args)` — snapshots the overlay when built, which is what
  a post-process pass wants: it reads the state at *its* slot.

`test_resolution_reads_the_declarations` pins direct field reads at zero over the
two scopes it can derive exactly (every `arg_groups` function taking a config,
every `ServerArgs` handler the dispatcher reaches). Readers the pipeline calls
from elsewhere (`ModelConfig`, the platform defaults, the spec-algo hook) have
moved to the view as well — a field read there is the same bug, just one the
derivation cannot enumerate.

One consequence worth knowing: because the fields are the raw input, resolving a
bare `dataclasses.replace` copy lands in the same place as the parent — the
pipeline reads only its own input. `replace_resolved` is the way to copy a
resolved record (it carries the declarations and the `model_config` memo, so the
copy does not re-resolve at all).

### Adding a model-specific config adjustment

Never assign `server_args` fields from model code. Declare instead
(`sglang/srt/arg_groups/overrides.py`):

- Constant per-arch values → `MODEL_OVERRIDES["MyArchForCausalLM"] = {...}`.
- Derived values → `@register_model_override("MyArchForCausalLM")` returning a dict; the
  callable receives *pristine* `server_args` + `hf_config` and must not write.
- Normalization that must see earlier declarations → a post-process pass invoked via
  `run_post_process_pass` at its slot (reads a view, returns a declaration dict).
- Values only knowable at load time are **per-runner state**, not declarations:
  there is no `declare_load_time_override` any more. A model-family decision that
  its checkpoint drives (shared-experts fusion) is a question the *loader* asks
  the model class — `shared_experts_fusion_disable_reason(hf_config,
  quant_config)`, a classmethod answering without an instance — at the single
  model-instantiation point, and
  `install_shared_experts_fusion_decision` writes the answer to the ACTIVE moe
  flag before that model's layers build and read it
  (`is_shared_experts_fusion_disabled`, config-intent fallback).
  `draft_model_build_scope` brackets every draft build and routes the draft's
  answer to the speculative leaf, so a draft's decision never overwrites the
  target's. A process-level load-time fact (the sm80 dtype fallback —
  device-driven, identical for every runner) records directly via
  `get_context().override`.

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
- `flags.moe` is materialized by `initialize_moe_config()` at scheduler init (it
  reads `exec.moe` / `spec` / `model`, and takes no record);
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
  `get_device().override(device="meta")`). The one carve-out is the deliberate
  single-accessor stub for isolating one predicate — the terms and the two
  sanctioned examples live under "Test doubles publish, they do not inject"
  above; anything wider than one named accessor is this rule.
- Mocked runners/managers still need the **per-runner instance attributes** the code
  under test reads (`kv_cache_dtype_str`, `server_args` for whole-object passes) — set
  them explicitly on the mock; `MagicMock(spec=...)` raises on attributes that only
  exist post-`__init__`, which is the fastest way to find a missed stub.
- `reset_context()` in teardown when a test publishes outside a scoped override.
- `ServerArgs(model_path="dummy")` early-returns the pipeline (few declarations, no
  strict guard) — fine for lightweight fixtures.
- **Asserting what resolution decided reads `resolution_result(sa, "field")`**, not
  `sa.field`: the field is the raw input. Assert the field only when the point of
  the case *is* that the record stayed pristine (the FA4 page-size and waterfill
  cases do exactly that, and say so).
- **Run changed test files per-file** (own process), the way CI does: a monolithic local
  pytest run lets a context published by an earlier file mask a missing-publish bug in a
  later one.

## Guardrails (these fail CI; what to do when they fire)

1. **Strict mutation guard** (always on): bare `server_args.x = ...` after resolution
   raises unconditionally in `ServerArgs.__setattr__` — this *is* the guarantee that
   no writer can desync the bags, so there is no writer ratchet any more. Change
   resolved config with `get_context().override`; hand a per-runner value to its
   runner as a constructor argument. Projected bags are sealed the same way (leaf
   assignment raises).
2. **Mutation ratchet** (`test_server_args_mutation_ratchet.py`, exact pin 0 over the whole
   package minus the pipeline / multimodal_gen): textual scan for assignment forms. Never
   raise the baseline.
3. **No-copy contract** (`test_server_args_no_instance_mutation_entry.py`): neither
   `ServerArgs.override` nor `ServerArgs.derive` exists, and nothing in the package
   calls either form. Rerouting a writer to the bags means flipping **all its readers
   in the same commit** (no transitional dual-write).
4. **Legacy-accessor ratchet** (`test_legacy_global_ratchet.py`): `get_global_server_args`
   call sites must not grow. The replacement for a *decision* is a bag leaf, a named
   accessor, or the owning runner's stamp — not `get_server_args().field`, which the
   read ratchet below pins at zero. `runtime_context.get_server_args()` is only for the
   whole-object shapes (dumps, provenance, a hand-off to a callee that takes a config).
5. **Global config read ratchet** (`test_global_config_read_ratchet.py`): baselines are
   **0** for both the direct `get_server_args().field` and the alias form (function-local
   — including local copies of an alias, `cfg = sa` — module-level, or parked on an
   instance attribute, plus the `getattr(..., "field")` spelling of each; a name
   computed at runtime or indirection deeper than a local name copy is census-tool
   territory, per the test's docstring). The scanner matches `get_server_args` by its
   literal name, and the same file *bans* `import ... as` renames of it so that
   matching stays sound. Exempt by owner
   module only (`runtime_context.py`, `server_args.py`, `arg_groups/`). The same file
   carries `_CONFIGURED_SIZE_CALL_SITES`, the (file, size) map of every
   `get_parallel().config.<live-shadowed size>` reader with the reason the live property
   cannot serve it — a new file or a new size in a listed file must be added there. Its
   subject set is *derived* (property names ∩ `parallel` NS leaves), and it resolves
   every spelling of the call itself — an aliased import, a module-qualified receiver
   (including the whole dotted path an unaliased `import` binds), a local bound to either
   hop — so neither a rename nor a new shadowed size escapes it.
   `TestParallelConfigReadSpellings` in that file runs each spelling, because a spelling
   the scanner cannot resolve drops the read instead of failing anything.
6. **Module-state ratchet** (`test_module_state_ratchet.py`): `global` statements in the
   flag-owning layers are pinned by name. A new module-level runtime global belongs on a
   flags group / resources slot instead; migrating a pinned survivor must shrink the pin.
7. **Namespace coverage** (`test_server_args_namespaces.py`,
   `test_runtime_context_config_bags.py`): every `ServerArgs` field carries `NS(...)`
   metadata and the projected bags must cover the fields exactly (two-way).

Never module-skip a test "until the migration settles" — seed the context instead
(the deferral ratchet that once pinned this is retired; the rule stands).

## Hard-won pitfalls (check these before/while refactoring)

- **Moving code drops first-line guards**: early returns (`if self.is_draft_worker: return`)
  are the easiest thing to lose when relocating a method body. A draft is built from
  the target's published config — there is no draft config copy and no nested publish
  any more — so a body moved out of a draft-aware call site keeps reading the target's
  bags, and only that guard tells the two apart. What the draft build *does* scope is
  narrower and named: `draft_model_build_scope()` for the MoE fusion gates,
  `speculative_moe_backend_context()` for the runner backends.
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
  exactly this reason, and the parallel config tier is read through the plain
  `ParallelContext.config` property for the same reason (`__getattr__` is
  error-only, and `object.__getattribute__` graph-breaks). Before moving such state,
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
`publish`, `_ConfigBag`, `override_server_args`),
`python/sglang/srt/arg_groups/overrides.py` (override registry, passes,
`declare_late_resolution`), `python/sglang/srt/server_args.py` (`NS` metadata,
`Arg(..., resolvable=True)`, `__setattr__` strict guard), and the guardrail tests under
`test/registered/unit/` (`test_server_args_mutation_ratchet.py`,
`test_global_config_read_ratchet.py`, `test_legacy_global_ratchet.py`,
`test_module_state_ratchet.py`, `test_server_args_namespaces.py`,
`test_runtime_context.py` — the last one doubles
as executable documentation of every tier's semantics).
