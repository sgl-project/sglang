# Code Cleanup Patterns

Patterns that refactor and cleanup PRs in this repo have applied, and the guardrails
their reverts taught. Each entry is a general rule, a way to spot it, the cases where
it does not apply, and example PRs; some add the proof method the PRs used. The PR
numbers are evidence, not definitions; replace them as newer examples appear.

Sections follow the questions an audit asks: should this code exist, is there one
implementation, is the configuration surface minimal, are the gates right, is it in the
right place, is it the right shape. Section G is about landing the change.

---

## A. Should this code exist?

**A1. Code nothing reaches is deleted together with its whole vertical slice, after a proof that covers lookups by string.**
Each symbol was re-checked with a whole-repo grep that includes string targets, then every
layer went in one change. For an AOT op: the kernel source, the build-list entry, the
`m.def` / `m.impl` in each `common_extension*.cc`, the header declarations (CUDA and ROCm
sections), the wrapper, the `sgl_kernel/__init__.py` import and its `__all__` entry, tests
and benchmarks (#9956, #44690). For a JIT kernel: the `.cuh`, the ops module and its
`KernelSpec` (#48366). A closure-aware AST scan found bindings assigned and never loaded
(#33888); a field that was always `""` went with the IPC field that carried it and the
branch that tested it (#47249).
Spot: a top-level def whose name appears only at its definition; a local assigned and never
loaded (`ruff check --select F841` lists candidates; the repo hook selects only F401, F821,
F823 and UP037); a field written once and never read, or constant for its whole life; a
predicate that is true (or always falsy, e.g. a selector branch missing its `return`) for
every registered value; an `m.def` whose wrapper is gone; a module whose only importer
reads one constant; a `KernelSpec` target that does not resolve; a write with no reader.
Not this pattern: names reached by string (registries, `EntryClass`, `pkgutil` walks,
`__init_subclass__`, argparse `dest`, YAML / JSON configs, Rust bindings); a call kept for
its side effect (#33888 kept the config-published probe); a test oracle shipped in
production code, which moves to `python/sglang/test/`; code an open PR is about to wire
(#48366 left the gfx950 kernels that #46693 wired two days later); a predicate true for
every in-tree value but not for plugin registrations (#27977 kept plugins with
`supports_overlap=False` running, with a warning); `from sgl_kernel import X` under an XPU
branch, which `sgl-kernel-xpu` serves (#32114, #48789). A `mock.patch` target or a test that
sets the dead field by hand to reach its branch goes with it (#33888, #47249).
Traps: a deletion updates importers on rarely run paths too (#23019 deleted a module that
`fused_moe_native.py` still imported, breaking the `torch.compile` MoE path until #24069);
after deleting an AOT op, rebuild from a clean tree, or a stale binding object fails at
import with `undefined symbol` (#9956).
Proof: #47249 compared the simplified loop against the old one over every input in a small
exhaustive domain; #33888 asserted each line's content before deleting it by line number,
then re-ran the scan until it reported zero.
Examples: #44690, #48366, #47249, #33888, #27977, #9956, #24069.

**A2. A feature, backend or flag nobody uses goes through a deprecation issue with a feedback window, then is deleted end to end in one PR.**
The issues list each feature with what it is and why it can go, give three weeks for
feedback, and ping the original author (#28543). They post a deletion list that names every
layer (kernel, build list, binding, header, wrapper, exports, backend, registry, hooks,
docs, tests) and the release it leaves in (#32111, #32112). The removal PR says what each
removed name does now (argparse rejects it) and who loses a fallback (#32114).
Spot: a backend gated to one SM version or already disabled on newer hardware; a subclass
that overrides one method and inherits the rest; coverage only under `test/manual/`; no
functional change in about a year; a maintained replacement that already auto-upgrades
users; a flag read only by its own validation hook.
Not this pattern: a user reports production use (#25664 became a reimplementation plan;
#6006 drew T4 and 2080 Ti users); a candidate a live path still calls (#28543's Marlin FP8
stayed because `compressed_tensors_w8a16_fp8.py` calls it); "nothing selects it" that has
not checked `@register_post_process` hooks keyed on the HF config (#32112 missed one); a
name another platform still serves (#32114 re-registered `gptq` for CPU). Removing without
an issue loses the signal: #7351 dropped `/v1/files` and `/v1/batches` inside a refactor and
users asked how to get them back; #43898 removed `--enable-pdmux` while a user PR extending
it (#30885) was open. Restoring a removed backend needs a maintainer sponsor and an owner
(#22992 after #23009).
Examples: #28543 then #31109, #32111 and #32112 then #32114, #25664, #43898.

**A3. A feature or flag whose every effect is an error or nothing is removed without a deprecation window; prove it for each accepted value, and re-grep after merging main.**
All five `--torchao-config` values raised `ImportError` after a pin jump that came with an
unrelated CI upgrade (#23119); a lazy import behind the flag kept the error out of startup,
there was no CI coverage, and no issue was filed for four months. #34295 showed a loop over
every value; #34304 removed the flag, module, docs, tests and dependency.
`--speculative-skip-dp-mlp-sync` did nothing except crash DP attention outside PD, and a
safe env var already covered that case (#45043).
Spot: imports of renamed upstream symbols behind a lazy import; tests only under
`test/manual/`; a documented option whose only effect is an error; a flag that skips a
safety step another switch already skips safely.
Not this pattern: "rarely used" is not "broken". A new reader can land while the PR is open:
#32017 added a read of `server_args.torchao_config` the day before #34304 merged, and #34869
fixed it. Tests that fake `server_args` with `MagicMock` never see a read of a deleted
field; build one real `ServerArgs` (#34869 added such a test).
Examples: #34295, #34304, #34869, #45043.

**A4. A vendored copy or version workaround goes as soon as upstream ships the code; a dependency that only an optional path uses becomes optional or goes.**
Diff the copy against the release before deleting it: #37206 was byte-identical to
FlashInfer 0.6.18 apart from imports; in #38124 the split-K copy had drifted, so the proof
was the same selected tactic and output on every allowlisted shape. A FlashInfer bump
removed a 5.6K-line patch applied at image build, a 528-line kernel copy, and the download,
env var and error text around them (#33997). A vendored FlashMLA build gave way to its
wheel (#48789). A hard dependency used only by a non-default loader went with the loader
(#48853).
Spot: `TODO(<project>#N): remove once ... in the <project> pin`; file names carrying an
upstream PR number; "port of" or "copied from" headers; `FetchContent_Declare` whose
wrappers nothing calls; patch files and URL+sha256 downloads in Dockerfiles and
`scripts/ci/`; version-gated branches; a `pyproject.toml` pin with a single importer; `try:
import vllm`.
Not this pattern: an unvalidated replacement (#11322 broke GPT-OSS's YaRN RoPE on ROCm and
was reverted by #12028); `from sgl_kernel import ...` under an XPU branch (#48789 left it);
vendored trees that sweeps leave alone and moves keep byte-identical (#47782, #30795); a
copy carried until an upstream PR lands, kept byte-for-byte with the TODO above (#44016).
Examples: #37206, #38124, #33997, #48789, #48853, #44016, #11322 / #12028.

**A5. Every documented flag, env var and recipe setting is read under that name by the code that runs it; otherwise the code or the doc changes.**
Doc rows presenting unread env vars as working knobs were removed (#44690, #48399). A
documented field with no `add_argument` got one, a doc naming a nonexistent flag was
rewritten, and a flag whose dest was not the field was fixed (#48366). Documented `SGLANG_*`
names that the code read as `SGL_*` or unprefixed, including C++ `std::getenv`, were
switched to `envs` (#44690, #48399): `_handle_deprecated_envs` copies `SGL_X` to `SGLANG_X`
and never the reverse, so a raw read of the legacy name ignores the documented one, and
native code sees neither.
Spot: names under `docs/` or the cookbook with no reader; `os.environ.get`,
`get_bool_env_var` or `std::getenv` of an `SGL_*` or unprefixed form of a documented name;
`envs.X.set(` with no `.get(`; an `add_argument` whose derived dest is not a field; doc rows
for deleted vars; cookbook flags equal to the default (verify before dropping).
Not this pattern: a recipe pinned to a preview image runs that image's code, not main's;
read the image's source before editing its cells (#48399 removed b12x selectors that the
DGX Spark preview image still reads; #49171 restored them). A flag equal to a default can still be needed: #28731
kept `--mem-fraction-static` (dropping it OOMed) and `--page-size 128`.
Proof: check each documented name against the real parser and `envs`; a static guard test
keeps a doc list in sync with the code (#36987).
Examples: #44690, #48366, #48399, #28731, #36987.

**A6. Removing a feature removes everything only it reached, including the model-side half and the hooks keyed on its config.**
#43898 removed PD multiplexing with the split prefill that only it created:
`ForwardMode.SPLIT_PREFILL`, `forward_split_prefill` in 18 model files, the per-stream graph
keys and the AOT `spatial` extension. #32114 removed the Dual Chunk Attention backend and
its auto-selection hook but left `DualChunkRotaryEmbedding`, its config check and the
`get_rope` plumbing in five models, so such a checkpoint built a RoPE only the deleted
backend could read and failed at the first forward (#49171 removed the rest).
Spot: enum values, forward modes, model methods and `get_rope` arguments named after the
feature; `@register_post_process` hooks that read an `hf_config` field; an AOT extension or
dependency with a single consumer.
Not this pattern: config fields or helpers another feature still reads; check every
consumer of the field first.
Examples: #43898, #32114 / #49171.

---

## B. Is there one implementation?

**B1. Two implementations are merged only after their outputs are shown identical; "same math" is not evidence.**
A dedup that claims no behavior change is checked bit for bit, base vs branch: `torch.equal`
on dumped outputs over every case (#48366: 88 tensors on H200, 112 on GB300), including fully
masked rows (#44690: 720 cases on two architectures); identical generated AMDGCN when the
hardware is missing (#48366). Re-run the check after each commit of a series, and force the
merged path end to end when the default config skips it (#23019: diff 0.0 over 11 shapes,
with a size that covers the deleted chunk loop). A fork that is not bit-exact stays, with a
one-line comment saying why. A consolidation onto a different kernel states its bar
instead: #29771 was within 1e-7 of a reference in fp32, MMLU identical, GSM8K within 2 of
1,319, with a bit-identical token hash and a counter proving the new branch ran.
Spot: near-identical norm, RoPE, quant or router helpers in per-model files; a unified entry
point that drops a parameter some caller passed (`positions`, `interleaved`) or changes an
intermediate dtype; a rewrite that moves a load above `PDLWaitPrimary` or across a stream
join.
Not this pattern: #18164 put diffusion RoPE behind one dispatcher that dropped `positions`
and `interleaved` and changed cos/sin to fp32, with perf numbers only; images changed and
#18929 reverted it the same day. Random inputs never tie: #29771's group score dropped tied
maxima until #48946. Tests that build every input before launch cannot see ordering: #47784
loaded a row before the PDL wait, lost about 5 GSM8K points in a nightly-only B200 test, was
reverted (#47994), and re-landed with the load after the wait and a 20-run repro (#48174).
Bit-exact is relative to the live reference: a library version that picks another kernel
moves it (#34412).
Examples: #44690, #48366, #23019, #29771, #18164 / #18929 / #33555, #47784 / #47994 / #48174, #48946.

**B2. Diverged v1/v2 copies converge by building the unified path, switching the default, then deleting the legacy copy once nothing reaches it.**
Radix caches (roadmap #20415): the factory routed hybrid models to the unified tree
(#30468); the two legacy classes and the env vars only they read went once no registered
backend constructed them (#40313, -5.2K); HiRadixCache went once a named release had
shipped without callers (#40787). Speculative V1 workers were deleted after the parity PRs
were listed (#25464); the last move carried an AST byte-equality check and a GSM8K
comparison of v1, v2 overlap and v2 sync (#27607); `supports_spec_v2()`, true for every
algorithm by then, went with its false branches (#27977). Context-parallel v1 went as a
stack: tests moved to v2 first, then compatibility args, runtime, renames, docs (#36222 to
#36230).
Spot: `*_v1` / `*_v2` pairs; `is_spec_v2`; `SGLANG_ENABLE_*_V2`; sibling classes every
base-class change must touch; a capability predicate true for every registered value; env
vars and bench configs only the legacy class reads.
Not this pattern: deleting the old path before parity; dropping platform callers (#36228
restored the MUSA files byte for byte and asserted HIP on the legacy fallback it kept);
losing a guard the old path supplied (#36228's stacked GPU runs found a dropped `spec_info
is None` check); a migration gate that is still default-off (`SGLANG_ENABLE_WEIGHT_LOADER_V2`,
RFC #24703), whose env var is not redundant.
Examples: #20415, #30468, #40313, #40787, #25464, #27607, #27977, #36222 / #36228.

**B3. Per-model copies of one operator become one parameterized kernel, with parity against every retired variant and per-shape benchmarks; the default flips only where it is faster.**
Issue #26771 tabled six gate/top-k kernels; its discussion set the target of one Triton and
one CUDA kernel. #29771 made the Triton router the default where it beat AOT and kept the
grouped path opt-in where it only matched. #47641 added a CUDA router with a written
coverage contract, dispatched to it whenever covered, and deleted four kernels. #48946
deleted two more after a 288-case comparison including ties and all-underflow rows.
Spot: kernels with the same contract that differ only in a shape or limit (hard-coded
`num_experts = 384`); a specialized branch reached only for inputs the general kernel's
guard rejects (fp16 logits, a misaligned bias); a JIT variant that does not compile on a
platform whose dispatch still calls it.
Not this pattern: a variant that is the only path on some architecture, dtype or page size
(XPU kept the AOT `topk_sigmoid`); retiring a variant before the survivor matches its tie,
NaN and underflow rules (#48946 had to fix the Triton group score first).
Examples: #26771, #29771, #47641, #48946.

**B4. Protocol logic shared by several backends lives in one place; each backend supplies only data movement.**
The PD transfer backends had drifted: one never started the heartbeat checker, another never
enforced the bootstrap timeout or told decode about a failure. Under tracker #34510,
behavior was aligned first in small backend-local PRs, each with a backend-parameterized
test (#34692, #35281, #36612); then the lifecycle moved into common sender and receiver
classes with every behavior change listed (#41140). The wire was pinned first, with
truth-table tests built from the frames senders write (#34977) and frame-by-frame assertions
of every backend's messages (#41140).
Spot: the same method in `mooncake/`, `nixl/` and `mori/` with diverging guards; `poll()` on
an unknown room raising in one backend and waiting in another; a `defaultdict` tracker a late
message re-creates after `clear()`.
Not this pattern: ad-hoc dedup outside the tracker's steps; wire-frame changes without a
version frame; copying one backend's broad per-message `except Exception` into the others
(#35281 dropped it in review); moving metrics that nightly baselines read.
Examples: #33861, #34510, #34692, #34977, #35281, #36612, #41140.

**B5. One operator abstraction has one dispatch implementation; the old class becomes an alias, and platform fallbacks are explicit.**
#33205 folded `MultiPlatformOp` into `BaseFusedOp`: one out-of-tree registry, the old class
kept as a deprecated attribute-compatible alias, dispatch cost measured (+38 ns per call),
and every referencing file migrated (16 on main, 0 left). A new kernel provenance is a
`KernelBackend` value with a `forward_<x>` hook in that dispatcher, not a second registry
(#35114, #45389).
Spot: two registries for the same op; class-name string matching (`"FusedMoE" in
cls.__name__` missed a class); `hasattr` availability checks on a base that stubs every
backend; `if current_platform.is_X(): rebind` at import time.
Not this pattern: implicit fall-through between platforms (#33205 removed MUSA to CUDA,
where `is_cuda()`-gated imports raise `NameError`, and kept HIP to CUDA); a seam whose
default branch changes what an unlisted platform gets (#35114 reported XPU as "cpu" and
lost five Triton kernels until #36654). List every platform and compare the selected
implementation before and after.
Proof: mocked-platform tests over the whole dispatch ladder and an old-vs-new differential
over every class (#33205).
Examples: #33205, #35114 / #36654, #45389.

**B6. Merging two paths makes their feature combinations reachable: test the cross product, and turn every invariant that held only because code was absent into a guard.**
When one copy had a fast path and the other a hook, consumer or platform the fast path never
met, the merged code runs them together. #23019 merged the LoRA runner into the fused-MoE
kernel sequence, so LoRA hooks saw the TMA path's sorted, padded buffers; #31608 added the
guard. #8515 moved the EP expert-id remap into the shared `FusedMoE.forward`, and a backend
that remaps ids itself went to 0 accuracy (#8630). #35114's shared seam sent XPU to the CPU
fallbacks (#36654).
Spot: a hook, callback or layout contract on one side and a layout-changing fast path (TMA,
sorted or padded buffers, in-place output) on the other; a transform hoisted into a shared
base that one consumer already applies; a lookup whose default branch names a real platform.
Not this pattern: parity runs that exercise each old configuration alone; they stay green
while the new combination is wrong.
Examples: #23019 / #31608, #8515 / #8630, #35114 / #36654.

---

## C. Is the configuration surface minimal?

**C1. An env var whose right value is a function of the GPU is deleted and the read site decides from the hardware; so is a kill switch of a default-on, architecture-gated path, and any var nothing reads.**
#48399 deleted 41 (9 selectors, 28 kill switches, 3 unread, 1 diffusion gate) and dispatches
on `sglang.kernels.arch.current_arch()`: `SGLANG_DEEPGEMM_ON_H20` became `arch.is_sm(9, 0) and
arch.num_sms == 78`, and a third-party var every gfx1250 recipe set is now set at startup
with `os.environ.setdefault`. #34492 replaced `SGLANG_USE_SGL_XPU` with `is_xpu()` plus a
`--moe-runner-backend` choice; #48366 dropped a kill switch for a bit-identical tile; #35060
deleted descriptors with no reader outside `environ.py`.
Spot: `envs.X.get() and <architecture check>`; a var the cookbook sets on exactly the
hardware its path needs; a var named after a SKU (`_ON_H20`); a third-party var every recipe
for one architecture sets; a kill switch whose comment says "speed only" or "bitwise
identical"; a var that is `.set()` and never read.
Not this pattern: a var a doc tells users to set, or a CI test uses as its A/B control; a
var a vendor library reads (#35060 kept one); a migration gate (B2); a model hook's default
for one model on one architecture, which is a model decision (#48399 restored a DeepSeek-V4
SM120 default in the hook after a read-site gate moved other models off a working kernel);
a default chosen on purpose (#48399 kept `SGLANG_AITER_CAPTURE_COPY_IN` off, as #43077
decided).
Examples: #48399, #34492, #48366, #35060.

**C2. A default-off fast path is turned on only on the architecture it was validated on: the cookbook already sets it there or it is bit-exact by construction, accuracy is checked end to end where a checkpoint exists, and a log or trace line proves the new default ran.**
#48399 turned on five candidates and left the rest opt-in: three gfx950 paths had GSM8K base
vs branch plus the new kernel in a trace, one SM90 path is bit-exact by construction, and
one GEMM had only a kernel check against fp32 and is listed as not run end to end. The flip
is per architecture: one path became the gfx950 default and stays opt-in on CUDA.
Spot: an env var set in every cookbook cell for one platform; `if is_cuda() and not
envs.X.is_set()`; a backend default changed in `server_args`.
Not this pattern: approximate paths; a candidate with incomplete evidence. A flip exposes
every helper the path pulls in, on every platform: csgmv as the LoRA default (#11488) broke
AMD through an always-on Triton kernel cache and was reverted (#11735); the re-land put that
helper behind a default-off var (#12176); enabling it on CUDA (#13600) broke B200 CI, was
reverted (#13707) and re-landed after the B200 test passed (#14277).
Examples: #48399, #11488 / #11735 / #12176, #13600 / #13707 / #14277.

**C3. Decide instead of adding an env var or flag, put a backend toggle into the existing `--*-backend` choice, and never let one setting overwrite a value the user set.**
A reviewer asked why `SGLANG_USE_SGL_XPU` should exist beside `is_xpu()` and
`--moe-runner-backend`; the PR deleted it and added a choice (#34492). Mega MoE became
`--moe-a2a-backend megamoe` with the env var kept as an alias (#24884); the revert and
re-land (#25317 / #25406) differ by one missing `is_megamoe()` in the disjunction that sizes
shared-expert TP. #48399 replaced unconditional `envs.X.set(v)` in DeepSeek-V4 hooks with
`if not envs.X.is_set(): envs.X.set(v)`.
Spot: a new env var beside an existing `--*-backend` choice; env reads inside argument
resolution that assign other fields; `envs.X.set()` in a hook without an `is_set()` guard;
after adding a choice value, every `is_<sibling>() or ...` disjunction it must join.
Not this pattern: the old env var kept as a logged alias of the new choice (C5).
Examples: #34492, #24884 / #25317 / #25406, #48399.

**C4. A CLI flag or choice value is retired through a deprecation shape that maps meaning, not spelling: the old spelling resolves to exactly the configuration it used to produce.**
`arg_groups/argparse_actions.py` documents five shapes: `DeprecatedAction`
(`error_message=` stops the launch), `DeprecatedStoreTrueAction`,
`DeprecatedStoreConstAction` and `DeprecatedAliasStoreAction` (`new_flag=`), and
`DeprecatedChoiceToFlagAction` (`flags={value: (dest, new_flag)}`). Choice values are aliased
in `handle_deprecated_args` and stay in `choices`: `tokenspeed_mla` becomes `cutedsl_mla` and
also forces the FP8 KV, prefill backend and verify mode the old backend implied (#44803).
`--dp-size N --enable-dp-attention` becomes `attn_dp_size=N, dp_size=1`, and passing both
spellings raises (#41818). #5005 mapped a flag to a choice value that was already the
default, so every MLA model switched backend; reverted (#5048), re-landed with an MLA-aware
default (#5052).
Spot: entries deleted from `choices`; an `add_argument` removed with no deprecation action;
a boolean folded into a choice whose default already equals the new value; an alias that
misses one of the fields accepting the value; YAML `--config` files using the old name.
Not this pattern: blind name-swap scripts (#38375 lists the traps: `store_const` aliases
carry a value, prose rewritten into nonsense, CRLF churn, the separate `multimodal_gen`
`ServerArgs`; it checked the result against the real parser). A deleted flag that prefixes
live flags is not "unrecognized": argparse prefix matching silently binds it to the one
match, or reports "ambiguous". Cutting a deprecated flag's choices to the values you know
(#45389 rejected one a published recipe and a pinned plugin used; #46625 restored it).
Examples: #41818, #44803, #5005 / #5048 / #5052, #38375, #45389 / #46625.

**C5. A deprecated user-facing name (flag, env var, endpoint, importable module) keeps working with a visible warning naming the replacement for at least two releases; it is removed in a batch once no in-repo caller uses it, and where out-of-tree code may still call it, it becomes a tombstone that raises.**
#40795 removed deprecations that had shipped two releases and had no in-repo users.
`/get_model_info` and `/get_server_info`, deprecated in #13273, are still served because the
frontend client, the Rust server and tests call them (#13607 moved the callers and was
reverted by #14533). #25821's NSA aliases went in three batches (#38375, #40795, #47990).
`get_global_server_args` stays importable and raises naming both replacements (#38375).
Spot: `*WithAlias(`, `deprecated_name=`, `_DEPRECATED_ENVS`, `Deprecated*Action`,
`FutureWarning` shims, dated `TODO: remove ... after <date>`. The warning must be visible
(`logger.warning`, `UserWarning`, `FutureWarning`): `DeprecationWarning` from library code is
filtered by default. An env var replaced by a flag keeps a `_DEPRECATED_ENVS` note naming
the flag (#35060).
Not this pattern: internal names need no alias; #25821's internal re-export shims only
became cleanup work for #40795 and #47990. A path a pinned out-of-tree plugin imports is
user-facing (#46625). External protocol surfaces are a separate decision. The repo has no
written window; #40795 (two releases and no in-repo users) is the precedent this catalog
follows.
Examples: #40795, #13273, #38375, #47990, #29108, #13607 / #14533.

**C6. A deprecated spelling is translated once, at one seam before any reader, and a test pins that it resolves to exactly the new configuration through every entry point.**
`handle_deprecated_dp_attention` runs before any handler reads the DP layout, and a test
checks the old spelling resolves like `--attn-dp-size`, the single warning and the rejected
combinations (#41818). The `tokenspeed_mla` alias resolves in one place, so downstream
overrides lost their branch for it (#44803). #35060 folded four env deprecation mechanisms
into `_DEPRECATED_ENVS` with `transform=`, and found an old mapping that copied a `DISABLE`
value into an `ENABLE` name without inverting it. #45389 found a `--config` YAML rejected the
deprecated alias and, through the shared dest, the new key too.
Spot: `or server_args.<old_name>` beside the new check; the old field read outside
resolution; `replacement=` without `transform=` where polarity or units differ; a deprecation
with no test of its CLI, YAML and env forms.
Not this pattern: a name whose meaning cannot be translated; stop with
`DeprecatedAction(error_message=...)` instead of guessing.
Examples: #41818, #44803, #35060, #45389, #38375.

---

## D. Are the gates right?

**D1. A kernel is gated on the architecture family it was built for: an allowlist (`is_sm(10)`, a `KernelSpec` `min_sm` / `max_sm` range, an exact `gcn_arch`), never `major >= N`, a negation, or a probe that is also true on another vendor.**
`major >= 10` and `is_blackwell` (majors 10-12) admitted SM120/SM121 into tcgen05/TMEM paths:
CuTe DSL GDN/KDA prefill became `major == 10` (#45112), the DeepSeek-V4.1 indexer `is_sm(10)`
and MSA-v1 `is_sm100` (#48399). `not DEEPGEMM_BLACKWELL` meant "Hopper" and let SM120 into
SM90 DeepGEMM kernels (#48399: `is_sm(9)`). `== 120` missed SM121 (#18750).
`torch.cuda.is_available()` is true on ROCm, where gfx950 reports (9, 5): NVIDIA checks
misfired (#28486) and PDL ran on gfx942 (#48366). A `"gfx95"` substring test admitted any
gfx95x (#48399: `gcn_arch == "gfx950"`).
Spot: `get_device_capability()[0] >=`; `is_sm_at_least(10)`, `is_blackwell` or `not
is_blackwell()` before a one-family kernel; `== 120`; `major >= N and major != M` (a denylist
still admits every future major, #35361); `torch.cuda.is_available() and
get_device_capability()`; gfx substring tests; capability read at import time; one helper
guarding two kernels with different needs (#48399 split one).
Not this pattern: a path later architectures support, verified against the library (#45112
kept FlashInfer KDA at `>= 10`); a `not is_blackwell()` site whose `is_sm(9)` rewrite would
change HIP or MUSA (#48399 left `single_batch_overlap.py`); `is_cuda()`, which is also true
on CUDA-compatible out-of-tree platforms (RFC #42173); SM count, a runtime query.
Proof: write the gate as a pure function of the capability tuple and unit-test it on CPU
across families (#35361); patch `current_arch` in tests (#48399).
Examples: #45112, #48399, #48366, #35361, #18750, #28486, #47782.

**D2. A gate decides whether a kernel applies; a contract violation raises.**
A `can_use_*` / `covered()` term that restates a caller contract adds no safety: the gate
runs first and turns the launcher's error into a silent fallback. #44690 removed four such
terms by the criteria in #42184. Wrappers that accepted parameters but compiled fixed values
now raise `ValueError` (#48366). Kernels that returned `None` for "unsupported" now raise
behind a public predicate (#35114).
Spot: gates comparing shapes, devices or ranks across operands; parameters accepted but
never read, or baked into a compiled kernel; `return None` meaning unsupported.
Not this pattern (keep the check in the gate): the launcher sees a flattened view and cannot
enforce it; the eager fallback computes the case correctly (a dtype mismatch promotes); the
call site swallows exceptions (`BitExactFusionGate`, failure caches), so raising would turn
one bad call into a permanent fallback. Shape or dtype support questions still return False.
Examples: #42184 (closed; the criteria), #44690, #48366, #35114.

**D3. A diffusion fast path runs only at the quality tier its numerics allow: `exact` is bit-identical to the reference path; `lossless` (the default) keeps the math and every operand and accumulator precision, and only moves rounding; `high` may change what is computed.**
#42370 defined the tiers and demoted three fusions that lowered a precision. Paths running at
`exact` without being bit-exact were removed or gated: a VAE QK-norm+RoPE (#44690), a VAE
residual+RMSNorm whose env var became the tier gate (#48399), and a fused QKV GEMM that takes
split-K at small M (#48366).
Spot: a default-on fusion with no tier check; `quality in ("lossless", "high")` instead of
`quality_allows(q, tier)`; an env var toggling a fusion; a concatenated-weight GEMM at
`exact`; hand-written verify-once blocks (use `BitExactFusionGate`).
Not this pattern: sites tuned for launch overhead stay hand-written (the gate costs about
160-180 ns of host time per call); no measurement admits a path that lowers precision or
accumulates below fp32.
Proof: `multimodal_gen/test/quality_tier_admission.py` scores the candidate and the current
path against fp64; a site test asserts the fused op is not called and `torch.equal` holds at
`exact` (#48399).
Examples: #42370, #44690, #48366, #48399.

**D4. Hardware differences leave shared code through a declared capability or a backend hook, not through new `is_xxx()` branches.**
`PlatformCapabilities` (#35594) replaced platform-name lists: `(_is_npu or _is_xpu or
_is_mps)` became `not capabilities.hicache_device_kernels`. #47173 deleted four NPU
graph-runner classes and moved their device steps into backend hooks (roadmap #46487).
Vendor whole-step kernels live out of tree behind one flag per vendor, with the contract in
`kernels/plugin_kernels/<hw>/` and the binding in `srt/kernel_plugins/<hw>/` (#43949, #45389).
Spot: `is_npu` / `_is_hip` / `is_out_of_tree()` branches in shared modules outside
`hardware_backend/`; a list of platform names standing for one property; a device subclass
copying a shared runner; per-model env vars toggling a vendor kernel.
Not this pattern: removing a branch before its hook exists; a platform class whose methods
equal the generic path is deleted, not moved, and whatever it inlines into shared code is
still a branch to move (#45950); `is_cuda_alike` while #42173 is open; plugin-facing hooks
retire with a warning (#35594).
Examples: #35594, #47173, #46487, #45389, #43949.

---

## E. Is it in the right place?

**E1. Moving an internal module updates every importer in the same PR and leaves no forwarding shim; names are imported from the module that defines them.**
#47990 moved 322 imports, checked each resolves to the same object, and deleted the
re-export shims; #47992 added `check-indirect-imports`, which fails when a module neither
defines nor exports a name it re-imports and names the real source. #45112 left 94 old
paths with zero references. A staged series may keep a shim only until its last step
deletes it (#32072).
Spot: a module that only re-imports; a forwarding `__getattr__`; `f as f` or `__all__` added
only to silence the hook; a rewrite that misses an import form (`from pkg import X` vs the
dotted form, function-local imports, a sibling import inside the moved module).
Not this pattern: names bound per platform inside `if` / `try`; a package facade
(`sglang.srt.utils` star-imports topic modules that declare `__all__`, #48781); documented
plugin-facing names; user-facing entry points (C5). The hook cannot see a module-level
`__getattr__`, `import a; a.f`, or `mock.patch` strings; grep for those.
Examples: #47990, #47992, #45112, #41243, #32072. Kernel placement: kernel-organization.

**E2. `srt.utils` is the bottom layer: no module-level imports of domain packages, the runtime never imports `sglang.utils`, and a helper with one importing package lives in that package.**
#48755 landed `check-srt-utils-imports` with a shrink-only list of 13 violators; #48758
moved them out (the offloader went to `layers`, because `model_loader` would form a cycle);
#48781 split `common.py` from 5,208 to 824 lines with byte-identical moves; #48827 kept the
runtime off `sglang.utils` (`import sglang` 1.00 to 0.73 s, `-X importtime`, median of 11).
Spot: a utils helper with one importer; a utils module importing `managers`, `layers` or
`multimodal` at module level; an annotation-only import outside `TYPE_CHECKING`; `logger`
imported from another module.
Not this pattern: imports inside a function or under `TYPE_CHECKING`; check for cycles before
choosing the new home.
Examples: #48755, #48758, #48781, #48827.

**E3. Model- or feature-specific branches move out of generic files into owner modules that declare, not mutate.**
The 601-line model-specific function in `server_args.py` (#20481) became `arg_groups/` hooks
(#24328, #24326, #25424; each move AST-equal), then `model_overrides/<family>.py` providers
registered with `@register_model_override(arch)` that return `{field: value}` without
mutating (#30067, #37087). Shared-expert configuration became owned by each model (#44913).
Spot: `if arch in (...)` blocks in server args, the scheduler or `ModelRunner`; a per-model
value threaded through constructors or saved and restored in process flags.
Not this pattern: moving config-time overrides into `models/` (the frontend must not import
GPU kernels at config time, #37087); setting an attribute on a wrapper the runner never sees
(F5).
Examples: #20481, #24328, #24326, #25424, #30067, #37087, #44913.

**E4. A god class is split into collaborators, not mixins, as prepare, mechanical move, follow-up.**
Scheduler mixins (#8483) became collaborators and the mixins were retired (#25610, #25624):
a prep commit makes the methods static over the collaborator, the move is a cut and paste
with byte-identical bodies, and call sites change by prefix. #26271 extracted 13
`init_<name>()` methods and proved them byte-equal with an AST script. #49144 gathered four
pause fields read across six files into `PauseState`.
Spot: `class \w+Mixin` in a large class's bases; methods annotated `self: "<Other>"`; private
fields of one concern read from several files.
Not this pattern: an extraction whose responsibilities the reviewer cannot state (#3115,
reverted by #3829); folding a thin subclass into its base, which is F2 (#48119).
Examples: #8483, #25610, #25624, #26271, #49144, #3115 / #3829. See large-class-style and
mechanical-refactor-verify.

**E5. A class's home follows its base class, and a new target directory lands with a ratchet guard, or the old file keeps growing.**
RFC #25371 mapped base classes to directories. Where the directory existed, the old file
shrank; where it did not, `memory_pool.py` grew from 2,257 to 5,042 lines in three months
(6,490 now), and #29678 added an allocator at the top level two months after `allocator/`
existed. The proposed ratchet never landed. The utils series did it in order: guard with a
shrink-only list (#48755), move (#48758), drop the list.
Spot: subclasses outside the directory their base maps to; new top-level modules beside a
package for their role; a README describing a directory that does not exist.
Not this pattern: drive-by moves into a directory that a tracked series owns.
Examples: #25371, #29678, #48755 / #48758.

**E6. Config is read-only after resolution, per-instance differences are constructor arguments, and readers switch to a new source of truth only after every writer writes it.**
`ServerArgs.override` (#33490) and `derive` (#33887) were retired; per-runner values became
constructor arguments. `ServerArgs` holds only the raw input and resolution declares instead
of writing (#36255); `get_context().override(...)` is the one mutation entry, and
`ServerArgs.__setattr__` raises after resolution. #32100 reverted five merged PRs that moved
readers to config bags before load-time overrides wrote them (a fused shared-expert fallback
desync, LoRA gates leaking between two `Engine`s, the draft load format ignored).
Spot: `deepcopy(server_args)`; `get_context().override` outside resolution; a value stored in
two places with late writers; a load-time override whose readers read a bag published
earlier.
Examples: #33490, #33887, #36255, #31813-#31817 / #32100.

---

## F. Is it the right shape?

**F1. No abstraction without a second user; prefer flat layouts and plain functions.**
An indirection whose only implementation passed values through became one function
(#31254). A wrapper whose fallback branch no subclass could reach was deleted (#21651). A
single-field wrapper class was inlined (#26085). Helpers kept "for future use" were deleted
in review, and so was a module duplicating `sampling_params` validation (#7167).
Bookkeeping for frees that torch mempools never perform was replaced by a comment stating
that invariant (#21392). Hook layouts stay flat (#24326).
Spot: an ABC or virtual method with one implementation or zero overrides; an
`@abstractmethod` base that also carries a fallback path; a single-field class around a
tensor; a helper that forwards its arguments unchanged; a public method whose only caller is
in a future PR (#3115); a lock or map for a case an invariant excludes.
Not this pattern: an extension point that out-of-tree plugins implement (#45389); a seam an
open tracked series is about to fill (G9).
Examples: #31254, #21651, #26085, #7167, #21392, #24326, #3115 / #3829.

**F2. Inherit only when the reuse is substantial; vary behavior through constructor arguments, and delete overrides that restate the parent.**
A subclass that only swapped kernel selections became a constructor argument; the proof was
a call-sequence comparison, a mutant the tests catch, and bit-identical greedy output
(#41442). Overrides identical to their parent were deleted (#42963). A thin worker subclass
was folded into its base (#48119). A backend that overrides every attention entry point and
allocates parent state it never uses should derive from the base class (#28808, open).
Spot: a subclass overriding every `init_*` and `forward_*`; a parent `__init__` building
objects the child replaces or never reads; a subclass chosen per layer only to change one
selection; an override whose body equals the parent's.
Not this pattern: a subclass that overrides one method is reuse; remove it only on A2
grounds. A wrapper forwarding through `__getattr__` is not a substitute for inheritance (F5).
Examples: #41442, #42963, #48119, #28808.

**F3. Rename only to fix meaning, in one sweep including tests and docs; never for taste in hot files.**
`InputMetadata` became `ForwardBatch` once it carried the inputs (#1543). NSA became DSA
(#25821). `--dp-size` meant replicas or attention-DP groups depending on another flag; it was
split into two arguments and each reader moved to the meaning it needs (#41818). A helper
named for SM90 that also ran SM120 was renamed in review (#32405).
Spot: a name whose meaning depends on another flag; a name that states one architecture,
dtype or model but serves several; a type name that describes half of what it holds.
Not this pattern: typo and capitalization sweeps over hot files (#6209, reverted by #6244);
`TokenizerManager` to `StdOrchestrator` (#3116 / #3828, never re-landed); wire-format keys,
which keep the old name (#41818); names pinned outside the repo, which change only after the
consumers import the new path (#45389); call sites about to migrate anyway, which get a
legacy docstring and a ratchet test instead of a sweep (#30065).
Examples: #1543, #25821, #41818, #32405, #30065, #6209 / #6244, #3116 / #3828.

**F4. Migrations to the typed conventions (direct field access, `isinstance` narrowing, `msgspec.Struct`) go one subsystem at a time, each step shown to leave the field set and behavior unchanged.**
Defensive `getattr` on fields every constructor sets became direct access (#25435, #31804);
`hasattr(obj, "batch")` became an `isinstance` check (#29549); one `getattr` named an
attribute that never existed, so a gate never fired (#38059). IPC structs moved to msgspec
behind an opt-in transport (#28688). The config tier moved in three green steps with the
field set counted before and after (#38753).
Spot: `getattr(x, "f", default)` where no class assigns `f` (a dead gate, not noise);
`hasattr` used as a type test; `__dataclass_fields__`, `object.__setattr__` or
`object.__new__` on a class being converted.
Not this pattern: duck-typed external objects (HF configs keep `getattr`); one class inside a
dataclass hierarchy (#32405); flipping a wire-format default without an audit path; drive-by
sweeps (`.claude/rules/no-dataclasses.md`).
Traps: `dataclasses.field(default_factory=)` must become `msgspec.field`; Struct fields reject
`object.__setattr__`; a `dict=True` namespace needs `__reduce__` to pickle; `SimpleNamespace`
test fakes need the new field (#38753, #31804).
Examples: #25435, #29549, #31804, #38059, #28688, #38753.

**F5. A wrapper model is an `nn.Module` subclass of what it wraps, never a plain object that forwards through `__getattr__`.**
Loaders keep what `model.eval()` returns; a forwarding wrapper returns the inner module, so
attributes and methods set on the wrapper vanish. Moving shared-expert configuration onto
models (#44913) crashed MiniCPM-V on ROCm and Mistral-Small-3.1 in nightly that way;
#45656 made the wrapper subclass the model it wraps.
Spot: `def __getattr__` on a model class delegating to `self.<inner>`; a
`*ForConditionalGeneration` that is not an `nn.Module`; per-class `eval()` overrides.
Examples: #45656, #44913.

---

## G. Landing it

**G1. A refactor PR carries no behavior change. A change it exposes lands separately, or is named in the body with a before/after and its own test.**
#28683 split `init_backends` and also charged draft weights to `--mem-fraction-static`; two
extra-stage jobs failed at startup and it was reverted. The redo named the change, made the
guard print the minimum viable fraction, and raised six test configs (#28855). #7163's
"simplifications" changed the draft KV layout; the redo fixed an indexing bug and documented
the layout for backends that build page tables (#7213).
Spot: "pure refactor" next to "fixes a latent bug" (#26134); a "split" or "simplify" title on
a diff that touches memory budgets, defaults or tensor layouts, or adds a startup `raise`; a
transform script that does not reproduce the whole diff.
Proof: compare greedy tokens and logprobs bitwise, base vs branch; in #32405 identical code
moved 0.03 on a 200-question GSM8K, so only the bitwise check resolves a difference.
Examples: #28683 / #28841 / #28855, #7163 / #7210 / #7213, #26134 / #26166 / #26665, #32405.

**G2. Wrong-looking logic is not deleted until you can say what it protected.**
Read the comment on the block and the commit that added it, and state in the PR why its case
cannot occur. #14167 removed KV headroom its own comment justified (#14259 bisected the MTP
memory growth to it). #14044 merged two graph pools and broke Qwen3-Next accuracy (#14278).
#20227 decoupled the health signal from batch output, which hides requests stuck in the
waiting queue (#20259). #12291 tried to drop `auto` MoE picks, one of which is the only
correct kernel for its case.
Spot: deletions in KV or memory sizing, graph pools, health or watchdog code, or `auto`
backend selection, with no statement of why it is safe; a body with no motivation; `if g is
not None: g = v` on a global that starts as `None`.
Examples: #14167 / #14259, #14044 / #14278, #20227 / #20259, #12291.

**G3. Everything importable from `sglang`, every `python -m sglang.<module>` entry point, every HTTP route and every `Engine` or SDK method name is public API.**
Unused-import removal broke `from sglang import RuntimeEndpoint`; the public surface is
`__all__` (#378 / #385). Moving the benchmark entry points without a shim was reverted the
same day (#809 / #815); the accepted move kept warning shims for two releases (#28625,
#29108).
Spot: F401 fixes in `__init__.py` (the ruff hook excludes them); imports kept for side
effects (`# noqa: F401`); renames under `lang/`, `entrypoints/` or top-level `sglang/*.py`;
route changes; module paths quoted in the README, docs or benchmark scripts.
Not this pattern: internal module moves (E1).
Examples: #378 / #385, #809 / #815, #3969, #11685, #28625, #29108.

**G4. Code leaving a shared package migrates every caller, call signature, platform and op registration first, and deletes the old copy last, in a separate PR.**
The Marlin kernels moved to JIT with bitwise JIT-vs-AOT tests (#18067, #19181); the AOT
copies went later (#19241, #19379). Two single-PR moves were reverted: #16879 deleted a
kernel that ROCm had registered four days earlier (re-landed keeping the AOT copy, #17658);
#19437 left callers passing a dropped argument and JIT ops outside `register_custom_op`
(#20012 fixed both).
Spot: one PR that deletes under `sgl-kernel/csrc` or `common_extension*.cc` and adds the JIT
twin; `common_extension_rocm.cc`, `setup_rocm.py` or an AMD workflow step naming the kernel;
`try: import <jit module> except: fallback` (the JIT import never fails, compilation is lazy);
wrapper arity changes.
Not this pattern: the `sgl_kernel` wheel's Python API may forward to the new implementation
for a release (#19437).
Trap: deleting the AOT copy deletes the bitwise reference too; keep a torch reference in the
JIT test (#18067).
Examples: #18067 / #19241, #19181 / #19379, #16879 / #17257 / #17658, #19437 / #20005 / #20012.

**G5. An operator's registration is part of its contract: the schema with mutation annotations, `mutates_args`, and mutated inputs that do not alias. A cleanup that moves, rewraps or merges an op keeps them and runs the CUDA graph, piecewise graph and `torch.compile` paths.**
#4890 replaced `m.def` schema strings with bare bindings and broke CUDA graph and
`torch.compile` (#4944 restored them). #19437 called JIT ops without `register_custom_op` and
broke piecewise CUDA graph (#20012). #47670 found two `mutates_args` tensors that were views
of one allocation, so AOTAutograd cloned the whole KV allocation every layer.
Spot: `m.def("x", fn)` with no schema; a dropped `Tensor!`; a kernel call that bypasses its
custom-op wrapper; two mutated inputs carved from one buffer; a rising
`TORCH_LOGS=graph_breaks` count (#20699).
Examples: #4890 / #4944, #19437 / #20012, #47670, #20699.

**G6. Before shared code changes, other platforms' paths are run or signed off by their owners; what cannot be run stays byte-identical.**
#36228 removed generic CP v1 but left every `hardware_backend/` file and the AMD / NPU test
trees identical to base, restored two MUSA files byte for byte after the owner's "MUSA CP
logics cannot be changed", and asserted HIP in the legacy fallback it kept. #25678's author
had no NPU access; "passes all NPU tests" was not accepted until the owner ran GSM8K for both
MoE paths on A3. #42245's re-review found a migrated fallback that called a concrete kernel
instead of the platform dispatcher and dropped XPU's path, and added a routing test.
Spot: a "migrated" fallback calling a concrete kernel instead of the dispatcher
(`get_*_ops()`, `forward_npu`, `forward_hip`); a shared module importing a vendor module at
module level (#25663 broke `import sglang.srt.layers.moe` everywhere); a platform branch in
a shared base-class `__init__`, which every subclass runs (#11322's ROCm cache view crashed
YaRN models, whose cache is longer; reverted by #12028).
Proof: `git diff --stat <base> -- python/sglang/srt/hardware_backend
test/registered/{amd,npu,musa}` is empty; `python -c "import sglang.srt.layers.moe"` on a
CPU-only box; a routing test that makes the wrong-platform kernel raise.
Not this pattern: CPU policy tests are not a kernel run on that platform (#42245); argued
platform decisions (#33205 dropped MUSA-to-CUDA fall-through and kept HIP-to-CUDA).
Examples: #36228, #25678, #42245, #25663, #11322 / #12028.

**G7. Green PR CI is not coverage: run the nightly and extra tests that exercise the change, on the hardware of the lane each test is registered on.**
#47784 passed PR CI and dropped GSM8K from about 0.95 to 0.90 in a nightly-only test that
failed only on 8-gpu-b200; the re-land ran it with `/rerun-test` before merge (#48174).
#28683 broke extra-stage jobs at a low `--mem-fraction-static`; the redo ran
`/tag-and-rerun-ci extra` (#28855). #36228's per-lane `/rerun-group cp` runs caught a shape
mismatch on B200 and a dropped guard on H200 before merge. #48366 found a CUDA test
registered `nightly=True` that no job ran, and a case that passed on sm_103 but returned NaN
on the B200 lane.
Spot: registrations no workflow runs (`nightly=True` on CUDA, suite names no workflow
invokes); nightly-only model tests of the family a kernel serves; memory-budget or default
changes with base CI only; a merge with a lane cancelled (#40475, reverted by #40611).
Proof: `/rerun-test <path>`, `/rerun-group <group>`, `/tag-and-rerun-ci extra`.
Examples: #47784 / #47994 / #48174, #28683 / #28855, #36228, #48366.

**G8. Removing a fallback, default backend, dependency or image variant needs a fallback story, an A/B on the target hardware, and a replacement that installs and whose downstream pieces are merged.**
#20583 was closed: with one library only "there is no fallback option". #16000 was 1.07x
faster and still closed because the replacement is hard to install in the image. #48325
folded EFA into the main image and was reverted the same day (#48637): DeepEP's EFA kernels
were not merged, and the AWS libfabric took over process-wide. #30438 was accepted with a
migration plan, a hard error for the retired choice, and SM120 kept via JIT. #44803's
interleaved A/B (A B B A, two launches per arm) found a 21% regression at 64K with
speculative decoding and fixed it before merge.
Spot: `hasattr(backend, "...")` / `getattr(backend, "...", None)` in model code, which
silently loses a fast path when the backend goes; an `auto` selection that changes as a side
effect (#30438 moved SM120 from Triton to CUTLASS); `LD_LIBRARY_PATH` edits in the main
Dockerfile.
Examples: #20583, #16000, #48325 / #48637, #30438, #44803.

**G9. Before starting, check open and recently closed PRs, trackers and RFCs for duplicates and in-flight migrations, including the re-land of anything you are redoing; a tracker closed for inactivity is stalled, not finished.**
#23087 redid a reverted JIT move while its re-land #22094 was open and was closed as a
duplicate; #9607 overlapped the open #7431 and was closed when #10479 landed; #28808 drew
about fifteen competing PRs and none merged; #48366 left gfx950 kernels that #46693 wired
two days later.
Spot: `gh pr list --state all --search "<symbol>"` and `--search "<reverted PR number>"`;
unchecked tracker items; issues closed by `github-actions` "due to inactivity" (#25371,
#31051, #38004, #8715); idle PRs closed by the bot and later revived (#42069 revives #31264).
Examples: #23087 / #22094, #9607 / #7431 / #10479, #28808, #46693; trackers #24703, #30696,
#34510, #46487.

**G10. Core-module refactors are agreed with the module owners on their tracker first, then ship as independently revertable steps across releases; no line-count or deletion quota.**
#24335 rejects "pure vibe coding PRs & naive changes without fully discussing"; #20415 turned
away a newcomer because the work needs someone who has fixed bugs in the cache; #25371
requires `[mem_cache][N/N]` titles that say whether a step is a mechanical move, no shims, and
mechanical-refactor-verify; #20481's review called the server-args split "dangerous while not
so urgent" and asked for smaller steps over several releases without breaking downstream
repos; #37405 dropped its line and CI-hour estimates and states "No deletion quota".
Not this pattern: a stack whose steps cannot land without sequential rebases is collapsed
into one PR (#26665 replaced four stacked PRs; #37436 folded three).
Examples: #24335, #20415, #25371, #20481, #37405.

**G11. Revert fast; re-land as a redo that names the reverted PR, says why it was reverted, and shows evidence that exercises the cause; hold release cherry-picks of a fresh change until it survives main.**
#47994 reverted within a day with a bisect (including a control rerun of the last good
commit), a suspected cause and a re-land plan, keeping helpers other code already used;
#48174 confirmed the cause with a 20-run repro table and SASS and re-ran the nightly test
that caught it. #28855 opens "Redo of #28683 (reverted in #28841)" and fixes what the revert
exposed. #20012 added the missing custom-op registrations and reported GSM8K with piecewise
CUDA graph on. An automated cherry-pick opened 41 s after merge needed its own revert when
main reverted (#27756 / #27760).
Spot: `Revert "..."` PRs with an empty body; a byte-identical re-land with no stated reason
(#40612); a revert that must re-apply fixes built on top (#26166).
Not this pattern: a cause that needs a purpose-built kernel to reproduce need not become a
registered test; the repro plus a rerun of the catching test is the evidence (#48174).
Examples: #47994 / #48174, #28841 / #28855, #20005 / #20012, #26166 / #26665, #27756 / #27760.

**G12. The diff stays on topic: no stray hunks, wrong-base commits, backup files, review notes or personal scaffolding.**
#22670 also rewrote the docs homepage banner from a stale branch and was reverted (#25044).
#33224 branched from another topic branch, carried six unrelated commits, and missed both
`pyproject.toml` files, so the dependency it set out to remove stayed. #32405 drew "Remove
this NOTE. It's for code review". #40475 uploaded a golden test "only for myself" whose
asserts ran only under `__main__`. #48366 committed a `.orig` merge backup.
Spot: `git diff --stat origin/main...HEAD` lists files outside the scope; `*.orig` / `*.rej`
in `git ls-files`; a review bot commenting on an untouched file; comments addressed to
reviewers; assertions only under `if __name__ == "__main__":`; a dependency removal where
`grep -ri <name>` still hits `pyproject*.toml`.
Examples: #22670 / #25044, #33224, #32405, #40475, #48366.

**G13. Every validation result names the base and head SHA it ran on; a result from an earlier head, a pre-rebase branch or another base is not evidence for what merges.**
#36228 kept a paragraph of which SHA each GPU run tested and marked earlier runs historical
after a main merge; #33224 re-ran its A/B after finding its base carried unrelated commits;
reviewers report exact-head runs (#42245). Findings are triaged as PR-introduced or
pre-existing against the base SHA (#25663), and failing test ids are compared base vs branch
(#48366).
Spot: result tables with no SHA; "verified" before a force-push or main merge; an A/B whose
base is a topic branch.
Examples: #36228, #33224, #42245, #25663, #48366.

---

## In-flight directions (2026-11; check the tracker before relying on this)

A cleanup must not fight these. Leave their dual paths, gates and half-moved code alone
unless the change is part of the tracked series. Several trackers were closed by the
inactivity bot with work left (G9); treat them as stalled, with their rules still binding.

- Weight loader v2: `SGLANG_ENABLE_WEIGHT_LOADER_V2` (default off) and the dual paths in
  `llama.py` / `qwen2.py` are a migration gate. Only the first step landed (#28671); RFC
  #24703 is open again, the revival #42069 is idle, and #47170 also redesigns loading.
- RuntimeContext (#30696, open): new code reads `get_<namespace>()`; `ServerArgs.override` /
  `derive` must not come back (#33490, #33887, #36255). Per-role publishing is not done.
- mem_cache: `pool/` has not started (#35647 open; RFC #25371 bot-closed, its rules stand:
  `[mem_cache][N/N]` titles, no shims, mechanical-refactor-verify).
- Radix cache (#20415): Rust is the default TreeCore since #39627, with a central fallback to
  the Python core (keep it). Removing `RadixCache` and its subclasses is still pending.
- PD transfer (#34510): transport steps are open (#46460, #46614); wire changes wait for a
  version frame. Do not hand-dedup the `conn.py` files.
- Quantization layout (#38004, bot-closed after step 1, #37552): no drive-by splits of the
  large method files; a split restarts the series with the RFC author.
- MoE runners (#8715, bot-closed): backend code inside quantization `apply()` is meant to move
  into `moe_runner/<backend>.py`; several methods have not moved.
- NPU (#46487), platform capabilities and the `is_cuda_alike` audit (#42173),
  parallelism-independent model code (#47170), the Rust frontend (#23206, #45816, #40920).
- Monokernels (#43949): `--monokernel-backend` is a deprecated alias of the per-vendor flags
  (#45389); keep the alias.
