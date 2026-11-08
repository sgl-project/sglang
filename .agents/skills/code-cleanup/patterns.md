# Code Cleanup Patterns

Patterns that refactor and cleanup PRs in this repo have applied, and the guardrails
their reverts taught. Each entry is a general rule, a way to spot it, the cases where
it does not apply, and example PRs. The PR numbers are evidence, not definitions;
replace them as newer examples appear.

Sections follow the questions an audit asks: should this code exist, is there one
implementation, is the configuration surface minimal, are the gates right, is it in the
right place, is it the right shape. Section G is about landing the change.

---

## A. Should this code exist?

**A1. Code nothing reaches is deleted together with its whole vertical slice, after a proof that covers lookups by string.**
The cleanup rounds re-checked every symbol with a whole-repo grep that includes string
targets (`module:callable` registry specs), then removed the source, the C++ op
declaration and binding, the Python wrapper, the `KernelSpec`, tests, benchmarks and docs
in one change. AST scans found bindings that are assigned and never loaded; a field that
was always `""` went together with the branch that tested it.
Spot: a top-level def whose name appears only at its definition; a field written once and
never read; a predicate that is true for every registered value; an `m.def` whose wrapper
is gone; registry metadata nothing reads; a write with no reader.
Not this pattern: names reached by string (registries, `EntryClass`, `pkgutil` walks,
`__init_subclass__`, argparse `dest`, `mock.patch` targets, YAML/JSON configs, Rust
bindings); a call kept for its side effect (#33888 kept a probe that checks whether config
has been published); production code that a test uses as its oracle, which moves to
`python/sglang/test/` instead of being deleted; code an open PR is about to wire (#48366
left gfx950 kernels that #46693 wired two days later); a symbol the AOT `.so` still links
(#9956 left an `undefined symbol`); a rarely run path that still imports the module
(#24069, the `torch.compile` MoE path).
Examples: #44690, #48366, #47249, #33888, #27977, #25432, #9956, #24069.

**A2. A feature, backend or flag nobody uses goes through a deprecation issue with a feedback window, then is deleted end to end in one PR.**
Deprecation issues list each feature with what it is and why it is removable, give a
feedback deadline (three weeks), and ping the original author; the removal PR then deletes
kernel, integration, flags, docs and tests together. The evidence in the PR body is
checkable: no CI enables it, no registry constructs the class, unused since a named release.
Spot: a backend gated to one SM version or already disabled on newer hardware; a subclass
that overrides one method and inherits the rest; coverage only under `test/manual/`; no
functional change in about a year; a maintained replacement already auto-upgrades users; a
flag read only by its own validation hook.
Not this pattern: a user reports production use. #25664 (Hopper SBO) became a
reimplementation plan instead of a deletion; #6006 (sm75 end of life) drew objections;
#7351 removed endpoints with no notice and users asked how to get them back.
Examples: #28543 then #31109, #32111 and #32112 then #32114, #43898, #40313, #40787, #23009.

**A3. A feature that fails for every accepted input can be removed without a deprecation window; smoke-test model loading after removing it.**
All five `--torchao-config` values raised `ImportError` after a pin bump, with no CI
coverage and no issue filed for four months; #34304 removed it, and #34869 fixed the
weight-loading break the removal caused the same day.
Spot: imports of renamed upstream symbols behind a lazy import; tests only under
`test/manual/`; a documented option whose only effect is an error.
Not this pattern: "rarely used" is not "broken". Prove it is broken with a repro that loops
over every accepted value.
Examples: #34295, #34304, #34869, #45043.

**A4. A vendored copy or version workaround goes as soon as upstream ships the code; a dependency that only an optional path uses becomes optional or goes.**
Diff the vendored copy against the upstream release, then import upstream instead (#37206
was byte-identical to a FlashInfer release); a FlashInfer bump deleted 6.5K lines of
workarounds (#33997); a vendored FlashMLA build was replaced by its wheel (#48789).
Spot: trees that mirror an upstream package; "port of" or "copied from" headers;
version-gated branches; a `pyproject.toml` pin with a single importer; `try: import vllm`.
Not this pattern: an unvalidated replacement (#11322 broke GPT-OSS on ROCm and was reverted
by #12028); a copy that has drifted from upstream, which needs a diff before deletion, not
after (#38124); vendored trees kept byte-identical to upstream so they stay diffable, which
moves preserve and sweeps leave alone (#30795, #47782).
Examples: #37206, #38124, #33997, #48789, #48853, #11322 and #12028.

**A5. Every documented flag, env var and recipe setting is read by code under that name; otherwise the code or the doc changes.**
Doc rows presenting unread env vars as working knobs were removed; a documented flag that
did not exist was implemented; documented `SGLANG_*` names that the code read as `SGL_*` or
unprefixed were switched to `envs`; a recipe pointing at a backend main does not have was
fixed.
Spot: names under `docs/` or the cookbook with no reader; `os.environ.get("SGL_...")` or an
unprefixed read of a documented name; doc rows for deleted vars; cookbook flags equal to
the default (verify before dropping).
Not this pattern: an explicit flag a recipe keeps on purpose (#28731 dropped redundant
flags but restored `--mem-fraction-static` after verification).
Examples: #44690, #48366, #48399, #28731, #36987.

---

## B. Is there one implementation?

**B1. Two implementations are merged only after their outputs are shown identical; "same math" is not evidence.**
Dedups were checked bit for bit (`torch.equal` over every output, identical AMDGCN) on every
architecture claimed; consolidations state their bar (#29771: 1,063 parity cases, fp32
epsilon across fp32 and bf16, GSM8K and MMLU unchanged). Forks that are not bit-exact stay,
each with a one-line comment saying why.
Spot: near-identical norm, RoPE or quant helpers in per-model files; branches keyed on
`is_neox_style`, `rotary_dim` or dtype being merged; intermediate math that changes dtype.
Not this pattern: #18164 unified diffusion RoPE with perf numbers only, changed images, and
was reverted the same day (#18929); the accepted #33555 limited the fast path and tested
NeoX and partial rotary. #47784's JIT rewrite dropped GSM8K from about 0.95 to 0.90 on B200
only, in a nightly-only test; #47994 reverted it and #48174 re-landed it with a 20-run
repro table. Bit-exact is relative to the live reference: a library update can move it
(#34412).
Examples: #44690, #48366, #29771, #18164 / #18929 / #33555, #47784 / #47994 / #48174, #8515.

**B2. Diverged v1/v2 copies converge by building the unified path, switching the default, then deleting the legacy copy once nothing reaches it.**
Radix caches: roadmap #20415, unified tree by default in #30468, legacy deleted in #40313
(-5.2K) and #40787. Speculative V1 workers were deleted only after the parity PRs were
listed (#25464); the last move carried an AST equivalence check and a GSM8K comparison
(#27607).
Spot: `*_v1` / `*_v2` pairs; `is_spec_v2`; `SGLANG_ENABLE_*_V2`; sibling classes that every
base-class change must touch.
Not this pattern: deleting the old path before parity; dropping platform callers on the way
(#36228 kept the HIP, NPU and MUSA paths); a migration gate that is still default-off
(weight loader v2 under RFC #24703), whose env var is not redundant.
Examples: #20415, #30468, #40313, #25464, #27607, #27977, #36228.

**B3. Per-model copies of one operator become one parameterized kernel, with parity against every retired variant and per-shape benchmarks; the default flips only where it is faster.**
Issue #26771 set the target of one Triton and one CUDA kernel; #29771 merged the gate/topk
kernels, benchmarked them on B200, H100 and H200, and kept the grouped path opt-in.
Spot: kernels with the same contract that differ only in a shape or limit; hard-coded
`num_experts = 384`.
Not this pattern: a specialized kernel whose inputs now reach a fallback with different
semantics (#48946 found tied maxima dropped); a variant that is the only path on some
architecture, dtype or page size.
Examples: #26771, #29771, #48946, #23019.

**B4. Protocol logic shared by several backends lives in one place; each backend supplies only data movement.**
The PD transfer backends had drifted (a heartbeat thread never started, a bootstrap timeout
never enforced, two frames swapped); behavior was aligned first, then the lifecycle was
consolidated, step by step under a tracker.
Spot: the same method in `mooncake/`, `nixl/` and `mori/` with diverging guards.
Not this pattern: ad-hoc dedup outside the tracker's steps; wire-frame changes without a
version frame.
Examples: #33861, #34510, #35281, #41140.

**B5. One operator abstraction has one dispatch implementation; the old class becomes an alias, and platform fallbacks are explicit.**
#33205 folded `MultiPlatformOp` into `BaseFusedOp`, measured the dispatch cost (+38 ns per
call) and migrated every referencing file; #35114 collapsed import-time platform chains
into one selection seam.
Spot: two registries for the same op; class-name string matching; `if
current_platform.is_X(): rebind` at import time.
Not this pattern: implicit fall-through between platforms (#33205 removed MUSA to CUDA and
kept HIP to CUDA deliberately).
Examples: #33205, #35114, #45389.

---

## C. Is the configuration surface minimal?

**C1. An env var whose right value is a function of the GPU is deleted and the code decides from the hardware; so is a kill switch of a default-on, architecture-gated path, and any var nothing reads.**
#48399 deleted 41 such vars and dispatches on `sglang.kernels.arch.current_arch()`; #35060
removed dead descriptors and folded four deprecation mechanisms into `_DEPRECATED_ENVS`.
Spot: `if envs.X.get() and <architecture check>`; a var the cookbook sets on exactly the
hardware its path needs; `DISABLE_*` on a path already gated by architecture; a descriptor
with no reader outside `environ.py`.
Not this pattern: a var a doc tells users to set, or a CI test uses as its A/B control; a
migration gate (B2); a model hook's default for one model on one architecture, which is a
model decision, not an architecture gate (#48399 kept DeepSeek-V4's SM120 mHC default in
the hook after a read-site gate moved another model off a working path); a default chosen
on purpose (#48399 kept `SGLANG_AITER_CAPTURE_COPY_IN` off, as #43077 decided).
Examples: #48399, #35060, #10479.

**C2. A default-off fast path is turned on for its architecture only with evidence: the cookbook already sets it there, CI covers it, and accuracy is checked end to end with a probe showing the new default ran.**
#48399 turned on five, each with GSM8K on the target hardware and a log or trace proving
the new path was taken.
Spot: an env var set in every cookbook cell for one platform.
Not this pattern: approximate paths; paths with no CI coverage on that hardware. Default
changes are reverted often (#488 / #492, #2217 / #2221, #11488 / #11735, #13707 / #14277).
Examples: #48399, #488 / #492, #11488 / #11735.

**C3. Decide instead of adding an env var or flag, and never let one setting silently rewrite another.**
A reviewer asked why a backend flag should exist at all, and selection moved into the
existing choice (#34492). An env var that silently set `ep_size = tp_size` was reverted and
re-landed as an explicit `--moe-a2a-backend megamoe` choice (#24884 / #25317 / #25406).
Spot: a new env var beside an existing `--*-backend` choice; env reads inside argument
resolution that assign other fields; hooks that overwrite values the user set (#48399 fixed
DeepSeek-V4 hooks that set nine vars unconditionally).
Examples: #34492, #24884 / #25317 / #25406, #48399.

**C4. A CLI flag or choice is retired through the argparse deprecation actions, keeping the defaults the old spelling implied.**
`arg_groups/argparse_actions.py` has five shapes (`DeprecatedAction`,
`DeprecatedStoreTrueAction`, `DeprecatedStoreConstAction`, `DeprecatedChoiceToFlagAction`,
`DeprecatedAliasStoreAction`), each with `new_flag=`; `handle_deprecated_args` translates
during resolution. The `tokenspeed_mla` alias maps to `cutedsl_mla` and also forces the FP8
KV cache the old backend implied (#44803).
Spot: entries deleted from `choices`; an `add_argument` removed with no deprecation action;
YAML `--config` files using the old name.
Not this pattern: blind name-swap scripts. #38375 lists the traps (old names that prefix a
live flag become "ambiguous", `store_const` aliases, CRLF churn, the separate
`multimodal_gen` `ServerArgs`) and checked its result against the real parser. Names that
external plugins already pin (#45389 reverted its own rename). A rename that costs more
than it fixes (#46885 kept `flashinfer`).
Examples: #41818, #44803, #38375, #45389, #5005 / #5048 / #5052, #23906.

**C5. A deprecated user-facing name (flag, env var, endpoint, importable module) lives at least two releases with a warning that names the replacement; it is then removed in a batch with its in-repo users, and the retired name raises instead of warning.**
`/get_*` endpoints were deprecated in #13273 and removed in #40795 ("past two releases");
the NSA to DSA aliases from #25821 were removed in #38375 and #47990; the benchmark shims
from #28625 and #29108 went in #47990. A retired API raises a hard error, because
`DeprecationWarning` is filtered by default (#38375).
Spot: `*WithAlias(`, `deprecated_name=`, `_DEPRECATED_ENVS`, `Deprecated*Action`,
`FutureWarning` shims, dated `TODO: remove ... after <date>`.
Not this pattern: internal names, which need no alias (#25821 renamed internals directly);
moving callers off a deprecated name before CI agreed (#13607, reverted by #14533);
external protocol surfaces such as a legacy gRPC service, which are a separate decision.
The repo has no written window; #40795 (two releases and no in-repo users) is the
precedent this catalog follows.
Examples: #13273, #40795, #25821, #38375, #47990, #29108.

---

## D. Are the gates right?

**D1. A kernel is gated on the architecture family it was built for, never on `major >= N`.**
`major >= 10` and `is_blackwell` (majors 10-12) admitted SM120/SM121 into tcgen05/TMEM and
SM100-only paths (#35361; #45112 for CuTe DSL GDN/KDA prefill; #48399 for the DeepSeek-V4.1
indexer and MSA-v1); `== 120` missed SM121 (#18750); on ROCm gfx950 reports capability
(9, 5), so NVIDIA checks misfired (#28486); `capability >= 9` emitted PDL on gfx942
(#48366); a `"gfx95"` substring test admitted any gfx95x (#48399 uses `gcn_arch ==
"gfx950"`).
Spot: `get_device_capability()[0] >=`; `is_sm_at_least(10)` or `is_blackwell()` in front of
an SM100-only kernel; `== 120`; substring tests on gfx names.
Not this pattern: a path every later architecture supports; a `not is_blackwell()` site
whose `is_sm(9)` rewrite would change HIP or MUSA (#48399 left `single_batch_overlap.py`
alone); SM count, which varies within one architecture and stays a runtime query.
Examples: #35361, #18750, #28486, #45112, #48366, #48399.

**D2. A gate decides whether a kernel applies; a contract violation raises.**
`can_use_*` gates that re-checked caller contracts turned caller bugs into silent fallbacks
although the launchers already raised (criteria in #42184, applied in #44690); wrappers
that accepted parameters but compiled fixed values now raise (#48366); `None` as
"unsupported" was replaced by raising plus a public predicate (#35114).
Spot: gates checking shape agreement between operands; parameters that are never read;
`return None` meaning unsupported; a selector branch missing its `return`.
Not this pattern: shape or dtype support questions, which still return False and take the
fallback.
Examples: #42184, #44690, #48366, #35114, #37552.

**D3. A diffusion fusion runs only at the quality tier its numerics allow: `exact` is bit-identical to native, `lossless` (the default) may move rounding, `high` may change what is computed.**
Non-bit-exact VAE paths were taken off `exact` (#44690, #48399); fused QKV was gated at
`lossless` (#48366); verify-once fusions go through `BitExactFusionGate`.
Spot: a default-on fusion with no tier check; hand-written verify-once blocks.
Not this pattern: sites tuned for launch overhead stay hand-written (the gate costs about
160-180 ns of host time per call).
Examples: #33453, #44690, #48366, #48399.

**D4. Hardware differences leave shared code through capabilities or a backend, not through new `is_xxx()` branches.**
`PlatformCapabilities` (#35594); the NPU communicator and graph runner moved out (#45950,
#47173) under roadmap #46487; vendor fast paths live out of tree behind a vendor-neutral
switch (#43949, #44353, #45389).
Spot: `is_npu` / `_is_hip` branches in shared modules outside `hardware_backend/`; per-model
env vars that toggle a vendor kernel path.
Not this pattern: removing a branch before its backend replacement exists; `is_cuda_alike`
while the #42173 audit is open.
Examples: #35594, #46487, #45950, #47173, #43949.

---

## E. Is it in the right place?

**E1. Moving an internal module updates every importer in the same PR and leaves no forwarding shim; names are imported from the module that defines them.**
#47990 moved 322 imports and deleted the re-export shims; #47992 added the
`check-indirect-imports` hook; #45112 left 94 old module paths with zero references.
Spot: a module that only re-imports; a module-level `__getattr__` that forwards; `from a
import f` where `a` only imports `f` (the hook flags it).
Not this pattern: names bound per platform inside `if` / `try`; a literal `__all__` or a
package `__init__`; documented plugin-facing names; user-facing entry points (C5).
Examples: #47990, #47992, #45112, #41243, #32072. Kernel placement: kernel-organization.

**E2. `srt.utils` is the bottom layer: no module-level imports of domain packages, and a helper with one importing package lives in that package.**
#48758 moved domain modules out; #48781 split `common.py` from 5,208 to 824 lines with
byte-identical moves; the `check-srt-utils-imports` hook enforces the layering.
Spot: a utils helper with one importer; a utils module importing `managers` or `layers` at
module level.
Not this pattern: imports inside a function or under `TYPE_CHECKING`; check for cycles
before choosing the new home.
Examples: #48758, #48781, #48827.

**E3. Model- or feature-specific branches move out of generic files into owner hooks and modules.**
The 601-line model-specific function in `server_args.py` (#20481) moved into
`arg_groups/<x>_hook.py` (#24328, #24326, #25424); shared-expert configuration became owned
by each model (#44913).
Spot: `if arch in (...)` blocks in server args, the scheduler or `ModelRunner`.
Not this pattern: setting an attribute on a wrapper the runner never sees (#44913 broke
MiniCPM-V on AMD that way).
Examples: #20481, #24328, #24326, #25424, #44913.

**E4. A god class is split into collaborators, not mixins, as prepare, mechanical move, follow-up.**
Scheduler mixins (#8483) later became collaborators and the mixins were retired (#25610,
#25624); #26271 extracted 13 `init_<name>()` methods with a byte-equality audit.
Not this pattern: an extraction whose responsibilities nobody can state (#3115, reverted by
#3829).
Examples: #8483, #25610, #25624, #26271, #48119, #49144. See large-class-style and
mechanical-refactor-verify.

**E5. A class's home follows its base class, and a new target directory gets a guard, or the old file keeps growing.**
The mem_cache README maps base classes to directories (#25371); after `allocator/` existed,
`memory_pool.py` still grew from 2,257 to 5,042 lines in three months and new allocators
landed at the top level (#29678).
Spot: subclasses outside the directory their base maps to.
Not this pattern: drive-by moves into a directory that a tracked series owns.
Examples: #25371, #29678.

**E6. Config is read-only after resolution, per-instance differences are constructor arguments, and readers switch to a new source of truth only after every writer writes it.**
`ServerArgs.override` and `derive` were retired (#33490, #33887); `ServerArgs` holds only
the raw input (#36255); `get_context().override()` is the one mutation entry. #32100
reverted five merged migration PRs that moved readers first (late-override desync, settings
leaking between two `Engine`s, the draft load format ignored).
Spot: `server_args.<x> =` writes; `deepcopy(server_args)`; a value stored in two places with
late writers.
Examples: #33490, #33887, #36255, #31813-#31817 / #32100, #11331 / #11520.

---

## F. Is it the right shape?

**F1. No abstraction without a second user; prefer flat layouts and plain functions.**
A three-level hook layout was flattened (#24326); helpers kept "for future use" were
deleted in review (#7167); an extra data structure and lock were rejected (#21392); an
auto weight loader was closed when no model adopted it (#12534).
Spot: a single-implementation ABC or registry; a wrapper that only forwards; a lock with no
concurrency; `class \w+Mixin`.
Not this pattern: an extension point out-of-tree plugins use (#45389).
Examples: #24326, #7167, #21392, #12534, #3115 / #3829.

**F2. Inherit only when the reuse is substantial.**
TRTLLMHA overrode all attention logic of its FlashInfer parent and allocated state it never
used (#28808); a backend that overrode only `forward_decode` was part of a removal case
(#32111).
Spot: a subclass that overrides every `init_*` and `forward_*`.
Examples: #28808, #32111.

**F3. Rename only to fix meaning, in one sweep including tests and docs; never for taste in hot files.**
`InputMetadata` to `ForwardBatch` (#1543), NSA to DSA (#25821), `--dp-size` split into two
flags (#41818), a "sm90" helper that also ran SM120 (#32405).
Not this pattern: typo and capitalization sweeps over hot files (#6209, reverted by #6244);
`TokenizerManager` to `StdOrchestrator` (#3116, reverted by #3828 and never re-landed);
names pinned outside the repo (#45389); code about to migrate anyway (#30065).
Examples: #1543, #25821, #41818, #32405, #6209 / #6244, #3116 / #3828.

**F4. Migrations to the typed conventions (direct field access, `msgspec.Struct`) go one subsystem at a time.**
Defensive `getattr` removals (#25435, #29549, #31804); one found a gate that never fired
(#38059); IPC structs moved to msgspec behind an opt-in first (#28688).
Not this pattern: duck-typed platform objects; flipping a wire format default without an
audit path; drive-by sweeps, which `.claude/rules/no-dataclasses.md` forbids.
Examples: #25435, #29549, #38059, #28688, #38753.

---

## G. Landing it

**G1. A refactor PR carries no behavior change; a change it exposes lands separately, or is listed and tested on its own.**
Spot: "pure" or "mechanical" in the body next to "fixes"; a "split" or "refactor" title on a
diff that touches memory budgets or defaults; a transform script that does not reproduce
the whole diff.
Examples: #28683 / #28841 / #28855, #26134 / #26166 / #26665, #7163 / #7210 / #7213.

**G2. Wrong-looking logic is not deleted until you can say what it protected.**
Spot: deletions in KV or memory sizing, graph pools, health or watchdog code, or `auto`
backend selection, with no statement of why it is safe.
Examples: #14167 / #14259, #14044 / #14278, #20227 / #20259, #12291.

**G3. Everything importable from `sglang`, every HTTP route and every `Engine` or SDK method name is public API.**
Spot: F401 fixes in `__init__.py`; renames under `lang/` or `entrypoints/`; route changes.
Examples: #378 / #385, #3969, #11685, #13607 / #14533, #809 / #815.

**G4. Code leaving a shared package migrates every caller and registration first and deletes the old copy last, in a separate PR.**
AOT to JIT went in two phases: JIT with a bitwise JIT-vs-AOT test and an end-to-end eval,
then the AOT deletion (#18067, #18543, #18949 then #19241; #19181 then #19379).
Spot: one PR that deletes under `sgl-kernel/csrc` or `common_extension.cc` and adds the JIT
twin; wrapper arity changes; edits to `m.def` or custom-op registration.
Examples: #16879 / #17257 / #17658, #19437 / #20005 / #20012, #4890 / #4944, #19241.

**G5. Other platforms are run, or signed off by their owners, before shared code changes.**
Examples: #11322 / #12028, #36228, #42245, #25678, #25663.

**G6. Green PR CI is not coverage: run the nightly and extra tests that exercise the change, and validate a test on the exact hardware of the lane it is registered on.**
#47784 failed a nightly-only test; #28683 broke extra-stage jobs; #40475 was merged with NPU
CI cancelled and reverted the same day (#40611); a test validated on sm_103 failed on the
sm_100 lane it was registered on (#48366).
Examples: #47784, #28683, #40475 / #40611, #48366.

**G7. Removing a fallback, default backend or image variant needs a fallback story and numbers on the target hardware.**
Examples: #20583 (closed), #16000 (closed), #48325 / #48637, #30438 (accepted with a
migration plan), #44803 (TTFT A/B).

**G8. Before starting, check open PRs, trackers and RFCs for duplicates and in-flight migrations.**
Examples: #9607 and #10479, #23087 and #22094, #46693; trackers #24703, #30696, #34510,
#46487.

**G9. Core-module refactors are discussed with their owners first and ship as small, independently revertable PRs; no line-count or deletion quotas.**
Examples: #24335, #20415, #20481, #37405.

**G10. Revert fast; re-land as "Redo of #X" with a why-it-was-reverted section and a test for the cause; never cherry-pick a refactor to a release branch the day it merges.**
Examples: #48174, #28855, #20012, #26665, #27659 / #27760.

**G11. The diff stays on topic: no stray hunks, review notes or personal scaffolding.**
Examples: #22670 / #25044, #33224, #32405, #40475.

---

## In-flight directions (2026-11; check the tracker before relying on this)

A cleanup must not fight these. Leave their dual paths, gates and half-moved code alone
unless the change is part of the tracked series.

- Weight loader v2: `SGLANG_ENABLE_WEIGHT_LOADER_V2` (default off) and the dual paths in
  `llama.py` / `qwen2.py` are a migration gate (#24703, tracker #31051).
- RuntimeContext: new code reads `get_<namespace>()`; `ServerArgs.override` / `derive` must
  not come back; the half reverted in #32100 is meant to re-land (#30696).
- mem_cache: `pool/` has not started; that series uses `[mem_cache][N/N]` titles and
  mechanical-refactor-verify (#25371, #35647).
- Radix cache: Python is still the default TreeCore backend; per-model caches converge on
  `unified_cache/` (#20415).
- PD transfer: transport steps are open (#34510); do not hand-dedup the `conn.py` files.
- Quantization layout: new methods follow the contribution guide; the large files are not
  split in drive-bys (#38004).
- MoE: backend code inside quantization `apply()` is meant to move into
  `moe_runner/<backend>.py` (#8715).
- NPU (#46487), platform capabilities (#42173), parallelism-independent model code (#47170),
  the Rust frontend (#23206, #45816, #40920), monokernels (#43949).
