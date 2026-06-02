# Loop 7 — Tier-2.A lifted-budget decode: ABI + design/disposition (AC-4 / task13)

The M0 oracle proved the budget-vs-scorer split is **regime-dependent**: 4K is
budget-limited (needle ranks just past 2048, recovered at 4096), 16K/64K are
scorer-limited (needle rank ≈ its sequence position). So Tier-2.A (a wider decode
budget) is a **bounded-secondary** lever — it can only recover the ≤~moderate
regime where the needle ranks in `(index_topk, lifted_budget_top_k]`; Tier-2.B
(the landed hybrid scorer) is the only lever for the long-context goal.

## Landed: the opt-in ABI (config) + fail-closed validator gate
- **Config fields (R10).** `DoubleSparsityConfig.enable_lifted_budget_decode: bool`
  (default `False`) and `lifted_budget_top_k: int` (default `0`). Parse-level
  validation: `lifted_budget_top_k` must be `> top_k` when enabled; set-without-flag
  and flag-without-budget both fail closed. The config still *parses* a lifted
  spec (the field is recognized) — the boot-time availability decision lives at
  the server validator, the layer that knows whether the backend exists.
- **Fail-closed validator gate (R11).** `validate_double_sparsity` raises a clear
  *recognized-but-not-implemented/selected* error whenever
  `enable_lifted_budget_decode` is set, because the opt-in decode backend path is
  not built yet. This is gated by a single capability seam
  `ds_lifted_budget_decode_available()` (returns `False` today; the decode-path
  landing flips it to `True`), mirroring `ds_scorer_is_graph_safe`. The gate is
  **independent of hf_config resolution** (it fires before the capability/model-topk
  block), so it cannot be skipped when the model config can't be resolved. This
  closes the R10-review hole where a lifted config booted into either a **silent
  no-op** (`top_k == index_topk` → the locked 2048 selector ignored the wider
  budget) or routed a **wider-than-2048 selection into the default `flashmla_kv`
  `indices.shape[-1] == dsa_index_topk` assert** (`top_k=4096, lifted_budget_top_k=8192`).
- **Steering preserved.** For the *no-flag* case, `top_k > index_topk` is still
  **rejected** with a message that steers to the ABI and forbids
  `SGLANG_DS_ALLOW_TOPK_MISMATCH` / `max_top_k` / Twilight as the mechanism. The
  model-topk block's lifted-specific validation (`lifted_budget_top_k > index_topk`,
  the info log) is retained as the *post-backend* layer: it becomes reachable once
  `ds_lifted_budget_decode_available()` flips. **Default-off leaves the DSA
  `dsa_index_topk` assert + the `SGLANG_DS_ALLOW_TOPK_MISMATCH` equality-mismatch
  ablation unchanged.**
- Tests: `TestLiftedBudgetABI` — config accept/reject matrix; the no-flag
  `top_k>index_topk` steering gate (monkeypatched `get_dsa_index_topk`); and the
  two R10-review fail-closed cases (`top_k=2048 + lifted_budget_top_k=4096`;
  `top_k=4096 + lifted_budget_top_k=8192`) now both RAISE.

## Decode-path design for task14–17 (Codex-reviewed)
> **Enablement seam.** task14 flips `ds_lifted_budget_decode_available()`
> (`selection_kernel.py`) to `True` once the path below is implemented and
> selected; that single change lifts the R11 fail-closed validator gate, after
> which the validator's model-topk block validates the lifted shape
> (`lifted_budget_top_k > index_topk`) and logs the enabled op-point.

1. **Selection.** When enabled, the selector picks `lifted_budget_top_k` logical
   positions (a FIXED, padded budget; `-1`-pad the tail) via the same graph-safe
   top-K + the R23 deterministic `(score-desc, position-asc)` tie-break. The
   effective count exceeds 2048 only in the budget-limited regime; elsewhere the
   lifted tail is `-1`.
2. **Compact remap (request-local).** logical pos → physical KV slot →
   `page_table_1_flattened` → a **request-local COMPACT dequant index** =
   `request_base + selected_rank` (an ORDINAL position in the compact dequant
   buffer, NOT a global lookup by physical-slot value). For `flash_mla_sparse_fwd`
   (no 2048 cap).
   - **Prefix-sharing hazard (Codex):** the same physical slot appearing in
     multiple requests' `page_table_1_flattened` is SAFE *iff* each request gets
     its own compact index space (request-local base). Duplicates **within one
     query row are NOT safe** — `flash_mla_sparse_fwd` would attend a duplicate
     valid index twice; the selector output is already unique per row (top-K over
     distinct logical positions), and the design **asserts/dedups uniqueness after
     the physical remap**.
3. **Padding safety (Codex).** `dequantize_k_cache_paged` blindly loads
   `page_table_1_flattened`, so a `-1` there is invalid: pad rows use a **safe
   physical placeholder** and their compact index stays `-1` (masked before any
   dequant/index op), or pack only the valid rows. No `-1`/pad reaches dequant.
4. **DSA default untouched** when the flag is off (the `dsa_index_topk` assert and
   the default `flashmla_kv` decode path are unchanged).

## Disposition (DEC-4 / DEC-6: landed-or-deferred-with-evidence)
Per Codex, deferring the alloc-free `out=`/scratch `dequantize_k_cache_paged`
variant + CUDA-graph landing to **task16** is consistent with DEC-4/DEC-6 ONLY if
task14/15 is **explicitly eager/research-gated** (proves the 4K recall recovery),
has recall evidence recorded, and **cannot enter production graph capture**
(the internally-allocating dequant is not graph-safe). Since the M0 evidence makes
Tier-2.A bounded-secondary (it does not address the long-context goal that
Tier-2.B already serves), the planned disposition is:
- **task14/15 (next): an EAGER research lifted-budget path + correctness/safety
  tests** — including the Codex-required **direct `flash_mla_sparse_fwd` 4K-topk
  smoke/accuracy test** (local coverage today is sparse-prefill top-k ≤ 512) —
  to prove the 4K budget recovery on the served path.
- **task16 (gated): production hardening** (alloc-free dequant + CUDA-graph
  capture + perf) pursued ONLY if the recall win justifies the heavy kernel; else
- **task17: a disposition record** that the recall evidence is recorded and
  production-hardening is carried to a follow-on with the DSA default untouched —
  which **closes AC-4** under the plan's "deferred-with-evidence" branch.

## Landed (decode index core + kernel proof)
The trap-laden correctness core of the decode path — the request-local
physical→compact remap — is implemented as a standalone, deterministic module
(`double_sparsity/lifted_budget.py::build_compact_decode_index`) with CPU unit
tests, and the kernel half is proven on GPU:
- **Remap.** Given per-request selected physical slots (selector order, fixed
  padded width) + `valid_lengths`, it emits `page_table_1_flattened` (valid
  slots only, **no `-1`** — `dequantize_k_cache_paged` blindly loads it) and
  request-local **compact-domain** ordinals (`request_base + rank`, `-1` pads).
  Prefix-sharing is isolated (each request its own compact span), within-row
  duplicates are collapsed to the highest-rank occurrence (and counted), and the
  selector's deterministic order is carried into the ordinals. CPU tests pin all
  cases (`test_lifted_budget_decode.py::TestCompactDecodeIndex`, 8 tests).
- **Kernel proof (GPU, H200/sm90).** `flash_mla_sparse_fwd` attends a request
  selecting **3000 > 2048** rows inside a 4096-wide padded budget and matches a
  reference attention (proves the no-cap behavior the lifted budget needs); a
  full **fp8 → `dequantize_k_cache_paged` → `flash_mla_sparse_fwd`** pipe with
  prefix-sharing matches a reference attending the dequantized selected slots,
  and the compact rows are bit-identical to the full-dequant gather
  (`TestLiftedBudgetKernelSmoke`, 2 tests).

### Kernel contract confirmed (binds the wiring)
- `flash_mla_sparse_fwd` masks indices that are `< 0` **or** `>= s_kv` — so the
  compact pad lane is simply `-1` (no safe-placeholder gymnastics needed in the
  COMPACT domain; the placeholder concern was about `page_table_1_flattened`,
  which we keep pad-free instead).
- **The padded index width (`lifted_budget_top_k`) must be a multiple of the
  kernel block — `topk % (2*B_TOPK) == 0`, i.e. a multiple of 128** (a `width=8`
  smoke hit `Assertion params.topk % (2*B_TOPK) == 0`; `width=256`/`4096` pass).
  The realistic budgets 4096/8192 satisfy this; the next-round config/validator
  must enforce `lifted_budget_top_k % 128 == 0`.

## Landed (served eager decode branch — wired + enabled)
The opt-in eager lifted-budget decode branch is wired end-to-end and the
availability seam is flipped (`ds_lifted_budget_decode_available()` → `True`):
- **Config**: `lifted_budget_top_k % 128 == 0` is enforced (the kernel-block
  constraint), alongside `> top_k`.
- **Validator**: when enabled, requires `top_k == index_topk` (base budget stays
  the DSA budget; `lifted_budget_top_k` is the separate wider width),
  `lifted_budget_top_k > index_topk`, `% 128 == 0`, **and `--disable-cuda-graph`**
  (the dequant is not graph-safe). The R11 fail-closed "not implemented" gate is
  replaced by these checks (kept as defense if a build ever ships the flag without
  the backend).
- **Selection width**: `DoubleSparsitySelector.max_top_k` and the backend's
  `ds_max_top_k` (which sizes `ds_topk_indices_out` + `ds_graph_state`) become
  `lifted_budget_top_k` when enabled, so the selector emits a lifted-width
  selection with the R23 tie-break unchanged.
- **Decode branch**: `DeepseekSparseAttnBackend.forward_decode` routes the lifted
  case (physical `page_table_1`, the FUSE_TOPK default) to `_forward_lifted_budget`
  → `build_lifted_compact_kv` (remap + `dequantize_k_cache_paged` for the fp8
  store, gather for bf16) → the existing `_forward_flashmla_sparse`
  (`flash_mla_sparse_fwd`). Everything is behind a default-off
  `getattr(self, "ds_lifted_budget_decode", False)` guard, so the default DSA/DS
  decode is byte-identical and the `flashmla_kv` `dsa_index_topk` assert is untouched.
- **Tests**: `TestLiftedBudgetABI` (config `%128` reject/accept; validator
  eager-required, `top_k==index_topk`, valid-config-passes); `test_lifted_budget_decode`
  GPU served-helper tests at **4096 and 8192** widths incl. prefix-sharing,
  `valid_lengths` < width, and an **interior `-1` from within-row dedup**, all vs a
  reference attention. 337 DS unit tests + 9 subtests pass.

## Open items (carry to hardening)
- ~~**Served recall**~~ **RESOLVED (R14, `m8_lifted_recall_finding.md`)**: served
  NIAH 4K, N=20, both eager same node — DS-lifted-4096 **95% [75.1,99.9]** vs
  DS-default-2048 **75% [50.9,91.3]**, **+20pp material** (lifted > base CI-high).
  Confirms the M0 4K budget-limited attribution on the served path (eager-mode number).
- ~~**TP=8 equality**~~ **RESOLVED (R14)**: lifted-width 4096/8192 selected-index +
  valid-length equality across 8 gloo ranks
  (`test_ds_scorer_tp_determinism.py::TestTP8LiftedWidthDeterminism`).
- ~~`flash_mla_sparse_fwd` accuracy at top-k > 512 unproven~~ **RESOLVED (R12)**.
- ~~selection budget widening / `%128` enforcement / decode wiring~~
  **RESOLVED (R13)**: selector + backend widen to `lifted_budget_top_k`, config
  enforces `%128`, the decode branch + validator gating are wired.
- **task16 hardening**: the dequant is not graph-safe → the path is eager-required
  (validator-enforced); an alloc-free `out=`/scratch dequant + CUDA-graph capture is
  needed before production graph use.

## Artifacts
`config.py` (ABI + `%128`), `validator.py` (lifted gating: eager-required,
`top_k==index_topk`, `%128`), `selector.py` + `dsa_backend.py` (lifted width +
`_forward_lifted_budget` decode branch),
`selection_kernel.py::ds_lifted_budget_decode_available` (seam → True),
`double_sparsity/lifted_budget.py` (`build_compact_decode_index` +
`build_lifted_compact_kv`), `test_scorer_variants.py::TestLiftedBudgetABI`,
`test_lifted_budget_decode.py` (remap + kernel/served-helper tests). Codex review:
`.humanize/skill/<ts>/output.md`.
