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

## Open risks (carry to the decode-branch wiring)
- The internally-allocating `dequantize_k_cache_paged` is not graph-safe → the
  research path must run eager (gate it off the production capture path; the
  validator already requires `--disable-cuda-graph` for the opt-in until the
  `out=`/scratch hardening lands).
- ~~`flash_mla_sparse_fwd` accuracy at top-k > 512 is unproven locally~~
  **RESOLVED (R12)**: the direct 4K smoke + the fp8 end-to-end smoke pass.
- The selection budget must be widened from `max_top_k = top_k` to
  `lifted_budget_top_k` for the opt-in path (selector + the `ds_graph_state` /
  `ds_topk_indices_out` scratch shapes); enforce the `%128` width constraint.
- Small oracle N at 4K → the recall recovery must be measured served at N≥20 with
  CIs, not inferred from the score-only oracle.

## Artifacts
`config.py` (ABI fields + validation), `validator.py` (fail-closed gate),
`selection_kernel.py::ds_lifted_budget_decode_available` (seam),
`double_sparsity/lifted_budget.py` (remap), `test_scorer_variants.py::TestLiftedBudgetABI`,
`test_lifted_budget_decode.py` (remap + kernel smokes), this doc. Codex review:
`.humanize/skill/<ts>/output.md`.
