# Refine Plan QA

## Summary

One comment block was extracted from `development/plan.md` (the converged output of the prior `humanize:gen-plan` run). It is a single, high-impact `change_request` from the user that rejects the plan's central architectural decision — integrating Double Sparsity as a HiSparse algorithm — and instructs a redesign in which Double Sparsity is a **standalone** SGLang feature, independent of HiSparse, with HiSparse and HiCache integration deferred to a downstream phase. The refinement applies the redirect throughout the plan: a new top-level CLI (`--enable-double-sparsity` / `--double-sparsity-config`), a new package location (`python/sglang/srt/layers/attention/double_sparsity/`), an independent validator, an independent metric namespace (`sglang_double_sparsity_*`), and a two-column (DS off / DS on) benchmark instead of the prior three-column HiSparse comparison. All 12 acceptance criteria, all 20 tasks, all 6 milestones, all 7 pending decisions, and the deliberation log were updated for consistency. Convergence status changes from `converged` to `partially_converged` because the seven pre-existing `PENDING` user decisions are still unresolved.

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1  | change_request | `## Acceptance Criteria` preamble (input lines 29–31, between "deterministic verification." and "- AC-1:") | "Absolutely not, at its current state hi-sparse requires PD-disaggreation to be turned on … Hicache integration was explicitly listed as a requirement only after client deliverables were already completed." | applied |

## Answers

*No `question`-type comments were extracted in this refinement.*

## Research Findings

The change_request embedded a factual claim about the current repository state ("hi-sparse requires PD-disaggregation to be turned on and is enabled on the decode instance only"). The refinement verified this claim via targeted reads before applying the redirect, so the architectural pivot is grounded.

### CMT-1 supporting research

**Research Scope:**
- `python/sglang/srt/managers/scheduler.py` — searched for `enable_hisparse`, `hisparse_coordinator`, `_build_hisparse_decode_batch`, `set_decode_producer_stream`.
- `python/sglang/srt/arg_groups/hisparse_hook.py` — re-read `validate_hisparse`.
- `python/sglang/srt/disaggregation/decode.py` — searched for `hisparse` references.
- `python/sglang/srt/server_args.py` — searched for the interaction between `enable_hisparse` and `disaggregation_mode`.
- `python/sglang/srt/mem_cache/sparsity/factory.py` and `core/sparse_coordinator.py` — confirmed `SparseCoordinator` is currently only constructed under the HiSparse path.

**Findings:**
- The scheduler hard-wires HiSparse to the decode path: `_build_hisparse_decode_batch` (line 2462), `set_decode_producer_stream(self.forward_stream)` (line 982), `self.hisparse_coordinator.collect_ready_reqs()` (line 2524), and an idle-check that depends on `hisparse_coordinator.has_ongoing_staging()` (line 3370). These are activated by `self.enable_hisparse` (line 387) but are themselves decode-side.
- `python/sglang/srt/disaggregation/decode.py` has extensive HiSparse hooks (`scheduler.enable_hisparse` branches around line 382, 787, 819, 896, 910, 1083, 1249) governing direct-to-host admission, request budgeting, and L1 radix-cache compatibility on the decode side.
- `validate_hisparse` (`hisparse_hook.py`) does not directly assert `disaggregation_mode != "null"`, but the surrounding scheduler / decode wiring assumes the request enters via the disaggregation decode path. Running HiSparse on a single-instance server is therefore not a supported configuration today.
- There is no public entry point to the `mem_cache/sparsity/` framework other than `--enable-hisparse`; building DS on that framework would inherit the HiSparse + PD-decode coupling.

**Impact on Plan:**
The user's claim is confirmed: HiSparse today is PD-decode-only. The refined plan therefore treats HiSparse as inspiration / borrow-source only and routes DS through a new, standalone module under `python/sglang/srt/layers/attention/double_sparsity/`. This conclusion is encoded in the new "Standalone, Not a HiSparse Algorithm" subsection of `## Goal Description`, in AC-1 / AC-2 / AC-7 / AC-10 / AC-12, in the Path Boundaries "Cannot use" list, and in the new DEC-6 framing.

## Plan Changes Applied

### CMT-1: Reframe DS as standalone; HiSparse / HiCache integration is downstream

**Original Comment:**
```
Absolutely not, at its current state hi-sparse requires PD-disaggreation to be turned on and is enabled on the decode instance only. Users must be able to use double sparsity even if PD-disaggregation is not being used. Hisparse was just provided as an example of a performance succesfull ship, and you can borrow some code from the implementation that will be useful for the double sparse implementation, but don't immediately just to integrate it with double sparse. Hicache integration was explicitly listed as a requirement only after client deliverables were already completed.
```

**Changes Made:**
- Plan title changed from "Deliver Double Sparsity for DeepSeek-V3.2 (FP8) via the HiSparse Framework" to "Deliver Standalone Double Sparsity for DeepSeek-V3.2 (FP8)".
- A new "Standalone, Not a HiSparse Algorithm" subsection was added to `## Goal Description` documenting the architectural pivot and citing the supporting code references discovered in the research scope above.
- The prior "Two Coordinators, One Plan" subsection was removed (no coordinator seam is required when DS does not invoke `SparseCoordinator`).
- The "Resume-vs-Restart Recommendation" was updated to specify a standalone implementation (`python/sglang/srt/layers/attention/double_sparsity/`) and a new branch name (`dev/double-sparsity-standalone`).
- AC-1 was rewritten: DS is now exposed via `--enable-double-sparsity` and `--double-sparsity-config`, not `--enable-hisparse --hisparse-config '{"algorithm":"double_sparsity"}'`. New negative tests cover the mutual-exclusion with `--enable-hisparse` and the artifact-path requirement.
- AC-2 was rewritten: the previously stubbed `NSABackendAdaptor.adapt_for_attn_metadata` is no longer the central enabling lift; instead the DS selector hooks into the V3.2 NSA attention path directly. A new negative test asserts zero net additions under `python/sglang/srt/mem_cache/sparsity/algorithms/` and zero registrations in `_ALGORITHM_REGISTRY`.
- AC-3's backend-pair language was updated to refer to "the V3.2 MLA backend rule" rather than naming `flashmla_kv` / `flashmla_sparse` as HiSparse-validator outputs.
- AC-7 was reduced from three columns to two (`DS off (native NSA)` / `DS on`). The HiSparse + native NSA column is moved to optional follow-up benchmarking under future-work notes.
- AC-10 renamed the Prometheus namespace from `sglang_hisparse_double_sparsity_*` to `sglang_double_sparsity_*`.
- AC-12 updated the allowed-paths whitelist to point at `python/sglang/srt/layers/attention/double_sparsity` (the standalone location) and explicitly listed forbidden additions under `mem_cache/sparsity/algorithms/double_sparsity/` and `arg_groups/hisparse_hook.py`.
- The Path Boundaries "Allowed Choices" `Can use` list was rewritten to permit "code patterns borrowed from HiSparse" (kernels, JSON shape) and to drop `SparseCoordinator`, `HiSparseCoordinator`, `NSABackendAdaptor`, and `BaseSparseAlgorithmImpl` inheritance from the allowed list. The `Cannot use` list now explicitly prohibits registering DS in `_ALGORITHM_REGISTRY`, calling `SparseCoordinator`, extending `hisparse_hook.py`, gating DS behind `--enable-hisparse`, and requiring `disaggregation_mode != "null"`.
- Feasibility Hints #1–#8 were rewritten end-to-end: package location, `DoubleSparsitySelector` class (not `DoubleSparsityAlgorithm`), a new `validator.py`, the V3.2 attention-path hook description, and the radix-cache decision moved out of `hisparse_hook.py` into the new DS validator.
- A new "Future-Work Notes" subsection in `## Feasibility Hints and Suggestions` documents the downstream integration paths to HiSparse, PD-Disagg, and HiCache so that the user's "Integration into all other sglang features" downstream requirement is acknowledged without inflating the initial scope.
- All seven milestones were restructured. Milestone 1 is now "Server args + validator + V3.2 attention-path seam" (replacing the prior "Coordinator seam + NSA adaptor"). Milestone 2's package skeleton points to the new location. Milestone 4 unchanged. Milestones 5 and 6 reflect the two-column benchmark and the namespace rename.
- Task Breakdown updated: task 3 (was "coordinator seam") is now "add server args"; task 4 (was "complete NSABackendAdaptor") is now "land V3.2 attention-path branch"; task 10 (was "register in `_ALGORITHM_REGISTRY` + extend `hisparse_hook.py`") is now "land `validate_double_sparsity`"; task 5 → task 15 dependency was made explicit (task 15 previously depended only on task 13, task 14, which under-specified the benchmark-script split sequence). All other task descriptions updated for consistency with the standalone architecture.
- Implementation Notes adds a "Symbol-name boundary" rule forbidding imports from `mem_cache/sparsity/`, `managers/hisparse_coordinator.py`, and `arg_groups/hisparse_hook.py` into the new `double_sparsity/` package.
- `## Claude-Codex Deliberation` adds a new resolved-disagreement entry "CMT-1 user redirect (architectural pivot)" with the user's verbatim quote and the supporting code references.
- `## Pending User Decisions` DEC-2 simplified: the per-algorithm radix-cache gate on `hisparse_hook.py` is no longer needed because DS doesn't flow through HiSparse; the page-stability fixture (M3-B) now governs the DS validator's own radix-cache permission. DEC-6 expanded to explicitly list HiSparse, PD-Disagg, and HiCache integration as deferred. DEC-7 updated to reference `--double-sparsity-config` instead of `--hisparse-config`'s `sparse_extra_config`.
- `### Convergence Status` updated from `converged` to `partially_converged` per the refine-plan rule that pending decisions must trigger partial-convergence.

**Affected Sections:**
- `## Goal Description` (title + new "Standalone, Not a HiSparse Algorithm" subsection; "Two Coordinators, One Plan" subsection removed; recommendation updated).
- `## Acceptance Criteria` (AC-1, AC-2, AC-3, AC-7, AC-10, AC-12 substantively rewritten; AC-4, AC-5, AC-6, AC-8, AC-9, AC-11 minor edits for consistency).
- `## Path Boundaries` (Upper Bound, Lower Bound, Allowed Choices — Can use and Cannot use lists rewritten).
- `## Feasibility Hints and Suggestions` (Conceptual Approach hints #1–#8 rewritten; new "Future-Work Notes" subsection added; Relevant References updated).
- `## Dependencies and Sequence` (all seven milestones updated).
- `## Task Breakdown` (task 3, task 4, task 10, task 15 dependency, task 20 PR-description requirement updated; task IDs preserved).
- `## Claude-Codex Deliberation` (Agreements updated; Resolved Disagreements gains CMT-1 entry; Convergence Status flips to `partially_converged`).
- `## Pending User Decisions` (DEC-2, DEC-6, DEC-7 updated).
- `## Implementation Notes` (new symbol-name boundary rule added).

**Cross-Reference Updates:**
- No `AC-N` identifiers were renumbered; all 12 ACs retain their original IDs.
- No task IDs were renumbered; all 20 tasks retain their original IDs.
- All `task-N` dependency references were re-checked; the missing `task5 → task15` link was added.
- All references to the old `mem_cache/sparsity/algorithms/double_sparsity/` package path were updated to `layers/attention/double_sparsity/`.
- The Prometheus metric prefix `sglang_hisparse_double_sparsity_*` was renamed to `sglang_double_sparsity_*` consistently in AC-10 and the deliberation log.
- The `--hisparse-config` references in DEC-7 were updated to `--double-sparsity-config`.
- The original draft appendix is preserved verbatim at the bottom of the refined plan.

## Remaining Decisions

CMT-1 was fully resolved by the change above. The seven pre-existing `PENDING` decisions inherited from the prior `gen-plan` output remain open, summarized below in the order they appear in the refined plan.

### DEC-1: SLO definition + hardware

**Related Comments:** none directly; inherited from `gen-plan` first-pass.

**Context:** "30 tokens/s with a P99 TTFT of < 22s" is ambiguous: per-request vs aggregate, P50 vs P99 for throughput, hardware undefined.

**Options:**
1. Per-request P50 ≥ 30 tok/s, P99 TTFT ≤ 22 s, H200 8-way TP (Claude default).
2. Aggregate throughput ≥ 30 tok/s (likely trivial and probably not what the client meant).
3. Per-request throughput on smaller hardware (H100 8-way TP) — kernel choices may need to change.

**Recommendation:** Option 1.

**Status:** PENDING

### DEC-2: Radix cache reconciliation

**Related Comments:** none directly; reframed by CMT-1's standalone architecture.

**Context:** With DS standalone, the existing HiSparse `assert disable_radix_cache` does not apply. The DS validator must instead decide whether to keep radix cache enabled by default.

**Options:**
1. Default-on; gate on the M3-B page-stability fixture passing (Claude default).
2. Default-off; require users to opt in to radix cache.

**Recommendation:** Option 1; the workload requires ~55 % prefix-cache hit and DS labels are deterministic functions of K pages.

**Status:** PENDING

### DEC-3: Quality threshold deltas vs native NSA

**Related Comments:** none directly; inherited.

**Context:** AC-9 needs concrete thresholds.

**Options:**
1. NIAH within 5 pp of native NSA; MMLU within 1.0 pp (Claude default).
2. Tighter (NIAH 3 pp / MMLU 0.5 pp).
3. Looser (NIAH 7 pp / MMLU 2 pp).

**Recommendation:** Option 1.

**Status:** PENDING

### DEC-4: Calibration ownership and artifact distribution

**Related Comments:** none directly; inherited.

**Context:** Should the DeepSeek-V3.2 FP8 calibration artifact be committed to the repo, hosted externally, or both?

**Options:**
1. Script in repo; artifact external to repo; tiny NSA fixture for CI (Claude default).
2. Script + artifact both committed (wheel bloat).
3. Script external too (deployment burden).

**Recommendation:** Option 1.

**Status:** PENDING

### DEC-5: Semantic relationship of DS to DeepSeek-V3.2 NSA

**Related Comments:** none directly; inherited (CMT-1 confirms DS is V3.2-attention-path-internal).

**Context:** DS can replace, augment, or be stacked atop V3.2's NSA selector.

**Options:**
1. Replace NSA's selector with DS (Claude default; Codex agrees).
2. Stack DS after NSA (likely quality regression).
3. Augment NSA's indexer internals.

**Recommendation:** Option 1.

**Status:** PENDING

### DEC-6: Scope of deferred-requirements coverage

**Related Comments:** CMT-1 (HiSparse + HiCache moved firmly out of initial scope).

**Context:** GLM-5, 128K ISL, FP4 weights, HiSparse integration, PD-Disagg integration, HiCache integration — which constrain the initial design?

**Options:**
1. All deferred; selector ABI + artifact schema shaped to admit them; task 6 produces a one-page schema-compatibility memo before the loader merges (Claude default).
2. Roll HiSparse integration back into initial scope (rejected by CMT-1).
3. Roll GLM-5 into initial scope (significant kernel-shape implications).

**Recommendation:** Option 1.

**Status:** PENDING

### DEC-7: "Extensions as a general knob for the sglang engine" interpretation

**Related Comments:** none directly; inherited.

**Context:** The draft mentions "Extensions as a general knob" as a downstream requirement. Concrete shape is unspecified.

**Options:**
1. Expose DS runtime knobs through `--double-sparsity-config` JSON; no plugin system (Claude default).
2. Introduce a generic plugin / extension system (separate design effort).

**Recommendation:** Option 1.

**Status:** PENDING

## Refinement Metadata

- **Input Plan:** `/sgl-workspace/sglang/development/plan.md`
- **Output Plan:** `/sgl-workspace/sglang/development/refined_plan.md`
- **QA Document:** `/sgl-workspace/sglang/.humanize/plan_qa/plan-qa.md`
- **Total Comments Processed:** 1
  - Questions: 0
  - Change Requests: 1
  - Research Requests: 0
- **Plan Sections Modified:** `## Goal Description` (title + new subsection + removed subsection + recommendation), `## Acceptance Criteria` (AC-1, AC-2, AC-3, AC-7, AC-10, AC-12 substantive; AC-4, AC-5, AC-6, AC-8, AC-9, AC-11 minor), `## Path Boundaries` (Upper Bound, Lower Bound, Allowed Choices), `## Feasibility Hints and Suggestions` (Conceptual Approach + new Future-Work Notes + Relevant References), `## Dependencies and Sequence` (all seven milestones), `## Task Breakdown` (task 3, task 4, task 10, task 15 dependency, task 20), `## Claude-Codex Deliberation` (Agreements, Resolved Disagreements, Convergence Status), `## Pending User Decisions` (DEC-2, DEC-6, DEC-7), `## Implementation Notes` (symbol-name boundary).
- **Convergence Status:** `partially_converged`
- **Refinement Date:** 2026-05-19
- **Mode:** `discussion`
