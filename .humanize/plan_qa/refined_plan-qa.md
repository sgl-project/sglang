# Refine Plan QA

## Summary

Nineteen critique comments were extracted from `development/refined_plan.md` — twelve `[Linus]` blocks authored in a prior session and seven `[Codex]` blocks added in a follow-up Codex review. All nineteen are dominantly `change_request`, with research and decision sub-items embedded in several. Sixteen are resolved as `applied` plan edits; two (`CMT-7` and `CMT-13`) are `researched` and converted into new `Pending User Decisions`; one (`CMT-18`) is `applied` as an execution-gate annotation and also surfaces an existing decision dependency. The refinement adds three new pending decisions (`DEC-8` HiSparse + DS coexistence, `DEC-9` TP rank synchronization, `DEC-10` V3.2-scope vs MLA-capability validator) and adjusts substantial portions of `## Goal Description`, `## Acceptance Criteria` (especially AC-1, AC-2, AC-4, AC-6, AC-7, AC-11), `## Path Boundaries`, `## Feasibility Hints and Suggestions`, `## Dependencies and Sequence`, `## Task Breakdown` (task 18 removed; task 20 dependencies updated), `## Claude-Codex Deliberation`, and `## Implementation Notes`. Convergence status is `partially_converged` because seven pre-existing decisions plus three new ones remain `PENDING`; per CMT-18, `DEC-1` and `DEC-3` are blocking for Milestone 5.

## Comment Ledger

| CMT-ID | Classification | Location | Original Text (excerpt) | Disposition |
|--------|----------------|----------|-------------------------|-------------|
| CMT-1  | change_request | `## Goal Description` (input line 7) | "[Linus] This is a plan document for a SELECTOR FUNCTION. Twelve ACs, twenty tasks, seven PENDING decisions, seven milestones … Strip this down …" | applied |
| CMT-2  | change_request | Between `## Goal Description` subsection "Two Different Labels" and `## Acceptance Criteria` (input line 30) | "[Linus] If you need a seventy-word section in your goal description to disambiguate two concepts called 'calibration artifact' and 'runtime label cache' … pick names that make confusion impossible …" | applied |
| CMT-3  | change_request | After `AC-1` (input line 46) | "[Linus] AC-1's '--enable-double-sparsity together with --enable-hisparse fails at startup with not yet integrated' is admitting upfront that two sparsity features in the same engine cannot compose …" | applied (DEC-8 added) |
| CMT-4  | change_request | After `AC-2` (input line 60) | "[Linus] 'DS replaces the NSA indexer's token-selection role' — replaces HOW … 'bit-for-bit identical attention output' on FP8 is a fantasy." | applied |
| CMT-5  | change_request | After CMT-4 (input line 66) | "[Codex] `selected_indices` is not a contract. FlashMLA needs physical block-table entries with causal lengths, not a bag of ranked logical page ids …" | applied |
| CMT-6  | change_request | After `AC-4` (input line 87) | "[Linus] Validating `model_revision_sha + head_dim + tp_world_size + dtype + page_size + label_dim` is paranoid bookkeeping that will not catch the actual bug class …" | applied |
| CMT-7  | research_request → change_request + decision | After CMT-6 (input line 91) | "[Codex] `tp_world_size` in the artifact is not enough for tensor parallel correctness …" | researched (DEC-9 added) |
| CMT-8  | change_request | After `AC-6` (input line 112) | "[Codex] Static output buffers do not make CUDA graph capture safe. Grid sizes, scratch buffers, page-table writes, valid page counts, dense sentinel branching, and radix-cache hit shapes also have to be replay-stable …" | applied |
| CMT-9  | change_request | After `AC-7` (input line 122) | "[Linus] 'DS off (native NSA)' vs 'DS on' is incoherent. Native NSA is not a flag …" | applied |
| CMT-10 | change_request | After `AC-11` (input line 159) | "[Linus] `selection_mode=TOPK / TOPP`, TOPK shipped, TOPP unit-test only — exactly the YAGNI flavor …" | applied |
| CMT-11 | change_request | After `Feasibility Hints` Conceptual Approach #5 (input line 209) | "[Codex] The page signature table is keyed by `max_pages`, but the plan never ties entries to the KV page allocator lifecycle …" | applied |
| CMT-12 | change_request | After CMT-11 (input line 213) | "[Codex] `[num_layers, max_pages, num_heads, label_dim]` has no memory budget …" | applied |
| CMT-13 | research_request → change_request + decision | After `Future-Work Notes` (input line 226) | "[Linus] The future-work plan says 'wrap DoubleSparsitySelector behind the HiSparse BaseSparseAlgorithm interface and register it in _ALGORITHM_REGISTRY['double_sparsity']' — i.e. the FINAL state is the architecture the user rejected at CMT-1 …" | researched (Future-Work section rewritten; DEC-8 captures the immediate coexistence decision) |
| CMT-14 | change_request | After Milestone 3 Phase B (input line 277) | "[Codex] 'Incrementally extend on decode for new pages' misses the hot page. During decode, the current KV page changes every token until it fills …" | applied |
| CMT-15 | change_request | After Milestone 6 Phase B (input line 296) | "[Linus] Hard-coding `is_deepseek_nsa(hf_config)` in the validator (M1-B, task10) is the special-case branch the project's own maxim — `eliminate-special-cases-by-redesigning-data-flow` — tells you to refactor away …" | applied (DEC-10 added) |
| CMT-16 | change_request | After task20 row (input line 337) | "[Codex] The FP8 path is underspecified (task 12). A `K_label` kernel reading FP8 cache bytes without the exact quantization scales used by `quant_k_cache` is computing scores in the wrong numeric space …" | applied |
| CMT-17 | change_request | After CMT-16 (input line 341) | "[Linus] PR #25304 commits `0b776ca05`, `1b5e52863`, `7fe8002a3` are cited ten times as 'cherry-pick targets'. They were on a FA3 + Llama path … tasks 1-20 form a near-linear critical path …" | applied |
| CMT-18 | change_request | After `### Convergence Status` (input line 371) | "[Linus] 'partially_converged' with seven PENDING decisions — including hardware (DEC-1), the radix-cache mechanism (DEC-2), and the quality thresholds (DEC-3) — means AC-7, AC-8, AC-9 are unmeasurable as written …" | applied (execution gate added to Milestone 5) |
| CMT-19 | change_request | After Implementation Notes "Symbol-name boundary" bullet (input line 430) | "[Linus] 'do NOT import from mem_cache/sparsity/ … Borrow patterns by *copying code*' is a code-duplication mandate dressed up as a discipline rule …" | applied |

## Answers

*No raw `question`-type comments were extracted. The embedded "why" / "how" probes inside several `[Linus]` blocks were treated as change-requests with concrete corrective edits.*

## Research Findings

### CMT-7 — TP rank correctness for page selection

**Original Comment:**
```
[Codex] `tp_world_size` in the artifact is not enough for tensor parallel correctness. If each TP rank computes page scores from its local shard, ranks can choose different page tables for the same request, which breaks backend metadata assumptions and makes output rank-dependent. Specify whether page selection is per-rank by design or globally synchronized; if it is global, define the reduction/all-gather path and test rank agreement.
```

**Research Scope:**
- `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` — the `Indexer` class (line 178) and its `forward_indexer` / `forward_cuda` paths.
- `python/sglang/srt/models/deepseek_v2.py` — `DeepseekV2AttentionMLA` (line 1309), `forward_core` (line 1704), how the indexer is invoked under TP.
- `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` and `deepseek_nsa.py` — for analogue TP behaviors in existing sparse algorithms.

**Findings:**
- DeepSeek-V3.2 MLA shards heads across the TP group, so each rank holds `H_local = H / TP` heads' worth of K cache. Channel selection is per-head, so each rank computes per-head partial scores.
- If each rank takes the local top-K independently, ranks will pick different page sets because they see different heads. FlashMLA expects all ranks to attend over the same page set; rank divergence either produces silently wrong attention or fails the kernel's metadata invariants.
- The only safe options are (a) all-reduce the per-page partial scores across the TP group and recompute top-K from the reduced scores (Claude default), or (b) compute scores on rank 0 and broadcast (cheaper communication but adds a serial dependency to the per-step DS path).

**Impact on Plan:**
- Added `DEC-9` with Claude default = per-layer per-step `SUM` all-reduce over partial head scores, with top-K computed from the reduced scores on every rank.
- Updated Feasibility Hint #5 to specify the all-reduce explicitly.
- Added a "rank agreement" test to `task16`.
- Updated Milestone 3 Phase B to land the all-reduce wiring alongside the page_signature_write kernel.

### CMT-13 — HiSparse + PD-disagg decoupling feasibility

**Original Comment:**
```
[Linus] The future-work plan says "wrap DoubleSparsitySelector behind the HiSparse BaseSparseAlgorithm interface and register it in _ALGORITHM_REGISTRY['double_sparsity']" — i.e. the FINAL state is the architecture the user rejected at CMT-1. You are proposing a two-step migration: build standalone now, integrate with HiSparse later. That is twice the work and twice the test surface. The right question is "why does HiSparse require PD-disaggregation?" and "can that be decoupled?" If yes, do the decoupling first and build DS as a HiSparse algorithm in one move. If no, say DS will NEVER be a HiSparse algorithm and delete this bullet.
```

**Research Scope:**
- `python/sglang/srt/managers/hisparse_coordinator.py` — `HiSparseCoordinator` class shape and its decode-side hooks (`set_decode_producer_stream`, `_build_hisparse_decode_batch`, `collect_ready_reqs`, `has_ongoing_staging`).
- `python/sglang/srt/managers/scheduler.py` — where HiSparse is constructed (`self.enable_hisparse`, line 387; `self.hisparse_coordinator`, line 388; assignment at 979–982).
- `python/sglang/srt/disaggregation/decode.py` — admission hooks that branch on `scheduler.enable_hisparse` (lines 382, 787, 819, 896, 910, 1083, 1249).
- `python/sglang/srt/arg_groups/hisparse_hook.py` — `validate_hisparse` model and dtype gating.

**Findings:**
- The HiSparse stack is deeply intertwined with the decode-side scheduler and PD-disagg decode-instance admission. Specifically: the coordinator owns a decode-producer stream, builds decode batches via a private `_build_hisparse_decode_batch`, and has admission hooks in `decode.py` that decide whether a request is direct-to-host vs decode-only-cached. None of this exists outside the PD-disagg decode path.
- "Decoupling" HiSparse from PD-disagg is therefore a real refactor with its own design (extract tiering from decode-side scheduling, decide single-instance admission semantics, settle whether radix cache works under HiSparse on a single instance, etc.). It is **not in scope** for this DS plan.
- The "wrap DS as a HiSparse algorithm later" bullet was therefore wishful — it implied HiSparse would be available standalone at the time DS wanted to layer in. It will not be, without a separate effort.

**Impact on Plan:**
- Rewrote the `Future-Work Notes` subsection to scope the HiSparse / PD-disagg decoupling as its own plan and to flag that the HiSparse adapter for DS is gated on that decoupling, not promised here.
- Added `DEC-8` to record the immediate composition decision (initial coexistence behavior between `--enable-double-sparsity` and `--enable-hisparse`).
- Updated AC-1's negative test to bind to DEC-8 rather than to a "not yet integrated" placeholder message.

## Plan Changes Applied

### CMT-1 — Add "Design at a glance" three-paragraph summary

**Original Comment:** see ledger row.

**Changes Made:**
- Added a `### Design at a glance` subsection at the top of `## Goal Description` containing three paragraphs: what DS does (replace NSA `Indexer.forward` selection), the two artifacts (channel mask file offline + page signature table online), and the single edit site (`DeepseekV2AttentionMLA.forward_core` line 1704).
- Did **not** delete the existing detail beneath; Codex's PARTIALLY_AGREE noted that the scope is bigger than a single selector, so the detail remains for reviewers who need it.

**Affected Sections:** `## Goal Description` (new subsection).

### CMT-2 — Rename "calibration artifact" / "runtime label cache"; delete disambiguation subsection

**Changes Made:**
- Renamed `calibration artifact` → `channel mask file` throughout (all section bodies, AC tests, hint paragraphs, task descriptions, DEC body text, milestone phases).
- Renamed `runtime label cache` → `page signature table` throughout (same scope).
- Renamed `calibration_artifact_path` → `channel_mask_path`; `calibration_artifact_valid` → `channel_mask_valid`; `_calibration_loaded` → `_channel_mask_loaded`.
- Renamed `K_label` write kernel → `page_signature_write` kernel.
- Renamed `runtime_label_cache.py` → `page_signature_table.py` and `calibration.py` → `channel_mask.py` in the package layout.
- Deleted the `### Two Different "Labels"` subsection from `## Goal Description`.

**Affected Sections:** every section that named these concepts, plus the package layout hint and the file/CLI metadata listings.

### CMT-3 — AC-1 mutual-exclusion guard tied to DEC-8

**Changes Made:**
- Rewrote AC-1's "DS + HiSparse not yet integrated" negative test to bind to DEC-8: either ship an honest "deliberately undesigned for v1" startup error or admit composition (whichever DEC-8 records).
- Added DEC-8 to `## Pending User Decisions` with explicit Claude position (ship the guard with the "deliberately undesigned" wording) and tradeoff summary.
- Removed the "downstream-integration roadmap" reference in the error message; the future-work narrative is rewritten under CMT-13's resolution.

**Affected Sections:** AC-1, Pending User Decisions (DEC-8 added).

### CMT-4 — Specify hook site; replace "bit-for-bit" FP8 test

**Changes Made:**
- Added an explicit "Hook site (one sentence, per CMT-4)" bullet to AC-2 naming `DeepseekV2AttentionMLA.forward_core` (line 1704 in `python/sglang/srt/models/deepseek_v2.py`) and the one-line config-gated branch.
- Added the `__init__` change (`self.double_sparsity_selector`) and confirmed no monkey-patching / no model fork / no new attention backend.
- Replaced the "single-decode-step golden test ... bit-for-bit identical" wording with a tolerance test: `max_abs_diff(out_fp8, out_ref) <= 5e-3` and `cosine_similarity(out_fp8, out_ref) >= 0.9999` averaged over 16 deterministic prompts.

**Affected Sections:** AC-2, Feasibility Hint #2 (hook-site repeats the spec for the implementer).

### CMT-5 — selected_indices contract is sequence-order

**Changes Made:**
- Added an explicit "Selector contract (per CMT-5)" bullet to AC-2: `selected_indices` are logical page IDs sorted ascending in **sequence order** with `-1` padding to `max_top_k`; `valid_lengths` is `[bs]` int32; the FlashMLA `block_table` is emitted in sequence order via the existing `req_to_token` lookup.
- Added a positive test to AC-2 asserting monotonic ascending order of non-padding entries.
- Updated Feasibility Hint #1 (`selector.py` description) to document the sequence-order return.
- Updated task7 description to require the placeholder return to also be sequence-order ascending so the FlashMLA adapter test passes from day one.

**Affected Sections:** AC-2, Feasibility Hint #1, task7.

### CMT-6 — AC-4 validator reworked

**Changes Made:**
- Added a "Validator fields kept / added / dropped" breakdown under AC-4 to be explicit about the design.
- Kept `dtype`, `head_dim`, `page_size`, `label_dim` (shape / dtype safety checks).
- Added `content_sha256` (SHA-256 over `channel_selection || channel_weights`, recomputed at load) and a startup NIAH-min sanity probe.
- Dropped `model_revision_sha` (passes for misaligned LoRA fine-tunes) and `tp_world_size` (derivable from runtime config; carrying it invites the DEC-9 rank-divergence bug).
- Added a negative test exercising content-hash mismatch and another exercising sanity-probe failure on a corrupted-channel-weights fixture.

**Affected Sections:** AC-4, Feasibility Hint #4 (channel mask file format), task8.

### CMT-8 — AC-6 strengthened for CUDA-graph capture

**Changes Made:**
- AC-6 added three new positive tests: scratch buffer preallocated to worst-case size, device-side branching only (no host-visible Python `if` on tensor values), and no allocations inside captured region.
- Added a negative test that intentionally calls `torch.empty(...)` inside the captured region and asserts capture fails — enforces the rule rather than just declaring it.
- Updated task13 to budget the preallocation + device-side branching + regression test.

**Affected Sections:** AC-6, task13.

### CMT-9 — Rename AC-7 benchmark columns

**Changes Made:**
- Renamed `double_sparsity_off (native NSA)` → `native_nsa` and `double_sparsity_on` → `double_sparsity` in the AC-7 positive test.
- Updated the report-publish negative test to match.
- Updated Milestone 5 Phase A to emit the renamed two-column report.
- Updated Path Boundaries Upper Bound and Lower Bound descriptions.

**Affected Sections:** AC-7, Milestone 5, Path Boundaries (upper + lower), task15.

### CMT-10 — Drop selection_mode (Twilight deferred)

**Changes Made:**
- Rewrote AC-11 to ship a single-mode top-K ABI: `retrieve_topk(queries, layer_id, req_pool_indices, sparse_mask) -> (selected_indices, valid_lengths)` and explicitly say no `selection_mode` parameter is added in initial scope.
- Updated AC-11 positive and negative tests accordingly (rejects `selection_mode` / `top_p` JSON fields).
- Dropped task18 entirely from the Task Breakdown; preserved the ID gap to keep cross-references stable.
- Updated task20 dependencies (was `task15, task16, task17, task18`; now `task15, task16, task17`).
- Updated Milestone 6 to remove the Twilight phase; it is now purely the ship-gate.
- Removed `selection_mode`, `top_p`, `min_top_k`, `max_top_k` from the `DoubleSparsityConfig` description in Feasibility Hint #1.
- Updated Path Boundaries to forbid `selection_mode` in initial scope under "Cannot use".

**Affected Sections:** AC-11, Task Breakdown (task18 removed, task20 deps updated), Milestone 6, Feasibility Hint #1, Path Boundaries.

### CMT-11 — Page signature lifecycle

**Changes Made:**
- Added an explicit "Allocation and ownership" sub-bullet under Feasibility Hint #5: page signature table is owned by the KV page allocator; assign / free / evict / retract callbacks write or invalidate entries; a `valid_mask[L, max_pages]` bool tensor backs invalidation.
- Updated task9 to wire the allocator lifecycle hooks and the `valid_mask`.
- Added the `valid_mask` invalidation behavior to the test list in task16.

**Affected Sections:** Feasibility Hint #5, task9, task16.

### CMT-12 — Memory budget

**Changes Made:**
- Added a "Memory budget" sub-bullet under Feasibility Hint #5 with worst-case math: V3.2 with 60 layers × 128 heads × `label_dim=16` × fp16 × 15,625 pages (1M context, page=64) ≈ 3.8 GB if unsharded; ≈ 480 MB / rank under TP=8 head-sharded.
- Locked the operating point: `dtype=fp16`, `label_dim=16`, TP-head-sharded allocation, KV-page-allocator-owned lifetime.

**Affected Sections:** Feasibility Hint #5, task9 (now specifies `dtype=fp16, label_dim=16`).

### CMT-14 — Hot page during decode

**Changes Made:**
- Added a "Hot page" sub-bullet under Feasibility Hint #5 with two rules: update the active page's signature every decode step; force the active page (and an optional small local window) into the selected set unconditionally.
- Added a positive test to AC-2 (hot-page test) that exercises the rule in a mid-page decode step.
- Updated task12 to land the hot-page rule alongside the page_signature_write kernel.

**Affected Sections:** Feasibility Hint #5, AC-2, task12.

### CMT-15 — Validator model-class check (DEC-10)

**Changes Made:**
- Added DEC-10 with Claude default: V3.2-specific via capability check (presence of `nsa.Indexer` on the attention layer), not a model-name string match like `is_deepseek_nsa`.
- Updated task10 description to reference DEC-10.
- Updated Milestone 1 Phase B description.

**Affected Sections:** Milestone 1 Phase B, task10, Pending User Decisions (DEC-10 added).

### CMT-16 — FP8 quant scale handling

**Changes Made:**
- Added an "FP8 scale-aware projection" sub-bullet under Feasibility Hint #5 specifying that the page_signature_write kernel reads the inline per-tile scales from `quant_k_cache`'s `nope_part` (bytes 512–528 per token) and dequantizes to BF16 before projecting through the channel mask.
- Documented the alternative (scale-aware projection directly in the kernel) as a future optimization.
- Updated task12 to require the FP8-dequant-then-project semantics and to reuse the existing `nsa/dequant_k_cache.py` reference.
- Added an FP8 dequant equivalence test to task16.

**Affected Sections:** Feasibility Hint #5, task12, task16.

### CMT-17 — Cherry-pick framing + linear theatre note

**Changes Made:**
- Replaced "cherry-pick from PR #25304" / "port from PR #25304 commits" wording with "reimplement using PR #25304 as reading material" throughout (Resume-vs-Restart rationale, Path Boundaries note, Feasibility Hints relevant references, task11 description).
- Added a `> Note (per CMT-17)` above the Task Breakdown table acknowledging the linear single-implementer scope; task IDs are anchors, not a parallel work-breakdown.
- Updated PR #25304 "Reference list" framing to "reading material" with explicit "rewrite the kernel for MLA + FP8" callout.

**Affected Sections:** Resume-vs-Restart, Path Boundaries Note, Feasibility Hint references, Task Breakdown preamble, task11.

### CMT-18 — Execution gate on PENDING DECs

**Changes Made:**
- Added a note to Milestone 0 Phase C: "Per CMT-18, DEC-1 (hardware) and DEC-3 (quality thresholds) must be resolved before Milestone 5 can publish a falsifiable SLO claim. DEC-8, DEC-9, DEC-10 should resolve before Milestone 1 begins (they affect server args, validator scope, and the all-reduce path)."
- Added an explicit gate annotation to Milestone 5's preamble: "Per CMT-18, this milestone cannot publish a passing SLO claim until DEC-1, DEC-2, and DEC-3 are resolved."
- Annotated DEC-1 and DEC-3 with "blocks Milestone 5 per CMT-18" in `## Pending User Decisions`.

**Affected Sections:** Milestone 0 Phase C, Milestone 5 preamble, DEC-1, DEC-3.

### CMT-19 — Replace "copy don't import" with neutral helper

**Changes Made:**
- Replaced the Implementation Notes "Symbol-name boundary" bullet with the new shared-helper rule: if a pattern from HiSparse code is useful, factor it into `python/sglang/srt/utils/sparse_helpers.py` (a new neutral module) and import from both packages. Direct imports from `mem_cache/sparsity/`, `managers/hisparse_coordinator.py`, or `arg_groups/hisparse_hook.py` into the `double_sparsity/` package are still forbidden.
- Updated AC-12 positive tests to allow `python/sglang/srt/utils/sparse_helpers.py` in the diff scope.
- Updated Path Boundaries "Can use" to mention shared helpers via the neutral helper module.

**Affected Sections:** Implementation Notes, AC-12, Path Boundaries.

### Cross-Reference Updates

- AC IDs: all 12 preserved (AC-1..AC-12).
- Task IDs: task18 removed; all other IDs preserved (task1..task17, task19, task20). task20 dependencies updated to `task15, task16, task17`.
- DEC IDs: DEC-1..DEC-7 preserved; DEC-8, DEC-9, DEC-10 added.
- Milestone IDs: M0..M6 preserved; M6 contents reduced (Twilight phase removed).
- Renamed identifiers cascaded consistently across AC text, hints, milestones, tasks, deliberation log, and DECs.
- Original draft appendix preserved verbatim.

## Remaining Decisions

Seven pre-existing PENDING decisions (DEC-1..DEC-7) carry over; three new ones (DEC-8, DEC-9, DEC-10) were added in this round.

### DEC-1: SLO definition + hardware (pre-existing; **blocks Milestone 5 per CMT-18**)

**Related Comments:** CMT-1 (mentions hardware bloat), CMT-18 (explicit blocker).

**Context:** "30 tokens/s with a P99 TTFT of < 22s" is ambiguous: per-request vs aggregate, P50 vs P99 for throughput, hardware undefined.

**Options:**
1. Per-request P50 ≥ 30 tok/s, P99 TTFT ≤ 22 s, H200 8-way TP (Claude default).
2. Aggregate throughput ≥ 30 tok/s (likely trivial and unlikely client intent).
3. Per-request on smaller hardware (H100 8-way TP) — kernel choices may need to change.

**Recommendation:** Option 1.

**Status:** PENDING.

### DEC-2: Radix cache reconciliation (pre-existing)

**Related Comments:** CMT-11 (page-signature lifecycle relies on radix-cache compatibility).

**Context:** DS no longer inherits HiSparse's `assert disable_radix_cache`. The DS validator gates radix cache on the M3-B page-stability fixture.

**Options:**
1. Default-on; gate on the M3-B fixture passing (Claude default).
2. Default-off; require users to opt in.

**Recommendation:** Option 1.

**Status:** PENDING.

### DEC-3: Quality threshold deltas vs native_nsa (pre-existing; **blocks Milestone 5 per CMT-18**)

**Related Comments:** CMT-18.

**Context:** AC-9 needs concrete thresholds.

**Options:**
1. NIAH within 5 pp of native_nsa; MMLU within 1.0 pp (Claude default).
2. Tighter (NIAH 3 pp / MMLU 0.5 pp).
3. Looser (NIAH 7 pp / MMLU 2 pp).

**Recommendation:** Option 1.

**Status:** PENDING.

### DEC-4: Calibration ownership and artifact distribution (pre-existing)

**Related Comments:** CMT-6 (`content_sha256` instead of repository-pinned model_revision).

**Context:** Should the DeepSeek-V3.2 FP8 channel mask file be committed to the repo, hosted externally, or both?

**Options:**
1. Script in repo; file external to repo; tiny NSA fixture for CI (Claude default).
2. Script + file both committed (wheel bloat, license issues).
3. Script external too (deployment burden).

**Recommendation:** Option 1.

**Status:** PENDING.

### DEC-5: Semantic relationship of DS to DeepSeek-V3.2 NSA (pre-existing)

**Related Comments:** CMT-4 (hook site), CMT-15 (validator scope).

**Context:** DS can replace, augment, or stack atop V3.2's NSA selector.

**Options:**
1. Replace the NSA `Indexer.forward()` selection role with DS (Claude default; Codex agrees).
2. Stack DS after NSA (likely quality regression).
3. Augment NSA's indexer internals.

**Recommendation:** Option 1.

**Status:** PENDING.

### DEC-6: Scope of deferred-requirements coverage (pre-existing)

**Related Comments:** CMT-13 (HiSparse decoupling), CMT-15 (V3.2-vs-generic scope).

**Context:** GLM-5, 128K ISL, FP4 weights, HiSparse integration, PD-Disagg integration, HiCache integration — which constrain the initial design?

**Options:**
1. All deferred; selector ABI + channel-mask schema shaped to admit them; task 6 produces a one-page schema-compatibility memo before the loader merges (Claude default).
2. Roll HiSparse integration back into initial scope (rejected by CMT-1).
3. Roll GLM-5 into initial scope (significant kernel-shape implications).

**Recommendation:** Option 1.

**Status:** PENDING.

### DEC-7: "Extensions as a general knob for the sglang engine" interpretation (pre-existing)

**Related Comments:** none directly; inherited.

**Context:** The draft mentions "Extensions as a general knob" as a downstream requirement.

**Options:**
1. Expose DS runtime knobs through `--double-sparsity-config` JSON; no plugin system (Claude default).
2. Introduce a generic plugin / extension system (separate design effort).

**Recommendation:** Option 1.

**Status:** PENDING.

### DEC-8: HiSparse + DS coexistence (new, from CMT-3 + CMT-13)

**Related Comments:** CMT-3 (the "not yet integrated" framing is the worst-of-both-worlds choice), CMT-13 (the future-work HiSparse adapter is a U-turn against CMT-1).

**Context:** For the initial standalone deliverable, should `--enable-double-sparsity` and `--enable-hisparse` be allowed together?

**Options:**
1. Ship a startup guard rejecting concurrent enablement with an honest "deliberately undesigned for v1" error message (Claude default). The future-work HiSparse adapter is gated on a separate HiSparse / PD-disagg decoupling effort and is not promised here.
2. Allow concurrent enablement and define composition now (significant scope creep; requires solving the rank-divergence + memory-tiering + page-table-merging design upfront).
3. Allow concurrent enablement but silently let HiSparse win — rejected as user-hostile.

**Recommendation:** Option 1.

**Status:** PENDING.

### DEC-9: TP rank synchronization of page selection (new, from CMT-7)

**Related Comments:** CMT-7 (Codex flagged that per-rank selection breaks backend invariants).

**Context:** Per-rank top-K (each rank picks its own pages from its head shard, may disagree) vs globally synchronized top-K (per-layer per-step all-reduce of partial head scores).

**Options:**
1. Globally synchronized via per-layer per-step `SUM` all-reduce of partial head scores; top-K computed from reduced scores on every rank (Claude default).
2. Per-rank selection — drops the all-reduce but breaks the kernel's same-page-table-across-ranks invariant. Silent wrong attention is the failure mode.
3. Rank-0 select + broadcast — cheaper communication but adds a serial dependency.

**Recommendation:** Option 1. The all-reduce is small (`label_dim × max_pages × num_heads_local` fp16) and bandwidth-cheap relative to attention itself.

**Status:** PENDING.

### DEC-10: V3.2-scope vs MLA-capability validator (new, from CMT-15)

**Related Comments:** CMT-15 (the `is_deepseek_nsa` name-string check is a special-case branch the project's own maxim forbids).

**Context:** Should the DS validator (M1-B / task10) check for `is_deepseek_nsa(hf_config)` (V3.2-specific by model name), check for the capability presence of `nsa.Indexer` on the attention layer (generic-but-V3.2-shaped), or refuse to gate by model entirely?

**Options:**
1. V3.2-specific via capability presence — check whether the attention layer has a `nsa.Indexer` field (Claude default). This is honest about current scope (V3.2 today) and generalizes naturally to any model that exposes the same `Indexer` interface (e.g. GLM-5 if it lands).
2. Pure model-name match (`is_deepseek_nsa`) — simpler to write but bakes a string check into the validator; fails the eliminate-special-cases-by-redesigning-data-flow maxim.
3. No gating; refuse to validate model class and let the attention path raise at first call — fragile, bad UX.

**Recommendation:** Option 1.

**Status:** PENDING.

## Refinement Metadata

- **Input Plan:** `/sgl-workspace/sglang/development/refined_plan.md`
- **Output Plan:** `/sgl-workspace/sglang/development/refined_plan_v2.md`
- **QA Document:** `/sgl-workspace/sglang/.humanize/plan_qa/refined_plan-qa.md`
- **Total Comments Processed:** 19
  - Questions: 0
  - Change Requests: 19 (dominant classification across the set; 2 contain embedded research sub-items handled as `researched`)
  - Research Requests: 0 (none classified primarily as research, but CMT-7 and CMT-13 received research treatment as part of their change-request resolution)
- **Plan Sections Modified:** `## Goal Description` (Design at a glance + Standalone subsection rewritten + Two Different Labels removed + Resume-vs-Restart rewritten), `## Acceptance Criteria` (AC-1 negative test; AC-2 hook site + contract + tolerance + hot page; AC-3 unchanged; AC-4 validator reworked + content hash + sanity probe; AC-5 file rename; AC-6 capture safety strengthened; AC-7 column rename; AC-8 metric rename; AC-9 baseline rename; AC-10 metric namespace + field rename; AC-11 simplified; AC-12 helper path allowance), `## Path Boundaries` (Upper / Lower / Allowed Choices updated; deterministic-design note rewritten), `## Feasibility Hints and Suggestions` (Conceptual Approach #1, #2, #4, #5, #6, #7, #8 updated; Future-Work Notes rewritten; Relevant References extended with `deepseek_v2.py:1309/1704/2573`, `nsa_indexer.py:178`, `quant_k_cache.py` scale layout; PR #25304 reframed as reading material), `## Dependencies and Sequence` (Milestones reworded; M0 Phase C DEC dependency note added; M5 execution-gate annotation added; M6 reduced), `## Task Breakdown` (note above table; task18 removed; task20 deps updated; task descriptions tightened), `## Claude-Codex Deliberation` (Agreements extended; Resolved Disagreements added 19 entries for this round; Convergence Status rewritten), `## Pending User Decisions` (DEC-8, DEC-9, DEC-10 added; DEC-1 and DEC-3 annotated as Milestone 5 blockers), `## Implementation Notes` (Code Style cascaded with renamed identifiers; Shared-helper rule replaces Symbol-name boundary).
- **Convergence Status:** `partially_converged` (10 PENDING decisions; DEC-1 and DEC-3 are blocking for Milestone 5 per CMT-18).
- **Refinement Date:** 2026-05-19
- **Mode:** `discussion`
