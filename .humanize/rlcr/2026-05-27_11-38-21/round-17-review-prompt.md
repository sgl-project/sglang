# Code Review - Round 17

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-17-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 17 Summary

## Work Completed

### task-ac6-cuda-graph — Truly Allocation-Free Graph-Safe Path

Codex Round 16 review left two gaps open:

1. `retrieve_topk_graph_safe` internally wrapped `retrieve_topk_via_labels` →
   ~42 new CUDA allocations even with pre-allocated output buffers.
2. `retrieve_topk_graph_safe` had no `per_request_valid` parameter, silently
   dropping the M2 ownership mask from the graph-safe path.

Both gaps are now closed.

**New Triton kernel `_logical_score_kernel`** (`selection_kernel.py`):
Grid `(bs, ceil(max_seq_len / TOKEN_BLOCK))`. Per (batch, logical_pos):
look up `physical_slot = req_to_token[pool_idx, pos]`, gather signatures at
that slot, max-over-heads dot product with projected query (using `ch_sel`
as gather index into queries), apply `written` and `seq_lens` masks. Writes
directly into pre-allocated `out [bs, max_seq_len]` fp32 — zero Python-level
allocations.

**Completely rewrote `retrieve_topk_graph_safe`** (`selection_kernel.py`):
On CUDA + Triton + scratch available, runs an allocation-free pipeline:

1. `_logical_score_kernel` fills `scratch_scores` (Triton).
2. (optional) all-reduce in place on `scratch_scores`.
3. (optional) apply `per_request_valid` via `scratch_pv_mask.copy_(...)` +
   in-place `logical_not` + `masked_fill_(-inf)`.
4. `topk(scratch_scores, k, sorted=False, largest=True,
   out=(scratch_topk_values, scratch_topk_indices))`.
5. `isneginf(scratch_topk_values, out=scratch_invalid_mask)`;
   `masked_fill_(invalid, max_seq_len)` sentinels invalid entries.
6. `topk(scratch_topk_indices, k, sorted=True, largest=False,
   out=(scratch_sorted_vals, scratch_throwaway_idx))` — ascending sort via
   smallest-first topk.
7. `out_indices.copy_(scratch_sorted_vals)`; `ge(...)` + `masked_fill_(-1)`
   converts sentinels to `-1`.
8. `searchsorted(scratch_sorted_vals, scratch_boundary, right=False,
   out=scratch_valid_i64)`; `out_lengths.copy_()`.

On CPU or with scratch missing: falls back to legacy `retrieve_topk_via_labels`
(allocating, fine for unit tests).

**Extended `DSGraphState`** (`cuda_graph.py`): added fields
`scratch_scores`, `scratch_topk_values`, `scratch_topk_indices`,
`scratch_invalid_mask`, `scratch_sorted_vals`, `scratch_boundary`,
`scratch_valid_i64`, `scratch_pv_mask`, **`scratch_throwaway_idx`**.

`scratch_throwaway_idx` was added after debugging the per_request_valid test
failure (see below). PyTorch `torch.topk(input=A, ...,
out=(values=B, indices=A))` corrupts the read when output indices alias
input. Symptom: input `[3, 1]` produced output values `[0, 1]` instead of
`[1, 3]`. Fix: route throwaway gather indices into a dedicated scratch.

**Extended `allocate_graph_state`**: now takes `num_local_heads: int = 0`,
`label_dim: int = 0` (accepted for API parity; scratch sizing is driven by
`max_seq_len`). When `max_seq_len > 0`, the eight scratch tensors above are
allocated; otherwise None (graph-safe fast path skipped).

**Updated `capture_decode_step` CUDA path**: when the selector is bound AND
`state.scratch_scores is not None`, the captured region calls
`retrieve_topk_graph_safe` with all scratch + `per_request_valid=sparse_mask`.
Otherwise falls back to `selector.retrieve_topk`. Eager+capture both use the
same `_call_into_state()` closure to guarantee warmup and capture paths are
identical.

### Tests

- **`test_retrieve_topk_graph_safe_zero_allocs_after_warmup`** (CUDA-only):
  Warm up with one direct call; wrap second call in `assert_no_alloc_in_region`.
  Asserts 0 new allocations AND correctness (`idx=[2,3]`, `valid=2`).
- **`test_retrieve_topk_graph_safe_per_request_valid_masks_position`**
  (CUDA-only): Masks logical position 2 (the would-be top score 9.0).
  Asserts position 2 is NOT in the output and remaining picks come from
  `[0, 1, 3]` → expected `sorted([1, 3])`.
- Existing CUDA tests updated to pass `num_local_heads=1, label_dim=1` so
  scratch is allocated. The 100-step bit-equal replay and the zero-alloc
  replay tests still pass.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`:
  - Added Triton kernel `_logical_score_kernel`.
  - Added wrapper `_logical_score_triton`.
  - Completely rewrote `retrieve_topk_graph_safe` (allocation-free fast
    path on CUDA + scratch; legacy fallback on CPU).
  - Added optional `per_request_valid`, `scratch_*`, `scratch_pv_mask`,
    `scratch_throwaway_idx` parameters.
- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`:
  - `DSGraphState`: added 9 scratch fields.
  - `allocate_graph_state`: added `num_local_heads`, `label_dim` params;
    allocates the scratch when `max_seq_len > 0`.
  - `capture_decode_step`: CUDA path routes through `retrieve_topk_graph_safe`
    when selector is bound + scratch available; passes
    `per_request_valid=sparse_mask`.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  - Added two CUDA-only tests above.
  - Updated existing CUDA tests with `num_local_heads=1, label_dim=1`.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
197 passed, 0 failed (was 195 before this round)
```

The two new CUDA tests pass on the available H200 (Triton path exercised).
Existing CUDA-only 100-step replay + zero-alloc replay tests still pass with
the new graph-safe path engaged.

Holistic loop audit (rounds 10–16 Codex reviews): no unresolved
HIGH-SEVERITY findings outside Round 17 scope. Hardware-gated items
(`task-ac4-hwrun`, `task-ac1-hwtest`, `task-ac8-*`, `task-ac12-*`) remain
queued unchanged.

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real
  V3.2 H200 cluster. The coding path is now complete (both Round 16 gaps
  closed; allocation-free Triton + topk pipeline verified locally).
- `task-ac4-hwrun`: hardware gate — H200 channel-mask calibration.
- Next coding task: `task-ac1-hwtest` or `task-ac10-radix` (per goal tracker).

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-torch-topk-aliasing-corrupts-input
Notes: PyTorch `torch.topk(input=A, ..., out=(values=B, indices=A))` is NOT
safe — output indices aliasing input corrupts the read. Symptom in this
loop: input `[3, 1]` produced sorted values `[0, 1]` (instead of `[1, 3]`).
Fix is to route throwaway output indices into a separate scratch tensor.
Worth recording because the contract explicitly suggested aliasing as an
allocation-saving trick, and the failure mode (looks like scoring returned
zeros) is misleading on the surface.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
cb6004a36 docs: restore CLUSTER.md on dev/double-sparsity-standalone
20bf84515 [Sparsity] Loop-4: plan + refined_plan_v1 + QA ledger
ae04e4c3d [Sparsity] Loop-4 Round-0: AC-0 token-level label rotation
7be9fd7a8 [DS Loop-4 Round-1] Fix 5 AC-0/AC-13 gaps: export, domain, bind-timing, Q-noPE, test renames
65cbd28e0 [Sparsity] Loop-4 Round-2: fix AC-0 slot-count authority + wire AC-1 write hooks
9ac489af3 [Sparsity] Loop-4 Round-3: fix kv_b_proj K-noPE extraction + FP8 latent-k preservation
ef16fa441 [Sparsity] Loop-4 Round-4: add AC-1 call-site tests for forward_extend/decode/TRT-LLM
a20cb5445 [Sparsity] Loop-5: AC-2 lifetime tests + AC-3 range-mask tests
178427b75 [Sparsity] Loop-6: AC-2 stale-slot invalidation + AC-3 logical-domain test
a81b6532e [DS] AC-2 live wiring + AC-7 MHA bypass for _select_topk_indices
8e2a18f03 [DS] Fix AC-7 MHA bypass: use ForwardContext + wire label write in _set_mla_kv_buffer
e2552fbe9 [Sparsity] Loop-9: AC-7 first-decode-after-prefill integration proof
8379cfdba [Sparsity] Loop-10: AC-4 Method 1 Q+K calibration + fixture tests
d2f82c93a [Sparsity] Loop-11: AC-4 fix MLA extraction + Pile-val calibration path
287a58231 [AC-4] Fix 3-D hook outputs, implement Pile-val token-block recipe, update calibration doc
104fedcab [AC-4] Fix V3.2 qk_rope_head_dim derivation; add --kv-cache-dtype for mask dtype
6cf32a884 [AC-5] TP=2 multiprocess all-reduce harness for Double Sparsity
cd7c071f3 [AC-6] Thread req_to_token through capture_decode_step; add AC-6 unit tests
0ce54a98d [AC-6] Fix CUDA graph capture-safety: remove host sync + add graph-safe API
18b22b5e3 [AC-6] Truly allocation-free CUDA graph-safe selector
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-16-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-16-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-15-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-15-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-14-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-14-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Implementation Review

- Your task is to conduct a deep critical review, focusing on finding implementation issues and identifying gaps between "plan-design" and actual implementation.
- Relevant top-level guidance documents, phased implementation plans, and other important documentation and implementation references are located under @docs.
- If Claude planned to defer any tasks to future phases in its summary, DO NOT follow its lead. Instead, you should force Claude to complete ALL tasks as planned.
  - Such deferred tasks are considered incomplete work and should be flagged in your review comments, requiring Claude to address them.
  - If Claude planned to defer any tasks, please explore the codebase in-depth and draft a detailed implementation plan. This plan should be included in your review comments for Claude to follow.
  - Your review should be meticulous and skeptical. Look for any discrepancies, missing features, incomplete implementations.
- If Claude does not plan to defer any tasks, but honestly admits that some tasks are still pending (not yet completed), you should also include those pending tasks in your review.
  - Your review should elaborate on those unfinished tasks, explore the codebase, and draft an implementation plan.
  - A good engineering implementation plan should be **singular, directive, and definitive**, rather than discussing multiple possible implementation options.
  - The implementation plan should be **unambiguous**, internally consistent, and coherent from beginning to end, so that **Claude can execute the work accurately and without error**.

## Part 2: Goal Alignment Check (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md and verify:

1. **Acceptance Criteria Progress**: For each AC, is progress being made? Are any ACs being ignored?
2. **Forgotten Items**: Are there tasks from the original plan that are not tracked in Active/Completed/Deferred?
3. **Deferred Items**: Are deferrals justified? Do they block any ACs?
4. **Plan Evolution**: If Claude modified the plan, is the justification valid?

Include a brief Goal Alignment Summary in your review:
```
ACs: X/Y addressed | Forgotten items: N | Unjustified deferrals: N
```

## Part 3: Required Finding Classification

You MUST classify your findings into these lanes:
- **Mainline Gaps**: plan-derived work or AC progress that is missing, incomplete, or regressing
- **Blocking Side Issues**: bugs or implementation issues that block the current mainline objective from succeeding safely
- **Queued Side Issues**: valid non-blocking follow-up issues that should be documented but must NOT take over the next round

Also include a one-line verdict:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
```

This verdict line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

If Claude mostly worked on queued side issues and failed to advance the mainline, say so explicitly.

## Part 4: ## Goal Tracker Update Requests (YOUR RESPONSIBILITY)

Claude should normally keep the **mutable section** of `goal-tracker.md` up to date directly. If Claude's summary contains a "Goal Tracker Update Request" section, or if you detect tracker drift during review, YOU must:

1. **Evaluate the tracker state**: Is the mutable section still aligned with the Ultimate Goal and current AC progress?
2. **If correction is needed**: Update @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md yourself with the requested changes:
   - Move tasks between Active/Completed/Deferred sections as appropriate
   - Add entries to "Plan Evolution Log" with round number and justification
   - Add new issues to "Blocking Side Issues" or "Queued Side Issues" as appropriate
   - **NEVER modify the IMMUTABLE SECTION** (Ultimate Goal and Acceptance Criteria)
3. **If you reject a requested tracker change**: Include in your review why it was rejected

Common update requests you should handle:
- Task completion: Move from "Active Tasks" to "Completed and Verified"
- New blocking issues: Add to "Blocking Side Issues"
- New queued issues: Add to "Queued Side Issues"
- Plan changes: Add to "Plan Evolution Log" with your assessment
- Deferrals: Only allow with strong justification; add to "Explicitly Deferred"

## Part 5: Output Requirements

- In short, your review comments can include: problems/findings/blockers; claims that don't match reality; implementation plans for deferred work (to be implemented now); implementation plans for unfinished work; goal alignment issues.
- Your output should be structured so Claude can tell which items are mainline gaps, blocking side issues, and queued side issues.
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-17-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
