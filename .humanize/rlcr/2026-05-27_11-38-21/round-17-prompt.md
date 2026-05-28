Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop4/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop4/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-17-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 16 Code Review

Mainline Progress Verdict: ADVANCED

Round 16 made real AC-6 progress: the illegal CUDA-capture host sync from `seq_lens.max().item()` is fixed when `max_seq_len` is supplied through `DSGraphState` / `capture_decode_step`, and the new CUDA graph replay fixture passes on H200. However, the round contract required a zero-allocation graph-safe selector path. That part is not implemented, so `task-ac6-cuda-graph` is still incomplete.

## Mainline Gaps

1. **AC-6 still allocates inside the selector path that `capture_decode_step` captures.**

   Evidence:
   - The CUDA graph body still calls `selector.retrieve_topk(...)` and then copies returned tensors into `state.selected_indices` / `state.valid_lengths` at `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py:203-217`. It does not call `retrieve_topk_graph_safe`.
   - `DSGraphState` only gained `max_seq_len`; it still has no logical-score/top-k scratch buffers required by the Round 16 contract (`cuda_graph.py:49-57`).
   - `retrieve_topk_graph_safe` is not actually graph-safe: it calls `retrieve_topk_via_labels(...)`, receives newly allocated `indices` / `valid`, then copies them into caller buffers (`selection_kernel.py:502-546`).
   - The underlying path still allocates intermediate tensors via `torch.gather` / multiply (`selection_kernel.py:198-200`), logical-position and physical-slot tensors (`selection_kernel.py:383-394`), score/mask tensors (`selection_kernel.py:397-407`), and top-k/sort/output tensors (`selection_kernel.py:301-334`).

   H200 verification after warmup:

   ```text
   selector retrieve allocation detected: selector-retrieve-max-seq-len: new CUDA allocation detected inside the captured region (47 new allocations)
   graph_safe allocation detected: retrieve-topk-graph-safe: new CUDA allocation detected inside the captured region (42 new allocations)
   ```

   Consequence: the Round 15 allocation gap remains. Graph replay being allocation-free is not enough; replay reuses allocations captured into the CUDA graph private pool. The plan/contract requires no dynamic allocation in the captured selector region itself.

   Required implementation plan:
   - Extend `DSGraphState` with the fixed-shape scratch required by the logical selector: `scratch_scores [max_bs, max_seq_len]`, `scratch_q_proj [max_bs, H_local, label_dim]`, top-k values/indices, sorted indices, final validity masks, valid lengths scratch, and any two-stage partial buffers needed by the chosen top-k implementation.
   - Add `num_local_heads` and `label_dim` to `allocate_graph_state` so those scratch tensors are sized at allocation time, not inferred during capture.
   - Replace `retrieve_topk_graph_safe` with a real into-buffer implementation. It must fill the scratch score buffer directly from `queries`, `req_pool_indices`, `req_to_token`, `seq_lens`, `written`, token signatures, channel selections/weights, and `per_request_valid`; it must not call `retrieve_topk_via_labels`.
   - Update `capture_decode_step` so the CUDA graph body calls only this into-buffer API. The CPU/eager fallback may keep using `selector.retrieve_topk`.
   - Add a CUDA regression that warms the graph-safe selector, then wraps the graph-safe selector call itself in `assert_no_alloc_in_region`. This test should fail on the current code with the allocation counts above and pass only after the scratch-backed path exists.
   - Keep the existing 100-step graph replay equality test, but do not use replay-only allocation checks as proof of capture-region allocation safety.

2. **The new `retrieve_topk_graph_safe` API cannot safely replace the production selector call as written.**

   Evidence:
   - `capture_decode_step` passes `sparse_mask` into `selector.retrieve_topk`, where it becomes `per_request_valid` (`selector.py:236-252`).
   - `retrieve_topk_graph_safe` has no `per_request_valid` / `sparse_mask` parameter and calls `retrieve_topk_via_labels` without one (`selection_kernel.py:502-540`).

   Consequence: if the next patch simply swaps `capture_decode_step` over to this helper, the M2 ownership/range mask is silently dropped from the captured selector path. That risks regressing AC-3 and violates the AC-6 dependency note that graph capture must include the ownership mask.

   Required implementation plan:
   - Add `per_request_valid: Optional[torch.Tensor]` to the graph-safe selector API.
   - Apply that mask inside the scratch-backed scoring/finalization path before top-k, exactly matching `retrieve_topk_via_labels` semantics.
   - Add a CUDA graph-safe test with a mask-off high-scoring logical position; the graph-safe API must not return that masked position.

## Blocking Side Issues

None separate from the mainline AC-6 gap. The allocation failure is the current mainline task itself.

## Queued Side Issues

1. AC-8 observability still reports token selections through page-named fields and computes sparsity against page counts in `_publish_ds_request_summary`.
2. Cleanup: stale DS bind/runtime comments still mention `req_to_token_pool.size` as max-token authority.
3. Cleanup: token-label lifetime docs still describe overwrite-before-read rather than invalidate-before-selection.

## Goal Alignment Summary

```text
ACs: 9/15 addressed (6 met, 3 partial: AC-1, AC-4, AC-6) | Forgotten items: 0 | Unjustified deferrals: 0
```

Status by AC:
- Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.
- Partial: AC-1, AC-4, AC-6.
- Not met: AC-1b, AC-8, AC-9, AC-10, AC-11, AC-12.

No explicit deferred tracker item is accepted. Hardware-gated work remains active pending work, not a deferral. `task-ac6-cuda-graph` also remains active because the coding contract is not complete.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 16 Review`.
- Added a Round 16 Review plan-evolution row reopening `task-ac6-cuda-graph`.
- Changed `task-ac6-cuda-graph` in Active Tasks from `complete` back to `pending`.

## Validation Run

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py::TestCUDAGraphCapture -q
9 passed

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
195 passed
```

Additional H200 allocation probes:
- `selector.retrieve_topk(..., max_seq_len=4)` still triggers `assert_no_alloc_in_region` with 47 new CUDA allocations.
- `retrieve_topk_graph_safe(...)` still triggers `assert_no_alloc_in_region` with 42 new CUDA allocations.

NOT COMPLETE
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-17-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-17-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
