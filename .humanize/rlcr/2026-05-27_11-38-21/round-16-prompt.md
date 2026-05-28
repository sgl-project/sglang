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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-16-contract.md

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
# Round 15 Code Review

Mainline Progress Verdict: ADVANCED

Round 15 fixed one real AC-6 prerequisite: `capture_decode_step` now threads `req_to_token` into all three `selector.retrieve_topk` calls. That is useful progress, but `task-ac6-cuda-graph` is not complete. On this H200, the actual CUDA graph path still fails during capture, and the new tests do not exercise that path.

## Mainline Gaps

1. **AC-6 CUDA graph capture still fails with a bound logical-domain selector.**

   Evidence:
   - `capture_decode_step` enters `torch.cuda.graph(graph)` and calls `selector.retrieve_topk` at `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py:179-187`.
   - The logical selector path then calls `_compute_logical_token_scores`, which computes `max_seq_len = int(seq_lens.max().item())` at `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:360-361`.
   - During CUDA graph capture, that `.item()` host sync is illegal. A minimal H200 repro using the same bound-selector fixture shape as the new unit test fails with:

   ```text
   torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
   ...
   selection_kernel.py:361, in _compute_logical_token_scores
       max_seq_len = int(seq_lens.max().item()) if bs > 0 else 0
   ```

   Consequence: the AC-6 positive criteria are not met. `capture_decode_step` does not complete on CUDA, so 100 graph replay steps and eager-vs-graph comparison cannot be claimed.

   Required fix:
   - Remove every host read of CUDA tensor values from the captured selector path. In particular, do not derive `max_seq_len` with `seq_lens.max().item()`.
   - Use a static graph-time sequence width from a CPU/static source, preferably `sparse_mask.shape[-1]` or a `max_seq_len` stored in `DSGraphState`, and mask dynamic request lengths with tensor operations only.
   - Add a CUDA regression test that binds a real `TokenLabelTable`/`ChannelMask`, calls `capture_decode_step` on CUDA with `req_to_token`, successfully captures, replays 100 times, and compares graph output to an eager direct selector call.

2. **The captured selector path still allocates tensors inside the region instead of using `DSGraphState` scratch.**

   Evidence:
   - `cuda_graph.py:180-190` calls `selector.retrieve_topk`, receives newly created `out_idx`/`out_len`, then copies them into `state.selected_indices` and `state.valid_lengths`. The preallocated state buffers are only a destination after allocation has already happened.
   - The selector path allocates multiple intermediates: `torch.arange` and gathers in `_compute_logical_token_scores` (`selection_kernel.py:375-387`), new score tensors at `selection_kernel.py:391-400`, and new top-k/sort/output tensors in `select_topk_sequence_order` (`selection_kernel.py:301-334`). The placeholder path also allocates via `torch.full`, `torch.arange`, and `torch.where` at `selector.py:272-288`.
   - After one warmup call, wrapping `sel.retrieve_topk(...)` in `assert_no_alloc_in_region` on the H200 reports:

   ```text
   RuntimeError selector-retrieve: new CUDA allocation detected inside the captured region (48 new allocations)
   ```

   Consequence: the Round 15 alloc-detector tests do not prove the DS decode capture region is allocation-free. They only prove the context manager can catch a standalone `torch.empty`, while the actual selector still violates the preallocation contract.

   Required fix:
   - Add a graph-safe selector API that writes into caller-owned buffers, for example `retrieve_topk_into(..., out_indices, out_lengths, scratch)`.
   - Extend `DSGraphState` with all fixed-shape scratch needed by logical scoring and selection: score buffer `[max_bs, max_seq_len]`, top-k values/indices, sorted indices, masks or reusable position buffers, and any query/physical-slot scratch the implementation keeps in torch rather than Triton.
   - Update `capture_decode_step` to call only the graph-safe API inside `torch.cuda.graph`.
   - Add a CUDA test that wraps the actual capture/replay setup with `assert_no_alloc_in_region`, not just a standalone allocation.

## Blocking Side Issues

None separate from the mainline AC-6 gap. The capture-safety failure is the mainline task itself, not a side issue.

## Queued Side Issues

1. AC-8 observability still reports token selections through page-named fields and computes sparsity against page counts in `_publish_ds_request_summary`.
2. Cleanup: stale DS bind/runtime comments still mention `req_to_token_pool.size` as max-token authority.
3. Cleanup: token-label lifetime docs still describe overwrite-before-read rather than invalidate-before-selection.

The old queued issue about missing `req_to_token` in `capture_decode_step` is fixed and should stay removed from the tracker.

## Goal Alignment Summary

```text
ACs: 9/15 addressed (6 met, 3 partial: AC-1, AC-4, AC-6) | Forgotten items: 0 | Unjustified deferrals: 0
```

Status by AC:
- Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.
- Partial: AC-1, AC-4, AC-6.
- Not met: AC-1b, AC-8, AC-9, AC-10, AC-11, AC-12.

No explicit deferred tracker item is justified or accepted. Hardware-gated items remain active pending work, not deferrals.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 15 Review`.
- Added a Round 15 Review plan-evolution row reopening `task-ac6-cuda-graph`.
- Changed `task-ac6-cuda-graph` in Active Tasks from `completed` back to `pending`.
- Removed the stale queued `req_to_token` omission issue because that part is fixed.
- Removed AC-6 from Completed and Verified because actual CUDA graph capture is not verified and currently fails.

## Validation Run

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py::TestCUDAGraphCapture -q
7 passed

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
193 passed
```

Additional H200 sanity checks:
- Bound logical-domain `capture_decode_step` on CUDA fails during `torch.cuda.graph` at `seq_lens.max().item()`.
- Warmed `selector.retrieve_topk(...)` still triggers `assert_no_alloc_in_region` with 48 new CUDA allocations.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-16-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-16-summary.md

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
