# Code Review - Round 16

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-16-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 16 Summary

## Work Completed

### task-ac6-cuda-graph — CUDA Capture-Safety Fix (two gaps from Round 15 Codex review)

**Gap 1 — Host sync in captured region (CRITICAL)**

`_compute_logical_token_scores` computed `max_seq_len = int(seq_lens.max().item())` inside the
path that `capture_decode_step` places inside `torch.cuda.graph(graph)`. The `.item()` is a
host sync that raises `CUDA error: operation not permitted when stream is capturing`.

Fix: Added `max_seq_len: int = 0` parameter to `_compute_logical_token_scores`. When nonzero,
the `.item()` call is skipped. Threaded the parameter up through:
- `retrieve_topk_via_labels` (`max_seq_len: int = 0`)
- `DoubleSparsitySelector.retrieve_topk` (`max_seq_len: int = 0`)
- `capture_decode_step` (`max_seq_len: int = 0`)

In `capture_decode_step`, a static `_max_seq_len` is resolved BEFORE the capture region:
```
priority: state.max_seq_len > max_seq_len parameter > one-time seq_lens.max().item()
```
That one-time `.item()` is safe because it happens before `torch.cuda.graph()` is entered.
All three call sites inside the function (eager warmup, CUDA warmup, CUDA capture) receive
`max_seq_len=_max_seq_len`.

**Gap 2 — Graph-safe selector API + DSGraphState extension**

Added `retrieve_topk_graph_safe` to `selection_kernel.py`: same contract as
`retrieve_topk_via_labels` but accepts `max_seq_len: int` (required, no default) and writes
results directly into caller-owned `out_indices` / `out_lengths` buffers. This is the
preferred API for graph capture call sites.

Extended `DSGraphState` with `max_seq_len: int = 0` (the static sequence width). Extended
`allocate_graph_state` to accept `max_seq_len: int = 0` and store it in the state. Callers
now pass `max_seq_len` at allocation time; `capture_decode_step` reads `state.max_seq_len`
automatically without requiring callers to pass the parameter twice.

**Two new CUDA-only tests** (decorated `@unittest.skipUnless(torch.cuda.is_available(), ...)`):

1. **`test_cuda_graph_100_step_replay_matches_eager`**: Creates a bound selector on CUDA with
   known sigs `[9.0, 8.0, 1.0, 2.0]` and `req_to_token = [[2, 3, 0, 1]]`. Captures a CUDA
   graph with `max_seq_len=4`. Calls `sel.retrieve_topk` eagerly to get reference result `[[2,3]]`.
   Replays the graph 100 times and verifies every replay is bit-equal to the eager reference.

2. **`test_cuda_graph_replay_zero_allocations`**: Same CUDA fixture. Wraps `replay()` in
   `assert_no_alloc_in_region("cuda-graph-replay")`. Verifies no `RuntimeError` is raised
   (i.e., 0 new CUDA allocations during graph replay). Also verifies correctness: `lens[0]=2`,
   `idx[0,:2] = [2, 3]`.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`:
  - `_compute_logical_token_scores`: added `max_seq_len: int = 0`; skip `.item()` when provided
  - `retrieve_topk_via_labels`: added `max_seq_len: int = 0`; threads to `_compute_logical_token_scores`
  - Added `retrieve_topk_graph_safe` (writes into `out_indices` / `out_lengths` in-place)
- `python/sglang/srt/layers/attention/double_sparsity/selector.py`:
  - `retrieve_topk`: added `max_seq_len: int = 0`; passes to `retrieve_topk_via_labels`
- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`:
  - `DSGraphState`: added `max_seq_len: int = 0`
  - `allocate_graph_state`: added `max_seq_len: int = 0`; stored in state
  - `capture_decode_step`: added `max_seq_len: int = 0`; resolves static `_max_seq_len` before
    capture region; passes to selector in all three call sites
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  - Added `_make_bound_selector_cuda` helper (parametric device version of existing CPU helper)
  - Added `test_cuda_graph_100_step_replay_matches_eager` (CUDA-only)
  - Added `test_cuda_graph_replay_zero_allocations` (CUDA-only)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
195 passed, 0 failed (was 193 before this round)
```

Commit: `0ce54a98d` — [AC-6] Fix CUDA graph capture-safety: remove host sync + add graph-safe API

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real V3.2 H200 cluster.
  The coding path is now complete (both capture-safety gaps closed).
- `task-ac4-hwrun`: hardware gate — H200 CUDA OOM on available machine.
- Next coding task: `task-ac1-hwtest` or `task-ac10-radix`.
- Hardware-gated: `task-ac1b-probe`, `task-ac8-*`, `task-ac12-*`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The two AC-6 gaps (.item() host sync, graph-safe output buffers) are a known class
of CUDA graph capture requirements. No surprising failure mode warranting a new entry;
the fix is textbook and fully documented in the capture_decode_step docstring.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-15-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-15-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-14-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-14-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-13-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-13-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-16-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
