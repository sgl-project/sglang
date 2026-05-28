# Code Review - Round 15

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-15-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 15 Summary

## Work Completed

### task-ac6-cuda-graph — CUDA Graph Decode Path Fix + Tests

**Root cause fixed**: `capture_decode_step` was missing `req_to_token` in all three
`selector.retrieve_topk` call sites. Without it, the captured graph would use the
physical-domain fallback path instead of logical-domain selection, producing wrong top-K
results when physical slots differ across TP ranks.

**Fix**: Added `req_to_token: Optional[torch.Tensor] = None` to `capture_decode_step`
and threaded it through to all three call sites:
1. Eager CPU path (`_eager_replay` closure)
2. CUDA warmup pass
3. CUDA graph capture region

**New unit tests (5)** in `TestCUDAGraphCapture`:

1. **`test_req_to_token_threads_to_logical_domain`**: Uses a bound selector with physical
   sigs `[9.0, 8.0, 1.0, 2.0]` and `req_to_token = [[2,3,0,1]]`. With this mapping,
   logical scores are `[1.0, 2.0, 9.0, 8.0]`, so top-2 logical positions are `[2, 3]`.
   Verifies the replay returns `[[2, 3]]` (not `[[0, 1]]` which would be the physical-domain answer).

2. **`test_eager_replay_output_matches_direct_call`**: Calls replay once and separately
   calls `selector.retrieve_topk` directly with the same arguments; asserts bit-equal output.

3. **`test_eager_replay_100_steps_stable`**: Calls replay 100 times and verifies all
   outputs are identical (placeholder-mode determinism).

4. **`test_alloc_detector_raises_on_cuda_alloc_in_region`**: On CUDA, allocating a tensor
   inside `assert_no_alloc_in_region` raises `RuntimeError`. On CPU, the context manager
   is a no-op (verified it doesn't raise).

5. **`test_alloc_detector_silent_when_prealloc_before_region`**: Preallocating a buffer
   before the region and doing writes-only inside does not raise.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`: added
  `req_to_token` parameter to `capture_decode_step`; threaded through all 3 call sites;
  updated docstring to explain the logical-domain requirement.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`: added
  `_make_bound_selector_with_known_sigs` helper and 5 new `TestCUDAGraphCapture` tests.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
193 passed, 0 failed (was 188 before this round)
```

Commit: `cd7c071f3` — [AC-6] Thread req_to_token through capture_decode_step; add AC-6 unit tests

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real V3.2 H200
  cluster. The coding path is complete.
- `task-ac4-hwrun`: hardware gate — H200 CUDA OOM on available machine.
- Next coding task: `task-ac1-hwtest` or `task-ac10-radix`. Hardware-gated tasks
  (`task-ac1b-probe`, `task-ac8-*`, `task-ac12-*`) need H200 cluster.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: No new lessons needed. The req_to_token threading omission was a straightforward
parameter gap, not a surprising failure mode warranting a BitLesson entry.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-14-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-14-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-13-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-13-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-12-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-12-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-15-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
