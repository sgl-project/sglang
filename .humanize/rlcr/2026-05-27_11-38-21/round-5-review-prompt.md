# Code Review - Round 5

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-5-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 5 Summary

## Work Completed

### AC-2 lifetime/stale-slot — `TestAC2Lifetime`

Added four tests that verify the token label table's lifetime and slot budget invariants:

**`test_boot_log_emits_gb_per_rank`**
Calls `allocate_token_label_table` inside `assertLogs` and verifies that the INFO-level message contains `token_label_table:`, `GB/rank`, and all four dimension fields (`L=`, `T=`, `H=`, `D=`). This confirms operators can read the sizing audit from boot logs.

**`test_slot_budget_covers_all_physical_kv_slots`**
Creates a table with `max_tokens = kv_pool_size + page_size` (= 192) and calls `token_label_write` targeting the last valid physical slot (`max_tokens - 1 = 191`). Asserts `written[0, 191]` is True and `table.max_tokens == 192`. Verifies the table covers the full `out_cache_loc` address space without OOB.

**`test_stale_slot_overwrite_replaces_prior_label`**
Writes label A (all-1.0) to slot 7, confirms it reads 1.0, then writes label B (all-2.0) to the same slot, and asserts the slot now reads 2.0 with no trace of 1.0. Verifies `token_label_write` performs an unconditional overwrite rather than accumulating.

**`test_label_visible_immediately_after_write`**
Writes sentinel value 42.0 to slot 3 and immediately reads it back. Asserts all label values equal 42.0 and `written[0, 3]` is True. Confirms no phantom/stale state is visible before the next overwrite.

### AC-3 range mask — `TestAC3RangeMask`

Added two tests verifying per-request token range ownership:

**`test_multi_request_picks_within_own_range_with_mask`** (positive)
Constructs bs=2, max_tokens=20 with disjoint ownership: req-0 owns slots 0..9, req-1 owns slots 10..19. Sets signatures so req-1's slots would outscore req-0's for any query without a mask. Passes `per_request_valid` (bool [2, 20]) to `retrieve_topk_via_labels`. Asserts all indices for req-0 are < 10 and all for req-1 are >= 10. Both requests produce at least one valid pick.

**`test_without_mask_cross_request_contamination_occurs`** (negative — mask is load-bearing)
Same setup but passes `per_request_valid=None`. Slots 10..19 have score 1000, slots 0..9 have score 0. With all-ones queries, req-0's top picks come from the high-score region 10..19 (which belongs to req-1). Asserts at least one index in req-0's results is >= 10. This confirms that `per_request_valid` is the mechanism preventing cross-request contamination and cannot be removed.

## Files Changed

- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — add `TestAC2Lifetime` (4 tests) and `TestAC3RangeMask` (2 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
166 passed, 0 failed
```

## Remaining Items

- `task-ac1-hwtest`: hardware population test on H200 — pending hardware access.
- `task-ac7-bypass` (AC-7): short-seq MHA bypass; confirm save_kv_cache=True still fires.
- `task-ac4-calibrate` (AC-4): Method 1 Q+K joint hooks in calibrate.py.
- `task-ac5-tp` (AC-5): TP=2 multiprocess test.
- `task-ac6-cuda-graph` (AC-6): decode-path graph capture.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: AC-2 tests use standard Python assertLogs and direct tensor writes; AC-3 tests use the existing per_request_valid parameter of retrieve_topk_via_labels. No project-specific new pattern emerged.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-4-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-4-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-3-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-3-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-2-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-2-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-5-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
