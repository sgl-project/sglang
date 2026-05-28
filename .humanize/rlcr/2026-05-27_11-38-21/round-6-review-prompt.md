# Code Review - Round 6

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-6-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 6 Summary

## Work Completed

### AC-2 stale-slot protection — production code

**`invalidate_token_label_slots` in `token_label_write.py`**
New function `invalidate_token_label_slots(written, layer_id, cache_loc)` sets `written[layer_id, cache_loc] = False` via `index_fill_`. This is capture-safe and O(1). Called by `_select_topk_indices` in `deepseek_v2.py` before `retrieve_topk` so that a newly-allocated physical KV slot cannot be selected based on a stale label left by a previously-evicted request.

**Wire in `deepseek_v2.py:_select_topk_indices._run()`**
Before `selector.retrieve_topk` is called, the `_run()` closure now retrieves `forward_batch.out_cache_loc` and `selector.token_label_table`, then calls `invalidate_token_label_slots`. Both guards are None-safe so the call is a no-op in unit tests that use placeholder selectors or SimpleNamespace forward batches.

**`validate_table_covers_kv_pool` in `token_label_table.py`**
New function `validate_table_covers_kv_pool(table, kv_pool_size, page_size)` raises `ValueError` if `table.max_tokens != kv_pool_size + page_size`. Called in the reuse branch of `_bind_double_sparsity_runtime_data` (the `else:` block that fires for layers 2+ when the table was already created by layer 0). Guards against a mis-sized pre-existing table reaching production.

### AC-2 regression tests in `TestAC2Lifetime`

**`test_invalidate_makes_stale_slot_unselectable`**
Writes a high-score stale label (1000.0) at physical slot 7, creates a new request with `req_to_token[0, 0] = 7` (logical pos 0 → slot 7). Runs `retrieve_topk_via_labels` in logical-domain mode WITHOUT invalidation → confirms stale slot IS selectable (shows the bug). Then calls `invalidate_token_label_slots` → runs again → confirms slot is NOT selectable.

**`test_after_invalidation_new_write_restores_selectability`**
Invalidates slot 7, then writes a new label. Confirms `written[0, 7]` is restored to True and slot is selectable again. Verifies the invalidation → write lifecycle.

**`test_validate_table_size_rejects_wrong_max_tokens`**
Creates a table with `max_tokens=100`. Calls `validate_table_covers_kv_pool(table, 36, 64)` → passes (36+64=100). Calls with `(64, 64)` → raises ValueError mentioning `max_tokens=100` and `128`.

### AC-3 production-path test in `TestAC3RangeMask`

**`test_logical_domain_req_to_token_isolates_per_request`**
Uses `req_to_token[0, 0..9] = [0..9]` and `req_to_token[1, 0..9] = [10..19]`. Sets signatures so physical slots 10..19 have score 1000 and 0..9 have score 0. Runs `retrieve_topk_via_labels` in logical-domain mode (with `req_pool_indices`, `req_to_token`, `seq_lens`). Converts logical output to physical via `logical_to_physical`. Asserts req-0 physical slots ⊆ [0,10) and req-1 physical slots ⊆ [10,20). This is the production path: logical-domain mode scores only each request's own `req_to_token` row, so foreign high-score slots are invisible to req-0.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/token_label_write.py` — add `invalidate_token_label_slots`
- `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` — add `validate_table_covers_kv_pool`
- `python/sglang/srt/models/deepseek_v2.py` — invalidation call in `_select_topk_indices._run()`; size guard in bind reuse branch
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — 4 new tests in `TestAC2Lifetime`; 1 new test in `TestAC3RangeMask`

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
170 passed, 0 failed
```

Commit: `178427b75`

## Remaining Items

- `task-ac1-hwtest`: H200 hardware population test — pending hardware access.
- `task-ac7-bypass` (AC-7): short-seq MHA bypass implementation + tests.
- `task-ac4-calibrate` (AC-4): Method 1 Q+K joint hooks in calibrate.py.
- `task-ac5-tp` (AC-5): TP=2 multiprocess all-reduce test.
- `task-ac6-cuda-graph` (AC-6): decode-path graph capture.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Stale-slot invalidation is a standard pre-selection invalidation pattern (mark invalid before selection, restore valid after write). No new generalizable project-specific lesson.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-5-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-5-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-4-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-4-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-3-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-3-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-6-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
