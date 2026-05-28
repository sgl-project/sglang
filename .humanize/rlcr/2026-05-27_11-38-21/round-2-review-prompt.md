# Code Review - Round 2

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-2-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 2 Summary

## Work Completed

### AC-0 slot-count authority fix

`TokenLabelTable.max_tokens` was incorrectly bound from `req_to_token_pool.size`
(request-row count, ~256). Any `out_cache_loc` value beyond that count would produce
an out-of-bounds write. Fixed by deriving `max_tokens = kv_pool.size + kv_pool.page_size`
where `kv_pool` is `token_to_kv_pool` — the object whose first tensor axis exactly
spans the physical KV slot address space addressed by `out_cache_loc`.

The physical pool is published to `server_args._ds_token_to_kv_pool` in the
`model_runner.py` DS bind block so it is available when `finalize_double_sparsity_bind()`
runs. A `RuntimeError` guard ensures the bind fails loudly if called before pool init.

Three regression tests confirm: (1) non-contiguous large-slot writes succeed with
correct sizing, (2) the old small sizing raises `IndexError`, (3) logical→physical
round-trip via `req_to_token` is bit-exact.

### AC-1 hook wiring

`_write_token_labels(layer, cache_loc, k)` added to `NativeSparseAttnBackend`:
- Projects `k_latent [T, kv_lora_rank]` through `layer.kv_b_proj` (no_grad) to
  get full `kv_proj_out [T, H_local*(nope+v)]`, slices the noPE K columns and
  reshapes to `[T, H_local, 128]`.
- Calls `token_label_write(signatures, written, layer_id, cache_loc, k_nope,
  channel_selection_layer)`.
- Guard: no-op if `enable_double_sparsity=False`, table/channel_selection is None,
  or `layer.kv_b_proj` is absent.

The hook is wired at all three `set_mla_kv_buffer` sites:
- Site 1: `forward_extend` native path
- Site 2: `forward_decode` native path
- Site 3: TRT-LLM extend path

`channel_selection` and `qk_nope_head_dim` are published from
`_bind_double_sparsity_runtime_data` via `server_args` attributes so the backend
can capture them at construction time without traversing the model hierarchy.

Two unit tests confirm: hook populates `signatures` and sets `written`, and is a
no-op when DS is disabled.

## Files Changed

- `python/sglang/srt/model_executor/model_runner.py` — publish `_ds_token_to_kv_pool`
- `python/sglang/srt/models/deepseek_v2.py` — use `kv_pool.size + kv_pool.page_size` for `max_tokens`; publish `_ds_channel_selection`, `_ds_qk_nope_head_dim`, `_double_sparsity_token_label_table`
- `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py` — fix docstrings
- `python/sglang/srt/layers/attention/dsa_backend.py` — capture table+channel selection in `__init__`; add `_write_token_labels`; wire at 3 sites
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — add `TestAC0RealSlotRegression` (3 tests) and `TestAC1HookUnit` (2 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
155 passed, 0 failed
```

Commit: `65cbd28e0`

## Remaining Items

Next round candidates (from queued side issues and active task table):
- AC-2: boot-time GB/rank log; stale-slot lifetime test
- AC-3: per-request range mask (M2); multi-request boundary test
- AC-7: short-seq MHA bypass
- AC-4: V3.2 channel mask calibration
- AC-6 graph helper: `capture_decode_step` still calls `retrieve_topk` without logical→physical conversion (blocks AC-6)
- AC-8 observability: `_publish_ds_request_summary` reports page-named fields (blocks AC-8 quality metrics)

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Implementation corrections (slot-count sizing and hook wiring). No new generalizable pattern emerged.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-1-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-1-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-0-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-0-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-2-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
