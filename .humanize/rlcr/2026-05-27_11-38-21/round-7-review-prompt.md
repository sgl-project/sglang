# Code Review - Round 7

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-7-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 7 Summary

## Work Completed

### AC-2 live-wiring regression — `TestAC2LiveWiring`

**`test_production_hook_invalidates_before_retrieve_topk`**
Constructs a `DeepseekV2AttentionMLA` fixture (via `object.__new__`) with `use_double_sparsity=True`, `IS_PLACEHOLDER=False`. Attaches a `TokenLabelTable` with `written[0, 7] = True` (stale). Sets `forward_batch.out_cache_loc = torch.tensor([7])`. Monkey-patches `selector.retrieve_topk` with a `side_effect` spy that captures `written[0, 7]` at call time. Calls `_select_topk_indices` normally. Asserts:
1. The spy was called exactly once (retrieve_topk fired).
2. `written[0, 7]` was `False` when the spy ran — i.e., the invalidation hook (lines 2087-2093 of `deepseek_v2.py`) fired before `retrieve_topk`. Test FAILS if those lines are deleted.

**`test_after_hook_written_is_restored_by_label_write`**
Exercises the full invalidate → label-write lifecycle: invalidates slot 7, asserts `written=False`, writes a new label, asserts `written=True`. Confirms the lifecycle in the call order that production uses.

### AC-7 bypass — `_select_topk_indices` in `deepseek_v2.py`

Added a guard immediately after the DS-path imports in `_select_topk_indices` (lines 2071-2079):

```python
# AC-7: skip sparse selection during short-seq MHA-mode prefill.
_attn_backend = getattr(forward_batch, "attn_backend", None)
if getattr(_attn_backend, "use_mha", False):
    return None
```

When `forward_batch.attn_backend.use_mha` is `True` (short extend below `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` on SM90/SM100), selection is skipped entirely. The label write in `dsa_backend._write_token_labels` (line 1510) fires unconditionally — it precedes the `if self.use_mha:` branch at line 1513 — and is unaffected by this change.

### AC-7 tests — `TestAC7MHABypass` (4 tests)

**`test_mha_bypass_returns_none_and_skips_retrieve_topk`**
`attn_backend.use_mha=True` → result is `None`, `retrieve_topk` not called.

**`test_no_bypass_when_use_mha_false`**
`attn_backend.use_mha=False` → `retrieve_topk` called once, result is non-None tensor.

**`test_bypass_when_no_attn_backend`**
`attn_backend=None` → `use_mha` defaults to `False` via `getattr` fallback → `retrieve_topk` called (no bypass).

**`test_mha_bypass_does_not_affect_nsa_path`**
`use_double_sparsity=False` with `attn_backend.use_mha=True` → NSA `indexer` called, MHA flag is irrelevant.

## Files Changed

- `python/sglang/srt/models/deepseek_v2.py` — AC-7 bypass (10 lines added in `_select_topk_indices`)
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — `TestAC2LiveWiring` (2 tests) + `TestAC7MHABypass` (4 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
176 passed, 0 failed
```

Commit: `a81b6532e`

## Remaining Items

- `task-ac1-hwtest`: H200 hardware population test — pending hardware access.
- `task-ac4-calibrate`: Method 1 Q+K joint hooks in calibrate.py.
- `task-ac4-hwrun`: Hardware run on H200 to generate dsv32-fp8-channel-mask.safetensors.
- `task-ac5-tp`: TP=2 multiprocess all-reduce test.
- `task-ac6-cuda-graph`: Decode-path graph capture.
- `task-ac6-hwrun`: Hardware full-graph capture at conc=64.
- `task-ac1b-probe`: Chunked-prefill probe.
- `task-ac8-server`, `task-ac8-quality`, `task-ac12-quality`: Server smoke + quality gates.
- `task-ac9-baseline`, `task-ac10-radix`, `task-ac11-compare`: Stretch comparators.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: AC-2 live wiring is a standard spy/side-effect pattern (capture assertion inside mock side_effect). AC-7 bypass is a standard guard-before-main-logic pattern. Neither introduces a project-specific lesson beyond what is already captured in the existing test fixtures.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-6-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-6-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-5-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-5-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-4-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-4-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-7-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
