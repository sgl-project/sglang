# Code Review - Round 8

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-8-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 8 Summary

## Work Completed

### AC-7 Fix 1 — Production bypass reads `ForwardContext` (`deepseek_v2.py`)

Replaced the dead `forward_batch.attn_backend` read with a `has_forward_context()` + `get_attn_backend()` guard that mirrors `handle_attention_dsa`:

```python
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend as _TboAttnBackend
from sglang.srt.model_executor.forward_context import (
    get_attn_backend as _get_attn_backend,
    has_forward_context as _has_forward_context,
)
if _has_forward_context():
    _fc_backend = _get_attn_backend()
    if isinstance(_fc_backend, _TboAttnBackend):
        _fc_backend = _fc_backend.primary
    if hasattr(_fc_backend, "use_mha") and _fc_backend.use_mha:
        return None
```

`ForwardBatch` has no `attn_backend` dataclass field — the old code was reading from a field that production never sets. The `has_forward_context()` guard preserves backward compatibility for unit tests that do not publish a `ForwardContext`.

### AC-7 Fix 2 — Label write in MHA_ONE_SHOT path (`forward_mha.py`)

Added after all pool-write branches in `_set_mla_kv_buffer`:

```python
if getattr(self, "use_double_sparsity", False):
    _ds_backend = _resolve_attn_backend(forward_batch)
    if hasattr(_ds_backend, "_write_token_labels"):
        _ds_backend._write_token_labels(
            self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1)
        )
```

`dsa_backend.forward_extend` guards `_write_token_labels` behind `save_kv_cache=True`, but `forward_normal_prepare` calls `_set_mla_kv_buffer` then passes `save_kv_cache=False` to the attention kernel — so labels were NEVER written for short dense prefills. This hook closes that gap. `_resolve_attn_backend` already handles `TboAttnBackend` unwrap.

### TestAC7MHABypass — updated to use real `ForwardContext` (6 tests)

**`test_bypass_fires_via_forward_context_use_mha_true`**
Runs under `forward_context(ForwardContext(attn_backend=mock_backend_use_mha_true))`.
`forward_batch` has NO `attn_backend` attribute — the production case.
Asserts: `result=None`, `retrieve_topk.assert_not_called()`.

**`test_no_bypass_when_forward_context_use_mha_false`**
`ForwardContext.use_mha=False` → `retrieve_topk` called, result non-None.

**`test_no_bypass_without_forward_context`**
No `ForwardContext` published → `has_forward_context()=False` → no bypass → `retrieve_topk` called.
Preserves backward compatibility for legacy unit tests.

**`test_mha_bypass_does_not_affect_nsa_path`**
`use_double_sparsity=False` → DS bypass block not entered → NSA indexer called.

**`test_mha_label_write_fires_in_set_mla_kv_buffer`**
Runs under `forward_context(ForwardContext(attn_backend=mock_backend_with_spy))`.
Calls `_set_mla_kv_buffer` directly with `T=3, kv_lora_rank=4` tensors.
Asserts: spy called once, `k.shape == [3, 1, 4]` (= `kv_a.unsqueeze(1)`).
Test FAILS if the `_write_token_labels` call is removed from `_set_mla_kv_buffer`.

**`test_no_label_write_when_not_double_sparsity`**
`use_double_sparsity=False` → `_write_token_labels` not called (negative).

## Files Changed

- `python/sglang/srt/models/deepseek_v2.py` — bypass reads `ForwardContext`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` — `_write_token_labels` call in `_set_mla_kv_buffer`
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — `TestAC7MHABypass` updated (6 tests vs 4)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
178 passed, 0 failed
```

Commit: `8e2a18f03`

## Remaining Items

- `task-ac1-hwtest`: H200 hardware population test.
- `task-ac4-calibrate` / `task-ac4-hwrun`: Method 1 calibration + mask generation.
- `task-ac5-tp`: TP=2 multiprocess all-reduce test.
- `task-ac6-cuda-graph` / `task-ac6-hwrun`: Graph capture + H200 replay.
- `task-ac1b-probe`, `task-ac8-*`, `task-ac12-quality`: Hardware/analyze gates.
- `task-ac9-baseline`, `task-ac10-radix`, `task-ac11-compare`: Stretch.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Both fixes follow patterns already in the codebase: `has_forward_context()` guard (mirrors `handle_attention_dsa`), `_resolve_attn_backend` for TBO unwrap (already in `forward_mha.py`), and conditional `_write_token_labels` hook (same pattern as the three existing call sites in `dsa_backend.py`). No new generalizable lesson.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-7-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-7-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-6-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-6-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-5-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-5-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-8-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
