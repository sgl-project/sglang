# Code Review - Round 13

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-13-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 13 Summary

## Objective

Fix the two AC-4 calibration coding blockers from Codex Round 12 review:
(1) Q RoPE width derivation broken on real V3.2 config shape;
(2) mask metadata dtype conflicts with FP8 serving validator.

## Changes Made

### 1. `python/sglang/srt/layers/attention/double_sparsity/calibrate.py`

**Blocker 1 — Q RoPE width derivation:**

Old code derived `qk_rope_head_dim = head_dim - qk_nope_head_dim` where `head_dim`
was itself derived from `hidden_size // num_heads` when the config lacks a `head_dim`
field. For V3.2 (`hidden_size=7168`, `num_heads=128`, `qk_nope=128`):
- `head_dim = 7168 // 128 = 56`
- `qk_rope_head_dim = 56 - 128 = -72`
- `full_mla_q_width = None` (negative guard), every `q_b_proj` hook skipped
- `_accumulate_method1` waited for Q indefinitely => "hooks did not fire" RuntimeError

New code reads `config.qk_rope_head_dim` directly when present. Only falls back to
`head_dim - qk_nope_head_dim` for configs that lack the explicit field. Raises a
clear RuntimeError with the derived values if the fallback would be non-positive.

**Blocker 2 — mask metadata dtype vs serving dtype:**

Added `mask_dtype = getattr(args, "kv_cache_dtype", None) or args.dtype`.
`save_channel_mask(dtype=mask_dtype, ...)` uses this value. `--dtype` remains
the model loading forward dtype; `--kv-cache-dtype` (optional, default None =>
falls back to `--dtype`) controls the mask metadata dtype.

**Parser update:** added `--kv-cache-dtype` optional arg; updated `--dtype` help.

**Module header:** production recipe now shows `--dtype bfloat16 --kv-cache-dtype fp8_e4m3`.

### 2. `docs/advanced_features/double_sparsity_calibration.md`

- Inputs table: updated `--dtype` description; added `--kv-cache-dtype` row.
- Recommended invocation: added `--kv-cache-dtype fp8_e4m3`.
- Explanation of the bf16 model-load vs fp8_e4m3 mask-metadata distinction.

### 3. `test/registered/unit/layers/attention/test_double_sparsity_unit.py`

**`test_dsv32_real_config_shape_q_hook_fires`** (new):
- Fake config: `qk_nope_head_dim=8`, `qk_rope_head_dim=4`, `v_head_dim=4`,
  `hidden_size=32`, `num_attention_heads=4` — no `head_dim` attribute.
- `hidden_size // num_heads = 8 != qk_nope + qk_rope = 12` — proves direct read.
- Asserts importance[0] shape is `(4, 8)`, is finite, and equals
  `mean(abs(Q_nope * K_nope))` ground truth.

## Test Results

```
188 passed, 0 failed (PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q)
```

Target was >= 188 (187 + 1 new). Target met exactly.

## Commit

`104fedcab` — [AC-4] Fix V3.2 qk_rope_head_dim derivation; add --kv-cache-dtype for mask dtype

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-mla-config-rope-dim-derivation
Notes: DeepSeek-V3.2 config has qk_rope_head_dim as an explicit field but no head_dim. Deriving qk_rope_head_dim from hidden_size//num_heads gives 56-128=-72, which silently skips all Q hooks. Fix reads config.qk_rope_head_dim directly first; only falls back to head_dim-qk_nope when the field is absent.

## AC Status After Round 13

- **AC-4** (`task-ac4-calibrate`): coding-complete — all blockers resolved across
  Rounds 10-13. `task-ac4-hwrun` unblocked for H200 generation with
  `--dtype bfloat16 --kv-cache-dtype fp8_e4m3`.
- All other ACs: unchanged from Round 12.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-12-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-12-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-11-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-11-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-10-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-10-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-13-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
