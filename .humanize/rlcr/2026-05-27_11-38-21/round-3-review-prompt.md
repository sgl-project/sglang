# Code Review - Round 3

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-3-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 3 Summary

## Work Completed

### AC-1 hook correctness fixes

Two bugs in `_write_token_labels` caused incorrect label population. Both are now fixed.

**Bug 1 — reshape before slice (wrong K extraction)**

`kv_b_proj` output layout is per-head `[K_nope | V]`: head 0 K, head 0 V, head 1 K, head 1 V, etc. The Round-2 code sliced `kv_proj_out[:, :H_local * nope_dim]` from the flat tensor before reshaping. For H_local=16, nope_dim=128, v_head_dim=128, the flat first 2048 values cover exactly 8 full heads worth of output (8 × 256), so the reshape to `(16, 128)` produces: head 0 → K0 (correct), head 1 → V0 (wrong), head 2 → K1 (correct), head 3 → V1 (wrong), etc.

Fix in `_write_token_labels`:
```python
# Before (wrong — flat slice then reshape)
k_nope = kv_proj_out[:, : H_local * nope_dim].view(T, H_local, nope_dim)

# After (correct — reshape first, then K-noPE prefix per head)
head_width = nope_dim + layer.v_head_dim
k_nope = kv_proj_out.view(T, H_local, head_width)[..., :nope_dim].contiguous()
```

`layer.v_head_dim` is set on `RadixAttention.attn_mha` in `deepseek_v2.py:1610` and equals `self.v_head_dim` (128 for V3.2).

**Bug 2 — TRT-LLM FP8 path passed post-quantized k to hook**

`_forward_trtllm` unconditionally called `_write_token_labels(layer, cache_loc, k)` after `mla_quantize_and_rope_for_fp8` overwrote `k` with FP8 cache data. The hook projects `k` through `kv_b_proj`, which expects the 512-d latent float K — feeding FP8 bytes produces garbage labels.

Fix: save `k_for_labels = k` before the FP8 block and pass it to the hook:
```python
k_for_labels = k  # preserve latent K before FP8 quantization
if self.kv_cache_dtype == torch.float8_e4m3fn:
    q, k, k_rope = mla_quantize_and_rope_for_fp8(q, q_rope, k.squeeze(1), ...)
...
if save_kv_cache:
    self.token_to_kv_pool.set_mla_kv_buffer(layer, cache_loc, k, k_rope)
    self._write_token_labels(layer, cache_loc, k_for_labels)  # pre-quantized
```

For non-FP8 paths, `k_for_labels = k` and `k` is never reassigned, so the fix is a no-op on those paths.

### Sentinel regression test

Added `test_write_token_labels_extracts_k_nope_not_v_columns` to `TestAC1HookUnit`:
- `kv_b_proj` stub returns per-head layout `[K_nope | V_sentinel]` where V columns = 999.0
- Head 0 K = `[1,2,3,4]`, Head 1 K = `[5,6,7,8]`, all V = `[999,999,999,999]`
- Asserts: no 999.0 in signatures; head-0 = `[1,2,3,4]`; head-1 = `[5,6,7,8]`
- This test fails deterministically with the old flat-slice code

Also added `v_head_dim=nope_dim` to the existing `test_write_token_labels_populates_table` layer stub so it remains correct under the new `layer.v_head_dim` access.

## Files Changed

- `python/sglang/srt/layers/attention/dsa_backend.py` — reshape fix in `_write_token_labels`; `k_for_labels` save in `_forward_trtllm`
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — add `test_write_token_labels_extracts_k_nope_not_v_columns`; add `v_head_dim` to existing stub

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
156 passed, 0 failed
```

Commit: `9ac489af3`

## Remaining Items

`task-ac1-hwtest` (hardware forward test on H200) remains pending per the plan — AC-1 unit tests pass but Codex hardware verification is still pending. Next coding tasks by dependency order:

- AC-2: boot-time GB/rank log; stale-slot lifetime test
- AC-3: per-request range mask (M2); multi-request boundary test
- AC-7: short-seq MHA bypass
- AC-4: calibrate.py Method 1 Q+K hooks

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Both bugs were reshape ordering and tensor preservation errors; no new generalizable lesson beyond what standard MLA projection reshape patterns already imply.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-2-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-2-review-result.md
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-3-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
