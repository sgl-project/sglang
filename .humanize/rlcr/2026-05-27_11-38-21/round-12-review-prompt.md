# Code Review - Round 12

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-12-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 12 Summary

## Objective

Fix three AC-4 calibration coding gaps identified by Codex Round 11 review so
`task-ac4-calibrate` is coding-complete and `task-ac4-hwrun` can proceed safely.

## Changes Made

### 1. `python/sglang/srt/layers/attention/double_sparsity/calibrate.py`

**Gap 1 — 3-D hook output handling in `_extract_mla_nope_prefix`:**
- Old code used `T = tensor.shape[0]` and then `tensor.reshape(T, num_heads, ...)`,
  which fails when real HF projection hooks emit `[batch, seq, width]` (3-D).
- Fixed by flattening all leading dimensions first:
  `flat = tensor.reshape(-1, tensor.shape[-1])`, then
  `flat.reshape(-1, num_heads, nope_dim + suffix_dim)[..., :nope_dim].contiguous()`
- Handles `[T, W]`, `[B, T, W]`, and any N-D shape where `shape[-1] == H*(nope+suffix)`.

**Gap 2 — Exact Pile-val token-block recipe:**
- Old code returned raw text strings truncated to `block_size` characters — not
  tokenized fixed-size blocks, and not concatenated across document boundaries.
- Added `_build_pile_val_token_blocks(tokenizer, num_blocks, block_size, seed)`:
  - Loads `mit-han-lab/pile-val-backup`, shuffles with `seed`
  - Tokenizes each doc with `add_special_tokens=False`
  - Concatenates token IDs across document boundaries
  - Splits into exactly `num_blocks` tensors of shape `[1, block_size]`
  - Raises `RuntimeError` if total tokens < `num_blocks * block_size`
- Added `use_pile_val` and `pile_val_seed` parameters to `_collect_channel_importance`
- When `use_pile_val=True`, calls `_build_pile_val_token_blocks` after tokenizer loads
  and feeds `model(input_ids=block.to(device))` for each block
- `calibrate()` production path (no `--dataset`, no `--allow-synthetic`) now sets
  `use_pile_val=True` — implements the exact Pile-val-256×512 recipe per AC-4

**Gap 3 — Module header updated:**
- Replaced K-only NIAH description with Method 1 Q+K noPE + Pile-val seed=42 recipe
- Production invocation example updated to match new parameters

### 2. `docs/advanced_features/double_sparsity_calibration.md`

- Inputs table: added `--block-size` (default 512), `--seed` (default 42),
  `--allow-synthetic` entries
- Recommended invocation: updated to include `--block-size 512 --seed 42`
- Dataset description: explains concatenated fixed-size block construction
- "What gets calibrated": describes Method 1 Q+K noPE with reshape-before-slice;
  removes stale K-only L2 + NIAH wording
- CI fixture: clarified `--allow-synthetic` opt-in and scope

### 3. `test/registered/unit/layers/attention/test_double_sparsity_unit.py`

Two new tests added to `TestCalibrateMethod1` (before `test_512d_channel_index_rejected`):

**`test_3d_hook_output_handled`**:
- Builds a fake model where `kv_b_proj` and `q_b_proj` return `[1, T, W]` (3-D
  with batch=1), using the same random values as `_make_fake_model` (seed=42)
- Runs `_collect_channel_importance` via `_run_calibration`
- Asserts `importance_3d[0]` is finite and `allclose` to `importance_2d[0]`
- Proves the flatten-before-reshape fix handles batch dimensions correctly

**`test_pile_val_blocks_concatenate_across_docs`**:
- Patches `datasets.load_dataset` with 3 short docs yielding 200 tokens each
  (IDs 0..199, 200..399, 400..599) — 600 total, need 512 for one block
- Calls `_build_pile_val_token_blocks(fake_tok, num_blocks=1, block_size=512, seed=42)`
- Asserts result is `[1, 512]` and that token at index 511 comes from doc 2
  (IDs 400..599 range), proving cross-document concatenation vs. truncation

## Test Results

```
187 passed, 0 failed (PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q)
```

Target was >= 187 (185 + 2 new). Target met exactly.

## Commit

`287a58231` — [AC-4] Fix 3-D hook outputs, implement Pile-val token-block recipe, update calibration doc

## BitLesson Delta

Action: none
Reason: `BL-20260527-reshape-before-slice-mla` covers the reshape-before-slice pattern
that this round extends to multi-dimensional inputs. No new distinct failure mode.

## AC Status After Round 12

- **AC-4** (`task-ac4-calibrate`): coding-complete — all three Round-11-review gaps
  closed; 187 tests pass. Ready for `task-ac4-hwrun` (H200 hardware run, analyze/Codex).
- All other ACs: unchanged from Round 11.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-11-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-11-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-10-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-10-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-9-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-9-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-12-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
