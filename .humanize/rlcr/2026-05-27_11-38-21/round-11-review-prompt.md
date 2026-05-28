# Code Review - Round 11

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-11-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 11 Summary

## Work Completed

### AC-4 — Fix 1: MLA extraction reshape-before-slice

Added `_extract_mla_nope_prefix(tensor, num_heads, nope_dim, suffix_dim)` helper in
`calibrate.py`. MLA projections interleave per-head blocks — `kv_b_proj` layout is
`[K_nope_h0 | V_h0 | K_nope_h1 | V_h1 | ...]` — so flat-slicing the first
`H * nope_dim` columns before reshape selects V columns from later heads (same class
of bug fixed in `dsa_backend.py` Round 3).

Changed both hooks to use the helper:
- K hook: `_extract_mla_nope_prefix(t, num_heads, k_head_dim, v_head_dim)` — replaces
  `tensor[..., :prefix].reshape(-1, num_heads, k_head_dim)`.
- Q hook: `_extract_mla_nope_prefix(t, num_heads, k_head_dim, qk_rope_head_dim)` —
  replaces `tensor[..., :prefix].reshape(-1, num_heads, k_head_dim)`.
  `qk_rope_head_dim = head_dim - qk_nope_head_dim` (derived from config).

Also added `is_mla_q` flag to `_make_q_hook` so standard-attention Q (`q_proj` or
`wq`) still uses a direct reshape instead of the per-head splitting logic.

### AC-4 — Fix 2: Pile-val seed=42, 256×512 calibration dataset

Added `_pile_val_blocks(num_blocks, block_size, seed)` function that loads
`mit-han-lab/pile-val-backup` via `datasets`, shuffles with `seed=42`, and returns
`num_blocks` text examples.

Changed `calibrate()` default dataset path:
- If `args.dataset`: use custom corpus file (unchanged).
- Elif `args.allow_synthetic`: use NIAH synthetic prompts (unchanged; CI path).
- Else (production path): use Pile-val blocks.

Added `--block-size` (default 512) and `--seed` (default 42) to the parser.
Changed `--num-samples` default from 64 to 256.
Added `dataset_source`, `seed`, `block_size` to output metadata.
Threaded `block_size` into `_collect_channel_importance` for tokenizer truncation
(`max_length=block_size, truncation=True` when set).

### AC-4 — Fix 3: Sentinel regression tests

Added 2 tests to `TestCalibrateMethod1`:

1. `test_mla_k_extraction_ignores_v_columns`: 2-head MLA, `kv_b_proj` output with
   K_nope=1.0 and V=100.0 (poison). Asserts `importance.max() < 10.0`. Fails under
   old flat-slice (head 1 gets V0 → importance ≈ 100.0).

2. `test_mla_q_extraction_ignores_rope_columns`: 2-head MLA, `q_b_proj` output with
   Q_nope=1.0 and Q_rope=100.0 (poison). Asserts `importance.max() < 10.0`. Fails
   under old flat-slice (head 1 gets Q0_rope[0:4] → importance ≈ 100.0).

Also fixed `_make_fake_model`'s `expected_importance` to use reshape-then-slice (to
match the now-correct calibrate.py extraction), and added `block_size=512, seed=42`
to the `test_label_dim_exceeds_k_head_dim_raises` argparse.Namespace fixture.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/calibrate.py` — added
  `_extract_mla_nope_prefix` + `_pile_val_blocks`; fixed K and Q hook extraction;
  added `block_size` parameter; updated `calibrate()` dataset path and metadata;
  added `--block-size` and `--seed` CLI args
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — fixed
  `_make_fake_model` expected_importance; added 2 sentinel tests; fixed argparse
  Namespace fixture

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
185 passed, 0 failed in 11.50s
```

Commit: `d2f82c93a`

## Remaining Items

- `task-ac4-hwrun`: H200 hardware run to generate `dsv32-fp8-channel-mask.safetensors`
  (analyze tag → Codex; hardware not available here)
- `task-ac5-tp`: TP=2 multiprocess all-reduce test
- `task-ac6-cuda-graph`: `req_to_token` through `capture_decode_step`
- Hardware / analyze gates: AC-1, AC-1b, AC-8, AC-9–12
- Queued: update `docs/advanced_features/double_sparsity_calibration.md` before
  `task-ac4-hwrun`

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-reshape-before-slice-mla

Notes: The flat-slice-before-reshape pattern on MLA projection outputs is a
recurring class of bug in this codebase. Round 3 fixed it in `dsa_backend.py`;
Round 11 fixed the same class in `calibrate.py`. The pattern manifests silently
— the resulting tensor has correct shape but wrong values for heads > 0. A
BitLesson should be written so future rounds catch this immediately rather than
waiting for Codex review.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-10-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-10-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-9-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-9-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-8-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-8-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-11-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
