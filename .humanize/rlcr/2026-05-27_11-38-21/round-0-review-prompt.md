# Code Review - Round 0

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-0-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 0 Summary

## What Was Implemented

Round 0 completed the AC-0 architecture rotation: migrated the entire Double
Sparsity implementation from page-level `[L, P, H, D]` label storage to
token-level `[L, T, H, D]` label storage (AC-0), and verified all 150 unit
tests pass on the new APIs (AC-13).

### Core files created

- `token_label_table.py` — `TokenLabelTable` dataclass (`signatures [L_local,
  max_tokens, H_local, label_dim]`, `written bool[L_local, max_tokens]`);
  `allocate_token_label_table`; `estimate_hbm_bytes` helper.

- `token_label_write.py` — `token_label_write(signatures, written, layer_id,
  cache_loc, k_nope, channel_selection_layer)` Triton-backed write path.
  Takes projected 128-d K_nope per token slot (no page averaging).

### Core files modified

- `page_table_adapter.py` — rewritten to < 150 LOC; exposes only
  `DSAdapterError` (base exception, kept for downstream imports) and
  `logical_to_physical(selected_indices, req_pool_indices, req_to_token, out)
  -> int`. The function performs a single `req_to_token` gather and returns a
  scalar `error_count` for bad pool indices.

- `selection_kernel.py` — `select_topk_sequence_order(token_scores, max_top_k)`
  (2 args; removed `hot_pages` and `per_request_valid`). `retrieve_topk_via_labels`
  updated to token-level shapes; removed `hot_pages` kwarg (kept `per_request_valid`).

- `selector.py` — `retrieve_topk` now returns logical token positions (sequence-
  ascending, `-1` padded) for token-level label table.

- `config.py` — `top_k` default changed from 64 (pages) to 2048 (tokens);
  docstring updated.

- `validator.py` — reads `dsa_prefill_backend` / `dsa_decode_backend` instead
  of old `nsa_*` attribute names. Added `top_k == get_dsa_index_topk(hf_config)`
  boot assertion (env-override: `SGLANG_DS_ALLOW_TOPK_MISMATCH=1`).

- `__init__.py` — re-exports updated to `TokenLabelTable`, `token_label_write`,
  `retrieve_topk`; old `PageSignatureTable` / `page_signature_write` exports
  removed.

- `channel_mask.py` — docstring updated to reference `token_label_write`.

- `deepseek_v2.py` — `_bind_double_sparsity_runtime_data` derives `max_tokens`
  from `req_to_token_pool.size`; `DSGraphState.selected_indices` is now
  `int32[max_bs, max_top_k]`.

- `model_runner.py` — updated DS bind call to pass `req_to_token_pool`.

### Files deleted

- `page_signature_table.py` (185 LOC) — `PageSignatureTable` page-level class
- `page_signature_write.py` (498 LOC) — page-level Triton write kernel

### Test file migrated (150 tests)

`test/registered/unit/layers/attention/test_double_sparsity_unit.py` — all 59
old API references removed and replaced:

| Old | New |
|-----|-----|
| `PageSignatureTable` / `page_signature_write` | `TokenLabelTable` / `token_label_write` |
| `expand_ds_selection_to_topk_indices` | `logical_to_physical` |
| `DSAdapterPageOutOfRange` | `DSAdapterError` / `RuntimeError` |
| `hot_pages=` kwarg | removed (no longer accepted) |
| `SchedulerOutputProcessorMixin` | `SchedulerBatchResultProcessor` |
| `nsa_decode_backend` | `dsa_decode_backend` |
| `m3b_page_stability_fixture` / `M3BFixture*` | `token_label_write` equivalents |
| Error trigger via OOB page position | Error trigger via OOB pool index |

## Files Changed

**Created:**
- `python/sglang/srt/layers/attention/double_sparsity/token_label_table.py`
- `python/sglang/srt/layers/attention/double_sparsity/token_label_write.py`

**Modified:**
- `python/sglang/srt/layers/attention/double_sparsity/__init__.py`
- `python/sglang/srt/layers/attention/double_sparsity/channel_mask.py`
- `python/sglang/srt/layers/attention/double_sparsity/config.py`
- `python/sglang/srt/layers/attention/double_sparsity/page_table_adapter.py`
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
- `python/sglang/srt/layers/attention/double_sparsity/selector.py`
- `python/sglang/srt/layers/attention/double_sparsity/validator.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`

**Deleted:**
- `python/sglang/srt/layers/attention/double_sparsity/page_signature_table.py`
- `python/sglang/srt/layers/attention/double_sparsity/page_signature_write.py`

## Validation

```
grep -c "def test_" test/registered/unit/layers/attention/test_double_sparsity_unit.py
150  ✓

grep -n "PageSignatureTable|page_signature_write|expand_ds_selection_to_topk_indices|DSAdapterPageOutOfRange|hot_pages=|SchedulerOutputProcessorMixin|nsa_decode_backend|m3b_page_stability_fixture" test/registered/unit/layers/attention/test_double_sparsity_unit.py
(empty) ✓

git diff --stat HEAD~1 | tail -1
14 files changed, 1280 insertions(+), 2220 deletions(-)  ✓
```

AC-0 positive checks satisfied at code level:
- `from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk` — re-exports present in `__init__.py`
- `page_table_adapter.py` is 72 LOC (< 150) ✓
- `token_label_table.py` allocates shape `[L_local, max_tokens, H_local, label_dim]` ✓
- `DoubleSparsityConfig.top_k` defaults to 2048 ✓
- `DSGraphState.selected_indices` shape is `int32[max_bs, max_top_k]` ✓
- `validator.py` reads `dsa_prefill_backend` / `dsa_decode_backend` ✓
- Importing `page_signature_table` or `page_signature_write` now raises `ModuleNotFoundError` (files deleted) ✓

## Remaining Items

All AC-0 and AC-13 code tasks complete. Remaining plan tasks start from AC-1:

- `task-m1-hook` (AC-1): Wire `token_label_write` at `dsa_backend.py` L1439/L1637/L2162
- `task-ac1-hwtest` (AC-1): Hardware test — `forward_extend` → non-zero signatures
- `task-ac2-lifetime` (AC-2): Boot-time GB/rank log; stale-slot negative test
- `task-m2-rangemask` (AC-3): Per-request token range mask in scorer
- … (remaining 20+ tasks per plan)

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: AC-0 rotation followed the plan exactly. No unexpected problems that would
generalize into a new lesson; all issues encountered (missing DSAdapterError,
wrong scheduler class name, wrong error trigger mechanism) were one-off
discovery-during-migration issues fully resolved in this round.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
cb6004a36 docs: restore CLUSTER.md on dev/double-sparsity-standalone
20bf84515 [Sparsity] Loop-4: plan + refined_plan_v1 + QA ledger
ae04e4c3d [Sparsity] Loop-4 Round-0: AC-0 token-level label rotation
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
(first round, no prior history)

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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-0-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
