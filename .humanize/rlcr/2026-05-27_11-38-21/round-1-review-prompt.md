# Code Review - Round 1

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-1-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 1 Summary

## Objective

Close AC-0 and AC-13 by repairing the 5 verified gaps reported in the Round-0 Codex review.

## Result

**150 passed, 0 failed** — all success criteria from the Round-1 contract are met.

## Work Completed

### Gap 1: `retrieve_topk` not exported (AC-0)

- `selection_kernel.py`: Added `retrieve_topk = retrieve_topk_via_labels` alias at module bottom.
- `__init__.py`: Added `from .selection_kernel import retrieve_topk` and `"retrieve_topk"` to `__all__`.
- Verification: `python -c "from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk; print('OK')"` prints OK.

### Gap 2: Selector domain — physical slots instead of logical positions (AC-0, AC-3)

- `selection_kernel.py`: Added `_compute_logical_token_scores()` helper that gathers physical labels per logical position via `req_to_token` gather, returning `[bs, max_seq_len]` fp32 scores with unwritten/OOB positions masked to `-inf`.
- `retrieve_topk_via_labels` detects optional `(req_pool_indices, req_to_token, seq_lens)` triple; when present, runs logical-domain scoring and returns logical positions (0-indexed within each request's sequence); when absent, falls back to original physical-slot path (preserves probe and unit test compatibility).
- `selector.py`: `retrieve_topk` passes the three new optional args through.

### Gap 3: Bind timing — DS bound before KV pool exists (AC-0, AC-1, AC-2)

- `deepseek_v2.py`: `__init__` now stores `self._ds_deferred_bind_args` instead of calling `_bind_double_sparsity_runtime_data` directly. New method `finalize_double_sparsity_bind()` calls it then clears the stored dict.
- `_bind_double_sparsity_runtime_data` removed the `device_buffer_size` fallback; now raises `RuntimeError` if `_ds_req_to_token_pool` is None (fail-fast invariant).
- `model_runner.py`: Added post-`init_memory_pool()` loop that calls `finalize_double_sparsity_bind()` on every module that exposes it.

### Gap 4: DS selector receives latent `q_lora` instead of projected Q-noPE (AC-0, AC-1, AC-8)

- `forward_mla.py`: Added `and not self.use_double_sparsity` to the alt-stream condition so DS always waits for `q_b_proj` to complete on the normal path. In the normal branch, derives `q_nope_for_ds = q[..., :self.qk_nope_head_dim]` after `q_b_proj` and passes it as `q_nope=q_nope_for_ds` to `_select_topk_indices`.
- `deepseek_v2.py:_select_topk_indices`: Added `q_nope: Optional[torch.Tensor] = None` parameter; uses `q_nope` when provided, falls back to `q_lora` otherwise.
- Gate alignment fix: alt-stream branch inner gate updated to `if (self.use_double_sparsity or not self.skip_topk or prev_topk_indices is None):` so both branches carry the identical DS-aware predicate required by `TestSkipTopkGateRespectsDS`.

### Gap 5: AC-13 test failures and stale `nsa_*` names (AC-13)

- Test file: renamed `nsa_prefill_backend`/`nsa_decode_backend` kwargs to `dsa_prefill_backend`/`dsa_decode_backend`.
- `test_nsametadata_has_ds_topk_indices_out_field`: import updated to `DSAMetadata as NSAMetadata`.
- `test_forward_decode_dispatches_to_flashmla_kv`: fixed `nsa_*` metadata fields to `dsa_*`; added `backend.dsa_decode_impl`, `backend.token_to_kv_pool`, and `backend.hisparse_coordinator = None` to mock.
- `test_ds_branch_returns_topk_indices_via_adapter`: added `req_to_token_pool` to `forward_batch` so the logical-domain path is exercised.
- `test_probe_finds_planted_needle`: fixed `max_tokens=16→512` and `needle_position=4→4*64` for token-level label layout.

### Side fix: `channel_mask.py` needle_position inconsistency

Both skip paths now use `needle_page * page_size` for consistent physical-address results.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/__init__.py`
- `python/sglang/srt/layers/attention/double_sparsity/channel_mask.py`
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`
- `python/sglang/srt/layers/attention/double_sparsity/selector.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
- `python/sglang/srt/models/deepseek_v2.py`
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`

## Validation

| Command | Result |
|---------|--------|
| `python -c "from sglang.srt.layers.attention.double_sparsity import TokenLabelTable, token_label_write, retrieve_topk; print('OK')"` | OK |
| `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` | 150 passed, 0 failed |

## Success Criteria Verification

| Criterion | Result |
|-----------|--------|
| `from ... import TokenLabelTable, token_label_write, retrieve_topk` | PASS |
| Non-contiguous fixture: logical [0,1,2,3] → physical [7,64,200,512] | PASS (test_ds_branch_returns_topk_indices_via_adapter) |
| 150 tests pass | PASS — 150 passed, 0 failed |
| `_bind_double_sparsity_runtime_data` raises if pool absent | PASS — RuntimeError raised, no silent fallback |
| `_select_topk_indices` passes `q[..., :qk_nope_head_dim]` to DS | PASS — q_nope_for_ds derived after q_b_proj |

## Remaining Items

None for this round's scope. AC-0 and AC-13 are closed. Next round targets AC-1 (live token-label cache population from KV-write path).

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: All fixes addressed pre-identified Round-0 gaps. No new generalizable lessons arose.
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
cb6004a36 docs: restore CLUSTER.md on dev/double-sparsity-standalone
20bf84515 [Sparsity] Loop-4: plan + refined_plan_v1 + QA ledger
ae04e4c3d [Sparsity] Loop-4 Round-0: AC-0 token-level label rotation
7be9fd7a8 [DS Loop-4 Round-1] Fix 5 AC-0/AC-13 gaps: export, domain, bind-timing, Q-noPE, test renames
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-1-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
