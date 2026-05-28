Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop4/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop4/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-7-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 6 Review Result

Mainline Progress Verdict: ADVANCED

Round 6 materially advanced the prior AC-2/AC-3 gaps. The stale-slot invalidation helper exists, `_select_topk_indices` calls it before `retrieve_topk`, the table-size guard exists, and the logical-domain `req_to_token` isolation test covers the scorer/adapter behavior. I verified the claimed unit suite:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
170 passed, 24 warnings in 11.58s
```

AC-3 is verified. AC-2 is still partial because the production invalidation hook itself is not protected by a regression test.

## Mainline Gaps

1. **AC-2 live invalidation wiring is untested, so AC-2 should not be marked verified yet.**

   Evidence:
   - The production hook is present at `python/sglang/srt/models/deepseek_v2.py:2087-2093`, before `selector.retrieve_topk` at `:2097-2105`.
   - The new stale-slot tests at `test/registered/unit/layers/attention/test_double_sparsity_unit.py:5080-5151` call `invalidate_token_label_slots` directly, then call `retrieve_topk_via_labels` directly. They do not call `_select_topk_indices`.
   - If the `_select_topk_indices` invalidation block were deleted, the new AC-2 tests would still pass while the original production stale-slot bug would return.

   Required fix: add a `TestSelectTopkIndicesHookBranch` regression that constructs an attention fixture with `use_double_sparsity=True`, a selector carrying `token_label_table.written` with slot 7 set `True`, and `forward_batch.out_cache_loc=torch.tensor([7])`. Replace `selector.retrieve_topk` with a spy/side effect that asserts `written[0, 7]` is already `False` when called, then returns a small logical result. This test must fail if lines `2087-2093` are removed.

2. **The original lower-bound work remains incomplete; Claude's Round 6 "Remaining Items" under-reports it.**

   Still pending from the plan:
   - AC-1 H200 real `forward_extend` population and AC-8 selector-read smoke.
   - AC-1b chunked-prefill probe.
   - AC-4 Method 1 calibration plus H200 mask generation. Current `calibrate.py` still describes and computes K-only L2 statistics (`pow(2)`), not `mean(abs(Q_nope * K_nope))`.
   - AC-5 TP=2 multiprocess all-reduce harness. The planned `test/registered/integration/test_double_sparsity_tp_multiprocess.py` does not exist.
   - AC-6 graph capture. `capture_decode_step` still has no `req_to_token` parameter and therefore cannot exercise the logical-domain selector path.
   - AC-7 short-seq prefill bypass. `forward_absorb_prepare` still calls `_select_topk_indices` whenever DS is enabled; no prefill-below-threshold bypass test exists.
   - AC-8 bench_serving + lightweight quality smoke.
   - AC-12 hard NIAH/MMLU quality gate. `test/manual/test_double_sparsity_v32.py` is still a skip-only scaffold.
   - AC-9 through AC-11 stretch baseline/radix/comparator work.

## Blocking Side Issues

None outside the mainline gaps above.

## Queued Side Issues

1. Before AC-6, update `capture_decode_step` to accept and pass `req_to_token`; otherwise graph capture uses physical-domain fallback.
2. Before AC-8, fix DS observability fields that still publish token selections as page metrics and divide by page counts.
3. Clean stale lifetime documentation: `token_label_table.py` still says reused slots are safe because writes happen before reads, but the actual invariant is now invalidate-before-selection, write-after-selection.

## Goal Alignment Summary

```text
ACs: 5/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Met/verified: AC-0, AC-3, AC-13. Partial: AC-1, AC-2. Not met: AC-1b, AC-4, AC-5, AC-6, AC-7, AC-8, AC-9, AC-10, AC-11, AC-12.

There are no accepted deferrals in the tracker. Hardware/analyze tasks remain active, not waived.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Kept `task-ac2-lifetime` Active as `partial-r6-review` pending the live `_select_topk_indices` invalidation regression.
- Moved AC-3 to Completed and Verified for Round 6.
- Added a Round 6 Review plan-evolution row.
- Added the stale token-label lifetime docstring as a queued side issue.

## Required Implementation Plan

1. Finish AC-2 verification first. Add the live `_select_topk_indices` invalidation regression described above, rerun the unit suite, then move `task-ac2-lifetime` to Completed and Verified.
2. Implement AC-7 next. In the DS branch, skip selector invocation during extend/prefill when the DSA backend is in dense MHA mode below `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`; keep the KV/label write path active. Add tests proving `retrieve_topk` is not called for short prefill, labels are written, and first decode calls `retrieve_topk`.
3. Implement AC-4 calibration. Replace K-only L2 accumulation in `calibrate.py` with paired Q-noPE and K-noPE hooks from the same forward pass, compute `mean(abs(Q_nope * K_nope))`, keep `qk_nope_head_dim=128`, use Pile validation with seed 42 and 256 x 512 tokens, and add fixture tests for Q/K pairing and 512-d index rejection. Then run the H200 mask generation and validate the safetensors file.
4. Implement AC-5. Create the TP=2 multiprocess test file, initialize a real process group, all-reduce `[bs, max_seq_len]` logical-domain score tensors, assert bit-equal logical indices, and include the no-op all-reduce negative.
5. Implement AC-6. Thread `req_to_token` through `capture_decode_step`, preallocate all DS output/scratch buffers before capture, add eager-vs-replay and allocation-negative tests, then run the H200 conc=64 capture/replay check.
6. Run AC-1 H200 population, AC-1b chunked-prefill probe, AC-8 bench_serving and quality smoke, and AC-12 full NIAH/MMLU gate in that order. Only after AC-12 passes should AC-9 through AC-11 stretch runs be attempted.

NOT COMPLETE
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-7-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-7-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
