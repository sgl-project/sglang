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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-6-contract.md

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
# Round 5 Review Result

Mainline Progress Verdict: ADVANCED

Round 5 added useful unit coverage and the local unit suite is green:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
166 passed, 24 warnings in 11.46s
```

However, AC-2 and AC-3 are not closed. The new tests verify boot-log text, direct overwrites, and a physical-domain `per_request_valid` helper mask, but they do not prove the allocator-lifetime stale-slot invariant or the production `req_to_token` ownership path required by the original plan.

## Mainline Gaps

1. **AC-2 stale-slot protection is missing; a reused slot can be selected before the new label write.**

   Evidence:
   - The new stale-slot test only writes A and then immediately writes B to the same slot (`test_double_sparsity_unit.py:5020`). It never simulates free/reallocation followed by selector read before B is written, which is the plan's negative fixture.
   - `token_label_write.py:53-54` sets `written[layer_id, slot] = True` and there is no corresponding lifecycle invalidation on slot reuse.
   - `_compute_logical_token_scores` gathers labels from `req_to_token` and masks only by `written_layer[safe_phys]` and `seq_lens` (`selection_kernel.py:383-400`). If the old request left `written=True`, the stale label is valid to the selector.
   - In the live decode path, `_select_topk_indices` is called before `attn_mqa` writes the current KV/label (`forward_mla.py:283`, then `dsa_backend.py:1703-1709`). That means the current step can read a reused slot before `_write_token_labels` overwrites it.

   I verified this with a synthetic fixture: old request writes `1000.0` at physical slot 7, new `req_to_token` maps logical position 0 to slot 7, no new write occurs, and `retrieve_topk_via_labels` returns `[[0]]` with `valid_lengths=[1]`. That is exactly the stale read AC-2 was meant to prevent.

2. **AC-2 slot-budget/fail-fast coverage is still incomplete.**

   `test_slot_budget_covers_all_physical_kv_slots` (`test_double_sparsity_unit.py:4994`) allocates an arbitrary `max_tokens = kv_pool_size + page_size` and writes the last slot. It does not exercise a server/bind-time invariant that token-label capacity cannot exceed the KV pool slot budget, and there is no boot-time HBM/fail-fast gate in `token_label_table.py` or `validator.py` that would reject an independently over-sized table as the plan requires.

3. **AC-3 production ownership is not tested.**

   Both new AC-3 tests call `retrieve_topk_via_labels` without `req_pool_indices`, `req_to_token`, or `seq_lens` (`test_double_sparsity_unit.py:5119` and `:5157`). That forces the physical-domain branch (`selection_kernel.py:440-468`), so the tests only prove that an explicit `[bs, max_tokens]` `per_request_valid` mask works.

   The production path is different: `DeepseekV2AttentionMLA._select_topk_indices` obtains `forward_batch.req_to_token_pool.req_to_token`, calls `selector.retrieve_topk(..., req_to_token=req_to_token)`, then maps logical positions back through `logical_to_physical` (`deepseek_v2.py:2061-2114`). Round 5 does not add a boundary test for that path, so a regression that drops `req_to_token` from the live call could still escape.

4. **Remaining original-plan work is still pending.**

   Claude's "Remaining Items" are unfinished tasks, not accepted deferrals: AC-1 H200 population, AC-1b chunked-prefill probe, AC-4 calibration and mask generation, AC-5 TP harness, AC-6 CUDA graph capture, AC-7 short-seq bypass, AC-8 bench/quality smoke, AC-12 full quality gate, and AC-9 through AC-11 stretch rows all remain open.

## Blocking Side Issues

None outside the mainline. The stale-slot bug above is itself an AC-2 mainline gap and must be fixed before AC-2 can be marked complete.

## Queued Side Issues

1. Before AC-6, update `capture_decode_step` to pass `req_to_token` and avoid the physical-domain fallback during graph capture.
2. Before AC-8, fix DS observability metrics that still use page-named fields and page-count denominators.
3. Clean stale `deepseek_v2.py` comments that still describe table sizing as `req_to_token_pool.size`.

## Goal Alignment Summary

```text
ACs: 5/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Met: AC-0, AC-13. Partial: AC-1, AC-2, AC-3. All other ACs remain not met. The tracker has no explicit deferred tasks; AC-9 through AC-11 remain active stretch work, not accepted deferrals.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Added a Round 5 Review plan-evolution row reopening AC-2/AC-3 as partial.
- Kept `task-ac2-lifetime`, `task-m2-rangemask`, and `task-ac3-test` in Active with `partial-r5-review`.
- Removed the pending AC-2/AC-3 rows from Completed and Verified.

## Required Implementation Plan

1. Fix AC-2 stale-slot lifetime now. Add a capture-safe invalidation helper for token-label slots, e.g. `invalidate_token_label_slots(table, layer_id, cache_loc)`, that sets `table.written[layer_id, cache_loc] = False` before the selector can read newly allocated slots. Call it from the DS branch of `_select_topk_indices` before `selector.retrieve_topk`, using `forward_batch.out_cache_loc` and the current `layer_id`. This makes reused slots unselectable until `_write_token_labels` rewrites them later in `dsa_backend.py`; it also protects `save_kv_cache=False` fused paths by leaving the slot invalid instead of stale.
2. Add the missing AC-2 regression: write old label A to slot N, reassign N through `req_to_token` for a new request, run selection before the new write and assert the old label is not selectable; then write label B and assert B becomes selectable. Add a bind-time shape guard so reused or preexisting token-label tables must have `max_tokens == kv_pool.size + kv_pool.page_size`, and test the oversize/undersize failure path.
3. Replace the Round 5 AC-3 helper-only test with a production-path boundary test. Build a real bound `DoubleSparsitySelector` or `_select_topk_indices` fixture with two `req_to_token` rows mapping logical positions to disjoint physical ranges; make request 1's physical slots outscore request 0's globally; assert the adapter output for each request stays inside that request's `req_to_token` row. The negative test should deliberately drop `req_to_token` or call the physical helper without a mask and show contamination.
4. Rerun the full unit suite. Only after AC-2/AC-3 are fixed and verified should the tracker move those tasks to Completed and Verified.
5. Continue in dependency order: AC-7 short-seq bypass including `save_kv_cache=True`/fused-path verification; AC-4 Method 1 calibration by capturing Q-noPE and K-noPE in the same forward pass and computing `mean(abs(Q_nope * K_nope))`; AC-5 TP=2 multiprocess logical-domain all-reduce; AC-6 graph capture with preallocated buffers and `req_to_token`; AC-1 H200 population, AC-1b chunked-prefill probe, AC-8 bench/quality smoke, AC-12 full NIAH/MMLU quality gate, then AC-9 through AC-11 stretch measurements.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-6-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-6-summary.md

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
