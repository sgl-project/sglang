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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-5-contract.md

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
# Round 4 Review Result

Mainline Progress Verdict: ADVANCED

Round 4 materially advanced AC-1. The prior blocking gap was production call-site coverage, and the new `TestAC1CallSites` class now exercises the three `_write_token_labels` call sites with `save_kv_cache=True`: `forward_extend`, `forward_decode`, and `_forward_trtllm`. I verified the hook sites are still present at `python/sglang/srt/layers/attention/dsa_backend.py:1510`, `:1709`, and `:2233`, and the local suite passes:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
160 passed, 24 warnings in 11.58s
```

AC-1 is still partial, not met. The plan still requires the H200 real `forward_extend` population test, and AC-1's integration smoke is tied to AC-8 `bench_serving` not crashing on selector reads.

## Part 1: Goal Tracker Audit

### Acceptance Criteria Status

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-0 | MET | Public import succeeds; `page_table_adapter.py` is 71 LOC; `TestAC0RealSlotRegression` covers non-contiguous physical slots and the corrected physical KV slot sizing. Note: the literal immutable text saying `req_to_token_pool.size` was superseded by Round 2 because it contradicted slot-indexing by `out_cache_loc`; current authority is `kv_pool.size + kv_pool.page_size`. |
| AC-1 | PARTIAL | Unit call-site coverage is now verified: tests at `test/registered/unit/layers/attention/test_double_sparsity_unit.py:4601` cover extend, decode, TRT-LLM FP8, and `save_kv_cache=False`. Blockers: `task-ac1-hwtest` on H200 and AC-8 selector-read smoke. |
| AC-1b | NOT MET | Chunked-prefill probe has not run. |
| AC-2 | NOT MET | Lifetime/stale-slot work is pending. No boot-time GB/rank log or stale-slot negative test evidence yet. |
| AC-3 | NOT MET | Per-request ownership mask and boundary negative test are pending. |
| AC-4 | NOT MET | Method 1 Q+K calibration code and H200 mask generation are pending. Design doc `development/past_implementations/study/07-mvp-proposed-architecture.md` sections 9-10 keep real calibration and quality smoke in MVP scope. |
| AC-5 | NOT MET | TP=2 multiprocess harness is pending. |
| AC-6 | NOT MET | CUDA graph capture work and H200 replay are pending; queued graph helper issue remains. |
| AC-7 | NOT MET | Short-seq MHA bypass verification is pending. |
| AC-8 | NOT MET | 8xH200 `bench_serving` and lightweight quality smoke are pending. |
| AC-9 | NOT MET | Stretch DSA baseline JSON is pending, not deferred. |
| AC-10 | NOT MET | Stretch radix-cache fixture and config flip are pending, not deferred. |
| AC-11 | NOT MET | Stretch comparator row is pending, not deferred. |
| AC-12 | NOT MET | Hard NIAH/MMLU quality gate is pending. The loop cannot close without this. |
| AC-13 | MET | Regression suite is green after shape migration. Current suite has grown to 160 tests and passes. |

### Forgotten Items Detection

No actionable forgotten plan item found after tracker correction. Remaining original tasks are in Active, Completed and Verified, or represented by queued side issues with explicit revisit triggers. The AC-0 task granularity is collapsed under verified AC-0 rows from earlier reviews; the stale `capture_decode_step` work is tracked as queued until `task-ac6-cuda-graph`.

One tracker drift item was corrected: `task-m1-hook` was duplicated between Active and Completed with `pending-codex`. I moved it out of Active and marked it verified in Round 4. I also added a queued cleanup for stale `deepseek_v2.py` comments that still say `max_tokens = req_to_token_pool.size`.

### Deferred Items Audit

There are no explicitly deferred items in the tracker. AC-9 through AC-11 are stretch tasks but still active/pending, not accepted deferrals. No deferral currently contradicts the Ultimate Goal.

### Goal Completion Summary

```text
Acceptance Criteria: 2/15 met (0 deferred, 1 partial)
Active Tasks: 17 remaining
Estimated remaining rounds: at least 8-10, hardware-window dependent
Critical blockers: H200 access for AC-1/AC-4/AC-6/AC-8/AC-12 hardware gates; generated V3.2 channel mask not available yet
```

## Part 2: Mainline Drift Audit

The current round's objective was clear and singular: close the AC-1 call-site verification gap from Round 3. Claude advanced the mainline by adding targeted production-path tests rather than clearing unrelated side issues.

The repeated AC-1 reopening over Rounds 2-4 is not stagnation because each review found a different concrete blocker and the next round fixed it: Round 2 fixed slot authority and wired hooks, Round 3 fixed projection/input correctness, and Round 4 added call-site tests.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 0
Queued Side Issues: 3
```

## Part 3: Implementation Review

No high-signal correctness issue found in the Round 4 test implementation.

Verified claims:
- `test_forward_extend_writes_token_labels` calls `forward_extend` with `save_kv_cache=True` and asserts writes at slots 5 and 10.
- `test_forward_decode_writes_token_labels` calls `forward_decode` with `save_kv_cache=True` and asserts writes at slots 7 and 15.
- `test_trtllm_hook_receives_pre_quantized_k` calls `_forward_trtllm`, patches FP8 quantization, spies on `_write_token_labels`, and verifies the hook receives the original float latent K rather than FP8 output.
- `test_no_labels_when_save_kv_cache_false` proves the hook is inside the KV-write guard.

Important limit: these are CPU/unit call-site tests with patched attention kernels. They prove the Python production calls are wired and guarded, but they do not replace `task-ac1-hwtest` against real V3.2/H200 execution.

Queued implementation cleanup:
- `python/sglang/srt/models/deepseek_v2.py:1541-1544` and `:1836-1840` still contain stale text saying table sizing comes from `req_to_token_pool.size`. The actual code at `:1906-1917` correctly uses physical KV slot capacity, so this does not reopen AC-0. It should be cleaned when DS bind/runtime comments are next touched.

## Part 4: Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Added a Round 4 Review plan-evolution row.
- Moved `task-m1-hook` from Active to Completed and Verified with Verified Round `4`.
- Left `task-ac1-hwtest` active, so AC-1 remains partial.
- Added the stale `deepseek_v2.py` sizing-comments issue to Queued Side Issues.

## Part 5: Progress Stagnation Check

No stagnation trigger. The loop had one regression in Round 0, then steady mainline recovery and advancement in Rounds 1-4. Feedback has not been circular: each repeated AC-1 review item was narrowed and addressed in the next round. The loop is incomplete, but not stalled.

## Action Items

### Mainline Gaps

1. Run `task-ac1-hwtest` on H200: real V3.2 `forward_extend`, assert `token_label_table.signatures[layer_id, out_cache_loc]` is non-zero for each written slot.
2. Continue dependency order after AC-1 hardware evidence: AC-2 lifetime/stale-slot, AC-3 range mask and boundary test, AC-7 short-seq bypass, AC-4 Method 1 Q+K calibration and H200 mask generation, AC-5 TP harness, AC-6 graph capture, AC-1b probe, AC-8 bench/quality smoke, AC-12 full quality.

### Blocking Side Issues

None verified in this round.

### Queued Side Issues

1. Before `task-ac6-cuda-graph`, update `capture_decode_step` to use the logical-domain selector path with `req_to_token`.
2. Before AC-8 server/quality smoke, fix DS observability metrics that still report token selections through page-named fields and page-count denominators.
3. Clean stale `deepseek_v2.py` comments/docstrings that still point to the old `req_to_token_pool.size` sizing authority.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-5-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

### Post-Alignment Check Action Items

This round follows a Full Goal Alignment Check. Pay special attention to:
- **Forgotten Items**: Codex may have identified tasks that were being ignored. Address them.
- **AC Status**: If any Acceptance Criteria were marked NOT MET, prioritize work toward those.
- **Deferred Items**: If any deferrals were flagged as unjustified, un-defer them now.
- **Queued Issues**: Keep non-blocking follow-up work queued unless it now clearly blocks mainline progress.

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-5-summary.md

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
