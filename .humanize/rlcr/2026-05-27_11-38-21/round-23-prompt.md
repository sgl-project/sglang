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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-23-contract.md

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
# Round 22 Code Review

Mainline Progress Verdict: ADVANCED

Round 22 fixes the two Round 21 AC-8 smoke-harness blockers and completes the residual page→token rename in `development/benchmark_compare.py`. I do not find a new Round 22 runtime defect that should block the paired H200 smoke run. The Loop 4 plan is still incomplete: hardware gates, AC-12, and the AC-8/AC-9/AC-11 run tooling remain active work and must not be treated as accepted deferrals.

## Implementation Review

Verified Round 22 claims:
- `test/manual/test_dsv32_quality_smoke.py:208-221` now implements first-n token overlap with set intersection, so shifted overlap is accepted.
- `test/manual/test_dsv32_quality_smoke.py:290-293` now counts prefix hits with only `ds[:32] == dsa[:32]`; the stale `len(dsa) >= 32` guard is gone.
- `development/benchmark_compare.py:61-63`, `211-213`, `274-283`, and `315-316` now use `selected_tokens_mean` / `total_tokens_mean`.
- `rg "selected_pages_mean|total_pages_mean" development/benchmark_compare.py test/registered/unit/layers/attention/test_double_sparsity_unit.py` returns no matches.
- The new helper tests exist at `test/registered/unit/layers/attention/test_double_sparsity_unit.py:5116-5174`.

Validation run:
```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
206 passed, 24 warnings

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_dsv32_quality_smoke.py -q
1 skipped, 1 warning
```

## Goal Alignment Summary

```text
ACs: 10/15 addressed | Forgotten items: 0 | Unjustified deferrals: 4
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.
Partial: AC-1, AC-4, AC-6, AC-8.
Not met: AC-1b, AC-9, AC-10, AC-11, AC-12.

No original-plan task is missing from Active, Completed, or Deferred after tracker reconciliation. `Explicitly Deferred` remains empty, which is correct. The “future round” language in Claude’s summary for AC-12, Option B script alignment, benchmark conc sweep, and AC-11 comparator enforcement is not accepted as justified deferral; these remain incomplete plan work.

## Findings By Lane

### Mainline Gaps

1. The Loop 4 MVP remains incomplete because the remaining original-plan gates have not run.

Required execution plan:
1. Generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200 with the AC-4 production command.
2. Run `task-ac1-hwtest`: real H200 `forward_extend`, then assert token-label rows at `out_cache_loc` are non-zero.
3. Run `task-ac6-hwrun`: V3.2 Option B conc=64 full-graph capture, 100 replays, no CUDA launch failure, eager/graph `max_abs_diff <= 1e-6`.
4. Run `task-ac1b-probe`: `chunked_prefill_size=4096`, compare labels 0..4095 against non-chunked baseline, and record the pass/fail decision.
5. Run AC-8 server smoke with locked Option B flags and conc 16/32/64, then run the corrected paired quality smoke.
6. Implement and run AC-12 before loop closure.
7. Complete AC-9, AC-10, and AC-11 rather than treating them as complete-by-deferral.

2. AC-12 is still a skip-only scaffold.

Evidence:
- `test/manual/test_double_sparsity_v32.py:62-93` still calls `self.skipTest(...)` for NIAH 4K, NIAH 16K, NIAH 64K, MMLU 5-shot, and both negative sensitivity checks.
- The plan states AC-12 is hard: the loop does not close without NIAH deltas ≤ 5 pp and MMLU delta ≤ 1 pp.

Required implementation plan:
1. Replace the scaffold with a paired-server harness requiring `DS_BASE_URL` and `DSA_BASE_URL`.
2. Implement deterministic NIAH datasets at 4K, 16K, and 64K with exact needle scoring.
3. Implement MMLU 5-shot evaluation through existing repo evaluation utilities where possible, persisting per-subject and aggregate scores.
4. Fail when `DSA - DS > 5 pp` for any NIAH length or `DSA - DS > 1.0 pp` for MMLU.
5. Convert the corrupt-mask and zero-signature sensitivity checks into executable fault-injection tests, not skips.

3. AC-8/AC-9/AC-11 run tooling is still not aligned with the locked operating point.

Evidence:
- `development/serve_double_sparsity.sh:44-54` and `development/serve_native_nsa.sh:32-39` still do not pass `--dsa-prefill-backend flashmla_kv`, `--dsa-decode-backend flashmla_kv`, `--disable-overlap-schedule`, or `--disable-piecewise-cuda-graph`.
- `development/benchmark.sh:28-30` and `development/benchmark_baseline.sh:32-34` still default to conc=64 only, while AC-8/AC-9 require conc 16/32/64.
- `development/benchmark_compare.py:21-25` and `254-259` still implement the older absolute SLO/no-op framing, not the AC-11 3-trial median with DS TPS within 5% of DSA and P99 TTFT ≤ 1.10x DSA.

Required implementation plan:
1. Update both server launch scripts to encode the locked Option B flags: FP8 KV cache, `flashmla_kv` prefill/decode backends, overlap off, piecewise CUDA graph off, page size 64, and the AC-1b chunked-prefill decision.
2. Keep `--disable-radix-cache` only on the DS AC-8 path until AC-10 passes.
3. Make benchmark scripts run conc 16/32/64 by default and persist commit SHA, full server args, seed, warmup, measurement window, and chunked-prefill setting.
4. Update `benchmark_compare.py` to consume three trials per concurrency, take medians, enforce DS TPS within 5% of DSA and P99 TTFT ≤ 1.10x DSA, and report profiling obligation on failures.

### Blocking Side Issues

None newly blocking from Round 22. The Round 21 AC-8 smoke-gate blockers are fixed by inspection and by the passing registered suite.

### Queued Side Issues

- The new prefix-match regression tests at `test_double_sparsity_unit.py:5143-5157` manually replicate the slicing expression instead of exercising the actual gate in `TestDSv32QualitySmoke.test_quality_smoke`. Current code is correct, so this should not take over the next round; next cleanup should extract `_prefix_match` or call `test_quality_smoke` with mocked `_run_paired` / `_record`.
- Existing queued docs drift remains: stale `deepseek_v2.py` slot-authority comments and `token_label_table.py` lifetime text.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 22 Review`.
- Added a Round 22 Review plan-evolution row with validation evidence.
- Removed the stale AC-8 smoke-gate blocker from Blocking Side Issues.
- Added the shallow prefix-match regression coverage as a queued side issue.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-23-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-23-summary.md

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
