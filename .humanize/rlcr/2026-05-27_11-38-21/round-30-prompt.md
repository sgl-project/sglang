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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-30-contract.md

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
# Round 29 Full Goal Alignment Review

Mainline Progress Verdict: ADVANCED

Round 29 fixed the two AC-12 harness defects found in Round 28. I found no new high-signal Round 29 code bug: the non-env `pandas` skip is gone, `_load_mmlu_examples` is stdlib-CSV only, and the `AC12_MMLU_SUBJECTS=beta` artifact path no longer references an undefined `subjects` name.

The overall Loop 4 goal is still incomplete. The design doc and refined plan still require real H200 evidence for calibration, CUDA graph capture/replay, bench_serving, AC-8 smoke, AC-12 NIAH/MMLU quality, AC-9 baseline, AC-10 radix, and AC-11 comparator. Those are active tasks, not deferrals.

## Part 1: Goal Tracker Audit

### 1.1 Acceptance Criteria Status

| AC | Status | Evidence (if MET / PARTIAL) | Blocker (if NOT MET / PARTIAL) | Justification (if DEFERRED) |
|----|--------|------------------------------|--------------------------------|-----------------------------|
| AC-0 | MET | Token-level architecture rotation verified in prior Codex reviews; current combined registered suite passed `280 passed`; tracker Completed row cites `task-ac0-slot-authority` with `TestAC0RealSlotRegression`. | - | - |
| AC-1 | PARTIAL | `task-m1-hook` code path verified by call-site tests; current registered suite still passes. | `task-ac1-hwtest` is still pending: real H200 `forward_extend` population and AC-8 selector-read smoke are not run. | - |
| AC-1b | NOT MET | - | Chunked-prefill probe has not run; depends on H200 server/capture path. | - |
| AC-2 | MET | Round 7 verified stale-slot invalidation live wiring; current registered suite passed `280 passed`. | - | - |
| AC-3 | MET | Round 6 verified logical-domain range-mask + adapter isolation; current registered suite passed `280 passed`. | - | - |
| AC-4 | PARTIAL | Calibration code path is verified locally through Round 13 regressions; current registered suite passed. | H200 production run has not generated and validated `/models/dsv32-fp8-channel-mask.safetensors`. | - |
| AC-5 | MET | `PYTHONPATH=python pytest test/registered/integration/test_double_sparsity_tp_multiprocess.py -q` passed `3 passed`. | - | - |
| AC-6 | PARTIAL | CUDA graph code-tier fixes were verified in Round 20 and remain covered by registered tests. | Real V3.2 conc=64 capture + 100 replay steps + eager/graph match on H200 are still pending. | - |
| AC-7 | MET | Round 9 verified short-prefill dense bypass plus first-decode sparse selection; current registered suite passed. | - | - |
| AC-8 | PARTIAL | Benchmark/smoke harnesses and sidecar metadata fixes exist. | No 8xH200 DS `bench_serving` evidence at conc 16/32/64; no same-session AC-8 quality smoke result. | - |
| AC-9 | NOT MET | - | DSA Option B baseline JSON is not produced. | - |
| AC-10 | NOT MET | - | M3-B radix hardware fixture, FP8 scale-stability check, radix flag flip, and launch-script update are not done. | - |
| AC-11 | NOT MET | - | Comparator still lacks the AC-11 3-trial median/directional gate semantics and no comparator row has been run. | - |
| AC-12 | PARTIAL | Round 29 harness readiness verified: AC-12 helper suite `47 passed`; empty-dir configured-server reproducer fails instead of skips; targeted no-pandas and subjects-filter regressions passed. | Full paired DS/DSA H200 quality run is still pending: NIAH @ 4K/16K/64K, MMLU 5-shot, and optional fault-injected sensitivity servers. | - |
| AC-13 | MET | Shape-migration regression suite remains green inside the current combined registered run (`280 passed`). | - | - |

### 1.2 Forgotten Items Detection

No original-plan task is forgotten. The tracker’s Active Tasks still cover the remaining original work: `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`, `task-ac12-quality`, `task-ac9-baseline`, `task-ac10-radix`, and `task-ac11-compare`.

The original AC-0 subtask names were consolidated by earlier plan-evolution rows and verified through completed AC-0/AC-13 evidence. That consolidation is acceptable and not a missing task.

No task is newly marked complete without verification. Round 29’s claim that the AC-12 harness is gate-tight is now verified at code-tier, but AC-12 itself remains partial because hardware execution has not happened.

### 1.3 Deferred Items Audit

The tracker has no entries under Explicitly Deferred. That is correct. Hardware-gated items are Active, not deferred, and must remain active until evidence exists.

### 1.4 Goal Completion Summary

```text
Acceptance Criteria: 6/15 met (0 deferred)
Active Tasks: 10 remaining
Estimated remaining rounds: 6-8 if H200 access is available; unbounded if hardware remains unavailable
Critical blockers: H200 execution for AC-1/1b/4/6/8/12; AC-10 radix work; AC-11 comparator semantics and runs
```

## Part 2: Mainline Drift Audit

The current round’s objective was clear and singular: close the two Round 28 AC-12 harness blockers. Claude did that.

Recent rounds have mostly cleared AC-12 harness issues rather than running hardware gates. That is still mainline work because the refined plan and design doc §9.5 B6 make AC-12 a hard quality gate, and a skip-prone harness could falsely close the loop. However, the next round should pivot to hardware evidence or the loop risks drifting into local harness polish.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 0
Queued Side Issues: 4
```

Blocking side issues: none active after Round 29. The Round 28 pandas skip and `subjects` NameError are resolved.

Queued side issues: the tracker still carries AC-11 comparator cleanup, AC-8 prefix-match regression cleanup, stale DS bind/runtime comments, and stale token-label lifetime docs. The AC-11 item is not optional for final loop completion; it is queued only because AC-12/hardware evidence should be tackled first.

## Part 3: Implementation Review

Verified Round 29 claims:

- `test/manual/test_double_sparsity_v32.py:329-335` uses `csv.reader` and has no `pandas` import.
- `test/manual/test_double_sparsity_v32.py:623-637` no longer contains the in-test `try: import pandas / self.skipTest(...)` branch.
- `test/manual/test_double_sparsity_v32.py:670-681` normalizes `subjects_for_artifact`, and `test/manual/test_double_sparsity_v32.py:732-745` records that value.
- `test/registered/unit/manual/test_ac12_helpers.py:445-481` blocks any `pandas` import and proves `_load_mmlu_examples` succeeds.
- `test/registered/unit/manual/test_ac12_helpers.py:501-590` drives the full `test_mmlu_5shot` subject-filter path with mocked generation and verifies `subjects == ["beta"]`.

Validation I ran:

```text
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
47 passed, 1 warning

PYTHONPATH=python pytest \
  test/registered/unit/layers/attention/test_double_sparsity_unit.py \
  test/registered/unit/development test/registered/unit/manual -q
280 passed, 24 warnings

PYTHONPATH=python pytest test/registered/integration/test_double_sparsity_tp_multiprocess.py -q
3 passed, 5 warnings

env -u DS_BASE_URL -u DSA_BASE_URL PYTHONPATH=python \
  python -m pytest test/manual/test_double_sparsity_v32.py -q
6 skipped, 1 warning

PYTHONPATH=python pytest \
  test/registered/unit/manual/test_ac12_helpers.py::TestAC12HarnessHelpers::test_mmlu_5shot_subjects_filter_does_not_crash \
  test/registered/unit/manual/test_ac12_helpers.py::TestAC12HarnessHelpers::test_load_mmlu_examples_works_without_pandas -q
2 passed, 1 warning
```

Round 27/28 empty-dir reproducer remains correctly loud:

```text
DS_BASE_URL=http://127.0.0.1:1 DSA_BASE_URL=http://127.0.0.1:2 \
  AC12_MMLU_DATA_DIR=<tmp-with-empty-dev-test> PYTHONPATH=python \
  python -m pytest test/manual/test_double_sparsity_v32.py::TestDoubleSparsityV32Quality::test_mmlu_5shot -q

Result: 1 failed, not skipped
```

No high-signal Round 29 implementation defect found.

Residual execution condition: AC-12 evidence must be a full gate run, not a 200-example smoke. The harness defaults `AC12_MMLU_NUM_EXAMPLES` to 200 for practicality, so the final AC-12 run must explicitly evaluate the intended full MMLU set (for example, set the cap high enough or disable the cap) before claiming AC-12.

## Part 4: Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md` mutable content only:

- Plan Version now says `Updated: Round 29 Review`.
- Added a Round 29 Review evolution row with the validation evidence above.
- Updated `task-ac12-quality` notes to say Codex verified the Round 29 harness fixes and that only H200 execution remains for AC-12.

No requested tracker change was rejected. I did not modify the immutable Ultimate Goal or Acceptance Criteria.

## Part 5: Progress Stagnation Check

Not stagnating yet.

There is a repeated pattern across Rounds 26-29: Codex keeps finding AC-12 harness bypasses after Claude claims the harness is ready. That is a process smell. But each round closed a distinct, verified blocker, and Round 29 did not introduce another code-tier AC-12 blocker. The loop has advanced from skip-only/zero-shot MMLU to a runnable, fail-loud 5-shot harness with fault-injection support.

The next round should move to H200 evidence. Another local-only AC-12 harness cleanup round without hardware progress would be a stronger stagnation signal.

## Action Items

### Mainline Gaps

1. Run `task-ac4-hwrun`: generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200.
2. Run `task-ac1-hwtest` and `task-ac6-hwrun`: real label population plus V3.2 conc=64 CUDA graph capture/replay evidence.
3. Run `task-ac1b-probe`, then lock the chunked-prefill launch decision for both DS and DSA.
4. Run `task-ac8-server` and `task-ac8-quality`: DS bench_serving conc 16/32/64 plus same-session quality smoke.
5. Run `task-ac12-quality` on paired DS/DSA servers with a full intended MMLU set and NIAH @ 4K/16K/64K.
6. Complete Phase B: `task-ac9-baseline`, `task-ac10-radix`, and `task-ac11-compare`.

### Blocking Side Issues

None active after Round 29.

### Queued Side Issues

- Align `benchmark_compare.py` with AC-11 before claiming AC-11: fixed seed, 120s warmup, 600s measurement, 3 trials, median aggregation, DS TPS >= 0.95 * DSA TPS, P99 TTFT <= 1.10 * DSA P99 TTFT.
- Replace shallow AC-8 prefix-match helper tests with coverage that exercises the actual smoke-harness gate.
- Fix stale `deepseek_v2.py` comments that still point at old `req_to_token_pool.size` slot authority.
- Fix stale `token_label_table.py` lifetime docs to describe invalidate-before-selection, not overwrite-before-read.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-30-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-30-summary.md

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
