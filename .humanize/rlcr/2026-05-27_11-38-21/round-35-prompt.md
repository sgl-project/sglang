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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-35-contract.md

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
# Round 34 Goal Alignment Review

Mainline Progress Verdict: ADVANCED

Round 34 fixed the three Round 33 AC-11 code-tier blockers. I found no new blocking implementation bug in the touched AC-11 producer/comparator paths. The loop is still incomplete because multiple original-plan hardware / execution ACs remain active, and AC-11 execution still depends on AC-10 radix-cache parity.

## Part 1: Goal Tracker Audit

| AC | Status | Evidence if MET / PARTIAL | Blocker if NOT MET | Justification if DEFERRED |
|----|--------|----------------------------|--------------------|----------------------------|
| AC-0 | MET | Tracker completed/verified; full suite still passes (`356 passed, 26 subtests`). | - | - |
| AC-1 | PARTIAL | `task-m1-hook` call-site tests verified; tracker keeps `task-ac1-hwtest` active. | Real H200 forward population test and AC-8 selector-read smoke still pending. | - |
| AC-1b | NOT MET | - | Chunked-prefill probe pending after AC-6 hardware run. | - |
| AC-2 | MET | Tracker completed/verified stale-slot invalidation + live wiring. | - | - |
| AC-3 | MET | Tracker completed/verified logical-domain ownership and adapter isolation. | - | - |
| AC-4 | PARTIAL | Calibration coding task verified. | H200 mask generation/validation at `/models/dsv32-fp8-channel-mask.safetensors` pending. | - |
| AC-5 | MET | TP=2 multiprocess integration accepted in tracker. | - | - |
| AC-6 | PARTIAL | CUDA-graph code-tier task verified. | Real V3.2 conc=64 H200 capture/replay pending. | - |
| AC-7 | MET | First-decode-after-short-prefill proof accepted in tracker. | - | - |
| AC-8 | PARTIAL | Bench/smoke tooling exists. | 8xH200 DS server run and same-session quality smoke pending. | - |
| AC-9 | NOT MET | - | Option B DSA baseline JSON pending. | - |
| AC-10 | NOT MET | - | M3-B radix-cache fixture, FP8 scale stability check, guard flip, and `--disable-radix-cache` removal pending. | - |
| AC-11 | PARTIAL | Round 34 closes code-tier blockers: multi-epoch metrics, per-epoch `num_prompts`, side identity. | AC-10 plus H200 3-trial DSA/DS sweep and comparator invocation pending. | - |
| AC-12 | PARTIAL | Harness code-tier is tracker-verified. | Paired DS/DSA NIAH + MMLU H200 quality execution pending. | - |
| AC-13 | MET | Regression suite remains green in the broader validation run. | - | - |

Forgotten items: none functionally forgotten. The tracker covers the remaining original-plan work in Active Tasks, with earlier implementation subtasks aggregated into Completed/Verified AC rows and Plan Evolution entries.

Deferred items: none. The Explicitly Deferred section is empty, so there is no current deferral contradiction.

Goal completion summary:

```text
Acceptance Criteria: 6/15 met (0 deferred)
Active Tasks: 10 remaining
Estimated remaining rounds: 4-6 after H200 access, unbounded while hardware execution is unavailable
Critical blockers: H200 availability/execution; AC-10 radix-cache fixture before AC-11; AC-4 mask artifact; AC-12 quality gate
```

## Part 2: Mainline Drift Audit

The current round objective was clear and singular: close the three AC-11 producer/comparator defects from Round 33. Claude advanced the mainline rather than spending the round on unrelated cleanup. The recent AC-11 rounds show repeated artifact-validity bugs, but they are not circular: each review found a distinct falsifiable hole, and each subsequent round closed it with regressions.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 0
Queued Side Issues: 4
```

True blocking side issues: none separate from the active original-plan tasks. AC-10 is not a side issue; it is an active original-plan dependency for AC-11.

Queued side issues:
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.
- Leftover `Round 33 (AC-11)` comments in `development/benchmark.sh:81` and `development/benchmark_baseline.sh:81`.

## Part 3: Implementation Review

No high-signal blocking implementation issue found in Round 34.

Verified claims:
- `bench_serving` now passes replicated per-epoch input rows to `calculate_metrics` for non-multi-turn multi-epoch runs (`python/sglang/bench_serving.py:1560-1570`).
- JSONL `num_prompts` is per-epoch and total attempts are carried separately as `measured_num_prompts` (`python/sglang/bench_serving.py:1700-1738`).
- `_validate_ac11_side_identity` enforces DSA vs DS identity before cross-side normalization drops DS-only args (`development/benchmark_compare.py:645-710`, called at `development/benchmark_compare.py:1070-1100`).
- Plan AC-11 requires only DS enablement/config to differ and requires 120s warmup, 600s measurement, three trials, and median reporting (`development/loop4/refined_plan_v1.md:123-125`). The code-tier now supports that evidence path; execution remains pending.

Validation run:

```text
PYTHONPATH=python pytest test/registered/unit/development/test_bench_serving_timing.py -q
9 passed

PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
57 passed, 26 subtests passed

PYTHONPATH=python pytest test/registered/unit/development/test_option_b_scripts.py -q
22 passed

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py test/registered/unit/development test/registered/unit/manual -q
356 passed, 26 subtests passed
```

Non-blocking cleanup finding:
- The plan forbids round markers in implementation comments (`development/loop4/refined_plan_v1.md:364`). Round 34 removed them from `bench_serving.py` and `benchmark_compare.py`, but the benchmark scripts still have `Round 33 (AC-11)` duration-guard comments at `development/benchmark.sh:81` and `development/benchmark_baseline.sh:81`. This is comment-only drift and should not displace AC-10 or hardware execution.

## Part 4: Goal Tracker Update

I updated only the mutable section of `goal-tracker.md`:
- Plan Version now says `Updated: Round 34 Review`.
- Added a Round 34 Review Plan Evolution row recording that no new blocking AC-11 implementation gap was found.
- Added the leftover benchmark-script round-marker comments as a queued side issue.

No immutable Ultimate Goal or Acceptance Criteria text was modified.

## Part 5: Stagnation Check

Not stagnating. Rounds 31-34 stayed on AC-11 and repeatedly uncovered comparator/producer contract bugs, but Round 34 closed the previous reproducers and the validation suite is green. The next meaningful progress should move back to active original-plan execution, especially AC-10 and the H200 gates.

## Action Items

Mainline Gaps:
- Complete `task-ac10-radix`: M3-B hardware fixture, FP8 scale-factor stability check, flip `_double_sparsity_radix_fixture_passed`, remove `--disable-radix-cache`, update launcher contract tests.
- Run the pending H200 gates: `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`, `task-ac11-compare`, and `task-ac12-quality`.

Blocking Side Issues:
- None.

Queued Side Issues:
- Replace the two remaining benchmark-script `Round 33 (AC-11)` comments with neutral duration-guard wording when those scripts are next touched.
- Preserve the existing queued cleanup items for AC-8 prefix-match tests and stale DS comments/docs.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-35-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-35-summary.md

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
