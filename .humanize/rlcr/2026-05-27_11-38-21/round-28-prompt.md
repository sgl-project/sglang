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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-28-contract.md

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
# Round 27 Code Review

Mainline Progress Verdict: ADVANCED

Round 27 fixed the exact Round 26 parser defect and the default missing-directory MMLU data skip. The registered helper suite and combined registered tests pass. However, AC-12 is still not acceptance-ready: with paired server env vars set, the MMLU test can still skip before contacting either server when an existing `dev/` + `test/` tree contains no usable examples. The original Loop 4 plan also remains incomplete: H200 evidence gates, AC-10 radix, and AC-11 comparator work are still pending.

## Implementation Review

Verified Round 27 claims:
- `test/manual/test_double_sparsity_v32.py` now parses answer tokens conservatively. `_parse_mmlu_letter("Answer: B")` returns `"B"`, lowercase `"b"` returns `"B"`, and narrative text without an answer marker returns `None`.
- `_ensure_mmlu_data_dir` downloads/extracts the Hendrycks tarball when `data_dir/dev` and `data_dir/test` are missing, and raises `RuntimeError` for download/extraction/missing-subdir failures.
- `test_mmlu_5shot` no longer contains the Round 26 missing-directory `skipTest("MMLU data not found...")` branch.
- `test/registered/unit/manual/test_ac12_helpers.py` contains the claimed parser and data-prep regressions.

Validation I ran:

```text
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
38 passed, 1 warning

env -u DS_BASE_URL -u DSA_BASE_URL PYTHONPATH=python \
  python -m pytest test/manual/test_double_sparsity_v32.py -q
6 skipped, 1 warning

PYTHONPATH=python pytest \
  test/registered/unit/layers/attention/test_double_sparsity_unit.py \
  test/registered/unit/development test/registered/unit/manual -q
271 passed, 24 warnings
```

Counter-evidence:

```text
tmp=$(mktemp -d); mkdir -p "$tmp/dev" "$tmp/test"
DS_BASE_URL=http://127.0.0.1:1 DSA_BASE_URL=http://127.0.0.1:2 \
  AC12_MMLU_DATA_DIR="$tmp" PYTHONPATH=python python -m pytest \
  test/manual/test_double_sparsity_v32.py::TestDoubleSparsityV32Quality::test_mmlu_5shot -q

Result: 1 skipped
```

The remaining skip is at `test/manual/test_double_sparsity_v32.py:587-588`: after `_ensure_mmlu_data_dir` returns existing `dev/` and `test/` directories, an empty or malformed dataset produces no examples and the test calls `self.skipTest(...)`. Because the class-level server env vars are set in this path, this is still a silent bypass of the hard AC-12 MMLU gate.

## Goal Alignment Summary

```text
ACs: 13/15 addressed | Forgotten items: 0 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-9, AC-11, AC-12.

Not met: AC-1b, AC-10.

No original-plan task is forgotten in the tracker: the remaining work is still in Active Tasks or Queued Side Issues. The unjustified deferral remains `benchmark_compare.py` AC-11 being described as a future queued code-tier item even though final Loop 4 completion still requires the AC-11 comparator semantics and evidence. Hardware-gated work remains pending, not complete.

## Mainline Gaps

1. AC-12 MMLU can still silently skip under configured-server conditions.

Evidence:
- `_ensure_mmlu_data_dir` returns immediately if `data_dir/dev` and `data_dir/test` exist, without validating CSV presence or example usability.
- `test_mmlu_5shot` then skips at `test/manual/test_double_sparsity_v32.py:587-588` when `examples` is empty.
- The reproducer above sets both `DS_BASE_URL` and `DSA_BASE_URL` and still gets `1 skipped` before either dummy server is contacted.
- The refined plan says AC-12 is hard and the loop does not close without MMLU 5-shot delta <= 1 pp.

Required implementation plan:
1. Add a pure helper, for example `_load_mmlu_examples(dev_dir, test_dir, subjects, max_examples)`, that discovers subjects, requires paired `{subject}_dev.csv` and `{subject}_test.csv`, requires at least five dev rows, and returns the deterministic shuffled example list plus per-subject metadata.
2. In `test_mmlu_5shot`, call the helper after `_ensure_mmlu_data_dir`.
3. Replace `self.skipTest("MMLU data dir present but produced no usable examples")` with `self.fail(...)` when the class-level server env vars are set. The message must name `AC12_MMLU_DATA_DIR`, the resolved `data_dir`, and the required `dev/{subject}_dev.csv` / `test/{subject}_test.csv` layout.
4. Keep the only clean skip as the class-level skip when `DS_BASE_URL` or `DSA_BASE_URL` is unset.
5. Add a registered regression that creates empty `dev/` + `test/` directories, runs the configured-server path with mocked generation disabled or unreachable, and asserts the result is a failure, not a skip.
6. Run the full AC-12 harness on H200 after this fix: NIAH @ 4K/16K/64K, MMLU 5-shot, corrupt-mask sensitivity, and zero-signature sensitivity.

2. Remaining original-plan gates are still active and must not be treated as complete-by-deferral.

Pending original-plan tasks: `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`, and `task-ac11-compare`.

Required execution plan:
1. Generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200 with the AC-4 production command.
2. Run AC-1 forward population and AC-6 real V3.2 CUDA-graph capture/replay.
3. Run AC-1b chunked-prefill probe and record the pass/fail launch decision.
4. Run AC-8 DS server smoke and paired AC-8 quality smoke.
5. Generate AC-9 DSA baseline JSON.
6. Complete AC-10: run the M3-B radix hardware fixture, verify FP8 cold/warm scale stability, set the radix fixture flag only after evidence, and remove DS `--disable-radix-cache`.
7. Implement and run AC-11 comparator semantics: at least 3 trials per mode/concurrency, fixed seed, 120s warmup, 600s measurement, median aggregation, DS TPS >= 0.95 * DSA TPS, and DS P99 TTFT <= 1.10 * DSA P99 TTFT.

## Blocking Side Issues

1. MMLU data preflight is still incomplete.

Evidence:
- `test/manual/test_double_sparsity_v32.py:320-323` treats existing `dev/` and `test/` directories as sufficient data readiness.
- `test/manual/test_double_sparsity_v32.py:587-588` skips if no usable examples are produced.
- The configured-server reproducer above reports `1 skipped`.

Required fix:
1. Validate usable CSV examples, not just directory existence.
2. Fail loudly when servers are configured and no examples can be built.
3. Add a regression for the empty-existing-directory case.

## Queued Side Issues

- AC-8 prefix-match helper regressions still manually replicate the slicing expression instead of exercising the actual smoke-harness gate.
- Stale `deepseek_v2.py` comments still point at the old `req_to_token_pool.size` slot authority.
- Stale `token_label_table.py` lifetime text still describes overwrite-before-read instead of the Round 6 invalidate-before-selection invariant.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 27 Review`.
- Added a Round 27 Review evolution row with validation and counter-evidence.
- Updated `task-ac12-quality` notes to include the remaining empty/malformed MMLU data skip.
- Added a blocking side issue for MMLU data preflight still skipping when configured/prepared directories contain no usable examples.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-28-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-28-summary.md

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
