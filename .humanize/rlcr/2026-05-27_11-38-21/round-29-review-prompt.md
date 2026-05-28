# FULL GOAL ALIGNMENT CHECK - Round 29

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 29 Summary

## Work Completed

Codex Round 28 review caught two more AC-12 harness defects:

1. `test_mmlu_5shot` opened with
   `try: import pandas as pd except ImportError: self.skipTest(...)`.
   That runs INSIDE the test, after the class-level `skipUnless` has
   already passed — so when servers are configured but pandas isn't
   installed, the gate silently skips. Round 28's contract had
   explicitly promised "the only acceptable MMLU skip is the
   class-level server-env skip", but the pandas guard slipped through.
2. Round 28 renamed `subjects → subjects_arg` but left
   `"subjects": subjects` in the `_record_artifact` payload, so
   `AC12_MMLU_SUBJECTS=beta` crashed with
   `NameError: name 'subjects' is not defined` AFTER both eval
   loops. The Round 28 registered regression tested
   `_load_mmlu_examples` directly with the `subjects=["beta"]` kwarg
   but never exercised the full harness path.

### Fix 1 — drop pandas; use stdlib csv

`test/manual/test_double_sparsity_v32.py`:

- Removed the `try: import pandas / self.skipTest(...)` guard from
  `test_mmlu_5shot`. The harness now has no non-env silent-skip
  paths.
- `_load_mmlu_examples` no longer imports pandas. Added an inline
  `_read_csv_rows(path)` helper using `csv.reader(open(path,
  newline=""))` that returns `List[List[str]]`. Drop-in replacement
  for the prior `pd.read_csv(...).values.tolist()` semantics.

### Fix 2 — `subjects` NameError

Defined `subjects_for_artifact = subjects_arg if subjects_arg is
not None else "all"` immediately after env parsing. The
`_record_artifact` payload now uses `subjects_for_artifact`. The
undefined `subjects` reference is gone.

### Fix 3 — Registered regressions (+2)

`test/registered/unit/manual/test_ac12_helpers.py`:

- `test_load_mmlu_examples_works_without_pandas`:
  - Pops `sys.modules["pandas"]` if present.
  - Monkeypatches `builtins.__import__` to raise
    `ImportError("simulated absence of pandas")` for any `pandas`
    import.
  - Builds a tiny valid CSV tree with one subject.
  - Calls `_load_mmlu_examples(...)` and asserts 2 examples
    returned with the right subject totals. No `SkipTest` or
    `ImportError` propagates.
  - Restores the original `__import__` and `sys.modules["pandas"]`
    in `finally` to avoid polluting other tests.

- `test_mmlu_5shot_subjects_filter_does_not_crash`:
  - Builds alpha + beta subjects under a temp `AC12_MMLU_DATA_DIR`.
  - Sets `AC12_MMLU_SUBJECTS=beta` + `DS_BASE_URL` + `DSA_BASE_URL`.
  - Mocks `_generate` to return the gold answer ("A") for every
    prompt so the |DSA - DS| gate passes.
  - Mocks `_record_artifact` to capture payloads.
  - Crucial: temporarily clears the class-level
    `__unittest_skip__` + `__unittest_skip_why__` attributes — the
    `@skipUnless` decorator evaluates its condition at module
    import time, so `patch.dict(os.environ)` alone doesn't unskip
    the class. Restores both in `finally`.
  - Runs the test via `unittest.TextTestRunner`. Asserts:
    - No errors and no failures (the gate path completed without
      `NameError`).
    - The recorded MMLU artifact has `subjects == ["beta"]`.
    - `alpha` does NOT appear in `dsa_per_subject` totals (the
      explicit filter actually narrowed evaluation).

## Files Changed

- `test/manual/test_double_sparsity_v32.py`:
  - `_load_mmlu_examples`: removed `import pandas`; inlined
    `_read_csv_rows(path)` stdlib-csv helper; updated CSV reads
    accordingly.
  - `test_mmlu_5shot`: removed the pandas import-guard skipTest
    block; defined `subjects_for_artifact`; replaced the undefined
    `subjects` reference in the recorder.
- `test/registered/unit/manual/test_ac12_helpers.py`:
  - +2 new regressions described above.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
47 passed, 0 failed (was 45)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
280 passed, 0 failed (was 278)

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_double_sparsity_v32.py -v
6 skipped — clean skip when env vars unset (the ONLY remaining skip path)
```

Round 28's empty-dir Codex reproducer still reports `1 failed`
(unchanged regression from Round 28). Round 28's subjects-filter
NameError is fixed; Round 28's pandas SkipTest is fixed.

Commit: `90942a402` — [AC-12] Drop pandas + fix subjects NameError;
harness fully gate-tight.

## Remaining Items

Code-tier items queued for future rounds:

- `benchmark_compare.py` AC-11 directional gate (3-trial median,
  DS TPS ≥ 95% of DSA, P99 TTFT ≤ 1.10× DSA).
- Shallow AC-8 prefix-match regression coverage cleanup.
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality` (harness fully gate-tight;
hardware execution pending).

## Push-to-remote Status

Branch is 30 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 29 applied the existing
`BL-20260527-conservative-llm-output-parser` principle ("fail loud
in eval harnesses, never silently skip") to two more sub-bugs in
the same harness. The general principle is already captured. The
specific `@unittest.skipUnless` class-decorator caching pattern
that bit Codex's subjects-filter integration test (forcing the
`__unittest_skip__` attribute clear in the regression) is too
narrow to generalize — it shows up whenever someone needs to
exercise an env-gated test class under monkeypatched env vars.
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
287a58231 [AC-4] Fix 3-D hook outputs, implement Pile-val token-block recipe, update calibration doc
104fedcab [AC-4] Fix V3.2 qk_rope_head_dim derivation; add --kv-cache-dtype for mask dtype
6cf32a884 [AC-5] TP=2 multiprocess all-reduce harness for Double Sparsity
cd7c071f3 [AC-6] Thread req_to_token through capture_decode_step; add AC-6 unit tests
0ce54a98d [AC-6] Fix CUDA graph capture-safety: remove host sync + add graph-safe API
18b22b5e3 [AC-6] Truly allocation-free CUDA graph-safe selector
38ef74765 [AC-6] Wire allocation-free DS path into production + production dtypes
8ab3c332a [AC-6] Resolve DS metadata via ForwardContext + int64 input dtypes
5c636760f [AC-6] Resolve ds_topk_indices_out via ForwardContext (mirror Round 19)
a586f814a [AC-8] Quality smoke harness + token-denominator observability fix
931949f99 [AC-8] Fix two AC-8 smoke gate bugs + finish R21 token rename
3ab86e868 [AC-8/9] Align Option B launchers + benchmark sweeps to plan §13
856ab7356 [AC-8/9] Fix Round 23 sidecar JSON injection bug; add chunked_prefill_size + AC-11 fields
0ae955cf2 [AC-12] Replace skip-only scaffold with real NIAH + MMLU + sensitivity harness
2ad8b9ee3 [AC-12] Real MMLU 5-shot + fault-injection gates + harness fixes
faa41438e [AC-12] Fix MMLU answer parser + auto-download Hendrycks data
9d39f544e [AC-12] Close last MMLU silent-skip path via _load_mmlu_examples
90942a402 [AC-12] Drop pandas + fix subjects NameError; harness fully gate-tight
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-28-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-28-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-27-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-27-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-26-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-26-review-result.md


Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Goal Tracker Audit (MANDATORY)

Read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md and verify:

### 1.1 Acceptance Criteria Status
For EACH Acceptance Criterion in the IMMUTABLE SECTION:
| AC | Status | Evidence (if MET) | Blocker (if NOT MET) | Justification (if DEFERRED) |
|----|--------|-------------------|---------------------|----------------------------|
| AC-1 | MET / PARTIAL / NOT MET / DEFERRED | ... | ... | ... |
| ... | ... | ... | ... | ... |

### 1.2 Forgotten Items Detection
Compare the original plan (@development/loop4/refined_plan_v1.md) with the current goal-tracker:
- Are there tasks that are neither in "Active", "Completed", nor "Deferred"?
- Are there tasks marked "complete" in summaries but not verified?
- List any forgotten items found.

### 1.3 Deferred Items Audit
For each item in "Explicitly Deferred":
- Is the deferral justification still valid?
- Should it be un-deferred based on current progress?
- Does it contradict the Ultimate Goal?

### 1.4 Goal Completion Summary
```
Acceptance Criteria: X/Y met (Z deferred)
Active Tasks: N remaining
Estimated remaining rounds: ?
Critical blockers: [list if any]
```

## Part 2: Mainline Drift Audit (MANDATORY)

Determine whether the recent rounds are still serving the original plan:
- Is the current round's mainline objective clear and singular?
- Has Claude been advancing mainline ACs, or mostly clearing side issues?
- Which findings are true **blocking side issues** versus merely **queued side issues**?

Include a short drift summary:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
Blocking Side Issues: N
Queued Side Issues: N
```

The `Mainline Progress Verdict` line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

## Part 3: Implementation Review

- Conduct a deep critical review of the implementation
- Verify Claude's claims match reality
- Identify any gaps, bugs, or incomplete work
- Reference @docs for design documents

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

## Part 5: Progress Stagnation Check (MANDATORY for Full Alignment Rounds)

To implement the original plan at @development/loop4/refined_plan_v1.md, we have completed **30 iterations** (Round 0 to Round 29).

The project's `.humanize/rlcr/2026-05-27_11-38-21/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-27_11-38-21/round-28-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-27-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-28-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-29-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-29-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
