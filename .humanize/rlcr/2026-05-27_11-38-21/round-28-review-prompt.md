# Code Review - Round 28

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-28-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 28 Summary

## Work Completed

Codex Round 27 review confirmed the parser + missing-directory
data-prep fixes from Round 27, but caught one remaining silent-skip
path:

```
tmp=$(mktemp -d); mkdir -p "$tmp/dev" "$tmp/test"
DS_BASE_URL=... DSA_BASE_URL=... AC12_MMLU_DATA_DIR="$tmp" \
  pytest test_mmlu_5shot
→ 1 skipped (should be 1 failed)
```

When `_ensure_mmlu_data_dir` saw existing `dev/`+`test/` directories,
it returned them as ready — even when the trees contained no usable
CSVs. The test loop then produced an empty `examples` list and fell
through to `self.skipTest("MMLU data dir present but produced no
usable examples")`, silently bypassing the hard AC-12 MMLU gate.

### Fix — `_load_mmlu_examples` pure validator

`test/manual/test_double_sparsity_v32.py`:

- New module-level helper `_load_mmlu_examples(dev_dir, test_dir,
  *, subjects=None, max_examples=200, seed=0xAC12)`:
  - Requires `dev_dir` and `test_dir` to be real directories.
  - Discovers subjects via `*_test.csv`.
  - Per subject: requires paired `{subject}_dev.csv` +
    `{subject}_test.csv`; requires ≥ 5 dev rows; requires test rows
    with ≥ 6 columns.
  - Rejects unusable subjects with a per-subject reason, collected
    into a single composite error message.
  - On no usable examples: raises `ValueError` with the resolved
    paths + rejection reasons + the expected layout.
  - Otherwise shuffles deterministically (`random.Random(seed)`),
    caps at `max_examples`, returns `(examples, per_subject_totals)`.
- Pure function — no unittest dependency, easy to CI-test.

`test_mmlu_5shot` now wraps the call in `try/except ValueError →
self.fail(...)`. The Round 27 silent
`skipTest("MMLU data dir present but produced no usable examples")`
branch is gone. The only acceptable skip remains the class-level
`@unittest.skipUnless` on `DS_BASE_URL` + `DSA_BASE_URL`.

### Verification of Codex's reproducer

```
tmp=$(mktemp -d); mkdir -p "$tmp/dev" "$tmp/test"
DS_BASE_URL=http://127.0.0.1:1 DSA_BASE_URL=http://127.0.0.1:2 \
  AC12_MMLU_DATA_DIR="$tmp" PYTHONPATH=python python -m pytest \
  test/manual/test_double_sparsity_v32.py::TestDoubleSparsityV32Quality::test_mmlu_5shot -v
```

→ `1 failed` with clear remediation message naming
`benchmark/mmlu/bench_sglang.py` + `AC12_MMLU_DATA_DIR`.

### Registered regressions (+7)

`test/registered/unit/manual/test_ac12_helpers.py`:

- `test_load_mmlu_examples_happy_path` — 5 dev rows + 3 test rows →
  3 examples, totals populated correctly.
- `test_load_mmlu_examples_raises_on_empty_test_dir` — Codex's
  reproducer scenario; ValueError "no subjects found".
- `test_load_mmlu_examples_raises_on_too_few_dev_rows` — 3 dev rows
  → ValueError "dev rows, need 5".
- `test_load_mmlu_examples_raises_on_malformed_test_rows` — test
  rows with 4 columns instead of 6 → ValueError "≥6 columns".
- `test_load_mmlu_examples_raises_on_missing_dev_csv` — test CSV
  present, dev CSV absent → ValueError "missing dev or test CSV".
- `test_load_mmlu_examples_deterministic_seed_and_cap` — same seed
  → same order; `max_examples=3` caps to exactly 3 examples.
- `test_load_mmlu_examples_explicit_subjects_filter` — explicit
  `subjects=["beta"]` returns only beta's examples; alpha excluded
  from `per_subject_totals`.

## Files Changed

- `test/manual/test_double_sparsity_v32.py`:
  - Added module-level `_load_mmlu_examples(...)`.
  - Replaced the inline subject-discovery loop in `test_mmlu_5shot`
    with a single call to the helper; converted the silent skip to
    `self.fail(...)`.
- `test/registered/unit/manual/test_ac12_helpers.py`:
  - +7 helper regressions exercising the loader's success path,
    every ValueError trigger, deterministic shuffle + cap,
    explicit subjects filter.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
45 passed, 0 failed (was 38)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
278 passed, 0 failed (was 271)

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_double_sparsity_v32.py -v
6 skipped — clean skip when env vars unset

# Codex's empty-dir reproducer now FAILS instead of SKIPS:
tmp=$(mktemp -d); mkdir -p "$tmp/dev" "$tmp/test"
DS_BASE_URL=http://127.0.0.1:1 DSA_BASE_URL=http://127.0.0.1:2 \
    AC12_MMLU_DATA_DIR="$tmp" python -m pytest \
    test/manual/test_double_sparsity_v32.py::TestDoubleSparsityV32Quality::test_mmlu_5shot -v
→ 1 failed (was 1 skipped)
```

Commit: `9d39f544e` — [AC-12] Close last MMLU silent-skip path via
`_load_mmlu_examples`.

## Remaining Items

Code-tier items queued for future rounds:

- `benchmark_compare.py` AC-11 directional gate (3-trial median,
  DS TPS ≥ 95% of DSA, P99 TTFT ≤ 1.10× DSA).
- Shallow AC-8 prefix-match regression coverage cleanup.
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality` (harness silent-skip paths
fully closed; hardware execution pending).

## Push-to-remote Status

Branch is 29 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: This round applied the existing
`BL-20260527-conservative-llm-output-parser` principle — "fail
loudly in eval harnesses, never silently skip" — to the data
preflight side of the same harness. The pattern is already
captured. No new entry warranted.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-27-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-27-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-26-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-26-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-25-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-25-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-28-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
