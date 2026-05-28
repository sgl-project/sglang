# Code Review - Round 22

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-22-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 22 Summary

## Work Completed

Codex Round-21-review verified the token-denominator observability
fix but reopened the AC-8 smoke harness as blocking because two gate
bugs would reject a perfect paired DS/DSA run. Round 22 fixes both
and finishes the residual page→token rename in `benchmark_compare.py`.

### Fix 1 — prefix-match gate accepts short exact matches

`test/manual/test_dsv32_quality_smoke.py`:
- Removed the `len(dsa) >= self.PREFIX_MATCH_CHARS` guard from the
  prefix-match-hit condition. The AC-8 contract is "first 32 chars
  match", not "DSA answer must be at least 32 chars long".
- ~12 of the 20 smoke prompts deliberately request short outputs
  ("Output only X", "Give just Y"). Under the old guard, exact
  short DS/DSA matches like `Au == Au`, `1969 == 1969`, or
  `Jupiter == Jupiter` were counted as misses, breaking the 80%
  prefix-match gate even with perfect quality.

### Fix 2 — `_first_n_tokens_match` uses set intersection

`test/manual/test_dsv32_quality_smoke.py`:
- Replaced the zipped same-position scan
  (`any(at == bt for at, bt in zip(a_toks, b_toks))`) with set
  intersection (`bool(set(a_toks) & set(b_toks))`). The docstring
  always said "any overlap"; the implementation only detected
  positionally-aligned overlap. Shifted overlap like
  `"alpha beta gamma"` vs `"beta gamma alpha"` returned False even
  though every token is shared.

### Fix 3 — Helper-level regression tests

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
new class `TestDSv32SmokeHelpers` loads the manual smoke module via
`importlib.util.spec_from_file_location` (no `__init__.py` under
`test/manual/`) and exercises the helpers in CI:

- `test_prefix_match_accepts_short_exact_outputs` — `"Au" == "Au"`
  prefix hit.
- `test_prefix_match_rejects_short_different_outputs` — `"Au" != "Ag"`.
- `test_first_n_tokens_match_shifted_overlap_is_true` —
  `("alpha beta gamma", "beta gamma alpha", n=3)` → True.
- `test_first_n_tokens_match_no_overlap_is_false` — `("a b c", "x y z")` → False.

### Fix 4 — `benchmark_compare.py` page→token rename

Residual from Round 21's per-request rename. After Round 21 the
per-request publication used `selected_tokens`, but the comparator
still consumed/reported `selected_pages_mean` / `total_pages_mean` —
a stale naming conflict.

`development/benchmark_compare.py`:
- `RunMetrics.selected_pages_mean` → `selected_tokens_mean`.
- `RunMetrics.total_pages_mean` → `total_tokens_mean`.
- JSON consumer reads `selected_tokens_mean` / `total_tokens_mean`.
- Missing-field reporter, no-op-detector message
  (`selected_pages == total_pages` → `selected_tokens == total_tokens`),
  and report row labels updated.
- Existing unit tests at lines 2577/2579/2584/2585/2593/2595/2754/2756
  in `test_double_sparsity_unit.py` updated to pass / assert the new
  field names.

The bigger AC-11 work (3-trial median, DS TPS within 5% of DSA, P99
TTFT ≤ 1.10× DSA) is explicitly out of Round 22 scope and stays
queued for a future round.

## Files Changed

- `test/manual/test_dsv32_quality_smoke.py`: prefix-match condition +
  `_first_n_tokens_match` semantics + clarifying comments.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  new `TestDSv32SmokeHelpers` class (4 tests); renamed
  `selected_pages_mean` / `total_pages_mean` references in existing
  benchmark_compare tests.
- `development/benchmark_compare.py`: page→token rename across
  RunMetrics, JSON consumer, no-op detector, report rows.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
206 passed, 0 failed (was 202 before this round)

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_dsv32_quality_smoke.py -v
1 skipped — clean skip when env vars unset

pytest -v -k TestDSv32SmokeHelpers
4 passed
```

Manual helper sanity:
```
_first_n_tokens_match("alpha beta gamma", "beta gamma alpha", n=3) = True
_first_n_tokens_match("a b c", "x y z", n=3) = False
_rouge_l_f("Au", "Au") = 1.0
_rouge_l_f("Au", "Ag") = 0.0
```

Branch state: 23 commits ahead of `jimmy/dev/double-sparsity-standalone`.
Commit `931949f99` — [AC-8] Fix two AC-8 smoke gate bugs + finish R21 token rename.

## Remaining Items

Mainline AC items still requiring hardware execution:
- `task-ac8-server` + `task-ac8-quality` (harness is now correct; needs
  paired DS+DSA H200 servers + same-session reference run).
- `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`,
  `task-ac1b-probe`, `task-ac12-quality`.

Code-tier items still queued for future rounds:
- Replace `test/manual/test_double_sparsity_v32.py` skip-only scaffolds
  with real NIAH 4K/16K/64K + MMLU 5-shot execution against paired
  servers (AC-12).
- `serve_double_sparsity.sh` + `serve_native_nsa.sh` Option B flag
  alignment.
- `benchmark.sh` + `benchmark_baseline.sh` conc 16/32/64 sweep.
- `benchmark_compare.py` 3-trial median + AC-11 directional gate
  enforcement.
- Stale DS bind/runtime comments + token-label lifetime docs.

## Push-to-remote Status

Branch is 23 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`; commits remain local only. To enable per-round pushing,
re-launch with `--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Both gate bugs were standard documentation-vs-implementation
mismatches in test fixtures — narrow enough that a specific BitLesson
would not generalize usefully. The token rename was mechanical cleanup
after Round 21's primary rename, also too narrow. The existing
`BL-20260527-reshape-before-slice-mla` covers the general "re-check
derived names after rotation" framing.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-21-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-21-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-20-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-20-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-19-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-19-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-22-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
