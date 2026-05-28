# Code Review - Round 30

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-30-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 30 Summary

## Work Completed

Codex Round 29 full goal alignment review found no Round 29 code
bug. The single piece of remaining code-tier mainline work the
plan still needed was the AC-11 directional comparator, which was
"queued only because AC-12/hardware evidence should be tackled
first." Hardware execution is outside my reach, so Round 30 lands
the AC-11 comparator.

Plan §AC-11 spec:

> Fixed seed, 600s window, 120s warmup, 3 trials, median.
> DS TPS within 5% of DSA TPS at conc=64 (directional gate).
> P99 TTFT ≤ DSA-on P99 TTFT × 1.10.

### Fix 1 — Pure helpers in `development/benchmark_compare.py`

- `_median(values)` — handles odd / even / single / None-filtered;
  returns `None` on empty or all-None.
- `_median_metrics(trials: List[RunMetrics]) -> RunMetrics` —
  per-field median across trials. `dense_fallback_total` is summed
  (counter, not sample). `concurrency` / `num_prompts` / `isl` /
  `osl` must agree across trials within one side — refuses
  otherwise so a misgrouped sweep cannot silently pass.
- `_group_by_concurrency(paths)` — groups trial JSONLs by
  resolved concurrency (per-row JSON preferred; `_c<N>.jsonl`
  filename suffix as fallback); raises `ValueError` when no
  resolution is possible.
- `_evaluate_ac11_gates(dsa_median, ds_median)` — computes
  `tps_ratio = ds_tps / dsa_tps` and `ttft_ratio = ds_ttft_p99 /
  dsa_ttft_p99`; gates: `tps_ratio >= 0.95` and `ttft_ratio <= 1.10`.
  Refuses on missing data or degenerate denominators with a
  `missing-data` reason.

Constants: `AC11_TPS_FLOOR_RATIO = 0.95`,
`AC11_TTFT_CEIL_RATIO = 1.10`, `AC11_MIN_TRIALS = 3`.

### Fix 2 — `--ac11` CLI mode

New flag `--ac11` plus arg lists `--ac11-baseline-results` and
`--ac11-ds-results` (each ≥ 3 paths per concurrency).
`_run_ac11_mode(args)`:

- Groups inputs by concurrency.
- Refuses if either side has < 3 trials at any concurrency, or if
  the concurrency sets disagree across sides, or if any per-trial
  operating-point invariant fires.
- For each concurrency, computes paired DSA and DS medians and
  evaluates both gates.
- Markdown report: per-concurrency
  `DSA TPS p50 | DS TPS p50 | TPS ratio | TPS gate | DSA TTFT p99
  | DS TTFT p99 | TTFT ratio | TTFT gate` table; final
  `AC-11 verdict: PASS|FAIL`. On any failure, the report lists each
  failing concurrency under a "Profiling obligation" header naming
  the failed gates + actual ratios.
- JSON report: `ac11_gates` constants + `per_concurrency` rows +
  top-level `verdict`.

Exit codes:
- `0` — all concurrencies pass both gates.
- `3` — at least one gate failed (profiling obligation triggered).
- `2` — input refusal (too few trials, mismatched concurrency
  sets, unresolvable concurrency, or `_median_metrics` invariant
  violation).

Module docstring updated to enumerate the two CLI modes
(single-trial AC-7/AC-8 legacy + new AC-11 directional).

### Fix 3 — Registered regressions

`test/registered/unit/development/test_ac11_comparator.py` (24
tests):

- `TestMedianHelper` × 6 — odd, even, single, None-filtered,
  empty → None, all-None → None.
- `TestMedianMetrics` × 3 — per-field medians, dense_fallback
  summed, concurrency-mismatch refusal.
- `TestGroupByConcurrency` × 3 — JSON-derived, filename fallback,
  unresolvable refusal.
- `TestEvaluateAC11Gates` × 6 — TPS pass at equality + at 0.95
  floor + fail below floor; TTFT pass at 1.10 ceiling + fail
  above; missing-data path marks both failed.
- `TestAC11EndToEnd` × 6 — full pass → exit 0 + Markdown +
  JSON output; TPS fail → exit 3 + obligation message; TTFT fail →
  exit 3; <3 trials → exit 2; concurrency-set mismatch → exit 2;
  legacy single-trial mode still works (no regression).

Loader uses `sys.modules["_bc"] = mod` before `exec_module` per
`BL-20260527-importlib-dataclass-sys-modules` so the `@dataclass`
decorators in `benchmark_compare.py` resolve correctly.

## Files Changed

- `development/benchmark_compare.py`: +4 pure helpers, +AC-11 CLI
  mode (`--ac11` + `--ac11-baseline-results` + `--ac11-ds-results`),
  +rendering + JSON output. Module docstring updated. Legacy
  single-trial path unchanged behaviorally.
- `test/registered/unit/development/test_ac11_comparator.py`:
  NEW — 24 registered regressions covering helpers and end-to-end
  CLI exit behavior.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
24 passed, 0 failed (new file)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
304 passed, 0 failed (was 280)
```

Sanity smoke (CLI happy path):
```
--ac11 + 3 DSA + 3 DS at conc=64 with paired DSA/DS TPS≈DS TPS,
P99 TTFT≈DSA TTFT → exit 0, "AC-11 verdict: PASS"

--ac11 + DS TPS = 50% of DSA → exit 3, "AC-11 verdict: FAIL",
"AC-11 TPS gate failed: DS/DSA = 0.5000 < 0.95", "Profiling obligation"
```

Commit: `00fdd6cb8` — [AC-11] Add 3-trial directional comparator
+ 24 CI regressions.

## Remaining Items

Code-tier items still queued for future rounds:

- Shallow AC-8 prefix-match regression coverage cleanup
  (Codex Round 22 queued item).
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

These are all minor; the loop's remaining mainline blockers are
purely hardware-gated now.

Hardware-gated tasks unchanged: `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac10-radix`, `task-ac11-compare` (comparator code-tier
landed — only the 3-trial H200 sweep + invocation remains),
`task-ac12-quality` (harness fully gate-tight — only the
H200 paired-server run remains).

## Push-to-remote Status

Branch is 31 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 30 applied three existing BitLessons:
`BL-20260527-shell-json-into-python-source` (sidecar JSON read via
the existing safe paths), `BL-20260527-conservative-llm-output-parser`
(fail-loud refusal rather than silent skip when AC-11 inputs are
malformed), `BL-20260527-importlib-dataclass-sys-modules` (loader
registers `_bc` in `sys.modules` before `exec_module`). No new
generalizable failure mode worth a fresh entry.
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
00fdd6cb8 [AC-11] Add 3-trial directional comparator + 24 CI regressions
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-29-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-29-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-28-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-28-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-27-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-27-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-30-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
