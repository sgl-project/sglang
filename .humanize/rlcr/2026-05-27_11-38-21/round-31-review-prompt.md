# Code Review - Round 31

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-31-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 31 Summary

## Work Completed

Codex Round 30 review caught three real AC-11 defects:

1. The comparator's gate evaluation ran on ratios alone — no
   sidecar reads, no cross-side workload/server-args checks, no
   warmup/window enforcement. A run with DSA `tp_size=8` /
   DS `tp_size=1`, mismatched `num_prompts`, missing `.meta.json`
   sidecars, etc. still published `AC-11 verdict: PASS`.
2. `_group_by_concurrency` swallowed JSON parse errors and let
   `_run_ac11_mode` re-trip them as an uncaught
   `JSONDecodeError` + exit 1 (instead of clean exit 2).
3. `benchmark.sh` / `benchmark_baseline.sh` overwrote
   `..._c${CONC}.jsonl` on every run — three trials produced one
   file, not three.

Round 31 closes all three with hard CI regressions.

### Fix 1 — Validation gauntlet (`development/benchmark_compare.py`)

New constants:

- `AC11_MIN_WARMUP_SECONDS = 120.0`
- `AC11_MIN_MEASUREMENT_WINDOW_SECONDS = 600.0`
- `_DS_ONLY_SERVER_ARG_KEYS = {enable_double_sparsity,
  double_sparsity_config, disable_radix_cache}` plus any
  `SGLANG_DS_FAULT_INJECT_*` flag — the only legitimate
  differences between DSA and DS server args.

New helpers:

- `_sidecar_path(p)` → `p + ".meta.json"`.
- `_read_ac11_meta(p)` raises `ValueError` on missing /
  malformed / non-object sidecar or non-null
  `server_args_error`.
- `_normalize_ac11_server_args(meta)` filters DS-only keys before
  cross-side comparison.
- `_validate_trial_metrics(metrics, side, path)` refuses trials
  missing `output_tps_p50` or `ttft_p99_s` (so `_median` can't
  turn 1 valid sample + 2 Nones into a passing 3-trial median).
- `_validate_meta_floors(meta, side, path)` refuses
  `warmup_seconds < 120` or `measurement_window_seconds < 600`.
- `_validate_per_side_agreement(metas, paths, side)` refuses
  any within-side seed / commit_sha / chunked_prefill_size /
  num_prompts / ISL / OSL / normalized_server_args disagreement.
- `_validate_cross_side_agreement(dsa_meta, ds_meta, conc)`
  refuses any cross-side seed / commit_sha / chunked / workload /
  server_args (after normalization) disagreement.
- Reuses existing `_match_or_refuse` for GPU/TP/page/concurrency on
  the JSONL context. Drops only the `disable_radix_cache`
  mismatch reason (AC-10 gap).

`_run_ac11_mode` now wraps the read passes in try/except so any
parse/refusal raises become clean exit 2 + log message naming the
side, trial path, and failing field. No tracebacks reach the
operator.

### Fix 2 — `_group_by_concurrency` crash fix

Stop swallowing `json.JSONDecodeError`; only fall back to filename
when the JSONL parsed cleanly but the parsed context lacks
concurrency. The second-read pass in `_run_ac11_mode` is also
wrapped so any parse / FileNotFound / refusal returns clean exit 2.

### Fix 3 — `_bench_meta_writer.py`: emit `warmup_seconds`

Added `warmup_seconds` field (float) from `WARMUP_SECONDS` env var.
`warmup_requests` retained for back-compat.

### Fix 4 — Benchmark scripts: 3-trial loop + non-overwriting filenames

`benchmark.sh` + `benchmark_baseline.sh`:

- `TRIALS="${TRIALS:-3}"`, `WARMUP_SECONDS="${WARMUP_SECONDS:-120}"`,
  `MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-600}"`.
- Outer trial loop `for TRIAL_ID in $(seq 1 "${TRIALS}")` writes
  `${MODE}_..._c${CONCURRENCY}_t${TRIAL_ID}.jsonl` (and matching
  `.meta.json`).
- Three trial runs produce three distinct files.
- Timing knobs reach `_bench_meta_writer.py` via env vars, so the
  comparator can refuse runs that don't meet AC-11 floors.

### Fix 5 — Registered regressions (+18 new)

`test_ac11_comparator.py` (+14):
- missing sidecar → exit 2;
- malformed JSON sidecar → exit 2;
- non-null `server_args_error` → exit 2;
- `warmup_seconds < 120` → exit 2;
- `measurement_window_seconds < 600` → exit 2;
- within-side seed mismatch → exit 2;
- cross-side seed mismatch → exit 2;
- cross-side commit_sha mismatch → exit 2;
- chunked_prefill_size mismatch → exit 2;
- num_prompts workload mismatch → exit 2;
- server_args tp_size mismatch → exit 2;
- DS-only differences (enable_double_sparsity + disable_radix_cache)
  still produce exit 0 (allowed via `_DS_ONLY_SERVER_ARG_KEYS`);
- missing `output_tps_p50` metric → exit 2;
- malformed `*_c64.jsonl` JSONL → exit 2 (no traceback — Codex's
  Round 30 review reproducer).

`test_option_b_scripts.py` (+4):
- both bench scripts default `TRIALS="${TRIALS:-3}"`;
- both loop `for TRIAL_ID in $(seq 1 "${TRIALS}")`;
- both filename include `_c${CONCURRENCY}_t${TRIAL_ID}.jsonl`;
- both default `WARMUP_SECONDS=120` + `MEASUREMENT_WINDOW_S=600`.

## Files Changed

- `development/benchmark_compare.py`: +AC-11 constants + 8 new
  validation helpers; `_group_by_concurrency` crash fix; full
  validation gauntlet wired into `_run_ac11_mode`.
- `development/_bench_meta_writer.py`: +`warmup_seconds` field.
- `development/benchmark.sh` + `development/benchmark_baseline.sh`:
  `TRIALS=3` outer loop, `_t${TRIAL_ID}.jsonl` filenames,
  `WARMUP_SECONDS`/`MEASUREMENT_WINDOW_S` defaults + pass-through.
- `test/registered/unit/development/test_ac11_comparator.py`:
  fixture helper grows `sidecar=` / `sidecar_overrides=` / `mode=`
  / `tp_size=` / `disable_radix_cache=` knobs; +14 validation
  regressions; existing 24 tests still pass.
- `test/registered/unit/development/test_option_b_scripts.py`:
  +4 script-contract regressions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
38 passed, 0 failed (was 24; +14)

PYTHONPATH=python pytest test/registered/unit/development -q
65 passed, 0 failed (was 47; +18)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
322 passed, 0 failed (was 304)

bash -n development/benchmark.sh         # OK
bash -n development/benchmark_baseline.sh # OK
```

Verified both Codex Round-30 reproducers:

```
# Mismatched DSA tp=8 / DS tp=1, no sidecars:
python development/benchmark_compare.py --ac11 \
    --ac11-baseline-results dsa_1.jsonl dsa_2.jsonl dsa_3.jsonl \
    --ac11-ds-results ds_1.jsonl ds_2.jsonl ds_3.jsonl
→ exit 2 with "AC-11 sidecar missing" + "AC-11 input refusal"
  (was exit 0 + "AC-11 verdict: PASS")

# Malformed *_c64.jsonl files:
python development/benchmark_compare.py --ac11 \
    --ac11-baseline-results malformed_t1_c64.jsonl ...
→ exit 2 + clean log (was uncaught JSONDecodeError + exit 1)
```

Commit: `732929181` — [AC-11] Comparator validation gauntlet
+ 3-trial scripts + crash fix.

## Remaining Items

Code-tier items queued for future rounds:

- Time-based warmup/measurement enforcement at the bench_serving CLI
  level (requires upstream CLI changes; the AC-11 comparator side
  enforces the floor via sidecar metadata).
- Shallow AC-8 prefix-match regression coverage cleanup
  (Codex Round 22 queued).
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

Hardware-gated execution unchanged: `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac10-radix`, `task-ac11-compare` (comparator + scripts now
plan-conformant; only the 3-trial H200 sweep + comparator
invocation remain), `task-ac12-quality` (harness fully gate-tight).

## Push-to-remote Status

Branch is 32 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 31 applied the existing
`BL-20260527-shell-json-into-python-source` lesson on the
benchmark-meta side (already implemented in Round 24) — the
comparator side just consumes the safe JSON the writer already
emits. The validation gauntlet itself is the kind of
"defense-in-depth" pattern that's better captured as code +
regressions than as a generalized BitLesson. No new entry warranted.
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
732929181 [AC-11] Comparator validation gauntlet + 3-trial scripts + crash fix
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-30-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-30-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-29-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-29-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-28-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-28-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-31-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
