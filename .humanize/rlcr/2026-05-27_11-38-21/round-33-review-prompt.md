# Code Review - Round 33

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-33-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 33 Summary

## Work Completed

Codex Round 32 review verified the four Round-31 reproducers are
closed but reopened AC-11 code-tier completeness with two gaps:

1. **Producer-side timing is still metadata-only.** Plan §AC-11 / DEC-2
   requires "120s warmup, 600s measurement window, 3 trials, median".
   Round 31's benchmark scripts only wrote those numbers into the
   sidecar; `bench_serving` had no `--warmup-seconds` /
   `--measurement-window-seconds` flag and consumed the finite request
   generator once. AC-11 evidence could not be produced.
2. **Server-args comparison silently under-compared launch flags.**
   Round 32's `_AC11_STABLE_LAUNCH_ARG_KEYS` was a 12-key hand-list;
   Codex's reproducer with `disable_cuda_graph=false` on DSA vs
   `disable_cuda_graph=true` on DS exited 0 with `AC-11 verdict: PASS`.

Round 33 closes both with hard CI regressions.

### Fix 1 — bench_serving seconds-based warmup + measurement window

`python/sglang/bench_serving.py`:

- New CLI flags `--warmup-seconds` and `--measurement-window-seconds`
  (default 0 — legacy single-pass behavior preserved when both unset).
- Refactored the request-dispatch loop (formerly lines 1348–1417) into
  a reusable inner closure `_dispatch_workload_once(with_pbar=...) ->
  (outputs, elapsed_s)`. The closure captures the outer-scope
  `backend`, `request_func`, `limited_request_func`, `input_requests`,
  `request_rate`, mooncake config, LoRA selection, extra request body,
  and progress-bar controls.
- **Seconds-based warmup**: while `elapsed < warmup_seconds`, run
  `_dispatch_workload_once(with_pbar=False)` and discard outputs.
  Before the first measured epoch, re-seed `random.seed(args.seed)`
  and `np.random.seed(args.seed)` so warmup does not perturb the
  measured request-arrival process.
- **Seconds-based measurement**: while `accumulated < window`, run
  `_dispatch_workload_once(...)` and extend the outputs list with each
  epoch's results. `benchmark_duration` is the accumulated wall-clock
  across epochs.
- **JSONL workload fields**: every result now carries `num_prompts` /
  `input_len` / `output_len` / `warmup_seconds` /
  `measurement_window_seconds` / `measured_epochs`. For
  `generated-shared-prefix` the effective ISL is
  `gsp_system_prompt_len + gsp_question_len`; for `random*` it is
  `args.random_input_len`. The comparator now cross-checks these
  against the sidecar's `num_prompts` / `isl_total_tokens` /
  `osl_tokens`.

### Fix 2 — Benchmark scripts pass timing flags + refuse short duration

`development/benchmark.sh` + `development/benchmark_baseline.sh`:

- Pass `--warmup-seconds "${WARMUP_SECONDS}"` and
  `--measurement-window-seconds "${MEASUREMENT_WINDOW_S}"` to
  `python3 -m sglang.bench_serving` (kept defaults: 120 / 600).
- After each trial, parse the JSONL `duration` field; if it is below
  `MEASUREMENT_WINDOW_S`, print `FATAL: ... refusing to publish
  AC-11 artifact` and exit non-zero. Catches the case where
  bench_serving's time-based loop itself bailed out early (e.g.
  workload epoch wall-clock > window so the loop ran exactly one
  epoch that was too short).

### Fix 3 — Comparator: full stable ServerArgs projection

`development/benchmark_compare.py`:

- Replaced the 12-key hand-list with the full set derived from
  `dataclasses.fields(sglang.srt.server_args.ServerArgs)` at module
  load (386 fields). Import failure raises a clear `RuntimeError`
  pointing at `PYTHONPATH=python` / `pip install -e python/` — the
  comparator cannot do its job without the schema and silent
  under-projection is exactly what Codex's review caught.
- `_normalize_ac11_server_args` projects `server_args` onto the full
  set and excludes only `_DS_ONLY_SERVER_ARG_KEYS`
  (`{enable_double_sparsity, double_sparsity_config}`). Dynamic
  `/server_info` telemetry (`internal_states`, `kv_events`,
  `last_gen_throughput`, `gpu_memory_used_bytes`, scheduler capacity,
  `step_time`, future endpoint additions, …) is dropped by
  construction — none of those keys are `ServerArgs` fields.
- New `_AC11_OPTION_B_LOCKED_FIELDS` + `_require_option_b_locked_fields`
  enforce per plan §13 / DEC-1 that every locked launch flag is
  present in the normalized projection: `model_path`, `tp_size`,
  `page_size`, `kv_cache_dtype`, `dsa_prefill_backend`,
  `dsa_decode_backend`, `disable_overlap_schedule`,
  `disable_piecewise_cuda_graph`, `disable_radix_cache`,
  `disable_cuda_graph`. Missing field → exit 2.
- Cross-side mismatch error names the differing fields with per-side
  values so the operator sees the diff immediately.

### Fix 4 — JSONL workload cross-check

`_validate_jsonl_workload_matches_sidecar(metrics, meta, side, path)`:
when the JSONL surfaces `num_prompts` / `input_len` / `output_len`,
they must equal the sidecar's `num_prompts` / `isl_total_tokens` /
`osl_tokens`. The sidecar is written from env vars; the JSONL is the
canonical record of what actually ran — a disagreement means the
sidecar is lying about the workload.

### Fix 5 — Test regressions (+16 named + 19 new subTests, +1 new file)

`test/registered/unit/development/test_ac11_comparator.py`:

- `test_server_args_non_ds_launch_flag_mismatch_refused` — subTest
  matrix over `disable_cuda_graph`, `trust_remote_code`, `dtype`,
  `max_total_tokens`, `attention_backend`, `mem_fraction_static` →
  each → exit 2 (closes Codex's `disable_cuda_graph` PASS-by-default
  reproducer).
- `test_server_args_locked_option_b_field_missing_refused` — subTest
  matrix over the 10 locked Option B fields → each → exit 2.
- `test_jsonl_workload_disagrees_with_sidecar_refused` — subTest
  matrix over `num_prompts` / `input_len` / `output_len` → each →
  exit 2.
- Updated `test_dynamic_server_info_drift_does_not_self_refuse` with
  the full Option B sidecar + wider dynamic telemetry
  (`scheduler_load`, `future_unknown_endpoint_field`); proves
  whitelist ignores non-ServerArgs telemetry while real launch-flag
  mismatch still refuses.
- Fixture (`_write_bench_jsonl` + `_make_trials` + new `_option_b_sa`
  helper) emits the full Option B `server_args` in every sidecar.

`test/registered/unit/development/test_option_b_scripts.py`:

- `test_{ds,dsa}_bench_passes_warmup_seconds_flag` — scripts must
  pass `--warmup-seconds`.
- `test_{ds,dsa}_bench_passes_measurement_window_seconds_flag`.
- `test_{ds,dsa}_bench_fails_on_short_observed_duration` — scripts
  must inspect the JSONL `duration` and refuse to publish if it falls
  below `MEASUREMENT_WINDOW_S`.

`test/registered/unit/development/test_bench_serving_timing.py` (NEW):

- `test_cli_exposes_warmup_seconds_flag` /
  `test_cli_default_zero_preserves_legacy_path` — `--help` shows both
  new flags.
- `test_seconds_warmup_runs_multiple_discarded_epochs` — mock-driven:
  50ms warmup with 5ms requests runs ≥ 2 epochs; warmup outputs do
  not appear in the measured result.
- `test_seed_reset_between_warmup_and_measured_phases` — wraps
  `random.seed` + `np.random.seed` with counters; asserts ≥ 1 call to
  each between warmup and measured phase.
- `test_measurement_window_loop_accumulates_outputs` — mock-driven:
  40ms window with 3ms epochs accumulates ≥ 8 outputs and
  `duration >= 0.04`.
- `test_legacy_single_pass_when_both_seconds_unset` — both flags 0 →
  exactly one epoch, 4 outputs.
- `test_jsonl_includes_workload_triple` — output JSONL carries
  `num_prompts` / `input_len` / `output_len`; for gsp dataset
  `input_len = gsp_system_prompt_len + gsp_question_len`.

## Files Changed

- `python/sglang/bench_serving.py`: +`--warmup-seconds` /
  `--measurement-window-seconds` flags;
  `_dispatch_workload_once` closure; seconds-based warmup with seed
  reset; seconds-based measurement loop; new JSONL fields
  `num_prompts` / `input_len` / `output_len` /
  `warmup_seconds` / `measurement_window_seconds` /
  `measured_epochs`. Net +284 / -86 lines.
- `development/benchmark.sh` + `benchmark_baseline.sh`: +new flags
  + post-trial duration guard. +18 lines each.
- `development/benchmark_compare.py`: full `ServerArgs` projection;
  `_AC11_OPTION_B_LOCKED_FIELDS` + `_require_option_b_locked_fields`;
  `_validate_jsonl_workload_matches_sidecar`. +154 / -64 lines.
- `test/registered/unit/development/test_ac11_comparator.py`: new
  `_option_b_sa` helper; full Option B fixture; +3 named tests with
  19 new subTests; updated dynamic-drift test. +186 / -33 lines.
- `test/registered/unit/development/test_option_b_scripts.py`: +6
  new script-contract regressions. +45 lines.
- `test/registered/unit/development/test_bench_serving_timing.py`
  (NEW): +7 mock-driven timing tests. +400 lines.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
51 passed, 26 subtests passed (was 48 + 7)

PYTHONPATH=python pytest test/registered/unit/development/test_option_b_scripts.py -q
22 passed (was 16; +6)

PYTHONPATH=python pytest test/registered/unit/development/test_bench_serving_timing.py -q
7 passed (new)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
348 passed, 26 subtests passed (was 332 + 7)

PYTHONPATH=python pytest test/registered/integration/test_double_sparsity_tp_multiprocess.py -q
3 passed (unchanged)
```

Verified Codex's `disable_cuda_graph` reproducer:

```
# DSA disable_cuda_graph=False, DS disable_cuda_graph=True, all other
# fields equal, valid JSONL duration:
python development/benchmark_compare.py --ac11 ...
→ exit 2 with
  "normalized launch-args server_args differ between DSA and DS on
   ['disable_cuda_graph']: [('disable_cuda_graph', False, True)].
   Only ['double_sparsity_config', 'enable_double_sparsity'] may
   differ between sides (plan §AC-11)."
(was exit 0 + "AC-11 verdict: PASS")
```

`PYTHONPATH=python python -m sglang.bench_serving --help` shows both
new flags:

```
  --warmup-seconds WARMUP_SECONDS
  --measurement-window-seconds MEASUREMENT_WINDOW_SECONDS
```

Commit: `4fcbb40ce` — [AC-11] real bench_serving warmup/window
enforcement + full ServerArgs projection.

## Remaining Items

Code-tier items queued for future rounds:

- AC-10 radix-cache fixture (`task-ac10-radix`) — must close before
  the H200 AC-11 sweep can run with DS launcher dropping
  `--disable-radix-cache`.
- Shallow AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

Hardware-gated execution unchanged: `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac10-radix`, `task-ac11-compare` (comparator + scripts +
bench_serving now plan-conformant; AC-10 must close before the
H200 3-trial sweep can run with DS radix-cache parity-ON),
`task-ac12-quality`.

## Push-to-remote Status

Branch is 34 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 33 applied the existing
`BL-20260527-shell-json-into-python-source` lesson on the
benchmark-script side (the new duration guard inspects the JSONL
via a small Python subprocess that uses `json.loads`, not heredoc
splicing). The bench_serving CLI refactor + comparator widening
are "defense-in-depth + schema-derivation" patterns that are better
captured as code + regressions than as a generalized BitLesson. No
new entry warranted.
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
48d6497b1 [AC-11] tighten comparator: JSONL duration floor, required-field sidecars, radix parity, launch-args whitelist
4fcbb40ce [AC-11] real bench_serving warmup/window enforcement + full ServerArgs projection
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-32-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-32-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-31-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-31-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-30-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-30-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-33-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
