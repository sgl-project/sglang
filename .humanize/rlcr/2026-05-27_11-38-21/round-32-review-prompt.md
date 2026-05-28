# Code Review - Round 32

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-32-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 32 Summary

## Work Completed

Codex Round 31 review verified that the validation gauntlet (sidecar
reads, per-side / cross-side agreement, JSONL malformed-input crash fix,
3-trial scripts) all landed correctly, but reopened AC-11 code-tier
completeness with four remaining defects:

1. **Mainline Gap 1** — Timing floors were sidecar-only. A JSONL with
   `duration: 5` and a sidecar claiming `measurement_window_seconds:
   600` still passed.
2. **Mainline Gap 2** — Sidecar validation treated missing required
   fields as agreement under `None == None`. Two sidecars omitting
   `seed` / `commit_sha` / `server_args` compared equal.
3. **Mainline Gap 3** — Radix mismatch was allowed via
   `_DS_ONLY_SERVER_ARG_KEYS`, but plan §AC-11 says only the DS
   enablement pair may differ; AC-11 depends on AC-10.
4. **Blocking Side Issue** — The raw `/get_server_info` payload (with
   `internal_states.last_gen_throughput` and other dynamic telemetry)
   was compared per-trial, so script-generated sequential trials could
   self-refuse on inert drift.

Round 32 closes all four with hard CI regressions.

### Fix 1 — JSONL `duration` floor (`development/benchmark_compare.py`)

`RunMetrics` gained a `duration_s: Optional[float]` field;
`_read_bench_jsonl` extracts `summary["duration"]` (bench_serving emits
this as the wall-clock duration of the measured phase).

New validator `_validate_jsonl_duration(metrics, side, path)`:

- refuses `duration_s is None` ("JSONL is missing the `duration` field;
  cannot verify the AC-11 measurement window"),
- refuses `duration_s < AC11_MIN_MEASUREMENT_WINDOW_SECONDS` ("wall-clock
  duration=Xs is below the AC-11 measurement-window floor of 600.0s").

Wired into `_run_ac11_mode` per trial, alongside `_validate_trial_metrics`
and `_validate_meta_floors`. The Codex reproducer (`duration=5` + sidecar
`measurement_window_seconds=600`) now exits 2 with a clear message.

### Fix 2 — Required-field sidecar validator

New helper `_require_sidecar_fields(meta, side, path)` enforces presence
and well-typedness of:

- `seed`: must be an `int` (and not a `bool`).
- `commit_sha`: must be a non-empty `str`, not `"unknown"` or `""`.
- `chunked_prefill_size`: must be a positive `int`, OR the string
  `"unknown"` (the `_bench_meta_writer.py` fallback when
  `/server_info` doesn't expose the knob).
- `num_prompts`, `isl_total_tokens`, `osl_tokens`: positive `int`.
- `server_args`: non-empty `dict`.

Called inside `_read_ac11_meta` before returning, so every later
agreement check operates on real values. `_read_ac11_meta` now takes
`side: str = "?"` so refusal messages can name "DSA" vs "DS".

`_run_ac11_mode` also asserts `meta["concurrency"] == grouping_conc`
per trial (catches sidecar/file-rename mishaps).

### Fix 3 — Remove radix-cache allowance

- `_DS_ONLY_SERVER_ARG_KEYS` shrunk to `{enable_double_sparsity,
  double_sparsity_config}` (was `{... , disable_radix_cache}`).
- The `hw_reasons = [r for r in hw_reasons if "disable_radix_cache"
  not in r]` post-filter in `_run_ac11_mode` is gone.

AC-11 now refuses radix mismatch on BOTH the launch-args server_args
path (via `_validate_cross_side_agreement` → `_normalize_ac11_server_args`)
AND the JSONL `RunContext` path (via `_match_or_refuse`).

Production note: AC-11 still requires AC-10 to close before the DS
launcher can drop `--disable-radix-cache`. The Round 31 launchers
(DSA: radix on, DS: radix off) now correctly refuse AC-11.

### Fix 4 — Stable launch-args whitelist

`_normalize_ac11_server_args` was a blocklist (`{k: v for k, v in sa.items()
if k not in _DS_ONLY_SERVER_ARG_KEYS and not k.startswith(
"SGLANG_DS_FAULT_INJECT_")}`). It now projects onto a fixed launch-args
whitelist:

```python
_AC11_STABLE_LAUNCH_ARG_KEYS = frozenset({
    "tp_size", "page_size", "model_path",
    "chunked_prefill_size", "dsa_prefill_backend", "dsa_decode_backend",
    "disable_overlap_schedule", "disable_piecewise_cuda_graph",
    "kv_cache_dtype",
    "enable_double_sparsity", "double_sparsity_config",
    "disable_radix_cache",
})
```

Dynamic `/get_server_info` telemetry (`internal_states`, `kv_events`,
`last_gen_throughput`, `gpu_memory_used_bytes`, `step_time`, …) is
dropped — sequential trials no longer self-refuse on inert drift. The
whitelist is schema-safe across future sglang versions, where a
blocklist would re-leak every new dynamic field.

### Fix 5 — `_filename_concurrency` recognizes `_c<N>_t<M>.jsonl`

Regex updated from `r"_c(\d+)\.jsonl$"` to
`r"_c(\d+)(?:_t\d+)?\.jsonl$"`. Round 31's three-trial sweep filenames
(e.g. `dsa_c64_t2.jsonl`) now resolve via the filename fallback when
the JSONL row lacks `max_concurrency`/`concurrency`. The legacy
`_c64.jsonl` form still works.

### Fix 6 — Test fixtures + 10 new validation regressions

`test/registered/unit/development/test_ac11_comparator.py`:

- `_write_bench_jsonl` gains `duration: float = 600.0` (AC-11 floor), so
  existing tests pass the JSONL duration validator. Tests that exercise
  it pass `duration=5.0` or `extra={"duration": None}` (writer pops the
  key for `None`).
- Sidecar `server_args` now always carries `disable_radix_cache`
  (matching real `/get_server_info`). The DS-mode block only adds
  `enable_double_sparsity` + `double_sparsity_config`.
- New `_OMIT = object()` sentinel: `sidecar_overrides={"seed": _OMIT}`
  removes the field entirely so missing-field validation can be
  exercised.
- `_make_trials` defaults both sides to `disable_radix=True` (radix-OFF
  parity, the pre-AC-10 state where the DS launcher passes
  `--disable-radix-cache`). The Round 32 comparator requires radix-cache
  parity, so the default fixture must keep it. Tests that exercise the
  mismatch refusal pass `disable_radix` explicitly.
- `test_workload_num_prompts_mismatch_exit_2` updated to keep radix
  parity so the refusal reason is the intended `num_prompts` mismatch,
  not a side-effect radix mismatch.
- `test_allowed_ds_only_differences_still_pass` doc-comment updated to
  the Round-32 semantics ("only `enable_double_sparsity` +
  `double_sparsity_config` may differ").

New regressions (10 named + 7 parameterized subTests = +17 cases):

- `test_short_jsonl_duration_refused` — JSONL `duration=5` + sidecar
  `measurement_window_seconds=600` → exit 2 (Codex Mainline Gap 1).
- `test_missing_jsonl_duration_refused` — JSONL has no `duration` key
  → exit 2.
- `test_radix_mismatch_refused` — DSA radix ON / DS radix OFF
  → exit 2 (Codex Mainline Gap 3).
- `test_missing_required_sidecar_field_refused` — subTest matrix over
  `seed` / `commit_sha` / `chunked_prefill_size` / `num_prompts` /
  `isl_total_tokens` / `osl_tokens` / `server_args` → each → exit 2
  (Codex Mainline Gap 2).
- `test_server_args_empty_dict_refused` — `server_args = {}` → exit 2.
- `test_commit_sha_unknown_refused` — `commit_sha = "unknown"` → exit 2.
- `test_commit_sha_empty_string_refused` — `commit_sha = ""` → exit 2.
- `test_sidecar_concurrency_mismatch_refused` — sidecar `concurrency=32`
  while JSONL/filename concurrency=64 → exit 2.
- `test_dynamic_server_info_drift_does_not_self_refuse` — 3 trials per
  side whose only difference is per-trial dynamic telemetry
  (`internal_states[0].last_gen_throughput`, `kv_events`,
  `last_gen_throughput`, `gpu_memory_used_bytes`, `step_time`) →
  exit 0 (blocking side issue from Codex Round 31).
- `test_chunked_prefill_size_unknown_string_allowed_when_consistent` —
  both sides have sidecar `chunked_prefill_size = "unknown"` (the
  `_bench_meta_writer.py` fallback) → exit 0.

## Files Changed

- `development/benchmark_compare.py`: + `_AC11_STABLE_LAUNCH_ARG_KEYS`
  whitelist, `_require_sidecar_fields`, `_validate_jsonl_duration`;
  updated `_read_ac11_meta` / `_normalize_ac11_server_args` /
  `_validate_cross_side_agreement` / `_run_ac11_mode`;
  `RunMetrics.duration_s`; updated `_filename_concurrency` regex;
  dropped `disable_radix_cache` from `_DS_ONLY_SERVER_ARG_KEYS`.
- `test/registered/unit/development/test_ac11_comparator.py`: fixture
  updates (`duration=600.0` default, `_OMIT` sentinel,
  `disable_radix_cache` always in sidecar server_args,
  `extra_summary` knob, `_make_trials` defaults radix-parity); existing
  Round 31 regressions adjusted for parity; +10 new tests + 7
  subTests.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
48 passed, 7 subtests passed (was 38; +10 + 7 subtests)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
332 passed, 7 subtests passed (was 322)
```

Verified both Codex Round-31 review reproducers:

```
# Mainline Gap 1: JSONL duration=5 + sidecar measurement_window_seconds=600
python development/benchmark_compare.py --ac11 \
    --ac11-baseline-results dsa_1.jsonl dsa_2.jsonl dsa_3.jsonl \
    --ac11-ds-results ds_1.jsonl ds_2.jsonl ds_3.jsonl
→ exit 2 with
  "AC-11 trial DS=...: bench_serving wall-clock duration=5.0s is below
   the AC-11 measurement-window floor of 600.0s"
  (was exit 0 + "AC-11 verdict: PASS")

# Mainline Gap 2: Sidecars with only timing + server_args_error: null
→ exit 2 with
  "AC-11 sidecar DSA=...: seed must be an int, got None"
  (was exit 0)
```

Commit: `48d6497b1` — [AC-11] tighten comparator: JSONL duration floor,
required-field sidecars, radix parity, launch-args whitelist.

## Remaining Items

Code-tier items queued for future rounds:

- bench_serving CLI changes (`--warmup-seconds` flag + warmup loop +
  measurement-window enforcement at the script/CLI layer). The
  comparator-side `duration` floor catches operator mistakes today;
  enforcing the floor at the bench_serving side will catch them earlier
  (before the trial JSONL is even written).
- Shallow AC-8 prefix-match regression coverage cleanup
  (Codex Round 22 queued).
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

Hardware-gated execution unchanged: `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac10-radix`, `task-ac11-compare` (comparator + scripts now
plan-conformant against all four Round-31 review reproducers; AC-10
must close before the H200 3-trial sweep can run with radix parity on),
`task-ac12-quality`.

## Push-to-remote Status

Branch is 33 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 32 applied the existing
`BL-20260527-shell-json-into-python-source` lesson on the comparator
side — the whitelist normalization is the consumer-side complement of
the writer-side lesson, ensuring dynamic `/get_server_info` payload
shapes don't break sidecar comparison. The validation tightening
itself is a "defense-in-depth" pattern that's better captured as code
+ regressions than as a generalized BitLesson. No new entry warranted.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-31-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-31-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-30-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-30-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-29-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-29-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-32-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
