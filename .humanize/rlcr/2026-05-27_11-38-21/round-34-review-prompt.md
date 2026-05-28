# FULL GOAL ALIGNMENT CHECK - Round 34

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 34 Summary

## Work Completed

Codex Round 33 review identified three real bugs in Round 33's
bench_serving timing + comparator code. None could be deferred without
leaving AC-11 unsafe.

### Fix 1 — Multi-epoch metrics IndexError

`python/sglang/bench_serving.py`: when `measurement_window_seconds > 0`
ran more than one epoch, `outputs` accumulated across epochs but
`calculate_metrics(input_requests=input_requests, outputs=outputs)`
indexed `input_requests[i]` for every successful output. As soon as
`len(outputs) > len(input_requests)`, the metric path raised
`IndexError: list index out of range` and the JSONL was never written.

Fix: replicate the per-epoch input row list `measured_epochs` times
before passing it to `calculate_metrics` for non-multi-turn runs. The
workload is deterministic per epoch (seed reset between warmup and
measured phases), so replication mirrors the dispatch order. Multi-
turn keeps the existing `input_requests=None` behavior. Result:

```python
if is_multi_turn:
    metrics_input_requests = None
elif measured_epochs > 1:
    metrics_input_requests = input_requests * measured_epochs
else:
    metrics_input_requests = input_requests
```

### Fix 2 — Per-epoch `num_prompts` in JSONL, total in `measured_num_prompts`

Round 33 emitted `num_prompts = args.num_prompts * measured_epochs`
in the JSONL, but the benchmark scripts wrote the per-epoch
`NUM_PROMPTS` into the sidecar. The Round 33 workload cross-check
then refused every legitimate multi-epoch artifact. The cross-check
itself was correct given matching semantics; only the producer was
off-by-multiplier.

Fix (`python/sglang/bench_serving.py`):

- `num_prompts` in the JSONL is now the PER-EPOCH workload shape
  (= `args.num_prompts`), matching the sidecar's `num_prompts`.
- New JSONL field `measured_num_prompts = per_epoch * measured_epochs`
  carries the total measured attempts across the multi-epoch window.
- `completed` (number of successful outputs) is unchanged.

### Fix 3 — DS/DSA side-identity validation

Because `_normalize_ac11_server_args` strips
`enable_double_sparsity` + `double_sparsity_config` from cross-side
comparison (those are the ONLY sanctioned differences per plan §AC-11),
the comparator could publish PASS when both columns were actually
DSA-on. Codex's reproducer: 3 DSA files + 3 fake "DS" files whose
sidecars also said `mode="native_nsa"` and lacked the DS flags →
exit 0 + `AC-11 verdict: PASS`.

New helper `_validate_ac11_side_identity(meta, *, expected_side, path)`
runs inside `_run_ac11_mode` before cross-side normalization:

- **DSA** column requires `mode == "native_nsa"`, no
  `enable_double_sparsity` (or it is False/absent), and no non-empty
  `double_sparsity_config`.
- **DS** column requires `mode == "double_sparsity"`,
  `server_args.enable_double_sparsity is True`, and a non-empty
  `double_sparsity_config` string.

### Cleanup — Plan markers removed from production code

Per plan §361-364, implementation code must not carry plan-specific
terminology. Round 33 introduced "Round 33" / "Codex Round NN review"
markers; Round 34 stripped them:

- `python/sglang/bench_serving.py`: 4 comments + 2 CLI help strings.
- `development/benchmark_compare.py`: 5 comments.

Kept references that are load-bearing for users: the `--ac11` CLI
flag, AC-11 in error messages (operator needs to know which gate
they're hitting), and AC-11 in the module docstring.

### Regressions (+8 named)

`test/registered/unit/development/test_bench_serving_timing.py` (+2):

- `test_multi_epoch_real_metrics_path_no_index_error` — drives
  bench_serving with 2 input rows and a 30ms window, WITHOUT
  monkeypatching `calculate_metrics`. The real metric path is
  exercised; asserts `measured_epochs >= 2` and
  `completed == num_input_rows * measured_epochs`. Round 33's code
  would have raised `IndexError` here.
- `test_multi_epoch_jsonl_consistency` — asserts `num_prompts` is
  per-epoch, `measured_num_prompts == num_prompts * measured_epochs`,
  and `duration >= window`.

`test/registered/unit/development/test_ac11_comparator.py` (+6):

- `test_side_identity_dsa_with_ds_flag_refused` — DSA sidecar has
  `enable_double_sparsity=True` → exit 2.
- `test_side_identity_ds_missing_enable_flag_refused` — DS sidecar
  lacks the enable flag → exit 2.
- `test_side_identity_ds_missing_config_refused` — DS sidecar has
  empty `double_sparsity_config` → exit 2.
- `test_side_identity_both_sides_native_refused` — closes Codex's
  reproducer: 3 DSA + 3 fake-DS all `mode='native_nsa'` → exit 2.
- `test_side_identity_mode_field_mismatch_refused` — DSA sidecar
  declares `mode='double_sparsity'` (copy-paste error) → exit 2.
- `test_jsonl_num_prompts_per_epoch_matches_sidecar` — legitimate
  multi-epoch artifact (`num_prompts=320`, `measured_epochs=2`,
  `measured_num_prompts=640`) matching sidecar `num_prompts=320`
  passes the comparator.

## Files Changed

- `python/sglang/bench_serving.py`: replicate per-epoch input rows
  for `calculate_metrics`; emit per-epoch `num_prompts` + new
  `measured_num_prompts` field; strip plan markers. +56 / -48 lines.
- `development/benchmark_compare.py`: `_validate_ac11_side_identity`
  helper + call from `_run_ac11_mode` (DSA + DS); plan-marker
  cleanup. +99 / -19 lines.
- `test/registered/unit/development/test_ac11_comparator.py`: +6
  identity / per-epoch regressions. +103 lines.
- `test/registered/unit/development/test_bench_serving_timing.py`:
  +2 multi-epoch real-metrics regressions. +147 lines.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
57 passed, 26 subtests passed (was 51 + 26)

PYTHONPATH=python pytest test/registered/unit/development/test_bench_serving_timing.py -q
9 passed (was 7)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
356 passed, 26 subtests passed (was 348 + 26)
```

Verified Codex's "both-sides-native" reproducer (3 DSA + 3 fake-DS
sidecars, all `mode='native_nsa'`, no DS flags, all other context
matching):

```
python development/benchmark_compare.py --ac11 \
    --ac11-baseline-results dsa_1.jsonl dsa_2.jsonl dsa_3.jsonl \
    --ac11-ds-results ds_1.jsonl ds_2.jsonl ds_3.jsonl
→ exit 2 with
  "AC-11 input refusal at conc=64: AC-11 DS trial ...: sidecar
   mode='native_nsa' but expected 'double_sparsity' for the DS-on
   column."
(was exit 0 + "AC-11 verdict: PASS")
```

Verified the multi-epoch real-metrics path: 2 input rows + 30ms
window completes without `IndexError` and writes a coherent JSONL.

```
grep -nE 'Round 3[0-9]|Codex Round' \
  python/sglang/bench_serving.py development/benchmark_compare.py
(no output — plan markers stripped)
```

Commit: `04e811aba` — [AC-11] fix multi-epoch metrics crash,
per-epoch num_prompts, side-identity gate.

## Remaining Items

Hardware-gated tasks (unchanged):
- `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`,
  `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`,
  `task-ac9-baseline`, `task-ac12-quality`.

Active original-plan task blocking AC-11 H200 execution:
- `task-ac10-radix` — M3-B radix-cache hardware fixture; FP8 scale
  stability check; flip `_double_sparsity_radix_fixture_passed =
  True`; remove `--disable-radix-cache` from
  `serve_double_sparsity.sh`; update launcher contract tests. This
  is an active original-plan task (corrected the tracker wording
  per the previous-round review feedback). After AC-10 passes,
  `task-ac11-compare` runs the H200 3-trial sweep + comparator.

Queued side issues (unchanged):
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

## Push-to-remote Status

Branch is 35 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 34 applied existing
`BL-20260527-shell-json-into-python-source` (sidecar/JSONL JSON
hand-off) and `BL-20260527-ds-metadata-via-forward-context` (DS path
identification — applied conceptually to the new side-identity
validator). The multi-epoch input-replication fix and the
identity-gate pattern are problem-specific defenses better captured
as code + regressions than as generalized lessons. No new entry
warranted.
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
04e811aba [AC-11] fix multi-epoch metrics crash, per-epoch num_prompts, side-identity gate
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-33-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-33-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-32-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-32-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-31-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-31-review-result.md


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

To implement the original plan at @development/loop4/refined_plan_v1.md, we have completed **35 iterations** (Round 0 to Round 34).

The project's `.humanize/rlcr/2026-05-27_11-38-21/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-27_11-38-21/round-33-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-32-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-33-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-34-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-34-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
