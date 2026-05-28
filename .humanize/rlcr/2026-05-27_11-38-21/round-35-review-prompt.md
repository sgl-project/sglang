# Code Review - Round 35

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-35-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 35 Summary

## Work Completed

Codex Round 34 review verdict was **ADVANCED** with no blocking
implementation bugs in AC-11. Action items moved the mainline back to
active original-plan execution, and the only original-plan task with
non-hardware code work still open was `task-ac10-radix`. Round 35
lands every piece of AC-10 that can be done without running V3.2.

### Fix 1 — CPU unit-level proof of DS label bit-stability

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`
+ `TestAC10RadixCacheLabelBitStability` (4 tests). The hardware
fixture's cold-vs-warm label equality reduces, at the labeling level,
to the deterministic property tested here: given the same projected
K-noPE input, `token_label_write` at the same `cache_loc` produces
bit-equal `signatures` rows even when the slot was just invalidated
(the radix-cache reuse semantic).

- `test_token_label_write_is_deterministic_for_same_kv_input` —
  identical input twice → bit-equal rows.
- `test_invalidate_then_rewrite_same_input_yields_equal_labels` —
  write `x` at slot S; `invalidate_token_label_slots(S)`; re-write
  `x` at S → bit-equal to the cold write. This is the labeling-side
  equivalent of the hardware fixture's cold-vs-warm equality
  property.
- `test_different_kv_input_yields_different_labels` (negative
  counterpart): different K-noPE → different label rows, so the
  bit-equality test above is real determinism, not a constant.
- `test_invalidate_does_not_clear_signature_bytes` — `invalidate`
  only clears the `written` flag; signature bytes are preserved.
  Matters because the "no stale picks" property is governed by
  `written` alone — a partial re-write must not be able to mix old
  and new bytes.

### Fix 2 — record_radix_fixture_passed helper

`python/sglang/srt/layers/attention/double_sparsity/validator.py`:
new `record_radix_fixture_passed(server_args)` sets
`_double_sparsity_radix_fixture_passed = True` on the args object and
emits a WARNING-level audit log line. The existing DEC-2 guard inside
`validate_double_sparsity` already reads this attribute (lines
211-223); the helper makes the operator flip explicit + grep-able.

New validator test `test_radix_on_refused_until_fixture_recorded`
proves the two-state guard:

1. `disable_radix_cache=False` + no helper call → validator raises
   `ValueError` with "M3-B page-stability fixture" in the message.
2. `disable_radix_cache=False` + `record_radix_fixture_passed(args)`
   → validator passes cleanly without `SGLANG_DS_RADIX_OVERRIDE=1`.

### Fix 3 — Hardware-gated M3-B fixture harness

`test/manual/test_dsv32_radix_cache_fixture.py` (NEW): mirrors the
AC-12 manual pattern (`@unittest.skipUnless(DS_BASE_URL, ...)`).
Issues a paired cold-prefix / warm-prefix request against a DS server
with radix cache ENABLED:

- **Cold request**: unique shared-prefix payload the server has never
  seen → fresh KV-slot allocation + DS label writes from a single-pass
  FP8 dequant of the just-written K-noPE.
- **Warm request**: same shared-prefix payload re-sent immediately.
  With radix cache ON, the shared-prefix slots are reused.

At `temperature=0` with `max_new_tokens=128`, equal continuations are
the operator-observable proxy for label bit-stability. Records
artifact at
`development/results/dsv32_radix_fixture_cold_warm_<ts>.json` with
`commit_sha`, `server_args`, prompts, continuations, and verdict.
`setUpClass` re-skips when the server reports
`disable_radix_cache=True` so the fixture cannot accidentally run
against the gated configuration it is meant to verify.

### Fix 4 — Launcher AC-10-FIXTURE-MARKER + contract test

`development/serve_double_sparsity.sh`: inline marker

```
`# AC-10-FIXTURE-MARKER: remove the next line after M3-B fixture` \
`# pass + record_radix_fixture_passed(server_args) flip (DEC-2).` \
--disable-radix-cache \
```

placed above the `--disable-radix-cache` flag. The post-AC-10
launcher edit is now a mechanical one-line deletion; the marker
persists for audit. The trailing comment now points operators at the
new `record_radix_fixture_passed` helper.

New `test_ds_server_has_ac10_fixture_marker` contract test in
`test_option_b_scripts.py` asserts (a) the marker is present, and
(b) it sits ABOVE the `--disable-radix-cache` flag so the edit-point
context is visible.

### Fix 5 (queued cleanup) — Strip Round 33 markers from benchmark scripts

`development/benchmark.sh` + `development/benchmark_baseline.sh`:
replaced

```
# Round 33 (AC-11): refuse the run if the observed JSONL `duration`
# is below MEASUREMENT_WINDOW_S — guards against bench_serving
# bailing out early before the time-based loop met its threshold.
```

with neutral wording that names the property, not the round (plan
§361-364). Closes the Codex Round-34 review queued item.

## Files Changed

- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +`TestAC10RadixCacheLabelBitStability` (4 tests),
  +`test_radix_on_refused_until_fixture_recorded` (1 test).
  +236 lines.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py`:
  +`record_radix_fixture_passed(server_args)` helper.
  +33 lines.
- `test/manual/test_dsv32_radix_cache_fixture.py` (NEW): manual
  hardware fixture + operator runbook + artifact recorder.
  +224 lines.
- `development/serve_double_sparsity.sh`: AC-10-FIXTURE-MARKER above
  `--disable-radix-cache` + updated trailing comment pointing at the
  helper. +9 / -3 lines.
- `test/registered/unit/development/test_option_b_scripts.py`:
  +`test_ds_server_has_ac10_fixture_marker`. +26 lines.
- `development/benchmark.sh` + `development/benchmark_baseline.sh`:
  plan-marker cleanup. +6 / -6 lines.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
362 passed, 26 subtests passed (was 356 + 26; +6 named)

env -u DS_BASE_URL PYTHONPATH=python pytest test/manual/test_dsv32_radix_cache_fixture.py -q
1 skipped (correct: harness skips when env var unset)

bash -n development/serve_double_sparsity.sh   # OK
bash -n development/benchmark.sh               # OK
bash -n development/benchmark_baseline.sh      # OK

grep -nE 'Round 3[0-9]|Codex Round' \
  development/benchmark.sh development/benchmark_baseline.sh
(no output — plan markers fully stripped)
```

Commit: `2d0336a81` — [AC-10] M3-B radix-cache fixture harness +
record_radix_fixture_passed helper.

## Remaining Items

AC-10 hardware execution (operator-driven, not in this loop):

1. Boot DS server with radix cache ON. The launcher still passes
   `--disable-radix-cache`; for this one-shot fixture run set
   `SGLANG_DS_RADIX_OVERRIDE=1` and edit the launcher to drop the
   flag (use the AC-10-FIXTURE-MARKER line as the edit point).
2. Run `DS_BASE_URL=http://...:30000 pytest
   test/manual/test_dsv32_radix_cache_fixture.py -v`. On pass, the
   artifact at `development/results/dsv32_radix_fixture_cold_warm_
   <ts>.json` records the verdict.
3. On pass, permanently remove the `--disable-radix-cache` line in
   `serve_double_sparsity.sh` and update the launcher (or downstream
   server-args parser) to call
   `record_radix_fixture_passed(server_args)` before
   `validate_double_sparsity` runs.
4. Update `test_ds_server_does_disable_radix_cache_until_ac10` to
   the post-AC-10 expectation (e.g., assert the flag is absent +
   the helper is invoked from the launcher).

After AC-10 closes, `task-ac11-compare` H200 3-trial DSA + DS sweep
+ comparator invocation runs.

Other hardware-gated tasks (unchanged): `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac12-quality`.

Queued cleanup (unchanged):
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

## Push-to-remote Status

Branch is 36 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 35 applied existing
`BL-20260527-ds-metadata-via-forward-context` (DS metadata flow),
`BL-20260527-importlib-dataclass-sys-modules` (manual-harness
loader pattern — the new harness uses standard import, no dataclass
loader issue), and `BL-20260527-shell-json-into-python-source`
(launcher's JSON-builder path remains via env-var helper). The
M3-B fixture pattern (CPU unit-level determinism proof + hardware-
gated harness + operator-helper-flip + launcher-marker) is the
AC-10-specific embodiment of a more general "fixture-gated launch
flag" pattern, but it is grounded in this AC's specifics and is
better captured as code + tests than as a generalized BitLesson. No
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
04e811aba [AC-11] fix multi-epoch metrics crash, per-epoch num_prompts, side-identity gate
2d0336a81 [AC-10] M3-B radix-cache fixture harness + record_radix_fixture_passed helper
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-34-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-34-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-33-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-33-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-32-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-32-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-35-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
