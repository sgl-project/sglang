# Code Review - Round 37

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-37-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 37 Summary

## Work Completed

Codex Round 36 review found three real defects: (1) the M3-B
label-capture fixture had a false-pass path because the capture log
lived in the SERVER process but the fixture imported it CLIENT-side
(different memory; empty list ⇒ `not mismatches and cached_tokens
> 0` PASSes); (2) even with a real server log, radix-cache HITS
skip the write path entirely so write-record overlap can never
prove warm-prefix label stability — `record_table_snapshot` existed
but was unused; (3) the FP8 fixture called
`sglang_per_token_group_quant_fp8` (row-wise quantizer) rather than
exercising the production `fused_store_index_k_cache` path. Round 37
closes all three.

### Fix 1 — Server-side per-request capture via `meta_info`

`python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:

- New pure helper `build_request_capture(*, signatures, written,
  req_to_token, req_pool_indices, seq_lens) -> list[dict]`.
- For each request in the batch, computes
  `slots = req_to_token[req_pool_indices[b], :seq_lens[b]]` and
  returns a dict with `prompt_len`, `slots_sha`,
  `per_layer_label_sha`, `per_layer_written_sha`,
  `per_layer_written_all_true`.
- Pure function: safe to call from the production per-request
  finalization site; no IO, no globals, no env reads.

`python/sglang/srt/models/deepseek_v2.py::_publish_ds_request_summary`:

- When `radix_fixture_capture.is_capture_enabled()` is True, looks
  up `self.double_sparsity_selector.token_label_table` and
  `forward_batch.req_to_token_pool.req_to_token`, calls
  `build_request_capture(...)`, and attaches the per-request
  snapshot records to
  `summary["double_sparsity_radix_capture"]`.
- Wrapped in try/except so a capture-path bug can never break
  production; the WARNING-style `summary[
  "double_sparsity_radix_capture_error"]` records the failure
  shape.
- Default (env unset) path pays exactly one `os.environ.get`
  lookup.

Snapshots (not write records) are the right primitive: on a
radix-cache HIT the warm pass skips the write path entirely, so
write-record-overlap comparison would find nothing. The snapshot
of `signatures[L, prompt_slots]` after the request captures the
slot state regardless of whether labels were re-written.

### Fix 2 — Pure verdict helper + 11 CPU regressions

`test/manual/_m3b_label_capture_verdict.py` (NEW): pure helper
`evaluate_m3b_label_capture_verdict(*, cold_capture, warm_capture,
cached_tokens) -> {verdict, reasons}`. PASS only when ALL of:

- both captures present and non-empty;
- `cached_tokens > 0` on warm pass;
- `slots_sha` matches between cold and warm;
- `per_layer_label_sha` matches;
- `per_layer_written_all_true` is True on both sides.

`test/registered/unit/manual/test_m3b_label_capture_verdict.py`
(NEW, 11 tests):

- `test_all_conditions_met_is_pass`
- `test_empty_cold_capture_fails_false_pass_guard` ← closes the
  Codex Round-36 false-pass class
- `test_empty_warm_capture_fails`
- `test_none_capture_fails`
- `test_zero_cached_tokens_fails`
- `test_slots_sha_mismatch_fails`
- `test_layer_label_sha_mismatch_fails` (with first-mismatch hint)
- `test_layer_label_length_mismatch_fails`
- `test_unwritten_slot_on_cold_fails`
- `test_unwritten_slot_on_warm_fails`
- `test_non_list_capture_treated_as_missing`

### Fix 3 — Capture-aware fixture reads `meta_info`

`test/manual/test_dsv32_radix_label_capture_fixture.py` rewritten:

- Dropped the broken client-process in-process import path AND
  the optional HTTP endpoint scaffolding.
- Reads `response["meta_info"]["double_sparsity_radix_capture"]`
  for each pass; calls the pure verdict helper; asserts PASS.
- Works against remote and local servers identically (capture
  travels with the response). Class-level skip remains on
  `DS_BASE_URL`; the helper reports if
  `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` was not set on the server
  (capture list empty → verdict FAILs the run with a clear
  message rather than silently scoring it as a pass).

### Fix 4 — FP8 fixture via production fused-store path

`test/manual/test_dsv32_fp8_scale_stability.py` rewritten:

- Directly calls `sglang.jit_kernel.fused_store_index_cache.
  fused_store_index_k_cache` — the JIT kernel that the production
  `DSAIndexer._store_index_k_cache` uses on the fused-store path.
- Allocates a real `index_k_with_scale_buffer`-shaped tensor of
  shape `(num_pages, page_size * (128 + 4))` (the production
  layout for DSv4SingleKVPool).
- Writes K0 alone into page 0 of `buf_singleton`; writes K0 + 63
  deterministic neighbours into page 0 of `buf_packed`.
- Reads back K0's FP8 bytes (`buf[0, 0:128]`) and per-token scale
  bytes (`buf[0, page_size*128 : +4]`) at the production byte
  offsets. Asserts bit-equality of both.
- Hardware-gated via `SGLANG_DS_FP8_SCALE_PROOF=1` + CUDA +
  `can_use_dsa_fused_store(...)` returning True.

### Fix 5 — `build_request_capture` CPU unit tests (+4)

`TestBuildRequestCapture` in `test_double_sparsity_unit.py`:

- `test_single_request_snapshot_matches_manual_hash` — proves the
  per-layer SHA equals a manual `hashlib.sha256(...).hexdigest()`
  computed on the same slot bytes.
- `test_identical_calls_produce_identical_records` — foundation
  of the cold/warm equality.
- `test_two_request_batch_per_request_records_independent` —
  per-request records use the correct `req_to_token` row.
- `test_unwritten_slots_flag_not_all_true` — when prompt slots are
  not all written, `written_all_true` is False so the fixture
  refuses the side.

### Fix 6 — Launcher comment names the M3-B fixtures

`development/serve_double_sparsity.sh` trailing comment rewritten
to name BOTH the label-capture fixture
(`SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`) AND the FP8 production-path
fixture (`SGLANG_DS_FP8_SCALE_PROOF=1`) as the M3-B evidence
required before any guard flip — the continuation smoke is now
correctly identified as pre-flight only.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:
  `build_request_capture(...)` pure helper.
- `python/sglang/srt/models/deepseek_v2.py`: env-gated capture
  attachment in `_publish_ds_request_summary`.
- `development/serve_double_sparsity.sh`: trailing comment now
  names the M3-B fixtures.
- `test/manual/_m3b_label_capture_verdict.py` (NEW): pure verdict
  helper.
- `test/manual/test_dsv32_radix_label_capture_fixture.py`:
  rewritten to read `meta_info` + use the pure verdict helper.
- `test/manual/test_dsv32_fp8_scale_stability.py`: rewritten to
  call `fused_store_index_k_cache` on a real
  `index_k_with_scale_buffer`-shaped buffer.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +`TestBuildRequestCapture` (4 tests).
- `test/registered/unit/manual/test_m3b_label_capture_verdict.py`
  (NEW): +11 verdict regressions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
389 passed, 26 subtests passed (was 374 + 26; +15 named)

PYTHONPATH=python pytest test/registered/unit/manual/test_m3b_label_capture_verdict.py -v
11 passed (every false-pass class Codex Round 36 named now FAILs)

env -u DS_BASE_URL -u SGLANG_DS_RADIX_FIXTURE_CAPTURE -u SGLANG_DS_FP8_SCALE_PROOF \
  PYTHONPATH=python pytest \
   test/manual/test_dsv32_radix_cache_fixture.py \
   test/manual/test_dsv32_radix_label_capture_fixture.py \
   test/manual/test_dsv32_fp8_scale_stability.py -q
3 skipped (all 3 manual fixtures skip cleanly when env unset)

bash -n development/serve_double_sparsity.sh   # OK
```

The verdict helper unit suite reproduces the exact false-pass
scenarios Codex Round-36 review identified and confirms they now
FAIL — the same helper is used by the manual hardware fixture, so
the hardware-side path is guarded by the same logic.

Commit: `b7ab80fc0` — [AC-10] M3-B direct evidence via meta_info +
production FP8 store-path proof.

## Remaining Items

AC-10 hardware execution (operator-driven, not in this loop):

1. Boot DS server with `SGLANG_DS_RADIX_OVERRIDE=1` +
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` + remove
   `--disable-radix-cache` for this one-shot run.
2. Run `test_dsv32_fp8_scale_stability.py` with
   `SGLANG_DS_FP8_SCALE_PROOF=1`. On PASS, save the artifact.
3. Run `test_dsv32_radix_label_capture_fixture.py`. On PASS, save
   the artifact.
4. Wire `record_radix_fixture_passed(server_args, artifact_path=
   "<label-capture-artifact.json>")` into a launcher init module
   BEFORE `validate_double_sparsity` runs.
5. Delete the `--disable-radix-cache` line in
   `serve_double_sparsity.sh` (the AC-10-FIXTURE-MARKER comment
   names the exact line).
6. Update `test_ds_server_does_disable_radix_cache_until_ac10`
   for the post-AC-10 expectation.

After AC-10 closes, `task-ac11-compare` H200 3-trial DSA + DS
sweep + comparator invocation runs.

Other hardware-gated tasks (unchanged): `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac12-quality`.

Queued cleanup (unchanged):
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

## Push-to-remote Status

Branch is 38 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 37 applied `BL-20260527-ds-metadata-via-forward-context`
to the new server-side capture attachment in
`_publish_ds_request_summary` — the DS metadata access lives on
`self.double_sparsity_selector.token_label_table` +
`forward_batch.req_to_token_pool`, not on
`forward_batch.attn_backend` (which production never sets). The
"capture-via-per-request-summary" pattern is one-shot
infrastructure for AC-10 specifically rather than a general
guideline. No new entry warranted.
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
a41b1d952 [AC-10] M3-B capture primitive + direct-evidence fixtures + identical-prompt smoke
b7ab80fc0 [AC-10] M3-B direct evidence via meta_info + production FP8 store-path proof
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-36-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-36-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-35-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-35-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-34-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-34-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-37-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
