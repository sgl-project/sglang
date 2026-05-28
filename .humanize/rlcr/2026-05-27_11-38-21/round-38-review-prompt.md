# Code Review - Round 38

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-38-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 38 Summary (Drift Recovery)

## Drift Cause + Recovery Anchor

Rounds 35 → 37 iteratively patched the M3-B label-capture fixture
but never verified the full producer→transport→consumer contract
in CI. Each round Codex review found a NEW seam:

* R35 — continuation-only proxy (not direct label evidence).
* R36 — capture log lived in the client process (different memory
  from the server's `_LOG`).
* R37 — capture published as a per-batch list under
  `summary["double_sparsity_radix_capture"]`, but the existing
  per-request summary transport in
  `_maybe_collect_per_request_summary` (`batch_result_processor.py`)
  unwraps `v[i]` per request — the client receives a **dict per
  request** in `meta_info`. My verdict helper rejected dicts as
  missing. Also: capture fired from `_publish_ds_request_summary`
  (called per-layer at the START of selection, BEFORE writes), so
  the current layer was stale. Also: only the fused FP8 path was
  implemented; fallback was `skipTest`.

**Recovery anchor**: a CPU integration test that reproduces the
actual transport unwrap and asserts the verdict helper accepts the
dict-shaped per-request meta_info. Future shape/timing drift on
either side now fails locally in CI, not in remote review.

## Mainline Objective + Target ACs

- AC-10 M3-B label-capture fixture delivers honest direct evidence
  on H200 with the correct response shape, post-write extend-only
  timing, cached-prefix-only comparison, AND both fused + fallback
  FP8 production-store paths.

## Work Completed

### Fix 1 — Verdict helper accepts the real transported shape

`test/manual/_m3b_label_capture_verdict.py::_records`:

- `dict` → `[dict]` (production transport: scheduler unwraps `v[i]`
  per request → tokenizer surfaces a single dict in meta_info).
- `list` → `list` (legacy / direct helper test).
- None / missing / other → None (treated as missing evidence).

### Fix 2 — Capture moved to `_write_token_labels` post-write, extend-only

`python/sglang/srt/layers/attention/dsa_backend.py`:

- New module-level helper `_ds_radix_publish_extend_snapshot(*,
  backend, forward_batch)` calls `build_request_capture` against
  the live table state and stashes the per-batch list on
  `forward_batch.ds_per_request_summary["double_sparsity_radix_capture"]`.
- Called from `_write_token_labels` AFTER `token_label_write(...)`
  returns AND only when `forward_batch.forward_mode.is_extend()`.
- Each layer's publish overwrites the previous → the LAST DS
  layer's call wins; at that point every layer's prompt-slot
  labels are fresh.
- Extend-only restriction prevents decode steps from clobbering
  the prefill snapshot.
- Wrapped in try/except; failure records the shape in
  `summary["double_sparsity_radix_capture_error"]` but never
  raises (capture must not break production).

`python/sglang/srt/models/deepseek_v2.py::_publish_ds_request_summary`:

- Removed the Round-37 capture branch (fired pre-write, per layer,
  every mode). Comment left in place naming the new emission site.

### Fix 3 — Per-token API in `build_request_capture`

`radix_fixture_capture.py`:

- Each record now includes `per_token_slot_sha: list[str]`
  (per-prompt-position slot SHA) and
  `per_layer_per_token_label_sha: list[list[str]]` (per layer,
  per position).
- `slots_sha` retained for back-compat (aggregate over the full
  range).
- New `compare_cached_prefix(*, cold, warm, cached_tokens) ->
  {ok, first_diverging_position, divergence_kind, reason}` —
  compares only the first `cached_tokens` positions. Clamps to
  the shorter capture so extra decode-allocated positions on
  either side don't force a false mismatch.

### Fix 4 — Producer→transport→consumer CPU integration test

`test/registered/unit/manual/test_m3b_label_capture_verdict.py`:

- `test_meta_info_transport_dict_shape_passes_verdict` — the
  drift-recovery anchor. Mirrors the scheduler's
  `_maybe_collect_per_request_summary` unwrap shim, asserts the
  per-request meta_info value is a DICT, then feeds it to the
  verdict helper → PASS.
- `test_meta_info_transport_dict_shape_with_extra_decode_slots` —
  warm has more positions than cold; `cached_tokens=5` → PASS
  (positions ≥5 ignored).
- `test_meta_info_transport_dict_shape_with_cached_prefix_diverging`
  — position 2 differs in the cached prefix → FAIL with
  `first_diverging_position=2`, `kind='slot'`.
- `test_meta_info_transport_label_divergence_within_cached_prefix`
  — slots match but layer-1 label SHA differs at position 3 →
  FAIL with `kind='label'`, `layer=1`.
- 9 verdict negatives: empty/None/non-list-non-dict capture,
  zero cached_tokens, written_all_true=False on either side,
  direct dict path PASS, direct list path PASS.

### Fix 5 — FP8 fixture with fused + fallback production paths

`test/manual/test_dsv32_fp8_scale_stability.py`:

- `_read_back_k0_bytes(buf, *, page_idx, position)` — shared
  helper using production byte offsets (`buf[page,
  position*128:(position+1)*128]` for FP8 and
  `buf[page, PAGE_SIZE*128 + position*4 : +4]` for the scale).
- `_try_fused_path` — `fused_store_index_k_cache(K0, buf, loc,
  page_size)` against a real `index_k_with_scale_buffer`-shaped
  buffer. Skips if `can_use_dsa_fused_store(...)` returns False
  or the JIT module import fails.
- `_try_fallback_path` — `act_quant(K, block_size=128)` →
  `SetKAndS.execute(pool=SimpleNamespace(page_size=64), buf=...,
  loc=..., index_k=fp8, index_k_scale=scale)` against the same
  buffer layout. Same readback helper.
- Runs every path that successfully executes; `skipTest` only
  when NEITHER runs. Artifact records `path_used` per branch +
  `per_path_verdict` + combined verdict.

### Fix 6 — `build_request_capture` per-token regressions (+6)

`TestBuildRequestCapture` in `test_double_sparsity_unit.py`:

- `test_per_token_slot_sha_lengths_match_prompt_len`.
- `test_compare_cached_prefix_first_position_diff`.
- `test_compare_cached_prefix_zero_cached_tokens_no_overlap`.
- `test_per_token_slot_sha_deterministic_across_calls`.
- `test_compare_cached_prefix_label_divergence_named_layer`.
- `test_compare_cached_prefix_clamps_to_shorter_capture`.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:
  per-token SHAs + `compare_cached_prefix`.
- `python/sglang/srt/layers/attention/dsa_backend.py`:
  `_ds_radix_publish_extend_snapshot` helper + call from
  `_write_token_labels`.
- `python/sglang/srt/models/deepseek_v2.py`: removed Round-37
  capture branch.
- `test/manual/_m3b_label_capture_verdict.py`: dict/list dual
  shape acceptance, uses `compare_cached_prefix`.
- `test/manual/test_dsv32_fp8_scale_stability.py`: fused +
  fallback production paths; shared readback helper.
- `test/registered/unit/manual/test_m3b_label_capture_verdict.py`:
  4 transport integration + 9 verdict negatives.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +6 per-token / compare_cached_prefix regressions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
397 passed, 26 subtests passed (was 389 + 26; +8 named)

PYTHONPATH=python pytest test/registered/unit/manual/test_m3b_label_capture_verdict.py -v
13 passed (4 transport integration + 9 verdict negatives)

env -u DS_BASE_URL -u SGLANG_DS_RADIX_FIXTURE_CAPTURE -u SGLANG_DS_FP8_SCALE_PROOF \
  PYTHONPATH=python pytest \
    test/manual/test_dsv32_radix_cache_fixture.py \
    test/manual/test_dsv32_radix_label_capture_fixture.py \
    test/manual/test_dsv32_fp8_scale_stability.py -q
3 skipped (all manual fixtures skip cleanly when env unset)
```

The drift-recovery integration test (`test_meta_info_transport_
dict_shape_passes_verdict`) reproduces the EXACT scheduler-side
unwrap that converted the producer's per-batch list into a
per-request dict — and asserts the verdict helper now accepts that
dict shape. The same helper is used by the manual hardware
fixture, so the dict-shape transport path is guarded under CI.

Commit: `ccbdac7c5` — [AC-10] M3-B drift recovery: transport
shape, post-write timing, per-token API, FP8 fallback.

## Recovery Verdict

Recovery is in CI: every Round-37-review claim is now exercised
either as a direct regression (slot divergence at position 2; layer
SHA divergence at layer 1 position 3) or as an integration test
(dict-shape transport unwrap → verdict). Failure modes are loud:

* Empty / wrong-shape capture → FAIL with "capture missing".
* `cached_tokens == 0` → FAIL with "radix cache was not exercised".
* Slot divergence within `cached_tokens` → FAIL naming the
  position.
* Per-layer per-token label divergence → FAIL naming layer +
  position.
* Either side's `written_all_true` False → FAIL naming the bad
  layers.

The FP8 fixture exercises BOTH production paths that
`_store_index_k_cache` uses on real hardware; CPU-only runs that
cannot execute the production kernel still skip cleanly.

## Remaining Items

AC-10 hardware execution (operator-driven):

1. Boot DS server with `SGLANG_DS_RADIX_OVERRIDE=1` +
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, removing
   `--disable-radix-cache` for this one-shot run.
2. Run `test_dsv32_fp8_scale_stability.py` with
   `SGLANG_DS_FP8_SCALE_PROOF=1`. Verify BOTH paths PASS.
3. Run `test_dsv32_radix_label_capture_fixture.py`. Verify the
   `meta_info["double_sparsity_radix_capture"]` records non-zero
   `cached_tokens` AND verdict PASS.
4. Wire `record_radix_fixture_passed(server_args, artifact_path=
   "<label-capture-artifact.json>")` into a launcher init module
   BEFORE `validate_double_sparsity` runs.
5. Delete the `--disable-radix-cache` line in
   `serve_double_sparsity.sh` (`# AC-10-FIXTURE-MARKER` names the
   line).
6. Update `test_ds_server_does_disable_radix_cache_until_ac10`
   for the post-AC-10 expectation.

After AC-10 closes, `task-ac11-compare` H200 sweep runs.

Other hardware-gated tasks (unchanged): `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac12-quality`.

Queued cleanup (unchanged):
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

## Push-to-remote Status

Branch is 39 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The drift-cause this round (producer/transport/consumer
mismatch from incremental patching) is a candidate for a future
generalized lesson ("when adding a server→client side-channel,
ship the CPU integration test FIRST and the producer side
SECOND"), but the pattern is too broad to phrase as a single
problem→solution entry. The specific failure modes are already
captured as registered regressions, which are the more durable
guardrail.
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
ccbdac7c5 [AC-10] M3-B drift recovery: transport shape, post-write timing, per-token API, FP8 fallback
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-37-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-37-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-36-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-36-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-35-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-35-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-38-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
