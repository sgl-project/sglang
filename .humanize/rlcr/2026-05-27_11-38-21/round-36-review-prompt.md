# Code Review - Round 36

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-36-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 36 Summary

## Work Completed

Codex Round 35 review surfaced five concrete defects in the manual
M3-B fixture: it (a) sent DIFFERENT prompts in its cold and warm
passes while asserting continuation equality, (b) asserted only
continuation byte-equality as a proxy for label bit-equality (the
plan requires direct label evidence), (c) did not record
`commit_sha`, (d) had no FP8 scale-stability proof, and (e) gave the
operator no way to actually capture DS label table state. Round 36
closes all five.

### Fix 1 — Server-side capture primitive

New module
`python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:

- Module-level `_LOG: list[dict]` guarded by a `threading.Lock`.
- `is_capture_enabled()` reads
  `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`.
- `record_write(layer_id, cache_loc, k_nope, written_after=None)`
  computes SHA256 of `cache_loc.long().contiguous()` bytes and
  `k_nope.float().contiguous()` bytes per write and appends a
  per-write record. No-op early exit when the env flag is off — the
  production hot path pays one `os.environ.get` lookup.
- `record_table_snapshot(signatures, written, slots, label=...)`
  emits per-layer SHA256 of `signatures[L, slots]` and
  `written[L, slots]` so cold/warm comparisons are by-slot,
  by-layer.
- `get_log()` / `clear_log()` accessors.

`python/sglang/srt/layers/attention/dsa_backend.py`: hooked into
`_write_token_labels` after `token_label_write(...)` returns. The
default path (env unset) is unchanged — zero overhead.

### Fix 2 — Identical-prompt continuation-smoke fixture

`test/manual/test_dsv32_radix_cache_fixture.py`:

- Replaced the `pass_id="cold"` / `pass_id="warm"` template with a
  single `_SHARED_PREFIX_PROMPT` constant; both passes use the
  identical string. The radix cache can now actually reuse slots
  across the two requests.
- Renamed the test
  `test_cold_then_warm_continuation_is_bit_equal` →
  `test_cold_warm_continuation_smoke`. Module docstring marks this
  fixture as necessary-but-not-sufficient pre-flight; the
  AC-10-conformant M3-B evidence comes from the new capture
  fixture + FP8 scale fixture.
- Artifact records `local_commit_sha` (from `git rev-parse HEAD`)
  + `server_commit_sha` (from `/get_server_info`'s `server_args`
  when present) per the audit-trail requirement.

### Fix 3 — Capture-aware M3-B label-equality fixture

New `test/manual/test_dsv32_radix_label_capture_fixture.py`:

- Class-level skip unless `DS_BASE_URL` set AND
  `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` set. setUpClass also re-skips
  when the server reports `disable_radix_cache=True` so the test
  cannot accidentally run against the gated configuration it is
  designed to verify.
- Reads the capture log via in-process import (operator co-locates
  fixture with the server process) or via an optional HTTP endpoint
  at `SGLANG_DS_RADIX_CAPTURE_LOG_URL` for remote-server setups.
- Issues two requests with IDENTICAL prompts at temperature=0;
  asserts:
  - `meta_info.cached_tokens > 0` on the warm pass — proves the
    radix cache actually reused slots (otherwise the test only
    re-proves the CPU unit determinism property).
  - Every overlap between cold-pass and warm-pass writes at the
    same `(layer_id, cache_loc_sha)` has matching `k_nope_sha`.
    This is the direct M3-B label bit-equality proof.
- Artifact records commit_sha (local + server), prompt,
  cached_tokens, write counts, mismatches (first 32 logged).

### Fix 4 — Hardware-gated FP8 scale-stability proof

New `test/manual/test_dsv32_fp8_scale_stability.py`:

- Class-level skip unless `SGLANG_DS_FP8_SCALE_PROOF=1` AND CUDA
  with FP8 support is available. Deliberate opt-in — a CPU-only
  run cannot be misread as a passing M3-B check.
- Invokes the production `sglang_per_token_group_quant_fp8` kernel
  for the same K0 row as:
  - Singleton input (1×128, just K0).
  - Packed input (64×128, K0 alongside 63 deterministic
    neighbours).
- Asserts `scale_single[0] == scale_packed[0]` and
  `fp8_single[0] == fp8_packed[0]`. If the kernel's per-block scale
  depends on neighbour tokens, the DS label-write hook would see
  different dequantized K-noPE in cold vs warm paths and the AC-10
  guard MUST stay in place.

### Fix 5 — `record_radix_fixture_passed` artifact audit trail

`python/sglang/srt/layers/attention/double_sparsity/validator.py`:

- Signature extended to
  `record_radix_fixture_passed(server_args, *, artifact_path:
  Optional[str] = None)`.
- When `artifact_path` is supplied, the audit WARNING line records
  the path + SHA256 of its contents. A grep over server logs
  surfaces both the flip event AND the evidence that authorized
  it.
- Three new validator tests cover (a) artifact path + SHA appear in
  the log, (b) no-artifact-path back-compat, (c) unreadable path
  still flips the guard but marks the artifact as
  `<unreadable:...>`.

### Fix 6 — CPU unit tests for the capture primitive (+9)

`TestRadixFixtureCapture` (in the existing DS unit test file):

- `test_record_write_noop_when_env_unset` — production hot path
  pays zero overhead.
- `test_record_write_appends_when_env_set` — full record shape.
- `test_identical_inputs_produce_identical_hashes` — the
  foundational determinism the capture fixture relies on.
- `test_different_cache_loc_produces_different_hash` and
  `test_different_k_nope_produces_different_hash` — sensitivity
  proofs.
- `test_int32_vs_int64_cache_loc_hashes_equal` — dtype stability
  so int32 cold vs int64 warm cannot spuriously disagree.
- `test_snapshot_equals_across_identical_label_writes` and
  `test_snapshot_differs_when_a_layer_row_changes` — per-layer
  hash sensitivity.
- `test_clear_log_resets_state`.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`
  (NEW): capture primitive. ~150 lines.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py`:
  `record_radix_fixture_passed` gains `artifact_path` + SHA audit.
- `python/sglang/srt/layers/attention/dsa_backend.py`: env-gated
  capture hook in `_write_token_labels`.
- `test/manual/test_dsv32_radix_cache_fixture.py`: identical-
  prompt fix, honest naming, commit_sha, docstring rewritten to
  point at the M3-B fixtures.
- `test/manual/test_dsv32_radix_label_capture_fixture.py` (NEW):
  M3-B label-equality fixture.
- `test/manual/test_dsv32_fp8_scale_stability.py` (NEW): FP8
  scale-stability fixture.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +`TestRadixFixtureCapture` (9 tests) +3 validator artifact-path
  tests.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
374 passed, 26 subtests passed (was 362 + 26; +12 named)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
   -k 'TestRadixFixtureCapture or test_record_radix_fixture' -q
12 passed (9 capture primitive + 3 record helper)

env -u DS_BASE_URL -u SGLANG_DS_RADIX_FIXTURE_CAPTURE \
    -u SGLANG_DS_FP8_SCALE_PROOF \
  PYTHONPATH=python pytest \
   test/manual/test_dsv32_radix_cache_fixture.py \
   test/manual/test_dsv32_radix_label_capture_fixture.py \
   test/manual/test_dsv32_fp8_scale_stability.py -q
3 skipped (all three manual fixtures skip cleanly when env unset)

bash -n development/serve_double_sparsity.sh   # OK
```

Commit: `a41b1d952` — [AC-10] M3-B capture primitive + direct-
evidence fixtures + identical-prompt smoke.

## Remaining Items

AC-10 hardware execution (operator-driven, not in this loop):

1. Boot DS server with `SGLANG_DS_RADIX_OVERRIDE=1` +
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` and radix cache ON.
2. Run `test_dsv32_fp8_scale_stability.py` with
   `SGLANG_DS_FP8_SCALE_PROOF=1`. On pass, save the artifact.
3. Run `test_dsv32_radix_label_capture_fixture.py`. On pass, save
   the artifact.
4. (Optional) Run the continuation smoke first as a sanity check.
5. Wire `record_radix_fixture_passed(server_args, artifact_path=
   "<label-capture-artifact.json>")` into a launcher init module
   BEFORE `validate_double_sparsity` runs.
6. Delete the `--disable-radix-cache` line in
   `serve_double_sparsity.sh` (the AC-10-FIXTURE-MARKER comment
   names the exact line).
7. Update `test_ds_server_does_disable_radix_cache_until_ac10`
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

Branch is 37 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 36 followed `BL-20260527-shell-json-into-python-source`
implicitly (the new capture fixture passes JSON-shaped payloads
through env vars + HTTP, not through Python source splicing).
Bitlesson-selector returned `NONE`: the existing entries do not
cover the "env-gated zero-cost capture hook + SHA audit trail"
pattern this round introduces, but the pattern is one-shot
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-35-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-35-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-34-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-34-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-33-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-33-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-36-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
