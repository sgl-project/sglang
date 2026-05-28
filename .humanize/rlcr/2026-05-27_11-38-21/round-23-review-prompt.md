# Code Review - Round 23

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-23-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 23 Summary

## Work Completed

Codex Round 22 review verified Round 22's gate fixes were clean and
listed the four remaining Loop 4 code-tier items: AC-12 scaffold
replacement, server-launcher Option B alignment, benchmark sweep
alignment, and `benchmark_compare.py` 3-trial-median + AC-11
directional gate. Round 23 lands the second and third (server +
benchmark scripts) so AC-8 / AC-9 hardware runs can produce
plan-conformant artifacts at the locked operating point.

### Fix 1 — `serve_double_sparsity.sh`

Added the four Option B locked flags Codex listed:
- `--dsa-prefill-backend flashmla_kv`
- `--dsa-decode-backend flashmla_kv`
- `--disable-overlap-schedule`
- `--disable-piecewise-cuda-graph`

Kept `--disable-radix-cache` (AC-10 still pending). Header docstring
updated to enumerate the locked Option B set.

### Fix 2 — `serve_native_nsa.sh`

Added the same four locked flags so DSA + DS launchers differ only
by DS-specific enablement and the AC-10 radix gate. Deliberately did
NOT add `--disable-radix-cache` — per plan §13 the DSA baseline runs
with radix cache ON so the DS-vs-DSA TPS gap reflects DS configuration
alone (the AC-10 radix gate is a separate AC).

### Fix 3 — `benchmark.sh`

- Default `CONCURRENCIES="${CONCURRENCIES:-16 32 64}"` (was `"64"`)
  matches the AC-8 / AC-9 spec.
- Outputs land in `${RESULTS_DIR}` (defaults to
  `$(pwd)/development/results/`) instead of cwd.
- Each run emits a `${OUTPUT_FILE}.meta.json` sidecar capturing
  `commit_sha`, `mode`, `concurrency`, `seed`, `num_prompts`,
  `isl_total_tokens`, `osl_tokens`, `timestamp_utc`, and the
  server's `/get_server_info` JSON (full server args). The AC-11
  comparator can verify both columns share the same operating point.
- Best-effort: `commit_sha` from `git rev-parse HEAD`;
  `server_args` from `curl -s --max-time 5 http://${HOST}:${PORT}/get_server_info`
  (writes `{}` if unreachable, so CI environments do not break).

### Fix 4 — `benchmark_baseline.sh`

Same updates as `benchmark.sh` (default conc 16/32/64, results dir,
meta sidecar), keyed by `MODE=native_nsa` to stay paired with the DS
benchmark.

### Fix 5 — Regression test locking the contract

New file `test/registered/unit/development/test_option_b_scripts.py`:

- **`TestOptionBLockedFlagsServerScripts`** (5 tests):
  - All 4 scripts exist.
  - Both servers carry all 4 locked-flag tokens.
  - DSA does NOT pass `--disable-radix-cache`.
  - DS does pass `--disable-radix-cache` (AC-10 gate).
- **`TestOptionBBenchmarkSweeps`** (4 tests):
  - Both bench scripts default `CONCURRENCIES` to `"16 32 64"`
    (regex on the bash parameter-expansion line).
  - Both bench scripts emit `.meta.json` sidecars carrying
    `commit_sha` + `server_args`.
- **`TestOptionBScriptsSyntax`** (1 test):
  - `bash -n` parses all 4 scripts cleanly.

Helper: `_non_comment_lines(path)` strips `#` lines so assertions
test the *active* command, not the docstring (so a comment
mentioning "does NOT pass --disable-radix-cache" does not
accidentally match the assertion).

## Files Changed

- `development/serve_double_sparsity.sh`: 4 locked Option B flags +
  header docstring update.
- `development/serve_native_nsa.sh`: same 4 locked flags + header
  docstring update explicitly explaining the no-radix-disable
  decision.
- `development/benchmark.sh`: rewritten with conc 16/32/64 default,
  results-dir output, and `.meta.json` sidecar emission.
- `development/benchmark_baseline.sh`: same as benchmark.sh with
  `MODE=native_nsa`.
- `test/registered/unit/development/test_option_b_scripts.py`: NEW
  10-test class locking the contract.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development/test_option_b_scripts.py -q
216 passed, 0 failed (was 206 before this round; +10 Option B regressions)

bash -n development/serve_double_sparsity.sh   # OK
bash -n development/serve_native_nsa.sh        # OK
bash -n development/benchmark.sh               # OK
bash -n development/benchmark_baseline.sh      # OK
```

Targeted greps:
```
grep -E 'dsa-prefill-backend flashmla_kv'      development/serve_*.sh   # 2 hits
grep -E 'dsa-decode-backend flashmla_kv'       development/serve_*.sh   # 2 hits
grep -E 'disable-overlap-schedule'             development/serve_*.sh   # 2 hits
grep -E 'disable-piecewise-cuda-graph'         development/serve_*.sh   # 2 hits
grep -E 'disable-radix-cache'                  development/serve_native_nsa.sh   # 1 (comment only — explanation)
grep -E 'disable-radix-cache'                  development/serve_double_sparsity.sh # 2 (active flag + AC-10 comment)
grep -E 'CONCURRENCIES:-16 32 64'              development/benchmark*.sh   # 2 hits
grep -E '\.meta\.json'                         development/benchmark*.sh   # multiple
```

Commit: `3ab86e868` — [AC-8/9] Align Option B launchers + benchmark
sweeps to plan §13.

## Remaining Items

Mainline AC items still requiring hardware execution:
- `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`,
  `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`,
  `task-ac9-baseline`, `task-ac10-radix`, `task-ac11-compare`,
  `task-ac12-quality`.

Code-tier items still queued for future rounds:
- Replace `test/manual/test_double_sparsity_v32.py` skip-only scaffolds
  with real NIAH 4K/16K/64K + MMLU 5-shot execution against paired
  servers (AC-12).
- `benchmark_compare.py` 3-trial median + AC-11 directional gate
  enforcement (DS TPS within 5% of DSA; P99 TTFT ≤ 1.10× DSA).
- Shallow prefix-match regression coverage cleanup (Codex Round 22
  queued).
- Stale DS bind/runtime comments + token-label lifetime docs.

## Push-to-remote Status

Branch is 24 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`; commits remain local only. Per-round pushing requires
re-launching with `--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: This was a mechanical, plan-derived alignment of dev-tier
scripts to the locked operating point. No surprising failure mode
or non-obvious workaround; the regression test exists so the
contract is locked, but the bug pattern itself is too narrow to
generalize. No new BitLesson entry warranted.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-22-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-22-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-21-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-21-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-20-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-20-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-23-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
