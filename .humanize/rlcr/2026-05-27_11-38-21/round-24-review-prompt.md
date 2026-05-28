# FULL GOAL ALIGNMENT CHECK - Round 24

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 24 Summary

## Work Completed

Codex Round 23 review verified the locked Option B flags and the
16/32/64 concurrency sweep, but caught a blocking bug in the new
benchmark sidecar writer:

> `development/benchmark.sh:71-83` splices raw `/get_server_info` JSON
> into a Python heredoc as source code. `/get_server_info` legitimately
> returns booleans (`true` / `false`) and nulls (`null`) — valid JSON
> tokens but invalid Python identifiers. After a successful H200
> benchmark run, sidecar generation raises `NameError`. The Round 23
> contract also required `chunked_prefill_size`; the implementation
> did not include it.

Same bug in `benchmark_baseline.sh`. Without sidecars the AC-11
reproducibility audit cannot run — both AC-8 and AC-9 hardware
artifacts would be missing the operating-point fingerprint.

### Fix 1 — `development/_bench_meta_writer.py` (new helper)

Pure-Python standalone script. Reads every metadata field from
environment variables. Parses `SERVER_ARGS_JSON` with `json.loads`
inside Python — never splices JSON as source. On parse failure
(empty / malformed / non-object) records `server_args = {}` and a
short `server_args_error` string so the operator can diagnose offline.

Schema (every field always present; missing values are `null`):

- `commit_sha`, `mode`, `concurrency`, `seed`, `num_prompts`,
  `isl_total_tokens`, `osl_tokens`, `timestamp_utc`.
- `chunked_prefill_size` — extracted from `server_args` (was missing
  in Round 23 despite the contract).
- `warmup_requests`, `measurement_window_seconds`, `trial_id` —
  AC-11 reproducibility fields Codex called out.
- `server_args` — parsed JSON object (or `{}` on error).
- `server_args_error` — `null` or a short reason.

The Round 23 JSON-as-Python-source splice is now structurally
impossible.

### Fix 2 — Both benchmark scripts call the helper

`development/benchmark.sh` and `benchmark_baseline.sh` replace the
inline `python3 - <<PYEOF` heredoc with:

```bash
COMMIT_SHA="${COMMIT_SHA}" MODE="${MODE}" ... \
SERVER_ARGS_JSON="${SERVER_ARGS_JSON}" \
TRIAL_ID="${TRIAL_ID:-1}" \
WARMUP_REQUESTS="${WARMUP_REQUESTS:-}" \
MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-}" \
python3 "$(dirname "$0")/_bench_meta_writer.py" > "${META_FILE}"
```

All JSON now travels via env-var data, not Python source. Operators
who want to record 3-trial AC-11 evidence can override
`TRIAL_ID=1/2/3`, `WARMUP_REQUESTS=120`, `MEASUREMENT_WINDOW_S=600`.

### Fix 3 — `test/registered/unit/development/test_bench_meta_writer.py`

10 registered tests invoke `_bench_meta_writer.py` as a subprocess
with controlled env vars:

- Realistic JSON with `true` / `false` / `null` / nested dicts
  produces valid JSON, preserves types (`is True`, `is None`,
  nested object identity), and extracts `chunked_prefill_size: 4096`.
- Empty `SERVER_ARGS_JSON` → `server_args = {}`, error contains
  `"empty"`.
- Malformed `SERVER_ARGS_JSON` (`"{not json"`) → error contains
  `"parse_error"`.
- Non-object JSON (`"[1, 2, 3]"`) → error contains `"not_object"`.
- `TRIAL_ID` defaults to `"1"`; env override works.
- AC-11 reproducibility fields always present (even when null);
  numeric overrides parse correctly.
- Output is multi-line pretty-printed JSON, re-parseable.

### Fix 4 — `test_option_b_scripts.py` reinforced

Replaced the shallow `commit_sha` / `server_args` string assertions
(Codex called them "shallow" in the review) with explicit
no-go assertions:

- Both benchmark scripts MUST reference `_bench_meta_writer.py`.
- Both benchmark scripts MUST NOT contain `PYEOF` (the unsafe
  heredoc terminator).
- Both benchmark scripts MUST NOT contain
  `"server_args": ${SERVER_ARGS_JSON` (the JSON-as-source splice).
- Both benchmark scripts MUST pass `SERVER_ARGS_JSON="${SERVER_ARGS_JSON}"`
  as an env-var assignment to the helper.

So any future "small cleanup" that reintroduces the Round 23 bug
fails this test.

## Files Changed

- `development/_bench_meta_writer.py`: NEW — standalone, env-var-driven,
  json.loads-safe sidecar writer.
- `development/benchmark.sh`, `development/benchmark_baseline.sh`:
  replaced embedded heredoc with subprocess call to the helper;
  added `TRIAL_ID`, `WARMUP_REQUESTS`, `MEASUREMENT_WINDOW_S` env
  pass-throughs.
- `test/registered/unit/development/test_bench_meta_writer.py`: NEW —
  10 tests against the writer.
- `test/registered/unit/development/test_option_b_scripts.py`:
  reinforced 2 sidecar-emit tests with helper-reference + heredoc-forbid
  assertions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development -q
23 passed, 0 failed (was 10)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development -q
229 passed, 0 failed (was 216 before this round)

bash -n development/benchmark.sh         # OK
bash -n development/benchmark_baseline.sh # OK
```

Manual writer smoke (the realistic JSON shape Codex named):

```
SERVER_ARGS_JSON='{"disable_radix_cache": true, "kv_events": null,
                   "chunked_prefill_size": 4096, "tp_size": 8}' \
python3 development/_bench_meta_writer.py
# → valid JSON, server_args["disable_radix_cache"]=true (round-trip),
#   chunked_prefill_size=4096, server_args_error=null.
```

Commit: `856ab7356` — [AC-8/9] Fix Round 23 sidecar JSON injection
bug; add chunked_prefill_size + AC-11 fields.

## Remaining Items

Code-tier items queued for future rounds:

- **AC-12 scaffold replacement** (`test/manual/test_double_sparsity_v32.py`
  still skip-only NIAH 4K/16K/64K + MMLU 5-shot). HARD loop-closure
  gate.
- **`benchmark_compare.py` AC-11 directional gate**: 3-trial median
  per concurrency, DS TPS within 5% of DSA, P99 TTFT ≤ 1.10× DSA;
  currently still the absolute-SLO single-trial framing.
- Shallow AC-8 prefix-match regression coverage (extract `_prefix_match`
  or mock through `test_quality_smoke`).
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality`.

## Push-to-remote Status

Branch is 25 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-shell-json-into-python-source
Notes: This was a clean, generalizable failure mode: shell scripts that
inject a captured JSON string into a Python heredoc as source code
work fine on test fixtures (objects of strings + ints) but fail on
real JSON containing `true` / `false` / `null` (valid JSON tokens,
invalid Python identifiers). Worth recording so future scripts use
env-var + `json.loads` from the start.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-23-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-23-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-22-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-22-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-21-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-21-review-result.md


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

To implement the original plan at @development/loop4/refined_plan_v1.md, we have completed **25 iterations** (Round 0 to Round 24).

The project's `.humanize/rlcr/2026-05-27_11-38-21/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-27_11-38-21/round-23-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-22-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-23-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-24-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-24-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
