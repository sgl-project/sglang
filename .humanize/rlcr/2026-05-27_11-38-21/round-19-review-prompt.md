# FULL GOAL ALIGNMENT CHECK - Round 19

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 19 Summary

## Work Completed

### task-ac6-cuda-graph — ForwardContext lookup + int64 input handling

Codex Round 18 review left two open gaps:

1. **Wrong production object path.** `_select_topk_indices` looked for
   `ds_graph_state` at `forward_batch.attn_backend.forward_metadata` — but
   real `ForwardBatch` has no `attn_backend` field. Production publishes the
   attention backend through `ForwardContext`, the same source the AC-7 MHA
   bypass already uses. Codex's spy probe recorded
   `retrieve_topk_graph_safe` call count 0 in production.

2. **Wrong dtype assumption.** Production `req_pool_indices` is int64
   (`schedule_batch.py:1507` + `cuda_graph_runner.py:178`); the new graph-
   safe fast path asserted int32. Codex's probe raised
   `AssertionError: req_pool_indices must be int32, got torch.int64`.

Both closed.

#### Metadata resolution

- `_select_topk_indices` resolves `ds_graph_state` in two steps (mirrors
  the AC-7 MHA bypass):
  1. `forward_batch.ds_graph_state` — primary, set by
     `dsa_backend.init_forward_metadata` for dynamic non-graph forwards.
  2. `has_forward_context() and get_attn_backend().forward_metadata.ds_graph_state`
     — fallback for the CUDA-graph capture/replay path.
- `dsa_backend.init_forward_metadata` now also assigns
  `forward_batch.ds_graph_state = ds_graph_state` next to
  `forward_batch.ds_topk_indices_out`.

#### int64 → int32 scratch via copy_

- Added two new fields to `DSGraphState`:
  `scratch_req_pool_indices: int32[max_bs]` and
  `scratch_seq_lens: int32[max_bs]`.
- `_select_topk_indices` does an in-place `copy_()` from
  production int64 `req_pool_indices` into the int32 scratch, then passes
  the scratch view to `retrieve_topk_graph_safe`.
- For `seq_lens`: prefers `DSAMetadata.cache_seqlens_int32[:bs]` when the
  metadata is reachable (the int32 view is already maintained per batch
  in `dsa_backend`); otherwise `copy_()` into `scratch_seq_lens`.

#### Bonus — allocation-free `logical_to_physical`

While tracing the production-path 0-alloc requirement we discovered
`logical_to_physical` allocated ~8 intermediates per call (clamp / sum /
where / full_like). Replaced the torch path with a single Triton kernel:

- `_logical_to_physical_kernel` (page_table_adapter.py): grid
  `(bs, ceil(max_top_k / BLOCK_K))`; gathers `req_to_token[safe_pool,
  safe_pos]`, masks padding + bad-pool rows to `-1`, atomically increments
  an int32 `error_scratch` for bad pool indices.
- `lp_error_scratch: int32[1]` is now part of `DSGraphState`. The
  `int(error_scratch.item())` host sync is skipped during stream capture
  (returns 0 conservatively); callers that need the count outside capture
  still get it.
- CPU + missing-scratch paths fall back to the original torch
  implementation — unit tests on CPU stay green.

#### Capture-safe `_publish_ds_request_summary`

The per-request CPU-side summary publication does
`valid_lengths.detach().to("cpu").tolist()` (a D2H sync that is illegal
during stream capture). Gated on
`not torch.cuda.is_current_stream_capturing()`, matching the
established pattern in `retrieve_topk_via_labels`.

### Tests

- **Replaced** `test_select_topk_indices_uses_graph_safe_when_metadata_state_present`
  with `..._via_forward_context`: publishes only a real
  `ForwardContext(attn_backend=...)`, no synthetic
  `forward_batch.attn_backend`. Production-dtype int64
  `req_pool_indices` + int64 `seq_lens` + int32 `sparse_mask`. Spies the
  dynamic import of `retrieve_topk_graph_safe` at the kernel module;
  asserts it is called exactly once.
- **Added** `test_select_topk_indices_zero_allocs_production_path`:
  captures `_select_topk_indices` into a real `torch.cuda.CUDAGraph`
  with the same production-dtype fixture; replays 5 times wrapped in
  `assert_no_alloc_in_region`; verifies 0 new CUDA allocations on every
  replay and that `ds_topk_indices_out` carries the expected physical
  slots from `logical_to_physical`. This mirrors how
  `cuda_graph_runner.py` captures the full decode forward in production.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/cuda_graph.py`:
  added `scratch_req_pool_indices`, `scratch_seq_lens`, `lp_error_scratch`
  fields to `DSGraphState`; allocated in `allocate_graph_state` under
  `max_seq_len > 0`.
- `python/sglang/srt/layers/attention/double_sparsity/page_table_adapter.py`:
  added Triton kernel `_logical_to_physical_kernel`; rewrote
  `logical_to_physical` with the alloc-free CUDA fast path + torch
  fallback; new `error_scratch` keyword argument; capture-safe error
  read.
- `python/sglang/srt/layers/attention/dsa_backend.py`:
  `init_forward_metadata` exposes `forward_batch.ds_graph_state` for
  the dynamic path.
- `python/sglang/srt/models/deepseek_v2.py::_select_topk_indices`:
  resolves `ds_graph_state` via `forward_batch` primary +
  `ForwardContext` fallback; `copy_()` int64 inputs into int32 scratch;
  passes `error_scratch` to `logical_to_physical`; gates
  `_publish_ds_request_summary` on capture state.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  added `_make_production_forward_batch` helper; replaced the
  attn_backend-mocked spy test with the `ForwardContext`-based one;
  added the CUDA-graph replay zero-alloc test.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
200 passed, 0 failed (was 199 before this round)
```

Targeted:
```
pytest -v -k "test_select_topk_indices_uses_graph_safe_via_forward_context"     # 1 passed
pytest -v -k "test_select_topk_indices_zero_allocs_production_path"             # 1 passed (5 replays x 0 allocs)
```

Commit: `8ab3c332a` — [AC-6] Resolve DS metadata via ForwardContext + int64 input dtypes.

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real
  V3.2 H200 cluster. The coding path is now production-ready: the
  graph-safe selector is reached via `ForwardContext`, production int64
  inputs are handled, `logical_to_physical` is alloc-free, and
  `_select_topk_indices` survives a real `torch.cuda.CUDAGraph` capture
  + replay with 0 new allocations.
- `task-ac4-hwrun`: hardware gate — H200 channel-mask calibration.
- Hardware-gated: `task-ac1-hwtest`, `task-ac1b-probe`, `task-ac8-*`,
  `task-ac12-*`.
- Queued (non-blocking): AC-8 page-vs-token unit mix-up in
  `_publish_ds_request_summary`; stale DS bind/runtime comments;
  token-label lifetime docs.

## Push-to-remote Status

User requested per-commit push to GitHub. The RLCR loop's
`loop-bash-validator.sh` hook still blocks `git push`; commits are saved
locally on `dev/double-sparsity-standalone` (Round 19 commit
`8ab3c332a`, 20 commits ahead of `jimmy/dev/double-sparsity-standalone`).
To enable per-round pushes, re-launch the loop with
`--push-every-round`. The commits will not be lost — local refs are
intact.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 19's lesson — "production metadata is published via
`ForwardContext`, not `forward_batch.attn_backend`" — is already
documented in the live AC-7 MHA bypass code at the top of
`_select_topk_indices`. The Round 19 fix copied that exact pattern.
No new BitLesson warranted; the existing inline comments in the bypass
make the pattern discoverable for future writers.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-18-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-18-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-17-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-17-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-16-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-16-review-result.md


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

To implement the original plan at @development/loop4/refined_plan_v1.md, we have completed **20 iterations** (Round 0 to Round 19).

The project's `.humanize/rlcr/2026-05-27_11-38-21/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-27_11-38-21/round-18-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-17-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-18-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-19-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-19-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
