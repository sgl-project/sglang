# Code Review - Round 18

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-18-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 18 Summary

## Work Completed

### task-ac6-cuda-graph — Production wiring + production dtypes

Codex Round 17 review left two open gaps:

1. The graph-safe `retrieve_topk_graph_safe` was only invoked from
   `cuda_graph.py::capture_decode_step` — a helper that is **not** referenced
   by the production DeepSeek decode path. Production
   `_select_topk_indices` was still calling
   `DoubleSparsitySelector.retrieve_topk` → `retrieve_topk_via_labels` (the
   allocating path, ~47 allocs per step).

2. The CUDA fast path of `retrieve_topk_graph_safe` cast `sig_layer` and
   non-fp32 `queries` with `.to(torch.float32)`. Under production dtypes
   (fp16 TokenLabelTable + bf16 / fp16 queries), those casts allocate fresh
   tensors. A Codex CUDA probe at production dtypes reported 2 new
   allocations after warmup.

Both are now closed.

#### Production wiring

- **`DSAMetadata` gained `ds_graph_state: Optional[DSGraphState]`** (dsa_backend.py).
- **Both metadata-init sites allocate it** when DS is enabled:
  - `init_forward_metadata` (extend / dynamic decode, ~line 715)
  - `init_forward_metadata_capture_cuda_graph` (CUDA graph capture, ~line 1015)
  - Sizing: `max_bs=bs, max_top_k=self.ds_max_top_k, max_seq_len=req_to_token.shape[1]`.
- **`deepseek_v2.py::_select_topk_indices`** detects
  `forward_batch.attn_backend.forward_metadata.ds_graph_state` and, when
  the selector is bound and tensors are CUDA, calls
  `retrieve_topk_graph_safe` directly with all scratch +
  `per_request_valid=sparse_mask`. Falls back to
  `DoubleSparsitySelector.retrieve_topk` only when scratch is absent (CPU
  tests / unbound selector / synthetic forward_batch without
  attn_backend).
- The downstream `logical_to_physical(..., out=ds_topk_indices_out)`
  conversion is unchanged.

#### Production dtypes

- **Removed all `.to(...)` casts** from the CUDA fast path of
  `retrieve_topk_graph_safe`. The `_logical_score_kernel` already loads
  fp16/bf16/fp32 q + sig pointers and casts via
  `tl.load(...).to(tl.float32)` inside the kernel.
- **Added contract asserts** in the fast path: `channel_selection int32`,
  `channel_weights fp32`, `req_pool_indices / req_to_token / seq_lens
  int32`. Any drift fails fast with a clear message instead of silently
  allocating per call.
- **Added bind-time asserts** in `DoubleSparsitySelector.bind_runtime_data`
  for `channel_selection.dtype == int32` and
  `channel_weights.dtype == float32`. The channel mask is the only tensor
  whose dtype the selector can enforce at bind time; the token
  signatures stay at the binder's choice (fp16 in production).

### Tests

- **`test_retrieve_topk_graph_safe_zero_allocs_production_dtypes`** (CUDA-only):
  Constructs a fp16 `TokenLabelTable` (production default) + bf16 queries
  + int32 `sparse_mask` (as `per_request_valid`). Warms up once; wraps
  the second call in `assert_no_alloc_in_region`. Closes Codex's
  prod-dtype CUDA probe (was 2 new allocs; is now 0).
- **`test_select_topk_indices_uses_graph_safe_when_metadata_state_present`**:
  Drives the actual `_select_topk_indices` method on a real-mode
  selector with a synthesized `forward_batch.attn_backend.forward_metadata.ds_graph_state`.
  Spies the dynamic import of `retrieve_topk_graph_safe` at the kernel
  module — asserts the spy is called exactly once. Closes Codex's
  production-path regression requirement.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`:
  removed `.to(torch.float32)` / `.to(torch.int32)` from CUDA fast path;
  added int32/fp32 contract asserts; documented the new caller contract.
- `python/sglang/srt/layers/attention/double_sparsity/selector.py`:
  added bind-time `channel_selection.dtype == int32` /
  `channel_weights.dtype == float32` asserts.
- `python/sglang/srt/layers/attention/dsa_backend.py`:
  added `DSGraphState` import; added
  `DSAMetadata.ds_graph_state: Optional[DSGraphState]`; allocated it in
  both `init_forward_metadata` sites (extend/decode dynamic + CUDA graph
  capture).
- `python/sglang/srt/models/deepseek_v2.py::_select_topk_indices`:
  added `_use_graph_safe` gate; when scratch is present + selector is
  bound + CUDA tensors, calls `retrieve_topk_graph_safe` directly with
  metadata-owned scratch; falls back to the legacy path otherwise.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  added `_make_bound_selector_cuda_fp16` helper; added the two new
  tests above.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
199 passed, 0 failed (was 197 before this round)
```

Targeted runs:
```
pytest -v -k zero_allocs_production           # 1 passed (CUDA fp16+bf16 0-alloc)
pytest -v -k test_select_topk_indices_uses_graph_safe  # 1 passed (production-path spy)
```

Commit: `38ef74765` — [AC-6] Wire allocation-free DS path into production + production dtypes.

## Remaining Items

- `task-ac6-hwrun`: hardware gate — full-graph capture at conc=64 on real
  V3.2 H200 cluster. The coding path is now production-ready (the
  graph-safe selector is invoked in the actual decode path, and the
  fast path is 0-alloc at production dtypes).
- `task-ac4-hwrun`: hardware gate — H200 channel-mask calibration.
- Hardware-gated: `task-ac1-hwtest`, `task-ac1b-probe`, `task-ac8-*`,
  `task-ac12-*`.
- Queued (non-blocking): AC-8 page-vs-token unit mix-up in
  `_publish_ds_request_summary`; stale DS bind/runtime comments;
  token-label lifetime docs.

## Push-to-remote Note

User requested per-commit push for crash safety. The RLCR loop's
`loop-bash-validator.sh` hook blocks `git push` unless the loop is
launched with `--push-every-round`. Commits are saved locally on
`dev/double-sparsity-standalone`; commit `38ef74765` is Round 18 and is
ahead of `jimmy/dev/double-sparsity-standalone` by 19 commits. Push
manually if you take over the shell, or relaunch the loop with the
`--push-every-round` flag.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 18 is a re-wiring round on top of Round 17's allocation-free
pipeline. The principle worth remembering — `tensor.to(dtype)` is NOT
allocation-free even when the caching allocator returns reused memory
(the `allocation.all.allocated` counter still increments) — is implicit
in the prior `BL-20260527-torch-topk-aliasing-corrupts-input` lesson's
"every host op must be allocation-free" framing. No new entry warranted.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-17-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-17-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-16-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-16-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-15-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-15-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-18-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
