# Code Review - Round 21

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-21-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 21 Summary

## Work Completed

Codex Round 20 review verified the AC-6 production-object fix and
opened the next plan-derived bottleneck: the AC-8 quality smoke harness
file (`test/manual/test_dsv32_quality_smoke.py`) did not exist, and
the observability publication still reported `selected_pages` against
`total_pages = (seq_len + page_size - 1) // page_size`, which is the
wrong unit after the AC-0 token-level rotation.

Round 21 lands both:

### Fix 1 — AC-8 quality smoke harness (new file)

`test/manual/test_dsv32_quality_smoke.py`:

- 20 deterministic prompts at `temperature=0`, `max_new_tokens=256`.
- 5 NIAH-mini needle-in-haystack prompts with unique sentinel needles
  (ZEBRA-7, MARLIN-42, ORCHID-99, GLACIER-13, PHARAOH-88).
- Paired DSA/DS HTTP queries per plan §9.4 (DSA reference first, same
  session immediately before DS).
- Four assertion gates:
  1. `prefix_match_rate >= 0.80` — DS first 32 chars match DSA's.
  2. `mean_rouge_l >= 0.85` — pure-Python LCS-based ROUGE-L F-measure
     (no `rouge_score` dependency; harness imports in any environment).
  3. `niah_mini_recall >= 4/5` — needle string present in DS response.
  4. `first_8_tokens_divergences == 0` — no prompt where DS and DSA's
     first 8 tokens fail to share any common token.
- Best-effort commit SHA capture from `/get_server_info` for both
  servers; written into `development/results/dsv32_quality_smoke_<ts>.json`.
- Cleanly skips when `DS_BASE_URL` or `DSA_BASE_URL` env vars are unset
  — `pytest test/manual/test_dsv32_quality_smoke.py` without env vars
  yields `1 skipped`.

### Fix 2 — token-vs-page units in `_publish_ds_request_summary`

After the AC-0 token-level rotation the selector emits TOKEN counts in
`valid_lengths`, but the observability layer still labeled the field
`selected_pages` and divided by `page_size` for the sparsity denominator.
That yielded `sparsity_rate = 1.0 - 30 / 2 = -14` when 30 tokens were
selected from a 100-token, page-64 request — wrong unit, wrong sign.

Renames (consistent across module + metrics):

- `DoubleSparsityRequestStats.selected_pages` → `selected_tokens`.
- `meta_info["selected_pages"]` → `meta_info["selected_tokens"]`.
- `record_selection(selected_pages, total_valid_pages)` →
  `record_selection(selected_tokens, total_valid_tokens)`.
- Prometheus metrics
  `sglang_double_sparsity_selected_pages_{sum,count}` →
  `..._selected_tokens_{sum,count}`. Pre-MVP rename — no external
  dashboard consumers depend on the old names yet.
- `_publish_ds_request_summary`: dropped the `page_size` division;
  `total_tokens = max(1, int(sl_cpu[b]))`; sparsity_rate computed
  against tokens. Error-path record (`error_class != ok` branch) also
  publishes `selected_tokens`.

### Regression unit test

`TestR5Coverage::test_publish_ds_request_summary_uses_token_denominator`:
constructs a bs=2 forward_batch (seq_lens = [100, 256], valid_lengths
= [30, 5]), drives `_publish_ds_request_summary` directly, asserts:
- `selected_pages` key is gone, `selected_tokens` is present.
- Row 0 sparsity_rate == 1 - 30/100; row 1 == 1 - 5/256.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/metrics.py`:
  field/param/metric renames; updated module docstring.
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`:
  updated `record_selection(...)` call site to new kw names.
- `python/sglang/srt/models/deepseek_v2.py::_publish_ds_request_summary`:
  token-denominator math; renamed published field; error-path record
  rename.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  renamed references in metric-counter assertion tests + meta_info
  shape tests + customized_info tests + test_select_topk_indices error
  fixtures; added the new regression test.
- `test/manual/test_dsv32_quality_smoke.py`: NEW file with the AC-8
  quality smoke harness.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
202 passed, 0 failed (was 201 before this round)

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_dsv32_quality_smoke.py -v
test_quality_smoke SKIPPED — clean skip when env vars are unset
```

Targeted:
```
pytest -v -k "test_publish_ds_request_summary_uses_token_denominator"   # 1 passed
```

Helper sanity:
```
_rouge_l_f(s, s)                = 1.000
_rouge_l_f(s, different)        = 0.167  (small, expected)
_first_n_tokens_match(...)      = True/False as expected
SMOKE_PROMPTS = 20, NIAH_MINI_PROMPTS = 5
```

Commit: `a586f814a` — [AC-8] Quality smoke harness + token-denominator
observability fix.

## Remaining Items

Mainline AC items still requiring hardware execution:
- `task-ac8-server` + `task-ac8-quality` (harness exists; needs paired DS+DSA H200 servers + run).
- `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`.
- `task-ac12-quality` (separate file `test_double_sparsity_v32.py` still skip-only — separate round).

Code-tier items still queued for future rounds (not Round 21 scope):
- Replace `test_double_sparsity_v32.py` skip scaffolds with real
  NIAH 4K/16K/64K + MMLU 5-shot logic.
- Update `serve_double_sparsity.sh` + `serve_native_nsa.sh` to encode
  the locked Option B flags.
- `benchmark.sh` + `benchmark_baseline.sh` conc 16/32/64 sweep.
- `benchmark_compare.py` 3-trial median + AC-11 directional gate.
- Stale DS bind/runtime comments + token-label lifetime docs.

## Push-to-remote Status

Branch is 22 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`; commits remain local on `dev/double-sparsity-standalone`.
Per-round pushing requires re-launching with `--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The page-vs-token unit mix-up was the kind of bug that comes from
incomplete refactors after an architecture rotation — but the BitLesson
file already covers reshape/slice ordering for MLA outputs
(`BL-20260527-reshape-before-slice-mla`) which is the closest general
principle ("always re-check derived names + per-head shapes after a
rotation"). A standalone "rename observability fields after rotation"
lesson would be too narrow to be useful. No new entry warranted.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-20-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-20-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-19-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-19-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-18-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-18-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-21-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
