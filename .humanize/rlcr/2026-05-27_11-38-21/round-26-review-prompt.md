# Code Review - Round 26

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-26-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 26 Summary

## Work Completed

Codex Round 25 review caught two blocking bugs in my AC-12 harness:

1. **MMLU broken** — passed root `DS_BASE_URL` to `run_eval_once`, but
   sglang's OpenAI-compatible endpoints live under `/v1`; worse, the
   reused `simple_eval_mmlu.MMLUEval` is 0-shot, not the plan-required
   5-shot.
2. **Server-side fault-injection plumbing missing** — Round 25
   deferred it; Codex rejected that as unjustified.

Plus one queued artifact-writer omission Codex flagged: NIAH JSON was
missing hit counts.

Round 26 closes all three plus all of Codex's requested helper
regressions.

### Fix 1 — Real MMLU 5-shot

`test/manual/test_double_sparsity_v32.py`:

- `_openai_base_url(url)` helper that normalizes any operator-supplied
  URL to `.../v1` (idempotent; case-insensitive suffix check).
  Reserved for future OpenAI-compatible callers; the 5-shot path uses
  `/generate` directly.
- New helpers `_format_mmlu_subject`, `_format_mmlu_example`,
  `_make_mmlu_5shot_prompt`, `_parse_mmlu_letter` — mirror the
  formatting from `benchmark/mmlu/bench_sglang.py:25-52`.
- `test_mmlu_5shot` body fully rewritten:
  - Loads MMLU CSVs from `benchmark/mmlu/data/` (operator override
    `AC12_MMLU_DATA_DIR`); skips cleanly if the data dir is absent
    (`benchmark/mmlu/bench_sglang.py` auto-fetches on first run).
  - Builds a single shuffled-deterministic list of test questions
    across all subjects (override `AC12_MMLU_SUBJECTS=foo,bar` to
    narrow); honors `AC12_MMLU_NUM_EXAMPLES` cap (default 200).
  - For each question: 5-shot prompt = header + dev[:5] (answered)
    + test question (bare `Answer:`). Queries `/generate` with
    `max_new_tokens=4`, `temperature=0`. First A-D character in the
    response is the prediction.
  - Per-subject + aggregate scoring persisted to
    `development/results/ac12_mmlu_5shot_<ts>.json`.
  - Gate: `dsa_score_pct - ds_score_pct ≤ 1.0`.

### Fix 2 — Server-side fault-injection gates

`python/sglang/srt/models/deepseek_v2.py` (after `slice_per_rank` at
line ~1893):

- `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK=1` replaces `channel_selection`
  with a deterministically-per-layer random selection from
  `[0, head_dim)`. Uses `torch.Generator().manual_seed(self.layer_id)`
  so the corruption is reproducible. Sampling without replacement
  when `label_dim ≤ head_dim` (the production case for V3.2
  `head_dim=128`, `label_dim=16`); falls back to `torch.randint` for
  the edge case. Preserves shape / dtype / device + `channel_weights`.
  Logs a clear WARNING per layer.

`python/sglang/srt/layers/attention/dsa_backend.py`:

- `NativeSparseAttnBackend.__init__` caches
  `self._ds_fault_zero_sig = os.getenv("SGLANG_DS_FAULT_INJECT_ZERO_SIG")
  == "1"`. Logs a WARNING once per process when set.
- `_write_token_labels` after the `token_label_write` call:
  ```python
  if getattr(self, "_ds_fault_zero_sig", False):
      self._ds_token_label_table.signatures[layer_id, cache_loc] = 0
  ```
  Uses `getattr` so existing `object.__new__(backend)` unit fixtures
  that don't run `__init__` keep working (6 R1 tests would otherwise
  regress).
- `written[layer_id, cache_loc] = True` stays — the selector sees
  intentionally bad labels, not absent slots, which is what the
  zero-signature negative gate exercises.

### Fix 3 — NIAH artifact JSON carries hit counts

`_niah_assert` + the two sensitivity recorders now include the
underlying `dsa_hits` / `ds_hits` (and `ds_corrupt_hits` /
`ds_zero_hits` for sensitivity variants) — the audit can replay the
gate without re-running the H200 servers.

### Fix 4 — Registered helper regressions (10 new)

`test/registered/unit/manual/test_ac12_helpers.py`:

- `_openai_base_url` × 5: appends `/v1`; strips trailing slash;
  idempotent on `/v1`; idempotent on `/v1/`; case-insensitive
  suffix preserves original case.
- MMLU formatting × 2: `_format_mmlu_example` with answer ends
  `"Answer: X\n\n"`; without answer ends bare `"Answer:"`.
- 5-shot prompt × 2: contains exactly 5 in-context `"Answer: X"`
  endings from dev set; requires ≥ 5 dev rows (`ValueError` otherwise).
- `_parse_mmlu_letter` × 1: finds first A-D char; None for letterless
  responses; first wins on `"AB"`.

### Fix 5 — Registered fault-injection regressions (4 new)

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`
new class `TestAC12FaultInjection`:

- `test_zero_sig_gate_default_off_keeps_signatures` — env unset →
  signatures are populated normally (sanity check that the change
  didn't introduce an unconditional zero).
- `test_zero_sig_gate_on_zeroes_just_written_row_keeps_written_true`
  — env on → `signatures[layer_id, cache_loc] == 0` AND
  `written == True`.
- `test_corrupt_mask_gate_random_selection_shape_dtype_range` —
  the algorithm preserves shape / dtype / device, all values in
  `[0, head_dim)`, and differs from the calibrated baseline.
- `test_corrupt_mask_gate_deterministic_per_seed` — same seed →
  same corruption (audit reproducibility); different seeds differ.

### Fix 6 — Docstring fix

`test_double_sparsity_v32.py` usage section now directs operators at
the pytest file-path form. The Round 25
`python -m unittest test.manual.test_double_sparsity_v32` form
collides with Python's stdlib `test` package and fails to import
(Codex Round 25 review queued issue).

## Files Changed

- `test/manual/test_double_sparsity_v32.py`: full MMLU rewrite + hit
  counts in artifacts + docstring fix.
- `python/sglang/srt/models/deepseek_v2.py`: `os` import + corrupt-mask
  env gate after `slice_per_rank`.
- `python/sglang/srt/layers/attention/dsa_backend.py`: zero-sig env
  gate cached in `__init__`; honored in `_write_token_labels`.
- `test/registered/unit/manual/test_ac12_helpers.py`: +10 helper
  regressions (URL norm + MMLU 5-shot + letter parse).
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +4 `TestAC12FaultInjection` regressions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
254 passed, 0 failed (was 240 before this round)

env -u DS_BASE_URL -u DSA_BASE_URL -u DS_CORRUPT_MASK_URL -u DS_ZERO_SIG_URL \
    python -m pytest test/manual/test_double_sparsity_v32.py -v
6 skipped — clean skip when env vars unset
```

Targeted runs:
```
pytest -v test/registered/unit/manual/test_ac12_helpers.py             # 21 passed (was 11)
pytest -v -k TestAC12FaultInjection                                    # 4 passed
```

Commit: `2ad8b9ee3` — [AC-12] Real MMLU 5-shot + fault-injection gates
+ harness fixes.

## Remaining Items

Code-tier items queued for future rounds:
- `benchmark_compare.py` AC-11 directional gate (3-trial median, DS
  TPS ≥ 95% of DSA, P99 TTFT ≤ 1.10× DSA).
- Shallow AC-8 prefix-match regression coverage.
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality` (harness + fault-injection
both ready; hardware execution pending).

## Push-to-remote Status

Branch is 27 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Both Round 26 fixes were straightforward — the MMLU bug was a
"I reused the wrong eval class" mistake (`simple_eval_mmlu.MMLUEval`
is documented as 0-shot in its own docstring), and the
fault-injection gates were a documented deferral now landed. Neither
generalizes into a useful BitLesson; the existing
`BL-20260527-importlib-dataclass-sys-modules` lesson already covered
the helper-test loader pattern I reused this round.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-25-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-25-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-24-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-24-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-23-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-23-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-26-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
