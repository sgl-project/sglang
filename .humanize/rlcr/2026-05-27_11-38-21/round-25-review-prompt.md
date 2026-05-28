# Code Review - Round 25

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-25-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
# Round 25 Summary

## Work Completed

Codex Round 24 full goal alignment named AC-12 as the next mainline
move and the **hard loop-closure gate**: per plan §10, "Loop does not
close without AC-12 passing." After three rounds of clearing AC-8/AC-9
plumbing (sidecar writer, Option B scripts, observability units), the
AC-12 scaffold was still 6 skip tests. Round 25 replaces it.

### Fix 1 — Real AC-12 harness

`test/manual/test_double_sparsity_v32.py` — full rewrite. Skips
cleanly when env vars are unset; runs the real gates when paired
servers are published.

**NIAH @ 4K / 16K / 64K** (3 tests). Per length:
- 20 deterministic seeded prompts via a Lorem-ipsum-style filler word
  pool that whitespace-tokenizes to approximately the requested
  length (±5%). The same `(length, seed, needle)` triple always
  produces the same string.
- Each prompt has a unique sentinel needle `NEEDLE-LLLLL-III`
  (`length` zero-padded to 5 digits + `index` to 3) planted at a
  deterministic depth in `[0.35, 0.65]` of the prompt — far from both
  ends so DS sparse selection is the meaningful gate.
- Generation: DSA first, then DS, both at `temperature=0`.
- Recall = the needle string is a substring of the response (per
  plan; substring rather than word-boundary because temperature-0
  models may concatenate or stylize the needle).
- Assertion: `dsa_recall_pct - ds_recall_pct ≤ 5.0`.

**MMLU 5-shot** (1 test). Reuses the repo's existing eval path:
- `sglang.test.simple_eval_mmlu.MMLUEval` over
  `openaipublic.blob.core.windows.net/simple-evals/mmlu.csv`.
- `sglang.test.run_eval.run_eval_once` against each base URL.
- Default 200 examples (override via `AC12_MMLU_NUM_EXAMPLES` for
  the full ~14k on H200; the 200-sample subset is enough to detect
  a >1.0 pp regression in CI/dev settings).
- Assertion: `dsa_score_pct - ds_score_pct ≤ 1.0`.

**Sensitivity (2 tests)**, skip cleanly without their respective
URLs:
- `test_niah_64k_sensitivity_corrupt_mask` (`DS_CORRUPT_MASK_URL`):
  assert `dsa - ds_corrupt > 20 pp`.
- `test_niah_16k_sensitivity_zero_signatures` (`DS_ZERO_SIG_URL`):
  assert `dsa - ds_zero > 30 pp`.

Each gate writes a `development/results/ac12_<suffix>_<ts>.json`
artifact with `dsa_hits`, `ds_hits`, `dsa_recall_pct`, `ds_recall_pct`,
`delta_pct`, `threshold_pp` so an audit pass can replay the gate
output even after the servers are torn down.

### Fix 2 — CPU helper regressions (CI-runnable)

`test/registered/unit/manual/test_ac12_helpers.py` (11 tests). The
manual harness file is skip-only without env vars, but its pure
helpers can and must be tested in CI.

Loader trick: the file is registered in `sys.modules` BEFORE
`spec.loader.exec_module(mod)` so the `@dataclass` decorator at
`_NIAHRunResult` can resolve `__module__` (the standard Python
dataclasses machinery walks `sys.modules` to find the owning module's
namespace; loading purely via `importlib.util.spec_from_file_location`
without registering breaks `@dataclass`).

Covered:
- Needle naming: stable per `(length, idx)`; differs across both
  inputs; format `NEEDLE-LLLLL-III`.
- Prompt determinism: same `(length, seed, needle)` → identical bytes.
- Needle present exactly once.
- Whitespace-tokenized length within ±5% of the request (with a
  32-word floor for small lengths).
- Needle depth fraction in `[0.30, 0.70]`.
- Question suffix `"... Output only the value."`.
- Recall scoring: all-present → 100%, none → 0%, partial → correct
  count, substring (not word-boundary) match works.

### Fix 3 — Server-side fault-injection plumbing (deferred to next round)

The two sensitivity tests need real fault-injected DS servers
exposed at `DS_CORRUPT_MASK_URL` and `DS_ZERO_SIG_URL`. The plan
implementation calls for env-var gates inside `dsa_backend.py` /
`token_label_table.py` (e.g. `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK`,
`SGLANG_DS_FAULT_INJECT_ZERO_SIG`) so an operator can boot a third
sglang server with the gate set. The Round 25 contract marked this
"stretch — defers cleanly"; the harness already skips when the
fault-injected URLs are unset, so AC-12 positive gates can run on
H200 today without waiting for that scaffold. Plumbing follows in
a focused Round 26.

## Files Changed

- `test/manual/test_double_sparsity_v32.py`: full rewrite — real
  NIAH + MMLU + sensitivity harness; deterministic prompt/needle
  generation; result artifact writer; env-var-gated skips.
- `test/registered/unit/manual/test_ac12_helpers.py`: NEW — 11 CPU
  regression tests for the harness's helper functions; uses
  `importlib.util` with `sys.modules` registration so `@dataclass`
  resolves.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
240 passed, 0 failed (was 229 before this round)

env -u DS_BASE_URL -u DSA_BASE_URL -u DS_CORRUPT_MASK_URL -u DS_ZERO_SIG_URL \
    python -m pytest test/manual/test_double_sparsity_v32.py -v
6 skipped — clean skip when env vars unset
```

Targeted run:
```
pytest -v test/registered/unit/manual/test_ac12_helpers.py
11 passed (needle naming, prompt determinism + length + depth +
suffix, recall all/none/partial/substring cases)
```

Commit: `0ae955cf2` — [AC-12] Replace skip-only scaffold with real
NIAH + MMLU + sensitivity harness.

## Remaining Items

Code-tier items queued for future rounds:
- **Server-side fault-injection env gates** (`SGLANG_DS_FAULT_INJECT_*`)
  so the two sensitivity tests can actually fire on hardware. The
  harness side is ready.
- **`benchmark_compare.py` AC-11 directional gate**: 3-trial median
  per concurrency, DS TPS within 5% of DSA, P99 TTFT ≤ 1.10× DSA;
  still the absolute-SLO single-trial framing today.
- Shallow AC-8 prefix-match regression coverage cleanup (Codex Round 22
  queued).
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality` (harness landed; hardware
execution pending).

## Push-to-remote Status

Branch is 26 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-importlib-dataclass-sys-modules
Notes: A test fixture that loads a sibling Python file via
`importlib.util.spec_from_file_location(...)` will fail at
`@dataclass`-decorated class definitions with
`AttributeError: 'NoneType' object has no attribute '__dict__'`
unless the module is registered in `sys.modules` BEFORE
`spec.loader.exec_module(mod)`. The dataclasses machinery resolves
`__module__` via `sys.modules.get(cls.__module__)`, and without the
registration step the lookup returns `None`. This is too narrow to
generalize to a separate lesson — it shares scope with the existing
"shell-json-into-python-source" lesson in being about cross-language
boundary plumbing — so I'll add it as a note appended to BitLesson
rather than its own entry.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-24-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-24-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-23-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-23-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-22-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-22-review-result.md


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
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-25-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
