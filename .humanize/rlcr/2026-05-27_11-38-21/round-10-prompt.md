Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@development/loop4/refined_plan_v1.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

## Round Re-anchor (REQUIRED FIRST STEP)

Before writing code:
- Re-read @development/loop4/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md
- Re-read the most recent round summaries/reviews that led to this round
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-10-contract.md

Your round contract must contain:
- Exactly one **mainline objective**
- The 1-2 target ACs for this round
- Which issues are truly **blocking** that mainline objective
- Which issues are **queued** and explicitly out of scope
- Concrete success criteria for this round

Do not start implementation until the round contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances this round's objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` work is the round's primary success condition
- `[blocking]` work is allowed only when it truly blocks the mainline objective
- `[queued]` work must be documented but must NOT replace the round objective
- If a new bug does not block the current objective, tag it `[queued]` and keep moving on mainline work

Before executing each task in this round:
1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 9 Goal Alignment Review

Mainline Progress Verdict: ADVANCED

Round 9 closes the AC-7 gap from Round 8. The new test is not just a mock-path assertion: it allocates a real `TokenLabelTable`, binds a real `DoubleSparsitySelector` with a synthetic `ChannelMask`, writes labels through `NativeSparseAttnBackend._write_token_labels` via `_set_mla_kv_buffer`, then decodes through `_select_topk_indices` and the real logical-to-physical adapter path. I reran the claimed suite:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
179 passed, 24 warnings in 11.51s
```

## Part 1: Goal Tracker Audit

| AC | Status | Evidence if MET | Blocker if NOT MET | Justification if DEFERRED |
|----|--------|-----------------|--------------------|---------------------------|
| AC-0 | MET | Tracker Completed and Verified: AC-0 slot authority verified Round 2; imports, token-level shapes, logical-domain adapter, validator, and bind sizing covered by unit tests. | - | - |
| AC-1 | PARTIAL | `task-m1-hook` call-site tests verified Round 4. | `task-ac1-hwtest` and AC-8 selector-read smoke remain pending; no real H200 `forward_extend` population evidence. | - |
| AC-1b | NOT MET | - | Chunked-prefill probe has not run. | - |
| AC-2 | MET | Tracker Completed and Verified Round 7; live `_select_topk_indices` invalidation test protects stale-slot behavior. | - | - |
| AC-3 | MET | Tracker Completed and Verified Round 6; logical-domain `req_to_token` isolation and adapter handoff are covered. | - | - |
| AC-4 | NOT MET | - | `calibrate.py` still needs Method 1 Q-noPE/K-noPE implementation and the H200 mask-generation run. The design doc requires the Pile-val 256x512 Method 1 recipe (`development/past_implementations/study/07-mvp-proposed-architecture.md` section 10). | - |
| AC-5 | NOT MET | - | `test/registered/integration/test_double_sparsity_tp_multiprocess.py` is still absent; no TP=2 multiprocess all-reduce proof. | - |
| AC-6 | NOT MET | - | CUDA graph capture remains pending; queued issue notes `capture_decode_step` still lacks `req_to_token`, so it would use the wrong selector domain before AC-6 is started. | - |
| AC-7 | MET | `test_first_decode_after_short_prefill_selects_prefill_slots` covers bypass, MHA label write, and first decode selecting prefill slots (`test_double_sparsity_unit.py:5754-5967`). The production bypass reads `ForwardContext` (`deepseek_v2.py:2071-2089`) and the MHA_ONE_SHOT write hook is present (`forward_mha.py:477-486`). | - | - |
| AC-8 | NOT MET | - | No 8xH200 `bench_serving`, no selected-token activity evidence, and no lightweight quality smoke. The design doc's quality-smoke gate remains unimplemented (`07-mvp-proposed-architecture.md` section 9.4). | - |
| AC-9 | NOT MET | - | No DSA baseline result JSON. Stretch remains active, not deferred. | - |
| AC-10 | NOT MET | - | Radix cache hardware fixture and FP8 cold/warm scale stability proof are pending. Stretch remains active, not deferred. | - |
| AC-11 | NOT MET | - | Comparator row has not run. Stretch remains active, not deferred. | - |
| AC-12 | NOT MET | - | Hard NIAH/MMLU quality gate has not run. | - |
| AC-13 | MET | Tracker Completed and Verified Round 1; the migrated unit suite remains green and now has 179 passing tests. | - | - |

### Forgotten Items Detection

No original plan tasks are missing from Active, Completed, or Deferred. The tracker had one bookkeeping drift: `task-ac7-bypass` was present in both Active and Completed/Verified after Round 9. I corrected the mutable tracker by removing it from Active and adding a Round 9 Review evolution row.

No summary-claimed completion remains unverified for Round 9: I verified AC-7 locally with the unit suite and code inspection.

### Deferred Items Audit

There are no explicitly deferred tasks in the tracker. No deferral currently contradicts the Ultimate Goal.

### Goal Completion Summary

```text
Acceptance Criteria: 5/15 met (0 deferred)
Active Tasks: 13 remaining
Estimated remaining rounds: 6-8 if hardware access is smooth; more if AC-4/AC-6/AC-8/AC-12 expose real-model failures
Critical blockers: no new blocking side issue; remaining critical gates are hardware/calibration/quality dependencies
```

## Part 2: Mainline Drift Audit

The current round's objective was clear and singular: finish the missing AC-7 first-decode-after-short-prefill proof. Claude advanced a mainline acceptance criterion rather than chasing a side issue. Recent rounds show correction rather than stagnation: Round 7's wrong production state was fixed in Round 8, and Round 8's missing decode-after-prefill proof was fixed in Round 9.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 0
Queued Side Issues: 4
```

The queued side issues remain the existing tracker items: graph capture must receive `req_to_token` before AC-6, observability must stop reporting token selections as pages before AC-8, and two stale documentation issues should be cleaned when those modules are next touched.

## Part 3: Implementation Review

No new high-signal Round 9 implementation bug found.

Claude's claims match reality:

- Bypass proof: the test enters `ForwardContext(attn_backend=backend)` with `backend.use_mha=True` and asserts `_select_topk_indices` returns `None` (`test_double_sparsity_unit.py:5884-5901`). The production code returns early from the DS branch on active `ForwardContext.use_mha` (`deepseek_v2.py:2071-2089`).
- Label-write proof: the test calls `_set_mla_kv_buffer` with prefill slots and then checks `table.written` and non-zero signatures (`test_double_sparsity_unit.py:5913-5934`). The production hook calls `_write_token_labels` after the MHA KV write when `use_double_sparsity` is true (`forward_mha.py:477-486`).
- First decode proof: the test flips `backend.use_mha=False`, passes real `q_nope`, and asserts the result contains a physical prefill slot (`test_double_sparsity_unit.py:5941-5967`). This exercises real selector scoring and the logical-to-physical adapter instead of a mocked `retrieve_topk`.

Residual risk: this remains a CPU fixture with a synthetic projection and does not replace the real V3.2 FP8/H200 gates. That is acceptable for AC-7's local proof but does not close AC-1, AC-8, or AC-12.

## Part 4: Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:

- Changed Plan Version update marker to Round 9 Review.
- Added a Round 9 Review row documenting AC-7 verification.
- Removed `task-ac7-bypass` from Active Tasks because it is already in Completed and Verified.

No requested tracker change was rejected.

## Part 5: Progress Stagnation Check

Development is not stagnating. The last four rounds closed AC-2 live wiring and AC-7 production correctness in sequence, with review findings addressed rather than repeated unchanged. The unit suite count moved 170 -> 176 -> 178 -> 179, and the current remaining work has shifted from local AC-7 proof to the larger calibration, TP, graph, hardware, and quality gates.

## Action Items For Claude

### Mainline Gaps

1. Implement AC-4 calibration next: replace K-only statistics with same-forward-pass Method 1 `mean(abs(Q_nope * K_nope))`, keep the 128-d noPE channel axis from the refined plan, add fixture tests for Q/K pairing and invalid 512-d indices, then run the H200 mask generation and `load_channel_mask` validation.
2. Add AC-5 TP=2 multiprocess coverage with real process-group all-reduce in logical-position space and a no-op all-reduce negative.
3. Fix AC-6 graph capture before hardware replay: thread `req_to_token` through `capture_decode_step`, preallocate DS output/scratch buffers, add eager-vs-replay and allocation-negative tests, then run the H200 conc=64 100-step replay.
4. Run the remaining hardware/analyze gates in dependency order: AC-1 H200 population, AC-1b chunked-prefill probe, AC-8 server benchmark and quality smoke, AC-12 full NIAH/MMLU quality.
5. Attempt stretch AC-9 through AC-11 only after hard gates are green.

### Blocking Side Issues

None.

### Queued Side Issues

1. Before AC-6, update `capture_decode_step` to pass `req_to_token`; otherwise graph capture validates the wrong selector domain.
2. Before AC-8, fix DS observability page-named fields and page-count sparsity math.
3. Clean stale bind/runtime sizing comments when touching `deepseek_v2.py`.
4. Clean stale token-label lifetime documentation when touching `token_label_table.py`.

NOT COMPLETE
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Which side issues are blocking vs queued
- Any Plan Evolution that has occurred
- The latest side-issue state that needs attention

**IMPORTANT**: Keep the mutable section of `goal-tracker.md` up to date during the round.
Do NOT change the immutable section after Round 0.
If you cannot safely reconcile the tracker yourself, include an optional "Goal Tracker Update Request" section in your summary (see below).

## Mainline Guardrails

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-10-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

### Post-Alignment Check Action Items

This round follows a Full Goal Alignment Check. Pay special attention to:
- **Forgotten Items**: Codex may have identified tasks that were being ignored. Address them.
- **AC Status**: If any Acceptance Criteria were marked NOT MET, prioritize work toward those.
- **Deferred Items**: If any deferrals were flagged as unjustified, un-defer them now.
- **Queued Issues**: Keep non-blocking follow-up work queued unless it now clearly blocks mainline progress.

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-10-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**Optional fallback**: if you could not safely update the mutable section of `goal-tracker.md` directly, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Blocking Side Issues: bug Y blocks AC-2"]
- [E.g., "Add to Queued Side Issues: cleanup Z is non-blocking"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and reconcile the Goal Tracker if justified.
