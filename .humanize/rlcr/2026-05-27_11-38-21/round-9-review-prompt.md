# FULL GOAL ALIGNMENT CHECK - Round 9

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 9 Summary

## Work Completed

### AC-7 — First-decode-after-short-prefill integration proof

Added `test_first_decode_after_short_prefill_selects_prefill_slots` to `TestAC7MHABypass`
in `test/registered/unit/layers/attention/test_double_sparsity_unit.py`.

The test uses a real `TokenLabelTable`, a real `DoubleSparsitySelector` bound with a
synthetic `ChannelMask` (not mocks), and a `NativeSparseAttnBackend` constructed via
`object.__new__` to provide the real `_write_token_labels` implementation.

**Steps proved end-to-end in order:**

1. **MHA bypass fires**: Under `forward_context(ForwardContext(attn_backend=backend))` with
   `backend.use_mha=True`, `_select_topk_indices` returns `None` without calling
   `retrieve_topk`. Asserts `bypass_result is None`.

2. **Labels written via MHA_ONE_SHOT path**: `_set_mla_kv_buffer` is called for 3 prefill
   tokens (physical slots 1, 2, 3). The DS hook at the end of `_set_mla_kv_buffer` calls
   `backend._write_token_labels(attn_layer, cache_loc, kv_a.unsqueeze(1))` using the real
   projection (`_FakeProj` stub returning `(x.float() @ W,)` in ColumnParallelLinear style).
   Asserts `table.written[0, slot] = True` for all 3 prefill slots and
   `table.signatures[0, slot0].abs().sum() > 0` for slot 1 (the strongest-signal token).

3. **Decode selection uses prefill labels**: Backend flipped to `use_mha=False`. Decode query
   `q_nope[0, :, 0] = 1.0` (strong in K_noPE channel 0) scores positively against prefill
   slot 1's label `[[5, 0], [5, 0]]`. `_select_topk_indices` is called with real
   `req_to_token` mapping logical positions 0–2 to physical slots 1–3. The real
   `retrieve_topk_via_labels` pipeline runs (scoring → `select_topk_sequence_order` →
   `logical_to_physical`). Asserts `decode_result` is not None and contains at least one
   entry from the prefill slot set `{1, 2, 3}`.

**What this test catches:**
- Removing the `_write_token_labels` call from `_set_mla_kv_buffer` → no labels written →
  scoring returns -inf for all slots → `decode_result` is all-(-1) → test fails.
- Removing the `has_forward_context()` bypass in `_select_topk_indices` → prefill step calls
  `retrieve_topk` instead of returning None → test fails on `assertIsNone`.
- Replacing real `retrieve_topk_via_labels` with a mock → physical-slot assertion may fail
  depending on what the mock returns.

## Files Changed

- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — added
  `test_first_decode_after_short_prefill_selects_prefill_slots` to `TestAC7MHABypass`
  (221 new lines, bringing the test class to 7 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
179 passed, 0 failed in 11.58s
```

Commit: `e2552fbe9`

## Remaining Items

- `task-ac1-hwtest`: H200 hardware population test.
- `task-ac4-calibrate` / `task-ac4-hwrun`: Method 1 calibration + mask generation.
- `task-ac5-tp`: TP=2 multiprocess all-reduce test.
- `task-ac6-cuda-graph` / `task-ac6-hwrun`: Graph capture + H200 replay.
- `task-ac1b-probe`, `task-ac8-*`, `task-ac12-quality`: Hardware/analyze gates.
- `task-ac9-baseline`, `task-ac10-radix`, `task-ac11-compare`: Stretch.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The pattern (real table + real selector bound with synthetic ChannelMask + fake backend via object.__new__) already appears in `test_real_logical_domain_scoring_and_adapter_roundtrip` (line 4320). No new generalizable lesson beyond what is already in the codebase.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-8-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-8-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-7-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-7-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-6-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-6-review-result.md


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

To implement the original plan at @development/loop4/refined_plan_v1.md, we have completed **10 iterations** (Round 0 to Round 9).

The project's `.humanize/rlcr/2026-05-27_11-38-21/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-27_11-38-21/round-8-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-7-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-8-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-9-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-9-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
