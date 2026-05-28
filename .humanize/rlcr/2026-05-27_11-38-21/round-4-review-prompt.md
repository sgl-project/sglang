# FULL GOAL ALIGNMENT CHECK - Round 4

This is a **mandatory checkpoint** (at configurable intervals). You must conduct a comprehensive goal alignment audit.

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@development/loop4/refined_plan_v1.md

You MUST read this plan file first to understand the full scope of work before conducting your review.

---
## Claude's Work Summary
<!-- CLAUDE's WORK SUMMARY START -->
# Round 4 Summary

## Work Completed

### AC-1 call-site verification — `TestAC1CallSites`

Added four tests that exercise the three production `_write_token_labels` call sites:

**`test_forward_extend_writes_token_labels`**
Calls `forward_extend` with `use_mha=True` and `save_kv_cache=True`. The MHA path
was chosen because it routes through the KV-write block (lines 1496-1510) and then
hits `_forward_standard_mha` (patched), avoiding all complex transform/paging logic
while still firing the production hook. Asserts `table.written[0, 5]` and
`table.written[0, 10]` are True, and signatures are non-zero.

**`test_forward_decode_writes_token_labels`**
Calls `forward_decode` with `dsa_decode_impl="flashmla_kv"`, `SGLANG_DSA_FUSE_TOPK=1`
(which routes `page_table_1 = topk_indices`, skipping `transform_index_page_table_decode`
and its metadata requirements), and `save_kv_cache=True`. `_forward_flashmla_kv` is
patched to return a dummy output. Asserts table populated at slots 7 and 15.

**`test_trtllm_hook_receives_pre_quantized_k`**
Calls `_forward_trtllm` directly (same production call site) with
`kv_cache_dtype=torch.float8_e4m3fn`. Patches `mla_quantize_and_rope_for_fp8` to
return a `float8_e4m3fn` k tensor (simulating the FP8 overwrite), and patches
`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` to avoid hardware.
Replaces `backend._write_token_labels` with a spy that records the `k` argument.
Asserts: exactly one call; dtype is `float32` (not `float8_e4m3fn`); values equal
the original `torch.ones(T, 1, kv_lora_rank)` latent k — proving `k_for_labels` is
saved before `mla_quantize_and_rope_for_fp8` runs.

**`test_no_labels_when_save_kv_cache_false`**
Same `forward_extend` setup but `save_kv_cache=False`. Asserts table.written remains
all-False — the KV-write block is guarded by `if save_kv_cache:` and the hook must
not fire outside that guard.

## Files Changed

- `test/registered/unit/layers/attention/test_double_sparsity_unit.py` — add `TestAC1CallSites` (4 tests)

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
160 passed, 0 failed
```

Commit: `ef16fa441`

## Remaining Items

- `task-ac1-hwtest`: hardware test on H200 with real `forward_extend` against V3.2 weights — pending hardware access; cannot be automated in unit suite.
- Next coding work (by dependency order): AC-2 lifetime/stale-slot, AC-3 M2 range mask, AC-7 short-seq bypass, AC-4 calibration.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Call-site test structure pattern (object.__new__ + instance-level patching + spy) is standard Python mocking; no new project-specific lesson needed.
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
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
- @.humanize/rlcr/2026-05-27_11-38-21/round-3-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-3-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-2-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-2-review-result.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-1-summary.md
- @.humanize/rlcr/2026-05-27_11-38-21/round-1-review-result.md


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

To implement the original plan at @development/loop4/refined_plan_v1.md, we have completed **5 iterations** (Round 0 to Round 4).

The project's `.humanize/rlcr/2026-05-27_11-38-21/` directory contains the history of each round's iteration:
- Round input prompts: `round-N-prompt.md`
- Round output summaries: `round-N-summary.md`
- Round review prompts: `round-N-review-prompt.md`
- Round review results: `round-N-review-result.md`

**How to Access Historical Files**: Read the historical review results and summaries using file paths like:
- `@.humanize/rlcr/2026-05-27_11-38-21/round-3-review-result.md` (previous round)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-2-review-result.md` (2 rounds ago)
- `@.humanize/rlcr/2026-05-27_11-38-21/round-3-summary.md` (previous summary)

**Your Task**: Review the historical review results, especially the **recent rounds** of development progress and review outcomes, to determine if the development has stalled.

**Signs of Stagnation** (circuit breaker triggers):
- Same issues appearing repeatedly across multiple rounds
- No meaningful progress on Acceptance Criteria over several rounds
- Claude making the same mistakes repeatedly
- Circular discussions without resolution
- No new code changes despite continued iterations
- Codex giving similar feedback repeatedly without Claude addressing it

**If development is stagnating**, write **STOP** (as a single word on its own line) as the last line of your review output @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-4-review-result.md instead of COMPLETE.

## Part 6: Output Requirements

- If issues found OR any AC is NOT MET (including deferred ACs), write your findings to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-4-review-result.md
- Include specific action items for Claude to address, classified into:
  - Mainline Gaps
  - Blocking Side Issues
  - Queued Side Issues
- **If development is stagnating** (see Part 4), write "STOP" as the last line
- **CRITICAL**: Only write "COMPLETE" as the last line if ALL ACs from the original plan are FULLY MET with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any AC is deferred
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals allowed
