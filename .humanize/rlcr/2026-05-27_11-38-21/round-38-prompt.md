Your work is not finished. Read and execute the below with ultrathink.

## Drift Recovery Mode

Codex judged the recent implementation rounds as failing to advance the mainline.

- Consecutive stalled/regressed rounds: 2
- Last mainline verdict: stalled

This round is a **drift recovery round**. Do not continue with normal issue-clearing behavior.

## Original Implementation Plan

**IMPORTANT**: Re-anchor on the original plan first:
@development/loop4/refined_plan_v1.md

## Required Recovery Re-anchor

Before changing code:
- Re-read @development/loop4/refined_plan_v1.md
- Re-read @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md
- Re-read the recent round summaries and review results that led here
- Rewrite the round contract at @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-38-contract.md

Your recovery contract must contain:
- Exactly one recovered **mainline objective**
- The 1-2 target ACs that prove mainline progress this round
- The root cause of recent drift or stagnation
- Which issues are truly **blocking** the recovered mainline objective
- Which issues remain **queued** and explicitly out of scope
- Concrete success criteria that would change the verdict back to `ADVANCED`

Do not start implementation until the recovery contract exists.

## Task Lane Rules

Use the Task system (TaskCreate, TaskUpdate, TaskList) with one required tag per task:
- `[mainline]` for plan-derived work that directly advances the recovered objective
- `[blocking]` for issues that prevent the recovered mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- This round must prove mainline movement, not just reduce noise
- `[blocking]` work is allowed only when it directly unblocks the recovered mainline objective
- `[queued]` work must stay documented but must NOT replace the recovered objective
- If a new issue does not block the recovered objective, tag it `[queued]` and keep moving on mainline work

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 37 Code Review

Mainline Progress Verdict: STALLED

Round 37 made useful local progress over Round 36, but AC-10 is still not code-tier complete. The new response-routed label capture cannot pass through the actual `per_request_summary` transport shape, the snapshot is taken at the wrong point and over the wrong slot range, and the FP8 fixture still does not cover the full production store contract from the Round 37 plan.

## Goal Alignment Summary

```text
ACs: 13/15 addressed (6 met, 7 partial, 2 not met) | Forgotten items: 0 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-10, AC-11, AC-12.

Not met: AC-1b, AC-9.

The tracker covers the remaining original-plan work, but it had drifted by marking the Round 37 AC-10 code tier as complete. I updated only the mutable section: Plan Version now says Round 37 Review, `task-ac10-radix` is reopened with the concrete transport/timing/range/fallback gaps, and the AC-10 blocking side issue is no longer marked resolved.

## Mainline Gaps

1. **The label-capture fixture rejects the shape that `/generate` will actually return.**

Evidence:
- `_publish_ds_request_summary` stores `summary["double_sparsity_radix_capture"] = build_request_capture(...)`, a per-batch list (`python/sglang/srt/models/deepseek_v2.py:2097`).
- The existing transport deliberately unwraps per-request summary lists: `_maybe_collect_per_request_summary` stores `entry = v[i]` on the request (`python/sglang/srt/managers/scheduler_components/batch_result_processor.py:256-262`), and the tokenizer emits `meta_info[k] = entry` (`python/sglang/srt/managers/tokenizer_manager.py:1799-1806`). Existing tests document this as a dict-per-request contract (`test/registered/unit/layers/attention/test_double_sparsity_unit.py:4381-4406`, `:5417-5424`).
- The manual fixture returns the raw `meta_info["double_sparsity_radix_capture"]` (`test/manual/test_dsv32_radix_label_capture_fixture.py:174-180`), but the verdict helper treats any non-list as missing (`test/manual/_m3b_label_capture_verdict.py:32-37`, `:59-69`). A real dict-shaped server response therefore fails as "capture missing or empty", not as evidence.

Required fix:
1. Add a registered transport regression that simulates `per_request_summary={"double_sparsity_radix_capture": [record]}` flowing through `_maybe_collect_per_request_summary` and tokenizer-style `meta_info` unpacking, then feeds the resulting dict-shaped `meta_info` value to the same verdict path the manual fixture uses.
2. Make the fixture/helper consume the real response shape. The simplest contract is: a dict is one capture record; a list is accepted only for backward-compatible direct helper tests and normalized to its first record for the single-request fixture.
3. Keep wrong shapes failing loudly, but do not classify the actual per-request dict as missing evidence.

2. **The snapshot is taken before current labels are written and hashes full `seq_lens`, not the reused cached prefix.**

Evidence:
- `build_request_capture` slices `req_to_token[row, :seq_lens[b]]` and stores that as `prompt_len` (`python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py:240-267`).
- In prefill, `seq_lens` is `len(r.fill_ids)` (`python/sglang/srt/managers/schedule_batch.py:1803-1808`); in decode it is incremented for every generated token (`python/sglang/srt/managers/schedule_batch.py:2459-2465`). This is the full current sequence, not the cached-prefix length.
- The capture is called from `_select_topk_indices` (`python/sglang/srt/models/deepseek_v2.py:2072-2105`). That function invalidates the current `out_cache_loc` before selection (`python/sglang/srt/models/deepseek_v2.py:2183-2194`).
- The DSA backend writes labels later, when attention core calls the backend (`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:278-290`, `:421-567`; label writes at `python/sglang/srt/layers/attention/dsa_backend.py:1577-1583`, `:1776-1782`, `:2305-2306`). So a decode capture can include just-invalidated current slots before they are restored.
- Because the final response carries the latest per-request summary, decode-time captures can overwrite the initial prefill evidence path. Those decode captures include generated-token slots that were freshly allocated for the warm request, not radix-reused prompt slots.

Required fix:
1. Remove radix label capture from `_publish_ds_request_summary` / `_select_topk_indices`.
2. Publish the capture after label writes are complete for the forward, not during selection. A concrete path is to emit it from the DSA backend after `_write_token_labels` on the final DS layer for an extend/prefill forward, writing to `forward_batch.ds_per_request_summary["double_sparsity_radix_capture"]` before `ModelRunner` copies the summary.
3. Capture only comparable prefix rows. The capture record must expose enough per-position hashed evidence for the fixture to compare the first `warm_cached_tokens` positions from cold and warm, instead of comparing aggregate hashes over full `seq_lens`. Per-token slot hashes plus per-layer per-token label/written hashes are acceptable for this one-request hardware fixture; full tensors are not needed.
4. Add regressions for both failure modes: a dict-shaped transported capture passes when hashes match, and a capture containing extra generated-token slots does not force a `slots_sha` mismatch when the cached prefix itself is stable.

3. **The FP8 fixture still skips the fallback production store path.**

Evidence:
- The Round 37 contract required trying fused and fallback production store paths, skipping only when neither can run.
- Production `_store_index_k_cache` uses `fused_store_index_k_cache` when available (`python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1187-1205`) and otherwise falls back to `act_quant(...)` plus `set_index_k_scale_buffer(...)` (`python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1232-1244`).
- The fixture exits with `skipTest` when `can_use_dsa_fused_store(...)` is false and explicitly says the fallback is not implemented (`test/manual/test_dsv32_fp8_scale_stability.py:105-112`).
- It also reads raw offsets directly (`test/manual/test_dsv32_fp8_scale_stability.py:153-161`) instead of sharing the production accessor math in `index_buf_accessor.py` (`python/sglang/srt/layers/attention/dsa/index_buf_accessor.py:398-468`).

Required fix:
1. Implement a fallback branch that uses the same fallback ingredients as production: `act_quant(key, block_size, scale_fmt)` followed by `set_index_k_scale_buffer`/`SetKAndS` against a real `index_k_with_scale_buffer`-shaped buffer.
2. Factor a tiny readback helper that shares the production offset/accessor math, and use it for both fused and fallback buffers.
3. Record `path_used` as `fused_store_index_k_cache` or `fallback_act_quant_set_index_k_scale_buffer`; skip only if CUDA/FP8 support is absent and neither production path can execute.

4. **The post-AC-10 guard flip and launcher parity remain unfinished original-plan work.**

This must not be done before the corrected fixtures pass, but it is still part of AC-10, not a completed item. After the fixed FP8 and label-capture fixtures pass on H200, wire `record_radix_fixture_passed(server_args, artifact_path=...)` before `validate_double_sparsity`, remove `--disable-radix-cache` from `development/serve_double_sparsity.sh`, update the launcher contract test, and only then run AC-11's radix-parity H200 sweep.

## Blocking Side Issues

None separate from the AC-10 mainline gaps. The response-shape and snapshot-placement defects directly block AC-10 and therefore block honest AC-11 execution.

## Queued Side Issues

- Preserve the existing queued cleanup items: AC-8 prefix-match helper regression coverage, stale `deepseek_v2.py` slot-authority comments, and stale `token_label_table.py` lifetime docs.
- Do not let those cleanup items displace the AC-10 fixture correctness work.

## Validation

Local checks run:

```text
PYTHONPATH=python pytest test/registered/unit/manual/test_m3b_label_capture_verdict.py -q
11 passed

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -k 'TestBuildRequestCapture or TestRadixFixtureCapture' -q
13 passed, 218 deselected

env -u DS_BASE_URL -u SGLANG_DS_RADIX_FIXTURE_CAPTURE -u SGLANG_DS_FP8_SCALE_PROOF \
  PYTHONPATH=python pytest \
  test/manual/test_dsv32_radix_cache_fixture.py \
  test/manual/test_dsv32_radix_label_capture_fixture.py \
  test/manual/test_dsv32_fp8_scale_stability.py -q
3 skipped

bash -n development/serve_double_sparsity.sh
```

I did not run H200 hardware gates. The local tests do not cover the actual dict-shaped `meta_info` response or the post-write cached-prefix capture timing.

## Tracker Update

I updated only the mutable section of `goal-tracker.md`:
- Plan Version: `Updated: Round 37 Review`.
- Added a Round 37 Review Plan Evolution row reopening AC-10 code-tier completeness.
- Rewrote `task-ac10-radix` active-task notes to reflect the remaining transport, snapshot, and FP8 fallback gaps.
- Reopened the AC-10 blocking side issue instead of leaving it marked resolved.

No immutable Ultimate Goal or Acceptance Criteria text was modified.

NOT COMPLETE
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference

Before starting work, **read and update** @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md as needed:
- Keep the immutable section unchanged
- Record the drift/stagnation cause in the mutable section if it changed planning
- Keep blocking vs queued issue classification accurate
- Ensure the tracker and contract now describe the same recovered mainline objective

## Recovery Guardrails

- Do not spend this round mostly on queued cleanup
- Do not broaden scope to compensate for previous stalls
- If the original approach was flawed, log the plan evolution explicitly instead of silently changing direction
- If you cannot produce a credible recovered mainline objective, say so in the summary with concrete blockers

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/sgl-workspace/sglang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Reference: @/sgl-workspace/sglang/.humanize/bitlesson.md

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-38-summary.md

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
