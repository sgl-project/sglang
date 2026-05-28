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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-37-contract.md

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
# Round 36 Code Review

Mainline Progress Verdict: STALLED

Round 36 fixed useful scaffolding from Round 35: the continuation smoke now uses identical prompts, the audit helper can log artifact hashes, and a capture module exists. It still does not close AC-10. The new M3-B label-capture fixture can pass with no server-side label evidence, and the FP8 proof does not exercise the production KV/index-cache store path that AC-10 is gating.

## Goal Alignment Summary

```text
ACs: 13/15 addressed (6 met, 7 partial, 2 not met) | Forgotten items: 0 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-10, AC-11, AC-12.

Not met: AC-1b, AC-9.

The tracker covers the original-plan pending work, but it had drifted by marking the Round 35 AC-10 fixture blocker as resolved. I updated only the mutable section: Plan Version now says Round 36 Review, AC-10 fixture completeness is reopened, `task-ac10-radix` notes describe the remaining code-tier gaps, and the blocking side-issue row is no longer marked resolved.

## Mainline Gaps

1. **The M3-B label-capture fixture can PASS with zero direct label evidence.**

Evidence:
- The only production hook appends write records inside the server/model process (`python/sglang/srt/layers/attention/dsa_backend.py:1499-1512`).
- The fixture's default collection path imports `radix_fixture_capture` in the pytest client process (`test/manual/test_dsv32_radix_label_capture_fixture.py:249-254`) and clears that same local client log before the request (`:259-263`). With the normal `DS_BASE_URL=http://...` server setup, this is a different process from the running server, so it reads an empty `_LOG`.
- There is no built-in HTTP endpoint or manager/control path exposing the server process capture log. `rg` finds `radix_fixture_capture` only in `dsa_backend.py` and the capture module; no entrypoint or manager route calls `get_log()` or `record_table_snapshot`.
- The test does not assert evidence exists. It computes `cold_writes` and `warm_writes` (`test_dsv32_radix_label_capture_fixture.py:268-281`), but the verdict is `PASS` when `not mismatches and cached_tokens > 0` (`:334-336`). The assertions only check `cached_tokens > 0` and `mismatches == []` (`:340-352`). Empty logs produce no mismatches.
- Even with a real server log, the comparison is against overlapping write records (`:289-315`). A warm radix-cache hit is exactly the path that skips rewriting the reused prefix slots, so the important warm-prefix state will usually have no write record. `record_table_snapshot` exists (`radix_fixture_capture.py:115-158`) but is never wired into request handling or the fixture.

Required implementation plan:
1. Stop treating client-process imports or operator-provided ad hoc endpoints as the primary evidence path. Add a first-class, env-gated server-side evidence path using the existing per-request summary channel (`forward_batch.ds_per_request_summary` → `BatchTokenIDOutput.per_request_summary` → response `meta_info`).
2. In the DS request path, when `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, compute the physical prompt/cached-prefix slots from `req_to_token[req_pool_idx, :prompt_len_or_cached_len]` after the request has populated the table. Use the warm response's `cached_tokens` to choose the exact comparable prefix length.
3. Snapshot `token_label_table.signatures[:, slots]` and `written[:, slots]` per layer with `record_table_snapshot` or an equivalent helper, and include those hashes in `meta_info["double_sparsity_radix_capture"]` for each request.
4. Update the fixture to assert: `num_cold_snapshots > 0`, `num_warm_snapshots > 0`, `cached_tokens > 0`, the compared slot/hash sets are non-empty, every `written` hash proves reachability, and cold/warm per-layer label hashes match for the cached prefix.
5. Add unit coverage for the false-pass case: mocked empty capture logs plus `cached_tokens > 0` must fail, not pass. Add a second regression where warm has no overlapping write records but has matching snapshots, and that is the only accepted path.

2. **The FP8 scale fixture does not prove the AC-10 production scale property.**

Evidence:
- The fixture calls `sglang_per_token_group_quant_fp8` directly (`test/manual/test_dsv32_fp8_scale_stability.py:100-134`).
- That helper allocates scales shaped per input row/group (`python/sglang/srt/layers/quantization/fp8_kernel.py:490-495`) and quantizes each row independently. Comparing `K0` as a 1-row input vs row 0 of a 64-row input is therefore not a proof about page/block fill level; neighbor rows are not part of `K0`'s scale computation.
- The relevant DSA index-cache store path is `_store_index_k_cache`, which prefers `fused_store_index_k_cache(key, buf, out_cache_loc, page_size)` (`python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1166-1205`) and falls back to `act_quant` plus `set_index_k_scale_buffer` (`:1233-1243`). The stored scale bytes live in the page buffer layout handled by `index_buf_accessor.py:400-511`. The Round 36 fixture exercises none of that.

Required implementation plan:
1. Replace the FP8 proof with a fixture that writes the same deterministic 128-d K row through the actual DS/DSA index-cache quant/store path used at the Option B operating point.
2. Allocate a real `index_k_with_scale_buffer`-shaped page buffer, write `K0` once at a singleton/block-start location and once into a fully populated page/block with deterministic neighbors, then read back the stored scale bytes for K0 using the production accessor.
3. Run the same assertion through the fused CUDA store path when available and the fallback path when the fused kernel is unavailable; skip only when neither production path can run on the target hardware.
4. Record the exact path used, page size, cache locations, scale bytes, and commit SHAs in the artifact.

3. **The operator runbook still points at the smoke fixture and the post-AC-10 launcher flip is still absent.**

Evidence:
- `development/serve_double_sparsity.sh` still passes `--disable-radix-cache` (`:67-69`), and the script contract still asserts that pre-AC-10 state (`test/registered/unit/development/test_option_b_scripts.py:90-97`).
- The trailing launcher comment says the operator flips the gate after `test/manual/test_dsv32_radix_cache_fixture.py` passes (`development/serve_double_sparsity.sh:72-77`). That file is now explicitly only a continuation smoke, not the M3-B evidence.
- `record_radix_fixture_passed(server_args, artifact_path=...)` is still not wired before `validate_double_sparsity`; this is acceptable only until the real M3-B evidence passes, but it remains incomplete original-plan work.

Required implementation plan:
1. Do not remove `--disable-radix-cache` or wire the guard flip until the fixed label snapshot fixture and corrected FP8 store-path proof pass on H200.
2. Immediately fix the launcher comments so they name the label-capture fixture plus FP8 proof, not the continuation smoke, as the evidence required before any flip.
3. After the real pass, add the durable pre-validation hook that calls `record_radix_fixture_passed(server_args, artifact_path=...)`, remove `--disable-radix-cache`, and update `test_ds_server_does_disable_radix_cache_until_ac10` to the post-AC-10 expectation.

## Blocking Side Issues

None separate from the AC-10 mainline gaps. The direct-evidence false-pass path blocks AC-10 and therefore blocks honest AC-11 execution.

## Queued Side Issues

- Preserve the existing queued cleanup items: AC-8 prefix-match helper regression coverage, stale `deepseek_v2.py` slot-authority comments, and stale `token_label_table.py` lifetime docs.
- Do not let those cleanup items displace AC-10 fixture correctness.

## Validation

Local checks run:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -k 'TestRadixFixtureCapture or record_radix_fixture or radix_on_refused' -q
13 passed, 214 deselected

env -u DS_BASE_URL -u SGLANG_DS_RADIX_FIXTURE_CAPTURE -u SGLANG_DS_FP8_SCALE_PROOF \
  PYTHONPATH=python pytest \
  test/manual/test_dsv32_radix_cache_fixture.py \
  test/manual/test_dsv32_radix_label_capture_fixture.py \
  test/manual/test_dsv32_fp8_scale_stability.py -q
3 skipped

PYTHONPATH=python pytest test/registered/unit/development/test_option_b_scripts.py -q
23 passed

bash -n development/serve_double_sparsity.sh
```

I did not run H200 hardware gates. The local tests passing does not clear the false-pass hole above because the manual fixture has no regression for empty server evidence or no overlapping writes.

## Tracker Update

I updated only the mutable section of `goal-tracker.md`:
- Plan Version: `Updated: Round 36 Review`.
- Added a Round 36 Review Plan Evolution row reopening AC-10 fixture completeness.
- Rewrote `task-ac10-radix` active-task notes to reflect that Round 36 is scaffolding, not code-tier closure.
- Reopened the AC-10 blocking side issue instead of leaving it marked resolved.

No immutable Ultimate Goal or Acceptance Criteria text was modified.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-37-contract.md stable for this round
- Do not let queued issues take over the round
- If Codex reported several findings, classify them into:
  - mainline gaps
  - blocking side issues
  - queued side issues
- Only mainline gaps and blocking side issues should drive the next code changes

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-37-summary.md

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
