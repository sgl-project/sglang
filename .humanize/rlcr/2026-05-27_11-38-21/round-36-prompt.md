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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-36-contract.md

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
# Round 35 Code Review

Mainline Progress Verdict: ADVANCED

Round 35 advanced AC-10 by adding useful scaffolding: CPU label-write determinism tests, the `record_radix_fixture_passed` helper, a manual harness shell, and launcher marker coverage. It does not complete AC-10. The current manual fixture is not sufficient evidence for the original plan's M3-B requirement, and the hardware pass / FP8 scale proof / guard flip / radix-on launcher edit remain active mainline work.

## Goal Alignment Summary

```text
ACs: 13/15 addressed (6 met, 7 partial, 2 not met) | Forgotten items: 0 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-10, AC-11, AC-12.

Not met: AC-1b, AC-9.

The tracker still covers the remaining original-plan work in Active Tasks. I updated the mutable tracker section for Round 35 Review: AC-10 remains active, and the manual fixture adequacy issue is now recorded as a blocking side issue for AC-10 / AC-11. The only explicit deferral I reject is treating AC-10 hardware execution and the post-pass launcher/validator flip as "not in this loop"; those are original-plan AC-10 requirements and must be completed before AC-11 can run honestly.

## Mainline Gaps

1. **AC-10's manual fixture does not prove cold/warm label bit-stability or FP8 scale stability.**

Evidence:
- Plan AC-10 requires the M3-B hardware fixture to prove cold-prefix vs warm-prefix labels are bit-stable and to explicitly verify FP8 block scale-factor stability for the same token under cold singleton vs fully-packed block writes (`development/loop4/refined_plan_v1.md:121`, `:303`).
- The new fixture only compares generated text (`test/manual/test_dsv32_radix_cache_fixture.py:204-225`). It never reads or hashes `token_label_table.signatures`, `written`, selected token indices, reused physical slots, or FP8 scale tensors.
- The fixture sends different prompts: `pass_id="cold"` for the first request and `pass_id="warm"` for the second (`test/manual/test_dsv32_radix_cache_fixture.py:182-189`) while asserting the continuations are byte-identical. That can fail because the prompt suffix changed, or pass because the model ignored the suffix; neither outcome proves label stability.
- The artifact does not include the `commit_sha` Claude claimed and the contract required. The payload only records URL, server args, prompts, texts, and `bit_equal` (`test/manual/test_dsv32_radix_cache_fixture.py:205-215`).

Required implementation plan:
1. Replace the continuation-only proxy with fixture-only direct evidence gated behind a clearly named debug knob such as `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`.
2. Run cold and warm requests with identical prompt text for the cacheable prefix path. Put the run id in the request id or artifact metadata, not in a differing prompt suffix.
3. Add a server-side capture path that, for the fixture only, records per-layer hashes of DS label rows for the reused shared-prefix physical slots, the corresponding `written` flags, selected physical slots, and `cached_tokens`/reuse evidence into response `meta_info` or a fixture endpoint. Hash bytes are enough; do not dump full tensors into the response.
4. Update the manual test to assert: radix cache is on, warm request reports cached/reused prefix tokens, cold and warm shared-prefix label hashes are bit-equal, selected physical/logical evidence is stable for the shared prefix, and the artifact records `commit_sha`, server args, prompts, captured hashes, cached-token counts, and verdict.
5. Add the explicit FP8 scale proof in the same manual fixture or a companion H200-only helper: call the production FP8 quantization path for the same deterministic K token as a singleton and as part of a packed batch/block, then compare the produced scale bytes/tensors for that token. If scales differ, document the failure and keep radix disabled per plan.
6. Only after that fixture passes on H200, wire a persistent operator config path before `validate_double_sparsity` runs, call `record_radix_fixture_passed(server_args)`, remove `--disable-radix-cache` from `development/serve_double_sparsity.sh`, and update `test_ds_server_does_disable_radix_cache_until_ac10` to the post-AC-10 expectation.

2. **AC-10 remains unfinished: the guard is not actually flipped and the DS launcher still disables radix cache.**

Evidence:
- `development/serve_double_sparsity.sh:67-70` adds a marker but still passes `--disable-radix-cache`.
- `ServerArgs.check_server_args` calls `validate_double_sparsity(self)` directly (`python/sglang/srt/server_args.py:7193-7198`). The new helper is not wired into any launcher/parser path, so a normal post-AC-10 boot still has no durable way to set `_double_sparsity_radix_fixture_passed` before validation.
- This is still correctly tracked as active, but Claude's summary frames the post-pass work as operator-driven future work. Original AC-10 requires `_double_sparsity_radix_fixture_passed = True` and removal of `--disable-radix-cache`; AC-11 depends on that parity.

Required implementation plan:
1. Do not call `record_radix_fixture_passed` until the fixed M3-B fixture and FP8 scale proof pass.
2. After the pass, add a durable ServerArgs/config mechanism for the verified operator flip, invoke `record_radix_fixture_passed(server_args)` before `validate_double_sparsity`, and include the fixture artifact path/hash in the audit log if available.
3. Remove `--disable-radix-cache` from the DS launcher and update script-contract tests so DS and DSA are both radix-on for AC-11.
4. Run the AC-11 3-trial DSA + DS H200 sweep only after the comparator sees radix parity.

## Blocking Side Issues

None separate from the AC-10 mainline gaps above. The fixture inadequacy is mainline because it blocks AC-10 and therefore AC-11.

## Queued Side Issues

- Preserve the existing queued cleanup items: AC-8 prefix-match helper regression coverage, stale `deepseek_v2.py` slot-authority comments, and stale `token_label_table.py` lifetime docs.
- The benchmark-script Round 33 marker cleanup is resolved by Round 35 and should not stay as an open queued item.

## Validation

I ran the Round 35 local checks that do not require H200:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -k 'AC10 or radix_on_refused' -q
5 passed, 210 deselected

PYTHONPATH=python pytest test/registered/unit/development/test_option_b_scripts.py -q
23 passed

env -u DS_BASE_URL PYTHONPATH=python pytest test/manual/test_dsv32_radix_cache_fixture.py -q
1 skipped

bash -n development/serve_double_sparsity.sh
bash -n development/benchmark.sh
bash -n development/benchmark_baseline.sh
grep -nE 'Round 3[0-9]|Codex Round' development/benchmark.sh development/benchmark_baseline.sh
```

The grep command returned no matches.

## Tracker Update

I updated only the mutable section of `goal-tracker.md`:
- Plan Version now says `Updated: Round 35 Review`.
- Added a Round 35 Review Plan Evolution row reopening AC-10 fixture completeness.
- Updated `task-ac10-radix` notes to say the Round 35 fixture is scaffolding, not sufficient evidence.
- Added a Blocking Side Issues row for the continuation-only manual fixture gap.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-36-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-36-summary.md

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
