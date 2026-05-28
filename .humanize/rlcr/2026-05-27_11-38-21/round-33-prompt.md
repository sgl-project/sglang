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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-33-contract.md

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
# Round 32 Code Review

Mainline Progress Verdict: ADVANCED

Round 32 advanced AC-11: the two Round-31 reproducers now refuse, radix-cache mismatch is no longer allowed, and dynamic `/server_info` telemetry no longer self-refuses sequential trials. The targeted tests pass:

```text
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
48 passed, 7 subtests passed

PYTHONPATH=python pytest test/registered/unit/development/test_option_b_scripts.py -q
16 passed
```

AC-11 is still not acceptance-ready. Claude explicitly deferred the bench_serving-side warmup/window implementation, and the new server-args whitelist can publish a PASS when DSA and DS differ on launch flags outside the small whitelist.

## Goal Alignment Summary

```text
ACs: 13/15 addressed | Forgotten items: 0 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-11, AC-12.

Not met: AC-1b, AC-9, AC-10.

The tracker still covers the remaining original-plan tasks. The unjustified deferral is the AC-11 producer-side timing contract: `bench_serving` and the benchmark scripts still do not run a 120s warmup or force a 600s measured window before writing the JSONL.

## Mainline Gaps

1. AC-11 timing enforcement is still deferred to future work.

Evidence:
- The plan requires a "minimum 600s measurement window after a 120s warmup" and says the timing evidence is required because noise can swing the 5% gate (`development/loop4/refined_plan_v1.md:123-125`).
- `task-ac11-compare` is explicitly the task that must run with fixed seed, 600s window, 120s warmup, 3 trials, and median aggregation (`development/loop4/refined_plan_v1.md:304`).
- `development/benchmark.sh:63-77` and `development/benchmark_baseline.sh:63-77` still invoke `python3 -m sglang.bench_serving` with no `--warmup-seconds` or `--measurement-window-seconds` flag. The `WARMUP_SECONDS` / `MEASUREMENT_WINDOW_S` variables are only written to sidecar metadata (`benchmark.sh:82-94`, `benchmark_baseline.sh:80-92`).
- `python/sglang/bench_serving.py` only has count-based `--warmup-requests` (`python/sglang/bench_serving.py:2284-2289`). The warmup is a fixed request count (`python/sglang/bench_serving.py:1250-1317`), and the measured run consumes the finite request generator once (`python/sglang/bench_serving.py:1348-1415`). `duration` is only an observed result (`python/sglang/bench_serving.py:1461`, `python/sglang/bench_serving.py:1598`).
- Round 32's `_validate_jsonl_duration` is useful after the fact, but it does not produce the required warmup/window evidence and cannot make the scripts generate valid AC-11 artifacts.

Required implementation plan:
1. Add `--warmup-seconds` and `--measurement-window-seconds` to `python/sglang/bench_serving.py`; keep `--warmup-requests` as the legacy path when the new seconds flag is unset.
2. Refactor the request execution block into one reusable measured-epoch helper that takes the prepared `input_requests`, request-rate scheduling, concurrency semaphore, LoRA selection, extra request body, and progress-bar controls, then returns outputs plus elapsed wall time.
3. Implement the seconds warmup by running full workload epochs until elapsed warmup time is at least `warmup_seconds`, discarding warmup outputs. Reset `random.seed(args.seed)` and `np.random.seed(args.seed)` before the first measured epoch so warmup does not perturb the measured request-arrival process.
4. Implement the measured window by running workload epochs and accumulating outputs until measured elapsed time is at least `measurement_window_seconds`. Calculate metrics over the accumulated measured outputs only and write `duration` as the accumulated measured wall time.
5. Add actual workload fields to the JSONL result for generated-shared-prefix runs: `num_prompts`, `input_len` or equivalent ISL, `output_len` or equivalent OSL, and the actual completed request count after any repeated measured epochs.
6. Update `development/benchmark.sh` and `development/benchmark_baseline.sh` to pass the new flags, keep `WARMUP_SECONDS=120` and `MEASUREMENT_WINDOW_S=600` defaults, and fail immediately if the JSONL result duration is below `MEASUREMENT_WINDOW_S`.
7. Extend `benchmark_compare.py --ac11` to require JSONL workload fields when present and verify they agree with sidecar workload fields. Add regressions for the scripts passing the new flags, a short observed duration refusing, and sidecar workload metadata lying about JSONL workload fields refusing.

2. AC-11 server-args comparison now under-compares launch args.

Evidence:
- Plan AC-11 says "Only `--enable-double-sparsity` and `--double-sparsity-config` differ between columns" and requires "full server args" beside each result JSON (`development/loop4/refined_plan_v1.md:124-125`).
- `/server_info` returns `dataclasses.asdict(server_args)` plus scheduler telemetry (`python/sglang/srt/entrypoints/http_server.py:631-651`), and `ServerArgs` contains many launch flags that affect the operating point, including `trust_remote_code`, `dtype`, `quantization`, `mem_fraction_static`, `max_total_tokens`, `attention_backend`, `disable_cuda_graph`, and many more (`python/sglang/srt/server_args.py:340-848`).
- Round 32's `_AC11_STABLE_LAUNCH_ARG_KEYS` contains only 12 keys (`development/benchmark_compare.py:470-483`), and `_normalize_ac11_server_args` silently drops every key outside that set (`development/benchmark_compare.py:577-595`).
- Reproducer I ran: three DSA sidecars with `server_args.disable_cuda_graph=false` and three DS sidecars with `server_args.disable_cuda_graph=true`, with all whitelisted fields equal and valid JSONL duration, exit 0 and publish `AC-11 verdict: PASS`. `disable_cuda_graph` is a real `ServerArgs` launch flag (`python/sglang/srt/server_args.py:700`) and is not one of the two DS-only allowed differences.

Required implementation plan:
1. Replace the small whitelist with a full stable `ServerArgs` projection. Prefer deriving the key set from `dataclasses.fields(ServerArgs)` inside `benchmark_compare.py`; if importing `sglang.srt.server_args` is too heavy for the development script, generate and maintain an explicit full key set from that dataclass.
2. Drop only dynamic `/server_info` additions that are not `ServerArgs` fields: `internal_states`, `kv_events`, scheduler capacity/info fields, throughput/memory/step-time telemetry, and future non-ServerArgs endpoint additions.
3. Keep removing only `enable_double_sparsity`, `double_sparsity_config`, and explicit DS fault-injection env fields from the cross-side comparison. Do not drop launch flags merely because they are not in a hand-written small whitelist.
4. Require the normalized server-args projection to be non-empty and to include the locked Option B fields (`model_path`, `tp_size`, `kv_cache_dtype`, DSA prefill/decode backends, overlap/piecewise flags, `page_size`, radix flag, and CUDA graph flag).
5. Add registered regressions where DSA and DS differ on non-DS launch fields outside the current whitelist: at minimum `disable_cuda_graph`, `trust_remote_code`, `dtype`, and `max_total_tokens`. Each must exit 2.
6. Keep the existing dynamic telemetry drift regression, but make it prove that dynamic non-ServerArgs endpoint fields are ignored while real ServerArgs launch fields still mismatch.

## Blocking Side Issues

No separate blocking side issue beyond the AC-11 mainline gaps above.

## Queued Side Issues

- AC-8 prefix-match helper regressions still manually replicate the slicing expression instead of exercising the actual smoke-harness gate.
- `deepseek_v2.py` still has stale slot-authority comments.
- `token_label_table.py` still has stale lifetime docs.

## Goal Tracker Update

I updated only the mutable section of `goal-tracker.md`:
- Plan Version now says `Updated: Round 32 Review`.
- Added a Round 32 Review evolution row reopening AC-11 code-tier completeness.
- Updated `task-ac11-compare` to require bench_serving timing enforcement and full stable ServerArgs comparison before H200 AC-11 execution is accepted.
- Updated the AC-11 timing blocker to make the remaining producer-side timing work blocking, not a queued future item.
- Added a new AC-11 blocking side issue for the under-broad server-args whitelist false PASS.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-33-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-33-summary.md

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
