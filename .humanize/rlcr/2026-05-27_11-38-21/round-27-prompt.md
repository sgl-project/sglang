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
- Write the current round contract to @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-27-contract.md

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
# Round 26 Code Review

Mainline Progress Verdict: ADVANCED

Round 26 made real AC-12 progress: the zero-shot MMLU implementation was replaced with 5-shot prompt construction, NIAH artifacts now include hit counts, and the two server-side fault-injection env gates exist. However, AC-12 is still not acceptance-ready. The MMLU gate can still skip before touching the paired servers on a normal checkout, and the MMLU answer parser can score a common prefixed answer incorrectly. The original Loop 4 plan also remains incomplete: all H200 evidence gates, AC-10 radix, and AC-11 comparator work are still pending.

## Implementation Review

Verified Round 26 claims:
- `test/manual/test_double_sparsity_v32.py` now contains 5-shot MMLU prompt helpers and no longer uses `simple_eval_mmlu.MMLUEval`.
- NIAH artifacts include `dsa_hits` / `ds_hits`, with fault variants using `ds_corrupt_hits` and `ds_zero_hits`.
- `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK=1` is wired in `deepseek_v2.py` after `slice_per_rank(...)`.
- `SGLANG_DS_FAULT_INJECT_ZERO_SIG=1` is wired in `dsa_backend.py` after `token_label_write`, while leaving `written=True`.
- The docstring now documents the pytest file-path invocation instead of the failing `python -m unittest test.manual...` form.

Validation I ran:

```text
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
21 passed, 1 warning

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -k TestAC12FaultInjection -q
4 passed, 206 deselected, 21 warnings

env -u DS_BASE_URL -u DSA_BASE_URL -u DS_CORRUPT_MASK_URL -u DS_ZERO_SIG_URL \
  PYTHONPATH=python python -m pytest test/manual/test_double_sparsity_v32.py -q
6 skipped, 1 warning
```

Counter-evidence:

```text
find benchmark/mmlu/data -maxdepth 2 -type f
find: 'benchmark/mmlu/data': No such file or directory

DS_BASE_URL=http://127.0.0.1:1 DSA_BASE_URL=http://127.0.0.1:2 \
  PYTHONPATH=python python -m pytest \
  test/manual/test_double_sparsity_v32.py::TestDoubleSparsityV32Quality::test_mmlu_5shot -q
1 skipped, 1 warning

_parse_mmlu_letter("Answer: B") -> "A"
```

## Goal Alignment Summary

```text
ACs: 13/15 addressed | Forgotten items: 0 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-9, AC-11, AC-12.

Not met: AC-1b, AC-10.

No original-plan task is forgotten in the tracker: the remaining work is still listed under Active Tasks or Queued Side Issues. The unjustified deferral is `benchmark_compare.py` AC-11 being described as a future queued code-tier item even though the original plan still requires the comparator before final completion. Hardware-gated work remains pending, not complete.

Plan evolution is mostly valid: replacing the OpenAI-client MMLU path with `/generate`-based 5-shot prompts is acceptable. The invalid part is converting the MMLU data dependency into `skipTest`; Round 26's own contract required downloading or preparing the data so the hard MMLU gate can fire.

## Mainline Gaps

1. AC-12 is still incomplete.

Evidence:
- `test/manual/test_double_sparsity_v32.py:416-420` skips MMLU when `benchmark/mmlu/data/{dev,test}` is absent.
- This checkout does not contain `benchmark/mmlu/data`.
- With `DS_BASE_URL` and `DSA_BASE_URL` set, the targeted MMLU test still reports `skipped` before contacting either server.
- The refined plan says AC-12 is hard and the loop does not close without MMLU 5-shot delta <= 1 pp.

Required implementation plan:
1. Add `_ensure_mmlu_data_dir(data_dir: str) -> tuple[str, str]` in `test/manual/test_double_sparsity_v32.py`.
2. If `data_dir/dev` and `data_dir/test` already exist, return them.
3. Otherwise download `https://people.eecs.berkeley.edu/~hendrycks/data.tar` into a temp file, safely extract only members under the archive `data/` directory into a temp directory, then atomically move `dev/` and `test/` into `data_dir`.
4. If download or extraction fails while `DS_BASE_URL` and `DSA_BASE_URL` are set, fail the test with a clear message; do not `skipTest`. The only acceptable clean skip is when the class-level server env vars are unset.
5. Add registered helper tests that monkeypatch the download/extract helper and prove the MMLU test does not call `skipTest` solely because the default data dir starts absent.
6. Run the full AC-12 harness on H200 after this fix: NIAH @ 4K/16K/64K, MMLU 5-shot, corrupt-mask sensitivity, and zero-signature sensitivity.

2. Remaining original-plan gates are still active and must not be treated as complete-by-deferral.

Pending original-plan tasks: `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`, and `task-ac11-compare`.

Required execution plan:
1. Generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200 with the AC-4 production command.
2. Run AC-1 forward population and AC-6 real V3.2 CUDA-graph capture/replay.
3. Run AC-1b chunked-prefill probe and record the pass/fail launch decision.
4. Run AC-8 DS server smoke and paired AC-8 quality smoke.
5. Generate AC-9 DSA baseline JSON.
6. Complete AC-10: run the M3-B radix hardware fixture, verify FP8 cold/warm scale stability, set the radix fixture flag only after evidence, and remove DS `--disable-radix-cache`.
7. Implement and run AC-11 comparator semantics: >=3 trials per mode/concurrency, fixed seed, 120s warmup, 600s measurement, median aggregation, DS TPS >= 0.95 * DSA TPS, and DS P99 TTFT <= 1.10 * DSA P99 TTFT.

## Blocking Side Issues

1. MMLU can still silently skip because the data dependency is not self-preparing.

Evidence:
- `test/manual/test_double_sparsity_v32.py:405-420` computes the default data dir and calls `self.skipTest(...)` if `dev/` and `test/` are missing.
- `benchmark/mmlu/bench_sglang.py:55-73` already has the intended download behavior, but Round 26 did not reuse or reimplement it.
- The repository checkout has no `benchmark/mmlu/data`, so the default AC-12 MMLU path is skip-only unless an operator manually primes the directory.

Required fix:
1. Implement the `_ensure_mmlu_data_dir` helper described above.
2. Call it from `test_mmlu_5shot` before subject discovery.
3. Replace the current missing-data skip with a hard failure if servers are configured and data preparation fails.
4. Keep `AC12_MMLU_DATA_DIR` as an override, but validate that the requested directory actually contains usable `dev/` and `test/` CSV trees after preparation.

2. `_parse_mmlu_letter` can mis-score common valid completions.

Evidence:
- `test/manual/test_double_sparsity_v32.py:250-255` returns the first uppercase A-D character anywhere in the response.
- A completion like `Answer: B` returns `A`, so correct B/C/D answers can be recorded as A if the model emits an answer prefix. This is more likely after Round 26 changed MMLU to `max_new_tokens=4`.

Required fix:
1. Replace `_parse_mmlu_letter` with answer-token parsing, not arbitrary character scanning.
2. Accept a standalone leading `[A-Da-d]` optionally wrapped in punctuation such as `(C)` or `D.`.
3. Otherwise search answer-introducer patterns such as `answer: B`, `answer is B`, `option C`, and extract the following standalone letter.
4. Add helper regressions for `Answer: B -> B`, lowercase `b -> B`, `(C) -> C`, `D. -> D`, and narrative text containing non-answer A-D letters before the final answer.

## Queued Side Issues

- AC-8 prefix-match helper regressions still manually replicate the slicing expression instead of exercising the actual smoke-harness gate.
- Stale `deepseek_v2.py` comments still point at the old `req_to_token_pool.size` slot authority.
- Stale `token_label_table.py` lifetime text still describes overwrite-before-read instead of the Round 6 invalidate-before-selection invariant.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 26 Review`.
- Added a Round 26 Review evolution row with validation and counter-evidence.
- Updated `task-ac12-quality` notes to include the MMLU data skip and parser defects.
- Added blocking side issues for MMLU data preparation and answer parsing.
- Removed the queued unittest-invocation doc issue because Round 26 fixed the docstring.

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

- Keep the mainline objective from @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-27-contract.md stable for this round
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
2. Write your work summary into @/sgl-workspace/sglang/.humanize/rlcr/2026-05-27_11-38-21/round-27-summary.md

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
