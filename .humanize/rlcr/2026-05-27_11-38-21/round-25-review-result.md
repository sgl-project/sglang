# Round 25 Code Review

Mainline Progress Verdict: ADVANCED

Round 25 made real mainline progress by replacing the six-test skip scaffold with executable NIAH/MMLU/sensitivity test bodies and adding 11 helper regressions. That advances AC-12 code-tier coverage, but it does not close AC-12. The MMLU path is not currently a runnable 5-shot gate against the documented server URLs, the sensitivity fault-injection server gates were deferred, and no H200 quality run has passed.

## Implementation Review

Verified claims:
- [test/manual/test_double_sparsity_v32.py](/sgl-workspace/sglang/test/manual/test_double_sparsity_v32.py:206) now has real NIAH execution and delta assertions for 4K, 16K, and 64K.
- [test/manual/test_double_sparsity_v32.py](/sgl-workspace/sglang/test/manual/test_double_sparsity_v32.py:299) has an MMLU test body instead of an unconditional skip.
- [test/manual/test_double_sparsity_v32.py](/sgl-workspace/sglang/test/manual/test_double_sparsity_v32.py:353) has executable sensitivity test bodies gated by optional fault-injected server URLs.
- [test/registered/unit/manual/test_ac12_helpers.py](/sgl-workspace/sglang/test/registered/unit/manual/test_ac12_helpers.py:16) covers deterministic needles, prompt construction, length/depth bounds, suffix, and recall scoring.

Validation I ran:

```text
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
11 passed, 1 warning

env -u DS_BASE_URL -u DSA_BASE_URL -u DS_CORRUPT_MASK_URL -u DS_ZERO_SIG_URL \
  PYTHONPATH=python python -m pytest test/manual/test_double_sparsity_v32.py -q
6 skipped, 1 warning
```

Counter-evidence:

```text
PYTHONPATH=python python -m unittest test.manual.test_double_sparsity_v32 -v
ERROR: ModuleNotFoundError: No module named 'test.manual'
```

The documented unittest module path fails because Python imports the stdlib `test` package, while the repo `test/` tree is not a package. Pytest-by-file works, so this is secondary to the AC-12 gate bugs below.

## Goal Alignment Summary

```text
ACs: 13/15 addressed | Forgotten items: 1 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-9, AC-11, AC-12.

Not met: AC-1b, AC-10.

Forgotten item: the server-side `SGLANG_DS_FAULT_INJECT_*` gates are not tracked as an active task even though Round 25 deferred them and the sensitivity tests cannot actually run without them.

Unjustified deferral: server-side AC-12 fault-injection plumbing. The Round 25 contract called it stretch, but the user instruction for this review explicitly rejects deferrals as incomplete.

## Mainline Gaps

1. AC-12 is still incomplete.

The NIAH harness is useful, but the hard gate has not passed on paired DS/DSA H200 servers. The sensitivity half is also incomplete because the harness only skips unless `DS_CORRUPT_MASK_URL` / `DS_ZERO_SIG_URL` are supplied, while this branch provides no way to boot those fault-injected servers. The plan says the loop does not close without AC-12 passing.

Required implementation plan:
1. Fix the MMLU harness defects in the blocking section below.
2. Add the two server-side fault-injection gates in the blocking section below.
3. Run AC-12 on H200 with `DS_BASE_URL` and `DSA_BASE_URL`: NIAH @ 4K/16K/64K, then MMLU 5-shot.
4. Boot fault-injected DS servers and run both sensitivity tests.
5. Persist artifacts with aggregate score, per-length/per-subject breakdown, hit counts, thresholds, URLs or server metadata, commit SHA, and env knobs used.

2. Remaining original-plan gates are still active and must not be treated as complete-by-deferral.

Pending original-plan tasks: `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`, and `task-ac11-compare`.

Required execution plan:
1. Generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200.
2. Run AC-1 forward population and AC-6 real V3.2 CUDA-graph capture/replay.
3. Run AC-1b chunked-prefill probe and record the pass/fail launch decision.
4. Run AC-8 server smoke and paired AC-8 quality smoke.
5. Generate AC-9 DSA baseline JSON.
6. Complete AC-10 radix fixture and only then remove DS `--disable-radix-cache`.
7. Implement and run AC-11 3-trial median comparator.

## Blocking Side Issues

1. AC-12 MMLU is not a runnable 5-shot gate.

Evidence:
- [test/manual/test_double_sparsity_v32.py](/sgl-workspace/sglang/test/manual/test_double_sparsity_v32.py:318) passes `base_url` directly to `run_eval_once`.
- The documented `DS_BASE_URL=http://localhost:30000` is a root URL for `/generate`; sglang exposes OpenAI endpoints under `/v1`, for example [http_server.py](/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py:1498) and [http_server.py](/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py:1608). The normal CLI adds `/v1` before `run_eval_once` in [run_eval.py](/sgl-workspace/sglang/python/sglang/test/run_eval.py:104), but the new harness does not.
- [simple_eval_mmlu.py](/sgl-workspace/sglang/python/sglang/test/simple_eval_mmlu.py:101) formats one multiple-choice question. [simple_eval_common.py](/sgl-workspace/sglang/python/sglang/test/simple_eval_common.py:244) contains no 5-shot examples. This is not the plan-required MMLU 5-shot.

Required fix:
1. Keep `DS_BASE_URL` / `DSA_BASE_URL` as root server URLs for `/generate`.
2. Add `_openai_base_url(base_url)` that returns `base_url.rstrip("/") + "/v1"` unless it already ends in `/v1`; use it only for OpenAI-compatible clients.
3. Replace the current MMLU body with true 5-shot prompt construction. Use the existing [benchmark/mmlu/bench_sglang.py](/sgl-workspace/sglang/benchmark/mmlu/bench_sglang.py:33) formatting helpers or equivalent in-harness code: five dev examples with answers, then each test question without answer, `max_new_tokens=1`, temperature 0, first stripped A-D token as prediction.
4. Add registered helper tests for URL normalization and 5-shot prompt construction so this cannot silently become zero-shot again.

2. Server-side fault injection is missing, so the sensitivity checks are skip-only in practice.

Evidence:
- The sensitivity tests require `DS_CORRUPT_MASK_URL` and `DS_ZERO_SIG_URL`, but `rg SGLANG_DS_FAULT_INJECT` only finds the manual test strings, not server implementation.
- The correct bind point for corrupting the local channel mask is [deepseek_v2.py](/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py:1893), before `bind_runtime_data`.
- The correct centralized write point for zeroing signatures is [dsa_backend.py](/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py:1432), after `token_label_write`.

Required fix:
1. In `deepseek_v2.py`, after `slice_per_rank(...)`, if `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK=1`, construct a new `ChannelMask` with deterministic random channel selections within `[0, head_dim)`, preserve shape/dtype, and log a warning once per process.
2. In `dsa_backend.NativeSparseAttnBackend.__init__`, cache `self._ds_fault_zero_sig = os.getenv("SGLANG_DS_FAULT_INJECT_ZERO_SIG") == "1"` and log a warning when enabled.
3. In `_write_token_labels`, after `token_label_write`, zero `signatures[layer_id, cache_loc]` while leaving `written[layer_id, cache_loc] = True`, so the selector sees intentionally bad labels rather than treating the slots as absent.
4. Add unit tests: corrupt-mask changes selections but keeps shape/range/dtype; zero-signature write leaves `written=True` and row values zero; default env leaves existing behavior untouched.
5. Run the two manual sensitivity tests against fault-injected servers before marking AC-12 complete.

## Queued Side Issues

- The AC-12 artifact writer omits hit counts in the positive NIAH artifacts even though the summary claims `dsa_hits` and `ds_hits` are recorded. This weakens auditability but is smaller than fixing the gate itself.
- The documented `python -m unittest test.manual.test_double_sparsity_v32` invocation fails; publish the pytest file-path command or make the repo test tree importable before operator handoff.
- Existing queued items remain valid: AC-8 prefix-match regression depth, stale DS bind/runtime comments, stale token-label lifetime docs, and the AC-11 comparator until it becomes the active benchmark-evidence task.

## Goal Tracker Update

I updated the mutable section of `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 25 Review`.
- Added a Round 25 Review evolution row with the validation and counter-evidence above.
- Tightened `task-ac12-quality` notes so it does not imply MMLU/fault-injection readiness.
- Added blocking side issues for the MMLU harness and missing `SGLANG_DS_FAULT_INJECT_*` gates.
- Added the failing unittest module path as a queued side issue.

NOT COMPLETE
