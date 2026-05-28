# Round 23 Code Review

Mainline Progress Verdict: ADVANCED

Round 23 made real mainline progress: the DS and DSA launch scripts now carry the locked Option B backend/scheduler flags, and both benchmark scripts default to the required 16/32/64 sweep. However, the new metadata sidecar path is not safe against real `/get_server_info` JSON, so the AC-8/AC-9 artifact path still needs a blocking fix before hardware runs can be accepted as reproducible evidence. The original Loop 4 plan remains incomplete: AC-12 is still a skip-only scaffold, AC-11 comparator enforcement is still missing, and all hardware gates remain active.

## Implementation Review

Verified Round 23 claims:
- `development/serve_double_sparsity.sh:55-68` now passes `--dsa-prefill-backend flashmla_kv`, `--dsa-decode-backend flashmla_kv`, `--disable-overlap-schedule`, `--disable-piecewise-cuda-graph`, and still passes `--disable-radix-cache`.
- `development/serve_native_nsa.sh:46-56` now passes the same four locked Option B flags and does not actively pass `--disable-radix-cache`.
- `development/benchmark.sh:40` and `development/benchmark_baseline.sh:43` default `CONCURRENCIES` to `16 32 64`.
- Both benchmark scripts write under `development/results/` by default and attempt `.meta.json` sidecars.
- `test/registered/unit/development/test_option_b_scripts.py` exists and locks the basic script contract.

Validation run:
```text
PYTHONPATH=python pytest test/registered/unit/development/test_option_b_scripts.py -q
10 passed, 1 warning

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
206 passed, 24 warnings

bash -n development/serve_double_sparsity.sh
bash -n development/serve_native_nsa.sh
bash -n development/benchmark.sh
bash -n development/benchmark_baseline.sh
all passed
```

## Goal Alignment Summary

```text
ACs: 11/15 addressed | Forgotten items: 0 | Unjustified deferrals: 2
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.
Partial: AC-1, AC-4, AC-6, AC-8, AC-9.
Not met: AC-1b, AC-10, AC-11, AC-12.

No original-plan task is missing from Active, Completed, or Deferred after tracker reconciliation. `Explicitly Deferred` remains empty, which is correct. Claude's "future rounds" language for AC-12 scaffold replacement and AC-11 comparator enforcement is not accepted as justified deferral; both remain incomplete plan work.

## Findings By Lane

### Mainline Gaps

1. AC-12 remains unimplemented even though it is a hard loop-closure gate.

Evidence:
- `test/manual/test_double_sparsity_v32.py:62-94` still skips NIAH 4K, NIAH 16K, NIAH 64K, MMLU 5-shot, and both negative sensitivity checks.
- `test/manual/test_double_sparsity_v32.py:31-34` still says the actual NIAH/MMLU evaluation is out of scope for the scaffold.
- The refined plan states AC-12 is hard and the loop does not close without NIAH deltas ≤ 5 pp and MMLU delta ≤ 1 pp.

Required implementation plan:
1. Replace `test/manual/test_double_sparsity_v32.py` with a paired-server harness that requires `DS_BASE_URL` and `DSA_BASE_URL`, plus optional `MODEL` auto-detected from `/server_info`.
2. Reuse the existing HTTP request helpers from `test/manual/test_dsv32_quality_smoke.py` for deterministic temperature-0 generation and result recording.
3. Implement deterministic NIAH prompts at 4K, 16K, and 64K with fixed seeds, a planted needle placed outside the local window, exact needle scoring, and per-length DS/DSA recall percentages.
4. Implement MMLU 5-shot through `sglang.test.run_eval` with `eval_name="mmlu"`, once against DSA and once against DS, persisting each score and the aggregate delta.
5. Fail when `DSA - DS > 5 pp` for any NIAH length or `DSA - DS > 1.0 pp` for MMLU.
6. Convert the corrupt-mask and zero-signature sensitivity tests into executable fault-injection checks; do not leave them as `skipTest`.

2. AC-11 comparator enforcement is still the old single-run/SLO report, not the plan-required 3-trial median directional gate.

Evidence:
- `development/benchmark_compare.py:21-25` still documents an AC-8-style absolute SLO/no-op detector.
- `development/benchmark_compare.py:334-340` accepts exactly one baseline JSONL and one DS JSONL.
- `development/benchmark_compare.py:254-259` checks fixed P50 TPS/P99 TTFT thresholds, not DS TPS within 5% of DSA and P99 TTFT ≤ 1.10x DSA.

Required implementation plan:
1. Change the CLI to accept trial sets per mode, for example `--baseline-results path1 path2 path3 --ds-results path1 path2 path3`, and group by concurrency.
2. Parse each JSONL plus its `.meta.json`; require fixed seed policy, matching concurrency, matching commit/server args except DS enablement/config and the radix-cache state allowed before AC-10.
3. For each concurrency, compute median TPS, P99 TTFT, TPOT, and goodput across at least three independent trials.
4. Enforce AC-11: DS TPS must be at least 95% of DSA TPS; DS P99 TTFT must be ≤ 1.10 × DSA P99 TTFT. A failure emits a profiling obligation in the report and exits non-zero.
5. Add registered unit tests for median selection, missing-trial rejection, server-arg mismatch rejection, TPS directional failure, and TTFT ratio failure.

3. The remaining original-plan hardware gates have not run.

Required execution plan:
1. Generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200 with the AC-4 production command.
2. Run `task-ac1-hwtest`: real H200 `forward_extend`, then assert token-label rows at `out_cache_loc` are non-zero.
3. Run `task-ac6-hwrun`: V3.2 Option B conc=64 full-graph capture, 100 replays, no CUDA launch failure, eager/graph `max_abs_diff <= 1e-6`.
4. Run `task-ac1b-probe`: `chunked_prefill_size=4096`, compare labels 0..4095 against non-chunked baseline, and record the pass/fail decision.
5. After fixing the metadata blocker below, run AC-8 server smoke and paired AC-8 quality smoke.
6. Complete AC-9, AC-10, and AC-11 rather than treating them as complete-by-deferral.

### Blocking Side Issues

1. The benchmark sidecar writer fails against real `/get_server_info` JSON and omits the promised chunked-prefill metadata.

Evidence:
- `development/benchmark.sh:52` and `development/benchmark_baseline.sh:53` capture raw `/get_server_info` JSON into `SERVER_ARGS_JSON`.
- `development/benchmark.sh:71-83` and `development/benchmark_baseline.sh:72-84` splice that raw JSON into a Python heredoc as source code.
- `/get_server_info` returns `dataclasses.asdict(server_args)` plus extra fields from `http_server.py:641-650`; that JSON contains booleans and nulls. JSON `true` / `null` are invalid Python identifiers, so sidecar generation raises `NameError` after a successful benchmark run.
- The Round 23 contract required `chunked_prefill_size`, but the sidecar only records commit, mode, concurrency, seed, prompt counts, ISL/OSL, timestamp, and `server_args`.

Required fix:
1. In both benchmark scripts, pass the curl output to Python as data, not source. Use an environment variable or temp file and call `json.loads`; on empty/invalid data, record `{}` plus an error field.
2. Record `chunked_prefill_size` explicitly from `server_args.get("chunked_prefill_size")`, falling back to `"unknown"` if unavailable.
3. Add the AC-11 reproducibility fields while touching this path: `warmup_requests`, `measurement_window_seconds` or actual `duration`, `trial_id`, and request-arrival seed/rate.
4. Add a registered regression that feeds `{"disable_radix_cache": true, "kv_events": null}` through the sidecar writer and proves valid JSON is emitted.

### Queued Side Issues

- The new Option B script tests are useful but shallow: they assert string presence, not that sidecar JSON survives real server-info shapes. This should be strengthened as part of the blocking sidecar fix.
- The AC-8 prefix-match helper tests still replicate the slicing condition instead of exercising `TestDSv32QualitySmoke.test_quality_smoke` through mocked `_run_paired` / `_record`.
- Existing docs drift remains: stale `deepseek_v2.py` slot-authority comments and `token_label_table.py` lifetime text.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 23 Review`.
- Added a Round 23 Review plan-evolution row.
- Added the metadata sidecar bug to Blocking Side Issues and noted it on `task-ac8-server` / `task-ac9-baseline`.
- Replaced the stale queued "Option B scripts/comparator" item with the remaining `benchmark_compare.py` AC-11 comparator gap, since the launcher flags and sweep defaults are now verified.

NOT COMPLETE
