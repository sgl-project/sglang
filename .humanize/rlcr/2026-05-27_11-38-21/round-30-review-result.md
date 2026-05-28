# Round 30 Code Review

Mainline Progress Verdict: ADVANCED

Round 30 advanced AC-11 by adding the median helpers, directional TPS/TTFT gate math, Markdown/JSON output, and registered regressions for those local semantics. The new targeted suite passes.

However, AC-11 is still not acceptance-ready. The original plan's AC-11 contract is not just "3 files and ratios"; it requires a fixed request-arrival seed, a 120s warmup, a minimum 600s measurement window, at least 3 independent trials, median aggregation, and recorded commit SHA + full server args + chunked-prefill setting. The comparator and scripts do not yet enforce or produce that evidence, so a false AC-11 PASS can still be published.

## Implementation Review

Validation I ran:

```text
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
24 passed, 1 warning
```

Counter-evidence:

```text
python development/benchmark_compare.py --ac11 \
  --ac11-baseline-results dsa_1.jsonl dsa_2.jsonl dsa_3.jsonl \
  --ac11-ds-results ds_1.jsonl ds_2.jsonl ds_3.jsonl

Result: exit 0, "AC-11 verdict: PASS"
```

The fabricated DSA files had `num_prompts=320`, `input_len=4096`, `output_len=512`, `tp_size=8`, `page_size=64`, `gpu_id=0`, `disable_radix_cache=false`. The fabricated DS files had `num_prompts=1`, `input_len=1`, `output_len=1`, `tp_size=1`, `page_size=1`, `gpu_id=1`, `disable_radix_cache=true`. No `.meta.json` sidecars existed. The comparator still passed because only the ratio fields and concurrency aligned.

I also reproduced an input-refusal crash path:

```text
python development/benchmark_compare.py --ac11 \
  --ac11-baseline-results malformed_t1_c64.jsonl malformed_t2_c64.jsonl malformed_t3_c64.jsonl \
  --ac11-ds-results ds_t1_c64.jsonl ds_t2_c64.jsonl ds_t3_c64.jsonl

Result: uncaught json.decoder.JSONDecodeError, exit 1
```

This happens because `_group_by_concurrency` catches JSON read errors and falls back to `_c<N>.jsonl`, but `_run_ac11_mode` reads the same files again outside a refusal wrapper.

## Goal Alignment Summary

```text
ACs: 13/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-9, AC-11, AC-12.

Not met: AC-1b, AC-10.

No original-plan task is forgotten in the tracker. The remaining hardware-gated work is still Active, not Deferred. I updated the mutable tracker to reopen AC-11 code-tier completeness and added two AC-11 blocking side issues.

## Mainline Gaps

1. AC-11 comparator does not enforce the reproducibility and apples-to-apples contract.

Evidence:
- `development/benchmark_compare.py:572-615` only groups by concurrency, checks trial counts, reads metrics, and computes medians/gates.
- `development/benchmark_compare.py:291-308` checks `concurrency`, `num_prompts`, `isl`, and `osl` only within one side's trial set, not between DSA and DS.
- AC-11 mode never reads `${result}.meta.json`, never calls `_match_or_refuse`, and never validates fixed seed, warmup/window length, commit SHA, server args, chunked-prefill, TP/page/radix/hardware, or "only DS flags differ".
- The reproducer above returns PASS with no sidecars and mismatched workload/server config.

Required implementation plan:
1. Add `_sidecar_path(result_path)`, `_read_ac11_meta(result_path)`, and `_normalize_ac11_server_args(meta)` in `development/benchmark_compare.py`.
2. For every input JSONL, require a sibling `.meta.json`; malformed or missing sidecars must return exit 2.
3. Validate every trial has all gate metrics needed for AC-11 (`output_tps_p50` and `ttft_p99_s`). Do not let `_median` turn one valid sample plus two `None`s into a 3-trial median.
4. Validate sidecar fields before computing medians:
   - `seed` must be present and equal across DSA and DS for the same concurrency/trial.
   - `num_prompts`, `isl_total_tokens`, and `osl_tokens` must match across DSA and DS and must agree with the JSONL metrics.
   - `warmup_seconds >= 120` or an equivalently explicit warmup-duration field must be present.
   - `measurement_window_seconds >= 600` must be present and must agree with the JSONL summary duration where available.
   - `commit_sha` and `chunked_prefill_size` must match across DSA and DS.
   - `server_args_error` must be null.
   - normalized `server_args` must match after removing only `enable_double_sparsity` / `double_sparsity_config` and any DS-only fault-injection flags that are not part of the benchmark run.
5. Reuse `_match_or_refuse` semantics for GPU id, TP size, page size, radix-cache setting, and concurrency in AC-11 mode.
6. Add registered negative tests for missing sidecar, malformed sidecar, short/missing warmup, short/missing measurement window, mismatched seed, mismatched workload, mismatched commit, mismatched chunked-prefill, disallowed server-arg mismatch, and missing gate metrics.

2. The benchmark scripts cannot safely produce the required 3 independent AC-11 trial set.

Evidence:
- `development/benchmark.sh:46-47` writes `..._c${CONCURRENCY}.jsonl` and `.meta.json`; no trial id appears in the filename.
- `development/benchmark_baseline.sh:49-50` has the same overwrite behavior.
- Both scripts set `TRIAL_ID="${TRIAL_ID:-1}"` only inside the sidecar env (`benchmark.sh:83`, `benchmark_baseline.sh:82`), so repeated runs overwrite the previous trial's JSONL/meta.
- Both scripts leave `WARMUP_REQUESTS` and `MEASUREMENT_WINDOW_S` empty by default (`benchmark.sh:84-85`, `benchmark_baseline.sh:83-84`) and the measured command does not enforce a 120s warmup or 600s window.

Required implementation plan:
1. Add an explicit AC-11 trial loop to both `development/benchmark.sh` and `development/benchmark_baseline.sh`: `TRIALS="${TRIALS:-3}"`, `TRIAL_ID=1..TRIALS`.
2. Include the trial id in output names, e.g. `${MODE}_gsp_isl4096_osl512_c${CONCURRENCY}_t${TRIAL_ID}.jsonl`, with matching `.meta.json`.
3. Add required `WARMUP_SECONDS="${WARMUP_SECONDS:-120}"` and `MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-600}"` controls. The scripts must not silently default these to empty metadata.
4. Run a pre-measurement warmup workload for `WARMUP_SECONDS` using the same dataset shape, seed family, and concurrency, discarding the warmup JSONL.
5. Run the measured workload so the JSONL summary duration is at least `MEASUREMENT_WINDOW_S`; fail the script if the summary duration is shorter.
6. Write `warmup_seconds`, `measurement_window_seconds`, `trial_id`, `seed`, workload shape, `commit_sha`, `chunked_prefill_size`, and full server args into the sidecar for each trial.
7. Add registered script tests that three trials create distinct output paths and that the comparator refuses any fixture missing the new timing/trial metadata.

3. Remaining original-plan gates are still active and must not be treated as complete-by-deferral.

Pending original-plan tasks:
- `task-ac1-hwtest`
- `task-ac4-hwrun`
- `task-ac6-hwrun`
- `task-ac1b-probe`
- `task-ac8-server`
- `task-ac8-quality`
- `task-ac12-quality`
- `task-ac9-baseline`
- `task-ac10-radix`
- `task-ac11-compare`

Required execution plan after fixing AC-11 code-tier:
1. Generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200.
2. Run AC-1 live label population and AC-6 V3.2 conc=64 CUDA graph capture/replay.
3. Run AC-1b chunked-prefill probe and record the pass/fail launch decision.
4. Run AC-8 DS bench_serving conc 16/32/64 and same-session AC-8 quality smoke.
5. Run paired AC-12 NIAH @ 4K/16K/64K and full intended MMLU 5-shot.
6. Generate AC-9 DSA baseline JSON.
7. Complete AC-10 radix: M3-B hardware fixture, FP8 scale stability, radix flag flip, and launch-script update.
8. Run the fixed AC-11 3-trial DSA+DS sweep and comparator.

## Blocking Side Issues

1. AC-11 input refusal is not robust for malformed JSONL files with a `_c<N>.jsonl` filename suffix.

Evidence:
- `_group_by_concurrency` catches all exceptions from `_read_bench_jsonl` and falls back to filename parsing at `development/benchmark_compare.py:342-346`.
- `_run_ac11_mode` re-reads those files at `development/benchmark_compare.py:607-608` without catching JSON parse errors.
- Malformed `*_c64.jsonl` inputs produce a traceback and exit 1, not the documented exit 2 refusal.

Required fix:
1. Stop swallowing parse errors in `_group_by_concurrency`; only fall back to filename when the JSONL parsed successfully but lacks concurrency.
2. Wrap the second `_read_bench_jsonl` pass in `_run_ac11_mode` and return exit 2 for any `FileNotFoundError`, `ValueError`, or `json.JSONDecodeError`.
3. Add a registered regression for malformed `_c64.jsonl` inputs asserting exit 2 and no traceback.

## Queued Side Issues

- AC-8 prefix-match helper regressions still manually replicate the slicing expression instead of exercising the actual smoke-harness gate.
- `deepseek_v2.py` still has stale comments/docstrings saying `max_tokens = req_to_token_pool.size` even though Round 2 corrected the authority to the physical KV slot address space.
- `token_label_table.py` module docstring still says reused slots are safe because the write hook overwrites before read, but Round 6 changed the invariant to invalidate before selection and rewrite later.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md` mutable content only:
- Plan Version now says `Updated: Round 30 Review`.
- Added a Round 30 Review plan-evolution row reopening AC-11 code-tier completeness.
- Updated `task-ac11-compare` notes to require sidecar/timing/cross-column validation and non-overwriting trial artifacts before H200 execution is accepted.
- Added AC-11 blocking side issues for missing comparator metadata validation and benchmark-script trial/timing gaps.
- Removed the stale queued row claiming the AC-11 comparator issue was fully resolved.

NOT COMPLETE
