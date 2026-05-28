# Round 30 Summary

## Work Completed

Codex Round 29 full goal alignment review found no Round 29 code
bug. The single piece of remaining code-tier mainline work the
plan still needed was the AC-11 directional comparator, which was
"queued only because AC-12/hardware evidence should be tackled
first." Hardware execution is outside my reach, so Round 30 lands
the AC-11 comparator.

Plan §AC-11 spec:

> Fixed seed, 600s window, 120s warmup, 3 trials, median.
> DS TPS within 5% of DSA TPS at conc=64 (directional gate).
> P99 TTFT ≤ DSA-on P99 TTFT × 1.10.

### Fix 1 — Pure helpers in `development/benchmark_compare.py`

- `_median(values)` — handles odd / even / single / None-filtered;
  returns `None` on empty or all-None.
- `_median_metrics(trials: List[RunMetrics]) -> RunMetrics` —
  per-field median across trials. `dense_fallback_total` is summed
  (counter, not sample). `concurrency` / `num_prompts` / `isl` /
  `osl` must agree across trials within one side — refuses
  otherwise so a misgrouped sweep cannot silently pass.
- `_group_by_concurrency(paths)` — groups trial JSONLs by
  resolved concurrency (per-row JSON preferred; `_c<N>.jsonl`
  filename suffix as fallback); raises `ValueError` when no
  resolution is possible.
- `_evaluate_ac11_gates(dsa_median, ds_median)` — computes
  `tps_ratio = ds_tps / dsa_tps` and `ttft_ratio = ds_ttft_p99 /
  dsa_ttft_p99`; gates: `tps_ratio >= 0.95` and `ttft_ratio <= 1.10`.
  Refuses on missing data or degenerate denominators with a
  `missing-data` reason.

Constants: `AC11_TPS_FLOOR_RATIO = 0.95`,
`AC11_TTFT_CEIL_RATIO = 1.10`, `AC11_MIN_TRIALS = 3`.

### Fix 2 — `--ac11` CLI mode

New flag `--ac11` plus arg lists `--ac11-baseline-results` and
`--ac11-ds-results` (each ≥ 3 paths per concurrency).
`_run_ac11_mode(args)`:

- Groups inputs by concurrency.
- Refuses if either side has < 3 trials at any concurrency, or if
  the concurrency sets disagree across sides, or if any per-trial
  operating-point invariant fires.
- For each concurrency, computes paired DSA and DS medians and
  evaluates both gates.
- Markdown report: per-concurrency
  `DSA TPS p50 | DS TPS p50 | TPS ratio | TPS gate | DSA TTFT p99
  | DS TTFT p99 | TTFT ratio | TTFT gate` table; final
  `AC-11 verdict: PASS|FAIL`. On any failure, the report lists each
  failing concurrency under a "Profiling obligation" header naming
  the failed gates + actual ratios.
- JSON report: `ac11_gates` constants + `per_concurrency` rows +
  top-level `verdict`.

Exit codes:
- `0` — all concurrencies pass both gates.
- `3` — at least one gate failed (profiling obligation triggered).
- `2` — input refusal (too few trials, mismatched concurrency
  sets, unresolvable concurrency, or `_median_metrics` invariant
  violation).

Module docstring updated to enumerate the two CLI modes
(single-trial AC-7/AC-8 legacy + new AC-11 directional).

### Fix 3 — Registered regressions

`test/registered/unit/development/test_ac11_comparator.py` (24
tests):

- `TestMedianHelper` × 6 — odd, even, single, None-filtered,
  empty → None, all-None → None.
- `TestMedianMetrics` × 3 — per-field medians, dense_fallback
  summed, concurrency-mismatch refusal.
- `TestGroupByConcurrency` × 3 — JSON-derived, filename fallback,
  unresolvable refusal.
- `TestEvaluateAC11Gates` × 6 — TPS pass at equality + at 0.95
  floor + fail below floor; TTFT pass at 1.10 ceiling + fail
  above; missing-data path marks both failed.
- `TestAC11EndToEnd` × 6 — full pass → exit 0 + Markdown +
  JSON output; TPS fail → exit 3 + obligation message; TTFT fail →
  exit 3; <3 trials → exit 2; concurrency-set mismatch → exit 2;
  legacy single-trial mode still works (no regression).

Loader uses `sys.modules["_bc"] = mod` before `exec_module` per
`BL-20260527-importlib-dataclass-sys-modules` so the `@dataclass`
decorators in `benchmark_compare.py` resolve correctly.

## Files Changed

- `development/benchmark_compare.py`: +4 pure helpers, +AC-11 CLI
  mode (`--ac11` + `--ac11-baseline-results` + `--ac11-ds-results`),
  +rendering + JSON output. Module docstring updated. Legacy
  single-trial path unchanged behaviorally.
- `test/registered/unit/development/test_ac11_comparator.py`:
  NEW — 24 registered regressions covering helpers and end-to-end
  CLI exit behavior.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
24 passed, 0 failed (new file)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
304 passed, 0 failed (was 280)
```

Sanity smoke (CLI happy path):
```
--ac11 + 3 DSA + 3 DS at conc=64 with paired DSA/DS TPS≈DS TPS,
P99 TTFT≈DSA TTFT → exit 0, "AC-11 verdict: PASS"

--ac11 + DS TPS = 50% of DSA → exit 3, "AC-11 verdict: FAIL",
"AC-11 TPS gate failed: DS/DSA = 0.5000 < 0.95", "Profiling obligation"
```

Commit: `00fdd6cb8` — [AC-11] Add 3-trial directional comparator
+ 24 CI regressions.

## Remaining Items

Code-tier items still queued for future rounds:

- Shallow AC-8 prefix-match regression coverage cleanup
  (Codex Round 22 queued item).
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

These are all minor; the loop's remaining mainline blockers are
purely hardware-gated now.

Hardware-gated tasks unchanged: `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac10-radix`, `task-ac11-compare` (comparator code-tier
landed — only the 3-trial H200 sweep + invocation remains),
`task-ac12-quality` (harness fully gate-tight — only the
H200 paired-server run remains).

## Push-to-remote Status

Branch is 31 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 30 applied three existing BitLessons:
`BL-20260527-shell-json-into-python-source` (sidecar JSON read via
the existing safe paths), `BL-20260527-conservative-llm-output-parser`
(fail-loud refusal rather than silent skip when AC-11 inputs are
malformed), `BL-20260527-importlib-dataclass-sys-modules` (loader
registers `_bc` in `sys.modules` before `exec_module`). No new
generalizable failure mode worth a fresh entry.
