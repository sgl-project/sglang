# Round 23 Summary

## Work Completed

Codex Round 22 review verified Round 22's gate fixes were clean and
listed the four remaining Loop 4 code-tier items: AC-12 scaffold
replacement, server-launcher Option B alignment, benchmark sweep
alignment, and `benchmark_compare.py` 3-trial-median + AC-11
directional gate. Round 23 lands the second and third (server +
benchmark scripts) so AC-8 / AC-9 hardware runs can produce
plan-conformant artifacts at the locked operating point.

### Fix 1 — `serve_double_sparsity.sh`

Added the four Option B locked flags Codex listed:
- `--dsa-prefill-backend flashmla_kv`
- `--dsa-decode-backend flashmla_kv`
- `--disable-overlap-schedule`
- `--disable-piecewise-cuda-graph`

Kept `--disable-radix-cache` (AC-10 still pending). Header docstring
updated to enumerate the locked Option B set.

### Fix 2 — `serve_native_nsa.sh`

Added the same four locked flags so DSA + DS launchers differ only
by DS-specific enablement and the AC-10 radix gate. Deliberately did
NOT add `--disable-radix-cache` — per plan §13 the DSA baseline runs
with radix cache ON so the DS-vs-DSA TPS gap reflects DS configuration
alone (the AC-10 radix gate is a separate AC).

### Fix 3 — `benchmark.sh`

- Default `CONCURRENCIES="${CONCURRENCIES:-16 32 64}"` (was `"64"`)
  matches the AC-8 / AC-9 spec.
- Outputs land in `${RESULTS_DIR}` (defaults to
  `$(pwd)/development/results/`) instead of cwd.
- Each run emits a `${OUTPUT_FILE}.meta.json` sidecar capturing
  `commit_sha`, `mode`, `concurrency`, `seed`, `num_prompts`,
  `isl_total_tokens`, `osl_tokens`, `timestamp_utc`, and the
  server's `/get_server_info` JSON (full server args). The AC-11
  comparator can verify both columns share the same operating point.
- Best-effort: `commit_sha` from `git rev-parse HEAD`;
  `server_args` from `curl -s --max-time 5 http://${HOST}:${PORT}/get_server_info`
  (writes `{}` if unreachable, so CI environments do not break).

### Fix 4 — `benchmark_baseline.sh`

Same updates as `benchmark.sh` (default conc 16/32/64, results dir,
meta sidecar), keyed by `MODE=native_nsa` to stay paired with the DS
benchmark.

### Fix 5 — Regression test locking the contract

New file `test/registered/unit/development/test_option_b_scripts.py`:

- **`TestOptionBLockedFlagsServerScripts`** (5 tests):
  - All 4 scripts exist.
  - Both servers carry all 4 locked-flag tokens.
  - DSA does NOT pass `--disable-radix-cache`.
  - DS does pass `--disable-radix-cache` (AC-10 gate).
- **`TestOptionBBenchmarkSweeps`** (4 tests):
  - Both bench scripts default `CONCURRENCIES` to `"16 32 64"`
    (regex on the bash parameter-expansion line).
  - Both bench scripts emit `.meta.json` sidecars carrying
    `commit_sha` + `server_args`.
- **`TestOptionBScriptsSyntax`** (1 test):
  - `bash -n` parses all 4 scripts cleanly.

Helper: `_non_comment_lines(path)` strips `#` lines so assertions
test the *active* command, not the docstring (so a comment
mentioning "does NOT pass --disable-radix-cache" does not
accidentally match the assertion).

## Files Changed

- `development/serve_double_sparsity.sh`: 4 locked Option B flags +
  header docstring update.
- `development/serve_native_nsa.sh`: same 4 locked flags + header
  docstring update explicitly explaining the no-radix-disable
  decision.
- `development/benchmark.sh`: rewritten with conc 16/32/64 default,
  results-dir output, and `.meta.json` sidecar emission.
- `development/benchmark_baseline.sh`: same as benchmark.sh with
  `MODE=native_nsa`.
- `test/registered/unit/development/test_option_b_scripts.py`: NEW
  10-test class locking the contract.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development/test_option_b_scripts.py -q
216 passed, 0 failed (was 206 before this round; +10 Option B regressions)

bash -n development/serve_double_sparsity.sh   # OK
bash -n development/serve_native_nsa.sh        # OK
bash -n development/benchmark.sh               # OK
bash -n development/benchmark_baseline.sh      # OK
```

Targeted greps:
```
grep -E 'dsa-prefill-backend flashmla_kv'      development/serve_*.sh   # 2 hits
grep -E 'dsa-decode-backend flashmla_kv'       development/serve_*.sh   # 2 hits
grep -E 'disable-overlap-schedule'             development/serve_*.sh   # 2 hits
grep -E 'disable-piecewise-cuda-graph'         development/serve_*.sh   # 2 hits
grep -E 'disable-radix-cache'                  development/serve_native_nsa.sh   # 1 (comment only — explanation)
grep -E 'disable-radix-cache'                  development/serve_double_sparsity.sh # 2 (active flag + AC-10 comment)
grep -E 'CONCURRENCIES:-16 32 64'              development/benchmark*.sh   # 2 hits
grep -E '\.meta\.json'                         development/benchmark*.sh   # multiple
```

Commit: `3ab86e868` — [AC-8/9] Align Option B launchers + benchmark
sweeps to plan §13.

## Remaining Items

Mainline AC items still requiring hardware execution:
- `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`,
  `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`,
  `task-ac9-baseline`, `task-ac10-radix`, `task-ac11-compare`,
  `task-ac12-quality`.

Code-tier items still queued for future rounds:
- Replace `test/manual/test_double_sparsity_v32.py` skip-only scaffolds
  with real NIAH 4K/16K/64K + MMLU 5-shot execution against paired
  servers (AC-12).
- `benchmark_compare.py` 3-trial median + AC-11 directional gate
  enforcement (DS TPS within 5% of DSA; P99 TTFT ≤ 1.10× DSA).
- Shallow prefix-match regression coverage cleanup (Codex Round 22
  queued).
- Stale DS bind/runtime comments + token-label lifetime docs.

## Push-to-remote Status

Branch is 24 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`; commits remain local only. Per-round pushing requires
re-launching with `--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: This was a mechanical, plan-derived alignment of dev-tier
scripts to the locked operating point. No surprising failure mode
or non-obvious workaround; the regression test exists so the
contract is locked, but the bug pattern itself is too narrow to
generalize. No new BitLesson entry warranted.
