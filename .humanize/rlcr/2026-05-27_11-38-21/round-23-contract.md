# Round 23 Contract

## Mainline Objective

Align the AC-8 / AC-9 launch + benchmark scripts with the locked
Option B operating point so the upcoming H200 hardware runs produce
plan-conformant evidence. Today, two of the four scripts are missing
required flags entirely, and both benchmark sweeps default to
concurrency=64 only instead of the AC-8/AC-9 spec of 16/32/64.

Codex Round-22-review framing: AC-6 coding is closed, Round 22's
gate fixes are clean. The next plan-derived bottleneck for the
remaining AC-8 / AC-9 / AC-11 hardware gates is the Option B run
tooling. Without these changes the operator either has to remember
to add the locked flags manually (error-prone), or the resulting
JSONL artifacts will be at a different operating point than the
plan locks down in §13 / DEC-1.

## Target ACs

- **AC-8** — the DS server launches with the locked Option B flags;
  the benchmark script sweeps conc 16/32/64 by default; each result
  carries the run metadata required for reproducibility.
- **AC-9** — the DSA baseline server launches with the SAME locked
  Option B flags as DS (modulo `--enable-double-sparsity` /
  `--disable-radix-cache`); the baseline benchmark sweeps the same
  three concurrencies and writes the JSONL into
  `development/results/` with a metadata sidecar.

## Required Implementation

### Fix 1: `development/serve_double_sparsity.sh`

- Add the four locked Option B flags Codex listed:
  - `--dsa-prefill-backend flashmla_kv`
  - `--dsa-decode-backend flashmla_kv`
  - `--disable-overlap-schedule`
  - `--disable-piecewise-cuda-graph`
- Keep `--disable-radix-cache` (AC-10 still pending).
- Header docstring updated to enumerate the locked Option B set.

### Fix 2: `development/serve_native_nsa.sh`

- Add the same four locked Option B flags (so DSA + DS are bit-identical
  except for DS-specific enablement).
- Do NOT add `--disable-radix-cache` — DSA baseline runs with radix on
  per plan §13.
- Header docstring updated.

### Fix 3: `development/benchmark.sh`

- Default `CONCURRENCIES="${CONCURRENCIES:-16 32 64}"` (sweep the three
  plan-locked concurrencies; operators can still override to a single
  value).
- Move output files into `development/results/` (so they sit next to
  the comparator's expected location and don't collide with cwd
  artifacts).
- Emit a metadata sidecar `${OUTPUT_FILE}.meta.json` capturing:
  `{commit_sha, mode, concurrency, seed, chunked_prefill_size,
  num_prompts, isl, osl, timestamp_utc, server_args_via_info_endpoint}`.
  Best-effort: `commit_sha` from `git rev-parse HEAD`; server args
  from `curl -s http://${HOST}:${PORT}/get_server_info`.

### Fix 4: `development/benchmark_baseline.sh`

- Same conc 16/32/64 default; same results-dir + metadata-sidecar
  treatment; `MODE=native_nsa` is unchanged.

### Fix 5: Lock the new behavior with a regression test

- `test/registered/unit/development/test_option_b_scripts.py` (new):
  - Reads each of the four scripts as text.
  - Asserts both server scripts contain the four locked-flag tokens.
  - Asserts `serve_native_nsa.sh` does NOT contain
    `--disable-radix-cache`.
  - Asserts both benchmark scripts default to `"16 32 64"` (regex on
    the parameter expansion line).
  - Asserts both benchmark scripts write metadata sidecars (regex on
    the `meta.json` literal).

Pure-Python; no hardware required; CI-runnable.

## Tests

- Existing 206 tests still pass.
- New test class adds ~5 assertions (one method or per-script methods).
- Expect ≥ 207 passed.

## Success Criteria

1. `grep -E "flashmla_kv|disable-overlap-schedule|disable-piecewise-cuda-graph"`
   on each server script matches 3 lines (one per Option B locked flag
   plus the two backend lines).
2. `serve_native_nsa.sh` does NOT contain `--disable-radix-cache`.
3. Both benchmark scripts: default `CONCURRENCIES` is the three-concurrency
   list `16 32 64`.
4. Both benchmark scripts emit a `.meta.json` sidecar containing
   `commit_sha` and `server_args` from `/get_server_info`.
5. `PYTHONPATH=python pytest test/registered -q` ≥ 207 passed, 0 failed
   (or equivalent including the new test class).
6. Scripts are still executable (`bash -n` passes — syntactic check).

## Blocking Issues

None.

## Queued (out of scope for Round 23)

- AC-12 scaffold replacement (real NIAH 4K/16K/64K + MMLU 5-shot
  paired-server harness) — separate round, the largest remaining
  code-tier item.
- `benchmark_compare.py` 3-trial median + AC-11 directional gate
  (DS TPS within 5% of DSA; P99 TTFT ≤ 1.10× DSA) — separate round.
- Shallow prefix-match regression coverage (Codex's Round 22
  queued observation — tests replicate slicing instead of calling
  the actual gate via mocks).
- Stale DS bind/runtime comments + token-label lifetime docs.
- Hardware-gated: `task-ac1-hwtest`, `task-ac4-hwrun`,
  `task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
  `task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
  `task-ac11-compare`, `task-ac12-quality`.
