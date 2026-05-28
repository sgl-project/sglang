# Round 21 Contract

## Mainline Objective

Land the AC-8 quality-smoke harness as a runnable file
(`test/manual/test_dsv32_quality_smoke.py`) per the plan's §9.4 contract,
**and** fix the precondition Codex flagged: the page-vs-token unit
mix-up in `_publish_ds_request_summary` (the smoke + downstream
operator metrics read from that field).

Codex Round-20-review framing: AC-6 coding is closed. The next
plan-derived bottleneck on the AC-8 path is the missing quality-smoke
harness file — `test/manual/test_dsv32_quality_smoke.py` does not exist
today. Without it, `task-ac8-quality` cannot run on hardware. The
observability unit mix-up was also explicitly listed as a precondition
to AC-8 server/quality smoke.

## Target ACs

- **AC-8** — the runnable AC-8 quality smoke harness exists, has the
  four assertion gates required by the plan (prefix-match ≥ 80%, mean
  ROUGE-L ≥ 0.85, NIAH-mini recall ≥ 4/5, no first-8-tokens divergence
  on any prompt), records the DSA server commit SHA, and skips cleanly
  when the env vars are not set.
- **AC-8 (observability)** — `_publish_ds_request_summary` reports
  `selected_tokens` against `total_tokens` (not `selected_pages` /
  `total_pages`). After the token-level rotation in AC-0, the selector
  emits token positions, not pages; the sparsity rate must use the
  correct denominator.

## Required Implementation

### Fix 1: Create AC-8 quality smoke harness

- New file: `test/manual/test_dsv32_quality_smoke.py`.
- Two env-var contracts (matching the existing AC-12 scaffold):
  - `DS_BASE_URL` — the DS HTTP server endpoint.
  - `DSA_BASE_URL` — the DSA reference HTTP server endpoint (run on the
    same binary, same restart session, generated immediately before
    the DS smoke per plan §9.4).
- Skip cleanly when env vars are unset (so `pytest` against the file
  doesn't error in CI).
- 20 deterministic prompts as a class attribute (mix: short conversational,
  factual recall, NIAH-mini needle templates). All at `temperature=0`,
  fixed `max_new_tokens` (256), seeded.
- For each prompt: query both endpoints, capture text + first-8 tokens.
- Compute four assertions per plan §9.4:
  1. `prefix_match_rate >= 0.80` — number of prompts whose generated
     output starts with the DSA reference's first 32 chars / total prompts.
  2. `mean_rouge_l >= 0.85` — mean ROUGE-L F-measure over the 20 prompts.
  3. `niah_mini_recall >= 4/5` — needle-in-haystack mini test on 5
     embedded-needle prompts; DS must locate the needle in ≥ 4.
  4. `first_8_tokens_divergence == 0` — no prompt where the first 8
     tokens of DS and DSA disagree completely.
- Record commit SHAs (DS and DSA) into the result artifact written
  under `development/results/dsv32_quality_smoke_<timestamp>.json`.
- Pure-Python ROUGE-L fallback when `rouge_score` is not importable
  (so the test file is importable in CI without an extra dep).

### Fix 2: token-vs-page units in `_publish_ds_request_summary`

- `deepseek_v2.py:1972-2031`: after the AC-0 rotation, `valid_lengths`
  carries the number of selected **tokens**, not pages.
- Rename the published fields:
  - `selected_pages` → `selected_tokens`.
  - `total_pages` (Python local) → `total_tokens`, computed as
    `int(sl_cpu[b])` (no page-size division — the AC-0 architecture
    uses token-level signatures with `page_size` still 64 but the
    selector emits token counts).
  - `sparsity_rate = 1.0 - selected_tokens / total_tokens`.
  - The error-path record at `deepseek_v2.py:2323-2330` updates from
    `selected_pages` → `selected_tokens` too.
- Update `metrics.meta_info_for_request` + the
  `DoubleSparsityRequestStats` field name accordingly.
- Update any unit test that asserts on the old field names.

## Tests

- Existing 201 tests must still pass.
- The new `test_dsv32_quality_smoke.py` must import without error and
  skip cleanly when env vars are unset (verified by running pytest
  against the file with no env vars set).
- Add a unit test asserting `_publish_ds_request_summary` writes
  `selected_tokens` (not `selected_pages`) and computes
  `sparsity_rate` against token totals.

## Success Criteria

1. `test/manual/test_dsv32_quality_smoke.py` exists and is importable.
2. Running `pytest test/manual/test_dsv32_quality_smoke.py` without
   env vars set: all tests skip cleanly with the env-var message.
3. `_publish_ds_request_summary` publishes `selected_tokens` against
   `total_tokens`; unit test asserts this.
4. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
   ≥ 202 passed, 0 failed.

## Blocking Issues

None.

## Queued (out of scope for Round 21)

- AC-12 full quality gate (`test_double_sparsity_v32.py`) — separate round.
- `serve_double_sparsity.sh` + `serve_native_nsa.sh` Option B flag
  alignment — separate round.
- `benchmark.sh` + `benchmark_baseline.sh` conc 16/32/64 sweep.
- `benchmark_compare.py` 3-trial median + AC-11 gate enforcement.
- Stale DS bind/runtime comments mentioning `req_to_token_pool.size`.
- Token-label lifetime docs (overwrite-before-read vs
  invalidate-before-selection).
- Hardware-gated: `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`,
  `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality` (the
  *hardware execution* of the harness landed in this round),
  `task-ac12-quality`.
