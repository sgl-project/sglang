# Round 21 Summary

## Work Completed

Codex Round 20 review verified the AC-6 production-object fix and
opened the next plan-derived bottleneck: the AC-8 quality smoke harness
file (`test/manual/test_dsv32_quality_smoke.py`) did not exist, and
the observability publication still reported `selected_pages` against
`total_pages = (seq_len + page_size - 1) // page_size`, which is the
wrong unit after the AC-0 token-level rotation.

Round 21 lands both:

### Fix 1 — AC-8 quality smoke harness (new file)

`test/manual/test_dsv32_quality_smoke.py`:

- 20 deterministic prompts at `temperature=0`, `max_new_tokens=256`.
- 5 NIAH-mini needle-in-haystack prompts with unique sentinel needles
  (ZEBRA-7, MARLIN-42, ORCHID-99, GLACIER-13, PHARAOH-88).
- Paired DSA/DS HTTP queries per plan §9.4 (DSA reference first, same
  session immediately before DS).
- Four assertion gates:
  1. `prefix_match_rate >= 0.80` — DS first 32 chars match DSA's.
  2. `mean_rouge_l >= 0.85` — pure-Python LCS-based ROUGE-L F-measure
     (no `rouge_score` dependency; harness imports in any environment).
  3. `niah_mini_recall >= 4/5` — needle string present in DS response.
  4. `first_8_tokens_divergences == 0` — no prompt where DS and DSA's
     first 8 tokens fail to share any common token.
- Best-effort commit SHA capture from `/get_server_info` for both
  servers; written into `development/results/dsv32_quality_smoke_<ts>.json`.
- Cleanly skips when `DS_BASE_URL` or `DSA_BASE_URL` env vars are unset
  — `pytest test/manual/test_dsv32_quality_smoke.py` without env vars
  yields `1 skipped`.

### Fix 2 — token-vs-page units in `_publish_ds_request_summary`

After the AC-0 token-level rotation the selector emits TOKEN counts in
`valid_lengths`, but the observability layer still labeled the field
`selected_pages` and divided by `page_size` for the sparsity denominator.
That yielded `sparsity_rate = 1.0 - 30 / 2 = -14` when 30 tokens were
selected from a 100-token, page-64 request — wrong unit, wrong sign.

Renames (consistent across module + metrics):

- `DoubleSparsityRequestStats.selected_pages` → `selected_tokens`.
- `meta_info["selected_pages"]` → `meta_info["selected_tokens"]`.
- `record_selection(selected_pages, total_valid_pages)` →
  `record_selection(selected_tokens, total_valid_tokens)`.
- Prometheus metrics
  `sglang_double_sparsity_selected_pages_{sum,count}` →
  `..._selected_tokens_{sum,count}`. Pre-MVP rename — no external
  dashboard consumers depend on the old names yet.
- `_publish_ds_request_summary`: dropped the `page_size` division;
  `total_tokens = max(1, int(sl_cpu[b]))`; sparsity_rate computed
  against tokens. Error-path record (`error_class != ok` branch) also
  publishes `selected_tokens`.

### Regression unit test

`TestR5Coverage::test_publish_ds_request_summary_uses_token_denominator`:
constructs a bs=2 forward_batch (seq_lens = [100, 256], valid_lengths
= [30, 5]), drives `_publish_ds_request_summary` directly, asserts:
- `selected_pages` key is gone, `selected_tokens` is present.
- Row 0 sparsity_rate == 1 - 30/100; row 1 == 1 - 5/256.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/metrics.py`:
  field/param/metric renames; updated module docstring.
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py`:
  updated `record_selection(...)` call site to new kw names.
- `python/sglang/srt/models/deepseek_v2.py::_publish_ds_request_summary`:
  token-denominator math; renamed published field; error-path record
  rename.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  renamed references in metric-counter assertion tests + meta_info
  shape tests + customized_info tests + test_select_topk_indices error
  fixtures; added the new regression test.
- `test/manual/test_dsv32_quality_smoke.py`: NEW file with the AC-8
  quality smoke harness.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
202 passed, 0 failed (was 201 before this round)

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_dsv32_quality_smoke.py -v
test_quality_smoke SKIPPED — clean skip when env vars are unset
```

Targeted:
```
pytest -v -k "test_publish_ds_request_summary_uses_token_denominator"   # 1 passed
```

Helper sanity:
```
_rouge_l_f(s, s)                = 1.000
_rouge_l_f(s, different)        = 0.167  (small, expected)
_first_n_tokens_match(...)      = True/False as expected
SMOKE_PROMPTS = 20, NIAH_MINI_PROMPTS = 5
```

Commit: `a586f814a` — [AC-8] Quality smoke harness + token-denominator
observability fix.

## Remaining Items

Mainline AC items still requiring hardware execution:
- `task-ac8-server` + `task-ac8-quality` (harness exists; needs paired DS+DSA H200 servers + run).
- `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`.
- `task-ac12-quality` (separate file `test_double_sparsity_v32.py` still skip-only — separate round).

Code-tier items still queued for future rounds (not Round 21 scope):
- Replace `test_double_sparsity_v32.py` skip scaffolds with real
  NIAH 4K/16K/64K + MMLU 5-shot logic.
- Update `serve_double_sparsity.sh` + `serve_native_nsa.sh` to encode
  the locked Option B flags.
- `benchmark.sh` + `benchmark_baseline.sh` conc 16/32/64 sweep.
- `benchmark_compare.py` 3-trial median + AC-11 directional gate.
- Stale DS bind/runtime comments + token-label lifetime docs.

## Push-to-remote Status

Branch is 22 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`; commits remain local on `dev/double-sparsity-standalone`.
Per-round pushing requires re-launching with `--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The page-vs-token unit mix-up was the kind of bug that comes from
incomplete refactors after an architecture rotation — but the BitLesson
file already covers reshape/slice ordering for MLA outputs
(`BL-20260527-reshape-before-slice-mla`) which is the closest general
principle ("always re-check derived names + per-head shapes after a
rotation"). A standalone "rename observability fields after rotation"
lesson would be too narrow to be useful. No new entry warranted.
