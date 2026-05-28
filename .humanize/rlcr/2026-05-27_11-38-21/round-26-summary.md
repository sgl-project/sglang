# Round 26 Summary

## Work Completed

Codex Round 25 review caught two blocking bugs in my AC-12 harness:

1. **MMLU broken** — passed root `DS_BASE_URL` to `run_eval_once`, but
   sglang's OpenAI-compatible endpoints live under `/v1`; worse, the
   reused `simple_eval_mmlu.MMLUEval` is 0-shot, not the plan-required
   5-shot.
2. **Server-side fault-injection plumbing missing** — Round 25
   deferred it; Codex rejected that as unjustified.

Plus one queued artifact-writer omission Codex flagged: NIAH JSON was
missing hit counts.

Round 26 closes all three plus all of Codex's requested helper
regressions.

### Fix 1 — Real MMLU 5-shot

`test/manual/test_double_sparsity_v32.py`:

- `_openai_base_url(url)` helper that normalizes any operator-supplied
  URL to `.../v1` (idempotent; case-insensitive suffix check).
  Reserved for future OpenAI-compatible callers; the 5-shot path uses
  `/generate` directly.
- New helpers `_format_mmlu_subject`, `_format_mmlu_example`,
  `_make_mmlu_5shot_prompt`, `_parse_mmlu_letter` — mirror the
  formatting from `benchmark/mmlu/bench_sglang.py:25-52`.
- `test_mmlu_5shot` body fully rewritten:
  - Loads MMLU CSVs from `benchmark/mmlu/data/` (operator override
    `AC12_MMLU_DATA_DIR`); skips cleanly if the data dir is absent
    (`benchmark/mmlu/bench_sglang.py` auto-fetches on first run).
  - Builds a single shuffled-deterministic list of test questions
    across all subjects (override `AC12_MMLU_SUBJECTS=foo,bar` to
    narrow); honors `AC12_MMLU_NUM_EXAMPLES` cap (default 200).
  - For each question: 5-shot prompt = header + dev[:5] (answered)
    + test question (bare `Answer:`). Queries `/generate` with
    `max_new_tokens=4`, `temperature=0`. First A-D character in the
    response is the prediction.
  - Per-subject + aggregate scoring persisted to
    `development/results/ac12_mmlu_5shot_<ts>.json`.
  - Gate: `dsa_score_pct - ds_score_pct ≤ 1.0`.

### Fix 2 — Server-side fault-injection gates

`python/sglang/srt/models/deepseek_v2.py` (after `slice_per_rank` at
line ~1893):

- `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK=1` replaces `channel_selection`
  with a deterministically-per-layer random selection from
  `[0, head_dim)`. Uses `torch.Generator().manual_seed(self.layer_id)`
  so the corruption is reproducible. Sampling without replacement
  when `label_dim ≤ head_dim` (the production case for V3.2
  `head_dim=128`, `label_dim=16`); falls back to `torch.randint` for
  the edge case. Preserves shape / dtype / device + `channel_weights`.
  Logs a clear WARNING per layer.

`python/sglang/srt/layers/attention/dsa_backend.py`:

- `NativeSparseAttnBackend.__init__` caches
  `self._ds_fault_zero_sig = os.getenv("SGLANG_DS_FAULT_INJECT_ZERO_SIG")
  == "1"`. Logs a WARNING once per process when set.
- `_write_token_labels` after the `token_label_write` call:
  ```python
  if getattr(self, "_ds_fault_zero_sig", False):
      self._ds_token_label_table.signatures[layer_id, cache_loc] = 0
  ```
  Uses `getattr` so existing `object.__new__(backend)` unit fixtures
  that don't run `__init__` keep working (6 R1 tests would otherwise
  regress).
- `written[layer_id, cache_loc] = True` stays — the selector sees
  intentionally bad labels, not absent slots, which is what the
  zero-signature negative gate exercises.

### Fix 3 — NIAH artifact JSON carries hit counts

`_niah_assert` + the two sensitivity recorders now include the
underlying `dsa_hits` / `ds_hits` (and `ds_corrupt_hits` /
`ds_zero_hits` for sensitivity variants) — the audit can replay the
gate without re-running the H200 servers.

### Fix 4 — Registered helper regressions (10 new)

`test/registered/unit/manual/test_ac12_helpers.py`:

- `_openai_base_url` × 5: appends `/v1`; strips trailing slash;
  idempotent on `/v1`; idempotent on `/v1/`; case-insensitive
  suffix preserves original case.
- MMLU formatting × 2: `_format_mmlu_example` with answer ends
  `"Answer: X\n\n"`; without answer ends bare `"Answer:"`.
- 5-shot prompt × 2: contains exactly 5 in-context `"Answer: X"`
  endings from dev set; requires ≥ 5 dev rows (`ValueError` otherwise).
- `_parse_mmlu_letter` × 1: finds first A-D char; None for letterless
  responses; first wins on `"AB"`.

### Fix 5 — Registered fault-injection regressions (4 new)

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`
new class `TestAC12FaultInjection`:

- `test_zero_sig_gate_default_off_keeps_signatures` — env unset →
  signatures are populated normally (sanity check that the change
  didn't introduce an unconditional zero).
- `test_zero_sig_gate_on_zeroes_just_written_row_keeps_written_true`
  — env on → `signatures[layer_id, cache_loc] == 0` AND
  `written == True`.
- `test_corrupt_mask_gate_random_selection_shape_dtype_range` —
  the algorithm preserves shape / dtype / device, all values in
  `[0, head_dim)`, and differs from the calibrated baseline.
- `test_corrupt_mask_gate_deterministic_per_seed` — same seed →
  same corruption (audit reproducibility); different seeds differ.

### Fix 6 — Docstring fix

`test_double_sparsity_v32.py` usage section now directs operators at
the pytest file-path form. The Round 25
`python -m unittest test.manual.test_double_sparsity_v32` form
collides with Python's stdlib `test` package and fails to import
(Codex Round 25 review queued issue).

## Files Changed

- `test/manual/test_double_sparsity_v32.py`: full MMLU rewrite + hit
  counts in artifacts + docstring fix.
- `python/sglang/srt/models/deepseek_v2.py`: `os` import + corrupt-mask
  env gate after `slice_per_rank`.
- `python/sglang/srt/layers/attention/dsa_backend.py`: zero-sig env
  gate cached in `__init__`; honored in `_write_token_labels`.
- `test/registered/unit/manual/test_ac12_helpers.py`: +10 helper
  regressions (URL norm + MMLU 5-shot + letter parse).
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +4 `TestAC12FaultInjection` regressions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
254 passed, 0 failed (was 240 before this round)

env -u DS_BASE_URL -u DSA_BASE_URL -u DS_CORRUPT_MASK_URL -u DS_ZERO_SIG_URL \
    python -m pytest test/manual/test_double_sparsity_v32.py -v
6 skipped — clean skip when env vars unset
```

Targeted runs:
```
pytest -v test/registered/unit/manual/test_ac12_helpers.py             # 21 passed (was 11)
pytest -v -k TestAC12FaultInjection                                    # 4 passed
```

Commit: `2ad8b9ee3` — [AC-12] Real MMLU 5-shot + fault-injection gates
+ harness fixes.

## Remaining Items

Code-tier items queued for future rounds:
- `benchmark_compare.py` AC-11 directional gate (3-trial median, DS
  TPS ≥ 95% of DSA, P99 TTFT ≤ 1.10× DSA).
- Shallow AC-8 prefix-match regression coverage.
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality` (harness + fault-injection
both ready; hardware execution pending).

## Push-to-remote Status

Branch is 27 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Both Round 26 fixes were straightforward — the MMLU bug was a
"I reused the wrong eval class" mistake (`simple_eval_mmlu.MMLUEval`
is documented as 0-shot in its own docstring), and the
fault-injection gates were a documented deferral now landed. Neither
generalizes into a useful BitLesson; the existing
`BL-20260527-importlib-dataclass-sys-modules` lesson already covered
the helper-test loader pattern I reused this round.
