# Round 26 Contract

## Mainline Objective

Close the AC-12 harness gaps Codex flagged in the Round 25 review so
the gate can actually fire on H200. Two blocking bugs in the Round 25
harness + one missing server-side feature:

1. **MMLU is broken** — Round 25 passes the root `DS_BASE_URL` to
   `run_eval_once`, but OpenAI-compatible endpoints live under `/v1`
   (sglang's own `run_eval.py` CLI appends it). Worse, the MMLU eval
   class (`simple_eval_mmlu.MMLUEval`) is **0-shot**, not the
   plan-required **5-shot**.
2. **Server-side fault-injection plumbing missing** — Round 25
   deferred it. Without `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK` and
   `SGLANG_DS_FAULT_INJECT_ZERO_SIG`, the two sensitivity tests can
   only skip on hardware, which violates the AC-12 negative gates.
3. **Artifact-writer omits hit counts** — Round 25 summary said the
   NIAH JSON records `dsa_hits` / `ds_hits`, but the writer only
   records the recall percents. Small fix.

## Target ACs

- **AC-12** — the harness's MMLU path is real 5-shot against the
  correct OpenAI base URL; both sensitivity tests can run when paired
  with fault-injected DS servers; positive NIAH artifacts include
  hit counts for downstream audit.

## Required Implementation

### Fix 1: MMLU 5-shot + /v1 URL normalization

`test/manual/test_double_sparsity_v32.py`:

- Add `_openai_base_url(base_url: str) -> str` that returns
  `base_url.rstrip('/') + '/v1'` unless the URL already ends with
  `/v1` (case-insensitive). This is used ONLY for OpenAI-compatible
  calls. `DS_BASE_URL` / `DSA_BASE_URL` stay root URLs for
  `/generate` (the NIAH path).
- Replace the `test_mmlu_5shot` body with real 5-shot prompt
  construction using the same data + format as
  `benchmark/mmlu/bench_sglang.py`:
  - Download MMLU (the bench script's `download_data` helper or a
    minimal in-harness version) into a cached dir.
  - For each subject (overridable via `AC12_MMLU_SUBJECTS` default
    `all`), take the first 5 dev rows as the 5-shot prefix; for each
    test row, build the prompt as `gen_prompt(dev, subject, k=5)` +
    `format_example(test, idx, include_answer=False)`.
  - Query each model with `max_new_tokens=1`, `temperature=0`; the
    first stripped `A`-`D` token in the response is the prediction.
  - Compare against the test row's gold label; aggregate accuracy
    over all examples (default cap `AC12_MMLU_NUM_EXAMPLES=200` for
    quick H200 run, override to full for the audit).
  - `dsa_score - ds_score ≤ 1.0` is the gate.
- Persist per-subject + aggregate scores into the artifact JSON.

### Fix 2: Server-side fault-injection gates

**Corrupt-mask gate** in `deepseek_v2.py`:

- After `local_mask = slice_per_rank(...)` (around line 1898), gate
  on `os.getenv("SGLANG_DS_FAULT_INJECT_CORRUPT_MASK") == "1"`. When
  set, construct a new `ChannelMask` with deterministically randomly
  permuted `channel_selection`:
  - Use `numpy.random.RandomState(layer_id)` (or a fixed module-level
    seed) so the corruption is reproducible.
  - Each `[L, H, label_dim]` row is a fresh random selection from
    `[0, head_dim)` (replace with `rng.choice(head_dim, label_dim,
    replace=False)` if `label_dim <= head_dim`, else with
    replacement).
  - Preserve dtype, device, and `channel_weights` (corrupt
    selection, not weights).
- Log a clear `WARNING` once per process at corrupt-mask init.

**Zero-signature gate** in `dsa_backend.py`:

- In `NativeSparseAttnBackend.__init__`, add `self._ds_fault_zero_sig
  = os.getenv("SGLANG_DS_FAULT_INJECT_ZERO_SIG") == "1"`. Log a
  warning once when True.
- In `_write_token_labels` (line ~1432), AFTER the `token_label_write`
  call, if `self._ds_fault_zero_sig`:
  - `signatures[layer_id, cache_loc] = 0` — zero the just-written
    row but keep `written[layer_id, cache_loc] = True` so the
    selector treats the slot as populated with intentionally bad
    labels (not absent).

### Fix 3: NIAH artifact includes hit counts

`test/manual/test_double_sparsity_v32.py::_niah_assert` + sensitivity
recorders: add `dsa_hits` and `ds_hits` (or `ds_corrupt_hits` /
`ds_zero_hits`) to the artifact dict so the audit can replay the
gate. The fields already exist on `_NIAHRunResult`; just include them
in `_record_artifact(...)`.

### Fix 4: Registered helper regressions

`test/registered/unit/manual/test_ac12_helpers.py`:

- `_openai_base_url` normalization tests:
  - `_openai_base_url("http://h:30000")` → `"http://h:30000/v1"`.
  - `_openai_base_url("http://h:30000/")` → `"http://h:30000/v1"`.
  - `_openai_base_url("http://h:30000/v1")` → `"http://h:30000/v1"`.
  - `_openai_base_url("http://h:30000/v1/")` → `"http://h:30000/v1"`.
  - `_openai_base_url("HTTPS://X/V1")` should also be idempotent
    (case-insensitive suffix check).
- 5-shot prompt structure tests (using a tiny in-memory `pd.DataFrame`
  stub or the helper's own renderer):
  - Output contains exactly 5 dev examples before the test question.
  - Each example ends with `"Answer: X\n\n"` for X in A-D.
  - The test question ends with `"Answer:"` (no trailing letter).

### Fix 5: Registered fault-injection regressions

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`:

- `test_corrupt_mask_gate_changes_selection_preserves_shape_and_range`:
  - With `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK=1`, the corrupted
    channel_selection has the same shape + dtype + device as the
    original; values are in `[0, head_dim)`; selection differs from
    the original.
  - With env unset (default), `slice_per_rank` output is untouched.
- `test_zero_sig_gate_zeroes_just_written_row_keeps_written_true`:
  - Build a minimal `TokenLabelTable` + `cache_loc` + `k_nope`;
    drive `_write_token_labels`; assert `signatures[layer_id,
    cache_loc] == 0` and `written[layer_id, cache_loc] == True`.
  - Negative: with env unset, the row holds nonzero signatures
    (existing behavior).

### Fix 6: Queued docs follow-up

Codex also flagged: `python -m unittest test.manual.test_double_sparsity_v32`
fails because the repo `test/` tree isn't a package. The harness
docstring should document the pytest file-path command form:

```
PYTHONPATH=python python -m pytest test/manual/test_double_sparsity_v32.py
```

Small docstring fix; non-blocking. Bundle into the harness file
since I'm already touching it.

## Tests

- Existing 240 tests must still pass.
- ~6 new registered helper regressions + 2 fault-injection regressions
  ≈ 8 new tests.
- Expect ≥ 248 passed total.

## Success Criteria

1. `_openai_base_url("http://h:30000")` returns `"http://h:30000/v1"`;
   variants pass CI tests.
2. MMLU 5-shot prompt construction produces a prompt with exactly 5
   answered dev examples + 1 unanswered test question; verified by CI.
3. `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK=1` is honored at `bind_runtime_data`;
   `SGLANG_DS_FAULT_INJECT_ZERO_SIG=1` is honored after every
   `token_label_write`; both verified by CI; both no-ops when env unset.
4. NIAH positive artifacts include `dsa_hits` and `ds_hits`.
5. `pytest test/registered -q` ≥ 248 passed.
6. Harness still skips cleanly without env vars.

## Blocking Issues

None.

## Queued (out of scope for Round 26)

- `benchmark_compare.py` 3-trial median + AC-11 directional gate.
- Shallow AC-8 prefix-match regression coverage cleanup.
- Stale DS bind/runtime comments + token-label lifetime docs.
- Hardware-gated execution of AC-12, AC-1, AC-1b, AC-4, AC-6, AC-8,
  AC-9, AC-10, AC-11.
