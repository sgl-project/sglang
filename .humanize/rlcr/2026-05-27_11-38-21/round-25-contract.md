# Round 25 Contract

## Mainline Objective

Replace the skip-only AC-12 scaffold at
`test/manual/test_double_sparsity_v32.py` with a real paired-server
quality-gate harness covering NIAH @ 4K / 16K / 64K and MMLU 5-shot,
with the two negative sensitivity checks also runnable when the
fault-injected servers are published. Add helper-level CI tests so
the harness's deterministic prompt generation, ROUGE/recall scoring,
and gate logic are locked in before the H200 run.

Codex Round 23/24 reviews and the full goal alignment in Round 24
both flagged AC-12 as the HARD loop-closure gate that no other code
or hardware milestone substitutes for. Quoting plan §10:
"Loop does not close without AC-12 passing."

## Target ACs

- **AC-12** — Replace `test/manual/test_double_sparsity_v32.py` with a
  harness that, when paired DS/DSA servers are running, enforces:
  - `|niah_recall_DS - niah_recall_DSA| <= 5 pp` at each of 4K, 16K, 64K.
  - `|mmlu_score_DS - mmlu_score_DSA| <= 1.0 pp` on MMLU 5-shot.
  - Negative: corrupt-mask DS drops NIAH @ 64K by > 20 pp below DSA baseline.
  - Negative: zero-signature DS drops NIAH @ 16K by > 30 pp below baseline.
  Skips cleanly when env vars are unset (so CI/loop import does not error).

## Required Implementation

### Fix 1: AC-12 NIAH harness

`test/manual/test_double_sparsity_v32.py` — rewrite from scratch:

- Env-var contract (matches the existing Round 21 quality smoke):
  - `DS_BASE_URL` — DS server.
  - `DSA_BASE_URL` — DSA reference server.
  - Optional `DS_CORRUPT_MASK_URL` — DS server booted with a corrupted
    channel mask (sensitivity test 1).
  - Optional `DS_ZERO_SIG_URL` — DS server booted with the
    `SGLANG_DS_FAULT_INJECT_ZERO_SIG=1` env (sensitivity test 2).
  - `AC12_NIAH_NUM_PROMPTS` (default 20): trials per NIAH length.
  - `AC12_NIAH_MAX_NEW_TOKENS` (default 64): generation cap.

- For each NIAH length L in {4096, 16384, 65536}:
  - Generate 20 deterministic prompts via a seeded RNG (one
    seed-per-length). Each prompt is `L`-token-ish (whitespace-tokenized
    Lorem-ipsum-style filler from a reproducible word list).
  - Plant a unique 8-character sentinel needle (e.g.
    `NEEDLE-####`) at a deterministic position deep inside the prompt
    (between 35% and 65% of the prompt) — far from both the prompt's
    start and the local window the model attends to without sparse
    selection.
  - Query DSA, then DS. Recall = needle string is a substring of the
    generated response.
  - `recall_pct = (hits / num_prompts) * 100`.
  - Assert `recall_pct_DSA - recall_pct_DS <= 5.0`.

### Fix 2: AC-12 MMLU 5-shot

- Reuse `sglang.test.run_eval` with `eval_name="mmlu"` if it exists in
  the repo (detect at import time; otherwise skip the MMLU test with a
  clear message naming what's missing).
- Run once against DSA, once against DS; persist aggregate scores plus
  per-subject scores into the same `development/results/` artifact dir
  as the quality smoke.
- Assert `mmlu_score_DSA - mmlu_score_DS <= 1.0`.

### Fix 3: Negative sensitivity tests

- `test_niah_64k_sensitivity_corrupt_mask` (requires `DS_CORRUPT_MASK_URL`):
  Same 64K NIAH fixture as the positive test, but the DS-corrupt server
  runs with a randomly-permuted `channel_mask.channel_selection`.
  Asserts `recall_pct_DSA - recall_pct_DS_CORRUPT > 20.0` — corrupt
  mask must drop recall significantly. (If `DS_CORRUPT_MASK_URL` is
  unset, the test skips with a clear message.)

- `test_niah_16k_sensitivity_zero_signatures` (requires `DS_ZERO_SIG_URL`):
  Same 16K NIAH fixture; DS-zero-sig server fault-injects an all-zero
  `token_label_table.signatures`. Asserts
  `recall_pct_DSA - recall_pct_DS_ZERO > 30.0`. (Skip if env unset.)

### Fix 4: Helper-level regression tests

`test/registered/unit/manual/test_ac12_helpers.py` (or extend the
existing development-tier helper test file): exercise the prompt
generation + scoring functions in CI without env vars:

- `_make_niah_prompt(length_tokens=4096, seed=42)` produces a string
  whose whitespace token count is approximately `length_tokens` (±5%)
  and contains the needle exactly once at a deterministic depth.
- `_niah_recall(prompts, responses_with_needles, responses_without)` —
  helper that takes the lists and returns recall percent.
- `_mmlu_delta(score_a, score_b)` returns `abs(a - b)`.
- All-needles-present case → 100% recall.
- No-needles-present case → 0% recall.
- Same seed → identical prompt strings.

### Fix 5: Sensitivity-test fault-injection plumbing (server-side)

This is a Round 25 stretch — it's required for the negative tests to
actually run on hardware, but the negative tests skip cleanly without
it. Implementation:

- Add an env-var gate in `dsa_backend.py` /
  `token_label_table.py` (whichever is closer to the population path)
  that reads `SGLANG_DS_FAULT_INJECT_ZERO_SIG` and, when set, zeroes
  the signature table immediately after every write. Log a clear
  warning on startup so an operator can't enable it silently.
- Same pattern for `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK`: after
  bind, randomly permute `channel_mask.channel_selection` per layer.

These two env-gates make the sensitivity tests executable on a
single H200 — the operator boots a third sglang server with the gate
set and exposes it at `DS_CORRUPT_MASK_URL` / `DS_ZERO_SIG_URL`.

(If this stretch portion proves too large within Round 25, it can be
deferred to a follow-up round; the harness side of Fix 3 must still
land with the env-gated skip path.)

## Tests

- Existing 229 tests must still pass.
- New helper-regression tests (~5) added.
- `pytest test/manual/test_double_sparsity_v32.py -v` without env vars
  must still skip cleanly (no import errors, no test failures).
- Expect ≥ 234 passed total.

## Success Criteria

1. `test/manual/test_double_sparsity_v32.py` is no longer
   `self.skipTest("...")`-only; each test has real prompt generation +
   scoring + assertions, and skips ONLY when its env vars are unset.
2. Helper functions (`_make_niah_prompt`, `_niah_recall`, etc.) are
   import-clean and have CI regression tests.
3. Manual smoke: `python -c "from test.manual import test_double_sparsity_v32"`
   succeeds outside the loop (no syntax/import errors).
4. `bash -n` and full unit-suite both still green.
5. `PYTHONPATH=python pytest test/registered -q` ≥ 234 passed.

## Blocking Issues

None — sensitivity-test fault injection is gated to "stretch" and
defers cleanly if too large.

## Queued (out of scope for Round 25)

- `benchmark_compare.py` AC-11 directional gate (3-trial median, DS
  TPS within 5% of DSA, P99 TTFT ≤ 1.10× DSA) — separate round.
- AC-10 radix fixture / FP8 cold-warm stability check / flag flip —
  hardware-gated.
- Shallow AC-8 prefix-match regression coverage.
- Stale `deepseek_v2.py` slot-authority comments + `token_label_table.py`
  lifetime docs.
- Hardware-gated tasks: `task-ac1-hwtest`, `task-ac4-hwrun`,
  `task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
  `task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
  `task-ac11-compare`, **`task-ac12-quality` (hardware execution)** —
  the harness lands in Round 25; the H200 run remains pending.
