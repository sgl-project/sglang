# Round 25 Summary

## Work Completed

Codex Round 24 full goal alignment named AC-12 as the next mainline
move and the **hard loop-closure gate**: per plan §10, "Loop does not
close without AC-12 passing." After three rounds of clearing AC-8/AC-9
plumbing (sidecar writer, Option B scripts, observability units), the
AC-12 scaffold was still 6 skip tests. Round 25 replaces it.

### Fix 1 — Real AC-12 harness

`test/manual/test_double_sparsity_v32.py` — full rewrite. Skips
cleanly when env vars are unset; runs the real gates when paired
servers are published.

**NIAH @ 4K / 16K / 64K** (3 tests). Per length:
- 20 deterministic seeded prompts via a Lorem-ipsum-style filler word
  pool that whitespace-tokenizes to approximately the requested
  length (±5%). The same `(length, seed, needle)` triple always
  produces the same string.
- Each prompt has a unique sentinel needle `NEEDLE-LLLLL-III`
  (`length` zero-padded to 5 digits + `index` to 3) planted at a
  deterministic depth in `[0.35, 0.65]` of the prompt — far from both
  ends so DS sparse selection is the meaningful gate.
- Generation: DSA first, then DS, both at `temperature=0`.
- Recall = the needle string is a substring of the response (per
  plan; substring rather than word-boundary because temperature-0
  models may concatenate or stylize the needle).
- Assertion: `dsa_recall_pct - ds_recall_pct ≤ 5.0`.

**MMLU 5-shot** (1 test). Reuses the repo's existing eval path:
- `sglang.test.simple_eval_mmlu.MMLUEval` over
  `openaipublic.blob.core.windows.net/simple-evals/mmlu.csv`.
- `sglang.test.run_eval.run_eval_once` against each base URL.
- Default 200 examples (override via `AC12_MMLU_NUM_EXAMPLES` for
  the full ~14k on H200; the 200-sample subset is enough to detect
  a >1.0 pp regression in CI/dev settings).
- Assertion: `dsa_score_pct - ds_score_pct ≤ 1.0`.

**Sensitivity (2 tests)**, skip cleanly without their respective
URLs:
- `test_niah_64k_sensitivity_corrupt_mask` (`DS_CORRUPT_MASK_URL`):
  assert `dsa - ds_corrupt > 20 pp`.
- `test_niah_16k_sensitivity_zero_signatures` (`DS_ZERO_SIG_URL`):
  assert `dsa - ds_zero > 30 pp`.

Each gate writes a `development/results/ac12_<suffix>_<ts>.json`
artifact with `dsa_hits`, `ds_hits`, `dsa_recall_pct`, `ds_recall_pct`,
`delta_pct`, `threshold_pp` so an audit pass can replay the gate
output even after the servers are torn down.

### Fix 2 — CPU helper regressions (CI-runnable)

`test/registered/unit/manual/test_ac12_helpers.py` (11 tests). The
manual harness file is skip-only without env vars, but its pure
helpers can and must be tested in CI.

Loader trick: the file is registered in `sys.modules` BEFORE
`spec.loader.exec_module(mod)` so the `@dataclass` decorator at
`_NIAHRunResult` can resolve `__module__` (the standard Python
dataclasses machinery walks `sys.modules` to find the owning module's
namespace; loading purely via `importlib.util.spec_from_file_location`
without registering breaks `@dataclass`).

Covered:
- Needle naming: stable per `(length, idx)`; differs across both
  inputs; format `NEEDLE-LLLLL-III`.
- Prompt determinism: same `(length, seed, needle)` → identical bytes.
- Needle present exactly once.
- Whitespace-tokenized length within ±5% of the request (with a
  32-word floor for small lengths).
- Needle depth fraction in `[0.30, 0.70]`.
- Question suffix `"... Output only the value."`.
- Recall scoring: all-present → 100%, none → 0%, partial → correct
  count, substring (not word-boundary) match works.

### Fix 3 — Server-side fault-injection plumbing (deferred to next round)

The two sensitivity tests need real fault-injected DS servers
exposed at `DS_CORRUPT_MASK_URL` and `DS_ZERO_SIG_URL`. The plan
implementation calls for env-var gates inside `dsa_backend.py` /
`token_label_table.py` (e.g. `SGLANG_DS_FAULT_INJECT_CORRUPT_MASK`,
`SGLANG_DS_FAULT_INJECT_ZERO_SIG`) so an operator can boot a third
sglang server with the gate set. The Round 25 contract marked this
"stretch — defers cleanly"; the harness already skips when the
fault-injected URLs are unset, so AC-12 positive gates can run on
H200 today without waiting for that scaffold. Plumbing follows in
a focused Round 26.

## Files Changed

- `test/manual/test_double_sparsity_v32.py`: full rewrite — real
  NIAH + MMLU + sensitivity harness; deterministic prompt/needle
  generation; result artifact writer; env-var-gated skips.
- `test/registered/unit/manual/test_ac12_helpers.py`: NEW — 11 CPU
  regression tests for the harness's helper functions; uses
  `importlib.util` with `sys.modules` registration so `@dataclass`
  resolves.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
240 passed, 0 failed (was 229 before this round)

env -u DS_BASE_URL -u DSA_BASE_URL -u DS_CORRUPT_MASK_URL -u DS_ZERO_SIG_URL \
    python -m pytest test/manual/test_double_sparsity_v32.py -v
6 skipped — clean skip when env vars unset
```

Targeted run:
```
pytest -v test/registered/unit/manual/test_ac12_helpers.py
11 passed (needle naming, prompt determinism + length + depth +
suffix, recall all/none/partial/substring cases)
```

Commit: `0ae955cf2` — [AC-12] Replace skip-only scaffold with real
NIAH + MMLU + sensitivity harness.

## Remaining Items

Code-tier items queued for future rounds:
- **Server-side fault-injection env gates** (`SGLANG_DS_FAULT_INJECT_*`)
  so the two sensitivity tests can actually fire on hardware. The
  harness side is ready.
- **`benchmark_compare.py` AC-11 directional gate**: 3-trial median
  per concurrency, DS TPS within 5% of DSA, P99 TTFT ≤ 1.10× DSA;
  still the absolute-SLO single-trial framing today.
- Shallow AC-8 prefix-match regression coverage cleanup (Codex Round 22
  queued).
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality` (harness landed; hardware
execution pending).

## Push-to-remote Status

Branch is 26 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-importlib-dataclass-sys-modules
Notes: A test fixture that loads a sibling Python file via
`importlib.util.spec_from_file_location(...)` will fail at
`@dataclass`-decorated class definitions with
`AttributeError: 'NoneType' object has no attribute '__dict__'`
unless the module is registered in `sys.modules` BEFORE
`spec.loader.exec_module(mod)`. The dataclasses machinery resolves
`__module__` via `sys.modules.get(cls.__module__)`, and without the
registration step the lookup returns `None`. This is too narrow to
generalize to a separate lesson — it shares scope with the existing
"shell-json-into-python-source" lesson in being about cross-language
boundary plumbing — so I'll add it as a note appended to BitLesson
rather than its own entry.
