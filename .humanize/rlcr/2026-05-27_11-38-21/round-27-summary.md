# Round 27 Summary

## Work Completed

Codex Round 26 review caught two correctness bugs in the AC-12 MMLU
harness:

1. `_parse_mmlu_letter("Answer: B")` returned `"A"` (the A in
   "Answer") — Round 26 scanned for the first A-D character anywhere
   in the response. With `max_new_tokens=4` and chat templates that
   often emit prefixes like `"Answer: B"`, `"B."`, `"(C)"`, correct
   B/C/D answers were silently scored as A.
2. MMLU silently skipped when `benchmark/mmlu/data/{dev,test}` was
   absent. The repo checkout does not ship that data, so every AC-12
   MMLU run skipped before contacting either server — the hard
   loop-closure gate could pass by virtue of skipping.

Round 27 closes both with hard CI regressions.

### Fix 1 — Robust answer-token parser

`test/manual/test_double_sparsity_v32.py`:

- `_LEADING_LETTER_RE = r'^[\s\(\[\{<"\'`]*([A-Da-d])(?!\w)'`
  matches isolated letters possibly wrapped in opening punctuation,
  followed by a non-word boundary so the parser never matches the
  `A` inside `"Answer"` or `"Awful"`.
- `_MARKER_RE = r'(?i)(?:answer\s*[:\-]?|answer\s+is|the\s+answer\s+is
  |option|choice)\s*[\(\[\{<"\'`:.]*\s*([A-Da-d])(?!\w)'`
  finds the next isolated A-D letter after a recognized
  answer-introducer marker.
- `_parse_mmlu_letter(response)` strips whitespace, tries the
  leading-letter regex, falls through to the marker regex, returns
  `None` if neither matches. Conservative — narrative text without
  a marker returns `None` rather than guessing.

### Fix 2 — `_ensure_mmlu_data_dir` self-prep

`test/manual/test_double_sparsity_v32.py`:

- If `data_dir/dev` and `data_dir/test` both exist → return
  immediately (idempotent fast path).
- Otherwise:
  1. `urllib.request.urlretrieve` the Hendrycks `data.tar` into a
     temp dir (no `wget` dep).
  2. `tarfile.extractall(filter='data')` — Python 3.12's safe
     filter rejects symlinks, devices, absolute paths,
     path-traversal via `..`. We also pre-filter members to only
     those under the archive's `data/` prefix.
  3. Atomic `shutil.move` of `data/dev` and `data/test` into the
     target.
- Any failure → `RuntimeError` with a clear message naming the URL
  + data_dir + failing step (download / extract / missing subdir).

### Fix 3 — `test_mmlu_5shot` calls the helper, fails loudly

The Round 26 `skipTest("MMLU data not found...")` branch is gone.
When `DS_BASE_URL`/`DSA_BASE_URL` are set (the operator is running
the gate), any data-prep failure becomes `self.fail(...)` with a
clear remediation message naming `benchmark/mmlu/bench_sglang.py`
and the `AC12_MMLU_DATA_DIR` override. Per plan §10: "Loop does
not close without AC-12 passing." Silent skips defeated that.

### Fix 4 — Registered regressions (+17 new)

`test/registered/unit/manual/test_ac12_helpers.py` adds:

**Parser (13 cases):**
- `"Answer: B" → "B"` (the exact Codex Round 26 review case)
- `"b" → "B"` (lowercase fold)
- `"(C)" → "C"` (paren-wrapped)
- `"D." → "D"` (trailing punctuation)
- `"[A]" → "A"` (bracket-wrapped)
- `"answer is C" → "C"` (lowercase marker)
- `"The answer is D." → "D"` (prefixed marker with punct)
- `"option B" → "B"` (marker variant)
- `"Choice (A)" → "A"` (marker + paren)
- `"Awful question, no marker." → None` (conservative narrative)
- `"" → None`
- `"   \n  " → None` (whitespace-only)
- `" (B), final" → "B"` (leading punctuation chain)

**Data prep (4 cases):**
- Downloads + extracts via monkeypatched `urlretrieve` over a
  fabricated `data/dev/foo_dev.csv` + `data/test/foo_test.csv` tar.
- Idempotent — second call no-op; `urlretrieve` called exactly once.
- `RuntimeError("...download failed...")` on `IOError` from
  urlretrieve.
- `RuntimeError("...missing data/dev/...")` when archive lacks the
  expected subdirs.

## Files Changed

- `test/manual/test_double_sparsity_v32.py`:
  - Added `re`/`shutil`/`tarfile`/`tempfile` imports + `Tuple` from
    `typing`.
  - Replaced `_parse_mmlu_letter` body with regex-driven parser.
  - Added `_ensure_mmlu_data_dir` helper.
  - `test_mmlu_5shot` calls the helper; failure path is now
    `self.fail`, not `self.skipTest`.
- `test/registered/unit/manual/test_ac12_helpers.py`:
  - Added `os` import.
  - +17 new helper regressions (parser × 13 + data-prep × 4).

## Validation

```
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
38 passed, 0 failed (was 21 — Round 26)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
271 passed, 0 failed (was 254)

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_double_sparsity_v32.py -v
6 skipped — clean skip when env vars unset
```

Manual parser smoke (the Round 26 broken cases):
```
_parse_mmlu_letter("Answer: B")  → "B"   (was "A")
_parse_mmlu_letter("b")           → "B"   (was None)
_parse_mmlu_letter("(C)")         → "C"
_parse_mmlu_letter("D.")          → "D"
_parse_mmlu_letter("Awful but C is right") → None
```

Commit: `faa41438e` — [AC-12] Fix MMLU answer parser + auto-download
Hendrycks data.

## Remaining Items

Code-tier items queued for future rounds:
- `benchmark_compare.py` AC-11 directional gate (3-trial median,
  DS TPS ≥ 95% of DSA, P99 TTFT ≤ 1.10× DSA).
- Shallow AC-8 prefix-match regression coverage.
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality` (harness now complete + 
self-preparing; hardware execution pending).

## Push-to-remote Status

Branch is 28 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-conservative-llm-output-parser
Notes: The Round 26 "first A-D character anywhere in the response"
parser is a recurring shape across LLM eval harnesses. Adding a
BitLesson so future eval writers default to the conservative pattern:
match the leading isolated token + answer-introducer markers, return
None rather than guess from narrative.
