# Round 27 Contract

## Mainline Objective

Close the two correctness bugs Codex flagged in the Round 26 AC-12
harness so MMLU can actually fire on H200:

1. **MMLU data not self-preparing.** Round 26 added a `skipTest`
   when `benchmark/mmlu/data/{dev,test}` is missing — but the
   default checkout does NOT have those dirs. The result: every
   AC-12 MMLU run skips silently before touching either server.
   The existing `benchmark/mmlu/bench_sglang.py:55-73` already
   has a `download_data` helper Round 26 should have reused.
2. **`_parse_mmlu_letter` mis-scores common answer prefixes.** It
   returns the first A-D character anywhere in the response, so
   `"Answer: B"` returns `"A"` (the `A` in `"Answer"`). With
   `max_new_tokens=4`, the model is very likely to emit prefixes
   like `"Answer: B"`, `"B."`, or `"(C)"`, so correct B/C/D
   answers can be silently scored as A.

## Target ACs

- **AC-12** — the MMLU gate is self-preparing (downloads + extracts
  Hendrycks data on first run; fails hard if download fails AND
  servers are configured); the answer parser correctly extracts the
  predicted letter from realistic model completions.

## Required Implementation

### Fix 1: MMLU data self-prep helper

`test/manual/test_double_sparsity_v32.py`:

- New `_ensure_mmlu_data_dir(data_dir: str) -> Tuple[str, str]`:
  - If `data_dir/dev` and `data_dir/test` already exist, return
    `(dev_dir, test_dir)`.
  - Otherwise:
    1. `os.makedirs(data_dir, exist_ok=True)`.
    2. Download `https://people.eecs.berkeley.edu/~hendrycks/data.tar`
       via `urllib.request.urlretrieve` into a temp file. (Using
       `urllib` not `wget` so the harness has no extra dep.)
    3. Open via `tarfile.open(...)`; extract ONLY members whose
       `name` starts with the archive's `data/` prefix into a temp
       extraction dir (rejects path-traversal via `..` and absolute
       paths per Python's tarfile filter docs).
    4. Move the extracted `data/dev` and `data/test` subdirs into
       `data_dir` (atomic `os.rename`).
    5. Clean up the temp dir + tar file.
  - If any step raises, re-raise with a clear message naming the
    URL + data_dir.
- `test_mmlu_5shot` calls `_ensure_mmlu_data_dir(...)` BEFORE
  subject discovery. The current `skipTest("MMLU data not found...")`
  branch becomes a `self.fail(...)` if the env vars
  `DS_BASE_URL`+`DSA_BASE_URL` are set (i.e. the operator is
  actually running the gate). It still `skipTest`-es when env vars
  are unset — that's the class-level skip the harness already uses.

### Fix 2: Robust MMLU letter parser

Replace `_parse_mmlu_letter` with a regex-driven parser:

```
1. Strip leading whitespace.
2. Leading isolated letter (possibly wrapped in punctuation):
   ^[\s\(\[\{<"']*([A-Da-d])\b   →  group 1 uppercased
   Matches "B", " B", "(C)", "D.", "[A]", etc.
3. Otherwise scan for an answer-introducer (case-insensitive):
   "answer:", "answer is", "the answer is", "option", "choice"
   Take the first standalone A-D letter after the marker.
4. Otherwise None.
```

The crucial change: the first A-D character in `"Answer: B"` is no
longer matched as `A` (the leading-letter regex sees `A` at
position 0 but then `\b` fails because `n` follows; the function
then falls through to the marker scan which finds the `:` followed
by `B`).

### Fix 3: Registered regressions

`test/registered/unit/manual/test_ac12_helpers.py` — add ~7 cases
to lock the parser contract:

- `"B" → "B"` (already covered) + `"b" → "B"` (NEW — case fold).
- `"Answer: B" → "B"`.
- `"answer is C" → "C"` (lowercase marker).
- `"The answer is D." → "D"`.
- `"(C)" → "C"`.
- `"D." → "D"`.
- `"Awful question, but C is right" → "C"` (narrative text with
  decoy A in `Awful`; the leading-letter regex matches `A` but
  `\b` after `w` fails, so it falls through to the marker scan;
  with no marker, falls through to None — but the narrative
  contains `C is right` which a future enhancement could pick up;
  Round 27 just locks the conservative parser: this case returns
  `None` until we have evidence a less-conservative parser
  reduces error).
- `"" → None`.

Also lock the data-prep helper:

- Monkeypatch `urllib.request.urlretrieve` + `tarfile.open` (or
  pass an alternate fetcher) to fabricate a fake "data.tar"
  containing `data/dev/foo_dev.csv` and `data/test/foo_test.csv`;
  call `_ensure_mmlu_data_dir(tmp)`; assert both subdirs exist
  with the expected files.
- Assert `_ensure_mmlu_data_dir` is idempotent (returns same paths
  the second time without re-downloading).

## Tests

- Existing 254 tests must still pass.
- ~9 new helper regressions (parser × 7-8 + data-prep × 2).
- Expect ≥ 262 passed total.

## Success Criteria

1. `_parse_mmlu_letter("Answer: B") == "B"` (and 6 other cases per
   above).
2. `_ensure_mmlu_data_dir` returns `(dev_dir, test_dir)` after
   either finding existing data or downloading/extracting; raises
   on any failure.
3. Harness still skips cleanly without env vars.
4. With `DS_BASE_URL` + `DSA_BASE_URL` set and missing data dir,
   the MMLU test no longer silently skips; it either downloads
   on its own or fails hard.
5. `pytest test/registered -q` ≥ 262 passed.

## Blocking Issues

None.

## Queued (out of scope for Round 27)

- `benchmark_compare.py` AC-11 directional gate.
- Shallow AC-8 prefix-match regression coverage.
- Stale DS bind/runtime comments + token-label lifetime docs.
- Hardware-gated execution of AC-12, AC-1, AC-1b, AC-4, AC-6, AC-8,
  AC-9, AC-10, AC-11.
