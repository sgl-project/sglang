# Run Unit Tests

SGLang uses the built-in library [unittest](https://docs.python.org/3/library/unittest.html) as the testing framework.

## Test Backend Runtime
```bash
cd sglang/test/srt

# Run a single file
python3 test_srt_endpoint.py

# Run a single test
python3 test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run a suite with multiple files
python3 run_suite.py --suite per-commit
```

## Test Frontend Language
```bash
cd sglang/test/lang

# Run a single file
python3 test_choices.py
```

## Adding or Updating Tests in CI

- Create new test files under `test/srt` or `test/lang` depending on the type of test.
- For nightly tests, place them in `test/srt/nightly/`. Use the `NightlyBenchmarkRunner` helper class in `nightly_utils.py` for performance benchmarking tests.
- Ensure they are referenced in the respective `run_suite.py` (e.g., `test/srt/run_suite.py`) so they are picked up in CI. For most small test cases, they can be added to the `per-commit-1-gpu` suite. Sort the test cases alphabetically by name.
- Ensure you added `unittest.main()` for unittest and `pytest.main([__file__])` for pytest in the scripts. The CI run them via `python3 test_file.py`.
- The CI will run some suites such as `per-commit-1-gpu`, `per-commit-2-gpu`, and `nightly-1-gpu` automatically. If you need special setup or custom test groups, you may modify the workflows in [`.github/workflows/`](https://github.com/sgl-project/sglang/tree/main/.github/workflows).

## CI Registry Quick Peek

Tests in `test/registered/` declare CI metadata via lightweight markers:

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=80, suite="stage-a-test-1")
```

## Writing Elegant Test Cases

- Learn from existing examples in [sglang/test/srt](https://github.com/sgl-project/sglang/tree/main/test/srt).
- Reduce the test time by using smaller models and reusing the server for multiple test cases. Launching a server takes a lot of time.
- Use as few GPUs as possible. Do not run long tests with 8-gpu runners.
- If the test cases take too long, considering adding them to nightly tests instead of per-commit tests.
- Keep each test function focused on a single scenario or piece of functionality.
- Give tests descriptive names reflecting their purpose.
- Use robust assertions (e.g., assert, unittest methods) to validate outcomes.
- Clean up resources to avoid side effects and preserve test independence.
- Reduce the test time by using smaller models and reusing the server for multiple test cases.


## Adding New Models to Nightly CI
- **For text models**: extend [global model lists variables](https://github.com/sgl-project/sglang/blob/85c1f7937781199203b38bb46325a2840f353a04/python/sglang/test/test_utils.py#L104) in `test_utils.py`, or add more model lists
- **For vlms**: extend the `MODEL_THRESHOLDS` global dictionary in `test/srt/nightly/test_vlms_mmmu_eval.py`
