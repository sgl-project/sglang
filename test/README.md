# Run Unit Tests

SGLang uses the built-in library [unittest](https://docs.python.org/3/library/unittest.html) as the testing framework.

## Test Backend Runtime
```bash
cd sglang/test/srt

# Run a single file
python3 test_srt_endpoint.py

# Run a single test
python3 -m unittest test_srt_endpoint.TestSRTEndpoint.test_simple_decode

# Run a suite with multiple files
python3 run_suite.py --suite per-commit
```

## Test Frontend Language
```bash
cd sglang/test/lang

# Run a single file
python3 test_srt_backend.py
```

## Adding or Updating Tests in CI

- Create new test files under `test/srt` or `test/lang` depending on the type of test.
- Ensure they are referenced in the respective `run_suite.py` (e.g., `test/srt/run_suite.py`) so they’re picked up in CI. For most small test cases, they can be added to the `per-commit` suite. Sort the test cases alphabetically.
- The CI will run the `per-commit` and `nightly` automatically. If you need special setup or custom test groups, you may modify the workflows in [`.github/workflows/`](https://github.com/sgl-project/sglang/tree/main/.github/workflows).


## Writing Elegant Test Cases

- Examine existing tests in [sglang/test](https://github.com/sgl-project/sglang/tree/main/test) for practical examples.
- Keep each test function focused on a single scenario or piece of functionality.
- Give tests descriptive names reflecting their purpose.
- Use robust assertions (e.g., assert, unittest methods) to validate outcomes.
- Clean up resources to avoid side effects and preserve test independence.
- Reduce the test time by using smaller models and reusing the server for multiple test cases.


## Adding new models to nightly-ci
- **For text models**: extend [global model lists variables](https://github.com/sgl-project/sglang/blob/85c1f7937781199203b38bb46325a2840f353a04/python/sglang/test/test_utils.py#L104) in `test_utils.py`, or add more model lists
- **For vlms**: extend global variables containing `ModelLaunchSettings` in `test_nightly_vlms_.*.py`, see [here](https://github.com/sgl-project/sglang/blob/85c1f7937781199203b38bb46325a2840f353a04/test/srt/test_nightly_vlms_mmmu_eval.py#L18)

