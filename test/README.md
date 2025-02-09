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
export OPENAI_API_KEY=sk-*****

# Run a single file
python3 test_openai_backend.py

# Run a single test
python3 -m unittest test_openai_backend.TestOpenAIServer.test_few_shot_qa

# Run a suite with multiple files
python3 run_suite.py --suite per-commit
```

## Adding or Updating Tests in CI

- Create new test files under `test/srt` or `test/lang` depending on the type of test.
- Ensure they are referenced in the respective `run_suite.py` (e.g., `test/srt/run_suite.py` or `test/lang/run_suite.py`) so they’re picked up in CI. For most small test cases, they can be added to the `per-commit` suite.
- The CI will run the `per-commit` and `nightly` automatically. If you need special setup or custom test groups, you may modify the workflows in [`.github/workflows/`](https://github.com/sgl-project/sglang/tree/main/.github/workflows).


## Writing Elegant Test Cases

- Examine existing tests in [sglang/test](https://github.com/sgl-project/sglang/tree/main/test) for practical examples.
- Keep each test function focused on a single scenario or piece of functionality.
- Give tests descriptive names reflecting their purpose.
- Use robust assertions (e.g., assert, unittest methods) to validate outcomes.
- Clean up resources to avoid side effects and preserve test independence.
