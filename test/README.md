# SGLang Test Suite

SGLang uses the built-in library [unittest](https://docs.python.org/3/library/unittest.html) as the testing framework.

## Test Directory Structure

```
test/
├── manual/                    # Unofficially maintained tests
│   └── ...                    # Code references for agents, not guaranteed to run in CI
│
├── registered/                # Officially maintained tests (per-commit + nightly)
│   ├── layers/                # Layer-level tests
│   │   ├── attention/
│   │   │   └── mamba/
│   │   └── mla/
│   ├── models/                # Model-specific tests
│   ├── openai_server/         # OpenAI API compatibility tests
│   │   ├── basic/
│   │   ├── features/
│   │   ├── function_call/
│   │   └── validation/
│   ├── lora/                  # LoRA tests
│   ├── quant/                 # Quantization tests
│   ├── rl/                    # RL/training tests
│   ├── hicache/               # HiCache tests
│   ├── ep/                    # Expert parallelism tests
│   ├── speculative/           # Speculative decoding tests
│   ├── runtime/               # Runtime tests
│   ├── cache/                 # Cache tests
│   ├── scheduler/             # Scheduler tests
│   ├── sampling/              # Sampling tests
│   ├── ops/                   # Operator tests
│   ├── cpu/                   # CPU backend tests
│   ├── ascend/                # Ascend NPU tests
│   └── xpu/                   # Intel XPU tests
│
├── srt/                       # Legacy test location (being migrated to registered/)
├── nightly/                   # Legacy nightly tests (being migrated to registered/)
│
├── run_suite.py               # Run registered tests by suite
└── run_suite_nightly.py       # Legacy nightly runner
```

## CI Registry

Tests in `registered/` use a registry system to declare their CI requirements:

```python
from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci

# Per-commit test (runs on every PR)
register_cuda_ci(est_time=80, suite="stage-a-test-1")
register_amd_ci(est_time=120, suite="stage-a-test-1")

# Nightly-only test
register_cuda_ci(est_time=200, suite="nightly-1-gpu", nightly=True)

# Temporarily disabled test (keeps all metadata for re-enabling)
register_cuda_ci(est_time=80, suite="stage-a-test-1", disabled="flaky - see #12345")
```

### Registry Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `est_time` | float | Estimated runtime in seconds |
| `suite` | str | Test suite name (e.g., "stage-a-test-1", "nightly-1-gpu") |
| `nightly` | bool | If True, only runs in nightly CI (default: False) |
| `disabled` | str | If provided, test is skipped with this reason |

## Running Tests

### Run Registered Tests

```bash
cd sglang/test

# Run per-commit tests for CUDA
python run_suite.py --hw cuda --suite stage-a-test-1

# Run nightly tests (includes per-commit tests)
python run_suite.py --hw cuda --suite nightly-1-gpu --nightly

# Run AMD tests
python run_suite.py --hw amd --suite stage-a-test-1
```

### Run Legacy Tests (test/srt/)

```bash
cd sglang/test/srt

# Run a single file
python3 test_srt_endpoint.py

# Run a single test
python3 test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run a suite
python3 run_suite.py --suite per-commit-1-gpu
```

## Adding New Tests

1. **Create test file** in the appropriate feature directory under `test/registered/`
2. **Add CI registry** at the top of the file:
   ```python
   from sglang.test.ci.ci_register import register_cuda_ci

   register_cuda_ci(est_time=60, suite="stage-a-test-1")
   ```
3. **Add `unittest.main()` or `pytest.main()`** at the bottom:
   ```python
   if __name__ == "__main__":
       unittest.main()
   ```

## CI Registry Quick Peek

Tests in `test/registered/` declare CI metadata via lightweight markers:

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=80, suite="stage-a-test-1")
```

## Writing Elegant Test Cases

- Use smaller models and reuse servers to reduce test time
- Use as few GPUs as possible
- Keep each test function focused on a single scenario
- Give tests descriptive names reflecting their purpose
- Clean up resources to avoid side effects
- For long-running tests, use `nightly=True` in the registry

## Adding New Models to Nightly CI

- **For text models**: extend [global model lists](https://github.com/sgl-project/sglang/blob/main/python/sglang/test/test_utils.py) in `test_utils.py`
- **For VLMs**: extend the `MODEL_THRESHOLDS` dictionary in `test/srt/nightly/test_vlms_mmmu_eval.py`
