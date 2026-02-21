# Run Unit Tests

SGLang uses the built-in library [unittest](https://docs.python.org/3/library/unittest.html) as the testing framework.

## Test Backend Runtime
```bash
# Run a single file
> cd test/registered
> python3 core/test_srt_endpoint.py

# Run a single test
> cd test/registered
> python3 core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run a suite with multiple files
> cd test
> python run_suite.py --hw cuda --suite stage-b-test-small-1-gpu
```

## Test Frontend Language
```bash
> cd test/manual/lang_frontend

# Run a single file
> python3 test_choices.py
```

## Adding or Updating Tests in CI

- Create new test files under `test/registered/` (organized by category) for CI tests, or `test/manual/` for manual tests.
- For nightly tests, use the CI registry with `nightly=True`. For performance benchmarking tests, use the `NightlyBenchmarkRunner` helper class in `python/sglang/test/nightly_utils.py`.
- Register tests using the CI registry system (see below). For most small test cases, use the `stage-b-test-small-1-gpu` suite. Sort the test cases alphabetically by name.
- Ensure you added `unittest.main()` for unittest and `sys.exit(pytest.main([__file__]))` for pytest in the scripts. The CI runs them via `python3 test_file.py`.
- The CI will run some suites such as `stage-b-test-small-1-gpu`, `stage-b-test-large-2-gpu`, and `nightly-1-gpu` automatically. If you need special setup or custom test groups, you may modify the workflows in [`.github/workflows/`](https://github.com/sgl-project/sglang/tree/main/.github/workflows).

## CI Registry System

Tests in `test/registered/` use a registry-based CI system for flexible backend/schedule configuration.

### Registration Functions

```python
from sglang.test.ci.ci_register import (
    register_cuda_ci,
    register_amd_ci,
    register_cpu_ci,
    register_npu_ci,
)

# Per-commit test (small 1-gpu, runs on 5090)
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu")

# Per-commit test (large 1-gpu, runs on H100)
register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")

# Per-commit test (2-gpu)
register_cuda_ci(est_time=200, suite="stage-b-test-large-2-gpu")

# Nightly-only test
register_cuda_ci(est_time=200, suite="nightly-1-gpu", nightly=True)

# Multi-backend test
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=120, suite="stage-b-test-small-1-gpu-amd")

# Temporarily disabled test
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu", disabled="flaky - see #12345")
```

### Choosing Between 1-GPU Suites (5090 vs H100)

When adding 1-GPU tests, choose the appropriate suite based on hardware compatibility:

| Suite | Runner | GPU | When to Use |
|-------|--------|-----|-------------|
| `stage-b-test-small-1-gpu` | `1-gpu-5090` | RTX 5090 (32GB, SM120) | 5090-compatible tests (preferred) |
| `stage-b-test-large-1-gpu` | `1-gpu-runner` | H100 (80GB, SM90) | Large models or 5090-incompatible tests |

**Use `stage-b-test-small-1-gpu` (5090) whenever possible** - this is the preferred suite for most 1-GPU tests.

**Use `stage-b-test-large-1-gpu` (H100) if ANY of these apply:**

1. **Architecture incompatibility (SM120/Blackwell)**:
   - FA3 attention backend (requires SM≤90)
   - MLA with FA3 backend
   - FP8/MXFP4 quantization (not supported on SM120)
   - Certain Triton kernels (shared memory limits)

2. **Memory requirements**:
   - Models >30B params or large MoE
   - Tests requiring >32GB VRAM

3. **Known 5090 failures**:
   - Weight update/sync tests
   - Certain spec decoding tests

If a test cannot run on 5090 due to any of the above, use `stage-b-test-large-1-gpu` which runs on H100.

### Available Suites

**Per-Commit (CUDA)**:
- Stage A: `stage-a-test-1` (locked), `stage-a-cpu-only`
- Stage B: `stage-b-test-small-1-gpu` (5090), `stage-b-test-large-1-gpu` (H100), `stage-b-test-large-2-gpu`
- Stage C (4-GPU): `stage-c-test-4-gpu-h100`, `stage-c-test-4-gpu-b200`, `stage-c-test-4-gpu-gb200`, `stage-c-test-deepep-4-gpu`
- Stage C (8-GPU): `stage-c-test-8-gpu-h20`, `stage-c-test-8-gpu-h200`, `stage-c-test-8-gpu-b200`, `stage-c-test-deepep-8-gpu-h200`

**Per-Commit (AMD)**:
- `stage-a-test-1-amd`, `stage-b-test-small-1-gpu-amd`, `stage-b-test-large-1-gpu-amd`, `stage-b-test-large-2-gpu-amd`

**Per-Commit (NPU)**:
- `stage-a-test-1`, `stage-b-test-1-npu-a2`, `stage-b-test-2-npu-a2`, `stage-b-test-4-npu-a3`, `stage-b-test-16-npu-a3`

**Nightly (CUDA)**:
- `nightly-1-gpu`, `nightly-2-gpu`, `nightly-4-gpu`, `nightly-8-gpu`, etc.
- Eval: `nightly-eval-text-2-gpu`, `nightly-eval-vlm-2-gpu`
- Perf: `nightly-perf-text-2-gpu`, `nightly-perf-vlm-2-gpu`

**Nightly (AMD)**:
- `nightly-amd`, `nightly-amd-1-gpu`, `nightly-amd-8-gpu`, `nightly-amd-vlm`

### Running Tests with run_suite.py

```bash
# Run per-commit tests
python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu

# Run nightly tests
python test/run_suite.py --hw cuda --suite nightly-1-gpu --nightly

# With auto-partitioning (for parallel CI jobs)
python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu \
    --auto-partition-id 0 --auto-partition-size 4
```

## Writing Elegant Test Cases

- Learn from existing examples in [sglang/test/registered](https://github.com/sgl-project/sglang/tree/main/test/registered).
- Reduce the test time by using smaller models and reusing the server for multiple test cases. Launching a server takes a lot of time.
- Use as few GPUs as possible. Do not run long tests with 8-gpu runners.
- If the test cases take too long, considering adding them to nightly tests instead of per-commit tests.
- Keep each test function focused on a single scenario or piece of functionality.
- Give tests descriptive names reflecting their purpose.
- Use robust assertions (e.g., assert, unittest methods) to validate outcomes.
- Clean up resources to avoid side effects and preserve test independence.


## Adding New Models to Nightly CI
- **For text models**: extend the `DEFAULT_MODEL_NAME_FOR_NIGHTLY_EVAL_*` variables in `python/sglang/test/test_utils.py`, or add new model constants.
- **For VLMs**: extend the `MODEL_THRESHOLDS` dictionary in `test/registered/eval/test_vlms_mmmu_eval.py`.
