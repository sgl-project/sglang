# Run Unit Tests

This page introduces the test system in sglang, including how to run and add tests.

## Three Stage CI Pipeline


## Folder organization
- `registered`: The registered test files. They will be run on CI. Most tests should be in this folder. It uses a custom registry system and use a file as the basic unit.
- `manual`: Test files that won't be run by CI. You need to manually run them. Typically, these are temporary tests, deprecated tests, or tests that not suitable for running on CI such as taking too long or requires special setup. We still would like to keep some files here in case someone wants to run them locally.
- `run_suite.py`: The launch script to run a test suite.
- others: some utility scripts, metadata folder. The `srt` folder is our old CI system and should be deprecated as soon as possible.

Since the system use a custom registry and launcher `run_suite.py`, it supports both python's built-in [unittest](https://docs.python.org/3/library/unittest.html) or the popular [pytest](https://docs.pytest.org/en/stable/).
The basic unit is a file and you can use either of them in your file.
The launcher will call `python filename.py` to run the test, so MAKE SURE you have the following lines in your file. Otherwise, a file won't be executed by CI.

```python
# for unittest
if __name__ == "__main__":
    unittest.main()
```

```python
# for pytest
if __name__ == "__main__":
    pytest.main([__file__])
```

## Run tests locally

### Run a single file or a single test
```bash
# Run a single file
python3 test/registered/core/test_srt_endpoint.py

# Run a single test
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode
```

### Run a suite with multiple files
```bash
# Run the CPU only tests
python3 test/run_suite.py --hw cpu --suite stage-a-test-cpu

# Run the small GPU test
python3 test/run_suite.py --hw cuda --suite stage-a-test-1-gpu-small
```


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
register_cuda_ci(est_time=80, suite="stage-b-test-1-gpu-small")

# Per-commit test (large 1-gpu, runs on H100)
register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-large")

# Per-commit test (2-gpu)
register_cuda_ci(est_time=200, suite="stage-b-test-2-gpu-large")

# Nightly-only test
register_cuda_ci(est_time=200, suite="nightly-1-gpu", nightly=True)

# Multi-backend test
register_cuda_ci(est_time=80, suite="stage-a-test-1-gpu-small")
register_amd_ci(est_time=120, suite="stage-a-test-1-gpu-small-amd")

# Temporarily disabled test
register_cuda_ci(est_time=80, suite="stage-b-test-1-gpu-small", disabled="flaky - see #12345")
```

### Available Suites

**Per-Commit (CUDA)**:
- Stage A: `stage-a-test-1-gpu-small` (5090), `stage-a-test-2`, `stage-a-test-cpu`
- Stage B: `stage-b-test-1-gpu-small` (5090), `stage-b-test-1-gpu-large` (H100), `stage-b-test-2-gpu-large`
- Stage C (4-GPU): `stage-c-test-4-gpu-h100`, `stage-c-test-4-gpu-b200`, `stage-c-test-4-gpu-gb200`, `stage-c-test-deepep-4-gpu-h100`
- Stage C (8-GPU): `stage-c-test-8-gpu-h20`, `stage-c-test-8-gpu-h200`, `stage-c-test-8-gpu-b200`, `stage-c-test-deepep-8-gpu-h200`

**Per-Commit (AMD)**:
- `stage-a-test-1-gpu-small-amd`, `stage-b-test-1-gpu-small-amd`, `stage-b-test-2-gpu-large-amd`

**Nightly**:
- `nightly-1-gpu`, `nightly-2-gpu`, `nightly-4-gpu`, `nightly-8-gpu`, etc.

### Choosing Between 1-GPU Suites (5090 vs H100)

When adding 1-GPU tests, choose the appropriate suite based on hardware compatibility:

| Suite | Runner | GPU | When to Use |
|-------|--------|-----|-------------|
| `stage-a-test-1-gpu-small` | `1-gpu-5090` | RTX 5090 (32GB, SM120) | Most small tests |
| `stage-a-test-1-gpu-small-amd` | AMD CI runners | ROCm | Stage A per-commit smoke (AMD) |
| `stage-b-test-1-gpu-small` | `1-gpu-5090` | RTX 5090 (32GB, SM120) | 5090-compatible tests (preferred) |
| `stage-b-test-1-gpu-large` | `1-gpu-h100` | H100 (80GB, SM90) | Large models or 5090-incompatible tests |

**Use `stage-b-test-1-gpu-small` (5090) whenever possible** - this is the preferred suite for most 1-GPU tests.

**Use `stage-b-test-1-gpu-large` (H100) if ANY of these apply:**

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

If a test cannot run on 5090 due to any of the above, use `stage-b-test-1-gpu-large` which runs on H100.

### Running Tests with run_suite.py

```bash
# Run per-commit tests
python test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small

# Run nightly tests
python test/run_suite.py --hw cuda --suite nightly-1-gpu --nightly

# With auto-partitioning (for parallel CI jobs)
python test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small \
    --auto-partition-id 0 --auto-partition-size 4
```


## Multi hardware backends


## Tips for Writing Elegant Test Cases
- Learn from existing examples in [test/registered](https://github.com/sgl-project/sglang/tree/main/test/registered).
- Reduce the test time by using smaller models and reusing the server for multiple test cases. Launching a server takes a lot of time, so please reuse a single server for many tests instead of launching many servers.
- Use as few GPUs as possible. Use 1-GPU runners whenever possible. Do not run long tests with 8-gpu runners.
- If the test cases take too long, considering adding them to nightly tests instead of per-commit tests.
- Each test file `test_xxx.py` should take less than 500s. If a single file takes more than that, split it into multiple files.

## Other Notes
### Adding New Models to Nightly CI
- **For text models**: extend [global model lists variables](https://github.com/sgl-project/sglang/blob/85c1f7937781199203b38bb46325a2840f353a04/python/sglang/test/test_utils.py#L104) in `test_utils.py`, or add more model lists
- **For vlms**: extend the `MODEL_THRESHOLDS` global dictionary in `test/srt/nightly/test_vlms_mmmu_eval.py`
