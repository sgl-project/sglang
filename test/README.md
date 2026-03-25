# Test and Continuous Integration (CI) System in SGLang

This page introduces the test system, including the CI pipeline, file organization, and how to add and run tests.

## Three Stage CI Pipeline

The CI pipeline runs in three sequential stages after building the kernel:

- **Stage A** (pre-flight check, ~3 min): Quick smoke tests on small GPUs and CPU to catch obvious breakages early.
- **Stage B** (basic tests, ~30 min): Core functional tests on both small GPUs (e.g., 5090) and large GPUs (e.g., H100), including 1-GPU and 2-GPU configurations. Kernel tests and multimodal generation tests also run in parallel at this stage.
- **Stage C** (advanced tests, ~30 min): Multi-GPU and specialized hardware tests (H100, H200, B200), plus advanced features such as DeepEP, PD disaggregation, and GB300.

Here is an illustration
```
 ┌──────────────┐
 │ build kernel │
 └──────┬───────┘
        │
        ├─────────────────────────────────────────────────────┐
        │                                                     │
        ▼                                                     │
 ┌─────────────────────────────────────┐                      │
 │          Stage A (~3 min)           │                      │
 │         pre-flight check            │                      │
 │                                     │                      │
 │  ┌─────────────────────────────┐    │                      │
 │  │ stage-a-test-1-gpu-small    │    │                      │
 │  │ (small GPUs)                │    │                      │
 │  └─────────────────────────────┘    │                      │
 │  ┌─────────────────────────────┐    │                      │
 │  │ stage-a-test-cpu            │    │                      │
 │  │ (CPU)                       │    │                      │
 │  └─────────────────────────────┘    │                      │
 └──────┬──────────────────────────────┘                      │
        │                                                     │
        ▼                                                     ▼
 ┌─────────────────────────────────────┐          ┌──────────────────────────┐
 │          Stage B (~30 min)          │          │      kernel test         │
 │           basic tests               │          └──────────────────────────┘
 │                                     │          ┌──────────────────────────┐
 │  ┌─────────────────────────────┐    │          │   multimodal gen test    │
 │  │ stage-b-test-1-gpu-small    │    │          └──────────────────────────┘
 │  │ (small GPUs, e.g. 5090)     │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ stage-b-test-1-gpu-large    │    │
 │  │ (large GPUs, e.g. H100)     │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ stage-b-test-2-gpu-large    │    │
 │  │ (large GPUs, e.g. H100)     │    │
 │  └─────────────────────────────┘    │
 └──────┬──────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────┐
 │          Stage C (~30 min)          │
 │          advanced tests             │
 │                                     │
 │  ┌─────────────────────────────┐    │
 │  │ stage-c-test-1-gpu-h100     │    │
 │  │ (H100 GPUs)                 │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ stage-c-test-8-gpu-h200     │    │
 │  │ (8 x H200 GPUs)             │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ stage-c-test-4-gpu-b200     │    │
 │  │ (4 x B200 GPUs)             │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ Other advanced tests        │    │
 │  │ (DeepEP, PD Disagg, GB300)  │    │
 │  └─────────────────────────────┘    │
 └─────────────────────────────────────┘
```

- Stage naming convention: `stage-{a,b,c}-test-{gpu_count}-gpu-{hardware}`
- CI runner naming convention: `{gpu_count}-gpu-{hardware}` (e.g., `1-gpu-5090`, `4-gpu-h100`, `8-gpu-h200`)


## Folder organization
- `registered`: The registered test files. They are run in CI. Most tests should live in this folder. The main exception is JIT kernel coverage, which lives under `python/sglang/jit_kernel/tests/` and `python/sglang/jit_kernel/benchmark/`.
- `manual`: Test files that CI does not run; you run them manually. Typically, these are temporary tests, deprecated tests, or tests that are not suitable for CI—such as those that take too long or require special setup. We would still like to keep some files here for anyone who wants to run them locally.
- `run_suite.py`: The launch script to run a test suite. It scans `test/registered/` and also the JIT kernel test / benchmark directories.
- Other: utility scripts and metadata folders. The `srt` folder holds our legacy CI setup and should be deprecated as soon as possible.

Because the system uses a custom registry and the `run_suite.py` launcher, it supports both Python's built-in [unittest](https://docs.python.org/3/library/unittest.html) and the popular [pytest](https://docs.pytest.org/en/stable/) framework.
The basic unit is a file, and you can use either framework in your file.
The launcher runs `python filename.py -f` to execute tests with **failfast enabled by default** — the first test method failure stops the file immediately. This avoids wasting CI time on remaining tests after a failure.

Make sure your file ends with **exactly** one of the following blocks. Do not add custom `argparse` or modify `sys.argv` before calling `unittest.main()` / `pytest.main()` — the CI runner appends `-f` for failfast, and custom argument parsing will break it.

```python
# for unittest
if __name__ == "__main__":
    unittest.main()
```

```python
# for pytest
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
```

## Run tests locally

### Run a single file or a single test
```bash
# Run a single file
python3 test/registered/core/test_srt_endpoint.py

# Run a single test
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run a single JIT kernel test file
python3 python/sglang/jit_kernel/tests/test_add_constant.py
```

### Run a suite with multiple files
```bash
# Run the CPU-only tests
python3 test/run_suite.py --hw cpu --suite stage-a-test-cpu

# Run the small GPU test
python3 test/run_suite.py --hw cuda --suite stage-a-test-1-gpu-small
```

### More examples
```bash
# Run nightly tests
python test/run_suite.py --hw cuda --suite nightly-1-gpu --nightly

# With auto-partitioning (for parallel CI jobs)
python test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small \
    --auto-partition-id 0 --auto-partition-size 4
```


## CI Registry System

CI-discovered tests use a registry-based CI system for flexible backend and schedule configuration.
This includes files under `test/registered/` and, for JIT kernels, files under `python/sglang/jit_kernel/tests/` and `python/sglang/jit_kernel/benchmark/`.
For every CI-discovered file you add, you need to register it in a suite and provide an estimated execution time in seconds.

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
register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)

# Temporarily disabled test
register_cuda_ci(est_time=80, suite="stage-b-test-1-gpu-small", disabled="flaky - see #12345")
```

### JIT kernel exception

JIT kernel files are discovered by `test/run_suite.py`, but they do not live under `test/registered/`:

- Correctness tests: `python/sglang/jit_kernel/tests/test_*.py`
- Benchmarks: `python/sglang/jit_kernel/benchmark/bench_*.py`

Use dedicated kernel suites:

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=6, suite="stage-b-kernel-benchmark-1-gpu-large")
# Optional nightly registration
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)
```

Keep `est_time` and `suite` as literal values. `run_suite.py` collects them by statically parsing the file AST.

## Available Suites

You can find the available suites for each hardware backend at [`test/run_suite.py`](run_suite.py) (`PER_COMMIT_SUITES`, `NIGHTLY_SUITES`). Here we briefly describe some suites.

### Per-commit (CUDA)

| Suite | Runner (label) | Description |
| --- | --- | --- |
| `stage-a-test-1-gpu-small` | `1-gpu-5090` | Quick checks on a small NVIDIA GPU before heavier stages |
| `stage-b-test-1-gpu-small` | `1-gpu-5090` | Core engine tests that fit a 5090-class card |
| `stage-b-test-1-gpu-large` | `1-gpu-h100` | Tests that need H100-class memory or kernels (e.g. FA3) |
| `stage-b-test-2-gpu-large` | `2-gpu-h100` | Two-GPU correctness and parallelism (TP/PP-style workloads) on H100 |
| `stage-b-test-4-gpu-b200` | `4-gpu-b200` | Early Blackwell coverage (e.g. SM100+ paths) on four GPUs |
| `stage-b-kernel-unit-1-gpu-large` | `1-gpu-h100` | JIT kernel correctness tests under `python/sglang/jit_kernel/tests/` |
| `stage-b-kernel-benchmark-1-gpu-large` | `1-gpu-h100` | JIT kernel benchmark files under `python/sglang/jit_kernel/benchmark/` |
| `stage-c-test-4-gpu-h100` | `4-gpu-h100` | Large 4-GPU H100 integration and scaling tests |
| `stage-c-test-8-gpu-h200` | `8-gpu-h200` | Large 8-GPU H200 runs for big models and parallelism |
| `stage-c-test-8-gpu-h20` | `8-gpu-h20` | Large 8-GPU H20 runs for big models |
| `stage-c-test-deepep-4-gpu-h100` | `4-gpu-h100` | DeepEP expert-parallel and related networking on four H100s. |
| `stage-c-test-deepep-8-gpu-h200` | `8-gpu-h200` | DeepEP at 8-GPU H200 scale. |
| `stage-c-test-4-gpu-b200` | `4-gpu-b200` | 4-GPU B200 suite for large models on blackwell |
| `stage-c-test-4-gpu-gb200` | `4-gpu-gb200`| 4-GPU GB200 suite for large models on grace blackwell |

Multimodal diffusion uses `python/sglang/multimodal_gen/test/run_suite.py`, not `test/run_suite.py`.

### Per-commit (CPU)

| Suite | Runner (label) | Description |
| --- | --- | --- |
| `stage-a-test-cpu` | `ubuntu-latest` | CPU-only unit tests |

### Per-commit (AMD)

| Suite | Runner (label) | Description |
| --- | --- | --- |
| `stage-a-test-1-gpu-small-amd` | `linux-mi325-1gpu-sglang` | Quick checks on one MI325-class GPU in the AMD CI container. |
| `stage-b-test-2-gpu-large-amd` | `linux-mi325-2gpu-sglang` | 2-GPU ROCm correctness and parallel setups. |
| `stage-b-test-large-8-gpu-35x-disaggregation-amd` | `linux-mi35x-gpu-8.fabric` | Prefill–decode disaggregation and RDMA-oriented tests on an 8×MI35x fabric runner. |
| `stage-c-test-large-8-gpu-amd` | `linux-mi325-8gpu-sglang` | 8-GPU MI325 scaling and integration. |

### Nightly

Nightly registry suites are listed in `NIGHTLY_SUITES` in [`test/run_suite.py`](run_suite.py). They are not driven by `pr-test.yml` / `pr-test-amd*.yml`; see workflows such as `nightly-test-nvidia.yml` and `nightly-test-amd.yml`. Examples:

- `nightly-1-gpu` (CUDA)
- `nightly-kernel-1-gpu` (CUDA, JIT kernel full grids)
- `nightly-8-gpu-h200` (CUDA)
- `nightly-eval-vlm-2-gpu` (CUDA)
- `nightly-amd` (AMD)
- `nightly-amd-8-gpu-mi35x` (AMD)

### Choosing a suite for your test

Use the lightest suite that still meets your test's needs.

- Prefer the CPU suite (`stage-a-test-cpu`) when no GPU is required.
- For most small GPU workloads that fit a 5090-class card in CI, use `stage-b-test-1-gpu-small`. Most tests should go here.
- If you really need more GPU memory capacity or Hopper-specific features, use `stage-b-test-1-gpu-large`.
- For JIT kernel work under `python/sglang/jit_kernel/`, use `stage-b-kernel-unit-1-gpu-large` for correctness tests and `stage-b-kernel-benchmark-1-gpu-large` for benchmarks.
- Use multi-GPU suites only when the test actually needs multiple GPUs or other advanced multi-GPU behavior.

In rare cases, if you need a new runner or custom setup, you might need to add a new suite.

## Steps for Adding a Test
Please refer to [.claude/skills/write-sglang-test/SKILL.md](../.claude/skills/write-sglang-test/SKILL.md)

## Multi-hardware backends
This README mostly describes the CI pipeline for NVIDIA GPU backends.
Other hardware backends should follow the same practices, use the multi-backend registry system, and build their own pipelines.
A scheduled job summarizes test coverage across all backends; [here is an example run](https://github.com/sgl-project/sglang/actions/runs/23424304300).

## Tips for Writing Elegant Test Cases
- Learn from existing examples in [test/registered](https://github.com/sgl-project/sglang/tree/main/test/registered).
- Reduce the test time by using smaller models and reusing the server for multiple test cases. Launching a server takes a lot of time, so please reuse a single server for many tests instead of launching many servers.
- Use as few GPUs as possible. Use 1-GPU runners whenever possible. Do not run long tests with 8-gpu runners.
- If the test cases take too long, consider adding them to nightly tests instead of per-commit tests.
- Each test file `test_xxx.py` should take less than 500 seconds. If a single file takes longer than that, split it into multiple files.
- Each GitHub Actions job should take less than 30 minutes. If a single job takes longer than that, split it into multiple jobs.

## Other Notes
### Adding New Models to Nightly CI
- **For text models**: Extend the [global model list variables](https://github.com/sgl-project/sglang/blob/85c1f7937781199203b38bb46325a2840f353a04/python/sglang/test/test_utils.py#L104) in `test_utils.py`, or add more model lists.
- **For VLMs**: Extend the `MODEL_THRESHOLDS` global dictionary in `test/srt/nightly/test_vlms_mmmu_eval.py`.
