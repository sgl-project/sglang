# Test and Continuous Integration (CI) System in SGLang

This page covers principles and essentials: folder layout, how to run tests, registration, and suite selection. For complete references, see the skill guides:

- **Writing tests** — templates, fixtures, model selection, complete suite tables, checklist: [`.claude/skills/write-sglang-test/SKILL.md`](../.claude/skills/write-sglang-test/SKILL.md)
- **CI pipeline internals** — stage flow diagrams, fast-fail layers, gating, partitioning, execution modes, debugging failures: [`.claude/skills/ci-workflow-guide/SKILL.md`](../.claude/skills/ci-workflow-guide/SKILL.md)

## CI Pipeline Overview

The CI pipeline runs in three sequential stages: **A** (pre-flight, ~3 min) → **B** (basic, ~30 min) → **C** (advanced, ~30 min). Kernel and multimodal-gen tests run in parallel with stage B. For details on stage gating, fast-fail mechanisms, execution modes (PR vs scheduled vs `/rerun-stage`), and debugging CI failures, see the [CI workflow guide](../.claude/skills/ci-workflow-guide/SKILL.md).

## Folder Organization

- `registered/`: CI test files, auto-discovered by `run_suite.py`. Most tests live here. JIT kernel tests are an exception (see below).
- `manual/`: Non-CI tests for local debugging or special setups.
- `run_suite.py`: CI runner — scans `registered/` and JIT kernel directories.
- `srt/`: Legacy CI setup, to be deprecated.

The system supports both [unittest](https://docs.python.org/3/library/unittest.html) and [pytest](https://docs.pytest.org/en/stable/). The launcher runs `python filename.py -f` with **failfast enabled by default**.

Make sure your file ends with **exactly** one of:

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

Do not add custom `argparse` or modify `sys.argv` before these calls — the CI runner appends `-f` for failfast.

## Run Tests Locally

```bash
# Single file
python3 test/registered/core/test_srt_endpoint.py

# Single test method
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Single JIT kernel test
python3 python/sglang/jit_kernel/tests/test_add_constant.py

# Run a suite
python3 test/run_suite.py --hw cpu --suite stage-a-test-cpu
python3 test/run_suite.py --hw cuda --suite stage-a-test-1-gpu-small

# Nightly tests
python3 test/run_suite.py --hw cuda --suite nightly-1-gpu --nightly

# With auto-partitioning (for parallel CI jobs)
python3 test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small \
    --auto-partition-id 0 --auto-partition-size 4
```

## CI Registration

Every CI-discovered test file must call a registration function at module level:

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=80, suite="stage-b-test-1-gpu-small")
```

Parameters: `est_time` (seconds), `suite` (target suite), `nightly=True` (nightly-only), `disabled="reason"` (temporarily disable).

Keep `est_time` and `suite` as **literal values** — `run_suite.py` collects them by AST parsing.

JIT kernel files live outside `test/registered/` but still use registration:
- Correctness tests: `python/sglang/jit_kernel/tests/test_*.py` → `stage-b-kernel-unit-1-gpu-large`
- Benchmarks: `python/sglang/jit_kernel/benchmark/bench_*.py` → `stage-b-kernel-benchmark-1-gpu-large`

## Choosing a Suite

Use the lightest suite that meets your test's needs. Full suite tables are in the [write-sglang-test skill](../.claude/skills/write-sglang-test/SKILL.md#all-ci-suites).

| Need | Suite |
|------|-------|
| No GPU required | `stage-a-test-cpu` |
| Small GPU (fits 5090, 32GB) | `stage-b-test-1-gpu-small` (most tests go here) |
| Large GPU memory or Hopper features | `stage-b-test-1-gpu-large` |
| JIT kernel correctness | `stage-b-kernel-unit-1-gpu-large` |
| JIT kernel benchmarks | `stage-b-kernel-benchmark-1-gpu-large` |
| Multi-GPU (2/4/8) | `stage-b-test-2-gpu-large`, `stage-c-test-*` |
| Long-running or experimental | `nightly-*` suites |

## Steps for Adding a Test

See the [write-sglang-test skill](../.claude/skills/write-sglang-test/SKILL.md) for templates, fixtures, model selection, and a complete checklist.

## Multi-Hardware Backends

This README mostly describes the NVIDIA GPU CI pipeline. Other hardware backends (AMD, NPU) follow the same practices and use the multi-backend registry system. A scheduled job summarizes test coverage across all backends; [here is an example run](https://github.com/sgl-project/sglang/actions/runs/23424304300).

## Tips

- Learn from existing examples in [test/registered](https://github.com/sgl-project/sglang/tree/main/test/registered).
- Reuse servers — launching is expensive. Share one server across many test methods via `setUpClass`.
- Use as few GPUs as possible. Prefer 1-GPU runners.
- Each test file should take < 500 seconds; split if longer.
- Each GitHub Actions job should take < 30 minutes; split if longer.
- If tests are too slow for per-commit, consider nightly suites.

## Other Notes

### Adding New Models to Nightly CI
- **Text models**: Extend the [global model list variables](https://github.com/sgl-project/sglang/blob/85c1f7937781199203b38bb46325a2840f353a04/python/sglang/test/test_utils.py#L104) in `test_utils.py`.
- **VLMs**: Extend the `MODEL_THRESHOLDS` dictionary in `test/srt/nightly/test_vlms_mmmu_eval.py`.
