---
name: write-sglang-test
description: Guide for writing SGLang CI/UT tests. Covers CustomTestCase, CI registration, server fixtures, model selection, mock testing, and test placement. Always read test/README.md for the full CI layout, how to run tests, and extra tips. Use when creating new tests, adding CI test cases, writing unit tests, or when the user asks to add tests for SGLang features.
---

# Writing SGLang CI / UT Tests

**Before or while applying this skill, read [`test/README.md`](../../../test/README.md)** in the repo root’s `test/` tree. It documents the three-stage CI pipeline, suite naming, how tests are discovered and executed, file layout, and practical tips that complement the sections below.

## Core Rules

1. **Always use `CustomTestCase`** — never raw `unittest.TestCase`. It ensures `tearDownClass` runs even when `setUpClass` fails, preventing resource leaks in CI.
2. **`tearDownClass` must be defensive** — use `hasattr`/null checks before accessing resources (e.g. `cls.process`) that `setUpClass` may not have finished allocating.
3. **Place tests in `test/registered/<category>/`** — except JIT kernel tests and benchmarks, which live in `python/sglang/jit_kernel/tests/` and `python/sglang/jit_kernel/benchmark/`
4. **Reuse server fixtures** — inherit from `DefaultServerBase` or write `setUpClass`/`tearDownClass` with `popen_launch_server`
5. **Prefer mock over real server** — when testing logic that doesn't need a server / engine launch (middleware, request routing, config validation, argument parsing), use `unittest.mock.patch` / `MagicMock` and place tests in `test/registered/unit/`. Only launch a real server when the test genuinely needs inference results or server lifecycle behavior.

JIT kernel exception:
- If the task is adding or updating code under `python/sglang/jit_kernel/`, prefer the `add-jit-kernel` skill first.
- JIT kernel correctness tests use `python/sglang/jit_kernel/tests/test_*.py`.
- JIT kernel benchmarks use `python/sglang/jit_kernel/benchmark/bench_*.py`.
- Those files are still executed by `test/run_suite.py`, but through dedicated kernel suites rather than `test/registered/`.

---

## Model & Backend Selection

| Scenario | Model | CI Registration | Suite |
|----------|-------|-----------------|-------|
| **Unit tests** (no server / engine launch) | None | `register_cpu_ci` (prefer) or `register_cuda_ci` | `stage-a-test-cpu` or `stage-b-test-1-gpu-small` |
| **Common / backend-independent** (middleware, abort, routing, config, arg parsing) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` only | `stage-b-test-1-gpu-small` |
| **Model-agnostic functionality** (sampling, session, OpenAI API features) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` (+ AMD if relevant) | `stage-b-test-1-gpu-small` |
| **General performance** (single node, no spec/DP/parallelism) | `DEFAULT_MODEL_NAME_FOR_TEST` (8B) | `register_cuda_ci` | `stage-b-test-1-gpu-large` |
| **Bigger features** (spec, DP, TP, disaggregation) | Case by case | Case by case | See suite table below |

**Key principle for E2E tests**: Do NOT add `register_amd_ci` unless the test specifically exercises AMD/ROCm code paths. Common E2E tests just need any GPU to run — duplicating across backends wastes CI time with no extra coverage.

### All model constants

Defined in `python/sglang/test/test_utils.py`:

| Constant | Model | When to use |
|----------|-------|-------------|
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` | Llama-3.2-1B-Instruct | Common features, model-agnostic tests |
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE` | Llama-3.2-1B | Base (non-instruct) model tests |
| `DEFAULT_MODEL_NAME_FOR_TEST` | Llama-3.1-8B-Instruct | General performance (single node) |
| `DEFAULT_MOE_MODEL_NAME_FOR_TEST` | Mixtral-8x7B-Instruct | MoE-specific tests |
| `DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST` | — | Embedding tests |
| `DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST` | — | Vision-language tests |

### All CI suites

| Suite | Runner | Scenario |
|-------|--------|----------|
| `stage-a-test-cpu` | CPU | CPU unit tests |
| `stage-b-test-1-gpu-small` | 1× 5090 (32GB) | Small model tests (1B, 8B) |
| `stage-b-test-1-gpu-large` | 1× H100 (80GB) | Large model tests or performance related |
| `stage-b-test-2-gpu-large` | 2× H100 | TP=2 tests |
| `stage-c-test-4-gpu-h100` | 4× H100 | TP=4 / EP tests |
| `stage-c-test-8-gpu-h200` | 8× H200 | Large-scale multi-GPU |
| `nightly-1-gpu` | 1 GPU | Nightly-only |
| `nightly-8-gpu` | 8 GPU | Nightly-only |

---

## Test File Templates

### Unit Tests (no server / engine launch)

See `test/registered/unit/README.md` for quick-start and rules. Unit tests live in `test/registered/unit/`, mirroring `python/sglang/srt/`:

```python
"""Unit tests for srt/<module>"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.<module> import TargetClass
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")
# Prefer CPU. Only use register_cuda_ci when the test truly needs a GPU.

class TestTargetClass(CustomTestCase):
    def test_basic_behavior(self):
        obj = TargetClass(...)
        self.assertEqual(obj.method(), expected)

    @patch("sglang.srt.<module>.some_dependency")
    def test_with_mock(self, mock_dep):
        mock_dep.return_value = MagicMock()
        # test logic with dependency mocked
        ...


if __name__ == "__main__":
    unittest.main()
```

Use `unittest.mock.patch` / `MagicMock` to mock dependencies and isolate the logic under test. If the module fails to import on CPU CI (e.g., imports `torch` or CUDA ops at module level), use `sys.modules` stubs to make the import succeed. See existing tests in `test/registered/unit/` for examples.

**Quality bar** — test real logic (validation boundaries, state transitions, error paths, branching, etc.). Skip tests that just verify Python itself works (e.g., "does calling an abstract method raise `NotImplementedError`?", "does a dataclass store the field I assigned?"). Consolidate repetitive patterns into parameterized tests. No production code changes in test PRs.

### E2E test (small model, server needed)

```python
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")


class TestMyFeature(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--arg1", "value1"],  # feature-specific args
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_basic_functionality(self):
        response = requests.post(
            self.base_url + "/generate",
            json={"text": "Hello", "sampling_params": {"max_new_tokens": 32}},
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=3)
```

### E2E test (8B model, server needed, performance)

```python
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="stage-b-test-1-gpu-large")


class TestMyFeaturePerf(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_latency(self):
        start = time.perf_counter()
        response = requests.post(
            self.base_url + "/generate",
            json={"text": "Hello", "sampling_params": {"max_new_tokens": 128}},
        )
        elapsed = time.perf_counter() - start
        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 5.0, "Latency exceeded threshold")


if __name__ == "__main__":
    unittest.main(verbosity=3)
```

---

## Server Fixture Reuse

For tests that only need a standard server, inherit from `DefaultServerBase` and override class attributes:

```python
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

class TestMyFeature(DefaultServerBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    other_args = ["--enable-my-feature"]

    def test_something(self):
        ...
```

Available fixtures in `python/sglang/test/server_fixtures/`:

| Fixture | Use case |
|---------|----------|
| `DefaultServerBase` | Standard single-server tests |
| `EagleServerBase` | EAGLE speculative decoding |
| `PDDisaggregationServerBase` | Disaggregated prefill/decode |
| `MMMUServerBase` | Multimodal VLM tests |

---

## CI Registration

Every CI-discovered test file must call a registration function at module level:

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-1-gpu-small")
```

Parameters:
- `est_time`: estimated runtime in seconds (used for CI partitioning)
- `suite`: which CI suite to run in (see suite table above)
- `nightly=True`: for nightly-only tests (default `False` = per-commit)
- `disabled="reason"`: temporarily disable with explanation

Only add `register_amd_ci` / `register_cpu_ci` when the test exercises backend-specific code paths.

For JIT kernel files:
- Place correctness tests in `python/sglang/jit_kernel/tests/`
- Place benchmarks in `python/sglang/jit_kernel/benchmark/`
- Use `register_cuda_ci` with kernel suites such as `stage-b-kernel-unit-1-gpu-large`, `stage-b-kernel-benchmark-1-gpu-large`, and optionally `nightly-kernel-1-gpu`
- Keep `est_time` and `suite` as literal values because `test/run_suite.py` collects them by AST parsing

---

## Test Placement

```
test/
├── registered/          # CI tests (auto-discovered by run_suite.py)
│   ├── unit/            # No server / engine launch (see test/registered/unit/README.md)
│   ├── kernels/         # CUDA kernel correctness (no server, GPU required)
│   ├── sampling/        # test_penalty.py, test_sampling_params.py ...
│   ├── sessions/        # test_session_control.py ...
│   ├── openai_server/   # basic/, features/, validation/ ...
│   ├── spec/            # eagle/, utils/ ...
│   ├── models/          # model-specific accuracy tests
│   ├── perf/            # performance benchmarks
│   └── <category>/      # create new category if needed
├── manual/              # Non-CI: debugging, one-off, manual verification
└── run_suite.py         # CI runner (scans registered/ plus jit_kernel test/benchmark files)

python/sglang/jit_kernel/
├── tests/               # JIT kernel correctness tests (CI-discovered by test/run_suite.py)
└── benchmark/           # JIT kernel benchmarks (CI-discovered by test/run_suite.py)
```

**Decision rule** (see also `test/registered/README.md`):
- Component logic, no server → `registered/unit/`
- JIT kernel correctness / benchmarks → `python/sglang/jit_kernel/tests/` or `python/sglang/jit_kernel/benchmark/`
- Other kernel correctness → `registered/kernels/`
- Server needed → `registered/<category>/`
- Local debugging → `manual/`

---

## Key Utilities

```python
from sglang.test.test_utils import (
    CustomTestCase,              # base class with retry logic
    popen_launch_server,         # launch server subprocess
    DEFAULT_URL_FOR_TEST,        # auto-configured base URL
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,  # 600s default
    run_bench_serving,           # benchmark helper (launch + bench)
)
from sglang.srt.utils import kill_process_tree  # cleanup server
```

---

## Checklist

Before submitting a test:

- [ ] Inherits from `CustomTestCase` (not `unittest.TestCase`)
- [ ] Has `register_*_ci(...)` call at module level
- [ ] Placed in `test/registered/<category>/`, unless this is a JIT kernel test/benchmark
- [ ] JIT kernel work: files live in `python/sglang/jit_kernel/tests/` or `python/sglang/jit_kernel/benchmark/`
- [ ] Backend-independent tests: `register_cuda_ci` only + smallest model
- [ ] Logic that doesn't need a server / engine launch → unit test in `registered/unit/` (see Unit Tests section)
- [ ] `setUpClass` launches server, `tearDownClass` kills it (if server-based)
- [ ] `tearDownClass` is defensive — uses `hasattr`/null checks before accessing resources that may not have been allocated
- [ ] Has `if __name__ == "__main__": unittest.main()`
- [ ] `est_time` is reasonable (measure locally)
