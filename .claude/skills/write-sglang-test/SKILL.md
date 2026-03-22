---
name: write-sglang-test
description: Guide for writing SGLang CI/UT tests following project conventions. Covers CustomTestCase, CI registration, server fixtures, model selection, and test placement. Use when creating new tests, adding CI test cases, writing unit tests, or when the user asks to add tests for SGLang features.
---

# Writing SGLang CI / UT Tests

## Core Rules

1. **Always use `CustomTestCase`** — never raw `unittest.TestCase`
2. **Place tests in `test/registered/<category>/`** — only use `test/manual/` for debugging / non-CI tests
3. **Reuse server fixtures** — inherit from `DefaultServerBase` or write `setUpClass`/`tearDownClass` with `popen_launch_server`
4. **Smallest model for model-agnostic functionality** — use `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (Llama-3.2-1B-Instruct) for basic features that don't depend on model size
5. **8B for general performance** — use `DEFAULT_MODEL_NAME_FOR_TEST` (Llama-3.1-8B-Instruct, single-node) for performance tests that don't involve spec / DP / parallelism
6. **Bigger features → discuss case by case** — spec, DP attention, tensor/pipeline parallelism etc. may need multi-GPU suites and specific models
7. **Common tests: CUDA-only + smallest model** — tests for backend-independent functionality (HTTP middleware, abort, API routing, config parsing, argument validation, etc.) should **only** call `register_cuda_ci` and use `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`. These tests don't care about model capability or GPU backend — they just need a running server. Do NOT add `register_amd_ci` / `register_cpu_ci` unless the test specifically exercises AMD/ROCm or CPU-specific code paths.
8. **Prefer `unittest.mock.patch` over launching a real server** — when testing logic that doesn't require end-to-end inference (middleware behavior, request routing, config validation, argument parsing), use `unittest.mock.patch` / `MagicMock` to isolate the unit under test. Only launch a real server (`popen_launch_server`) when the test genuinely needs inference results or server lifecycle behavior. Mock-based tests are faster, more deterministic, and don't consume GPU CI time.

---

## Test File Template

### Functional correctness test (small model)

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

register_cuda_ci(est_time=60, suite="stage-b-test-small-1-gpu")


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

### General performance test (8B model, single node, no spec/DP/parallelism)

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

register_cuda_ci(est_time=300, suite="stage-b-test-large-1-gpu")


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

Every test file in `test/registered/` **must** call a registration function at module level:

```python
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-test-small-1-gpu")

# Only add register_amd_ci / register_cpu_ci when the test exercises
# backend-specific code paths (e.g., AMD kernel, ROCm integration).
# Common tests (middleware, abort, routing, config) should NOT register
# non-CUDA backends — it wastes CI time with no extra coverage.
```

Parameters:
- `est_time`: estimated runtime in seconds (used for CI partitioning)
- `suite`: which CI suite to run in (see below)
- `nightly=True`: for nightly-only tests (default `False` = per-commit)
- `disabled="reason"`: temporarily disable with explanation

### Suite selection guide

**Default cases (1 GPU):**

| Scenario | Model | Suite |
|----------|-------|-------|
| Model-agnostic basic functionality | 1B (smallest) | `stage-b-test-small-1-gpu` |
| General performance (no spec/DP/parallelism) | 8B | `stage-b-test-large-1-gpu` |

**Bigger features (case by case):**

| Scenario | Suite |
|----------|-------|
| 2 GPU (e.g. TP=2) | `stage-b-test-large-2-gpu` |
| 4 GPU (H100) | `stage-c-test-4-gpu-h100` |
| 8 GPU (H200) | `stage-c-test-8-gpu-h200` |
| Nightly, 1 GPU | `nightly-1-gpu` |
| Nightly, 8 GPU | `nightly-8-gpu` |

For spec, DP attention, parallelism, disaggregation, etc., discuss with the team to determine the appropriate suite and GPU configuration.

---

## Model Constants

All defined in `python/sglang/test/test_utils.py`:

| Constant | Model | When to use |
|----------|-------|-------------|
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` | Llama-3.2-1B-Instruct | Model-agnostic basic functionality |
| `DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE` | Llama-3.2-1B | Base (non-instruct) model tests |
| `DEFAULT_MODEL_NAME_FOR_TEST` | Llama-3.1-8B-Instruct | General performance (single node) |
| `DEFAULT_MOE_MODEL_NAME_FOR_TEST` | Mixtral-8x7B-Instruct | MoE-specific tests |
| `DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST` | — | Embedding tests |
| `DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST` | — | Vision-language tests |

---

## Test Placement

```
test/
├── registered/          # CI tests (auto-discovered by run_suite.py)
│   ├── sampling/        # test_penalty.py, test_sampling_params.py ...
│   ├── sessions/        # test_session_control.py ...
│   ├── openai_server/   # basic/, features/, validation/ ...
│   ├── spec/            # eagle/, utils/ ...
│   ├── models/          # model-specific accuracy tests
│   ├── perf/            # performance benchmarks
│   └── <category>/      # create new category if needed
├── manual/              # Non-CI: debugging, one-off, manual verification
└── run_suite.py         # CI runner (scans registered/ only)
```

**Decision rule**: if the test should run in CI → `registered/`. If it's for local debugging or requires special hardware not in CI → `manual/`.

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
- [ ] Placed in `test/registered/<category>/`
- [ ] Model selection: smallest for model-agnostic features, 8B for general perf, case-by-case for other complex features
- [ ] `setUpClass` launches server, `tearDownClass` kills it
- [ ] Has `if __name__ == "__main__": unittest.main(verbosity=3)`
- [ ] `est_time` is reasonable (measure locally)
- [ ] Backend-independent tests only register `register_cuda_ci` (no unnecessary AMD/CPU registration)
- [ ] Logic that doesn't need inference uses `unittest.mock.patch` instead of launching a server
