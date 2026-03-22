---
name: write-sglang-test
description: Guide for writing SGLang CI/UT tests following project conventions. Covers CustomTestCase, CI registration, server fixtures, model selection, mock testing, and test placement. Use when creating new tests, adding CI test cases, writing unit tests, or when the user asks to add tests for SGLang features.
---

# Writing SGLang CI / UT Tests

## Core Rules

1. **Always use `CustomTestCase`** — never raw `unittest.TestCase`
2. **Place tests in `test/registered/<category>/`** — only use `test/manual/` for debugging / non-CI tests
3. **Reuse server fixtures** — inherit from `DefaultServerBase` or write `setUpClass`/`tearDownClass` with `popen_launch_server`
4. **Prefer mock over real server** — when testing logic that doesn't need inference (middleware, request routing, config validation, argument parsing), use `unittest.mock.patch` / `MagicMock`. Only launch a real server when the test genuinely needs inference results or server lifecycle behavior.

---

## Model & Backend Selection

| Scenario | Model | CI Registration | Suite |
|----------|-------|-----------------|-------|
| **Common / backend-independent** (middleware, abort, routing, config, arg parsing) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` only | `stage-b-test-small-1-gpu` |
| **Model-agnostic functionality** (sampling, session, OpenAI API features) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` (+ AMD if relevant) | `stage-b-test-small-1-gpu` |
| **General performance** (single node, no spec/DP/parallelism) | `DEFAULT_MODEL_NAME_FOR_TEST` (8B) | `register_cuda_ci` | `stage-b-test-large-1-gpu` |
| **Bigger features** (spec, DP, TP, disaggregation) | Case by case | Case by case | See suite table below |

**Key principle**: Do NOT add `register_amd_ci` / `register_cpu_ci` unless the test specifically exercises AMD/ROCm or CPU-specific code paths. Common tests just need any GPU to run — duplicating across backends wastes CI time with no extra coverage.

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
| `stage-b-test-small-1-gpu` | 1× 5090 (32GB) | Small model tests |
| `stage-b-test-large-1-gpu` | 1× H100 (80GB) | 8B model tests |
| `stage-b-test-large-2-gpu` | 2× H100 | TP=2 tests |
| `stage-c-test-4-gpu-h100` | 4× H100 | TP=4 / EP tests |
| `stage-c-test-8-gpu-h200` | 8× H200 | Large-scale multi-GPU |
| `nightly-1-gpu` | 1 GPU | Nightly-only |
| `nightly-8-gpu` | 8 GPU | Nightly-only |

---

## Test File Templates

### Mock test (no server needed)

Use this for testing logic that doesn't require inference — fastest, most deterministic.

```python
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")


class TestMyLogic(CustomTestCase):
    def test_config_validation(self):
        """Test that invalid config raises ValueError."""
        from sglang.srt.server_args import ServerArgs

        with self.assertRaises(ValueError):
            ServerArgs.from_cli_args(["--invalid-flag"])

    @patch("sglang.srt.utils.common.some_function")
    def test_middleware_behavior(self, mock_fn):
        mock_fn.return_value = MagicMock(status_code=200)
        # test middleware logic without launching a server
        ...


if __name__ == "__main__":
    unittest.main(verbosity=3)
```

### Integration test (small model, server needed)

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

### Performance test (8B model)

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
```

Parameters:
- `est_time`: estimated runtime in seconds (used for CI partitioning)
- `suite`: which CI suite to run in (see suite table above)
- `nightly=True`: for nightly-only tests (default `False` = per-commit)
- `disabled="reason"`: temporarily disable with explanation

Only add `register_amd_ci` / `register_cpu_ci` when the test exercises backend-specific code paths.

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
- [ ] Backend-independent tests: `register_cuda_ci` only + smallest model
- [ ] Logic that doesn't need inference uses `unittest.mock.patch` instead of a real server
- [ ] `setUpClass` launches server, `tearDownClass` kills it (if server-based)
- [ ] Has `if __name__ == "__main__": unittest.main(verbosity=3)`
- [ ] `est_time` is reasonable (measure locally)
