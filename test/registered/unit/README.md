# Unit Tests

Component-level tests that do **not** launch a server or load model weights.
Tests can use CPU or GPU — the key criterion is **no server process**.

## Quick Start

1. Find the source file under `python/sglang/srt/`.
2. Create the corresponding test here, mirroring the source tree:
   ```
   srt/mem_cache/radix_cache.py       →  unit/mem_cache/test_radix_cache.py
   srt/sampling/sampling_params.py    →  unit/sampling/test_sampling_params.py
   ```
3. Register for CI at the **top of the file** (after imports, before test classes):
   ```python
   from sglang.test.ci.ci_register import register_cpu_ci
   register_cpu_ci(est_time=5, suite="stage-a-cpu-only")
   # or: register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
   ```
4. Run locally:
   ```bash
   pytest test/registered/unit/ -v            # all unit tests
   pytest test/registered/unit/mem_cache/ -v  # one module
   ```
5. Run with coverage:
   ```bash
   # summary
   pytest test/registered/unit/ --cov --cov-config=.coveragerc -v

   # PR incremental check (require ≥60% on changed lines)
   pytest test/registered/unit/ --cov --cov-config=.coveragerc --cov-report=xml
   diff-cover coverage.xml --compare-branch=origin/main --fail-under=60
   ```

## Example

```python
"""Unit tests for <module> — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest

from sglang.srt.<module> import TargetClass
from sglang.test.test_utils import CustomTestCase


class TestTargetClass(CustomTestCase):
    def test_basic_behavior(self):
        obj = TargetClass(...)
        self.assertEqual(obj.method(), expected)


if __name__ == "__main__":
    unittest.main()
```

## Rules

- **No** `popen_launch_server()` or `Engine(...)`.
- **No** model weight loading.
- Use `CustomTestCase` (from `sglang.test.test_utils`, adds CI retry).
- Use `unittest.mock` for dependencies that are expensive to construct.
