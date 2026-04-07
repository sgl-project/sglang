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
   register_cpu_ci(est_time=5, suite="stage-a-test-cpu")
   # or: register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
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

## Examples

### Basic unit test

```python
"""Unit tests for <module> — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

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

### Stubbing GPU-only imports for CPU tests

Some modules (e.g. `scheduler.py`, `io_struct.py`) transitively import packages like
`sgl_kernel` that require a GPU to initialize. To run pure-mock tests against these
modules on CPU-only CI, stub the problematic package **before** importing it.

`maybe_stub_sgl_kernel()` in `test_utils.py` does this for `sgl_kernel`: it's a no-op
on GPU machines, and on CPU it installs a `sys.meta_path` finder that auto-creates empty
stub modules for all `sgl_kernel.*` submodules.

```python
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.managers.io_struct import FlushCacheReqInput
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")
```

The same pattern (`sys.meta_path` finder) can be applied to other GPU-only packages.
See `maybe_stub_sgl_kernel()` in `python/sglang/test/test_utils.py` for the
implementation. Do not directly mutate `sys.modules` at module level — pytest
imports all test files before running any, so such mutations pollute the entire
process. If you must stub, use `patch.dict("sys.modules", ...)` with proper cleanup.

## Rules

- **No** `popen_launch_server()` or `Engine(...)`.
- **No** model weight loading.
- Use `CustomTestCase` (from `sglang.test.test_utils`, adds CI retry).
- Use `unittest.mock` for dependencies that are expensive to construct.
