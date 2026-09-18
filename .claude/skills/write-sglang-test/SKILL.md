---
name: write-sglang-test
description: Guide for writing SGLang CI/UT tests. Covers CustomTestCase, CI registration, server fixtures, model selection, mock testing, and test placement. Always read test/README.md for the full CI layout, how to run tests, and extra tips. Use when creating new tests, adding CI test cases, writing unit tests, or when the user asks to add tests for SGLang features.
---

# Writing SGLang CI / UT Tests

This skill covers **how to write and register tests**. For CI pipeline internals (stage ordering, fast-fail, gating, partitioning, debugging CI failures), see the [CI workflow guide](../ci-workflow-guide/SKILL.md). Whether a case is worth adding at all is decided by [`unit-test-admission`](../../rules/unit-test-admission.md) — read it before writing the case, not after.

## Core Rules

1. **Always use `CustomTestCase`** — never raw `unittest.TestCase`. It ensures `tearDownClass` runs even when `setUpClass` fails, preventing resource leaks in CI.
2. **`tearDownClass` must shut the server down gracefully** — call `terminate_and_kill_process_tree(cls.process)`, never a bare `kill_process_tree`. SIGKILL alone skips the server's userspace cleanup and leaves its GPU memory charged to the dead process; the next class then OOMs while loading weights. Keep it defensive too: `hasattr`/null checks before accessing resources (e.g. `cls.process`) that `setUpClass` may not have finished allocating.
3. **Place non-kernel tests in `test/registered/<kind>/<subsystem>/`** — `<kind>` is `unit`, `e2e`, `accuracy`, `perf`, or `stress`; kernel tests use `test/registered/kernels/{ops,benchmark}/<group>/`; hardware belongs in registrations, not directory names
4. **Reuse server fixtures** — inherit from `DefaultServerBase` or write `setUpClass`/`tearDownClass` with `popen_launch_server`
5. **Mock boundaries, not SGLang behavior** — mock slow or external dependencies only when the assertion still checks an observable result, state transition, or error. A test whose evidence is only `assert_called*` mirrors its mock and is not admissible. Launch a real server only when inference results or lifecycle behavior are the contract under test.

> **Existing files are not the reference.** About 290 test files still call the bare
> `kill_process_tree(cls.process.pid)`, against 43 on the current helper. They predate
> rule 2 and are being migrated, so grepping the repo for a teardown pattern finds the
> wrong one roughly seven times out of eight. The same goes for `register_cuda_ci(suite=...)`:
> four files still pass it, all of them under `test/registered/stress/`.

```python
# Bad:  kill_process_tree(cls.process.pid)           # SIGKILL only; GPU memory lingers
# Good: terminate_and_kill_process_tree(cls.process)

# Bad:  register_cuda_ci(est_time=80, suite="base-b-test-1-gpu-small")
# Good: register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")
```

JIT kernel notes:
- If the task is adding or updating code under `python/sglang/kernels/jit/`, prefer the `add-jit-kernel` skill first.
- JIT kernel correctness tests use `test/registered/kernels/ops/<group>/test_*.py`.
- JIT kernel benchmarks use `test/registered/kernels/benchmark/<group>/bench_*.py`.
- Those files are executed by `test/run_suite.py` through dedicated kernel suites (`base-b-kernel-*`); a `register_*_ci(...)` call placed under `python/sglang/` is rejected by the `check-no-registered-tests-in-package` pre-commit hook.

---

## Model & Backend Selection

| Scenario | Model | CI Registration | Suite |
|----------|-------|-----------------|-------|
| **Unit tests** (no server / engine launch) | None | `register_cpu_ci` | `base-a-test-cpu` |
| **Common / backend-independent** (middleware, abort, routing, config, arg parsing) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` only | `base-b-test-1-gpu-small` |
| **Model-agnostic functionality** (sampling, session, OpenAI API features) | `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (1B) | `register_cuda_ci` (+ AMD if relevant) | `base-b-test-1-gpu-small` |
| **General performance** (single node, no spec/DP/parallelism) | `DEFAULT_MODEL_NAME_FOR_TEST` (8B) | `register_cuda_ci` | `base-b-test-1-gpu-large` |
| **Bigger features** (spec, DP, TP, disaggregation) | Case by case | Case by case | See **Choosing a Suite** below |

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

### Naming Conventions

A per-commit suite name is **generated** from registration metadata as `{stage}-test-{runner_config}` — you don't hand-write it:

- **`stage`** — the CI stage (e.g. `base-b`, `base-b-kernel-unit`, `base-c`).
- **`runner_config`** — a runner-pool key from `scripts/ci/runner_configs.yml`, which maps it to the physical runner label (so `1-gpu-large` runs on `1-gpu-h100`). AMD/NPU use their own keys (e.g. `amd`).
- **Suite** — `register_cuda_ci(stage="base-b", runner_config="1-gpu-small")` → `base-b-test-1-gpu-small`, the name you pass to `run_suite.py --suite`. The `-test-` is just the connector; never put it in `register_*_ci`.

> CUDA nightly uses the same shape with `stage="nightly"` (e.g. `stage="nightly", runner_config="1-gpu-large"` → `nightly-test-1-gpu-large`) and **no** `nightly=True` — the stage name carries the cadence, and setting the flag makes the test silently never run. Legacy single-string `suite=` is left only for `stress` and some AMD/CPU/NPU pools.

### All CI Suites

Do not work from a list copied into this file; it goes stale silently. Read the
current one:

```bash
grep -n "_SUITES = {" test/run_suite.py   # PER_COMMIT_SUITES, NIGHTLY_SUITES, OTHER_SUITES
cat scripts/ci/runner_configs.yml         # runner_config -> physical runner label
```

`scripts/ci/runner_configs.yml` calls itself the single source of truth for the
`runner_config` field, and `run_suite.py` is what actually dispatches, so those two
files settle any disagreement with prose anywhere else.

Nightly suites live in `NIGHTLY_SUITES` and run via `nightly-test-nvidia.yml`,
`nightly-test-amd.yml`, and `nightly-test-npu.yml`, not `pr-test.yml`. CUDA nightly is
named `nightly-test-{runner_config}` — one suite per machine type, holding everything
that runs nightly on it, with `auto_partition` splitting the work. There is no
per-purpose split; kernel, eval, perf, and precision all share their machine's suite.

> **Note**: Multimodal diffusion uses `python/sglang/multimodal_gen/test/run_suite.py`, not `test/run_suite.py`.

### Choosing a Suite

Use the lightest suite that meets your test's needs:

- **No GPU required** → `base-a-test-cpu`
- **Most small GPU tests** → `base-b-test-1-gpu-small` (default choice)
- **Need H100 memory or Hopper features** → `base-b-test-1-gpu-large`
- **JIT kernel correctness** → `base-b-kernel-unit-test-1-gpu-large`
- **JIT kernel correctness for B200 / SM100 paths** → `base-b-kernel-unit-test-4-gpu-b200`
- **JIT kernel benchmarks** → `base-b-kernel-benchmark-test-1-gpu-large`
- **Multi-GPU** → only when the test actually needs multiple GPUs

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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
# Unit tests are CPU-only. GPU operator tests belong under `kernel/`.

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

Use `unittest.mock.patch` / `MagicMock` only at dependency boundaries. Assert the
resulting value, state, protocol output, or error—not merely that the mock was
called. If the module transitively imports GPU-only packages (e.g. `sgl_kernel`),
they can be stubbed so the test runs on CPU CI. Do not modify `sys.modules` at
module level—use `patch.dict` (as a class decorator or with `start`/`stop`) to
ensure cleanup and avoid cross-test pollution. See
`test/registered/unit/README.md` for details and examples.

**Quality bar** — test real logic (validation boundaries, state transitions, error paths, branching, etc.). Skip tests that just verify Python itself works (e.g., "does calling an abstract method raise `NotImplementedError`?", "does a dataclass store the field I assigned?"). Consolidate repetitive patterns into parameterized tests. No production code changes in test PRs.

### E2E test (small model, server needed)

```python
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


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
            terminate_and_kill_process_tree(cls.process)

    def test_basic_functionality(self):
        response = requests.post(
            self.base_url + "/generate",
            json={"text": "Hello", "sampling_params": {"max_new_tokens": 32}},
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=3)
```

Copy the `tearDownClass` above verbatim. Most existing E2E files still show the bare
`kill_process_tree`; that form is being migrated out and must not be reproduced.

### E2E test (8B model, server needed, performance)

```python
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")


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
            terminate_and_kill_process_tree(cls.process)

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
from sglang.test.ci.ci_register import (
    register_cuda_ci,
    register_amd_ci,
    register_cpu_ci,
    register_npu_ci,
)

# Per-commit test (small 1-gpu, runs on 5090)
register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")

# Per-commit test (large 1-gpu, runs on H100)
register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

# Nightly-only test (same shape as per-commit, stage is just "nightly")
register_cuda_ci(est_time=200, stage="nightly", runner_config="1-gpu-large")

# Multi-backend test (only when testing backend-specific code paths)
register_cuda_ci(est_time=80, stage="base-a", runner_config="1-gpu-small")
register_amd_ci(est_time=120, suite="stage-a-test-1-gpu-small-amd")
register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)

# Temporarily disabled test
register_cuda_ci(
    est_time=80, stage="base-b", runner_config="1-gpu-small", disabled="flaky - see #12345"
)
```

Parameters:
- `est_time`: estimated runtime in seconds (used for CI partitioning)
- `stage` + `runner_config`: the canonical pair for CUDA; the suite name is generated from them (see Naming Conventions)
- `suite`: legacy single-string form. Only `stress` and some AMD/CPU/NPU pools still take it; `register_cpu_ci(suite="base-a-test-cpu")` is correct and is not being migrated
- `nightly=True`: legacy cadence flag, for non-CUDA nightly suites only. CUDA nightly uses `stage="nightly"` and must leave this unset
- `disabled="reason"`: temporarily disable with explanation

**Key principle**: Only add `register_amd_ci` / `register_npu_ci` when the test exercises backend-specific code paths. Common E2E tests just need `register_cuda_ci` — duplicating across backends wastes CI time.

### JIT Kernel Registration

`run_suite.py` discovers every `test/registered/**/*.py`, JIT kernel files included.
They are ordinary registered tests; only their stage differs:

```python
from sglang.test.ci.ci_register import register_cuda_ci

# Correctness tests in test/registered/kernels/ops/<group>/
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="8-gpu-h200")

# Benchmarks in test/registered/kernels/benchmark/<group>/
register_cuda_ci(est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large")

# Optional nightly registration — same form, stage is just "nightly"
register_cuda_ci(est_time=120, stage="nightly", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, stage="nightly", runner_config="8-gpu-h200")
```

Every call generates a suite named `{stage}-test-{runner_config}`, e.g. `base-b-kernel-unit-test-1-gpu-large` and `nightly-test-1-gpu-large`. Keep `est_time`, `stage`, `runner_config`, and `suite` as **literal values** — `run_suite.py` collects them by AST parsing.

---

## Test Placement

```
test/
├── registered/          # CI tests (auto-discovered by run_suite.py)
│   ├── unit/<subsystem>/      # CPU-only; no server or model weights
│   ├── kernel/<group>/        # accelerator operator correctness/benchmarks
│   ├── e2e/<subsystem>/       # engine/server integration
│   ├── accuracy/<family>/     # scheduled eval floors
│   ├── perf/<family>/         # scheduled latency/throughput contracts
│   └── stress/<subsystem>/    # stress/weekly coverage
├── manual/              # Non-CI: debugging, one-off, manual verification
└── run_suite.py         # CI runner (globs test/registered/**/*.py; nothing outside it)

python/sglang/kernels/jit/   # implementation + test-only helpers, never registered tests
```

A `register_*_ci(...)` under `python/sglang/` is rejected by the
`check-no-registered-tests-in-package` pre-commit hook.

**Decision rule** (see also `test/registered/README.md`):
- CPU component logic, no server → `registered/unit/<subsystem>/`
- JIT kernel correctness → `registered/kernels/ops/<group>/`
- JIT kernel benchmarks → `registered/kernels/benchmark/<group>/`
- Other accelerator operator correctness → `registered/kernels/ops/<group>/`
- Server needed → `registered/e2e/<subsystem>/`
- Eval floor / performance contract → `registered/{accuracy,perf}/<family>/`
- Local debugging → `manual/`

---

## Eval Accuracy Mixins

**Design philosophy**: Most test files don't care about eval logic — they only need a "does this feature break model output quality?" sanity check. The mixin pattern separates **what to test** (threshold) from **how to test** (run_eval, assertions, CI summary). Test classes declare thresholds as class attributes; the mixin provides the `test_*` method. Override when you need extra assertions (e.g. EAGLE accept length).

Available mixins in `python/sglang/test/kits/eval_accuracy_kit.py`: `MMLUMixin`, `HumanEvalMixin`, `MGSMEnMixin`, `GSM8KMixin`. Can be combined freely. Read the source for attrs and defaults.

```python
class TestMyFeature(CustomTestCase, MMLUMixin):
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32
    # test_mmlu is inherited — no code needed
```

---

## Key Utilities

```python
from sglang.test.test_utils import (
    CustomTestCase,              # base class with retry logic
    popen_launch_server,         # launch server subprocess
    terminate_and_kill_process_tree,    # SIGTERM, then SIGKILL, then wait for
                                        # the GPU memory to come back
    DEFAULT_URL_FOR_TEST,        # auto-configured base URL
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,  # 600s default
    run_bench_serving,           # benchmark helper (launch + bench)
)
```

---

## Checklist

Before submitting a test:

- [ ] Inherits from `CustomTestCase` (not `unittest.TestCase`)
- [ ] Has `register_*_ci(...)` call at module level
- [ ] Placed in `test/registered/<kind>/<subsystem>/`
- [ ] JIT kernel work: correctness tests live in `test/registered/kernels/ops/<group>/`, benchmarks live in `test/registered/kernels/benchmark/<group>/`, and only test-only helpers stay under `python/sglang/kernels/jit/`
- [ ] Backend-independent tests: `register_cuda_ci` only + smallest model
- [ ] Logic that doesn't need a server / engine launch → unit test in `registered/unit/` (see Unit Tests section)
- [ ] `tearDownClass` is defensive — uses `hasattr`/null checks before accessing resources that may not have been allocated
- [ ] Every case answers "what future diff turns this red?" — see [`unit-test-admission`](../../rules/unit-test-admission.md)
- [ ] `est_time` is reasonable (measure locally)

Run these against the new file and paste the output rather than self-attesting:

```bash
f=<your new test file>
grep -n "kill_process_tree" $f          # every hit must be terminate_and_kill_process_tree
grep -n "register_.*_ci(" $f            # CUDA: stage= + runner_config=, never suite=
grep -n "CustomTestCase\|unittest.main" $f   # both must appear
```
