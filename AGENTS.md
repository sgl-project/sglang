# AGENTS.md - AI Coding Agent Guidelines for SGLang

SGLang is a high-performance serving framework for large language models and multimodal models.

## Project Structure

- `python/sglang/srt/` — Server runtime: `models/` (130+ architectures), `layers/` (attention, MoE, quantization), `managers/` (scheduler, tokenizer), `sampling/`, `speculative/`, `lora/`, `utils/`
- `python/sglang/lang/` — Frontend DSL
- `python/sglang/jit_kernel/` — JIT-compiled kernels
- `sgl-kernel/` — CUDA/C++ kernel package (separate PyPI package)
- `sgl-model-gateway/` — Rust model gateway / load balancer
- `test/` — Integration and unit tests
- `benchmark/` — Performance benchmarks

## Build Commands

```bash
pip install -e "python[dev]"                    # Install from source (editable)
pip install -e "python[dev,diffusion,tracing]"  # With optional extras

# Linting and formatting (run twice if first run auto-fixes)
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

## Test Commands

Main tests use **unittest**; kernel tests use **pytest**.

```bash
# Single test file
python3 test/srt/test_srt_endpoint.py

# Single test method
python3 test/srt/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode
# or:
python3 -m unittest test.srt.test_srt_endpoint.TestSRTEndpoint.test_simple_decode

# Test suite (legacy, defined in test/srt/run_suite.py)
python3 test/srt/run_suite.py --suite per-commit-1-gpu

# Test suite (new registry system, defined in test/run_suite.py)
python3 test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu

# Kernel tests (from sgl-kernel/ directory)
cd sgl-kernel && pytest tests/
pytest tests/test_activation.py              # Single kernel test file
```

Legacy suites: `per-commit-1-gpu`, `per-commit-2-gpu`, `per-commit-4-gpu`, `quantization_test`.
New suites (CUDA): `stage-a-test-1`, `stage-b-test-small-1-gpu`, `stage-b-test-large-1-gpu`, `stage-b-test-large-2-gpu`, `stage-c-test-large-4-gpu`.

## Code Style

### Toolchain

- **Formatter**: Black (v24.10.0)
- **Import sorting**: isort (v5.13.2, profile=black, first-party=`sglang`)
- **Linter**: Ruff (v0.11.7, rules: F401 unused imports, F821 undefined names)
- **Spell checker**: codespell (v2.4.1)
- **C++/CUDA**: clang-format (v18.1.8, Google style, 2-space indent, 120 col limit)

### File Header

All Python source files must include the Apache 2.0 license header (`# Copyright 2023-2024 SGLang Team` ... `# ==============================================================================`).

### Import Order

Three groups separated by blank lines, each alphabetized internally:

```python
from __future__ import annotations           # Always first when used

import logging                               # 1. Standard library
from typing import TYPE_CHECKING, Optional

import torch                                 # 2. Third-party

from sglang.srt.utils.common import get_device  # 3. Local (sglang.*)

if TYPE_CHECKING:                            # 4. Type-checking-only imports
    from sglang.srt.server_args import ServerArgs
```

### Type Annotations

- Always type-hint function signatures and return types.
- Use `from __future__ import annotations` for forward references.
- Use `TYPE_CHECKING` guard for imports that would cause circular deps or are heavy.

### Naming Conventions

| Entity | Convention | Examples |
|--------|-----------|----------|
| Functions/methods | `snake_case` | `get_token_ids`, `run_batch` |
| Classes | `PascalCase` | `TokenizerManager`, `LlamaForCausalLM` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_TIMEOUT`, `FP8_E4M3_MAX` |
| Files | `snake_case.py` | `server_args.py`, `model_runner.py` |
| Private/internal | `_leading_underscore` | `_ModelRegistry`, `_is_hip` |
| Test files/classes/methods | `test_<feature>.py`, `Test<Feature>`, `test_<scenario>` |

### Logging

```python
logger = logging.getLogger(__name__)  # Set up immediately after imports
logger.warning(f"Something happened: {detail}")  # Use f-strings
```

### Error Handling

- Raise `ValueError` for bad inputs, `RuntimeError` for system issues.
- Use `assert` for internal invariants only.
- Catch specific exceptions; log with context before re-raising.

### Environment Variables

Centralized in `python/sglang/srt/environ.py` via descriptors. Never use scattered `os.getenv()`:

```python
from sglang.srt.environ import envs
value = envs.SGLANG_SOME_FLAG.get()  # Never use envs.X directly as bool
```

## Performance Guidelines

- **No device sync in hot paths**: Avoid `tensor.item()`, `tensor.cpu()` during inference.
- **Cache runtime checks**: Compute once and store as `bool` if constant across layers.
- **Vectorize**: Prefer batch tensor ops over Python loops.
- **File size limit**: Keep files under 2,000 lines; split if larger.

## Test Writing

- Use `CustomTestCase` from `sglang.test.test_utils` (adds retry logic).
- Launch servers in `setUpClass`; tear down in `tearDownClass` with `kill_process_tree`.
- Use `DEFAULT_SMALL_MODEL_NAME_FOR_TEST` (`Llama-3.2-1B-Instruct`) for fast tests.
- Each test method should test one scenario. Keep test files under 500 seconds.
- End every test file with `if __name__ == "__main__": unittest.main()`.
- New tests must be registered in `test/srt/run_suite.py` (alphabetical order).

```python
class TestFeature(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST, DEFAULT_URL_FOR_TEST)
    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
    def test_specific_scenario(self):
        pass
```

## Adding New Hardware / Features

- Prefer adding new files over modifying existing ones (e.g., `allocator_ascend.py`).
- In `if/else` blocks, put the common path (NVIDIA/existing) first.
- Don't drastically restructure existing code.

## Updating sgl-kernel

sglang and sgl-kernel are separate PyPI packages. Kernel changes require multiple PRs:

1. PR to update kernel source (without calling it from sglang yet).
2. Bump `sgl-kernel` version in `sgl-kernel/pyproject.toml` (triggers PyPI release).
3. Update `sgl-kernel` version in `python/pyproject.toml` and add caller code.
