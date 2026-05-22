# Registered Tests

Tests under this directory are auto-discovered by `run_suite.py` via CI registration decorators.

## Where Should I Put My New Test?

### No server / engine launch required

| What you're testing | Directory | Requires |
|---|---|---|
| Component logic in isolation (cache, scheduler, config, parser, etc.) | [`unit/<module>/`](unit/README.md) | CPU or GPU |
| CUDA kernel correctness | `kernels/` | GPU |

### Server / engine launch required (E2E)

| What you're testing | Directory | Requires |
|---|---|---|
| Model inference correctness | `models/`, `4-gpu-models/`, `8-gpu-models/` | GPU |
| Feature-specific (OpenAI API, LoRA, speculative, distributed, VLM, etc.) | `openai_server/`, `lora/`, `spec/`, `distributed/`, ... | GPU |
| Benchmarks (performance, accuracy, stress) | `benchmark/` | GPU |
| Platform-specific | `amd/`, `ascend/` | Vendor GPU |

See [`unit/README.md`](unit/README.md) for unit test conventions.
