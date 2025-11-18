# KT-Kernel Tests

Tests for SGLang integration with KT-Kernel (KTransformers) AMX CPU inference.

## Prerequisites

1. **Hardware Requirements**:
   - Intel CPU with AMX support (e.g., Xeon 8488C)
   - NVIDIA GPUs (1, 4, or 8 depending on test)

2. **Software Requirements**:
   - Python 3.10+
   - PyTorch 2.x with CUDA support
   - SGLang installed
   - KT-Kernel installed from source

3. **Model Weights**:
   - GPU weights: `models/DeepSeek-R1-0528-GPU-weight`
   - CPU weights: `models/DeepSeek-R1-0528-CPU-weight`

## Test Structure

```
test/srt/kt/
├── __init__.py                 # Package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── utils.py                    # Test utilities (get_kt_server_args, etc.)
├── test_kt_basic.py            # Basic inference workflow tests
├── test_kt_performance_*.py    # Performance tests (to be implemented)
└── test_kt_correctness.py      # Correctness tests (to be implemented)
```

## Key Design

Following SGLang's standard test patterns:

1. **Inherits from `CustomTestCase`** - Provides auto-retry on failure in CI
2. **Uses `setUpClass/tearDownClass`** - Manages server lifecycle per test class
3. **Uses `popen_launch_server`** - Standard SGLang server launcher
4. **Imports from `sglang.test.test_utils`** - Reuses common test infrastructure

## Running Tests

### Run All Basic Tests

```bash
cd test/srt
python3 -m pytest kt/test_kt_basic.py -v -s
```

### Run Specific GPU Configuration

```bash
# 1 GPU test
python3 -m pytest kt/test_kt_basic.py::TestKTBasic1GPU -v -s

# 4 GPU test
python3 -m pytest kt/test_kt_basic.py::TestKTBasic4GPU -v -s

# 8 GPU test
python3 -m pytest kt/test_kt_basic.py::TestKTBasic8GPU -v -s
```

### Run with Markers

```bash
# Run only basic tests
python3 -m pytest kt/ -m basic -v -s

# Run only 1 GPU tests
python3 -m pytest kt/ -m gpu1 -v -s

# Run only 8 GPU tests
python3 -m pytest kt/ -m gpu8 -v -s
```

### Run with unittest

```bash
cd test/srt/kt
python3 test_kt_basic.py
```

## Test Details

### test_kt_basic.py

Tests basic end-to-end inference workflow with different GPU configurations.

**Test Classes**:

1. **TestKTBasic1GPU**
   - Configuration: 1 GPU, kt_num_gpu_experts=1, kt_cpuinfer=60
   - Base URL: http://127.0.0.1:30000
   - Expected duration: ~10 minutes

2. **TestKTBasic4GPU**
   - Configuration: 4 GPU (TP=4), kt_num_gpu_experts=80, kt_cpuinfer=60
   - Base URL: http://127.0.0.1:30001
   - Expected duration: ~10 minutes

3. **TestKTBasic8GPU**
   - Configuration: 8 GPU (TP=8), kt_num_gpu_experts=200, kt_cpuinfer=60
   - Base URL: http://127.0.0.1:30002
   - Expected duration: ~10 minutes

**Test Method**: `test_basic_inference_workflow`
1. Start SGLang server with KT-kernel configuration (in setUpClass)
2. Send 5 test prompts (max_tokens=50)
3. Validate responses (non-empty, correct count)
4. Stop server gracefully (in tearDownClass)

## Environment Variables

```bash
# Optional: Override default model paths
export SGLANG_CI_MODEL_DIR=/path/to/models
export SGLANG_CI_KT_WEIGHT_DIR=/path/to/kt-weights

# Required for server startup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Troubleshooting

### Server fails to start

Check logs at `/tmp/sglang_server_<port>.log`

### AMX not supported

Verify AMX support:
```bash
python3 -c "import torch; print(torch._C._cpu._is_amx_tile_supported())"
```

### Port conflicts

Tests use different ports (30000, 30001, 30002) to avoid conflicts. Ensure these ports are available.

### Clean up

If tests fail and leave servers running:
```bash
pkill -9 -f sglang.launch_server
```

## CI Integration

These tests are integrated into SGLang CI:

- **Per-commit tests**: Run basic tests only (~15 min)
- **Nightly tests**: Run all tests including performance and correctness (~3.4 hours)

See `.github/workflows/pr-test-kt-amx.yml` for CI configuration.
