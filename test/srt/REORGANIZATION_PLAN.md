# Test Directory Reorganization Plan

## Overview

This PR moves all tests marked as "__not_in_ci__" from test/srt/ to test/manual/ for easier organization and review. It also updates the pr-test.yml CI configuration.

## Changes Made

### 1. Moved Manual Tests
- **77 tests** moved from test/srt/ to test/manual/
- These were previously in the "__not_in_ci__" section of run_suite.py
- Tests are not run in CI and are kept for manual testing purposes
- Directory structure preserved (e.g., models/, lora/, nightly/, etc.)

### 2. Updated CI Configuration
- Updated `.github/workflows/pr-test.yml` with improvements

### 3. Cleaned Up run_suite.py
- Removed the "__not_in_ci__" section from test/srt/run_suite.py
- All manual tests are now in test/manual/

## Directory Structure

```
test/
├── manual/                         # ✅ NEW: Manual tests not run in CI
│   ├── ascend/                     # Ascend-specific manual tests
│   ├── cpu/                        # CPU manual tests
│   ├── debug_utils/                # Debug utilities
│   ├── entrypoints/                # Entrypoint tests
│   ├── hicache/                    # HiCache tests
│   ├── layers/                     # Layer-specific tests
│   ├── lora/                       # LoRA manual tests
│   ├── models/                     # Model manual tests
│   ├── nightly/                    # Nightly performance tests
│   ├── openai_server/              # OpenAI server tests
│   ├── quant/                      # Quantization tests
│   ├── rl/                         # RL tests
│   └── test_*.py                   # Various manual tests
│
└── srt/                            # All CI tests remain here
    ├── run_suite.py                # Test runner (cleaned up)
    └── ... (all other test files)
```

## Manual Tests Moved (77 tests)

Tests moved to test/manual/:
- Ascend NPU tests (2)
- CPU communication tests (1)
- Debug utilities (1)
- Entrypoint tests (1)
- HiCache tests (1)
- Layer tests (2)
- LoRA tests (3)
- Model tests (7)
- Nightly performance tests (7)
- OpenAI server tests (3)
- Quantization tests (1)
- RL tests (2)
- Various other manual tests (46)

## Benefits

1. **Cleaner organization**: Manual tests clearly separated from CI tests
2. **Easier review**: No more "__not_in_ci__" section in run_suite.py
3. **Better structure**: test/manual/ directory clearly indicates purpose
4. **Simpler CI logic**: No special handling for "__not_in_ci__" tests
