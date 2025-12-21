# CI Suite Organization & Test Migration Plan

**Tracking Issue**: [#13808 - CI suites organization](https://github.com/sgl-project/sglang/issues/13808)
**Last Updated**: 2025-12-20
**Assignees**: HaiShaw, Kangyan-Zhou, iforgetmyname, mingfeima, alisonshao, hnyls2002

---

## Overview

This document tracks the progress of migrating SGLang's CI test suite from the legacy structure to a new registry-based, feature-organized system. The goal is to improve test organization, enable flexible backend/schedule configuration, and reduce CI resource waste.

---

## Current State Summary

### Directory Structure (Current)
```
test/
├── manual/           # Unofficially maintained tests (57+ files)
├── registered/       # New official tests with CI registry
│   ├── function_call/
│   ├── lora/         # MIGRATED (PR #15176)
│   │   ├── test_lora.py
│   │   ├── test_lora_backend.py
│   │   ├── test_lora_eviction.py
│   │   ├── test_lora_eviction_policy.py (nightly)
│   │   ├── test_lora_hf_sgl_logprob_diff.py (nightly)
│   │   ├── test_lora_openai_api.py (nightly)
│   │   ├── test_lora_openai_compatible.py (nightly)
│   │   ├── test_lora_qwen3.py (nightly)
│   │   ├── test_lora_radix_cache.py (nightly)
│   │   ├── test_lora_tp.py
│   │   ├── test_lora_update.py
│   │   └── test_multi_lora_backend.py
│   ├── spec/         # MIGRATED (PR #14529)
│   │   ├── eagle/
│   │   │   ├── test_eagle_constrained_decoding.py
│   │   │   ├── test_eagle_infer_a.py
│   │   │   ├── test_eagle_infer_b.py
│   │   │   └── test_eagle_infer_beta.py
│   │   └── utils/
│   │       └── test_build_eagle_tree.py
│   ├── cuda_graph/   # MIGRATED (in progress)
│   │   ├── test_piecewise_cuda_graph_small_1_gpu.py
│   │   ├── test_piecewise_cuda_graph_large_1_gpu.py
│   │   └── test_piecewise_cuda_graph_2_gpu.py
│   ├── stress/
│   └── test_srt_backend.py
├── srt/              # Legacy tests (137+ files) - TO BE MIGRATED
├── nightly/          # Legacy nightly tests (31 files) - TO BE MIGRATED
├── run_suite.py      # New runner for registered/ tests
└── run_suite_nightly.py
```

### Infrastructure Components

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| CI Registry | Done | `python/sglang/test/ci/ci_register.py` | Supports CUDA, AMD, CPU, NPU |
| Registry Decorators | Done | Same file | `register_*_ci(est_time, suite, nightly, disabled)` |
| New run_suite.py | Done | `test/run_suite.py` | Scans `registered/`, supports `--nightly`, auto-partition |
| Per-commit suites | Partial | `test/run_suite.py` | Defined but few tests migrated |
| Nightly suites | Partial | `test/run_suite.py` | Defined but few tests migrated |

---

## CUDA Per-Commit Suite Organization

### Stage A - Fast Tests (Priority: Highest)

| Suite Name | GPU Requirement | Time Threshold | Partition Rule | Purpose |
|------------|-----------------|----------------|----------------|---------|
| `stage-a-test-1` | N/A | N/A | **LOCKED** | Already setup, DO NOT add tests |
| `stage-a-test-2` | 24GB GPU | **8 minutes** | +1 partition if exceeded | Basic utility tests only |
| `stage-a-test-cpu` | CPU only (no GPU) | **5 minutes** | +1 partition if exceeded | Tests not requiring GPU |

### Stage B - Core Tests (Priority: High)

| Suite Name | GPU Requirement | Time Threshold | Partition Rule | Purpose |
|------------|-----------------|----------------|----------------|---------|
| `stage-b-test-small-1-gpu` | 1x 24GB GPU | **30 minutes** | +1 partition if exceeded | 1-GPU tests, small GPU sufficient |
| `stage-b-test-large-1-gpu` | 1x 80GB GPU | **30 minutes** | +1 partition if exceeded | 1-GPU tests requiring large GPU |
| `stage-b-test-large-2-gpu` | 2x 80GB GPUs | **30 minutes** | +1 partition if exceeded | 2-GPU tests |

### Stage C - Extended Tests (Priority: Medium)

Use Stage C when Stage B becomes overloaded. Move larger tests from Stage B to Stage C.

| Suite Name | GPU Requirement | Time Threshold | Partition Rule | Purpose |
|------------|-----------------|----------------|----------------|---------|
| `stage-c-test-large-1-gpu` | 1x 80GB GPU | **30 minutes** | +1 partition if exceeded | Overflow from stage-b-test-large-1-gpu |
| `stage-c-test-large-2-gpu` | 2x 80GB GPUs | **30 minutes** | +1 partition if exceeded | Overflow from stage-b-test-large-2-gpu |
| `stage-c-test-large-4-gpu-b200` | 4x B200 GPUs | **30 minutes** | +1 partition if exceeded | 4-GPU tests on B200 |
| `stage-c-test-large-8-gpu-b200` | 8x B200 GPUs | **30 minutes** | +1 partition if exceeded | 8-GPU tests on B200 |

### Suite Selection Guidelines

1. **Prioritize Stage B** for 1-GPU and 2-GPU tests
2. Only move to Stage C when Stage B suites exceed time thresholds
3. When moving tests to Stage C, prioritize moving the **larger/slower** tests first
4. Stage A is reserved for critical fast tests; `stage-a-test-1` is locked

---

## Time Threshold & Partitioning Rules

**IMPORTANT**: When total estimated time exceeds threshold, increment partition count.

| Suite Category | Time Threshold | Action When Exceeded |
|----------------|----------------|----------------------|
| Stage A test-2 | 8 minutes | Add 1 to partition number |
| Stage A test-cpu | 5 minutes | Add 1 to partition number |
| All Stage B suites | 30 minutes | Add 1 to partition number |
| All Stage C suites | 30 minutes | Add 1 to partition number |

Example:
- If `stage-b-test-small-1-gpu` has total est_time of 35 minutes and currently uses 2 partitions
- Increase to 3 partitions so each partition stays under threshold

---

## Existing Issues (from #13808)

1. Many CI tests under `test/srt` are never executed by current workflows
2. Tests in `sglang.test` module are not connected to any CI workflow
3. CI tests are tightly coupled to hardware backends (CUDA, AMD, XPU, NPU)
4. Cannot easily switch tests between nightly and per-commit pipelines
5. CI workflow pipelines lack proper fast-fail mechanism
6. Timeout settings are not fine-grained
7. Some performance tests require manual triggering
8. CI monitoring is tightly coupled to standalone summary steps
9. No tracking/management of flaky tests
10. CPU-only tests are mixed with other platform tests

---

## Refactoring Roadmap

### Phase 1: Infrastructure Setup
- [x] Deprecate most CI tests under `sglang.test` or move to `test/manual/`
- [x] Introduce CI registry for backend selection, nightly/per-commit inclusion, disable flags (PR #13345)
- [ ] Setup test suite infrastructure for staged migration (PR #13653 - POC, not to be merged)

### Phase 2: Test Migration (Feature-by-Feature)

#### Immediate Migration Candidates (1-GPU focused)

| Option | Feature | Files | GPU | Complexity |
|--------|---------|-------|-----|------------|
| 1 | Attention Backends | 12 | 1-GPU | Medium |
| 2 | MLA | 6 | 1-GPU | Low |
| 3 | MoE | 5 | 1-GPU | Low |
| 4 | Speculative Decoding | 3 | 1-GPU | Low |
| 5 | Constrained Decoding | 3 | 1-GPU | Low |
| 6 | Quantization | 11+ | 1-GPU | Medium |
| 7 | Vision & Multimodal | 6 | 1-GPU | Low |

#### Migration Status

| Feature | Source Files | Target Dir | Status | PR |
|---------|--------------|------------|--------|-----|
| **Eagle/Spec** | `test/srt/test_eagle_*.py` | `registered/spec/eagle/` | **DONE** | #14529 |
| **LoRA** | `test/srt/lora/`, `test/nightly/test_lora_*.py` | `registered/lora/` | **DONE** | #15176 |
| **CUDA Graph** | `test/srt/test_piecewise_cuda_graph_*.py` | `registered/cuda_graph/` | **DONE** | #15436 |
| **Attention Backends** | `test/srt/test_*attention*.py`, etc. | `registered/attention/` | **IN PROGRESS** | TBD |
| Function Calling | `test/srt/openai_server/` | `registered/function_call/` | In Progress | |
| Stress Tests | `test/srt/` stress tests | `registered/stress/` | Started | |
| OpenAI Server | `test/srt/openai_server/` | `registered/openai/` | Not Started | |
| Quantization | `test/srt/quant/` | `registered/quant/` | Not Started | |
| Multi-GPU (EP) | `test/srt/ep/` | `registered/ep/` | Not Started | |
| Models | `test/srt/models/` | `registered/models/` | Not Started | |
| Layers | `test/srt/layers/` | `registered/layers/` | Not Started | |
| HiCache | `test/srt/hicache/` | `registered/hicache/` | Not Started | |
| RL | `test/srt/rl/` | `registered/rl/` | Not Started | |
| CPU Tests | `test/srt/cpu/` | `registered/cpu/` | Not Started | |
| Ascend/NPU | `test/srt/ascend/` | `registered/ascend/` | Not Started | |
| Nightly Perf | `test/nightly/test_*_perf.py` | `registered/perf/` | Not Started | |
| Nightly Eval | `test/nightly/test_*_eval.py` | `registered/eval/` | Not Started | |
| Eagle DP/Large | `test/srt/test_eagle_dp_*.py`, `test/nightly/test_eagle_*.py` | `registered/spec/eagle/` | Not Started | |

### Phase 3: Workflow Reorganization
- [ ] Reorganize CI workflow pipeline to follow new unified structure
- [ ] Introduce fine-grained timeout settings:
  - Per-file timeout
  - Per-unit-test timeout
  - Server boot timeout
- [ ] Make all performance tests compatible with `run_suite`
- [ ] Refactor CI monitor summary/reporting to work with `run_suite` API

### Phase 4: Cleanup
- [ ] Remove legacy `test/srt/run_suite.py` once migration complete
- [ ] Update all workflow files to use new paths
- [ ] Archive or remove empty legacy directories

---

## Completed Migrations

### Eagle 1-GPU Tests (PR #14529) - MERGED

**Files Migrated to `test/registered/spec/eagle/`:**
| File | Est. Time | Suite |
|------|-----------|-------|
| test_build_eagle_tree.py | 3s | stage-b-test-small-1-gpu |
| test_eagle_constrained_decoding.py | 100s | stage-b-test-small-1-gpu |
| test_eagle_infer_a.py | 470s | stage-b-test-small-1-gpu |
| test_eagle_infer_b.py | 473s | stage-b-test-small-1-gpu |
| test_eagle_infer_beta.py | 194s | stage-b-test-small-1-gpu |

**Total Est. Time**: ~1240s (~20.7 min)

### LoRA Tests (PR #15176) - MERGED

**Per-commit tests (stage-b-test-small-1-gpu):**
| Test | Est Time |
|------|----------|
| test_lora.py | 82s |
| test_lora_backend.py | 200s |
| test_lora_eviction.py | 224s |
| test_lora_update.py | 451s |
| test_multi_lora_backend.py | 60s |

**Per-commit tests (stage-b-test-large-2-gpu):**
| Test | Est Time |
|------|----------|
| test_lora_tp.py | 116s |

**Nightly tests (nightly-1-gpu):**
| Test | Est Time |
|------|----------|
| test_lora_hf_sgl_logprob_diff.py | 300s |
| test_lora_eviction_policy.py | 200s |
| test_lora_openai_api.py | 30s |
| test_lora_openai_compatible.py | 150s |
| test_lora_qwen3.py | 97s |
| test_lora_radix_cache.py | 200s |

**Infrastructure fixes in this PR:**
- Added `python/sglang/test/ci/__init__.py`
- Moved `lora_utils.py` to `python/sglang/test/lora_utils.py`
- Added `stage-b-test-2-gpu` workflow job
- Added `stage-b-test-large-2-gpu` to `PER_COMMIT_SUITES`

### CUDA Graph Tests (PR #15436) - MERGED

**Files Migrated to `test/registered/cuda_graph/`:**

**Small 1-GPU tests (stage-b-test-small-1-gpu):**
| Test Class | Model | Est. Time |
|------------|-------|-----------|
| TestPiecewiseCudaGraphCorrectness | Llama 8B | ~60s |
| TestPiecewiseCudaGraphBenchmark | Llama 8B | ~30s |
| TestPiecewiseCudaGraphLlama31FP4 | Llama 8B FP4 (Blackwell only) | ~100s |
| TestPiecewiseCudaGraphDeepSeek | DeepSeek MLA | ~80s |
| TestPiecewiseCudaGraphFP8 | Llama 8B FP8 | ~80s |
| TestPiecewiseCudaGraphQwen25VL | Qwen 7B VLM | ~60s |
| TestPiecewiseCudaGraphInternVL25 | InternVL 8B | ~60s |
| TestPiecewiseCudaGraphQwen25VLEmbedding | Qwen 3B VLM | ~30s |

**Total Est. Time**: 460s (~7.7 min)

**Large 1-GPU tests (stage-b-test-large-1-gpu):**
| Test Class | Model | Est. Time |
|------------|-------|-----------|
| TestPiecewiseCudaGraphQwen3MoE | Qwen3 30B MoE | ~200s |
| TestPiecewiseCudaGraphGPTQ | Qwen3 30B GPTQ | ~140s |
| TestPiecewiseCudaGraphAWQ | QwQ 32B AWQ | ~140s |

**Total Est. Time**: 480s (~8 min)

**2-GPU tests (stage-b-test-large-2-gpu):**
| Test Class | Model | Est. Time |
|------------|-------|-----------|
| TestPiecewiseCudaGraphQwen3OmniMOE | Qwen3 Omni 30B MoE (tp=2) | ~200s |
| TestPiecewiseCudaGraphFusedMoE | Qwen3 Coder 30B MoE (tp=2, ep=2) | ~55s |

**Total Est. Time**: 255s (~4.3 min)

**Changes made:**
- Reorganized tests by GPU size (small 24GB vs large 80GB)
- Removed invalid `stage-b-test-small-2-gpu` suite
- Added `stage-b-test-large-1-gpu` and `stage-b-test-large-2-gpu` suites
- Deleted old test files from `test/srt/`

### Attention Backends (IN PROGRESS)

**Files Migrated to `test/registered/attention/`:**

**Per-commit tests (stage-b-test-small-1-gpu):**
| Test File | Description | Est. Time |
|-----------|-------------|-----------|
| test_radix_cache_unit.py | RadixCache unit tests (CPU-based) | ~5s |
| test_create_kvindices.py | Triton KV indices kernel | ~10s |
| test_triton_attention_kernels.py | Triton attention kernels (decode, extend, prefill) | ~30s |
| test_wave_attention_kernels.py | Wave attention kernels | ~60s |
| test_radix_attention.py | RadixAttention server integration | ~60s |
| test_torch_native_attention_backend.py | Torch native attention + MMLU | ~90s |
| test_triton_attention_backend.py | Triton attention + MMLU | ~120s |
| test_triton_sliding_window.py | Sliding window attention (Gemma-3) | ~150s |

**Total Est. Time**: 525s (~8.8 min)

**Per-commit tests (stage-c-test-large-4-gpu):**
| Test File | Description | Est. Time |
|-----------|-------------|-----------|
| test_local_attn.py | Local attention with FA3 (tp=4, SM 90+) | ~200s |

**Total Est. Time**: 200s (~3.3 min)

**Nightly tests (nightly-1-gpu):**
| Test File | Description | Est. Time |
|-----------|-------------|-----------|
| test_fa3.py | FlashAttention3 + MLA + SpecDecode (SM 90+) | ~300s |
| test_hybrid_attn_backend.py | Hybrid FA3+FlashInfer (SM 90+) | ~200s |
| test_flash_attention_4.py | FlashAttention4 (SM 100+ Blackwell) | ~200s |

**Total Est. Time**: 700s (~11.7 min)

**Infrastructure changes:**
- Created `test/registered/attention/` directory
- Added `stage-c-test-large-4-gpu` suite to `test/run_suite.py`
- Added `stage-c-test-large-4-gpu` job to `.github/workflows/pr-test.yml`
- Added `stage-c-test-large-4-gpu` to `scripts/ci/slash_command_handler.py`
- Deleted 12 original test files from `test/srt/`
- **Removed test entries from legacy `test/srt/run_suite.py`** (per-commit-1-gpu, per-commit-4-gpu, per-commit-4-gpu-b200, per-commit-amd)

---

## CI Registry Reference

### Registration Functions
```python
from sglang.test.ci.ci_register import (
    register_cuda_ci,
    register_amd_ci,
    register_cpu_ci,
    register_npu_ci,
)

# Per-commit test (small 1-gpu)
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu")

# Per-commit test (large 1-gpu)
register_cuda_ci(est_time=120, suite="stage-b-test-large-1-gpu")

# Per-commit test (2-gpu)
register_cuda_ci(est_time=200, suite="stage-b-test-large-2-gpu")

# Per-commit test (4-gpu H100)
register_cuda_ci(est_time=200, suite="stage-c-test-large-4-gpu")

# CPU-only test
register_cuda_ci(est_time=30, suite="stage-a-test-cpu")

# Nightly-only test
register_cuda_ci(est_time=200, suite="nightly-1-gpu", nightly=True)

# Multi-backend test
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=120, suite="stage-a-test-1")

# Temporarily disabled test
register_cuda_ci(est_time=80, suite="stage-b-test-small-1-gpu", disabled="flaky - see #12345")
```

### Available Suites Summary

**Per-Commit (CUDA)**:
- Stage A: `stage-a-test-1` (locked), `stage-a-test-2`, `stage-a-test-cpu`
- Stage B: `stage-b-test-small-1-gpu`, `stage-b-test-large-1-gpu`, `stage-b-test-large-2-gpu`
- Stage C: `stage-c-test-large-1-gpu`, `stage-c-test-large-2-gpu`, `stage-c-test-large-4-gpu`, `stage-c-test-large-4-gpu-b200`, `stage-c-test-large-8-gpu-b200`

**Per-Commit (AMD)**:
- `stage-a-test-1`

**Per-Commit (CPU)**:
- `default`

**Nightly (CUDA)**:
- `nightly-1-gpu`, `nightly-2-gpu`, `nightly-4-gpu`, `nightly-4-gpu-b200`
- `nightly-8-gpu`, `nightly-8-gpu-h200`, `nightly-8-gpu-h20`, `nightly-8-gpu-b200`

**Nightly (AMD)**:
- `nightly-amd`

**Nightly (NPU)**:
- `nightly-1-npu-a3`, `nightly-2-npu-a3`, `nightly-4-npu-a3`, `nightly-16-npu-a3`

---

## Running Tests

```bash
# Run per-commit tests
python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu

# Run nightly tests
python test/run_suite.py --hw cuda --suite nightly-1-gpu --nightly

# With auto-partitioning (for parallel CI jobs)
python test/run_suite.py --hw cuda --suite stage-b-test-small-1-gpu \
    --auto-partition-id 0 --auto-partition-size 4
```

---

## Migration Checklist (Per Feature)

When migrating a feature:

1. [ ] Create `test/registered/<feature>/` directory
2. [ ] Move test files from `test/srt/` or `test/nightly/`
3. [ ] Add CI registry decorator to each test file
4. [ ] Update imports if necessary
5. [ ] Remove tests from legacy `run_suite.py` files
6. [ ] Verify tests run with new `test/run_suite.py`
7. [ ] Update workflow files if needed:
   - [ ] Add new job in `.github/workflows/pr-test.yml` if using a new suite
   - [ ] Add stage name to `scripts/ci/slash_command_handler.py` (nvidia_stages list) for `/rerun-stage` support
8. [ ] **RUN CI TO VERIFY ESTIMATED TIMES ARE CORRECT**
9. [ ] Check if total suite time exceeds threshold; adjust partitions if needed
10. [ ] Create PR with clear description
    - [ ] Mention roadmap issue #13808 in PR description (e.g., "Part of #13808")

### Adding New Workflow Stages

When adding a new suite that requires a new workflow job:

1. **Add job to `pr-test.yml`**: Copy an existing similar job (e.g., `stage-b-test-small-1-gpu`) and modify:
   - Job name (e.g., `stage-b-test-large-1-gpu`)
   - `inputs.target_stage` condition
   - `runs-on` runner (use appropriate GPU runner)
   - Suite name in `run_suite.py` command

2. **Add to slash command handler**: Update `scripts/ci/slash_command_handler.py`:
   - Add stage name to `nvidia_stages` list (for NVIDIA tests) or `amd_stages` list (for AMD tests)
   - This enables `/rerun-stage <stage-name>` command for the new stage

**REMINDER**: After creating a migration PR, always run CI to verify that the `est_time` values are accurate. Adjust times based on actual CI run durations.

---

## Related PRs

| PR | Description | Status |
|----|-------------|--------|
| #13345 | CI refactor: introduce CI register | Merged |
| #13653 | POC: Setup test suite infrastructure for staged migration | Open (POC) |
| #13941 | Add nightly test support to unified run_suite.py | Merged |
| #14529 | Migrate Eagle 1-GPU tests to test/registered/ | **Merged** |
| #15176 | Migrate LoRA tests to test/registered/lora/ | **Merged** |
| #15436 | Migrate CUDA Graph tests to test/registered/cuda_graph/ | **Merged** |

---

## Notes & Decisions

- Tests are organized by **feature** not behavior (per-commit vs nightly)
- The `nightly` flag in registry determines schedule, not directory location
- `test/manual/` is for unofficially maintained tests (code references for AI agents)
- `test/registered/` is for officially maintained tests guaranteed to run in CI
- Migration should happen incrementally - one feature at a time in separate PRs
- **Always verify est_time by running CI after migration**
- **Monitor total suite times and increase partitions when thresholds exceeded**

---

## Session Log

### 2025-12-20
- Updated CUDA Graph tests status to DONE (PR #15436 merged)
- Eagle 1-GPU, LoRA, and CUDA Graph migrations all complete
- **Started Attention Backends migration**:
  - Created `test/registered/attention/` directory with 12 test files
  - Added `stage-c-test-large-4-gpu` suite for 4-GPU H100 tests
  - Per-commit tests: 8 files for stage-b-test-small-1-gpu (~8.8 min)
  - Per-commit tests: 1 file for stage-c-test-large-4-gpu (~3.3 min)
  - Nightly tests: 3 files for nightly-1-gpu (~11.7 min)
  - Updated pr-test.yml workflow with new stage-c-test-large-4-gpu job
  - Updated slash_command_handler.py for /rerun-stage support

### 2025-12-18
- Created this planning document
- Analyzed current state from issue #13808
- Documented existing infrastructure and migration roadmap
- Added detailed CUDA per-commit suite organization (Stage A, B, C)
- Added time threshold and partitioning rules
- Documented completed migrations: Eagle (#14529), LoRA (#15176)
- **Migrated CUDA Graph (piecewise) tests**:
  - Created `test/registered/cuda_graph/` directory
  - Split tests by GPU size: small (24GB), large (80GB), 2-GPU
  - Fixed invalid `stage-b-test-small-2-gpu` suite → `stage-b-test-large-2-gpu`
  - Added `stage-b-test-large-1-gpu` suite
  - Updated `test/run_suite.py` with new suites
  - Updated `pr-test.yml` workflow (added `stage-b-test-large-1-gpu` job)
  - Updated `test_lora_tp.py` to use correct suite
  - Added `stage-b-test-large-1-gpu` to `scripts/ci/slash_command_handler.py` for `/rerun-stage` support
  - Renamed `stage-b-test-2-gpu` to `stage-b-test-large-2-gpu` in both `pr-test.yml` and `slash_command_handler.py` for consistency
  - Added documentation for adding new workflow stages to migration checklist

---
