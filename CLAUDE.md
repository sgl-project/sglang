# CLAUDE.md — SGLang Source Analysis Guide

You are working inside the SGLang repository at `/home/liang/sglang`.

---

## 1. Repository Purpose

SGLang is a fast serving framework for large language model (LLM) inference.
It supports structured generation, radix cache (prefix caching), chunked prefill,
speculative decoding, multi-LoRA, disaggregated prefill-decode, and multiple
hardware backends (NVIDIA CUDA, AMD ROCm, Ascend NPU, Intel XPU, Apple MLX).

The core runtime lives under `python/sglang/srt/`.

## 2. Important Directories

| Path | Description |
|------|-------------|
| `python/sglang/srt/managers/` | Scheduler, tokenizer manager, schedule batch data structures |
| `python/sglang/srt/model_executor/` | Forward batch info, model runner, CUDA graph runners |
| `python/sglang/srt/mem_cache/` | KV cache pools, radix cache, allocators, page-major layout |
| `python/sglang/srt/layers/attention/` | Attention backends (FlashInfer, Triton, AITER, DSA, etc.) |
| `python/sglang/srt/layers/` | Model layers: MLP, MoE, rotary, quantization, logits processor |
| `python/sglang/srt/models/` | Per-model implementations (GLM, Qwen, Llama, DeepSeek, etc.) |
| `python/sglang/srt/disaggregation/` | PD disaggregation: prefill, decode, NIXL, KV transfer |
| `python/sglang/srt/hardware_backend/` | Backend-specific code: NPU, GPU, MLX, XPU |
| `python/sglang/srt/speculative/` | Speculative decoding: EAGLE, DFLASH, MTP |
| `python/sglang/srt/sampling/` | Sampling logic: top-p, top-k, temperature, penalties |
| `test/registered/` | Registered CI tests (sorted by category) |
| `docs/` | Documentation and developer guides |
| `reports/` | Local analysis reports (see section 10) |

### Key Source Files

| File | Lines | Subsystem |
|------|-------|-----------|
| `managers/scheduler.py` | ~4305 | Core scheduler: `get_next_batch_to_run()`, `run_batch()`, `process_batch_result()` |
| `managers/schedule_batch.py` | ~2940 | `Req`, `ScheduleBatch`: `prepare_for_extend()`, `prepare_for_decode()`, `filter_batch()` |
| `model_executor/forward_batch_info.py` | ~1575 | `ForwardBatch.init_new()`, `ForwardMode` enum |
| `mem_cache/memory_pool.py` | ~3698 | `ReqToTokenPool`, `MHATokenToKVPool`, `MLATokenToKVPool`, `PageMajorMHATokenToKVPool` |
| `mem_cache/radix_cache.py` | ~836 | `RadixCache`: `match_prefix()`, `insert()`, `cache_finished_req()` |
| `mem_cache/common.py` | ~700+ | `alloc_for_extend()`, `alloc_for_decode()`, `release_kv_cache()` |
| `model_executor/model_runner.py` | ~3342 | `ModelRunner.forward()`, model loading, attention backend init |
| `layers/attention/base_attn_backend.py` | ~251 | `AttentionBackend` ABC: `forward_extend()`, `forward_decode()` |
| `layers/attention/flashinfer_backend.py` | ~2045 | `FlashInferAttnBackend` |
| `layers/attention/triton_backend.py` | ~1945 | `TritonAttnBackend` |
| `model_executor/runner/decode_cuda_graph_runner.py` | ~1120 | Decode CUDA graph replay |
| `model_executor/runner/prefill_cuda_graph_runner.py` | ~976 | Prefill piecewise CUDA graph |
| `model_executor/runner_backend/cuda_graph_dedup_mixin.py` | ~375 | `cudaGraphExecUpdate` dedup |
| `managers/hisparse_coordinator.py` | ~500+ | HiSparse sparse KV cache coordinator |
| `mem_cache/layout/page_major.py` | ~100+ | Page-major envelope layout builders |

## 3. Safe Inspection Commands

These commands are read-only and safe to run:

```bash
# Git metadata
git log --oneline -20
git diff HEAD~5 --stat
git status --short

# File discovery
find python/sglang/srt -name "*.py" | head -50
ls python/sglang/srt/managers/
ls python/sglang/srt/mem_cache/

# Content search (use rg / grep)
rg -n "class.*Backend" python/sglang/srt/layers/attention/
rg -n "def prepare_for_extend|def prepare_for_decode" python/sglang/srt/managers/schedule_batch.py
rg -n "class.*Pool" python/sglang/srt/mem_cache/memory_pool.py
rg -n "ForwardMode\\." python/sglang/srt/model_executor/forward_batch_info.py

# Line counts
wc -l python/sglang/srt/managers/scheduler.py

# Read files
head -100 python/sglang/srt/managers/scheduler.py
```

## 4. Dangerous Commands to Avoid

Do NOT run these without explicit user permission:

```bash
# GPU jobs
python -m sglang.launch_server --model-path <model>   # starts a server
python -m sglang.bench_one_batch                       # GPU benchmark
python -m sglang.bench_serving                         # serving benchmark

# Heavy tests
pytest test/registered/ -x                             # may launch servers/GPU
python -m sglang.test.srt                              # may run GPU inference

# Model downloads
huggingface-cli download <model>                       # downloads weights

# Destructive git
git reset --hard                                        # loses uncommitted work
git clean -fd                                           # deletes untracked files
git push --force                                        # force push

# Heavy builds
pip install -e .                                        # may compile CUDA kernels
python -c "import sglang; sglang.compile()"             # JIT compilation
```

## 5. How to Analyze Latest Upstream Updates

```bash
# See recent commits
git log --oneline -30

# See what changed in the last pull
git diff ORIG_HEAD..HEAD --stat | head -100
git log ORIG_HEAD..HEAD --oneline

# Check specific areas
git log ORIG_HEAD..HEAD --oneline -- python/sglang/srt/mem_cache/
git log ORIG_HEAD..HEAD --oneline -- python/sglang/srt/layers/attention/
git log ORIG_HEAD..HEAD --oneline -- python/sglang/srt/model_executor/

# See if a specific feature was added/changed
git log --all --oneline --grep="radix\|KV cache\|page.major\|HiSparse"
```

## 6. How to Inspect Prefill/Decode/KV Cache Code Paths

### Prefill (EXTEND) path:
```
scheduler.py::get_next_batch_to_run()
  -> _get_new_batch_prefill_raw()
    -> PrefillAdder (schedule_policy.py)
    -> ScheduleBatch.init_new()
    -> batch.prepare_for_extend()      [schedule_batch.py:2011]
      -> alloc_for_extend()             [mem_cache/common.py:459]
    -> ForwardBatch.init_new()          [forward_batch_info.py:613]
    -> ModelRunner.forward()            [model_runner.py:2954]
      -> attn_backend.forward_extend()  [flashinfer_backend.py:952]
        -> token_to_kv_pool.set_kv_buffer()
```

### Decode path:
```
scheduler.py::get_next_batch_to_run()
  -> update_running_batch()             [scheduler.py:3019]
    -> batch.check_decode_mem()
    -> batch.prepare_for_decode()       [schedule_batch.py:2618]
      -> alloc_for_decode()             [mem_cache/common.py:593]
    -> ForwardBatch.init_new()
    -> ModelRunner.forward() / CUDA graph replay
      -> attn_backend.forward_decode()  [flashinfer_backend.py:1086]
        -> token_to_kv_pool.set_kv_buffer()
```

### KV Cache:
```
ReqToTokenPool                         [memory_pool.py:242]
  req_to_token[req_pool_idx, pos] -> physical KV slot index

TokenToKVPool subclasses:
  MHATokenToKVPool                     [memory_pool.py:1258]
  PageMajorMHATokenToKVPool            [memory_pool.py:2174]
  MLATokenToKVPool                     [memory_pool.py:2559]
  MHATokenToKVPoolFP4                  [memory_pool.py:2024]

RadixCache                             [radix_cache.py:286]
  match_prefix() -> prefix_indices
  insert() -> tree node with KV indices
  cache_finished_req() -> insert full sequence
  evict() -> LRU eviction
```

## 7. How to Inspect Registered Tests

```bash
# List test categories
ls test/registered/

# Find tests by keyword
find test/registered -name "*glm*" -o -name "*radix*" -o -name "*kv_cache*"
rg -l "glm5|GLM.5|hisparse|disaggregation" test/registered/

# Read a test file
head -50 test/registered/unit/models/test_glm4_moe_gate_fp32_cache.py

# List unit tests
ls test/registered/unit/
ls test/registered/unit/models/
```

## 8. How to Inspect Backend-Specific Changes

### CUDA (default):
```bash
ls python/sglang/srt/layers/attention/  # FlashInfer, Triton, TRT-LLM backends
rg -n "is_cuda|cuda_graph" python/sglang/srt/model_executor/
```

### Ascend/NPU:
```bash
ls python/sglang/srt/hardware_backend/npu/
rg -n "is_npu|_is_npu" python/sglang/srt/
ls python/sglang/srt/hardware_backend/npu/attention/
```

### AMD/ROCm:
```bash
rg -n "is_hip|_is_hip|aiter|rocm" python/sglang/srt/
ls python/sglang/srt/layers/attention/aiter_backend.py
```

### GB300/NVFP4:
```bash
rg -n "nvfp4|fp4|gb300|GB300" python/sglang/srt/
ls test/registered/gb300/ 2>/dev/null
```

## 9. Local User Constraints

- **Do not run heavy GPU jobs** — no `launch_server`, `bench_one_batch`, or GPU-based `pytest`.
- **Do not start training** — no training loops or fine-tuning scripts.
- **Do not download models** — no `huggingface-cli download` or weight fetching.
- **Prefer static analysis** — read files, grep, git log, lightweight Python imports.
- **Ask before expensive commands** — if a command may take >10 seconds or use significant CPU/RAM, explain it first and wait for approval.
- **Do not delete files** — no `rm` on source or report files.
- **Do not modify core source code** — analysis only unless the user explicitly requests changes.

## 10. Notes About Existing Reports in `reports/`

| File | Description |
|------|-------------|
| `reports/latest_update_prefill_decode_kv_cache_analysis.md` | Detailed English analysis of the latest upstream pull (175 commits, 970 files). Covers GLM-5.2 test renames, NPU expansion, disaggregation, HiSparse, DCP, page-major KV, CUDA graph dedup, speculative decoding, GB300/NVFP4, JIT kernels. |
| `reports/latest_update_prefill_decode_kv_cache_summary_zh.md` | Chinese summary of the same update, covering prefill/decode/KV cache lifecycle, radix cache role, CUDA graph role, and test implications. |
| `reports/source_walk_prefill_decode_kv_cache_zh.md` | Source-code guided Chinese report with exact file paths, class/function names, and call chains. (This session) |
| `reports/source_walk_file_index_zh.md` | File index of important source files with subsystem tagging. (This session) |

Reports in `docs/glm52_analysis/` may also exist for GLM-5.2 specific analysis.

## 11. Suggested Investigation Commands

```bash
# Find GLM-related code
rg -n "GLM|glm|Glm|THUDM" python/ test/
find python -iname "*glm*" -o -iname "*chatglm*"

# Find attention backends
rg -n "class.*Backend" python/sglang/srt/layers/attention/

# Find KV cache pool types
rg -n "class.*Pool|class.*Allocator" python/sglang/srt/mem_cache/

# Find forward mode definitions
rg -n "class ForwardMode" python/sglang/srt/model_executor/forward_batch_info.py

# Find scheduler entry points
rg -n "def get_next_batch_to_run|def run_batch|def process_batch_result" python/sglang/srt/managers/scheduler.py

# Find radix cache operations
rg -n "def match_prefix|def insert|def cache_finished|def cache_unfinished|def evict" python/sglang/srt/mem_cache/radix_cache.py
```

---

*Last updated: 2026-07-02*

---

## GLM-5.2 EAGLE3 PP Development Status

**Date:** 2026-07-17
**Branch:** `liang/glm52-eagle3-pp`

### 1. Current Branch and Safety State

```
Repository: /home/liang/sglang
Branch: liang/glm52-eagle3-pp
Do not switch branches.
liang/pp-spec-rework belongs to another worktree and must not be touched.
Working tree contains valuable uncommitted work.
No automatic staging, committing, resetting, cleaning, stashing, rebasing, or merging.
```

### 2. Production Target

```
GLM-5.2 target model
true separate EAGLE3 draft checkpoint
target TP=4
target PP=2
world size=8
draft only on physical PP1
draft logical TP=4
topk=1
non-overlap scheduler
8× NVIDIA H200 PCIe
no NVLink/NVSwitch
```

### 3. Current Host Limitations

The validation host contains only:

```
2× NVIDIA H100 PCIe 80 GB
PHB topology
no CUDA P2P between GPU 0 and GPU 1
NCCL SHM/host-mediated transport
```

These two-GPU results **cannot** prove:

- TP4×PP2 versus TP8 performance.
- The optimality of a 40/38 layer split.
- 8×H200 performance.
- Real GLM-5.2 correctness.
- Real separate EAGLE3 draft correctness or acceptance.
- Real PP1-only draft initialization at world size 8.
- 200K+ context performance.
- Production PCIe traffic behavior.
- Production CUDA Graph replay.

### 4. Completed EAGLE3 PP Runtime Work

The following runtime features are implemented and tested:

- GLM-5.2 EAGLE3 PP auxiliary hidden-state propagation (`python/sglang/srt/speculative/glm52_eagle3_pp.py`).
- RID-keyed speculative chain state in `scheduler_pp_mixin.py`.
- Capture-layer validation (startup and runtime).
- Single PP partition source of truth (`SGLANG_PP_LAYER_PARTITION` only; `SGLANG_GLM52_PP_SPLIT` removed).
- PP1-only draft collective hardening.
- CUDA Graph auxiliary static buffers.
- Required PP proxy-key validation.
- Draft identity validation (separate EAGLE3 checkpoint, not MTP/NextN).
- Real two-process Gloo distributed test (`test_glm52_eagle3_pp_distributed.py`).
- Real two-process NCCL distributed test (same file, `--backend nccl`).
- Corrected Tensor Core benchmark (`scripts/perf/bench_tensor_core_util.py`).
- Corrected communication benchmark with one-way-with-ack and ping-pong (`scripts/perf/bench_glm52_pp_transport_v2.py`).
- Real SGLang `GroupCoordinator` transport benchmark (`scripts/perf/bench_sglang_pp_transport_real.py`).
- DeepGEMM dense FP8 benchmark (`scripts/perf/bench_glm52_fp8_deepgemm.py`).
- Nsight Systems 2026.3.1 compatibility discovery.

These do **not** prove the final eight-GPU topology is correct or optimal.

### 5. Packed PP Transport Implementation Status

**Feature flag:**

```
SGLANG_PP_PACKED_TRANSPORT
disabled by default (EnvBool(False) in environ.py)
```

**Implemented components (in `python/sglang/srt/distributed/pp_packed_transport.py`):**

- Versioned packed protocol (protocol version 1).
- Separate floating-point data buffer and integer/control buffer.
- Explicit schema and presence bitmask handling.
- Bounded LRU schema cache (max 64 entries) with hit, miss, and eviction counters.
- Static buffer registry with bucket-based allocation and stable backing pointers.
- Active-row views.
- Capacity and validation checks.
- Old tensor-dict path retained as the default/fallback path.
- Focused pack/unpack correctness tests (27 tests in `test/unittest/perf_glm52/test_packed_transport.py`).

**Important limitation:**

```
Helper implementation and benchmark validation completed.
Full scheduler/model-runner production-chain integration still requires explicit confirmation.
```

The production scheduler (`scheduler_pp_mixin.py`) does **not** currently import or invoke `pp_packed_transport.py`. The packed transport is validated through dedicated benchmark scripts (`scripts/perf/bench_pp_transport_comparison.py`, `scripts/perf/soak_pp_transport.py`) that call the pack/unpack helpers directly. Wiring the packed path into `_pp_send_dict_to_next_stage()` and `_pp_recv_typed_dict()` behind the feature flag is pending work.

### 6. Correctness and Test Status

Latest confirmed results:

```
compile check: passed (RC=0)
focused test suite: 108 passed, 0 failed
packed transport helper tests: 27 passed
git diff --check: passed at the recorded validation point
```

Covered cases in packed transport tests:

- BF16 and FP16 data tensors.
- int32 control tensors.
- Contiguous and non-contiguous inputs.
- Missing optional keys (topk absent).
- Missing required keys (hidden_states absent).
- Capacity overflow detection.
- Shape mismatch detection.
- Dtype mismatch detection.
- Protocol-version mismatch detection.
- Schema mismatch detection.
- Cache eviction behavior.
- Active-row transitions: 16→1, 16→4, 8→2, 32→1, 1→16→1, 64→4→64.
- No stale rows exposed.
- No stale optional values exposed.
- Stable static-buffer pointers in focused tests.

These are focused helper and distributed transport validation tests, not full-model validation.

### 7. Reproduced Transport Measurements

Corrected benchmark results from the successful run (hidden+residual+aux+topk, BF16, GPU 0→1):

| Mode                 |    M=1 |    M=4 |   M=16 |   M=64 |  M=256 |  M=1024 |
| -------------------- | -----: | -----: | -----: | -----: | -----: | ------: |
| Existing tensor-dict | 431 µs | 424 µs | 387 µs | 508 µs | 745 µs | 1743 µs |
| Packed static        | 186 µs | 186 µs | 187 µs | 277 µs | 465 µs | 1553 µs |
| Speedup              |  2.31× |  2.28× |  2.07× |  1.83× |  1.60× |   1.12× |

Environment:

```
2× NVIDIA H100 PCIe
PHB topology
no CUDA P2P
NCCL SHM/host-mediated transport
GPUs classified as LIGHTLY_CONTENDED during the recorded run
```

These numbers do **not** predict 8×H200 performance.

### 8. Benchmark Failure-History Clarification

- The first background packed-transport benchmark attempt failed.
- The failure was caused by benchmark synchronization and transport misuse:
  - CPU tensor communication attempted through the NCCL process group.
  - A mismatched barrier/object-message design caused rank desynchronization.
- The benchmark harness was corrected (header moved to GPU tensor, barrier removed from packed path).
- The corrected benchmark then completed successfully with return code 0.
- All four tested modes completed: old, packed, packed_cached, packed_static.
- All six tested token-row sizes completed: 1, 4, 16, 64, 256, 1024.
- The successful result file was:

```
/tmp/sglang-perf-corrected-20260716-182226/pp_comparison.json
```

The original failed background notification is **not** a remaining runtime failure. It was a benchmark-harness bug, not a production-code bug.

`/tmp` artifacts are ephemeral and must not be committed.

### 9. Soak and Failure-Validation Status

Soak test results:

```
10,000 transport rounds
0 correctness errors
0 timeouts
0 deadlocks
bounded CUDA memory (12.6–25.1 MiB, no growth)
stable RSS (1401.1–1401.9 MiB, no growth)
bounded schema cache (2 entries)
stable p50 latency (~440 µs, no observed drift)
```

Tested validation failures (focused helper tests):

- Unknown schema ID → RuntimeError with diagnostic.
- Negative active rows → RuntimeError.
- Capacity overflow → RuntimeError.
- Presence-bitmask mismatch → RuntimeError.
- Cache eviction behavior → bounded, no unbounded growth.

These were the completed focused validation paths. Comprehensive distributed fault tolerance (real rank-loss, partial-send, timeout, or cache-desynchronization under live process failure) was **not** tested.

### 10. Nsight Systems Status

```
Nsight Systems version: 2026.3.1.157
Binary path: /home/AIShared/liang_temp/docker/overlay2/b4e2c37e70c567440860b77f98bd0e0eacb715abd40b9b6372e326b302664ef8/diff/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/nsys
profile return code: 0
CUDA kernel rows: 770
NVTX rows: 2,332
GPU metrics: unavailable without required privileges
dominant communication kernel: ncclDevKernel_SendRecv
```

The existing tensor-dict path submitted more small NCCL sends (metadata + per-tensor) than the packed path (header + data + control = 3 sends). This observation explains the measured overhead difference. It does **not** constitute an 8×H200 performance conclusion.

### 11. TP1×PP2 Tiny Full-Stack Integration

```
Not attempted.
```

A dedicated tiny-model fixture and full-stack harness still need to be built. Suggested future validation levels:

1. TP1×PP2 tiny target model without speculative decoding.
2. Synthetic `PPProxyTensors` through the real scheduler/model-runner chain.
3. Tiny separate draft model when an appropriate fixture is available.

These are future tasks only.

### 12. DeepGEMM Status

Directly observed facts:

- Dense FP8 GEMM (`deep_gemm.fp8_gemm_nt`) worked in the isolated `ds2` environment with CUDA 12.9 and PDL disabled (`SGLANG_DEEPGEMM_PDL=0`).
- Grouped GEMM (`deep_gemm.fp8_m_grouped_gemm_nt_masked`) produced `CUDA_ERROR_ILLEGAL_ADDRESS`.
- The environment involved a CUDA 12.9 JIT/toolchain path (`CUDA_HOME=/home/AIShared/liang_temp/envs/ds2`) and a CUDA 13.0-capable runtime/driver (torch 2.11.0+cu130, driver 580.159.03).

A CUDA toolchain/runtime mismatch is one hypothesis, not yet a proven root cause. Grouped-GEMM shape handling, generated kernel correctness, JIT cache contents, and the exact DeepGEMM commit also remain possible causes.

Future diagnosis should use an isolated environment and record:

- DeepGEMM commit hash.
- `torch.version.cuda`.
- `nvcc --version`.
- Driver version.
- Generated compiler command.
- JIT cache directory contents.
- Minimal failing grouped shape.
- Compute Sanitizer result where available.

### 13. Proposed Future Commit Split

This is a proposed manual split only. No commit, staging, or push was performed automatically.

```
A. Core EAGLE3 PP runtime
   - python/sglang/srt/speculative/glm52_eagle3_pp.py
   - python/sglang/srt/managers/scheduler_pp_mixin.py
   - python/sglang/srt/managers/scheduler.py
   - python/sglang/srt/managers/tp_worker.py
   - python/sglang/srt/managers/utils.py
   - python/sglang/srt/speculative/eagle_worker_v2.py
   - python/sglang/srt/models/deepseek_v2.py
   - python/sglang/srt/model_executor/ (existing changes)
   - python/sglang/srt/server_args.py
   - python/sglang/srt/distributed/utils.py
   - python/sglang/srt/environ.py (EAGLE3 PP env vars only)
   - test/unittest/test_eagle3_pp_*.py

B. Packed transport runtime
   - python/sglang/srt/distributed/pp_packed_transport.py
   - python/sglang/srt/environ.py (SGLANG_PP_PACKED_TRANSPORT only)

C. Packed transport tests
   - test/unittest/perf_glm52/test_packed_transport.py

D. Performance and diagnostic harnesses
   - scripts/perf/bench_tensor_core_util.py
   - scripts/perf/bench_glm52_fp8_deepgemm.py
   - scripts/perf/bench_glm52_pp_transport_v2.py
   - scripts/perf/bench_sglang_pp_transport_real.py
   - scripts/perf/bench_pp_transport_comparison.py
   - scripts/perf/soak_pp_transport.py
   - test/unittest/perf_glm52/test_benchmark_correctness.py
   - test/unittest/perf_glm52/test_perf_calculations.py
```

The following must **not** be committed:

```
/tmp/sglang-perf-corrected-*/*
*.nsys-rep
*.sqlite
*.ncu-rep
temporary logs
generated profiler artifacts
```

Do not use `git add .`.

### 14. Remaining Two-H100 Work

Highest-value work that can still be honestly validated on the current host:

- Confirm packed transport is wired into the real production scheduler call chain in both directions (PP0→PP1 and PP1→PP0).
- Remove GPU-header-to-CPU synchronization such as `.tolist()` or `.item()` from the fast path.
- Audit token/control dtype preservation, especially int32 versus int64.
- Add batched P2P submission.
- Add dedicated PP communication streams.
- Add ping-pong or bounded ring static buffers for asynchronous send safety.
- Validate CUDA Graph packed-buffer mechanics using fixed buckets and device-side active-row masking.
- Build TP1×PP2 tiny full-stack smoke tests.
- Add two-process schema-cache desynchronization tests.
- Add delayed receiver and partial-message failure tests.
- Add logical four-lane TP concurrency stress without claiming TP4 performance.
- Continue DeepGEMM grouped-GEMM diagnosis in a separate environment.

These are pending.

### 15. Remaining Eight-H200 Production Gates

All of the following remain untested:

- Real TP4×PP2.
- Real GLM-5.2 target weights.
- Real separate EAGLE3 draft checkpoint.
- Real PP1-only draft initialization.
- Real draft logical TP4.
- Real 40/38 split.
- TP8 versus TP4×PP2 A/B.
- Full-model CUDA Graph replay.
- Real EAGLE3 acceptance length and rate.
- Real 200K+ context behavior.
- Real PCIe TX/RX measurements.
- Production concurrency and soak.
- Failure recovery on the eight-GPU topology.

```
These gates require the 8×H200 PCIe production environment and remain unvalidated.
```
