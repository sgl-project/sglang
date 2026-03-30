# SGLang CudaProfilerApi Integration - Complete Analysis

## 📋 Documentation Index

This directory contains a comprehensive analysis of the CudaProfilerApi (cudaProfilerStart/Stop) integration in the SGLang codebase.

### 📄 Documents Included

#### 1. **CUDAPROFILER_INTEGRATION_ANALYSIS.md** (18 KB, 526 lines)
The **complete technical reference** with detailed explanations.

**Contents:**
- Executive Summary
- 4 primary implementation locations with full code snippets
- How capture-range cudaProfilerApi integration works
- Multi-GPU and distributed considerations
- Profile activities configuration
- Profile Manager and stage-based profiling
- Environment variables and configuration
- Multimodal generation (diffusion) profiler details
- Comparison table: cudaProfilerStart vs torch.profiler
- Search results summary
- Integration flow diagram
- Configuration examples
- Summary table of all components

**Best for:** In-depth understanding, detailed implementation review, architecture documentation

---

#### 2. **CUDAPROFILER_QUICK_REFERENCE.txt** (20 KB)
A **quick lookup guide** with formatted ASCII tables and examples.

**Contents:**
- Primary implementation locations with emojis and quick facts
- How it works with nsys (step-by-step diagram)
- Key features highlighted
- Configuration & usage examples (4 examples)
- Profile activities reference
- Environment variables table
- Comparison chart (cudaProfilerStart vs torch.profiler)
- Troubleshooting guide

**Best for:** Quick lookups, copy-paste examples, troubleshooting, CLI usage

---

## 🎯 Quick Facts

| Aspect | Detail |
|--------|--------|
| **Primary File** | `sglang/multimodal_gen/runtime/managers/scheduler.py` (Lines 368-386) |
| **Total Implementations** | 4 main locations + supporting code |
| **Key Technology** | `torch.cuda.cudart().cudaProfilerStart/Stop()` |
| **External Tool** | Nsight Systems (nsys) with `--capture-range=cudaProfilerApi` |
| **Main Benefit** | Precise GPU profiling with warmup exclusion |
| **Overhead** | Minimal (2 API calls, no tracing overhead) |
| **Multi-GPU Safe** | Yes (only base_gpu_id issues profiler calls) |

---

## 🚀 Quick Start

### Use Case 1: Profile a Benchmark
```bash
python -m sglang.bench_one_batch \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --batch 1 --input-len 256 \
    --profile --profile-activities CUDA_PROFILER
```

### Use Case 2: Profile with Nsys
```bash
nsys profile --force-overwrite=true -o trace \
    --capture-range=cudaProfilerApi \
    python -m sglang.bench_one_batch \
        --model-path meta-llama/Meta-Llama-3-8B-Instruct \
        --batch 1 --input-len 256 \
        --profile --profile-activities CUDA_PROFILER
```

### Use Case 3: Multimodal Generation (Automatic)
```python
# In scheduler.py event_loop() - automatically called:
if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.cudart().cudaProfilerStart()
    # ... execute handler ...
    torch.cuda.cudart().cudaProfilerStop()
```

---

## 📍 Implementation Locations

### PRIMARY (Auto-Profiling)
- **Multimodal Generation Scheduler**
  - File: `sglang/multimodal_gen/runtime/managers/scheduler.py`
  - Lines: 368-386
  - Behavior: Automatically profiles non-warmup requests

### SECONDARY (Manual Control)
- **LLM Scheduler Profiler Mixin**
  - File: `sglang/srt/managers/scheduler_profiler_mixin.py`
  - Lines: start (212-215), stop (322-324)
  - Behavior: Controlled via profiling API

### INFRASTRUCTURE
- **Profile Manager (_ProfilerCudart)**
  - File: `sglang/srt/utils/profile_utils.py`
  - Lines: 335-344
  - Behavior: Pluggable backend system

- **Benchmark Tool**
  - File: `sglang/bench_one_batch.py`
  - Lines: start_profile (93-104), stop_profile (124-141)
  - Behavior: CLI integration

---

## 🔑 Key Features

✅ **Automatic Warmup Exclusion**
- Scheduler checks `req.is_warmup` flag
- Only non-warmup requests trigger profiler
- Warmup latency doesn't pollute profiling data

✅ **Minimal Overhead**
- Just 2 API calls
- No torch.profiler tracing overhead
- Suitable for production profiling

✅ **Multi-GPU Safe**
- Only base_gpu_id issues profiler commands
- Prevents trace file conflicts
- Works in distributed settings

✅ **GPU Synchronization**
- `torch.cuda.synchronize()` called before stop
- Ensures all GPU work is complete
- Prevents incomplete traces

✅ **Stage-Based Profiling**
- Can profile specific stages (prefill/decode)
- ProfileManager supports stage-based triggering
- Useful for identifying bottleneck phases

---

## 📊 Profile Activities

| Activity | Backend | Output | Use Case |
|----------|---------|--------|----------|
| **CPU** | torch.profiler | JSON trace | CPU profiling |
| **GPU** | torch.profiler | JSON trace | General GPU profiling |
| **CUDA_PROFILER** | CUDA Runtime API | nsys trace | Production profiling with nsys |
| **MEM** | torch.cuda | pickle | Memory profiling |
| **RPD** | ROCm (AMD) | JSON trace | ROCM profiling |
| **XPU** | Intel | JSON trace | Intel GPU profiling |

**Note:** Only `CUDA_PROFILER` uses `cudaProfilerStart/Stop`

---

## 🔧 Configuration

### Environment Variables
```bash
# Output directory for torch profiler traces
export SGLANG_TORCH_PROFILER_DIR=/root/profiles

# Enable new stage-based profiling system
export SGLANG_PROFILE_V2=1

# Flag for nsys profiling (internal use)
export SGLANG_NSYS_PROFILING=1
```

### CLI Flags (bench_one_batch.py)
```bash
--profile                      # Enable profiling
--profile-activities           # Which activities (CPU/GPU/CUDA_PROFILER)
--profile-record-shapes        # Record tensor shapes
--profile-stage               # Which stage (all/prefill/decode)
--profile-filename-prefix     # Output file prefix
--profile-start-step          # When to start
--profile-steps               # How many steps
```

---

## 🆚 Comparison: cudaProfilerStart vs torch.profiler

### cudaProfilerStart (CUDA_PROFILER)
✓ Works with `nsys --capture-range=cudaProfilerApi`
✓ Minimal overhead (2 API calls)
✓ No framework initialization captured
✓ Production-suitable
✗ Requires nsys to be running
✗ Need manual warmup exclusion (scheduler does it)

### torch.profiler (CPU/GPU activities)
✓ Standalone, no external tool needed
✓ Rich trace information
✓ Works on all ranks
✗ Higher overhead
✗ Captures framework initialization
✗ Larger trace files
✗ Not compatible with nsys capture-range

---

## 🐛 Troubleshooting

### "Failed to start CUDA profiler"
- Check: `nsys --version` (is nsys installed?)
- Check: Using `--capture-range=cudaProfilerApi` flag
- Fix: `nsys profile --capture-range=cudaProfilerApi python ...`

### Multiple trace files on multi-GPU
- Cause: All GPUs calling cudaProfilerStart()
- Fix: Only base_gpu_id should call (already enforced)
- Verify: Check `base_gpu_id` in server_args

### Trace file is empty/corrupted
- Cause: GPU ops incomplete before cudaProfilerStop()
- Fix: `torch.cuda.synchronize()` is called (already in code)
- Manual fix: Ensure proper GPU sync before stopping

### nsys not capturing GPU operations
- Cause: Application not calling cudaProfilerStart()
- Check: `--profile-activities CUDA_PROFILER` is set
- Verify: Requests not all marked as warmup

---

## 📚 Related Documentation

- [nsight-profiler.md](../../.claude/skills/diffusion-kernel/nsight-profiler.md) - Comprehensive nsys/ncu guide
- [diffusion-benchmark-and-profile.md](../../.claude/skills/diffusion-kernel/diffusion-benchmark-and-profile.md) - Profiling workflows
- SGLang documentation for profiling configuration

---

## 📋 File Summary

| File | Lines | Purpose |
|------|-------|---------|
| CUDAPROFILER_INTEGRATION_ANALYSIS.md | 526 | Complete technical reference |
| CUDAPROFILER_QUICK_REFERENCE.txt | ~300 | Quick lookup guide |
| README_CUDAPROFILER.md | this file | Navigation and index |

---

## ✅ Verification Checklist

When implementing profiling in SGLang:

- [ ] Check if warmup requests should be excluded
- [ ] Verify `torch.cuda.cudart()` is available
- [ ] Use `torch.cuda.synchronize()` before `cudaProfilerStop()`
- [ ] For multi-GPU, guard profiler calls with `base_gpu_id == self.gpu_id`
- [ ] For multi-node, guard profiler calls with `first_rank_in_node`
- [ ] Set `--profile-activities CUDA_PROFILER` for nsys capture-range
- [ ] Wrap with `nsys profile --capture-range=cudaProfilerApi` if using nsys
- [ ] Test with `--profile` flag first before enabling in production

---

## 🎓 Learning Path

1. **Start here:** Read CUDAPROFILER_QUICK_REFERENCE.txt (5 min)
2. **For details:** Read CUDAPROFILER_INTEGRATION_ANALYSIS.md (15 min)
3. **Try it:** Use Example 1-2 from Quick Reference (5 min)
4. **Deep dive:** Review actual code in the 4 implementation locations
5. **Advanced:** Configure custom profiling stages with ProfileManager

---

**Last Updated:** 2026-03-27  
**Analysis Scope:** sglang/python/sglang/ (multimodal_gen + srt)  
**Grep Coverage:** Complete search for cudaProfiler, ProfilerApi, nsys, profiler keywords
