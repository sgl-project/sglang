# SGLang CudaProfilerApi Integration Analysis

## Executive Summary

The sglang codebase implements CUDA profiler integration (`torch.cuda.cudart().cudaProfilerStart/Stop`) primarily for compatibility with **Nsight Systems (nsys) profiling tool** using the `--capture-range=cudaProfilerApi` flag. This allows profiling to be triggered programmatically within the inference pipeline, enabling precise capture windows while excluding warmup phases and framework initialization.

---

## 1. CudaProfilerApi Implementation Locations

### 1.1 Multimodal Generation Scheduler (PRIMARY)
**File**: `/data/home/rhyshen/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py`

**Lines**: 368-386

```python
# Start CUDA profiler for non-warmup requests when using
# nsys --capture-range=cudaProfilerApi, so warmup is excluded
is_non_warmup_req = (
    isinstance(processed_req, Req) and not processed_req.is_warmup
)
if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.cudart().cudaProfilerStart()

handler = self.request_handlers.get(type(processed_req))
if handler:
    output_batch = handler(reqs)
else:
    output_batch = OutputBatch(
        error=f"Unknown request type: {type(processed_req)}"
    )

if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
```

**Context**: This is in the main event loop (`event_loop()` method) of the Scheduler class
- **When Called**: For every non-warmup request in the inference pipeline
- **Key Feature**: Warmup requests are explicitly excluded to focus profiling on actual inference
- **Synchronization**: `torch.cuda.synchronize()` is called before `cudaProfilerStop()` to ensure GPU operations complete

### 1.2 LLM Runtime Scheduler Profiler Mixin
**File**: `/data/home/rhyshen/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_profiler_mixin.py`

**Lines for start**: 212-215
```python
if "CUDA_PROFILER" in activities:
    if self.gpu_id == get_global_server_args().base_gpu_id:
        torch.cuda.cudart().cudaProfilerStart()
    self.profile_in_progress = True
```

**Lines for stop**: 322-324
```python
if "CUDA_PROFILER" in self.profiler_activities:
    if self.gpu_id == get_global_server_args().base_gpu_id:
        torch.cuda.cudart().cudaProfilerStop()
```

**Context**: Part of the `SchedulerProfilerMixin` class, integrated into the main Scheduler
- **When Called**: When profiling is explicitly initiated via profile requests
- **Multi-GPU Handling**: Only runs on the base GPU (`base_gpu_id`) to avoid multiple trace collection
- **Profile Activities**: Only triggered when `"CUDA_PROFILER"` is in the activities list

### 1.3 Profiling Utilities (Profile Manager)
**File**: `/data/home/rhyshen/sgl-workspace/sglang/python/sglang/srt/utils/profile_utils.py`

**Lines for _ProfilerCudart class**: 335-344

```python
class _ProfilerCudart(_ProfilerConcreteBase):
    def start(self):
        if self.first_rank_in_node:
            logger.info(f"Call cudaProfilerStart")
            torch.cuda.cudart().cudaProfilerStart()

    def stop(self):
        if self.first_rank_in_node:
            logger.info(f"Call cudaProfilerStop")
            torch.cuda.cudart().cudaProfilerStop()
```

**Context**: Part of a pluggable profiler system supporting multiple profiling backends
- **Multi-Node Handling**: Only calls on the first rank in node
- **Logging**: Explicit logging when profiler starts/stops
- **Integration**: Works with the `ProfileManager` class for stage-based profiling

### 1.4 Bench Utilities (bench_one_batch.py)
**File**: `/data/home/rhyshen/sgl-workspace/sglang/python/sglang/bench_one_batch.py`

**Lines 93-104 (start_profile)**:
```python
def start_profile(profile_activities, profile_record_shapes=False, rank_print=print):
    """
    Abstracted function to start profiling based on profile_activities.
    Returns profiler object (or None).
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            rank_print("CUDA Profiler started (nsys will begin capturing)")
        except Exception as e:
            rank_print(f"Failed to start CUDA profiler: {e}")
        return None
```

**Lines 124-141 (stop_profile)**:
```python
def stop_profile(
    profiler,
    profile_activities,
    rank_print=print,
    save_trace=False,
    trace_filename=None,
    stage=None,
):
    """
    Abstracted function to stop profiling based on profile_activities.
    Optionally saves trace results and prints completion messages.
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            rank_print("CUDA Profiler stopped (nsys should dump traces)")
        except Exception as e:
            rank_print(f"Failed to stop CUDA profiler: {e}")
    elif profiler is not None:
        profiler.stop()
```

**Context**: Used for benchmarking with profiling capability
- **CLI Integration**: Part of `bench_one_batch` benchmarking tool
- **Separation**: Keeps CUDA_PROFILER path separate from torch.profiler path
- **User Feedback**: Clear messages indicating when profiling starts/stops

---

## 2. How Capture-Range CudaProfilerApi Integration Works

### 2.1 The nsys --capture-range=cudaProfilerApi Mechanism

```bash
# Example usage:
nsys profile --force-overwrite=true -o bench_one_batch \
    --capture-range=cudaProfilerApi \
    python -m sglang.bench_one_batch \
        --model-path meta-llama/Meta-Llama-3-8B-Instruct \
        --batch 1 --input-len 256 --profile --profile-activities CUDA_PROFILER
```

**How it works**:
1. `nsys` starts the system tracer in the background
2. The application runs and calls `cudaProfilerStart()` 
3. nsys detects this CUDA API call and begins capturing GPU operations
4. When `cudaProfilerStop()` is called, nsys stops capturing
5. Only GPU operations between the API calls are recorded

### 2.2 Benefits

| Benefit | Explanation |
|---------|-------------|
| **Precise Capture Window** | Excludes model loading, initialization, and warmup phases |
| **Warmup Exclusion** | Scheduler explicitly skips profiler calls for warmup requests |
| **Reduced Trace Size** | Only captures relevant inference operations, not framework overhead |
| **Flexible Profiling** | Can be combined with prefill/decode stage tracking |
| **Production-like** | Minimal overhead - just two API calls around the critical path |

### 2.3 Integration Points in Inference Pipeline

```
┌─ Request Receives ──────────┐
│                             │
├─ Check if warmup?           │
│  ├─ YES: Skip profiling     │
│  └─ NO: Call cudaProfilerStart()
│                             │
├─ Execute Handler           │
│  └─ Handle generation       │
│                             │
├─ Synchronize GPU           │
│  └─ torch.cuda.synchronize()
│                             │
└─ Call cudaProfilerStop()   │
```

---

## 3. Multi-GPU and Distributed Considerations

### 3.1 Single-GPU Profile Calls

In `scheduler_profiler_mixin.py`:
```python
if self.gpu_id == get_global_server_args().base_gpu_id:
    torch.cuda.cudart().cudaProfilerStart()
```

**Rationale**: Only the primary GPU should issue profiler commands to avoid conflicting trace files

### 3.2 Multi-Node Awareness

In `profile_utils.py` (_ProfilerCudart):
```python
if self.first_rank_in_node:
    logger.info(f"Call cudaProfilerStart")
    torch.cuda.cudart().cudaProfilerStart()
```

**Rationale**: Avoids duplicate CUDA profiler calls on multi-node setups

### 3.3 Rank Printing

In `bench_one_batch.py`:
```python
rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
```

Ensures status messages only print once per profiling session

---

## 4. Profile Activities Configuration

### 4.1 Activity Types Supported

```python
# From bench_one_batch.py BenchArgs
profile_activities: Tuple[str] = ("CPU", "GPU")  # Default

# Choices available:
choices=["CPU", "GPU", "CUDA_PROFILER", "XPU"]
```

### 4.2 Activity Routing

```python
# torch.profiler activities (CPU/GPU/XPU)
if "CPU" in profile_activities:
    activities.append(torch.profiler.ProfilerActivity.CPU)
if "GPU" in profile_activities:
    activities.append(torch.profiler.ProfilerActivity.CUDA)

# CUDA profiler (nsys integration)
if "CUDA_PROFILER" in profile_activities:
    torch.cuda.cudart().cudaProfilerStart()
```

### 4.3 CLI Usage Examples

```bash
# Using torch profiler (generates JSON trace)
python -m sglang.bench_one_batch \
    --profile \
    --profile-activities CPU GPU

# Using CUDA profiler (works with nsys)
python -m sglang.bench_one_batch \
    --profile \
    --profile-activities CUDA_PROFILER

# Using with nsys profiling
nsys profile --capture-range=cudaProfilerApi -o trace \
    python -m sglang.bench_one_batch \
        --profile-activities CUDA_PROFILER
```

---

## 5. Profile Manager and Stage-Based Profiling

### 5.1 Profile Manager Architecture

**File**: `/data/home/rhyshen/sgl-workspace/sglang/python/sglang/srt/utils/profile_utils.py`

The `ProfileManager` provides a pluggable profiler system:

```python
class _ProfilerBase(ABC):
    @staticmethod
    def create(activities, with_stack, record_shapes, **kwargs):
        inners = []
        if ("CPU" in activities) or ("GPU" in activities):
            inners.append(_ProfilerTorch(...))
        if "MEM" in activities:
            inners.append(_ProfilerMemory(...))
        if "CUDA_PROFILER" in activities:
            inners.append(_ProfilerCudart(...))
        if "RPD" in activities:  # for ROCM
            inners.append(_ProfilerRPD(...))
        return _ProfilerList(inners)
```

### 5.2 Stage-Based Triggering

```python
class _StageBasedTrigger:
    def step(self, stage: str):
        # Increment counter for current stage
        if (s := self.running_state) is not None:
            s.curr_count += 1
        
        # Maybe stop
        if ((s := self.running_state) is not None) and (
            (s.curr_count > self.stage_configs[s.curr_stage].target_count)
            or (stage != s.curr_stage)
        ):
            del self.stage_configs[s.curr_stage]
            self.running_state = None
            self.on_stop()
        
        # Maybe start
        if (self.running_state is None) and (stage in self.stage_configs):
            self.running_state = self._RunningState(...)
            self.on_start(stage=stage)
```

**Supported Stages**: `"prefill"` and `"decode"`

---

## 6. Environment Variables and Configuration

### 6.1 Torch Profiler Output Directory

```python
output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
```

**Default**: `/tmp` if not specified

### 6.2 Profile V2 Detection

```python
if envs.SGLANG_PROFILE_V2.get():
    self._profile_manager = ProfileManager(...)
```

Enables the new stage-based profiling system if the environment variable is set.

### 6.3 nsys Integration Flag

From `srt/utils/bench_utils.py`:
```python
using_nsys = int(os.environ.get("SGLANG_NSYS_PROFILING", 0))
```

**Purpose**: Disable torch.profiler when nsys profiling is active to avoid conflicts

---

## 7. Multimodal Generation (Diffusion) Profiler

**File**: `/data/home/rhyshen/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/utils/profiler.py`

### 7.1 SGLDiffusionProfiler Class

```python
class SGLDiffusionProfiler:
    def __init__(
        self,
        request_id: str | None = None,
        rank: int = 0,
        full_profile: bool = False,
        num_steps: int | None = None,
        num_inference_steps: int | None = None,
        log_dir: str | None = None,
    ):
```

**Features**:
- Full pipeline profiling or denoising-stage-only profiling
- Scheduled profiling with warmup/active step configuration
- Support for NPU (Ascend) devices via torch_npu
- Trace integrity checking

### 7.2 Step-Based Profiling

```python
def step_denoising_step(self):
    if not self.full_profile:
        if self.num_active_steps >= 0:
            self._step()
            self.num_active_steps -= 1
        else:
            # early exit when enough steps are captured
            self.stop(dump_rank=0)
```

---

## 8. Key Differences: cudaProfilerStart vs torch.profiler

| Aspect | cudaProfilerStart | torch.profiler |
|--------|------------------|-----------------|
| **API** | `torch.cuda.cudart().cudaProfilerStart()` | `torch.profiler.profile()` |
| **Nsys Integration** | Works with `--capture-range=cudaProfilerApi` | Generates Chrome trace JSON |
| **Overhead** | Minimal (2 API calls) | Higher (records all CPU/GPU ops) |
| **Warmup Exclusion** | Manual (via conditional checks) | Via `schedule()` parameter |
| **Multi-GPU** | Only call on base GPU | Works on all ranks |
| **Output Format** | nsys trace (.nsys-rep) | JSON + gzip (.trace.json.gz) |
| **Runtime Compatibility** | Requires nsys to be running | Standalone, no external tool needed |

---

## 9. Search Results Summary

### Files with cudaProfilerStart/Stop Implementations:
1. ✅ `multimodal_gen/runtime/managers/scheduler.py` (Primary - Lines 374, 386)
2. ✅ `srt/managers/scheduler_profiler_mixin.py` (Lines 214, 324)
3. ✅ `srt/utils/profile_utils.py` (_ProfilerCudart - Lines 339, 344)
4. ✅ `bench_one_batch.py` (Lines 100, 138)

### Files with nsys/capture-range References:
1. ✅ `bench_one_batch.py` (Line 18, 101, 139)
2. ✅ `bench_offline_throughput.py` (Line 180)
3. ✅ `srt/utils/bench_utils.py` (Lines 56, 73, 81, 92, 96)
4. ✅ `.claude/skills/diffusion-kernel/nsight-profiler.md` (Documentation)
5. ✅ `.claude/skills/diffusion-kernel/diffusion-benchmark-and-profile.md` (Documentation)

### GPU Memory and Graph Runner References:
- `srt/model_executor/cuda_graph_runner.py` (Lines 763-790 - "capture_range" variable for tqdm)
- `srt/model_executor/piecewise_cuda_graph_runner.py`
- `srt/model_executor/cpu_graph_runner.py`

---

## 10. Integration Flow Diagram

```
┌─ CLI / Python API / HTTP Server ──────────────────┐
│                                                   │
│  python -m sglang.bench_one_batch                 │
│    --profile-activities CUDA_PROFILER             │
│    nsys profile --capture-range=cudaProfilerApi   │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
         ┌─ Scheduler / GPU Worker ───┐
         │                            │
         │  Event Loop:               │
         │  1. Receive Request        │
         │  2. Check warmup flag      │
         │  3. If not warmup:         │
         │     cudaProfilerStart()◄───┼─ nsys captures from here
         │  4. Execute inference      │
         │  5. torch.cuda.sync()      │
         │  6. cudaProfilerStop() ◄───┼─ nsys stops capturing
         │  7. Return results         │
         └────────────────────────────┘
                    │
                    ▼
         ┌─ Nsight Systems (nsys) ────┐
         │                            │
         │  Output: trace.nsys-rep    │
         │  Captures only the marked  │
         │  range between API calls   │
         └────────────────────────────┘
```

---

## 11. Configuration Examples

### 11.1 Multimodal Generation Server with Profiling

```python
# In scheduler.py event_loop(), lines 368-386
# Automatically called for every non-warmup request
if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.cudart().cudaProfilerStart()
    # ... execute handler ...
    torch.cuda.cudart().cudaProfilerStop()
```

### 11.2 LLM Runtime with Profiling Request

```python
# In scheduler_profiler_mixin.py start_profile(), line 214
if "CUDA_PROFILER" in activities:
    if self.gpu_id == get_global_server_args().base_gpu_id:
        torch.cuda.cudart().cudaProfilerStart()
    self.profile_in_progress = True
```

### 11.3 Benchmark with CUDA Profiler

```bash
# Run with CUDA profiler activity
python -m sglang.bench_one_batch \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --batch 1 --input-len 256 \
    --profile --profile-activities CUDA_PROFILER

# With nsys wrapper
nsys profile --force-overwrite=true -o bench_trace \
    --capture-range=cudaProfilerApi \
    python -m sglang.bench_one_batch \
        --model-path meta-llama/Meta-Llama-3-8B-Instruct \
        --batch 1 --input-len 256 \
        --profile --profile-activities CUDA_PROFILER
```

---

## Summary Table

| Component | Location | Purpose | Key Method |
|-----------|----------|---------|------------|
| **Multimodal Scheduler** | `multimodal_gen/.../scheduler.py` | Auto-profile non-warmup reqs | `event_loop()` |
| **LLM Scheduler Mixin** | `srt/.../scheduler_profiler_mixin.py` | Manual profile control | `start_profile()` / `stop_profile()` |
| **Profile Manager** | `srt/utils/profile_utils.py` | Pluggable profiler backend | `ProfileManager` / `_ProfilerCudart` |
| **Benchmark Tool** | `bench_one_batch.py` | CLI profiling support | `start_profile()` / `stop_profile()` |
| **Diffusion Profiler** | `multimodal_gen/.../profiler.py` | Stage-specific profiling | `SGLDiffusionProfiler` |
| **Documentation** | `.claude/skills/.../nsight-profiler.md` | Usage guide | nsys CLI examples |

---

## Recommendations for Usage

1. **For LLM inference profiling**: Use multimodal scheduler's automatic profiling (excludes warmup automatically)
2. **For detailed kernel analysis**: Combine with `nsys profile --capture-range=cudaProfilerApi`
3. **For multi-GPU setups**: Ensure only base_gpu_id calls cudaProfilerStart/Stop
4. **For reducing trace size**: Use `CUDA_PROFILER` instead of `GPU` activity to capture only GPU operations
5. **For stage-specific analysis**: Use `profile_by_stage=True` with `ProfileManager` for prefill/decode separation

