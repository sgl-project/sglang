# SGLang Profiler - Implementation Patterns & Code Examples

## Overview

This guide shows how torch.profiler is integrated into sglang at key points, with code snippets you can use as reference when adding profiling to your own code.

---

## Pattern 1: Scheduler-Based Profiling (Multimodal)

**Location**: `sglang/multimodal_gen/runtime/managers/scheduler.py` (Lines 368-386)

**Pattern**: Automatic per-request profiling with warmup exclusion

### Implementation

```python
class Scheduler:
    def event_loop(self):
        """Main event loop that processes requests"""
        while True:
            # Get next request
            processed_req = self.get_next_request()
            
            # Check if this is a non-warmup request
            is_non_warmup_req = (
                isinstance(processed_req, Req) and 
                not processed_req.is_warmup
            )
            
            # START PROFILING for non-warmup requests
            if is_non_warmup_req and torch.cuda.is_available():
                torch.cuda.cudart().cudaProfilerStart()
            
            # Execute the handler
            try:
                handler = self.request_handlers.get(type(processed_req))
                if handler:
                    output_batch = handler([processed_req])
                else:
                    output_batch = OutputBatch(
                        error=f"Unknown request type: {type(processed_req)}"
                    )
            except Exception as e:
                output_batch = OutputBatch(error=str(e))
            
            # STOP PROFILING with GPU sync
            if is_non_warmup_req and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure GPU work complete
                torch.cuda.cudart().cudaProfilerStop()
            
            # Return results
            self.return_results(output_batch)
```

### Key Points
- ✅ Automatic: No explicit profiling API calls needed
- ✅ Warmup-aware: Uses `is_warmup` flag to exclude initialization
- ✅ GPU-safe: Calls `synchronize()` before stop
- ✅ No overhead: Only 2 CUDA API calls

### When to Use
- Long-running servers with multiple requests
- Want to exclude warmup overhead automatically
- Need per-request profiling data

---

## Pattern 2: Explicit Profiling with torch.profiler

**Location**: `sglang/bench_one_batch.py` (Lines 93-141)

**Pattern**: Manual control with activity selection and trace export

### Implementation

```python
def start_profile(profile_activities, profile_record_shapes=False):
    """Start profiling based on activities"""
    
    # Handle CUDA_PROFILER activity (nsys integration)
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            print("CUDA Profiler started (nsys will begin capturing)")
        except Exception as e:
            print(f"Failed to start CUDA profiler: {e}")
        return None  # No profiler object for CUDA_PROFILER
    
    # Handle torch.profiler activities
    activities = []
    if "CPU" in profile_activities:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if "GPU" in profile_activities:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    if not activities:
        return None
    
    # Create torch.profiler instance
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=profile_record_shapes,
        with_stack=True,
        use_kineto=True,  # Enable CUDA tracing
    )
    
    profiler.__enter__()
    return profiler


def stop_profile(profiler, profile_activities, trace_filename=None):
    """Stop profiling and save trace"""
    
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            print("CUDA Profiler stopped (nsys should dump traces)")
        except Exception as e:
            print(f"Failed to stop CUDA profiler: {e}")
    
    elif profiler is not None:
        profiler.__exit__(None, None, None)
        
        # Export trace to file
        if trace_filename:
            profiler.export_chrome_trace(trace_filename)
            print(f"Trace saved to: {trace_filename}")
```

### Usage

```python
def main():
    args = parse_args()
    
    # Start profiling
    profiler = start_profile(
        profile_activities=args.profile_activities,
        profile_record_shapes=args.profile_record_shapes
    )
    
    # Run inference
    for i in range(num_steps):
        output = model.forward(...)
        if profiler and not isinstance(profiler, type(None)):
            profiler.step()
    
    # Stop profiling and save
    stop_profile(
        profiler,
        profile_activities=args.profile_activities,
        trace_filename=f"trace_{i}.trace.json.gz"
    )

if __name__ == "__main__":
    main()
```

### Key Points
- ✅ Flexible: Different activity types supported
- ✅ Controllable: Manual start/stop
- ✅ Exportable: Chrome trace format
- ⚠️ Higher overhead: Records all CPU/GPU operations

### When to Use
- Single runs or benchmarks
- Need detailed Python-side overhead
- Can afford extra tracing overhead

---

## Pattern 3: Multi-GPU Safe Profiling

**Location**: `sglang/srt/managers/scheduler_profiler_mixin.py` (Lines 212-324)

**Pattern**: Guard profiler calls with GPU ID checks

### Implementation

```python
class SchedulerProfilerMixin:
    def start_profile(self, activities):
        """Start profiling only on base GPU"""
        
        # Multi-GPU guard: only base GPU should profile
        if "CUDA_PROFILER" in activities:
            if self.gpu_id == get_global_server_args().base_gpu_id:
                logger.info(f"GPU {self.gpu_id}: Starting CUDA profiler")
                torch.cuda.cudart().cudaProfilerStart()
            
            self.profile_in_progress = True
        
        elif "CPU" in activities or "GPU" in activities:
            # torch.profiler can run on all GPUs
            activity_list = []
            if "CPU" in activities:
                activity_list.append(torch.profiler.ProfilerActivity.CPU)
            if "GPU" in activities:
                activity_list.append(torch.profiler.ProfilerActivity.CUDA)
            
            self.profiler = torch.profiler.profile(
                activities=activity_list,
                with_stack=True,
            )
            self.profiler.__enter__()
            self.profile_in_progress = True
    
    def stop_profile(self):
        """Stop profiling only on base GPU"""
        
        if "CUDA_PROFILER" in self.profiler_activities:
            # Only base GPU should stop
            if self.gpu_id == get_global_server_args().base_gpu_id:
                logger.info(f"GPU {self.gpu_id}: Stopping CUDA profiler")
                torch.cuda.synchronize()  # Important!
                torch.cuda.cudart().cudaProfilerStop()
        
        elif self.profiler is not None:
            # torch.profiler can stop on all GPUs
            self.profiler.__exit__(None, None, None)
            self.profiler.export_chrome_trace(
                f"profile_gpu_{self.gpu_id}.trace.json.gz"
            )
        
        self.profile_in_progress = False
```

### Key Points
- ✅ Multi-GPU safe: Only base GPU calls API
- ✅ Multi-rank compatible: Logs GPU ID
- ✅ Thread-safe: Guards prevent race conditions
- ✅ Configurable: Activity-based routing

### When to Use
- Multi-GPU distributed systems
- Want to avoid duplicate traces
- Need coordination between ranks

---

## Pattern 4: Pluggable Profiler Backend

**Location**: `sglang/srt/utils/profile_utils.py` (Lines 335-344)

**Pattern**: Abstraction for multiple profiler backends

### Implementation

```python
class _ProfilerBase(ABC):
    """Abstract base for profilers"""
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass


class _ProfilerTorch(_ProfilerBase):
    """torch.profiler backend"""
    def __init__(self, activities, with_stack, record_shapes):
        self.activities = activities
        self.profiler = torch.profiler.profile(
            activities=activities,
            with_stack=with_stack,
            record_shapes=record_shapes,
        )
    
    def start(self):
        self.profiler.__enter__()
    
    def stop(self):
        self.profiler.__exit__(None, None, None)


class _ProfilerCudart(_ProfilerBase):
    """CUDA profiler (nsys) backend"""
    def __init__(self, first_rank_in_node=True):
        self.first_rank_in_node = first_rank_in_node
    
    def start(self):
        if self.first_rank_in_node:
            logger.info("Call cudaProfilerStart")
            torch.cuda.cudart().cudaProfilerStart()
    
    def stop(self):
        if self.first_rank_in_node:
            logger.info("Call cudaProfilerStop")
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()


class _ProfilerMemory(_ProfilerBase):
    """Memory profiling backend"""
    def start(self):
        torch.cuda.reset_peak_memory_stats()
    
    def stop(self):
        peak_mem = torch.cuda.max_memory_allocated()
        logger.info(f"Peak memory: {peak_mem / 1e9:.2f} GB")


class _ProfilerList(_ProfilerBase):
    """Composite profiler (multiple backends)"""
    def __init__(self, profilers):
        self.profilers = profilers
    
    def start(self):
        for p in self.profilers:
            p.start()
    
    def stop(self):
        for p in self.profilers:
            p.stop()


class ProfileManager:
    """Factory for profilers"""
    
    @staticmethod
    def create(activities, with_stack, record_shapes):
        """Create appropriate profiler(s)"""
        inners = []
        
        if ("CPU" in activities) or ("GPU" in activities):
            inners.append(_ProfilerTorch(
                [getattr(torch.profiler.ProfilerActivity, a) 
                 for a in ["CPU", "GPU"] if a in activities],
                with_stack=with_stack,
                record_shapes=record_shapes
            ))
        
        if "MEM" in activities:
            inners.append(_ProfilerMemory())
        
        if "CUDA_PROFILER" in activities:
            inners.append(_ProfilerCudart())
        
        if "RPD" in activities:  # ROCM
            inners.append(_ProfilerRPD())
        
        if len(inners) == 1:
            return inners[0]
        return _ProfilerList(inners)


# Usage
profiler = ProfileManager.create(
    activities=["CPU", "GPU", "MEM"],
    with_stack=True,
    record_shapes=False
)

profiler.start()
# ... do work ...
profiler.stop()
```

### Key Points
- ✅ Extensible: Easy to add new backends
- ✅ Composable: Mix and match profilers
- ✅ Flexible: Support multiple GPUs/frameworks
- ✅ Clean: Single interface for all backends

### When to Use
- Complex profiling pipelines
- Need flexibility in profiler selection
- Multiple profiling backends required

---

## Pattern 5: Stage-Based Profiling (Diffusion)

**Location**: `sglang/multimodal_gen/runtime/utils/profiler.py`

**Pattern**: Profile specific pipeline stages

### Implementation

```python
class SGLDiffusionProfiler:
    """Profiles specific stages of diffusion pipeline"""
    
    def __init__(
        self,
        request_id=None,
        full_profile=False,      # Profile all stages?
        num_steps=None,          # Num denoising steps
        log_dir="./logs",
    ):
        self.request_id = request_id
        self.full_profile = full_profile
        self.log_dir = log_dir
        self.profilers = {}
        
        if full_profile:
            # Profile all stages: TextEnc, Denoise, Decode
            self.profilers["full"] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
            )
            self.profilers["full"].__enter__()
        else:
            # Profile only denoising steps
            self.num_active_steps = num_steps if num_steps else 0
            self.denoising_profiler = None
    
    def step_denoising_step(self):
        """Called after each denoising step"""
        if not self.full_profile:
            if self.num_active_steps > 0:
                if not self.denoising_profiler:
                    self.denoising_profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA],
                    )
                    self.denoising_profiler.__enter__()
                
                self.denoising_profiler.step()
                self.num_active_steps -= 1
            else:
                # Profiling complete
                self.stop()
    
    def stop(self, dump_rank=0):
        """Stop all profilers and save traces"""
        import torch.distributed as dist
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == dump_rank:
            for stage, prof in self.profilers.items():
                if prof is not None:
                    prof.__exit__(None, None, None)
                    filename = f"{self.log_dir}/{self.request_id}_{stage}.trace.json.gz"
                    prof.export_chrome_trace(filename)
                    logger.info(f"Saved {stage} trace: {filename}")


# Usage
profiler = SGLDiffusionProfiler(
    request_id="gen_001",
    full_profile=False,
    num_steps=9  # Profile 9 denoising steps
)

for step in range(num_denoising_steps):
    # ... denoising logic ...
    profiler.step_denoising_step()

profiler.stop(dump_rank=0)
```

### Key Points
- ✅ Stage-aware: Profile specific parts of pipeline
- ✅ Distributed: Only save on rank 0
- ✅ Flexible: Full or partial profiling
- ✅ Memory-efficient: Don't profile all steps

### When to Use
- Long inference pipelines (>10s)
- Want to focus on specific stages
- Memory/storage constraints
- Distributed diffusion generation

---

## Integration Checklist

When adding profiling to your code:

- [ ] **Identify profiling point**: Where should profiling start/stop?
- [ ] **Choose activity types**: CPU, GPU, or both?
- [ ] **Handle multi-GPU**: Add GPU ID guards if needed
- [ ] **Add GPU sync**: Call `torch.cuda.synchronize()` before stop
- [ ] **Export trace**: Use `export_chrome_trace()` for JSON output
- [ ] **Handle exceptions**: Wrap with try/except for robustness
- [ ] **Add logging**: Log start/stop for debugging
- [ ] **Test with small data**: Profile short runs first
- [ ] **Document flags**: Add CLI flags for profiling control
- [ ] **Consider overhead**: Choose appropriate activity types

---

## Performance Impact

| Activity | Overhead | Use Case |
|----------|----------|----------|
| **None** | 0% | Production |
| **GPU only** | 2-5% | Standard profiling |
| **CPU+GPU** | 10-15% | Detailed analysis |
| **CUDA_PROFILER** | <1% | nsys integration |
| **CPU+GPU+MEM** | 15-25% | Full debugging |

**Recommendation**: Use `GPU` only for production profiling, add `CPU` during development.

---

## Trace Analysis Template

```python
import gzip
import json
from collections import defaultdict, Counter

def analyze_trace(trace_file):
    """Analyze a profiler trace"""
    
    with gzip.open(trace_file, 'rb') as f:
        data = json.loads(f.read())
    
    events = data.get("traceEvents", [])
    
    # 1. Event statistics
    cat_count = Counter(e.get("cat") for e in events)
    print("Event counts by category:")
    for cat, count in cat_count.most_common(10):
        print(f"  {cat}: {count}")
    
    # 2. Time breakdown
    cat_time = defaultdict(float)
    for e in events:
        if "dur" in e:
            cat_time[e.get("cat", "unknown")] += e["dur"]
    
    total_time = sum(cat_time.values())
    print(f"\nTime breakdown (total {total_time/1e6:.2f}s):")
    for cat in sorted(cat_time, key=lambda x: cat_time[x], reverse=True):
        pct = cat_time[cat] / total_time * 100
        print(f"  {cat}: {cat_time[cat]/1e6:.2f}s ({pct:.1f}%)")
    
    # 3. Top kernels
    kernels = [e for e in events if e.get("cat") == "kernel"]
    kernel_time = defaultdict(lambda: {"total": 0, "count": 0})
    for k in kernels:
        name = k.get("name", "unknown")
        dur = k.get("dur", 0)
        kernel_time[name]["total"] += dur
        kernel_time[name]["count"] += 1
    
    print("\nTop 10 kernels by time:")
    for name, info in sorted(
        kernel_time.items(), 
        key=lambda x: x[1]["total"], 
        reverse=True
    )[:10]:
        print(f"  {name}: {info['total']/1e6:.3f}s ({info['count']} calls)")

# Run analysis
analyze_trace("trace_0.trace.json.gz")
```

---

## References

- **Official torch.profiler docs**: https://pytorch.org/docs/stable/profiler.html
- **Chrome DevTools tracing**: chrome://tracing
- **Perfetto trace viewer**: https://ui.perfetto.dev
- **CUDA profiling guide**: `.claude/skills/diffusion-kernel/nsight-profiler.md`

