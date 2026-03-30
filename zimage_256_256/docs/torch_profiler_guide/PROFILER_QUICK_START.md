# SGLang torch.profiler - Quick Start Cheat Sheet

## TL;DR

```bash
# 1. Profile your model (generates JSON trace)
python -m sglang.bench_one_batch \
    --model-path z-image-turbo \
    --batch 1 \
    --profile \
    --profile-activities CPU GPU

# 2. View trace in Chrome
#    Open: chrome://tracing
#    Load: sglang_bench_one_batch_model_trace_0.trace.json.gz

# 3. Analyze with script
python analyze_profile.py \
    --trace-dir ./logs \
    --output-dir ./analysis_report
```

---

## 1️⃣ Basic Profiling

### Profile CPU+GPU
```bash
python -m sglang.bench_one_batch \
    --profile \
    --profile-activities CPU GPU
```

### Profile GPU Only (smaller file)
```bash
python -m sglang.bench_one_batch \
    --profile \
    --profile-activities GPU
```

### Profile with nsys (minimal overhead)
```bash
nsys profile --capture-range=cudaProfilerApi -o trace \
    python -m sglang.bench_one_batch \
        --profile-activities CUDA_PROFILER
```

---

## 2️⃣ Diffusion Model Profiling

```bash
python -m sglang.gen \
    --model-path z-image-turbo \
    --prompt "A beautiful sunset" \
    --height 256 --width 256 \
    --profile \
    --profile-activities GPU

# Output: sglang_gen_model_trace_0.trace.json.gz
```

---

## 3️⃣ Environment Variables

```bash
# Set output directory
export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles

# Enable stage-based profiling
export SGLANG_PROFILE_V2=1
```

---

## 4️⃣ What Each Activity Captures

| Activity | Captures | File Size | Use Case |
|----------|----------|-----------|----------|
| **GPU** | CUDA kernels, memory ops | 10-50MB | Fastest; GPU analysis |
| **CPU** | Python functions, ATen ops | 50-100MB | Python overhead |
| **CPU+GPU** | Both | 100-500MB | Complete analysis |
| **CUDA_PROFILER** | GPU only (via nsys) | 1-500MB | Minimal overhead |
| **MEM** | Memory allocations | 50-200MB | Memory leaks |

---

## 5️⃣ Viewing Traces

### Chrome DevTools (JSON traces)
```
1. Open: chrome://tracing
2. Click "Load" button
3. Select: sglang_*.trace.json.gz
```

### Perfetto (web-based)
```
1. Go to: https://ui.perfetto.dev
2. Upload: sglang_*.trace.json.gz
```

### Nsight Systems (nsys traces)
```bash
nsys stats trace.nsys-rep
nsys-ui trace.nsys-rep
```

---

## 6️⃣ Event Categories in Traces

### Python-Side Overhead (CPU time)
- `python_function` - Python execution
- `cpu_op` - ATen CPU operations
- `overhead` - Profiler overhead

### GPU-Side Overhead (Kernel launch gaps)
- `cuda_driver` - Kernel launch (~1-10μs per kernel)
- `cuda_runtime` - cudaMalloc, cudaMemcpy, etc.

### GPU Computation
- `kernel` - Actual CUDA kernel execution
- `gpu_memcpy` - GPU memory transfers
- `gpu_memset` - GPU memory initialization

---

## 7️⃣ Analyzing Output

### Load and inspect trace
```python
import gzip, json

with gzip.open("sglang_*.trace.json.gz", "rb") as f:
    data = json.loads(f.read())

# Get event categories
categories = set()
for event in data["traceEvents"]:
    categories.add(event.get("cat"))
print(sorted(categories))

# Count events by category
from collections import Counter
cat_count = Counter(e.get("cat") for e in data["traceEvents"])
print(cat_count.most_common(10))
```

### Use analyze_profile.py
```bash
python analyze_profile.py \
    --trace-dir ./logs \
    --baseline-dir ./zimage_bench \
    --output-dir ./analysis_report

# Generates:
# - PNG charts with breakdown
# - Markdown report
# - Kernel statistics
```

---

## 8️⃣ Finding Bottlenecks

### CPU Overhead
```
Look for large "python_function" events
→ Suggests framework or data prep overhead
→ Optimize Python code or use batch operations
```

### Kernel Launch Gaps
```
Look for gaps between "cuda_driver" and "kernel" events
→ Typically 1-10μs per kernel
→ Total gap = num_kernels × launch_overhead
```

### Large Kernels
```
Sort by "kernel" event duration
→ Top 5 kernels = 80% of GPU time
→ Focus optimization on these
```

### Memory Overhead
```
Add "gpu_memcpy" and "gpu_memset" times
→ If >10% of total, memory transfer is bottleneck
→ Use fused kernels or reduce data movement
```

---

## 9️⃣ Multi-GPU Profiling

```bash
# Profile on specific GPU
CUDA_VISIBLE_DEVICES=0 python -m sglang.bench_one_batch \
    --profile-activities GPU

# With distributed training
torchrun --nproc_per_node=8 -m sglang.bench_one_batch \
    --profile-activities GPU
# ⚠️ Only base GPU (gpu_id=0) records trace
```

---

## 🔟 CLI Flags Reference

```bash
--profile                           # Enable profiling
--profile-activities CPU GPU        # Which activities
--profile-record-shapes             # Include tensor shapes
--profile-stage all                 # all/prefill/decode
--profile-filename-prefix myrun     # Output prefix
--profile-start-step 0              # When to start
--profile-steps 10                  # How many steps
```

---

## ❌ Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Huge file (>500MB) | Profiling too many steps | Use `--profile-steps 10` |
| Empty trace | GPU work not complete | Already fixed (torch.cuda.sync) |
| CUDA_PROFILER fails | nsys not running | Use `nsys profile --capture-range=cudaProfilerApi` |
| No Python overhead | Not using CPU activity | Use `--profile-activities CPU GPU` |
| Trace not loading | Wrong format | Use `.trace.json.gz` files with Chrome |

---

## 📚 Key Files to Know

| File | Purpose |
|------|---------|
| `bench_one_batch.py` | Entry point for profiling |
| `scheduler.py` | Where profiler calls happen |
| `analyze_profile.py` | Trace analysis tool |
| `profile_utils.py` | Profile manager backend |
| `SGLANG_PROFILER_COMPLETE_GUIDE.md` | Full technical reference |

---

## 🚀 One-Liner Examples

```bash
# Profile and view immediately
python -m sglang.bench_one_batch --profile-activities GPU && \
    python -c "print('Open: chrome://tracing')"

# Profile + analyze
python -m sglang.bench_one_batch --profile-activities GPU && \
    python analyze_profile.py --trace-dir ./logs --output-dir ./analysis

# nsys profiling (minimal overhead)
nsys profile --capture-range=cudaProfilerApi -o bench \
    python -m sglang.bench_one_batch --profile-activities CUDA_PROFILER && \
    nsys stats bench.nsys-rep

# Profile diffusion
SGLANG_TORCH_PROFILER_DIR=/tmp/profiles \
    python -m sglang.gen --model-path z-image-turbo \
    --prompt "test" --height 256 --width 256 --profile-activities GPU
```

---

## 📊 Expected Output

```
benchmark output:
  Steps: 100
  Avg latency: 42.5 ms
  Throughput: 23.5 req/s

profiling output:
  ✅ Trace saved: sglang_bench_one_batch_model_trace_0.trace.json.gz (15.2 MB)
  GPU kernel time: 38.2 ms (89.8%)
  CPU overhead: 4.3 ms (10.2%)
  
View trace at: chrome://tracing
```

---

**For detailed information, see: SGLANG_PROFILER_COMPLETE_GUIDE.md**
