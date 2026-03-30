# NSys Profiling Tools - Quick Index

## 📄 Main Documentation
- **NSYS_PROFILING_COMPREHENSIVE_GUIDE.md** ← **START HERE** (809 lines)
  - Complete overview of all tools, data, and documentation
  - Technical details and code examples
  - Key findings from existing analysis

## 🛠️ Primary Tools

### 1. GPU Trace Analysis Tool
**File**: `examples/profiler/nsys_profile_tools/gputrc2graph.py` (347 lines)
- Converts `.nsys-rep` binary traces to HTML/CSV visualizations
- Handles concurrent kernel deduplication
- Generates stacked bar charts showing kernel category breakdown
- Uses JSON-based kernel classification

**Quick Start**:
```bash
python gputrc2graph.py \
    --in_file trace.nsys-rep,sglang,llama,100 \
    --out_dir results
```

**Output**: 
- `result.html` - Interactive stacked bar chart
- `result.csv` - Kernel-to-category mapping

### 2. Torch Profiler Analyzer
**File**: `zimage_256_256/analyze_profile.py` (784 lines)
- Analyzes torch.profiler traces (JSON/gzip format)
- Categorizes CUDA kernels
- Generates 6 matplotlib charts
- Creates markdown analysis reports

**Quick Start**:
```bash
python analyze_profile.py \
    --trace-dir ./logs \
    --baseline-dir ./zimage_bench \
    --output-dir ./analysis_report
```

**Output**:
- 6 PNG charts (breakdown, comparison, waterfall, etc.)
- ANALYSIS_REPORT.md markdown

### 3. CSV Export Utility
**File**: `zimage_256_256/export_nsys_stats.sh` (33 lines)
- Bash script to export nsys statistics CSVs
- Uses: `nsys stats --report cuda_gpu_kern_sum`

**Quick Start**:
```bash
bash export_nsys_stats.sh
```

## 📊 Existing Analysis Data

**Location**: `zimage_256_256/`

| Type | Files | Purpose |
|------|-------|---------|
| Binary profiles | 8 `.nsys-rep` | Raw GPU traces |
| CSV exports | 6 files | Kernel statistics |
| PNG charts | 10+ images | Visualizations |
| Reports | 3+ markdown | Analysis findings |

### Key Analysis Charts
- `01_pipeline_and_kernel_breakdown.png` - E2E + kernel categories
- `04_fp32_vs_bf16_gemm.png` - **Core bottleneck insight** (most important!)
- `multi_res_*.png` - Resolution-specific analysis

## 📚 Documentation Guides

**All Located in `zimage_256_256/`:**

1. **PROFILER_QUICK_START.md** (294 lines)
   - Quick reference for profiling commands
   - Environment variables
   - ViewingTraces tips

2. **SGLANG_PROFILER_COMPLETE_GUIDE.md** (709 lines)
   - Complete technical architecture
   - Integration points in code
   - Event categories and trace structure
   - Multi-GPU considerations

3. **PROFILER_IMPLEMENTATION_GUIDE.md** (564 lines)
   - 5 profiling patterns with code
   - Pattern 1: Scheduler-based (auto)
   - Pattern 2: Explicit torch.profiler
   - Pattern 3: Multi-GPU safe
   - Pattern 4: Pluggable backends
   - Pattern 5: Stage-based (diffusion)

4. **README.md** (examples/profiler/nsys_profile_tools/)
   - gputrc2graph.py user guide
   - How to collect nsys profiles
   - Examples and extensions

## 🔧 Configuration Files

**`examples/profiler/nsys_profile_tools/sglang_engine_model.json`**
- Kernel name → category mapping
- Regex-based classification
- Supports multiple engines/models
- Extensible via additional JSON files

**Example**:
```json
{
  "sglang": {
    "llama": {
      "gemm|nvjet": "gemm",
      "flash|fmha": "attn",
      ".*": "misc"
    }
  }
}
```

## 🔍 Integration Points in Code

All profiling integrations use `torch.cuda.cudart().cudaProfiler*()`:

1. **Multimodal Scheduler** (`sglang/multimodal_gen/runtime/managers/scheduler.py`, line 368)
   - Automatic per-request profiling
   - Warmup exclusion

2. **LLM Scheduler** (`sglang/srt/managers/scheduler_profiler_mixin.py`, line 212)
   - Multi-GPU safe
   - Activity routing

3. **Bench Tool** (`sglang/bench_one_batch.py`, line 93)
   - Manual torch.profiler control

## 📖 Common Workflows

### 1. Analyze Single Profile
```bash
cd examples/profiler/nsys_profile_tools/
python gputrc2graph.py \
    --in_file /path/to/trace.nsys-rep,sglang,llama,100 \
    --out_dir ./results
# View: results/result.html
```

### 2. Compare Two Profiles
```bash
python gputrc2graph.py \
    --in_file baseline.nsys-rep,sglang,llama,100 \
              optimized.nsys-rep,sglang,llama,98 \
    --out_dir comparison
# View: comparison/result.html (side-by-side)
```

### 3. Export CSV for Spreadsheet
```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
    -o output trace.nsys-rep
# Result: output_cuda_gpu_kern_sum.csv
```

### 4. Analyze Torch Traces
```bash
cd zimage_256_256/
python analyze_profile.py \
    --trace-dir ./logs \
    --baseline-dir ./zimage_bench \
    --output-dir ./my_analysis
# View: my_analysis/*.png + ANALYSIS_REPORT.md
```

## 🎯 Key Findings Summary

**Z-Image-Turbo 256×256 (from existing analysis):**

**Baseline**: E2E = 362ms
- TextEncoding: 81ms (22.4%)
- Denoising: 269ms (74.3%) ⚠️
- Decoding: 10ms (2.7%)

**GPU Kernels**:
- BF16 GEMM: 251ms (53.0%)
- FP32 GEMM: 157ms (33.0%) ⚠️ PRIMARY BOTTLENECK
- Attention: 10ms (2.1%) - NOT a bottleneck

**Optimization Impact**:
- FP32→BF16: -17% E2E
- + FP8 DiT: Additional -5-10%

## 🚀 Advanced Usage

### Add New Kernel Classification
Create new JSON in `examples/profiler/nsys_profile_tools/`:
```json
{
  "my_engine": {
    "my_model": {
      "pattern": "category"
    }
  }
}
```

### Custom Trace Analysis
```python
import gzip, json
from collections import defaultdict

with gzip.open("trace.trace.json.gz", "rb") as f:
    events = json.loads(f.read())["traceEvents"]

# Aggregate by category
times = defaultdict(float)
for e in events:
    if "dur" in e:
        times[e.get("cat")] += e["dur"]

for cat in sorted(times, key=times.get, reverse=True):
    print(f"{cat}: {times[cat]/1e6:.2f}s")
```

## 📋 Troubleshooting

| Issue | Solution |
|-------|----------|
| Huge trace file (>1GB) | Use `nsys profile -d 10` (10 sec max) |
| Empty kernel list | Ensure model is on GPU |
| Slow `nsys stats` | File size ÷ 240 = time (minutes) |
| Missing GPU kernels | Update nsys to match profile version |
| "Unknown engine/model" | Add to sglang_engine_model.json |

## 📞 Quick Command Reference

```bash
# Collect profile
nsys profile -t cuda -o trace python my_model.py

# Export kernel summary
nsys stats --report cuda_gpu_kern_sum --format csv -o out trace.nsys-rep

# Export execution trace
nsys stats --report cuda_gpu_trace --format csv -o out trace.nsys-rep

# View in GUI
nsys-ui trace.nsys-rep

# Analyze with gputrc2graph
python gputrc2graph.py --in_file trace.nsys-rep,sglang,llama,100

# Analyze torch traces
python analyze_profile.py --trace-dir ./logs --output-dir ./results
```

---

**Created**: March 27, 2026  
**Status**: Complete exploration of nsys profiling infrastructure  
**Next Steps**: Use NSYS_PROFILING_COMPREHENSIVE_GUIDE.md for deep-dive documentation
