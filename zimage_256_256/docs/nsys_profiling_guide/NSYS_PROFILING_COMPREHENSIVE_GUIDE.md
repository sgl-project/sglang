# Complete NSys Profiling Tools & Analysis in SGLang Codebase

**Date**: March 27, 2026  
**Author**: Comprehensive exploration of nsys profiling infrastructure  
**Scope**: SGLang codebase - profiler tools, analysis scripts, documentation

---

## 1. Executive Summary

The SGLang codebase contains a sophisticated nsys profiling infrastructure with:

- **Primary Tool**: `gputrc2graph.py` - Converts NVIDIA nsys GPU traces to HTML/CSV visualizations
- **Analysis Script**: `analyze_profile.py` - Torch profiler trace analysis with CUDA kernel breakdown
- **Export Utility**: `export_nsys_stats.sh` - Bash script for nsys stats CSV export
- **Real Data**: 7 `.nsys-rep` binary profiles + 6 CSV export reports + 5+ PNG analysis charts
- **Documentation**: 3 comprehensive markdown guides + example JSON configurations

---

## 2. Primary Tools & Scripts

### 2.1 `gputrc2graph.py` - Main Nsys GPU Trace Analysis Tool

**Location**: `examples/profiler/nsys_profile_tools/gputrc2graph.py` (347 lines)

**Purpose**: Process NVIDIA Nsys GPU trace files (`.nsys-rep`) and generate:
- HTML stacked bar charts showing GPU kernel time breakdown
- CSV mapping of kernel names to categories
- Non-overlapping GPU cycles (deduplicates concurrent kernels)

**Key Methods**:

```python
class GPUTrace2Graph:
    def gen_sum_file(file, nsys_cmd):
        """Generate summary CSV from nsys trace using:
        $ nsys stats -r cuda_gpu_trace <file> -o <output>
        """
        
    def sum_non_overlapping_intervals(df):
        """Calculate non-overlapped GPU cycles using vectorized pandas ops
        - Handles kernel concurrency
        - Returns "Elapsed Time (ns)" column with overlap-adjusted duration
        """
        
    def anno_gpu_kernname(df, mapping):
        """Classify kernels using regex patterns from JSON config
        - Maps kernel names to high-level categories (GEMM, Attention, etc.)
        - Uses engine_model.json for classification rules
        """
        
    def make_html(df, output_dir, title):
        """Generate interactive visualization:
        - Plotly stacked histogram by Model_Engine
        - Pivot table showing category breakdown per config
        """
```

**Usage**:

```bash
# Single profile
python gputrc2graph.py \
    --in_file run1.nsys-rep,sglang,llama,100 \
    --out_dir results \
    --title "My Analysis"

# Multiple profiles (for comparison)
python gputrc2graph.py \
    --in_file run1.nsys-rep,sglang,llama,100 \
                run2.nsys-rep,sglang,gpt-oss,102 \
    --out_dir results

# Custom nsys path
python gputrc2graph.py \
    --in_file trace.nsys-rep,sglang,llama,100 \
    --nsys_cmd /usr/bin/nsys
```

**Key Parameters**:

| Parameter | Example | Notes |
|-----------|---------|-------|
| `--in_file` | `trace.nsys-rep,sglang,llama,100` | Format: `<nsys-rep>,<engine>,<model>,<elapsed_sec_without_profiling>` |
| `--out_dir` | `./results` | Output location for HTML/CSV |
| `--title` | `"BF16 vs FP8"` | Chart title |
| `--nsys_cmd` | `nsys` (default) or `/path/to/nsys` | nsys binary location |

**Output Files**:
- `result.html` - Interactive stacked bar chart with kernel category breakdown
- `result.csv` - Kernel-to-category mapping table

**Technical Details**:

1. **Calls nsys internally**: 
   ```bash
   nsys stats -r cuda_gpu_trace <file> -o <output>
   # Generates: <output>_cuda_gpu_trace.csv
   ```

2. **Non-overlapping calculation**:
   - Sorts kernels by start time
   - Tracks current GPU end time
   - Subtracts overlap from duration when kernels run concurrently
   - Result: "Elapsed Time (ns)" = wall-clock GPU time (no double-counting)

3. **Kernel categorization**:
   - Loads all `.json` files from script directory
   - Each JSON defines engine → model → kernel_regex → category mapping
   - Example: `"nvjet|gemm"` → `"gemm_kernel"` category

---

### 2.2 `sglang_engine_model.json` - Kernel Classification Configuration

**Location**: `examples/profiler/nsys_profile_tools/sglang_engine_model.json`

**Purpose**: Define kernel name → category mappings for visualization

**Structure**:

```json
{
  "sglang": {
    "llama": {
      "gemm|nvjet": "gemm",
      "flash|fmha": "attn",
      "moe|sigmoid": "moe",
      "fp8_quant|cvt_|quantize": "quantize",
      "CUDA mem": "non-gpu-H_D_memops",
      ".*": "misc"
    },
    "gpt-oss": {
      "gemm|nvjet": "gemm",
      "fused_moe_kernel|_group_gemm": "moe_gemm",
      ...
    },
    "ds": { ... }
  }
}
```

**Key Mappings**:
- `gemm|nvjet` → GEMM kernels
- `flash|fmha` → Flash Attention
- `moe|sigmoid` → MoE (Mixture of Experts)
- `fp8_quant|cvt_|quantize` → Quantization kernels
- `CUDA mem` → Memory operations
- `.*` → Catch-all for misc kernels

**Extensibility**:
Add new engine/model by creating JSON in same directory:
```json
{
  "my_engine": {
    "my_model": {
      "kernel_pattern": "category",
      ...
    }
  }
}
```

---

### 2.3 `analyze_profile.py` - Torch Profiler Trace Analysis

**Location**: `zimage_256_256/analyze_profile.py` (784 lines)

**Purpose**: Analyze torch.profiler traces (JSON/gzip format) and generate:
- CUDA kernel categorization
- Pipeline stage breakdown (TextEncoding → Denoising → Decoding)
- Multi-config comparison charts
- Markdown analysis reports

**Key Functions**:

```python
def load_baselines(baseline_dir):
    """Load all baseline_*.json files from directory"""
    
def load_trace(trace_path):
    """Load torch.profiler trace:
    - .trace.json.gz (gzipped)
    - *.trace.json (plain JSON)
    Returns: traceEvents list
    """
    
def classify_kernel(name: str) -> str:
    """Classify CUDA kernel by name pattern:
    - "nvjet" → "BF16 GEMM (DiT)"
    - "sm80_xmma_gemm_f32f32" → "FP32 GEMM (TextEncoder)"
    - "flashattn|flash.*fwd" → "FlashAttention"
    - "rmsnorm|qknorm" → "RMSNorm / QKNorm"
    - etc.
    """
    
def analyze_kernels(events) -> dict:
    """Aggregate kernel statistics:
    - Per-kernel: total time, count, max duration
    - Per-category: aggregated time
    Returns: kernel_stats + category_stats + totals
    """
    
def generate_charts(baselines, analysis, output_dir):
    """Create 6 matplotlib PNG charts:
    1. Pipeline stage breakdown (pie)
    2. CUDA kernel categories (pie)
    3. Config comparison E2E latency (bar)
    4. Per-step denoising latency (line)
    5. Top-N kernels waterfall (barh)
    6. FP32 vs BF16 GEMM breakdown (bar + annotation)
    """
    
def generate_markdown_report(baselines, analysis, output_dir):
    """Generate markdown with:
    - E2E latency table
    - Kernel breakdown table with optimization notes
    - Key findings summary
    - Optimization roadmap
    """
```

**Usage**:

```bash
python analyze_profile.py \
    --trace-dir ./logs \
    --baseline-dir ./zimage_bench \
    --output-dir ./analysis_report
```

**Output**:
- `analysis_report/01_pipeline_and_kernel_breakdown.png` - Pie charts
- `analysis_report/02_config_comparison.png` - Multi-config bars
- `analysis_report/03_top_kernels_waterfall.png` - Top 15 kernels
- `analysis_report/04_fp32_vs_bf16_gemm.png` - Core bottleneck visualization
- `analysis_report/ANALYSIS_REPORT.md` - Full markdown report

**Key Classification Examples**:

| Kernel Name Pattern | Category | Notes |
|-------------------|----------|-------|
| `nvjet_tst_*` | BF16 GEMM (DiT) | TensorRT's nvJET kernel |
| `sm80_xmma_gemm_f32f32` | FP32 GEMM (TextEncoder) | CUTLASS BF16 on H100-like |
| `deep_gemm::sm90_fp8_gemm_1d2d_impl` | FP8 GEMM | DeepGemm library |
| `flashattn*` | FlashAttention | Flash-v3 attention |
| `per_token_group_quant_8bit_kernel` | Quantization | FP8 quantization |
| `fused_qknorm_warp` | RMSNorm | Fused QK normalization |
| `act_and_mul_kernel` | Activation | SiLU + scale |

---

### 2.4 `export_nsys_stats.sh` - Bash Export Utility

**Location**: `zimage_256_256/export_nsys_stats.sh` (33 lines)

**Purpose**: Export nsys statistics CSV from `.nsys-rep` binary files

**Content**:

```bash
#!/bin/bash
NSYS_DIR="/path/to/nsys/files"
OUT_DIR="/path/to/output"
mkdir -p "$OUT_DIR"

# Export BF16 baseline kernel summary
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUT_DIR/nsys_bf16_kernels" \
    "${NSYS_DIR}/zimage_1gpu_256x256_te16.nsys-rep"

# Export FP8 optimized kernel summary
nsys stats \
    --report cuda_gpu_kern_sum \
    --format csv \
    --output "$OUT_DIR/nsys_fp8_kernels" \
    "${NSYS_DIR}/zimage_1gpu_256x256_fp8.nsys-rep"
```

**Key nsys Commands Used**:

```bash
# Kernel summary statistics
nsys stats --report cuda_gpu_kern_sum --format csv -o output file.nsys-rep
# Output: output_cuda_gpu_kern_sum.csv

# Detailed kernel execution trace
nsys stats --report cuda_gpu_trace --format csv -o output file.nsys-rep
# Output: output_cuda_gpu_trace.csv
```

---

## 3. Real Data - Analysis Reports

### 3.1 Existing Analysis Files

**Location**: `zimage_256_256/analysis_report/`

**Contents**:

| File | Type | Size | Purpose |
|------|------|------|---------|
| `01_pipeline_and_kernel_breakdown.png` | PNG | 107 KB | E2E + kernel category pie charts |
| `02_config_comparison.png` | PNG | ~100 KB | Multi-config E2E + per-step latency |
| `03_top_kernels_waterfall.png` | PNG | ~100 KB | Top 15 individual kernels |
| `04_fp32_vs_bf16_gemm.png` | PNG | ~100 KB | Core GEMM comparison (key insight chart) |
| `multi_res_e2e_latency.png` | PNG | 107 KB | E2E across 3 resolutions |
| `multi_res_fp8_denoising_speedup.png` | PNG | 101 KB | FP8 speedup by resolution |
| `multi_res_speedup.png` | PNG | 99 KB | Config speedup comparison |
| `multi_res_ss_step.png` | PNG | 204 KB | Steady-state step latency |
| `multi_res_stage_breakdown.png` | PNG | 70 KB | Stage breakdown by resolution |
| `multi_res_vram.png` | PNG | 70 KB | Memory usage by config |
| `ANALYSIS_REPORT.md` | MD | 40 KB | Comprehensive analysis report |
| `README_CUDAPROFILER.md` | MD | 9 KB | CUDA profiler quick reference |
| `CUDAPROFILER_QUICK_REFERENCE.txt` | TXT | 20 KB | Integration quick reference |
| `CUDAPROFILER_INTEGRATION_ANALYSIS.md` | MD | 18 KB | Technical integration details |

### 3.2 Existing Nsys CSV Data

**Location**: `zimage_256_256/zimage_bench/nsys/csv/`

**CSV Files** (from `nsys stats --report cuda_gpu_kern_sum`):

```
├── nsys_bf16_kernels_cuda_gpu_kern_sum.csv      # Baseline BF16 profile
├── nsys_fp8_kernels_cuda_gpu_kern_sum.csv       # Optimized FP8 profile
├── nsys_fp8_disable_deepgemm_cuda_gpu_kern_sum.csv  # Alternative FP8
├── nsys_bf16_cuda_kern_exec_trace.csv           # Detailed execution trace
├── nsys_fp8_cuda_kern_exec_trace.csv
└── nsys_fp8_disable_deepgemm_cuda_kern_exec_trace.csv
```

**CSV Schema** (from `nsys stats`):

```
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
62.7,1802181977,26265,68615.3,68768.0,67200,72672,594.8,"void at::native::vectorized_elementwise_kernel<(int)4, ...>"
13.1,377211583,33117,11390.3,6688.0,1920,134208,19091.4,_layer_norm_fwd_1pass_kernel
5.4,156564829,72,2174511.5,2174544.0,2172704,2177408,898.4,sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3_...
```

**Columns**:
- `Time (%)` - Percentage of total time
- `Total Time (ns)` - Sum of all invocations
- `Instances` - Number of times kernel ran
- `Avg/Med/Min/Max (ns)` - Duration statistics
- `StdDev (ns)` - Variance
- `Name` - Full CUDA kernel name

### 3.3 Existing Nsys Binary Files

**Location**: `zimage_256_256/zimage_bench/nsys/`

```
├── zimage_1gpu_256x256.nsys-rep                 # Baseline
├── zimage_1gpu_256x256_fp8.nsys-rep             # FP8 optimized
├── zimage_1gpu_256x256_fp8_disable_deepgemm.nsys-rep
├── zimage_1gpu_256x256_te16.nsys-rep
├── zimage_1gpu_cachedit_256x256.nsys-rep
├── zimage_1gpu_compile_256x256.nsys-rep
├── zimage_2gpu_256x256.nsys-rep
└── nsys_no_warmup/
    ├── 256_256/
    │   ├── tebf16_256x256.nsys-rep
    │   └── tebf16_fp8_deepgemm_256x256.nsys-rep
    ├── 512_512/
    │   ├── tebf16_512x512.nsys-rep
    │   └── tebf16_fp8_deepgemm_512x512.nsys-rep
    └── 1024_1024/
        ├── tebf16_1024x1024.nsys-rep
        └── tebf16_fp8_deepgemm_1024x1024.nsys-rep
```

---

## 4. Documentation

### 4.1 Quick Start Guides

**Location**: `zimage_256_256/PROFILER_QUICK_START.md` (294 lines)

**Covers**:
- Basic profiling with torch.profiler
- Diffusion model profiling
- Environment variables (SGLANG_TORCH_PROFILER_DIR, SGLANG_PROFILE_V2)
- ProfileActivity types (GPU, CPU, CUDA_PROFILER, MEM)
- Viewing traces in Chrome DevTools & Perfetto
- Event categories in traces
- Analyzing with analyze_profile.py
- Finding bottlenecks
- Multi-GPU profiling

**Key Content**:

```bash
# Profile GPU only
python -m sglang.bench_one_batch --profile --profile-activities GPU

# Profile CPU+GPU
python -m sglang.bench_one_batch --profile --profile-activities CPU GPU

# Profile with nsys (minimal overhead)
nsys profile --capture-range=cudaProfilerApi -o trace \
    python -m sglang.bench_one_batch --profile-activities CUDA_PROFILER
```

### 4.2 Complete Technical Guide

**Location**: `zimage_256_256/SGLANG_PROFILER_COMPLETE_GUIDE.md` (709 lines)

**Covers**:
- How `--profile` flag works
- ProfilerActivity types and mapping
- Trace file structure (JSON schema)
- Event categories (cpu_op, kernel, cuda_driver, cuda_runtime, overhead)
- Python-side overhead capture
- Multi-GPU safety
- Integration with nsys (CudaProfilerApi)
- Output format analysis
- CLI flags reference
- Troubleshooting common issues
- Integration table for all profiler implementations

**Key Sections**:
- Section 3: Where profiler is started/stopped in code
- Section 4: Output format and trace structure
- Section 5: Capturing Python-side overhead
- Section 8: Integration summary table

### 4.3 Implementation Patterns Guide

**Location**: `zimage_256_256/PROFILER_IMPLEMENTATION_GUIDE.md` (564 lines)

**Covers 5 profiling patterns**:

1. **Scheduler-Based (Multimodal)** - Auto per-request
   ```python
   # In sglang/multimodal_gen/runtime/managers/scheduler.py
   if is_non_warmup_req and torch.cuda.is_available():
       torch.cuda.cudart().cudaProfilerStart()
   # ... do work ...
   torch.cuda.synchronize()
   torch.cuda.cudart().cudaProfilerStop()
   ```

2. **Explicit with torch.profiler** - Manual control
   ```python
   profiler = torch.profiler.profile(activities=[...])
   profiler.__enter__()
   # ... work ...
   profiler.__exit__()
   profiler.export_chrome_trace("trace.json.gz")
   ```

3. **Multi-GPU Safe** - With GPU ID guards
4. **Pluggable Backend** - ProfileManager factory pattern
5. **Stage-Based (Diffusion)** - Profile specific pipeline stages

Each pattern includes full code examples and when to use it.

### 4.4 README Files

**`examples/profiler/nsys_profile_tools/README.md`** (177 lines):
- Usage guide for gputrc2graph.py
- How to collect nsys profiles
- Example 1: Single profile analysis
- Example 2: Multiple profile comparison
- Example 3: Adding new classification for new models
- Kernel-to-category mapping explanation

---

## 5. Nsys Integration in Code

### 5.1 Scheduler Integration (Multimodal)

**File**: `sglang/multimodal_gen/runtime/managers/scheduler.py` (Lines 368-386)

```python
# Automatic CUDA profiler control for non-warmup requests
is_non_warmup_req = (
    isinstance(processed_req, Req) and not processed_req.is_warmup
)
if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.cudart().cudaProfilerStart()  # START

handler = self.request_handlers.get(type(processed_req))
output_batch = handler(reqs)  # Execute

if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.synchronize()  # GPU sync
    torch.cuda.cudart().cudaProfilerStop()   # STOP
```

**Features**:
- ✅ Automatic warmup exclusion
- ✅ GPU synchronization before stop
- ✅ Only 2 CUDA API calls (minimal overhead)
- ✅ Works with nsys `--capture-range=cudaProfilerApi`

### 5.2 LLM Scheduler Integration

**File**: `sglang/srt/managers/scheduler_profiler_mixin.py` (Lines 212-324)

```python
class SchedulerProfilerMixin:
    def start_profile(self, activities):
        if "CUDA_PROFILER" in activities:
            if self.gpu_id == get_global_server_args().base_gpu_id:
                torch.cuda.cudart().cudaProfilerStart()
        # ... torch.profiler code for CPU/GPU ...
    
    def stop_profile(self):
        if "CUDA_PROFILER" in self.profiler_activities:
            if self.gpu_id == get_global_server_args().base_gpu_id:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
```

**Features**:
- ✅ Multi-GPU safe (only base_gpu_id)
- ✅ Conditional activity routing
- ✅ Multi-rank aware

### 5.3 Bench Tool Integration

**File**: `sglang/bench_one_batch.py` (Lines 93-141)

```python
def start_profile(profile_activities, profile_record_shapes=False):
    if "CUDA_PROFILER" in profile_activities:
        torch.cuda.cudart().cudaProfilerStart()
        return None  # CUDA_PROFILER uses external nsys
    
    activities = []
    if "CPU" in profile_activities:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if "GPU" in profile_activities:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=profile_record_shapes,
        with_stack=True,
        use_kineto=True,
    )
    profiler.__enter__()
    return profiler
```

---

## 6. Profiling Methodology

### 6.1 Collecting Nsys Profiles

```bash
# Basic collection (all GPU activity)
nsys profile -t cuda -o output_file -f true \
    python -m sglang.launch_server --model meta-llama/Llama-3.1-8B

# With fork tracing (for multiprocess)
nsys profile -t cuda -o output_file -f true --trace-fork-before-exec=true \
    python -m sglang.launch_server --model ...

# Minimal overhead with CudaProfilerApi (recommended for production)
nsys profile --capture-range=cudaProfilerApi -o output_file \
    python -m sglang.launch_server --model ...
```

### 6.2 Exporting Kernel Statistics

```bash
# Kernel summary (aggregated per kernel name)
nsys stats --report cuda_gpu_kern_sum --format csv \
    -o output_prefix file.nsys-rep
# Creates: output_prefix_cuda_gpu_kern_sum.csv

# Detailed execution trace (every kernel invocation)
nsys stats --report cuda_gpu_trace --format csv \
    -o output_prefix file.nsys-rep
# Creates: output_prefix_cuda_gpu_trace.csv

# View in Nsight Systems GUI
nsys-ui file.nsys-rep
```

### 6.3 Analyzing Results

**Option 1: Using gputrc2graph.py**
```bash
python gputrc2graph.py \
    --in_file baseline.nsys-rep,sglang,llama,100 \
              optimized.nsys-rep,sglang,llama,98 \
    --out_dir comparison
# Outputs: comparison/result.html (interactive chart)
```

**Option 2: Using analyze_profile.py**
```bash
python analyze_profile.py \
    --trace-dir ./logs \
    --baseline-dir ./zimage_bench \
    --output-dir ./analysis
# Outputs: 6 PNG charts + markdown report
```

---

## 7. Key Findings from Existing Analysis

### 7.1 Z-Image-Turbo 256×256 Baseline

**E2E Latency**: 362ms (1 GPU, FP32 TextEncoder)

| Stage | Time | % |
|-------|------|---|
| TextEncoding | 81ms | 22.4% |
| Denoising | 269ms | 74.3% ⚠️ |
| Decoding | 10ms | 2.7% |

### 7.2 GPU Kernel Breakdown (FP32 Baseline)

| Category | Time | % | Note |
|----------|------|---|------|
| BF16 GEMM (DiT) | 251ms | 53.0% | Optimization target |
| FP32 GEMM (TextEncoder) | 157ms | 33.0% | **#1 Bottleneck** |
| Attention | 10ms | 2.1% | Not a bottleneck |
| Others | 56ms | 11.9% | |

### 7.3 Optimization Results

**After FP32→BF16 TextEncoder**:
- TextEncoding: 81ms → ~20ms (-75%)
- Total E2E: 362ms → ~300ms (-17%)

**After BF16 DiT + FP8 Quantization**:
- Additional savings: 5-10% depending on FP8 implementation
- DeepGemm FP8: Consistent across resolutions
- CUTLASS FP8: Variable by shape

---

## 8. Quick Reference: Using Nsys Tools

### 8.1 One-Liner Examples

```bash
# Collect profile
nsys profile -t cuda -o trace python my_model.py

# Export kernel summary
nsys stats --report cuda_gpu_kern_sum --format csv \
    -o stats trace.nsys-rep

# Export execution trace
nsys stats --report cuda_gpu_trace --format csv \
    -o trace trace.nsys-rep

# Analyze with gputrc2graph.py
python gputrc2graph.py \
    --in_file trace.nsys-rep,sglang,llama,100

# Analyze with analyze_profile.py
python analyze_profile.py --trace-dir ./logs --output-dir ./results
```

### 8.2 Nsys Commands Reference

| Command | Purpose | Output |
|---------|---------|--------|
| `nsys profile` | Collect GPU traces | `.nsys-rep` (binary) |
| `nsys stats` | Export statistics | CSV files |
| `nsys export` | Format conversion | `.sqlite`, `.json` |
| `nsys-ui` | Interactive GUI | Visual timeline |

### 8.3 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Huge file (>1GB) | Tracing too long | Use `nsys profile -d 10` (10 sec) |
| Empty kernel list | No GPU work | Ensure model is on GPU |
| Slow nsys stats | Large file | File size / 240 = time (min) |
| GPU kernels missing | nsys version | Update nsys to match profile version |

---

## 9. File Structure Summary

```
sglang/
├── examples/profiler/nsys_profile_tools/
│   ├── gputrc2graph.py              ← Main tool (347 lines)
│   ├── sglang_engine_model.json     ← Kernel classification config
│   └── README.md                    ← Tool documentation (177 lines)
│
├── zimage_256_256/
│   ├── analyze_profile.py           ← Torch trace analyzer (784 lines)
│   ├── generate_multi_res_charts.py ← Chart generator
│   ├── export_nsys_stats.sh         ← Export utility
│   │
│   ├── PROFILER_QUICK_START.md      ← Quick start (294 lines)
│   ├── SGLANG_PROFILER_COMPLETE_GUIDE.md  ← Full guide (709 lines)
│   ├── PROFILER_IMPLEMENTATION_GUIDE.md   ← Patterns (564 lines)
│   ├── README_PROFILER_DOCS.md
│   │
│   ├── analysis_report/
│   │   ├── *.png                    ← 10 analysis charts
│   │   └── ANALYSIS_REPORT.md       ← Markdown report (40KB)
│   │
│   └── zimage_bench/nsys/
│       ├── *.nsys-rep               ← 8 binary profiles
│       └── csv/
│           └── *.csv                ← 6 exported kernel CSVs
│
└── sglang/multimodal_gen/
    └── runtime/
        ├── managers/scheduler.py    ← Profiler integration (line 368)
        └── utils/profiler.py        ← SGLDiffusionProfiler class
```

---

## 10. Advanced Usage

### 10.1 Adding a New Kernel Classification

Edit `sglang_engine_model.json` or create `sglang_engine_new.json`:

```json
{
  "my_engine": {
    "my_model": {
      "kernel_pattern_1": "category_1",
      "kernel_pattern_2|alt_pattern": "category_2",
      "CUDA mem": "memory_ops",
      ".*": "misc"
    }
  }
}
```

Then run:
```bash
python gputrc2graph.py \
    --in_file trace.nsys-rep,my_engine,my_model,100
```

### 10.2 Comparing Two Profiles

```bash
python gputrc2graph.py \
    --in_file baseline.nsys-rep,sglang,llama,100 \
              optimized.nsys-rep,sglang,llama,100 \
    --out_dir comparison \
    --title "Baseline vs Optimized"

# Output: comparison/result.html with side-by-side stacked bars
```

### 10.3 Custom Trace Analysis

Load and analyze traces programmatically:

```python
import gzip, json
from collections import defaultdict

# Load trace
with gzip.open("trace.trace.json.gz", "rb") as f:
    data = json.loads(f.read())

events = data["traceEvents"]

# Count events by category
cat_count = defaultdict(int)
for e in events:
    cat_count[e.get("cat")] += 1

# Time breakdown
cat_time = defaultdict(float)
for e in events:
    if "dur" in e:
        cat_time[e.get("cat")] += e["dur"]

# Print
for cat in sorted(cat_time, key=lambda x: cat_time[x], reverse=True):
    print(f"{cat}: {cat_time[cat]/1e6:.2f}s ({cat_count[cat]} events)")
```

---

## Conclusion

The SGLang codebase provides a comprehensive profiling infrastructure:

✅ **Collection**: Multiple nsys integration points for automatic profile capture  
✅ **Export**: Bash scripts and gputrc2graph.py for CSV extraction  
✅ **Analysis**: Two-level analysis (gputrc2graph for GPU timelines, analyze_profile for kernel breakdown)  
✅ **Visualization**: HTML charts + PNG reports + markdown summaries  
✅ **Documentation**: 3 markdown guides + code examples + real data  
✅ **Extensibility**: JSON-based kernel classification, pluggable backends  

Perfect for understanding GPU bottlenecks and tracking optimization progress.
