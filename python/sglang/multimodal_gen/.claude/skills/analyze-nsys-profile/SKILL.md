---
name: analyze-nsys-profile
description: Analyze nsys profiling results (cuda_gpu_kern_sum + cuda_gpu_trace CSVs) to compare baseline vs optimized kernel performance. Use when the user has nsys profile data and needs to identify new/removed kernels, GEMM speedup per shape, FP8 overhead breakdown, and bottleneck shifts across resolutions. Also validates that CudaProfilerApi warmup exclusion is correctly implemented in the scheduler.
---

# nsys Profile Kernel Analysis

Systematic workflow for analyzing NVIDIA Nsight Systems profiling data from SGLang-Diffusion inference runs. Compares two configurations (e.g., BF16 baseline vs FP8 DeepGemm) at the CUDA kernel level.

## When to Use

- User provides nsys `cuda_gpu_kern_sum.csv` and/or `cuda_gpu_trace.csv` files
- User wants to compare baseline vs optimized (FP8, torch.compile, new kernel, etc.)
- User asks "which kernels are new/removed", "is GEMM faster", "what's the overhead"
- User has multiple resolutions and wants cross-resolution trend analysis
- User wants to verify warmup was properly excluded from profiling

## Required Inputs

The user should provide **pairs** of CSV files (baseline + optimized) per resolution:

```
<dir>/
├── <baseline>_cuda_gpu_kern_sum.csv    # Kernel summary (aggregated)
├── <baseline>_cuda_gpu_trace.csv       # Individual kernel launches (optional)
├── <optimized>_cuda_gpu_kern_sum.csv
└── <optimized>_cuda_gpu_trace.csv      # (optional)
```

Also useful but optional:
- `sglang generate --perf-dump-path` JSON files for E2E stage breakdown
- `.nsys-rep` files (binary, need `nsys stats` to export)
- `.sqlite` files (queryable with sqlite3)

### CSV Format Validation

The `cuda_gpu_kern_sum.csv` must have these columns:
```
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
```

If the user only has `.nsys-rep` files, provide export commands:
```bash
nsys stats --report cuda_gpu_kern_sum --format csv -o <output_prefix> <file>.nsys-rep
nsys stats --report cuda_gpu_trace --format csv -o <output_prefix> <file>.nsys-rep
```

---

## Step 0: Verify CudaProfilerApi Warmup Exclusion

**CRITICAL**: Before analyzing, verify the profiling data excludes warmup. Read the scheduler source:

```
python/sglang/multimodal_gen/runtime/managers/scheduler.py
```

Look for lines near the request execution loop (around line 368-386). The correct pattern is:

```python
# Start CUDA profiler for non-warmup requests when using
# nsys --capture-range=cudaProfilerApi, so warmup is excluded
is_non_warmup_req = (
    isinstance(processed_req, Req) and not processed_req.is_warmup
)
if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.cudart().cudaProfilerStart()

# ... handler executes the request ...

if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
```

**Checklist**:
- [ ] `cudaProfilerStart()` is called **only** when `not processed_req.is_warmup`
- [ ] `cudaProfilerStop()` is called after `torch.cuda.synchronize()` (ensures all GPU work completes)
- [ ] Both calls are guarded by `torch.cuda.is_available()`
- [ ] The nsys command used `--capture-range=cudaProfilerApi` flag

If any check fails, warn the user that **warmup kernels may be included** in the profile data, inflating overhead numbers (especially DeepGemm JIT transpose).

**How the user should have run nsys**:
```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o <output> sglang generate --model-path <MODEL> --warmup ...
```

---

## Step 1: Parse & Classify Kernels

Read each `cuda_gpu_kern_sum.csv` file using Python (via Bash tool). Classify every kernel into categories using these rules (order matters — first match wins):

| Priority | Pattern in kernel name | Category | Notes |
|:---:|------|------|------|
| 1 | `deep_gemm::sm90_fp8_gemm` | **FP8_DeepGEMM** | FP8 GEMM compute |
| 2 | `per_token_group_quant_8bit_kernel` | **FP8_Quantize** | BF16→FP8 dynamic quantization |
| 3 | `nvjet_tst_` + `_bias_` | **BF16_GEMM_bias** | Small GEMM with bias (TextEncoder/adaLN) |
| 4 | `nvjet_tst_` (no bias) | **BF16_GEMM** | DiT BF16 GEMM (nvjet/cuBLAS) |
| 5 | `FlashAttn` or `fmha_` | **FlashAttention** | Attention kernels |
| 6 | `RMSNorm` | **RMSNorm** | |
| 7 | `act_and_mul_kernel` | **SiLU_Gate** | SwiGLU activation |
| 8 | `fused_qknorm` | **QKNorm** | |
| 9 | `RotaryPos` or `Rotary` | **RoPE** | Rotary position embedding |
| 10 | `xmma_fprop_implicit_gemm` or `cudnn` | **Conv_VAE** | VAE convolutions |
| 11 | `splitKreduce` | **GEMM_splitK** | cuBLAS split-K reduction |
| 12 | `cublas` or `cutlass_80` or `cutlass::Kernel2` | **Other_GEMM** | Other GEMM backends |
| 13 | `elementwise` or `vectorized_elementwise` or `unrolled_elementwise` | **Elementwise** | |
| 14 | `CatArrayBatchedCopy` | **CatCopy** | Tensor concatenation |
| 15 | `RowwiseMoments` or `layer_norm` or `GroupNorm` or `ComputeFusedParams` | **Norm_Other** | |
| 16 | `reduce_kernel` | **Reduce** | |
| 17 | (everything else) | **Other** | |

Aggregate by category: sum `Total Time (ns)` and `Instances` for each category.

---

## Step 2: Cross-config Comparison

Generate a comparison table:

```markdown
| Category | Baseline (ms) | 次数 | Optimized (ms) | 次数 | Delta (ms) | 说明 |
|----------|:---:|:---:|:---:|:---:|:---:|------|
```

Sort by `max(baseline, optimized)` descending. Mark key changes:
- Categories that went from >0 to 0: **"被替换"**
- Categories that went from 0 to >0: **"新增"**
- Categories with |delta| > 1ms: show delta

Also output totals:
```
Baseline GPU kernel total: XXX.XX ms
Optimized GPU kernel total: XXX.XX ms
Delta: ±XX.XX ms (±XX.X%)
```

---

## Step 3: Identify New/Removed Kernels

Compute set differences on kernel **names**:

- **Only in Optimized**: kernels present in optimized but not baseline
- **Only in Baseline**: kernels present in baseline but not optimized

List top entries sorted by `Total Time` descending, with classification.

---

## Step 4: GEMM Shape-level Speedup

### DiT Instance Count Patterns

For a model with `L` layers and `S` denoising steps:
- **L × S** instances = per-layer-per-step kernel (e.g., FFN w13: 30×9 = 270)
- **4 × L × S** instances = per-projection-per-step (Q,K,V,out: 4×30×9 = 1080)
- Smaller counts (e.g., 18, 72) = TextEncoder or autotuning variants

### DeepGemm Shape Extraction

From template parameters in the kernel name:
```
void deep_gemm::sm90_fp8_gemm_1d2d_impl<..., (unsigned int)N, (unsigned int)K, (unsigned int)1, ...>
```
Extract N and K with regex: `unsigned int\)(\d+), \(unsigned int\)(\d+), \(unsigned int\)1,`

Common shapes:
- N=20480, K=3840 → **FFN w13** (gate+up projection)
- N=3840, K=3840 → **QKV + output_proj**
- N=3840, K=10240 → **FFN w2** (down projection)

### Matching BF16 ↔ FP8

Match by instance count:
- 270 instances in both → same per-layer kernel
- 1080 instances in both → same QKV kernel
- Multiple BF16 kernels may map to different nvjet tile configs for the same logical GEMM

### Speedup Table

```markdown
| GEMM 用途 | Shape (M×K→N) | BF16 nvjet (ms) | FP8 DeepGemm (ms) | Speedup | BF16 avg (μs) | FP8 avg (μs) |
```

---

## Step 5: Overhead Accounting

Classify FP8 DeepGemm kernels as **main** vs **autotune**:
- **Main**: instance count matches expected L×S or 4×L×S
- **Autotune**: smaller instance counts (different tile configs being explored)

Build overhead table:

```markdown
| Overhead Type | Total Time (ms) | % of Optimized GPU Total | Launch Count |
|--------------|:---:|:---:|:---:|
| FP8 Quantize | X.XX | X.X% | N |
| DeepGemm autotune | X.XX | X.X% | N |
| **Total Overhead** | **X.XX** | **X.X%** | **N** |
```

Build time budget:

```markdown
| Component | Time (ms) | Notes |
|-----------|:---:|-------|
| BF16 GEMM (replaced) | XXX.XX | Baseline reference |
| → FP8 main GEMM | XXX.XX | Faster by XX.XX ms ✅ |
| → FP8 autotune | X.XX | Tile exploration |
| → FP8 quantize | X.XX | Per-token quantization |
| **FP8 Total** | **XXX.XX** | — |
| **Net Savings** | **-XX.XX ms** | **-XX.X%** |
```

---

## Step 6: Non-GEMM Verification

Verify non-GEMM kernel categories are unchanged between baseline and optimized:

```markdown
| Category | Baseline (ms) | Optimized (ms) | Delta | Status |
|----------|:---:|:---:|:---:|:---:|
| FlashAttention | X.X | X.X | ≈0 | ✅ |
| Elementwise | X.X | X.X | ≈0 | ✅ |
```

Flag any unexpected changes (>5% delta) as potential issues.

Note bottleneck shifts: after GEMM optimization, which non-GEMM category becomes the largest?

---

## Step 7: Generate Report

Produce a structured markdown section suitable for appending to an existing analysis report. Follow this template:

```markdown
## N. nsys Kernel Analysis: <Baseline> vs <Optimized>

> **Method**: nsys profile with `--capture-range=cudaProfilerApi` (warmup excluded)
> **Data**: `<path to CSV files>`

### N.1 GPU Kernel Total Time

| Config | GPU Kernel Total | Delta |
|--------|:---:|:---:|
| Baseline | **XXX.XX ms** | — |
| Optimized | **XXX.XX ms** | **±XX.XX ms (±XX.X%)** |

### N.2 Per-Category Breakdown
(table from Step 2)

### N.3 New/Removed Kernels
(tables from Step 3)

### N.4 GEMM Shape Speedup
(table from Step 4)

### N.5 Overhead Accounting
(tables from Step 5)

### N.6 Key Findings
(numbered list of insights)
```

---

## Cross-resolution Analysis

When the user provides data for multiple resolutions, add:

### Summary Table

```markdown
| Resolution | M | BL GPU (ms) | Opt GPU (ms) | Kernel Δ% | BL GEMM (ms) | Opt GEMM (ms) | GEMM Speedup | Quant (ms) | Overhead (ms) |
```

### Trend Analysis

- Does GEMM speedup increase/decrease/stay constant with resolution?
- How does overhead % change with resolution?
- Which non-GEMM category grows fastest (e.g., FlashAttention O(n²))?
- At what resolution does a bottleneck shift occur?

---

## Common Pitfalls

1. **Warmup not excluded**: If CSV includes JIT/autotuning kernels (e.g., massive transpose counts), the data includes warmup. Re-profile with `--capture-range=cudaProfilerApi`.

2. **nvjet merges shapes at large M**: At 1024×1024, nvjet may use a single tile config for all DiT GEMM shapes (e.g., 1620 = 1080+270+270 instances). Don't try to split these into individual shapes.

3. **DeepGemm autotune ≠ warmup**: Even with CudaProfilerApi, DeepGemm may run alternate tile configs during early denoise steps. These are the "autotune" kernels with smaller instance counts — they're part of real inference, not warmup.

4. **E2E vs kernel time mismatch**: If E2E is slower but kernel time is faster, the bottleneck is host-side (Python dispatch, CUDA graph absence). Flag this prominently.

5. **Instance count arithmetic**: `instances = num_layers × num_steps × projections_per_layer`. For ZImage-Turbo: 30 layers × 9 steps = 270 (per-layer), ×4 for QKV+out = 1080. Adjust for other models.

---

## Reference

For detailed kernel classification rules, regex patterns, and DeepGemm template parsing, see:
[nsys-kernel-analysis-guide.md](./nsys-kernel-analysis-guide.md)
