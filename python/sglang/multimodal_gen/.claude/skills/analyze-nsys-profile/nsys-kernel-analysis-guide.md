# nsys Kernel Analysis — Detailed Reference Guide

Technical reference for kernel classification, shape extraction, and analysis patterns used by the `analyze-nsys-profile` skill.

---

## 1. Kernel Classification — Python Implementation

Use this Python function to classify kernels from nsys CSV data:

```python
def classify_kernel(name):
    """Classify a CUDA kernel name into analysis categories.
    Order matters — first match wins."""
    if 'deep_gemm::sm90_fp8_gemm' in name:
        return 'FP8_DeepGEMM'
    if 'per_token_group_quant_8bit_kernel' in name:
        return 'FP8_Quantize'
    if 'nvjet_tst_' in name:
        return 'BF16_GEMM_bias' if '_bias_' in name else 'BF16_GEMM'
    if 'FlashAttn' in name or 'fmha_' in name:
        return 'FlashAttention'
    if 'RMSNorm' in name:
        return 'RMSNorm'
    if 'act_and_mul_kernel' in name:
        return 'SiLU_Gate'
    if 'fused_qknorm' in name:
        return 'QKNorm'
    if 'RotaryPos' in name or 'Rotary' in name:
        return 'RoPE'
    if 'xmma_fprop_implicit_gemm' in name or 'cudnn' in name:
        return 'Conv_VAE'
    if 'splitKreduce' in name:
        return 'GEMM_splitK'
    if 'cublas' in name or 'cutlass_80' in name or 'cutlass::Kernel2' in name:
        return 'Other_GEMM'
    if any(x in name for x in ['elementwise', 'vectorized_elementwise', 'unrolled_elementwise']):
        return 'Elementwise'
    if 'CatArrayBatchedCopy' in name:
        return 'CatCopy'
    if any(x in name for x in ['RowwiseMoments', 'layer_norm', 'GroupNorm', 'ComputeFusedParams']):
        return 'Norm_Other'
    if 'reduce_kernel' in name:
        return 'Reduce'
    if 'upsample' in name:
        return 'Upsample'
    if 'nchwToNhwc' in name or 'nhwcToNchw' in name or 'tensorTransform' in name:
        return 'Layout_Transform'
    return 'Other'
```

### Extending for New Kernel Types

When a new optimization introduces unfamiliar kernels, add classification rules **before** the `Other` fallback. Common additions:

| Optimization | New Kernel Pattern | Suggested Category |
|------|------|------|
| CUTLASS FP8 | `cutlass::device_kernel<...fp8...>` | `FP8_CUTLASS` |
| Triton custom | `triton_` prefix | `Triton_Custom` |
| torch.compile | `triton::` or `inductor::` | `Compile_Fused` |
| INT8 quantize | `int8` in name | `INT8_Quantize` |

---

## 2. CSV Parsing — Python Implementation

```python
import csv

def parse_kern_sum(filepath):
    """Parse nsys cuda_gpu_kern_sum CSV file.
    Returns list of dicts with name, total_ns, instances, avg_ns."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Name'].strip('"')
            total_ns = int(row['Total Time (ns)'])
            instances = int(row['Instances'])
            results.append({
                'name': name,
                'total_ns': total_ns,
                'instances': instances,
                'avg_ns': total_ns / instances if instances > 0 else 0,
            })
    return results

def aggregate_by_category(kernels):
    """Aggregate kernel list into category totals."""
    cats = {}
    for k in kernels:
        cat = classify_kernel(k['name'])
        if cat not in cats:
            cats[cat] = {'total_ns': 0, 'instances': 0}
        cats[cat]['total_ns'] += k['total_ns']
        cats[cat]['instances'] += k['instances']
    return cats
```

---

## 3. DeepGemm Shape Extraction

DeepGemm kernel names encode GEMM dimensions in template parameters:

```
void deep_gemm::sm90_fp8_gemm_1d2d_impl<
  (cute::UMMA::Major)0,
  (unsigned int)0,
  (unsigned int)N,        ← output dimension (or combined)
  (unsigned int)K,        ← reduction dimension
  (unsigned int)1,        ← batch (always 1 for standard GEMM)
  ...tile parameters...
>
```

### Extraction Regex

```python
import re

def get_deepgemm_shape(name):
    """Extract (N, K) from DeepGemm kernel template parameters."""
    m = re.search(
        r'unsigned int\)(\d+), \(unsigned int\)(\d+), \(unsigned int\)1,',
        name
    )
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None
```

### Shape → GEMM Type Mapping

| N | K | GEMM Type | DiT Component |
|:---:|:---:|------|------|
| 20480 | 3840 | FFN w13 | gate+up projection (SwiGLU) |
| 3840 | 3840 | QKV+out | Q, K, V, output projection |
| 3840 | 10240 | FFN w2 | down projection |

**Note**: N and K depend on the model's hidden dimension (`dim`) and FFN multiplier. For ZImage-Turbo: `dim=3840`, FFN hidden = `dim × 8/3 × 2 ≈ 20480` (SwiGLU gate+up fused).

For other models, derive from:
```
FFN w13: N = ffn_hidden_dim * 2, K = dim    (or N = ffn_hidden_dim for separate gate/up)
QKV:     N = dim, K = dim
FFN w2:  N = dim, K = ffn_hidden_dim
```

---

## 4. Instance Count → Logical Operation Mapping

### General Formula

```
instances = num_layers × num_steps × ops_per_layer
```

### ZImage-Turbo Specifics

| Instance Count | Computation |Logical Operation |
|:---:|------|------|
| 270 | 30 layers × 9 steps × 1 | FFN w13 OR FFN w2 (one per layer) |
| 1080 | 30 layers × 9 steps × 4 | Q + K + V + output_proj |
| 288 | 32×9 or 30×9 + autotune extra | FFN w13 at large M (1024×1024) |
| 1620 | 30 × 9 × 6 | All DiT projections combined (nvjet at 1024) |
| 1350 | 30 × 9 × 5 | QKV+out + w13 combined (nvjet at 512) |
| 72 | 8 layers × 9 steps | TextEncoder GEMM |
| 18 | 2 layers × 9 steps | TextEncoder small GEMM or autotune |
| 1836 | (270+1080+270+18+...) × 1 | Total FP8 quantize calls (1 per GEMM) |

### Identifying Main vs Autotune DeepGemm Kernels

For a given shape (N, K), the **main** kernel has the highest instance count. Other variants of the same shape with fewer instances are **autotune** variants (DeepGemm exploring different tile configurations).

```python
def separate_main_vs_autotune(fp8_kernels, expected_main_counts={270, 1080}):
    """Separate DeepGemm kernels into main compute and autotune variants."""
    main_total_ns = 0
    autotune_total_ns = 0
    for k in fp8_kernels:
        if k['instances'] in expected_main_counts:
            main_total_ns += k['total_ns']
        else:
            autotune_total_ns += k['total_ns']
    return main_total_ns, autotune_total_ns
```

---

## 5. nvjet Tile Config Patterns

BF16 GEMM uses NVIDIA nvjet (cuBLAS) with auto-selected tile configurations. The kernel name encodes the tile:

```
nvjet_tst_{Mtile}x{Ntile}_64x{K}_..._{layout}
```

### Common Tile → Shape Mapping by Resolution

**256×256 (M=768)**: nvjet selects per-shape configs
- `nvjet_192x160` → FFN w13 (270 instances)
- `nvjet_256x64` → QKV (1080 instances)
- `nvjet_128x288` → FFN w2 (270 instances)

**512×512 (M=3072)**: nvjet merges some shapes
- `nvjet_128x224` → QKV + FFN w13 combined (1350 = 1080+270)
- `nvjet_384x96` → FFN w2 (270 instances)

**1024×1024 (M=12288)**: nvjet uses single config for all
- `nvjet_256x160` → ALL DiT GEMM (1620 = 1080+270+270)

**Key insight**: At larger M, nvjet's tile selection becomes more "generic" (one tile for all shapes), while DeepGemm maintains per-shape optimal tiling — this is one reason DeepGemm outperforms at large M.

---

## 6. CudaProfilerApi Verification Checklist

### Source Location

```
python/sglang/multimodal_gen/runtime/managers/scheduler.py
```

### Required Pattern (search near event loop, around line 365-390)

```python
# 1. Warmup detection
is_non_warmup_req = (
    isinstance(processed_req, Req) and not processed_req.is_warmup
)

# 2. Start profiler BEFORE handler execution
if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.cudart().cudaProfilerStart()

# 3. Handler executes the inference
output_batch = handler(reqs)

# 4. Stop profiler AFTER synchronize
if is_non_warmup_req and torch.cuda.is_available():
    torch.cuda.synchronize()     # ← CRITICAL: ensures GPU work completes
    torch.cuda.cudart().cudaProfilerStop()
```

### Verification Checks

| # | Check | Why It Matters |
|:---:|------|------|
| 1 | `not processed_req.is_warmup` guard | Excludes warmup requests (JIT compile, first-run caching) |
| 2 | `cudaProfilerStart()` before handler | Captures all inference kernels |
| 3 | `torch.cuda.synchronize()` before Stop | Without sync, async GPU work may be cut off |
| 4 | `cudaProfilerStop()` after sync | Clean profiling boundary |
| 5 | Both guarded by `torch.cuda.is_available()` | Safe on CPU-only environments |

### nsys Command Verification

The nsys command must include:
```bash
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...
```

Without `--capture-range=cudaProfilerApi`, nsys captures everything (including warmup), making the scheduler code irrelevant.

### Red Flags (warmup NOT excluded)

If you see these in the kernel data, warmup may be included:
- **Huge transpose kernel counts** (e.g., 49,152 transpose launches) — DeepGemm JIT scale preprocessing
- **Many distinct DeepGemm tile variants** with very small instance counts (e.g., 1-3 instances each)
- **Total kernel time far exceeds** expected inference time (e.g., 800ms kernel time for a 400ms E2E run)

---

## 7. Report Template

Use this template to structure the final analysis output:

```markdown
## N. nsys No-Warmup Kernel Analysis: {Baseline} vs {Optimized}

> **Method**: nsys profile with `--capture-range=cudaProfilerApi` (warmup excluded)
> **Data**: `{path}`

### N.1 GPU Kernel Total Time

| Config | GPU Kernel Total | Delta |
|--------|:---:|:---:|
| {Baseline} | **{bl_total:.2f} ms** | — |
| {Optimized} | **{opt_total:.2f} ms** | **{delta:+.2f} ms ({pct:+.1f}%)** |

### N.2 Per-Category Kernel Breakdown

| Category | Baseline (ms) | 次数 | Optimized (ms) | 次数 | Delta (ms) | 说明 |
|----------|:---:|:---:|:---:|:---:|:---:|------|
| ... | ... | ... | ... | ... | ... | ... |

### N.3 New/Removed Kernels

#### New in Optimized
| Kernel | Total (ms) | Instances | Category |
|--------|:---:|:---:|------|

#### Removed from Baseline
| Kernel | Total (ms) | Instances | Category |
|--------|:---:|:---:|------|

### N.4 GEMM Shape Speedup

| GEMM Type | Shape (M×K→N) | Baseline (ms) | Optimized (ms) | Speedup | BL avg (μs) | Opt avg (μs) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|

### N.5 Overhead Accounting

| Component | Time (ms) | Notes |
|-----------|:---:|-------|
| Baseline GEMM (replaced) | {bl_gemm:.2f} | Reference |
| → Optimized main GEMM | {opt_gemm:.2f} | Faster by {saving:.2f}ms ✅ |
| → Autotune variants | {auto:.2f} | Tile exploration |
| → Quantize overhead | {quant:.2f} | Per-token quantization |
| **Optimized GEMM Total** | **{opt_total_gemm:.2f}** | — |
| **Net Savings** | **{net:+.2f} ms** | **{net_pct:+.1f}%** |

### N.6 Key Findings

1. ...
2. ...
```

---

## 8. Cross-Resolution Summary Template

```markdown
### Cross-Resolution Summary

| Resolution | M | BL GPU (ms) | Opt GPU (ms) | Kernel Δ% | GEMM Speedup | Quant OH (ms) | Total OH (ms) | OH % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

### Trends
- GEMM speedup trend: {constant/increasing/decreasing} with resolution
- Overhead % trend: {decreasing} — FP8 more efficient at larger M
- Bottleneck shift: at {resolution}, {category} becomes #{rank} ({pct}% of FP8 time)
```

---

## 9. sqlite3 Queries (Alternative to CSV)

If the user has `.sqlite` files from nsys, these queries extract equivalent data:

```sql
-- Kernel summary (equivalent to cuda_gpu_kern_sum.csv)
SELECT s.value AS name,
       SUM(k.end - k.start) AS total_ns,
       COUNT(*) AS instances
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
GROUP BY k.shortName
ORDER BY total_ns DESC;

-- Total GPU kernel time
SELECT SUM(end - start) / 1e6 AS total_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL;

-- Kernel launch count
SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;
```

---

## 10. Adapting to Other Models

This guide was developed on ZImage-Turbo (30-layer DiT, dim=3840, 9 steps). To adapt:

1. **Update instance count formula**: `num_layers × num_steps × ops_per_layer`
2. **Update shape mapping**: derive N, K from model's `dim` and `ffn_hidden_dim`
3. **Add new kernel categories** if the model uses different operators (e.g., cross-attention, grouped convolution)
4. **Adjust "main vs autotune" threshold**: use the model's expected L×S as the main instance count

| Model | Layers | Typical Steps | dim | FFN hidden |
|-------|:---:|:---:|:---:|:---:|
| ZImage-Turbo | 30 | 9 | 3840 | 20480 |
| FLUX.1 | 19+38 | 20-50 | 3072 | 12288 |
| HunyuanVideo | 40 | 50 | 3072 | — |
| Wan-T2V-A14B | 40 | 50 | 5120 | — |
