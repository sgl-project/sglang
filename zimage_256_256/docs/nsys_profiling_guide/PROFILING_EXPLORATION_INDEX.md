# Profiling Exploration Index
**Complete Guide to ZImage-Turbo 256×256 Performance Analysis**

---

## 📋 Quick Navigation

### Main Summary Document
- **[PROFILING_ANALYSIS_SUMMARY.md](./PROFILING_ANALYSIS_SUMMARY.md)** — Read this first! 
  - Executive summary of all findings
  - 16 comprehensive sections covering every aspect
  - Performance bottlenecks, optimization recommendations, next steps
  - **Size**: ~50 KB, **Time to read**: 30-45 minutes

---

## 📂 Repository Structure & Files

### 1. Baseline Performance Data
```
zimage_256_256/zimage_bench/
├── baseline_1gpu.json                              # BF16 reference (485ms E2E)
├── baseline_1gpu_text_encoder_bf16.json           # TextEncoder in BF16
├── baseline_1gpu_tebf16_cachedit.json             # + Cache-DiT
├── baseline_1gpu_tebf16_cachedit_fp8_deepgemm.json  # Full FP8
├── baseline_1gpu_tebf16_cachedit_fp8_cutlass.json   # FP8 + CUTLASS
├── baseline_2gpu.json, baseline_2gpu_fp8_deepgemm.json  # Multi-GPU variants
├── 256_256/, 512_512/, 1024_1024/                 # Multi-resolution configs
└── nsys/                                           # GPU profiling data
```

**File Purpose**: End-to-end latency measurements across configurations
**Key Data Points**:
- E2E latency (total inference time)
- Stage breakdown (TextEnc, Denoise, Decode)
- Per-step denoising latencies
- Warmup vs steady-state performance

---

### 2. GPU Kernel Profiling (nsys CSV Data)

```
zimage_256_256/zimage_bench/nsys/csv/
├── nsys_bf16_kernels_cuda_gpu_kern_sum.csv        # BF16 kernel statistics
├── nsys_bf16_cuda_kern_exec_trace.csv             # BF16 detailed trace (73K lines)
├── nsys_fp8_kernels_cuda_gpu_kern_sum.csv         # FP8 + DeepGEMM kernel stats
├── nsys_fp8_cuda_kern_exec_trace.csv              # FP8 detailed trace (65K lines)
├── nsys_fp8_disable_deepgemm_cuda_gpu_kern_sum.csv  # FP8 fallback stats
└── nsys_fp8_disable_deepgemm_cuda_kern_exec_trace.csv  # FP8 fallback trace
```

**File Format**: CSV with columns: `Time (%), Total Time (ns), Instances, Avg (ns), Med (ns), Min (ns), Max (ns), Name`

**Key Analysis**:

| File | Primary Finding |
|------|-----------------|
| nsys_bf16_kernels_cuda_gpu_kern_sum.csv | FP32 GEMM = 5.4%, BF16 GEMM = 4.1% of total GPU time |
| nsys_fp8_kernels_cuda_gpu_kern_sum.csv | Scale transpose overhead = 16.5% + 13.0% (1.9 ms total) |
| nsys_fp8_disable_deepgemm_... | Triton fallback kernels are slower but no transpose overhead |

**How to Read**:
1. Look at `Time (%)` column to identify bottlenecks
2. Check `Avg (ns)` to see per-kernel efficiency
3. Cross-reference kernel names with `analyze_profile.py` classification

---

### 3. Analysis Scripts

#### 3.1 export_nsys_stats.sh (1 KB)
**Purpose**: Extract GPU kernel statistics from binary nsys profiling files

```bash
# Run this on GPU server to export stats
bash export_nsys_stats.sh

# Generates CSV files in debug directory
```

**Outputs**: The `.csv` files documented in section 2 above

---

#### 3.2 analyze_profile.py (26 KB)
**Purpose**: Main performance analysis tool

**Capabilities**:
- Loads baseline JSON files
- Parses torch profiler traces
- Classifies CUDA kernels (GEMM, Attention, Normalization, etc.)
- Generates 4 comparison charts
- Produces markdown analysis report

**Usage**:
```bash
python analyze_profile.py \
    --trace-dir ./logs \
    --baseline-dir ./zimage_bench \
    --output-dir ./analysis_report
```

**Outputs**:
- `01_pipeline_and_kernel_breakdown.png` — Pipeline stages & kernel categories
- `02_config_comparison.png` — E2E latency & per-step denoising across configs
- `03_top_kernels_waterfall.png` — Top-15 CUDA kernels by GPU time
- `04_fp32_vs_bf16_gemm.png` — FP32 TextEncoder vs BF16 DiT comparison
- `ANALYSIS_REPORT.md` — Markdown summary of findings

---

### 4. DeepGEMM Technical Analysis

```
zimage_256_256/deep_gemm/
├── introduce.md   # What is DeepGEMM? Key concepts and capabilities
└── report.md      # Deep technical analysis
```

#### 4.1 introduce.md
**Reading Time**: 5-10 minutes  
**Topics**:
- What is DeepGEMM? (DeepSeek AI's FP8 GEMM library)
- 2.7× faster than CUTLASS on aligned shapes
- Block-wise quantization (1×128 activations, 128×128 weights)
- Hopper architecture utilization (WGMMA, TMA, persistent kernels)
- Alignment requirements (N%64==0, K%128==0)
- Best/worst use cases

**Key Insight**: DeepGEMM is optimized for **long sequences**; short sequences like 256×256 are **bandwidth-bound** where it shows no benefit.

---

#### 4.2 report.md
**Reading Time**: 15-20 minutes  
**Topics**:
- FP8 precision trade-offs (block quantization vs per-tensor)
- Two-level accumulation (FP8 speed + FP32 accuracy)
- Hopper hardware features utilized
- **Scale transpose bottleneck** (~70ms per forward)
- H20 GPU specifics (4.0 TB/s HBM, 296 TFLOPS FP8)
- **Verdict on 256×256**: BF16 baseline (485ms) beats FP8+DeepGEMM (548ms) due to transpose overhead
- Multi-resolution analysis (where FP8 becomes beneficial)

**Critical Finding**: The 70ms runtime transpose overhead overwhelms FP8 compute savings on short sequences. At 1024×1024+, savings exceed overhead.

---

### 5. Code Review & Optimization Proposal

```
zimage_256_256/review/
└── pre_transpose.md  # Weight scale pre-transpose optimization review
```

**Reading Time**: 20-30 minutes  
**Status**: Round 2 review (critical issue found)

**What It Proposes**:
- Move weight scale transpose from **runtime** to **model load time**
- Expected to save 70ms per forward pass

**Changes Accepted** ✅:
- Extracted shared helper function
- Eliminated code duplication between `srt/` and `multimodal_gen/`
- Improved fallback path

**CRITICAL Issue Found** ⚠️:
- **SM90 assertion failure** when weight scale has TMA padding
- Affected shapes: N where `(N/128) % 4 != 0` (e.g., N=768, 384, 640, 896)
- Will crash at runtime without fix

**Recommended Fix**:
```python
# Skip pre-transpose when padding would be added
if (n_scale_dim % 4) != 0:
    return  # TMA alignment would add padding incompatible with SM90
```

**Must Do Before Merge**: Test on GPU to confirm issue and validate fix.

---

### 6. Generated Reports & Charts

```
zimage_256_256/analysis_report/
├── 01_pipeline_and_kernel_breakdown.png    # Stage breakdown & kernel categories
├── 02_config_comparison.png                 # E2E latency across configs
├── 03_top_kernels_waterfall.png            # Top-15 CUDA kernels
├── 04_fp32_vs_bf16_gemm.png                # FP32 vs BF16 comparison (core evidence)
└── ANALYSIS_REPORT.md                      # Markdown report
```

---

### 7. Profiler Documentation

```
zimage_256_256/
├── PROFILER_IMPLEMENTATION_GUIDE.md   (16.9 KB) - How to use SGLang profiling
├── PROFILER_QUICK_START.md             (6.8 KB)  - Quick reference
└── SGLANG_PROFILER_COMPLETE_GUIDE.md   (21.9 KB) - Full documentation
```

---

## 🎯 Key Findings at a Glance

### Performance Bottleneck Hierarchy
```
#1: TextEncoding (FP32)     — 300 ms (62% of E2E)  ◀◀ PRIMARY TARGET
#2: Denoising (BF16 DiT)    — 150 ms (31% of E2E)
#3: Decoding (VAE)          — 35 ms (7% of E2E)
```

### GPU Kernel Time Distribution (BF16)
```
GEMM kernels:       2.6 ms (91.5%) ◀◀ DOMINATES
├─ FP32 (TextEnc):  0.31 ms
├─ BF16 (DiT):      2.29 ms
Attention:          0.06 ms
Normalization:      0.15 ms
Other:              0.08 ms
────────────────────────────
Total GPU time:     2.9 ms
```

### FP8 + DeepGEMM Hidden Overhead
```
Scale transpose:    1.9 ms (27% of GPU kernel time) ◀◀ KILLS BENEFIT
├─ transpose_fp32<512,64,30>: 1.24 ms
└─ transpose_fp32<512,64,80>: 0.617 ms

DeepGEMM kernels:   1.2 ms (faster individually)
Quantization:       0.01 ms
Fallback kernels:   2.0 ms
Other overhead:     1.4 ms
────────────────────────────
Total GPU time:     6.5 ms (vs 2.9 ms BF16 — 2.2× slower!)
```

---

## 📊 Data Summary

| Metric | Value | Source |
|--------|-------|--------|
| BF16 E2E Latency | 485 ms | baseline_1gpu.json |
| TextEncoding %age | 62% | E2E breakdown |
| FP8+DeepGEMM E2E | 548 ms | baseline_1gpu_tebf16_cachedit_fp8_deepgemm.json |
| FP8 Regression | +63 ms | Comparison |
| GEMM GPU time | 91.5% | nsys_bf16_kernels_cuda_gpu_kern_sum.csv |
| FP32 GEMM %age | 5.4% | nsys_bf16_kernels_cuda_gpu_kern_sum.csv |
| BF16 GEMM %age | 4.1% | nsys_bf16_kernels_cuda_gpu_kern_sum.csv |
| Scale transpose %age | 29.5% | nsys_fp8_kernels_cuda_gpu_kern_sum.csv |
| FlashAttention %age | 0.4% | nsys_bf16_kernels_cuda_gpu_kern_sum.csv |

---

## 🚀 Optimization Roadmap

### Priority 1: TextEncoder FP32 → BF16 ✅ HIGH IMPACT, LOW EFFORT
- **Expected Savings**: 200-300 ms per image
- **Implementation**: 1 line of code
- **Risk**: Low
- **ROI**: -38% to -59% E2E latency

### Priority 2: DeepGEMM Weight Scale Pre-Transpose 🔴 CRITICAL ISSUE BLOCKS THIS
- **Expected Savings**: 70 ms per image
- **Implementation**: Medium (requires bug fixes first)
- **Risk**: Medium (SM90 assertion issue)
- **ROI**: Makes FP8 viable on 256×256

### Priority 3: Activation Scale UE8M0 Quantization (Blackwell only)
- **Expected Savings**: 10-20 ms (large batches only)
- **Risk**: Low
- **Note**: H20 (SM90) doesn't benefit much

### Priority 4: Cache-DiT Integration 🚫 LOW PRIORITY
- **Finding**: No effect on 9-step inference
- **Better use**: Larger batches or longer chains

---

## 📖 Recommended Reading Order

1. **[PROFILING_ANALYSIS_SUMMARY.md](./PROFILING_ANALYSIS_SUMMARY.md)** (30-45 min)
   - Overview of all findings
   - Context for everything else

2. **zimage_256_256/deep_gemm/introduce.md** (5-10 min)
   - Understand what DeepGEMM is
   - Why it's fast and what it requires

3. **CSV Kernel Statistics** (10-15 min)
   - Read `nsys_bf16_kernels_cuda_gpu_kern_sum.csv` (top 10 kernels)
   - Compare with `nsys_fp8_kernels_cuda_gpu_kern_sum.csv`
   - Understand where time is spent

4. **zimage_256_256/deep_gemm/report.md** (15-20 min)
   - Deep analysis of FP8 precision, hardware utilization
   - Why 256×256 is worst case for DeepGEMM

5. **zimage_256_256/review/pre_transpose.md** (20-30 min)
   - Optimization proposal and critical issue
   - What needs to be fixed before merge

6. **Generated Charts** (5-10 min)
   - Visualize all findings
   - Share with stakeholders

---

## 🔬 How to Use the Data

### For Performance Investigation
```
1. Check baseline_1gpu.json for E2E latency
2. Review nsys_bf16_kernels_cuda_gpu_kern_sum.csv for bottlenecks
3. Use analyze_profile.py to visualize findings
```

### For Optimization Planning
```
1. Read PROFILING_ANALYSIS_SUMMARY.md sections 8-9
2. Review deep_gemm/report.md for multi-resolution scaling
3. Check review/pre_transpose.md for implementation status
```

### For Code Review
```
1. Read review/pre_transpose.md for all issues found
2. Check SM90 assertion failure details
3. Validate fixes on GPU before merge
```

### For Presentations
```
1. Use 04_fp32_vs_bf16_gemm.png as evidence chart
2. Reference PROFILING_ANALYSIS_SUMMARY.md section 8 for metrics
3. Show multi-resolution breakdown (section 10)
```

---

## ✅ Verification Checklist

Before deploying any optimizations:

- [ ] Verify transpose overhead source (sfb vs sfa)
- [ ] Test SM90 assertion with current code
- [ ] Run diagnose_deepgemm_transpose.py
- [ ] Benchmark TextEncoder BF16 conversion
- [ ] Validate FP8 precision on VAE decoder
- [ ] Test pre-transpose on multi-resolution (256×256, 512×512, 1024×1024)
- [ ] Confirm image output quality
- [ ] Validate on both H20 (SM90) and Blackwell (SM100) if available

---

## 📞 Contact & Questions

For questions about:
- **Profiling data**: See PROFILING_ANALYSIS_SUMMARY.md
- **DeepGEMM specifics**: See deep_gemm/report.md
- **Code issues**: See review/pre_transpose.md
- **Optimization roadmap**: See section 9 of main summary

---

## 📄 Document Versions

- **PROFILING_ANALYSIS_SUMMARY.md**: v1.0 (2026-03-27)
- **PROFILING_EXPLORATION_INDEX.md** (this file): v1.0 (2026-03-27)
- **Analysis generated by**: analyze_profile.py + manual deep review
- **Profiling data collected on**: H20 GPU (SM90), ZImage-Turbo 256×256

