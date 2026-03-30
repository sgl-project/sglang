# Z-Image-Turbo 性能分析与优化报告

> **Hardware**: NVIDIA H20 (SM90, 96GB HBM3)
> **Model**: Z-Image-Turbo (30-layer DiT, dim=3840, 9 denoising steps, no CFG)
> **Date**: 2026-03-30 (updated)

---

## 目录

- [1. 瓶颈分析（优化前）](#1-瓶颈分析优化前)
  - [1.1 E2E 时延分解](#11-e2e-时延分解)
  - [1.2 CUDA Kernel 分类分析](#12-cuda-kernel-分类分析)
  - [1.3 DiT Denoising 细粒度 Kernel 分解](#13-dit-denoising-细粒度-kernel-分解)
  - [1.4 关键发现](#14-关键发现)
- [2. 关于 TextEncoder 精度的说明](#2-关于-textencoder-精度的说明)
- [3. 优化 #1：DiT FP8 量化](#3-优化-1dit-fp8-量化)
  - [3.1 量化方案设计](#31-量化方案设计)
  - [3.2 FP8 三层 Bug 修复](#32-fp8-三层-bug-修复)
  - [3.3 FP8 实测结果与根因分析](#33-fp8-实测结果与根因分析)
  - [3.4 CUTLASS vs DeepGemm vs BF16 对比](#34-cutlass-vs-deepgemm-vs-bf16-对比)
  - [3.5 DeepGemm Transpose 开销分析](#35-deepgemm-transpose-开销分析)
- [4. 多分辨率全面 Benchmark](#4-多分辨率全面-benchmark)
  - [4.1 E2E 时延总表](#41-e2e-时延总表)
  - [4.2 跨分辨率加速比对比](#42-跨分辨率加速比对比)
  - [4.3 Stage 瓶颈转移趋势](#43-stage-瓶颈转移趋势)
  - [4.4 VRAM 节省](#44-vram-节省)
  - [4.5 关键发现总结](#45-关键发现总结)
- [5. Cache-DiT 调优分析](#5-cache-dit-调优分析)
  - [5.1 默认配置与实际缓存行为](#51-默认配置与实际缓存行为)
  - [5.2 调优方案与测试结果](#52-调优方案与测试结果)
  - [5.3 Cache-DiT 原理简述](#53-cache-dit-原理简述)
- [6. torch.compile 收益分析](#6-torchcompile-收益分析)
- [7. 256×256 GEMM Memory-Bound 分析](#7-256256-gemm-memory-bound-分析)
  - [7.1 H20 Roofline 理论分析](#71-h20-roofline-理论分析)
  - [7.2 Wave Quantization 效应](#72-wave-quantization-效应)
- [8. H100 vs H20 芯片对比](#8-h100-vs-h20-芯片对比)
- [9. 优化路线图](#9-优化路线图)
- [10. nsys No-Warmup Kernel 精细对比](#10-nsys-no-warmup-kernel-精细对比baseline-vs-fp8-deepgemm)
  - [10.1 跨分辨率 GPU Kernel 总时间对比](#101-跨分辨率-gpu-kernel-总时间对比)
  - [10.2 分类 Kernel 时间对比](#102-分类-kernel-时间对比)
  - [10.3 FP8 新增与移除 Kernel](#103-fp8-新增与移除-kernel)
  - [10.4 逐 Shape GEMM 加速比分析](#104-逐-shape-gemm-加速比分析)
  - [10.5 FP8 Overhead 时间收支](#105-fp8-overhead-时间收支)
  - [10.6 非 GEMM Kernel 跨分辨率对比](#106-非-gemm-kernel-跨分辨率对比)
  - [10.7 关键结论](#107-关键结论)
- [11. 优化 #3：CUDA Graph 消除 Host Dispatch 开销](#11-优化-3cuda-graph-消除-host-dispatch-开销)
  - [11.1 问题根因](#111-问题根因)
  - [11.2 方案设计](#112-方案设计)
  - [11.3 多分辨率 E2E 实测结果](#113-多分辨率-e2e-实测结果)
  - [11.4 Per-Step 分析](#114-per-step-分析)
  - [11.5 CUDA Graph 收益分析](#115-cuda-graph-收益分析)
  - [11.6 关键发现](#116-关键发现)
- [附录 A：与 Yikai 1024×1024 Profile 的对比](#附录-a与-yikai-10241024-profile-的对比)
- [附录 B：复现命令](#附录-b复现命令)

---

## 1. 瓶颈分析（优化前）

### 1.1 E2E 时延分解

**Baseline**: 1 GPU, BF16 TextEncoder（ZImage 默认配置）, 256×256, E2E = 332ms

| Stage | Time (ms) | 占比 | 说明 |
|-------|-----------|------|------|
| TextEncodingStage | 51 | 15.4% | Qwen3 text encoder (BF16) |
| DenoisingStage | 268 | 80.7% | **#1 瓶颈** — 9 steps, ~33ms/step |
| DecodingStage | 10 | 3.0% | VAE decoder |

> 注：ZImage pipeline config 默认 `text_encoder_precisions: ("bf16",)`（zimage.py 第 49 行），TextEncoder 始终以 BF16 运行。早期报告中的"FP32 TextEncoder"数据来自旧版 sglang 的通用基准（base.py 默认 `("fp32",)`），不适用于 ZImage。本报告所有 baseline 数据均基于 ZImage 默认配置（BF16 TextEncoder）。

![Pipeline Breakdown](01_pipeline_and_kernel_breakdown.png)

### 1.2 CUDA Kernel 分类分析

**Total GPU kernel time: ~341ms (Torch Profiler, BF16 baseline)**

| Category | Time (ms) | % | 说明 |
|----------|-----------|---|------|
| BF16 GEMM (DiT + TE) | 279.6 | 81.8% | **优化目标** (FP8) |
| Elementwise Ops | 18.2 | 5.3% | |
| Convolution (VAE) | 12.6 | 3.7% | |
| FlashAttention | 10.7 | 3.1% | 短序列(768 tokens), 非瓶颈 |
| RMSNorm / QKNorm | 5.7 | 1.7% | |

![GEMM Breakdown](04_fp32_vs_bf16_gemm.png)

### 1.3 DiT Denoising 细粒度 Kernel 分解

使用 torch profiler trace + python call stack 分析，按功能标签分类每个 CUDA kernel：

**Denoising GPU kernel time: 287.55 ms**

| Tag | Duration (ms) | % | 说明 |
|-----|---------------|---|------|
| feedforward:gemm | 163.80 | 56.96% | **#1** — SwiGLU FFN (w13+w2) |
| qkv_projection:gemm | 64.02 | 22.26% | **#2** — Q, K, V 线性投影 |
| output_projection:gemm | 21.24 | 7.39% | **#3** — Attention output (to_out) |
| rms_norm_gate:elementwise | 10.99 | 3.82% | adaLN gate/scale |
| usp_attention:attention | 9.72 | 3.38% | FlashAttention |
| rms_norm:norm | 4.16 | 1.45% | RMSNorm |
| 其他 (silu, rope, adaln, copy) | 13.62 | 4.74% | |

**GEMM 汇总**：占 Denoising 的 **87.2%**（250.74ms / 287.55ms）

> **核心结论**：FP8 量化覆盖 feedforward + qkv + output_projection 即可覆盖 **99.3%** 的 Denoising GEMM。

### 1.4 关键发现

1. **GEMM 占 ~82%** GPU kernel 时间（DiT BF16 GEMM + TextEncoder BF16 GEMM）
2. **FlashAttention 仅 3.1%** — 短序列(768 tokens)下 attention 非瓶颈
3. **torch.compile 导致严重回退** — 256×256 下 +48%，Triton kernel 在小矩阵上比 cuBLAS/nvJET 慢
4. **2-GPU SP 无收益** — 768 tokens 太短，通信开销 > 并行收益
5. **Cache-DiT 默认配置效果有限** — 9 步仅 1 步命中缓存

---

## 2. 关于 TextEncoder 精度的说明

> ⚠️ **勘误**：早期报告（及部分图表）中记录了"FP32 TextEncoder → BF16"的优化步骤，这基于旧版 sglang-diffusion 的通用默认配置（`base.py` 第 195 行 `text_encoder_precisions: ("fp32",)`）。但 **ZImage 的 pipeline config 从一开始就将 text encoder 默认为 BF16**（`zimage.py` 第 49 行 `text_encoder_precisions: ("bf16",)`），因此 ZImage 模型从未以 FP32 TextEncoder 运行。
>
> 早期报告中的 FP32 数据（E2E=362ms、TextEnc=81ms）**不是 ZImage 的真实 baseline**，仅反映旧版 sglang 的通用配置行为。本报告的所有 baseline 数据已更正为 ZImage 默认配置（BF16 TextEncoder），来源为 `zimage_bench/*/baseline_1gpu.json`。
>
> Section 1 中引用的图表（`01_pipeline_and_kernel_breakdown.png`、`04_fp32_vs_bf16_gemm.png`、`07_bottleneck_shift_before_after.png`、`08_kernel_comparison_fp32_vs_bf16.png`）是基于早期旧基准生成的，其中包含 FP32 TextEncoder 的数据，**仅供历史参考，不代表 ZImage 的实际优化路径**。

**ZImage 的实际优化起点**是 BF16 TextEncoder baseline，直接进入 DiT FP8 量化（Section 3）。

---

## 3. 优化 #1：DiT FP8 量化

### 3.1 量化方案设计

**决策总结**：

| 维度 | 选择 | 理由 |
|------|------|------|
| 量化方案 | **FP8 W8A8** | sglang-diffusion NVIDIA GPU 上唯一方案，INT8 仅华为 NPU |
| 格式 | **E4M3 (float8_e4m3fn)** | 工业标准(NVIDIA TE, vLLM, SGLang)，E5M2 仅用于训练梯度 |
| 粒度 | **Block 128×128** | 最优精度/速度权衡，DeepGemm 原生支持 |
| 标定 | **动态 per-token** | Diffusion 激活分布跨 timestep 变化大，静态 scale 不适用 |
| GEMM 后端 | **DeepGemm（H20 默认）** | JIT + shape autotuning，比 CUTLASS 固定 tile 更快 |

**量化层选择**：

| 层 | 量化？ | GEMM 占比 |
|----|:---:|:---:|
| feedforward.w13 (gate+up) | ✅ | 56.96% |
| feedforward.w2 (down) | ✅ | (含在 FFN 内) |
| to_q, to_k, to_v | ✅ | 22.26% |
| to_out | ✅ | 7.39% |
| adaLN_modulation | ❌ | ~0.6%，对量化敏感 |
| TextEncoder / VAE / 嵌入层 | ❌ | 分开处理 |

> ⚠️ 原 `convert_hf_to_fp8.py` 从 FLUX 继承了 `"feed_forward" not in key` 排除规则，导致 FFN（占 GEMM 57%）不被量化。已移除此排除。

### 3.2 FP8 三层 Bug 修复

初次 FP8 测试出现 **2× 性能回退 + 图像损坏**。根因：`weight_scale_inv` 全部未被加载（静默跳过），模型使用默认初始值 `float32.min ≈ -3.4e+38`。

**三层 Bug 及修复**：

| Bug | 问题 | 修复文件 |
|-----|------|---------|
| **Bug 1** | `FeedForward` 类不接受 `quant_config` → w13/w2 无量化 → 无 `weight_scale_inv` 参数 | `runtime/models/dits/zimage.py` |
| **Bug 2** | `param_names_mapping` 缺少 `w1.weight_scale_inv → w13.weight_scale_inv` 映射 | `configs/models/dits/zimage.py` |
| **Bug 3** | `BlockQuantScaleParameter.weight_loader_v2` 未实现（raise NotImplementedError） | `runtime/layers/linear.py` |

修复后验证：
- ✅ 所有 `weight_scale_inv` 正确加载
- ✅ 图像质量完全正确
- ✅ VRAM 从 20.2GB 降至 14.5GB（**-28%**）

### 3.3 FP8 实测结果与根因分析

**FP8 修复后仍未达到预期加速**：

| 配置 | E2E (ms) | Denoise (ms) | SS Step (ms) | Peak VRAM |
|------|:---:|:---:|:---:|:---:|
| BF16 TextEnc | 332 | 267 | 32.9ms | 20,167MB |
| BF16+FP8 DeepGemm (含FFN) | 404 | 343 | 40.9ms | **14,453MB** |
| BF16+FP8 CUTLASS | 384 | 289 | 34.1ms | **14,452MB** |

> 256×256 下 FP8 **未加速反而略慢**，但 VRAM 节省 28%。

**nsys Kernel 级根因（早期含 warmup 的 profile）**：

| 指标 | BF16 nvjet | FP8 DeepGemm | FP8 CUTLASS |
|------|:---:|:---:|:---:|
| DiT GEMM 计算 | 260.0ms | **171.8ms** ✅ | 238.8ms |
| 量化开销 | 0ms | +11.8ms | +16.2ms |
| **GEMM 总计** | **260.0ms** | **183.6ms** | **255.0ms** |

> ⚠️ **勘误**：此表数据来自早期**未排除 warmup** 的 nsys profile。原版本包含 139.8ms transpose 开销，后经 Section 10 的 no-warmup 精确分析证明该 transpose 仅发生在 DeepGemm JIT 编译的 warmup 阶段（weight scale 预处理），推理阶段无 transpose 开销。上表已移除 transpose 行。正确的推理阶段 kernel 数据见 Section 10。

### 3.4 CUTLASS vs DeepGemm vs BF16 对比

**逐 GEMM Shape 分析**：

| GEMM 用途 | Shape (M×K×N) | BF16 nvjet (μs) | DeepGemm FP8 (μs) | CUTLASS FP8 (μs) | DeepGemm 加速 |
|-----------|:---:|:---:|:---:|:---:|:---:|
| FFN w13 | 768×3840×10240 | 396 | **227** | ~117 | **1.75×** |
| QKV+output | 768×3840×3840 | 73.8 | **47.8** | ~117 | **1.54×** |
| FFN w2 | 768×10240×3840 | 175 | **120** | ~117 | **1.46×** |
| **DiT 合计** | — | **260.0ms** | **171.8ms** | **238.8ms** | **1.51×** |

> **DeepGemm GEMM 计算快 1.51×**（早期含 warmup 数据；Section 10 no-warmup 数据显示 ~1.9× 加速）。

### 3.5 早期 Transpose 开销分析（已证伪）

> ⚠️ **已证伪**：本节原记录了 139.8ms transpose 开销，认为是 FP8 DeepGemm 在推理阶段的主要瓶颈。Section 10 的 `cudaProfilerStart()`/`cudaProfilerStop()` 精确框选推理区间的 nsys 分析证明：**该 transpose 仅发生在 DeepGemm JIT 编译（warmup）阶段**，用于将 weight scale 从行优先转为 Hopper TMA 要求的列优先格式。推理阶段 scale 已预处理完毕，无额外 transpose。
>
> 早期 profile 观察到 transpose 的原因是 nsys 录制范围包含了 JIT warmup，未使用 CUDA Profiler API 排除。这一误导性结论促使了后续的 no-warmup 精细分析（Section 10），最终定位到真正的 E2E 瓶颈——host-side dispatch 开销（Section 11 CUDA Graph 解决）。
>
> **原始数据（仅供历史参考）**：Transpose 总时间 139.8ms，49,152 次 kernel launch，GEMM 计算 171.8ms — 均为 warmup 阶段数据。

---

## 4. 多分辨率全面 Benchmark

> **测试时间**：2026-03-23 | **硬件**：1×H20 / 2×H20 (Ulysses SP)
> **配置**：多种优化组合 × 3 种分辨率
>
> 注：Baseline 为 ZImage 默认配置（BF16 TextEncoder），非 FP32。下表中"1GPU BF16-TE"即为 baseline。早期表格中的"1GPU FP32 (baseline)"标注有误，已更正。

### 4.1 E2E 时延总表

#### 256×256 (seq_len ≈ 768)

| 配置 | E2E (ms) | TextEnc | Denoise | SS Step | vs Baseline | VRAM |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1GPU BF16-TE (baseline) | **332** | 51 | 268 | 33.0ms | — | 20.2GB |
| BF16-TE+CacheDiT | **336** | 63 | 264 | 31.9ms | **1.08×** | 20.2GB |
| CacheDiT+FP8-CUTLASS | 384 | 86 | 289 | 29.7ms | 0.94× | **14.5GB** |
| CacheDiT+FP8-DeepGemm | 404 | 52 | 343 | 35.4ms | 0.90× | **14.5GB** |
| BF16-TE+Compile | 491 | 52 | 431 | 47.0ms | 0.74× ❌ | 20.2GB |
| 2GPU SP | 385 | 51 | 326 | 35.7ms | 0.94× | 20.2GB |

#### 512×512 (seq_len ≈ 3,072)

| 配置 | E2E (ms) | TextEnc | Denoise | SS Step | vs Baseline | VRAM |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1GPU BF16-TE (baseline) | **923** | 51 | 838 | 104.5ms | — | 21.1GB |
| BF16-TE+CacheDiT | **865** | 78 | 779 | 94.4ms | **1.07×** | 21.1GB |
| **CacheDiT+FP8-CUTLASS** | **753** | 187 | 558 | 65.5ms | **1.23×** | **15.4GB** |
| **CacheDiT+FP8-DeepGemm** | **632** | 68 | 555 | 66.8ms | **1.46×** | **15.4GB** |
| 2GPU SP | **635** | 51 | 564 | 67.7ms | **1.45×** | 21.1GB |
| **2GPU+FP8-DeepGemm** | **575** | 52 | 515 | 56.9ms | **1.61×** | **15.4GB** |

#### 1024×1024 (seq_len ≈ 12,288)

| 配置 | E2E (ms) | TextEnc | Denoise | SS Step | vs Baseline | VRAM |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1GPU BF16-TE (baseline) | **3609** | 51 | 3413 | 422.7ms | — | 24.7GB |
| BF16-TE+CacheDiT | **3237** | 56 | 3069 | 365.2ms | **1.12×** | 24.8GB |
| **CacheDiT+FP8-CUTLASS** | **2388** | 51 | 2252 | 269.9ms | **1.52×** | **19.1GB** |
| **CacheDiT+FP8-DeepGemm** | **2186** | 51 | 2082 | 248.4ms | **1.66×** | **19.1GB** |
| 2GPU SP | **2166** | 65 | 2038 | 244.3ms | **1.68×** | 24.7GB |
| **2GPU+FP8-DeepGemm** | **1639** | 51 | 1540 | 184.4ms | **2.21×** | **19.0GB** |

### 4.2 跨分辨率加速比对比

![E2E Speedup by Resolution](multi_res_speedup.png)

| 配置 | 256×256 | 512×512 | **1024×1024** |
|------|:---:|:---:|:---:|
| BF16-TE+CacheDiT | 1.08× | 1.07× | **1.12×** |
| CacheDiT+FP8-CUTLASS | 0.94× | 1.23× | **1.52×** |
| **CacheDiT+FP8-DeepGemm** | 0.90× | **1.46×** | **1.66×** |
| 2GPU SP | 0.94× | 1.45× | **1.68×** |
| **2GPU+FP8-DeepGemm** | — | **1.61×** | **2.21×** |

> **核心发现**：FP8+DeepGemm 在 1024×1024 单 GPU 达到 **1.66× 加速**，几乎追平 2GPU SP（1.68×）；2GPU+FP8 达到 **2.21×**。

### 4.3 Stage 瓶颈转移趋势

![Stage Breakdown](multi_res_stage_breakdown.png)

| 分辨率 | TextEncoder 占比 | Denoising 占比 | Decoding 占比 |
|--------|:---:|:---:|:---:|
| 256×256 | 22.4% | **74.3%** | 2.7% |
| 512×512 | 5.6% | **90.7%** | 3.4% |
| 1024×1024 | 2.0% | **94.1%** | 3.9% |

> 随分辨率增大，Denoising 占比从 74% 升到 94%，TextEncoder 优化被稀释。

### 4.4 VRAM 节省

![Peak VRAM](multi_res_vram.png)

| 配置 | 256×256 | 512×512 | 1024×1024 |
|------|:---:|:---:|:---:|
| BF16 | 20.2GB | 21.1GB | 24.7GB |
| **FP8** | **14.5GB** | **15.4GB** | **19.1GB** |
| **节省** | **-28%** | **-27%** | **-23%** |

> FP8 在所有分辨率下稳定节省 ~5.5GB VRAM。

### 4.5 关键发现总结

#### ✅ 有效的优化

1. **FP8+DeepGemm + CacheDiT 是 1GPU 最优方案**（≥512×512）
   - 512×512: **1.46×**（924ms → 632ms）
   - 1024×1024: **1.66×**（3628ms → 2186ms），几乎追平 2GPU
2. **2GPU+FP8-DeepGemm 是最强方案**（1024×1024: **2.21×**）
3. **CacheDiT** 在所有分辨率稳定提升 +7~12%
4. **BF16-TE** 在 256×256 有效（TextEnc 占比高），大分辨率无感

#### ❌ 无效的优化

1. **torch.compile**：256×256 严重回退(-36%)，大分辨率收益 ≤3%
2. **FP8 在 256×256（不含 CUDA Graph）**：GPU kernel 已快 25-30%，但 host dispatch 开销抵消了全部加速（已通过 CUDA Graph 解决，见 Section 11）
3. **2GPU SP 在 256×256**：通信开销 > 并行收益

#### 分辨率适配推荐

| 分辨率 | 推荐配置 | E2E | 加速 |
|--------|---------|:---:|:---:|
| 256×256 | **FP8-DeepGemm + CUDA Graph** | **263ms** | **1.26×** |
| 512×512 | **FP8-DeepGemm + CUDA Graph** | **693ms** | **1.33×** |
| 1024×1024 | FP8-DeepGemm (+ CUDA Graph) | 2,604ms | 1.38× |
| 1024×1024 (2GPU) | 2GPU + FP8-DeepGemm | 1,639ms | 2.21× |

> 注：256×256 推荐从 BF16-TE 更新为 FP8+CUDA Graph（Section 11 验证后）。CUDA Graph 在 256×256 收益最大（-45% per-step），在 1024×1024 收益微弱（-1% per-step）但无副作用。

---

## 5. Cache-DiT 调优分析

### 5.1 默认配置与实际缓存行为

**默认参数**：

| 参数 | 默认值 | 含义 |
|------|:---:|------|
| W (Warmup) | **4** | 前 4 步不缓存 |
| R (Threshold) | **0.24** | 残差 L1 阈值 |
| MC (Max Continuous) | **3** | 最大连续缓存步数 |
| Fn / Bn | 1 / 0 | 前 1 层始终计算，无尾部校准 |

**实际缓存模式（9 步）**：

```
Step:  0    1    2    3    4       5     6    7    8
       W    W    W    W    Build   ☑HIT  C    C    C
```

**问题**：9 步中仅 **1 步命中缓存**（11%）。W=4 消耗了 44% 的步数。

**各分辨率 Cache-DiT 默认配置净收益**：

| 分辨率 | Step 5 节省 | Step 4 overhead | 净收益 | 占 Denoising % |
|--------|:---:|:---:|:---:|:---:|
| 256×256 | -11.1ms | +3.8ms | ~7ms | 2.6% |
| 512×512 | -77.0ms | +13.0ms | ~64ms | 7.6% |
| 1024×1024 | -363.3ms | +14.0ms | ~349ms | 10.2% |

### 5.2 调优方案与测试结果

测试了多组 Cache-DiT 参数调优和 SCM 预设：

#### 256×256 Cache-DiT 调优结果

| 配置 | E2E (ms) | Denoise (ms) | 缓存命中步数 | vs Baseline |
|------|:---:|:---:|:---:|:---:|
| 默认 CacheDiT | 336 | 264 | 1/9 | 1.08× |
| W=2, R=0.35, MC=5 | **318** | **234** | ~3/9 | **1.14×** |
| W=2, R=0.45, MC=6 | **301** | **235** | ~3/9 | **1.20×** |
| **W=2, R=0.50, MC=6** | **268** | **208** | **~4/9** | **1.35×** |
| W=1, R=0.40, MC=6 | **321** | **237** | ~3/9 | **1.13×** |
| SCM fast | 335 | 248 | ~2/9 | 1.08× |
| SCM ultra | **319** | **251** | ~3/9 | **1.13×** |

#### 1024×1024 Cache-DiT 调优结果

| 配置 | E2E (ms) | Denoise (ms) | vs Baseline |
|------|:---:|:---:|:---:|
| 默认 CacheDiT | 3237 | 3069 | 1.12× |
| W=2, R=0.35, MC=5 | **2858** | **2687** | **1.27×** |
| W=1, R=0.40, MC=6 | **2852** | **2689** | **1.27×** |
| SCM fast | 3249 | 3036 | 1.12× |
| SCM ultra | 3264 | 3036 | 1.11× |

> **发现**：
> - **DBCache 参数调优**（降 W、提 R、增 MC）比 SCM 预设更有效
> - **W=2, R=0.50, MC=6** 在 256×256 达到 **1.35×**（最佳单项优化）
> - ⚠️ R 越高质量风险越大，需固定 seed 对比验证

### 5.3 Cache-DiT 原理简述

**DBCache 核心算法**：将 DiT 30 层分为 Fn（始终算）/ Mn（动态判断）/ Bn（尾部校准）三段。每步计算 Fn 层输出的 L1 残差距离，低于 R 则缓存命中（跳过 Mn 层），否则完整计算。

**SCM（Steps Computation Masking）**：在 DBCache 之上叠加步级掩码，预标记哪些步可缓存。`dynamic` 模式结合 R 判断，`static` 严格按掩码。

**TaylorSeer**：用 Taylor 展开预测特征替代简单复用，**不适合 ZImage-Turbo**（9 步蒸馏模型，步间变化剧烈，Taylor 近似不准确）。

---

## 6. torch.compile 收益分析

| 分辨率 | 无 compile SS Step (ms) | 有 compile SS Step (ms) | 变化 | E2E 变化 |
|--------|:---:|:---:|:---:|:---:|
| **256×256** | ~33 | ~47 | **+42% 更慢** | **+48% 负收益** |
| **512×512** | ~104 | ~102 | -2% | -1.8% |
| **1024×1024** | ~422 | ~415 | -1.7% | -2.6% |

**根本原因**：模型关键路径已被 custom CUDA kernel 全覆盖

| 操作 | 占比 | 已有优化 | compile 能否优化 |
|------|------|---------|:---:|
| FlashAttention | ~40-50% | sgl_kernel | ❌ 不透明 custom op |
| Linear (GEMM) | ~30-40% | cuBLAS | ❌ 无法被 Inductor 改写 |
| RMSNorm / SiLU / QKNorm / ROPE | ~8-12% | Triton/sgl_kernel/FlashInfer | ❌ 已是 custom kernel |
| **AdaLN modulation + reshape** | **~5%** | 纯 PyTorch eager | **✅ 唯一可优化** |

torch.compile 只能优化最后 ~5% 的 "胶水代码"，但引入的 graph break overhead（每个 custom op 断图一次）和 Triton kernel launch overhead 在小矩阵场景下超过收益。

**结论**：256×256 应**关闭** torch.compile。若需优化 AdaLN，应直接手写 Triton fused kernel。

---

## 7. 256×256 GEMM Memory-Bound 分析

### 7.1 H20 Roofline 理论分析

| 精度 | H20 TFLOPS | HBM BW | Ridge Point (FLOP/byte) |
|------|:---:|:---:|:---:|
| BF16 | 148 | 4.0 TB/s | **37** |
| FP8 | 296 | 4.0 TB/s | **74** |

256×256 下 M=768 的 GEMM，理论 arithmetic intensity 虽高（~650 FLOP/byte），但**实际运行效率受限于 tile utilization**：

- cuBLAS/nvjet 典型 tile: 128×128
- M=768 只有 6 个 tile rows → weight tile 最多复用 6 次
- 对比 M=16384 (1024×1024) 有 128 个 tile rows → 复用 128 次

### 7.2 Wave Quantization 效应

H20 有 78 个 SM。以 QKV projection (M=768, N=3840) 为例：

```
tile 128×128: grid = ceil(768/128) × ceil(3840/128) = 6 × 30 = 180 tiles
wave 数 = ceil(180/78) = 3 waves
最后一个 wave: 180 - 2×78 = 24 tiles → 24/78 = 31% SM 利用率

对比 M=16384 (1024×1024):
grid = 128 × 30 = 3840 tiles → ~49 waves (nearly full)
→ 平均 SM 利用率远高于 256×256
```

> ⚠️ **勘误**：本节原结论为"256×256 GEMM 是 bandwidth-bound，FP8 无法转化为实际加速"。该结论基于 Roofline 理论分析和 wave quantization 推导，但**未被 nsys 实测数据验证**。Section 10.4 的 no-warmup kernel 数据显示，DeepGemm FP8 在 256×256（M=768）下实现了 **1.93× GEMM 加速**，与 512×512（1.84×）和 1024×1024（1.90×）高度一致，接近 FP8 理论 2× 极限。这说明 DeepGemm 的 JIT tile autotuning 在小 M 值下仍能有效利用 FP8 TFLOPs，所有测试分辨率下 GEMM 均表现为 compute-bound。
>
> 256×256 E2E 未加速的真正原因是 **host-side dispatch 开销**（Section 11 CUDA Graph 解决），而非 GPU kernel 层面的 bandwidth-bound。

---

## 8. H100 vs H20 芯片对比

| 规格 | **H100 SXM5** | **H20** | H20/H100 |
|------|:---:|:---:|:---:|
| SM 数量 | 132 | 78 | 59% |
| **BF16 TFLOPs** | **989** | **148** | **15%** |
| **FP8 TFLOPs** | **1,979** | **296** | **15%** |
| **HBM 带宽** | **3.35 TB/s** | **4.0 TB/s** | **119%** |
| 显存容量 | 80 GB | 96 GB | 120% |

> H20 是出口管制下的 H100 缩减版：保留高带宽内存，削减算力。

**不同场景速度预测**：

| 场景 | H20 特性 | H100 vs H20 |
|------|---------|:---:|
| 256×256 GEMM (M=768) | H20 HBM BW 优势 | H100 受限于算力削减，预期加速有限 |
| 1024×1024 GEMM (M=12288) | 计算密集 | **H100 快 5-6.7×** |

> **结论**：H20 的算力弱势在大分辨率下最为显著。

---

## 9. 优化路线图

> **Updated**: 2026-03-30，基于 CUDA Graph 实测结果（Section 11）更新

#### 已完成

| 状态 | 优化 | 目标 | 收益 | 说明 |
|:---:|------|------|------|------|
| ✅ | TextEncoder FP32→BF16 | TextEncoding | -35% E2E (旧基准) | CLI flag, 零代码 |
| ✅ | FP8 scale 加载修复（3 层 Bug） | Bug fix | VRAM -28%, 图像恢复 | 4 处代码修复 |
| ✅ | CUTLASS FP8 验证 | Denoising GEMM | 256×256 无 transpose 但 GEMM 不如 DeepGemm | 对照组 |
| ✅ | 多分辨率 Benchmark | 全面验证 | 确认 FP8 在 ≥512 E2E 有效 | 24 次测试 |
| ✅ | Cache-DiT 调优 | Denoising skip | 256×256 最高 1.35× | W/R/MC 参数调优 |
| ✅ | nsys no-warmup Kernel 分析 | 根因定位 | **GEMM ~1.9× 加速（所有分辨率）** | 推翻 transpose 瓶颈假说 |
| ✅ | **CUDA Graph capture/replay** | **E2E latency** | **256×256: 1.26×, 512×512: 1.33×** | **消除 host dispatch，Graph 跨请求缓存** |

#### 优化路线总览

三步优化路线在所有分辨率下的累计效果：

| 分辨率 | BF16 Baseline | + FP8 DeepGemm | + CUDA Graph | **总加速** |
|--------|:---:|:---:|:---:|:---:|
| 256×256 | 330ms | 443ms (1.34× 慢 ❌) | **263ms** | **1.26×** ✅ |
| 512×512 | 923ms | 720ms (1.28×) | **693ms** | **1.33×** ✅ |
| 1024×1024 | 3,607ms | 2,621ms (1.38×) | **2,604ms** | **1.38×** ✅ |

> **核心成果**：CUDA Graph 解决了 256×256 FP8 反而更慢的问题（443ms → 263ms），使 FP8 在**所有分辨率**下都实现了 E2E 加速。

#### 待做优化（优先级重排）

| 优先级 | 优化 | 目标 | 预期收益 | 说明 |
|:---:|------|------|------|------|
| **P0** | **FlashAttention 优化（1024×1024）** | Attention | **-100~200ms（1024×1024 下 625ms, 占 FP8 GPU 21.7%）** | FP8 将 GEMM 减半后 Attention 成为 #2 瓶颈；可考虑 FP8 Attention / CuTe DSL / 更优 tile config |
| **P1** | **CUDA Graph + Cache-DiT 联合优化** | Denoising skip + replay | **各分辨率额外 +10~30%** | 当前 CUDA Graph 与 Cache-DiT 互斥（Cache-DiT 改变每步 kernel 序列）；可为 full/cached 两种模式各 capture 一个 graph，按 Cache-DiT 决策选择 replay |
| **P2** | FP8 量化 kernel 融合 | Quantize overhead | -10~97ms（视分辨率） | 当前每个 GEMM 前单独 launch quant kernel（1836 次），可尝试 fuse 到前序 kernel |
| **P3** | Fused adaLN + gate kernel | Elementwise | -5~10ms | Triton fused kernel，替代当前 ~20ms elementwise |
| **P4** | Better RMSNorm (Quack) | Norm | -2~5ms | 38ms@1024×1024，bandwidth-bound |
| **P5** | 多分辨率 Graph 缓存 + 预热 | 首请求延迟 | 消除首请求 3× step overhead | 当前首请求需 2× warmup + capture；可在 server 启动时预热常用分辨率 |
| ~~P0~~ | ~~CUDA Graph / 减少 host dispatch~~ | ~~E2E latency~~ | ~~已完成~~ | **Section 11，256×256 从 0.75× 恢复到 1.26×** |
| ~~P0~~ | ~~Pre-transpose weight scale~~ | ~~DeepGemm transpose~~ | ~~已无需~~ | **nsys no-warmup 证明推理阶段无 transpose 开销** |

> **路线图变更说明**：
> - **CUDA Graph 已完成（原 P0）**：256×256 从 FP8 反而更慢（0.75×）恢复到 1.26× 加速；graph 跨请求缓存，serve 场景下首次请求后无额外开销
> - **P0 更新为 FlashAttention 优化**：GEMM 已被 FP8+CUDA Graph 充分优化，1024×1024 下 FlashAttention (625ms) 成为最大非 GEMM 瓶颈
> - **新增 P1 CUDA Graph + Cache-DiT 联合**：两者当前互斥，联合后可在 CUDA Graph 的低 dispatch 开销基础上叠加 Cache-DiT 的步级跳过

---

## 10. nsys No-Warmup Kernel 精细对比（Baseline vs FP8 DeepGemm）

> **方法**：使用 `cudaProfilerStart()`/`cudaProfilerStop()` API 精确框选推理区间，排除 JIT 编译、权重加载、autotuning 等 warmup 开销。
> **数据源**：`zimage_bench/nsys_no_warmup/{256_256,512_512,1024_1024}/`

### 10.1 跨分辨率 GPU Kernel 总时间对比

| 分辨率 | seq_len | Baseline GPU (ms) | FP8 GPU (ms) | Delta (ms) | Kernel 加速 |
|--------|:---:|:---:|:---:|:---:|:---:|
| 256×256 | 768 | **348.13** | **260.87** | **-87.26 (-25.1%)** | **1.33×** |
| 512×512 | 3,072 | **1,045.60** | **740.13** | **-305.47 (-29.2%)** | **1.41×** |
| 1024×1024 | 12,288 | **4,131.08** | **2,875.79** | **-1,255.29 (-30.4%)** | **1.44×** |

> ⚠️ **关键发现**：去除 warmup 后，FP8 在**所有分辨率**下的 GPU kernel 总时间均快于 BF16（25-30%）。此前 E2E benchmark 中 256×256 FP8 反而更慢 (404ms vs 332ms)，说明小分辨率下 E2E 时延的主要 overhead 来自 **host-side 开销**（Python dispatch、DeepGemm JIT 路径、CUDA graph 缺失等），而非 GPU kernel 本身。

### 10.2 分类 Kernel 时间对比

#### 256×256 (M=768)

| Category | Baseline (ms) | 次数 | FP8 (ms) | 次数 | Delta (ms) | 说明 |
|----------|:---:|:---:|:---:|:---:|:---:|------|
| **BF16 GEMM (DiT)** | **277.83** | 2,088 | **0** | 0 | **-277.83** | 被 FP8 DeepGemm 替换 |
| **FP8 DeepGemm** | 0 | 0 | **154.56** | 1,836 | **+154.56** | 新增：FP8 GEMM 计算 |
| **FP8 Quantize** | 0 | 0 | **10.57** | 1,836 | **+10.57** | 新增：per-token BF16→FP8 量化 |
| BF16 GEMM (TE+adaLN) | 28.90 | 594 | 28.90 | 594 | 0 | TextEncoder + adaLN 未量化 |
| GEMM splitKreduce | 3.13 | 333 | 0.02 | 9 | -3.11 | BF16 splitK 被消除 |
| FlashAttention | 14.14 | 343 | 14.17 | 343 | 0 | 不变 |
| Elementwise | 21.93 | 5,265 | 21.94 | 5,265 | 0 | 不变 |
| Conv (VAE) | 13.88 | 134 | 13.93 | 134 | 0 | 不变 |
| RMSNorm | 4.20 | 1,233 | 4.20 | 1,233 | 0 | 不变 |
| SiLU Gate | 2.37 | 306 | 2.31 | 306 | 0 | 不变 |
| QKNorm | 1.55 | 306 | 1.44 | 306 | -0.11 | 略快 |
| RoPE | 1.24 | 306 | 1.23 | 306 | 0 | 不变 |

#### 512×512 (M=3072)

| Category | Baseline (ms) | 次数 | FP8 (ms) | 次数 | Delta (ms) | 说明 |
|----------|:---:|:---:|:---:|:---:|:---:|------|
| **BF16 GEMM (DiT)** | **858.29** | 2,088 | **0** | 0 | **-858.29** | 被 FP8 DeepGemm 替换 |
| **FP8 DeepGemm** | 0 | 0 | **496.67** | 1,836 | **+496.67** | 新增：FP8 GEMM 计算 |
| FlashAttention | 54.06 | 343 | 54.07 | 343 | 0 | 不变 |
| Conv (VAE) | 48.57 | 146 | 47.63 | 146 | -0.95 | 略快 |
| Elementwise | 41.29 | 5,265 | 41.41 | 5,265 | 0 | 不变 |
| **FP8 Quantize** | 0 | 0 | **28.27** | 1,836 | **+28.27** | 新增 |
| Norm_Other | 11.51 | 69 | 11.52 | 69 | 0 | 不变 |
| RMSNorm | 10.70 | 1,233 | 10.55 | 1,233 | -0.16 | 不变 |
| SiLU Gate | 7.22 | 306 | 7.38 | 306 | 0 | 不变 |
| QKNorm | 4.16 | 306 | 4.12 | 306 | 0 | 不变 |
| RoPE | 2.87 | 306 | 2.87 | 306 | 0 | 不变 |

#### 1024×1024 (M=12288)

| Category | Baseline (ms) | 次数 | FP8 (ms) | 次数 | Delta (ms) | 说明 |
|----------|:---:|:---:|:---:|:---:|:---:|------|
| **BF16 GEMM (DiT)** | **3,044.55** | 2,088 | **0** | 0 | **-3,044.55** | 被 FP8 DeepGemm 替换 |
| **FP8 DeepGemm** | 0 | 0 | **1,660.99** | 1,836 | **+1,660.99** | 新增：FP8 GEMM 计算 |
| **FlashAttention** | **624.97** | 343 | **624.57** | 343 | 0 | **成为 FP8 下第二大瓶颈** |
| Conv (VAE) | 170.52 | 161 | 171.75 | 161 | +1.24 | 不变 |
| Elementwise | 138.90 | 5,265 | 139.86 | 5,265 | +0.96 | 不变 |
| **FP8 Quantize** | 0 | 0 | **96.77** | 1,836 | **+96.77** | 新增 |
| Norm_Other | 46.76 | 69 | 46.76 | 69 | 0 | 不变 |
| RMSNorm | 38.44 | 1,233 | 39.30 | 1,233 | +0.85 | 不变 |
| SiLU Gate | 27.11 | 306 | 27.30 | 306 | +0.19 | 不变 |
| QKNorm | 14.89 | 306 | 14.87 | 306 | 0 | 不变 |
| RoPE | 12.92 | 306 | 12.85 | 306 | 0 | 不变 |

**Kernel Launch 数量对比**：

| 分辨率 | Baseline | FP8 | Delta |
|--------|:---:|:---:|:---:|
| 256×256 | 11,346 | 12,858 | +1,512 |
| 512×512 | 11,070 | 12,870 | +1,800 |
| 1024×1024 | 11,076 | 12,876 | +1,800 |

> 新增 launch 主要来自 FP8 DeepGemm (1,836) + FP8 Quantize (1,836)，同时消除了 BF16 nvjet GEMM (~1,836) 和 splitKreduce (~324)。

### 10.3 FP8 新增与移除 Kernel

#### 各分辨率新增 Kernel（FP8 独有，top 5）

**256×256**：

| Kernel | 总时间 (ms) | 次数 | 说明 |
|--------|:---:|:---:|------|
| `deep_gemm::sm90_fp8_gemm` (N=20480,K=3840) | 61.26 | 270 | FFN w13 |
| `deep_gemm::sm90_fp8_gemm` (N=3840,K=3840) | 51.54 | 1,080 | QKV+out |
| `deep_gemm::sm90_fp8_gemm` (N=3840,K=10240) | 32.45 | 270 | FFN w2 |
| `per_token_group_quant_8bit_kernel` | 10.57 | 1,836 | FP8 量化 |
| DeepGemm autotune 变体 | 9.32 | 234 | tile 探索 |

**512×512**：

| Kernel | 总时间 (ms) | 次数 | 说明 |
|--------|:---:|:---:|------|
| `deep_gemm::sm90_fp8_gemm` (N=20480,K=3840) | 210.02 | 270 | FFN w13 |
| `deep_gemm::sm90_fp8_gemm` (N=3840,K=3840) | 156.28 | 1,080 | QKV+out |
| `deep_gemm::sm90_fp8_gemm` (N=3840,K=10240) | 101.43 | 270 | FFN w2 |
| `per_token_group_quant_8bit_kernel` | 28.27 | 1,836 | FP8 量化 |
| DeepGemm autotune 变体 | 28.94 | 234 | tile 探索 |

**1024×1024**：

| Kernel | 总时间 (ms) | 次数 | 说明 |
|--------|:---:|:---:|------|
| `deep_gemm::sm90_fp8_gemm` (N=20480,K=3840) | 708.04 | 288 | FFN w13 |
| `deep_gemm::sm90_fp8_gemm` (N=3840,K=3840) | 537.61 | 1,080 | QKV+out |
| `deep_gemm::sm90_fp8_gemm` (N=3840,K=10240) | 353.61 | 270 | FFN w2 |
| `per_token_group_quant_8bit_kernel` | 96.77 | 1,836 | FP8 量化 |
| DeepGemm autotune 变体 | 61.73 | 180 | tile 探索 |

#### 各分辨率被移除 Kernel（Baseline 独有，top 3）

| 分辨率 | Kernel | 总时间 (ms) | 次数 | 说明 |
|--------|--------|:---:|:---:|------|
| 256×256 | `nvjet_192x160` (coopB, TNT) | 106.80 | 270 | FFN w13 |
| | `nvjet_256x64` (TNT) | 79.65 | 1,080 | QKV |
| | `nvjet_128x288` (splitK, TNT) | 47.22 | 270 | FFN w2 |
| 512×512 | `nvjet_128x224` (coopA, TNT) | 442.09 | 1,350 | QKV+w13 |
| | `nvjet_384x96` (coopA, TNT) | 337.81 | 270 | FFN w2 |
| | `nvjet_80x128` (TNN) | 26.29 | 90 | TE GEMM |
| 1024×1024 | `nvjet_256x160` (coopA, TNT) | 2,826.14 | 1,620 | **全部 DiT GEMM** |
| | `nvjet_320x128` (coopB, TNT) | 104.72 | 90 | TE GEMM |
| | `nvjet_128x256` (coopA, TNN) | 96.02 | 90 | TE GEMM |

> **发现**：
> - 256×256：nvjet 为每种 GEMM shape 选择不同 tile config（3 种主要 nvjet 配置）
> - 512×512：nvjet 将 QKV+out 和 FFN w13 合并到同一 tile config（`128x224`，1350=1080+270）
> - 1024×1024：nvjet 将所有 DiT GEMM 合并到单一 tile config（`256x160`，1620=1080+270+270）
> - 这种"大矩阵用一个通用 tile"的策略在 1024 下 **不如 DeepGemm 的 per-shape autotuning 高效**

### 10.4 逐 Shape GEMM 加速比分析

#### 256×256 (M=768)

| GEMM 用途 | Shape (M×K→N) | BF16 nvjet (ms) | FP8 DeepGemm (ms) | Speedup | BF16 avg (μs) | FP8 avg (μs) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| FFN w13 (gate+up) | 768×3840→20480 | 106.80 | 61.26 | **1.74×** | 395.6 | 226.9 |
| QKV + output_proj | 768×3840→3840 | 79.65 | 51.54 | **1.55×** | 73.8 | 47.7 |
| FFN w2 (down) | 768×10240→3840 | 47.22 | 32.45 | **1.46×** | 174.9 | 120.2 |
| **DiT GEMM 合计** | — | **233.67** | **145.25** | **1.61×** | — | — |

#### 512×512 (M=3072)

| GEMM 用途 | Shape (M×K→N) | BF16 nvjet (ms) | FP8 DeepGemm (ms) | Speedup | BF16 avg (μs) | FP8 avg (μs) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| QKV+out + FFN w13 | 3072×3840→{3840,20480} | 442.09 | 366.30 | **1.21×** | — | — |
| FFN w2 (down) | 3072×10240→3840 | 337.81 | 101.43 | **3.33×** | 1,251.1 | 375.7 |
| **DiT GEMM 合计** | — | **779.90** | **467.73** | **1.67×** | — | — |

> 注：512×512 下 BF16 nvjet 将 QKV+out (1080) 和 FFN w13 (270) 合并到同一 tile config `128x224`，共 1350 次。FFN w2 单独使用 `384x96` config。DeepGemm 对 FFN w2 加速最大 (3.33×)，因其矩阵最大 (K=10240) 最能利用 FP8 计算密度优势。

#### 1024×1024 (M=12288)

| GEMM 用途 | Shape (M×K→N) | BF16 nvjet (ms) | FP8 DeepGemm (ms) | Speedup | BF16 avg (μs) | FP8 avg (μs) |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| FFN w13 (gate+up) | 12288×3840→20480 | — | 708.04 | — | — | 2,458.5 |
| QKV + output_proj | 12288×3840→3840 | — | 537.61 | — | — | 497.8 |
| FFN w2 (down) | 12288×10240→3840 | — | 353.61 | — | — | 1,309.7 |
| **DiT GEMM 合计** | — | **2,826.14** | **1,599.26** | **1.77×** | — | — |

> 注：1024×1024 下 BF16 nvjet 将所有 DiT GEMM (1620 次) 合并到单一 `256x160` config，平均 1,744.5μs/call。DeepGemm 对每种 shape 使用不同 tile config，实现 1.77× 整体加速。

#### 跨分辨率 GEMM 加速趋势

| 分辨率 | M | BF16 DiT GEMM (ms) | FP8 main GEMM (ms) | **GEMM Speedup** | 理论极限 |
|--------|:---:|:---:|:---:|:---:|:---:|
| 256×256 | 768 | 280.96 | 145.24 | **1.93×** | 2.0× |
| 512×512 | 3,072 | 858.51 | 467.73 | **1.84×** | 2.0× |
| 1024×1024 | 12,288 | 3,044.75 | 1,599.26 | **1.90×** | 2.0× |

> **发现**：
> - **GEMM 纯计算加速在所有分辨率下均接近 1.9×**，非常接近 FP8 的理论 2× TFLOPs 极限
> - 跨分辨率 GEMM 加速非常一致（1.84-1.93×），说明 DeepGemm 在各 M 值下都能有效利用 FP8 TFLOPs，所有测试分辨率下 GEMM 均为 compute-bound
> - Section 7 中关于 256×256 bandwidth-bound 的理论分析**未被实测验证**——DeepGemm 的 JIT tile autotuning 在小 M 值下仍能达到接近理论极限的加速比

### 10.5 FP8 Overhead 时间收支

#### 各分辨率 Overhead 汇总

| 分辨率 | FP8 Quantize (ms) | Autotune (ms) | **Overhead 合计** | 占 FP8 GPU 总时间 | GEMM 净节省 (ms) |
|--------|:---:|:---:|:---:|:---:|:---:|
| 256×256 | 10.57 | 9.32 | **19.89** | 7.6% | -88.42 |
| 512×512 | 28.27 | 28.94 | **57.22** | 7.7% | -333.56 |
| 1024×1024 | 96.77 | 61.73 | **158.51** | 5.5% | -1,286.98 |

#### 256×256 时间收支明细

| 组件 | 时间 (ms) | 说明 |
|------|:---:|------|
| BF16 nvjet DiT GEMM（被替换） | 280.96 | 基准参考 |
| → FP8 DeepGemm 主 GEMM | 145.24 | 快了 135.72ms ✅ |
| → FP8 DeepGemm autotune 变体 | 9.32 | 首 2 步 tile 探索 |
| → FP8 per-token 量化 | 10.57 | 1836 次量化 launch |
| → splitKreduce 消除 | -3.11 | BF16 splitK 不再需要 |
| **FP8 GEMM 总计** | **162.02** | — |
| **净节省** | **-118.94 ms** | **GEMM 路径 -42.3%** |

#### 1024×1024 时间收支明细

| 组件 | 时间 (ms) | 说明 |
|------|:---:|------|
| BF16 nvjet DiT GEMM（被替换） | 3,044.75 | 基准参考 |
| → FP8 DeepGemm 主 GEMM | 1,599.26 | 快了 1,445.49ms ✅ |
| → FP8 DeepGemm autotune 变体 | 61.73 | tile 探索 |
| → FP8 per-token 量化 | 96.77 | 1836 次量化 launch |
| **FP8 GEMM 总计** | **1,757.76** | — |
| **净节省** | **-1,286.99 ms** | **GEMM 路径 -42.3%** |

> **与 Section 3.5 的对比**：
> - Section 3.5 分析的 nsys 数据**包含 warmup**，transpose 开销 139.8ms 是 JIT 编译时的 scale transpose
> - **去除 warmup 后，transpose 开销消失**，说明 DeepGemm 在 JIT compile 阶段完成 scale 预处理，推理时无额外 transpose
> - **推理阶段**的 FP8 真实开销仅为量化 + autotune

### 10.6 非 GEMM Kernel 跨分辨率对比

非 GEMM 类 kernel 在 BF16 和 FP8 之间几乎完全一致，验证了 FP8 仅影响 GEMM 路径：

| Category | 256 BL | 256 FP8 | 512 BL | 512 FP8 | 1024 BL | 1024 FP8 | 说明 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|------|
| FlashAttention | 14.1 | 14.2 | 54.1 | 54.1 | **625.0** | **624.6** | O(n²) 增长，1024 下成为 #2 瓶颈 |
| Conv (VAE) | 13.9 | 13.9 | 48.6 | 47.6 | 170.5 | 171.8 | 线性增长 |
| Elementwise | 21.9 | 21.9 | 41.3 | 41.4 | 138.9 | 139.9 | 线性增长 |
| RMSNorm | 4.2 | 4.2 | 10.7 | 10.6 | 38.4 | 39.3 | 线性增长 |
| SiLU Gate | 2.4 | 2.3 | 7.2 | 7.4 | 27.1 | 27.3 | 线性增长 |
| QKNorm | 1.6 | 1.4 | 4.2 | 4.1 | 14.9 | 14.9 | 线性增长 |
| RoPE | 1.2 | 1.2 | 2.9 | 2.9 | 12.9 | 12.9 | 线性增长 |

> **瓶颈转移**：FP8 将 GEMM 时间减半后，1024×1024 下 FlashAttention (625ms) 从微不足道变为 FP8 GPU 时间的 **21.7%**，成为下一个优化目标。

### 10.7 关键结论

1. **GEMM 纯计算加速跨分辨率一致：~1.9×**，接近 FP8 理论 2× 极限，所有测试分辨率均为 compute-bound
2. **FP8 新增 overhead 仅 5.5-7.7%**，远小于 GEMM 节省量
3. **256×256 E2E 未加速的根因是 host-side dispatch**，GPU kernel 本身已快 25%（Section 11 CUDA Graph 已解决）
4. **去除 warmup 后 transpose 开销消失**，推翻了 Section 3.5 的结论——DeepGemm 在推理阶段无额外 transpose，transpose 仅发生在 JIT warmup 阶段
5. **1024×1024 FP8 后 FlashAttention 成为 #2 瓶颈**（625ms, 21.7%），是后续优化方向
6. **量化 overhead 随分辨率线性增长**（10→28→97ms），占比从 4.1% 降至 3.4%，说明 FP8 越大分辨率越划算

---

## 11. 优化 #3：CUDA Graph 消除 Host Dispatch 开销

> Section 10 的 nsys no-warmup 分析揭示了一个关键矛盾：**FP8 GPU kernel 在所有分辨率下都快了 25-30%，但 256×256 E2E 反而慢了 34%**（443ms vs 330ms）。根因是 host-side 开销——FP8 增加了量化 kernel（+1,836 次 launch），加上 DeepGemm 的 Python dispatch 路径比 cuBLAS 的 C++ 路径更重，导致 CPU 侧 kernel launch latency 成为瓶颈。在 256×256 这种 GPU kernel 执行极快（~22ms/step）的场景下，CPU 侧的 ~18ms dispatch 开销占据了每步时间的近一半。

### 11.1 问题根因

**nsys timeline 观察**：
- **BF16 场景**：CPU kernel launch 时间短，GPU kernel 时间长，GPU kernel 之间几乎没有 bubble
- **FP8 场景**：CPU kernel launch 时间比 BF16 长 2× 以上（量化 kernel 增加了 launch 数量），但 FP8 GPU kernel 计算时间只有 BF16 的一半，导致 GPU kernel 之间出现大量 bubble——GPU 在等 CPU launch 下一个 kernel

**本质**：FP8 把瓶颈从 GPU compute 转移到了 CPU dispatch。GPU 算得越快，CPU launch 的开销占比越大。

### 11.2 方案设计

**方案**：将 `dit.forward()` 的完整 kernel 序列 capture 到一个 CUDA Graph 中，replay 时通过单次 `cudaGraphLaunch()` 提交所有 kernel，消除逐 kernel 的 CPU launch 延迟。

**Graph 边界**（与 LLM 侧 `CudaGraphRunner` 一致）：

```
Graph 内：  dit.forward()（完整模型 forward，包含 FP8 quantize + GEMM + attention + norm + FFN）
Graph 外：  scheduler.step()、timestep 展开、CFG 合并、profiling
```

**跨请求缓存**：Graph 在 `DenoisingStage` 实例上按 latent shape 缓存。首次请求 capture（2× warmup + 1× capture = 3× eager forward 开销），后续同分辨率请求直接 replay。由于 ZImage tokenizer 将所有 prompt pad 到 512 tokens（`padding="max_length", max_length=512`），不同 prompt 在 text encoder 输出后 shape 完全一致，因此**同分辨率不同 prompt 的请求可以直接复用已 capture 的 graph**，仅需将新的 `encoder_hidden_states` 数据 copy 到持久 buffer 中。

**实现细节**：
- `DiffusionCudaGraphRunner` 类：管理 capture/replay/buffer 生命周期
- 所有输入（timestep、latents、encoder_hidden_states、freqs_cis、guidance）都存储在持久 buffer 中，通过 `.copy_()` 更新数据
- 使用 `torch.cuda.graph_pool_handle()` 共享内存池，避免跨分辨率 graph 的内存碎片化
- Runtime assert 确保 attention backend 不依赖 `current_timestep` 做 kernel 路径选择（FlashAttention 安全，STA/VSA/SVG2 不兼容）

### 11.3 多分辨率 E2E 实测结果

> **测试条件**：1×H20, `sglang generate --warmup`（graph 在 warmup 请求时 capture，实际请求全部 replay）

#### 完整三步优化路线对比

| 分辨率 | BF16 TE (Baseline) | + FP8 DeepGemm | + CUDA Graph | vs Baseline |
|--------|:---:|:---:|:---:|:---:|
| **256×256** | **330ms** | 443ms (+34% ❌) | **263ms** | **1.26× ✅** |
| **512×512** | **923ms** | 720ms (-22%) | **693ms** | **1.33× ✅** |
| **1024×1024** | **3,607ms** | 2,621ms (-27%) | **2,604ms** | **1.38× ✅** |

#### Stage 级分解

**256×256**：

| Stage | BF16 TE | FP8 DeepGemm | FP8 + CUDA Graph | CUDA Graph 收益 |
|-------|:---:|:---:|:---:|:---:|
| TextEncoding | 50ms | 62ms | 51ms | — |
| **Denoising** | **268ms** | **371ms** | **204ms** | **-167ms (-45%)** |
| Decoding | 10ms | 6ms | 6ms | — |
| **E2E** | **330ms** | **443ms** | **263ms** | **-180ms (-41%)** |

**512×512**：

| Stage | BF16 TE | FP8 DeepGemm | FP8 + CUDA Graph | CUDA Graph 收益 |
|-------|:---:|:---:|:---:|:---:|
| TextEncoding | 51ms | 57ms | 52ms | — |
| **Denoising** | **838ms** | **655ms** | **632ms** | **-23ms (-4%)** |
| Decoding | 31ms | 6ms | 7ms | — |
| **E2E** | **923ms** | **720ms** | **693ms** | **-27ms (-4%)** |

**1024×1024**：

| Stage | BF16 TE | FP8 DeepGemm | FP8 + CUDA Graph | CUDA Graph 收益 |
|-------|:---:|:---:|:---:|:---:|
| TextEncoding | 51ms | 51ms | 61ms | — |
| **Denoising** | **3,412ms** | **2,561ms** | **2,535ms** | **-26ms (-1%)** |
| Decoding | 142ms | 7ms | 6ms | — |
| **E2E** | **3,607ms** | **2,621ms** | **2,604ms** | **-17ms (-1%)** |

### 11.4 Per-Step 分析

#### 256×256 — CUDA Graph 效果最显著

| Step | BF16 TE (ms) | FP8 DeepGemm (ms) | FP8 + CUDA Graph (ms) |
|:---:|:---:|:---:|:---:|
| 0 | 21.9 | 43.1 | 22.9 |
| 1 | 20.6 | 40.8 | 22.2 |
| 2 | 24.7 | 40.6 | 22.2 |
| 3 | 33.2 | 40.4 | 22.2 |
| 4 | 32.8 | 40.9 | 22.2 |
| 5 | 32.8 | 41.1 | 22.2 |
| 6 | 32.9 | 40.2 | 22.2 |
| 7 | 33.0 | 40.3 | 22.2 |
| 8 | 32.8 | 40.6 | 22.2 |
| **Avg (step 1-8)** | **30.4** | **40.6** | **22.2** |

> **关键观察**：
> - FP8 eager：40.6ms/step — 比 BF16 的 30.4ms 慢 34%（GPU 快了但 host dispatch 拖后腿）
> - FP8 + CUDA Graph：22.2ms/step — 比 BF16 快 27%，比 FP8 eager 快 45%
> - CUDA Graph 消除了 ~18ms/step 的 host dispatch 开销

#### 1024×1024 — CUDA Graph 效果微弱

| Step | BF16 TE (ms) | FP8 DeepGemm (ms) | FP8 + CUDA Graph (ms) |
|:---:|:---:|:---:|:---:|
| 0 | 27.7 | 283.9 | 281.8 |
| 1-8 avg | 422.7 | 284.1 | 281.0 |

> 1024×1024 下 GPU kernel 执行时间 ~280ms/step，host dispatch 开销相对微不足道（~3ms/step），CUDA Graph 仅节省 ~3ms/step。

### 11.5 CUDA Graph 收益分析

| 分辨率 | FP8 Eager Per-Step | FP8+CG Per-Step | Host Dispatch 消除 | 收益占比 |
|--------|:---:|:---:|:---:|:---:|
| **256×256** | 40.6ms | **22.2ms** | **-18.4ms/step (-45%)** | **显著** |
| **512×512** | 72.4ms | **69.6ms** | -2.8ms/step (-4%) | 中等 |
| **1024×1024** | 284.1ms | **281.0ms** | -3.1ms/step (-1%) | 微弱 |

**收益与分辨率的关系**：CUDA Graph 消除的是**固定的** CPU-side dispatch 开销（~3-18ms/step，与 kernel 数量成正比但与矩阵大小无关）。当 GPU kernel 执行时间短（256×256: ~22ms）时，这个固定开销占比大（45%），收益显著；当 GPU kernel 执行时间长（1024×1024: ~280ms）时，占比可忽略（1%）。

**VRAM 影响**：CUDA Graph 额外占用约 33-86MB 用于 graph 内部 tensor 缓存和 pool 管理，对总 VRAM（14-19GB）影响可忽略。

| 分辨率 | FP8 Eager Peak VRAM | FP8+CG Peak VRAM | Delta |
|--------|:---:|:---:|:---:|
| 256×256 | 14,448MB | 14,481MB | +33MB (+0.2%) |
| 512×512 | 15,365MB | 15,399MB | +34MB (+0.2%) |
| 1024×1024 | 19,020MB | 19,056MB | +36MB (+0.2%) |

### 11.6 关键发现

1. **CUDA Graph 解决了 256×256 FP8 反而更慢的核心问题**：443ms → 263ms，E2E 从 0.75× 翻转为 1.26×
2. **收益随分辨率递减**：256×256 下 -45% per-step（host-bound），1024×1024 下仅 -1%（compute-bound）
3. **Graph 跨请求缓存有效**：同分辨率不同 prompt 直接 replay，无需重新 capture（ZImage tokenizer pad to 512 保证 shape 一致）
4. **与 FP8 协同效应**：FP8 让 GPU kernel 变快 → host dispatch 占比升高 → CUDA Graph 消除 dispatch 的收益更大。FP8 + CUDA Graph 是互补组合
5. **VRAM 开销可忽略**：+33~36MB（<0.2%），不影响实际部署
6. **与 Cache-DiT 当前互斥**：CUDA Graph 要求每步 kernel 序列一致，而 Cache-DiT 的缓存命中步会跳过大部分 layer，改变 kernel 序列。联合优化需要为 full/cached 两种模式各 capture 一个 graph

---

## 附录 A：与 Yikai 1024×1024 Profile 的对比

**Source**: Z-Image Sgl-Diffusion Profile | **Hardware**: H100 | **Resolution**: 1024×1024 (~12K tokens)

| Tag | **256×256 (H20)** | **1024×1024 (H100)** | 差异根因 |
|-----|:---:|:---:|------|
| feedforward:gemm | **56.96%** | 34.32% | H100 GEMM 6.7× 更快 → 占比降 |
| usp_attention | 3.38% | **17.11%** | O(n²) + 更长序列 |
| qkv_projection:gemm | **22.26%** | 13.15% | 同 feedforward |
| rms_norm | 1.83% | **11.44%** | bandwidth-bound，H100 GEMM 缩后占比升 |
| output_projection:gemm | 7.39% | 4.47% | |

**差异归因**：

| 因素 | GEMM % | RMSNorm % | Attention % |
|------|:---:|:---:|:---:|
| 硬件 (H20→H100) | ↓↓ 计算强→GEMM缩 | ↑↑ 带宽近似→占比升 | ↑ |
| 序列长度 (768→12K) | ↓ O(n²)抢占 | ↓ 同理 | ↑↑ O(n²)增长 |
| **净效应** | **↓↓↓** (87%→52%) | **↑** (1.8%→11.4%) | **↑↑↑** (3.4%→17.1%) |

> **结论**：两个 profile 差异主要源于**硬件差异**（H100 计算力 6.7×）和**序列长度**（O(n²) attention），需在同一 GPU 上对比才能隔离纯序列长度效应。

---

## 附录 B：复现命令

```bash
export MODEL=/mnt/geminihzceph/rhyshen/models/Z-Image-Turbo
export PROMPT="A beautiful sunset over the ocean with golden clouds"

# FP32 baseline
sglang generate --model-path $MODEL --prompt "$PROMPT" \
    --height 256 --width 256 --warmup --save-output \
    --perf-dump-path ./baseline_fp32.json

# BF16 TextEncoder
sglang generate --model-path $MODEL --prompt "$PROMPT" \
    --height 256 --width 256 --warmup --save-output \
    --text-encoder-precisions bf16 \
    --perf-dump-path ./baseline_bf16.json

# FP8 DeepGemm
sglang generate --model-path $MODEL \
    --transformer-weights-path $MODEL/transformer-FP8-block128 \
    --text-encoder-precisions bf16 \
    --prompt "$PROMPT" --height 256 --width 256 --warmup --save-output \
    --perf-dump-path ./baseline_fp8_deepgemm.json

# FP8 CUTLASS (disable DeepGemm)
SGLANG_ENABLE_JIT_DEEPGEMM=0 sglang generate --model-path $MODEL \
    --transformer--weights-path $MODEL/transformer-FP8-block128 \
    --text-encoder-precisions bf16 \
    --prompt "$PROMPT" --height 256 --width 256 --warmup --save-output \
    --perf-dump-path ./baseline_fp8_cutlass.json

# FP8 DeepGemm + CUDA Graph
sglang generate --model-path $MODEL \
    --transformer--weights-path $MODEL/transformer-FP8-block128 \
    --text-encoder-precisions bf16 \
    --enable-diffusion-cuda-graph \
    --prompt "$PROMPT" --height 256 --width 256 --warmup --save-output \
    --perf-dump-path ./baseline_fp8_deepgemm_cuda_graph.json

# Cache-DiT 调优
SGLANG_CACHE_DIT_ENABLED=true SGLANG_CACHE_DIT_WARMUP=2 \
SGLANG_CACHE_DIT_RDT=0.50 SGLANG_CACHE_DIT_MC=6 \
sglang generate --model-path $MODEL --prompt "$PROMPT" \
    --text-encoder-precisions bf16 \
    --height 256 --width 256 --warmup --save-output

# Profile (kernel-level analysis)
sglang generate --model-path $MODEL --prompt "$PROMPT" \
    --height 256 --width 256 --warmup \
    --profile --profile-all-stages --save-output
```

---

*Generated by analyze_profile.py + generate_multi_res_charts.py + manual analysis*
*Charts: analysis_report/*.png*
*Benchmark data: zimage_bench/{256_256,512_512,1024_1024}/*.json*
*CUDA Graph implementation: python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py*
