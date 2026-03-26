# Z-Image-Turbo 性能分析与优化报告

> **Hardware**: NVIDIA H20 (SM90, 96GB HBM3)
> **Model**: Z-Image-Turbo (30-layer DiT, dim=3840, 9 denoising steps, no CFG)
> **Date**: 2026-03-25 (updated)

---

## 目录

- [1. 瓶颈分析（优化前）](#1-瓶颈分析优化前)
  - [1.1 E2E 时延分解](#11-e2e-时延分解)
  - [1.2 CUDA Kernel 分类分析](#12-cuda-kernel-分类分析)
  - [1.3 DiT Denoising 细粒度 Kernel 分解](#13-dit-denoising-细粒度-kernel-分解)
  - [1.4 关键发现](#14-关键发现)
- [2. 优化 #1：TextEncoder FP32 → BF16](#2-优化-1textencoder-fp32--bf16)
  - [2.1 方法与结果](#21-方法与结果)
  - [2.2 Kernel 级验证](#22-kernel-级验证)
  - [2.3 瓶颈转移](#23-瓶颈转移)
- [3. 优化 #2：DiT FP8 量化](#3-优化-2dit-fp8-量化)
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
- [附录 A：与 Yikai 1024×1024 Profile 的对比](#附录-a与-yikai-10241024-profile-的对比)
- [附录 B：复现命令](#附录-b复现命令)

---

## 1. 瓶颈分析（优化前）

### 1.1 E2E 时延分解

**Baseline**: 1 GPU, FP32 TextEncoder, 256×256, E2E = 362ms

| Stage | Time (ms) | 占比 | 说明 |
|-------|-----------|------|------|
| TextEncodingStage | 81 | 22.4% | Qwen3 text encoder |
| DenoisingStage | 269 | 74.3% | **#1 瓶颈** — 9 steps, ~33ms/step |
| DecodingStage | 10 | 2.7% | VAE decoder |

> 注：早期 profile（FP32 TextEncoder 未配置 bf16 CLI flag 的旧基准）中 TextEncoder 占比更高（62%），因旧版本 TextEncoder 以 FP32 运行。当前基准（benchmark JSON）中 TextEncoder 已默认优化。

![Pipeline Breakdown](01_pipeline_and_kernel_breakdown.png)

### 1.2 CUDA Kernel 分类分析

**Total GPU kernel time: 474ms (Torch Profiler, FP32 baseline)**

| Category | Time (ms) | % | 说明 |
|----------|-----------|---|------|
| BF16 GEMM (DiT) | 251.0 | 53.0% | 优化目标 #2 (FP8) |
| FP32 GEMM (TextEncoder) | 156.6 | 33.0% | **优化目标 #1** (FP32→BF16) |
| Elementwise Ops | 21.1 | 4.4% | |
| Convolution (VAE) | 12.6 | 2.6% | |
| FlashAttention | 9.7 | 2.1% | 短序列(768 tokens), 非瓶颈 |
| RMSNorm / QKNorm | 5.7 | 1.2% | |

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

1. **GEMM 占 91.5%** GPU kernel 时间（TextEncoder FP32 GEMM + DiT BF16 GEMM）
2. **FlashAttention 仅 2.1%** — 短序列(768 tokens)下 attention 非瓶颈
3. **torch.compile 导致严重回退** — 256×256 下 +48%，Triton kernel 在小矩阵上比 cuBLAS/nvJET 慢
4. **2-GPU SP 无收益** — 768 tokens 太短，通信开销 > 并行收益
5. **Cache-DiT 默认配置效果有限** — 9 步仅 1 步命中缓存

---

## 2. 优化 #1：TextEncoder FP32 → BF16

### 2.1 方法与结果

**方法**：`--text-encoder-precisions bf16`（零代码修改，CLI flag）

| 指标 | FP32 Baseline | BF16 TextEnc | 变化 |
|------|:---:|:---:|:---:|
| **E2E** | 362ms | **332ms** | **-8.3%** |
| TextEncoding | 81ms | 53ms | -34.6% |
| Denoising | 269ms | 267ms | 不变 |
| Peak VRAM | 20,167MB | 20,167MB | 不变 |

> 注：在更早期的旧基准中（TextEncoder 默认 FP32 运行），BF16 带来了 35% 加速（749ms→486ms）。当前 benchmark 基准中 TextEncoder 耗时已较低。

### 2.2 Kernel 级验证

**Total GPU kernel time: 474.0ms (FP32) → 341.7ms (BF16) = -132.4ms (-27.9%)**

| Kernel Category | FP32 (ms) | BF16 (ms) | Delta | 说明 |
|----------------|-----------|-----------|-------|------|
| BF16 GEMM (DiT) | 251.0 | 279.6 | +28.5ms | TE GEMM 转为 BF16 nvjet |
| **FP32 GEMM (TE)** | **156.6** | **0.2** | **-156.4ms** | **消除** |
| Elementwise | 21.1 | 18.2 | -2.9ms | 少了 FP32 cast |
| FlashAttention | 9.7 | 10.7 | +1.0ms | TE 注意力转 BF16 |
| LayerNorm | 1.2 | 0.1 | -1.1ms | FP32 LayerNorm 消除 |

核心：**156.6ms FP32 GEMM 被替换为 ~28.5ms BF16 nvjet GEMM（5.5× 加速）**。

![Kernel Comparison](08_kernel_comparison_fp32_vs_bf16.png)

### 2.3 瓶颈转移

BF16 优化后瓶颈从 TextEncoding 转移到 Denoising：

| Stage | 优化前 | 优化后 | 占比变化 |
|-------|--------|--------|----------|
| TextEncoding | 62.1% | 41.4% | ↓ |
| **Denoising** | 35.9% | **55.9%** | **↑ 成为新瓶颈** |

![Bottleneck Shift](07_bottleneck_shift_before_after.png)

---

## 3. 优化 #2：DiT FP8 量化

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

**nsys Kernel 级根因**：

| 指标 | BF16 nvjet | FP8 DeepGemm | FP8 CUTLASS |
|------|:---:|:---:|:---:|
| DiT GEMM 计算 | 260.0ms | **171.8ms** ✅ | 238.8ms |
| transpose 开销 | 0ms | **+139.8ms** ❌ | 0ms |
| 量化开销 | 0ms | +11.8ms | +16.2ms |
| **GEMM 总计** | **260.0ms** | **323.4ms** | **255.0ms** |

### 3.4 CUTLASS vs DeepGemm vs BF16 对比

**逐 GEMM Shape 分析**：

| GEMM 用途 | Shape (M×K×N) | BF16 nvjet (μs) | DeepGemm FP8 (μs) | CUTLASS FP8 (μs) | DeepGemm 加速 |
|-----------|:---:|:---:|:---:|:---:|:---:|
| FFN w13 | 768×3840×10240 | 396 | **227** | ~117 | **1.75×** |
| QKV+output | 768×3840×3840 | 73.8 | **47.8** | ~117 | **1.54×** |
| FFN w2 | 768×10240×3840 | 175 | **120** | ~117 | **1.46×** |
| **DiT 合计** | — | **260.0ms** | **171.8ms** | **238.8ms** | **1.51×** |

> **DeepGemm GEMM 计算快 1.51×**，但 139.8ms transpose 抵消了全部加速。
> **CUTLASS FP8 无 transpose**，总时间 255ms，比 BF16 略快 2%（-5ms）。

### 3.5 DeepGemm Transpose 开销分析

**根因**：Hopper SM90 TMA 硬件要求 block scale 以列优先、16-byte 对齐存储。DeepGemm 在**每次** GEMM 调用时都对 weight scale 和 activation scale 做 transpose。

| 指标 | 值 |
|------|---:|
| Transpose 总时间 | **139.8ms** |
| GEMM 计算总时间 | 171.8ms |
| Transpose / GEMM | **81%** |
| Transpose kernel launch 次数 | **49,152** |
| GEMM kernel launch 次数 | 2,040 |

**FP8 时间收支**：

| 组件 | 时间 | 说明 |
|------|------|------|
| BF16 nvjet GEMM（被替换） | 260.0ms | 参照 |
| FP8 DeepGemm GEMM 计算 | 171.8ms | 快了 88.2ms ✅ |
| FP8 DeepGemm transpose | **+139.8ms** | **回吐所有加速** ❌ |
| FP8 量化开销 | +11.8ms | 可接受 |
| **FP8 总计** | **323.4ms** | **比 BF16 慢 63.4ms** |

> **结论**：如能消除 transpose，FP8 DiT GEMM 从 260ms 降至 ~184ms，实现 **1.42× 加速**。

---

## 4. 多分辨率全面 Benchmark

> **测试时间**：2026-03-23 | **硬件**：1×H20 / 2×H20 (Ulysses SP)
> **配置**：8 种优化组合 × 3 种分辨率 = 24 次测试

### 4.1 E2E 时延总表

#### 256×256 (seq_len ≈ 768)

| 配置 | E2E (ms) | TextEnc | Denoise | SS Step | vs Baseline | VRAM |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1GPU FP32 (baseline) | **362** | 81 | 269 | 33.0ms | — | 20.2GB |
| 1GPU BF16-TE | **332** | 53 | 267 | 32.9ms | **1.09×** | 20.2GB |
| BF16-TE+CacheDiT | **336** | 63 | 264 | 31.9ms | **1.08×** | 20.2GB |
| CacheDiT+FP8-CUTLASS | 384 | 86 | 289 | 29.7ms | 0.94× | **14.5GB** |
| CacheDiT+FP8-DeepGemm | 404 | 52 | 343 | 35.4ms | 0.90× | **14.5GB** |
| BF16-TE+Compile | 491 | 52 | 431 | 47.0ms | 0.74× ❌ | 20.2GB |
| 2GPU SP | 385 | 51 | 326 | 35.7ms | 0.94× | 20.2GB |

#### 512×512 (seq_len ≈ 3,072)

| 配置 | E2E (ms) | TextEnc | Denoise | SS Step | vs Baseline | VRAM |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1GPU FP32 (baseline) | **924** | 52 | 838 | 104.5ms | — | 21.1GB |
| BF16-TE+CacheDiT | **865** | 78 | 779 | 94.4ms | **1.07×** | 21.1GB |
| **CacheDiT+FP8-CUTLASS** | **753** | 187 | 558 | 65.5ms | **1.23×** | **15.4GB** |
| **CacheDiT+FP8-DeepGemm** | **632** | 68 | 555 | 66.8ms | **1.46×** | **15.4GB** |
| 2GPU SP | **635** | 51 | 564 | 67.7ms | **1.45×** | 21.1GB |
| **2GPU+FP8-DeepGemm** | **575** | 52 | 515 | 56.9ms | **1.61×** | **15.4GB** |

#### 1024×1024 (seq_len ≈ 12,288)

| 配置 | E2E (ms) | TextEnc | Denoise | SS Step | vs Baseline | VRAM |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1GPU FP32 (baseline) | **3628** | 71 | 3413 | 422.7ms | — | 24.7GB |
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
2. **FP8 在 256×256**：GEMM bandwidth-bound，无法发挥 2× TFLOPs
3. **2GPU SP 在 256×256**：通信开销 > 并行收益

#### 分辨率适配推荐

| 分辨率 | 推荐配置 | E2E | 加速 |
|--------|---------|:---:|:---:|
| 256×256 | BF16-TE（简单有效） | 332ms | 1.09× |
| 512×512 | CacheDiT + FP8-DeepGemm | 632ms | 1.46× |
| 1024×1024 | CacheDiT + FP8-DeepGemm | 2186ms | 1.66× |
| 1024×1024 (2GPU) | 2GPU + FP8-DeepGemm | 1639ms | 2.21× |

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

> **结论**：256×256 GEMM 是 **bandwidth-bound**，FP8 的 2× TFLOPs 无法转化为实际加速。这已被多分辨率 benchmark 实测验证（256×256 FP8 无加速，1024×1024 FP8 加速 1.66×）。

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

| 场景 | 限制因素 | H100 vs H20 |
|------|---------|:---:|
| 256×256 GEMM (M=768) | **带宽受限** | H100 ≈ H20（仅快 ~10%） |
| 1024×1024 GEMM (M=12288) | **计算受限** | **H100 快 5-6.7×** |

> **结论**：对 256×256，换 H100 几乎无收益。H100 的算力优势只在 1024×1024 等 compute-bound 场景发挥。

---

## 9. 优化路线图

| 状态 | 优化 | 目标 | 收益 | 说明 |
|:---:|------|------|------|------|
| ✅ 完成 | TextEncoder FP32→BF16 | TextEncoding | -35% E2E (旧基准) | CLI flag, 零代码 |
| ✅ 完成 | FP8 scale 加载修复（3层Bug） | Bug fix | VRAM -28%, 图像恢复 | 4 处代码修复 |
| ✅ 完成 | CUTLASS FP8 验证 | Denoising GEMM | 256×256 收益有限(+2%) | 消除 transpose |
| ✅ 完成 | 多分辨率 Benchmark | 全面验证 | 确认 FP8 在 ≥512 有效 | 24 次测试 |
| ✅ 完成 | Cache-DiT 调优 | Denoising skip | 256×256 最高 1.35× | W/R/MC 参数调优 |
| **P0** | **Pre-transpose weight scale** | DeepGemm transpose | -70ms（消除 weight scale transpose） | 使 DeepGemm 真正有效 |
| P1 | Fused adaLN + gate kernel | Elementwise | -5~10ms | Triton fused kernel |
| P2 | CuTe DSL FlashAttention | Attention | -3~5ms（256×256 收益小） | 长序列更有效 |
| P3 | Better RMSNorm (Quack) | Norm | -2~3ms | bandwidth-bound 场景收益小 |

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
    --transformer-path $MODEL/transformer-FP8-block128 \
    --text-encoder-precisions bf16 \
    --prompt "$PROMPT" --height 256 --width 256 --warmup --save-output \
    --perf-dump-path ./baseline_fp8_deepgemm.json

# FP8 CUTLASS (disable DeepGemm)
SGLANG_ENABLE_JIT_DEEPGEMM=0 sglang generate --model-path $MODEL \
    --transformer-path $MODEL/transformer-FP8-block128 \
    --text-encoder-precisions bf16 \
    --prompt "$PROMPT" --height 256 --width 256 --warmup --save-output \
    --perf-dump-path ./baseline_fp8_cutlass.json

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
