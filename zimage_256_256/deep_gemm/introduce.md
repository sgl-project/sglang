# DeepGEMM 技术分析报告

> 基于 SGLang 源码分析 + DeepSeek 官方 repo，针对 H20 GPU + ZImage-Turbo 推理场景

---

## 1. 什么是 DeepGEMM？

DeepGEMM 是 DeepSeek AI 开源的高性能 FP8 GEMM 库，是 DeepSeek-V3/R1 训练推理基础设施的核心组件之一。它只有约 300 行核心 CUDA 代码，却在特定矩阵形状下比高度调优的 CUTLASS 3.6 快 **2.7×**。

**GitHub**：https://github.com/deepseek-ai/DeepGEMM

---

## 2. 核心技术优势

### 2.1 细粒度块量化（Fine-Grained Block Scaling）

标准 FP8 量化通常用 per-tensor 或 per-channel scaling，精度损失大：

```
per-tensor: 整个矩阵共用一个 scale → 大值和小值争夺动态范围 → 精度差
```

DeepGEMM 采用 **块量化（Block Scaling）**：
- **Activation（A 矩阵）**：每 **1×128** 块共用一个 scale（每个 token 的每 128 个 channel 一个 scale）
- **Weight（B 矩阵）**：每 **128×128** 块共用一个 scale

这样每个局部区域有自己的动态范围，FP8 的精度损失大幅降低，数值表现接近 BF16。

### 2.2 两级累加（Two-Level Accumulation）

```
FP8 TensorCore 计算：主 GEMM（速度快，精度有限）
         ↓
每个 128×128 块完成后，用 CUDA Core 乘以对应的 FP32 scale 并累加
         ↓
最终结果精度等效于更高精度，同时享受 FP8 的带宽优势
```

这是 DeepGEMM 能在 FP8 精度与 BF16 数值质量之间取得平衡的根本原因。

### 2.3 充分利用 Hopper 架构硬件特性

| Hopper 特性 | DeepGEMM 的使用方式 | 收益 |
|---|---|---|
| **FP8 Tensor Core (WGMMA)** | Warp-group 级矩阵运算，直接在每个 128×128 tile 上应用 scale | FP8 峰值算力比 BF16 高 2× |
| **TMA（Tensor Memory Accelerator）** | 异步加载 scale 数据，与 WGMMA 计算重叠 | 消除 scale 加载的 latency |
| **Persistent Kernel** | 单个 kernel 处理完整 GEMM，无论 M 多大 | 避免 kernel launch overhead，GPU 保持满负载 |
| **Warp Specialization** | 不同 warp 专门负责数据加载 vs 计算 | 进一步提升 pipeline 利用率 |

> **结论**：DeepGEMM 是专门为 Hopper 架构设计的，完整利用了 SM90 的 FP8 算力路径。

### 2.4 JIT 编译（NVRTC）

不需要预先编译安装，运行时通过 NVRTC 按需编译 CUDA kernel，支持针对当前 GPU 的 SM 型号自动调优。SGLang 的实现中还预编译了 3072 种常见 M 值的 kernel，做到了**热启动零开销**。

---

## 3. DeepGEMM vs CUTLASS vs Triton（SGLang 中的对比）

SGLang 的 FP8 GEMM 有三条代码路径，dispatch 逻辑如下：

```
输出 dtype == BF16？
  └─ SM90+（Hopper）？
       └─ DeepGEMM 已安装？ → 使用 DeepGEMM          ← 最优路径
       └─ 否 → CUTLASS FP8
  └─ SM85-89（Ampere）→ CUTLASS / Triton
输出 dtype 为 FP32/FP16 → Triton
```

| 维度 | DeepGEMM | CUTLASS FP8 | Triton |
|---|---|---|---|
| **峰值算力利用率** | 最高（专为 Hopper 设计） | 高 | 中 |
| **块量化支持** | 原生支持 | 有限 | 有限 |
| **MoE grouped GEMM** | 原生支持 | 不支持 | 不支持 |
| **代码复杂度** | ~300 行 | 极复杂 | 中等 |
| **对齐要求** | 严格（N%64, K%128） | 较宽松 | 较宽松 |
| **适用硬件** | H100/H800/H20（SM90+） | Ampere+ | 通用 |

---

## 4. 对齐约束与限制

DeepGEMM 对矩阵维度有严格要求：

| 约束 | 要求 | 原因 |
|---|---|---|
| N（输出维度） | N % 64 == 0 | WGMMA warp-group tile 宽度 |
| K（内积维度） | K % 128 == 0 | 块量化 scale 粒度 |
| 输出 dtype | 必须为 BF16 | 不支持 FP16/FP32 输出 |
| Scale 内存布局 | 列主序（column-major），16 字节对齐 | TMA 硬件寻址要求 |

**不满足约束时**，SGLang 自动回退到 Triton kernel，不会报错。

---

## 5. 在 H20 上的实际表现（ZImage 256×256 场景）

根据 SGLang 源码分析和之前的 benchmark 结果：

| 配置 | 端到端时延 | 备注 |
|---|---|---|
| BF16 baseline | ~485ms | 参考基线 |
| FP8 + CUTLASS | 与 BF16 接近或略快 | 中等序列长度下收益有限 |
| FP8 + DeepGEMM | ~548ms（**比 BF16 慢**） | 存在 scale transpose 开销 |

### 为什么 256×256 场景下 DeepGEMM 没有收益？

原因分析：

**1. scale 转置开销（~70ms）**

DeepGEMM 要求 scale 为列主序，但模型权重加载后 scale 是行主序。当前实现在**每次 forward 时临时转置**：

```python
# 每次调用都做一次转置 → ~70ms overhead
weight_scale_T = weight_scale.t().contiguous()
```

这 70ms 基本抵消了 FP8 计算节省的所有时间。

**2. 序列长度太短，计算量不够大**

256×256 图片 → latent token 数较少 → 每个 GEMM 的 M 维度小 → GPU 利用率低 → FP8 相对 BF16 的计算优势不明显。

### 1024×1024 场景的预期

- Token 数增加 16× → GEMM 计算量增加 16×
- 但 scale transpose overhead 是固定的（仍然 ~70ms）
- FP8 节省的计算时间远超 overhead → **预期有明显正收益**

---

## 6. 已识别的优化机会

### 优化项：预转置 Weight Scale（高优先级）

**问题**：每次 forward 都做 scale 转置，约 70ms 开销。

**方案**：在模型加载阶段（一次性）预先将 weight scale 转置为列主序，推理时直接使用。

**预期收益**：节省 ~70ms/forward，对 256×256 场景约提升 **13%**，对大尺寸场景效果更显著。

**相关文件**：
- `python/sglang/multimodal_gen/runtime/layers/linear.py` — scale transpose 发生位置
- `python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py` — 权重加载位置

---

## 7. MoE 支持（面向未来）

DeepGEMM 专门提供了 MoE grouped GEMM kernel：

- `grouped_gemm_nt_f8f8bf16_masked()`：masked layout，适合训练
- `grouped_gemm_nt_f8f8bf16_contig()`：contiguous layout，适合推理

这是 CUTLASS 和 Triton 当前不具备的能力，是 DeepSeek-V3 等 MoE 模型高效推理的关键。ZImage 当前不是 MoE 模型，此优势暂不适用。

---

## 8. 硬件对比（H20 在 FP8 场景下的定位）

| GPU | SM 数量 | BF16 算力 | FP8 算力 | HBM 带宽 |
|---|---|---|---|---|
| H20 | 78 | 148 TFLOPS | **296 TFLOPS** | 4.0 TB/s |
| H100 SXM | 132 | 989 TFLOPS | 1,979 TFLOPS | 3.35 TB/s |
| H200 | 144 | 1,457 TFLOPS | 2,914 TFLOPS | 4.8 TB/s |

H20 的 HBM 带宽（4.0 TB/s）略高于 H100（3.35 TB/s），在**带宽受限**（bandwidth-bound）的场景（如短序列推理）表现接近 H100。但 FP8 算力仅为 H100 的 15%，**计算受限**（compute-bound）场景（如长序列、大 batch）与 H100 差距显著。

**关键结论**：H20 上用 DeepGEMM 获得 FP8 加速的前提是序列足够长，让 GEMM 进入 compute-bound 区间。对于 256×256 图片（短序列），当前仍处于 bandwidth-bound，FP8 收益有限。

---

## 9. 总结

| 维度 | 结论 |
|---|---|
| **算法创新** | 块量化 + 两级累加，在 FP8 精度下保持接近 BF16 的数值质量 |
| **Hopper 利用** | 完整利用 WGMMA + TMA + Persistent Kernel，是目前最充分利用 H100 FP8 算力的 GEMM 实现之一 |
| **vs CUTLASS** | 在对齐的矩阵形状上快 2.7×；短序列/小 M 时优势减小 |
| **当前瓶颈** | Scale transpose overhead（~70ms）在短序列场景抵消了 FP8 收益 |
| **最适场景** | 长序列（1024×1024+）、大 batch、MoE 模型（grouped GEMM） |
| **ZImage 256×256 结论** | 暂时无正收益；建议测试 1024×1024 并预转置 weight scale 后再评估 |

---

*报告生成时间：2026-03-23*
*参考来源：SGLang 源码 + DeepSeek DeepGEMM GitHub + 实测 benchmark 数据*
