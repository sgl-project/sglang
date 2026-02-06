# Roadmap 26Q1 核心目标分析

## 核心目标概览

Roadmap 26Q1 的核心目标是：**让 SGLang-Diffusion 更快、更稳定、支持更多功能**

主要分为三大类：
1. **性能提升**（Performance Improvements）- 核心中的核心
2. **功能扩展**（Features）- 增加新能力
3. **平台支持**（Platform & Backend）- 扩大适用范围

---

## 一、核心目标详解

### 1. 性能提升（Performance Improvements）- 最高优先级

这是 roadmap 的核心，目标是让推理速度更快。

#### 1.1 Lossless（无损优化）- 不改变模型精度

**核心目标**：通过优化计算和并行化，在不损失精度的情况下提升速度

| 任务 | 负责人 | 核心目标 | 难度 | 你能帮忙吗 |
|:-----|:-------|:---------|:-----|:----------|
| **Cuda graph for small models** | @zyksir | 小模型（如Qwen-Image）用CUDA graph加速 | 中 | ⭐⭐⭐ 可以 |
| **Parallel vae decoding** | - | VAE解码并行化，提升整体pipeline速度 | 高 | ⭐⭐ 可以尝试 |
| **Kernel optimizations** | - | 各种kernel优化（Norm, RoPE, QKV等） | 高 | ⭐⭐⭐ 可以 |
| **B200 Kernel Optimization** | @HydraQYH | 针对B200 GPU的256B load/store优化 | 高 | ⭐ 需要B200硬件 |
| **Better comm kernels for USP** | - | 改进Ulysses Parallel的通信kernel | 高 | ⭐⭐ 可以研究 |

**核心价值**：
- 这些优化直接影响所有模型的性能
- 是 SGLang-Diffusion 竞争力的核心

#### 1.2 Lossy（有损优化）- 可能改变精度但大幅提升速度

**核心目标**：通过量化等技术，在可接受的精度损失下大幅提升速度

| 任务 | 负责人 | 核心目标 | 难度 | 你能帮忙吗 |
|:-----|:-------|:---------|:-----|:----------|
| **Quantization** | @fsygd @RubiaCx | 支持多种量化格式（E4M3, INT8, MXFP8等） | 很高 | ⭐⭐ 可以测试 |
| **More Sparse Attention** | - | 更多稀疏注意力后端 | 高 | ⭐ 需要专业知识 |

**核心价值**：
- 让模型在消费级GPU上也能运行
- 大幅降低内存使用

### 2. 功能扩展（Features）

**核心目标**：增加新功能，提升易用性和应用场景

| 任务 | 负责人 | 核心目标 | 难度 | 你能帮忙吗 |
|:-----|:-------|:---------|:-----|:----------|
| **Postprocessing** | @yhyang201 @mickqian | 帧插值和超分辨率 | 中 | ⭐⭐⭐ 可以 |
| **Distillation support** | @RubiaCx | 支持模型蒸馏 | 高 | ⭐⭐ 可以研究 |
| **LoRA** | @niehen6174 | 提升LoRA覆盖和性能 | 中 | ⭐⭐⭐ 可以测试 |

### 3. 平台和后端支持

**核心目标**：扩大 SGLang-Diffusion 的适用范围

| 任务 | 负责人 | 核心目标 | 难度 | 你能帮忙吗 |
|:-----|:-------|:---------|:-----|:----------|
| **Refine ComfyUI plugin** | @niehen6174 | 改进ComfyUI集成 | 中 | ⭐⭐⭐ 可以 |
| **Improve diffusers backend** | @DefTruth | 提升diffusers后端性能 | 中 | ⭐⭐⭐⭐ **你在做** |
| **MacOS support** | - | 支持MacOS平台 | 中 | ⭐ 需要Mac |
| **Consumer-level GPU optimizations** | @ryang-max | 消费级GPU优化 | 中 | ⭐⭐⭐ 可以 |

---

## 二、你能帮忙的"边角料"任务

### ⭐⭐⭐⭐ 高优先级（你已经在做的）

#### 1. **Kernel optimizations** - 你正在做！

**你的工作（bug 18077）**：
- ✅ 已经完成了 GLM-Image 的基准测试（SGLang-D vs Diffusers 作为 baseline）
- 🔄 正在分析和优化 **SGLang-D 的性能**
- 🎯 目标：优化 SGLang-D 对 GLM-Image 的推理性能

**你的实际工作方向**：
- **Kernel 优化**：优化 SGLang-D 的 kernel（QKV, Norm, RoPE 等）
- **Sequence Parallelism**：为 GLM-Image 集成 SP 支持
- **Profiling 和优化**：找出 SGLang-D 的瓶颈并优化
- **多GPU 优化**：改进多GPU 并行性能

**注意**：
- Diffusers backend 的优化是 @DefTruth 在做，不是你的任务
- 你对比 Diffusers 只是为了建立性能 baseline
- 你的核心工作是**优化 SGLang-D 本身**

**价值**：你的工作属于 "Performance Improvements - Kernel optimizations"，这是 roadmap 的核心任务！

---

### ⭐⭐⭐ 中优先级（容易上手）

#### 2. **Kernel optimizations - Norm related**

**现状**：
- @yingluosanqian 已经有一个实现（`diffusion-norm-fusion-for-zimage` branch）
- @qimcis 准备接手

**你能做什么**：
- **测试和验证**：测试这个 kernel 在不同模型上的效果
- **Benchmark**：对比优化前后的性能
- **文档**：记录使用方法和效果
- **GLM-Image 适配**：看看能否应用到 GLM-Image

**具体行动**：
```bash
# 1. 查看那个 branch
git fetch origin diffusion-norm-fusion-for-zimage
git checkout diffusion-norm-fusion-for-zimage

# 2. 测试效果
# 3. 写测试报告
```

#### 3. **CUDA graph for small models**

**目标**：为小模型（如 Qwen-Image, GLM-Image）启用 CUDA graph

**你能做什么**：
- **研究现有实现**：看看哪些模型已经支持
- **GLM-Image 测试**：尝试为 GLM-Image 启用 CUDA graph
- **Benchmark**：测试效果
- **文档**：记录如何启用和使用

**价值**：GLM-Image 是"small model"，正好符合这个任务！

#### 4. **LoRA 测试和优化**

**目标**：提升 LoRA 的覆盖和性能

**你能做什么**：
- **测试更多 LoRA**：测试 GLM-Image 相关的 LoRA
- **性能对比**：对比不同 LoRA 的性能
- **问题报告**：发现 LoRA 相关的问题
- **文档贡献**：补充 LoRA 使用文档

#### 5. **Postprocessing - 帧插值和超分辨率**

**目标**：支持视频后处理功能

**你能做什么**：
- **需求调研**：了解用户需要什么样的后处理
- **测试现有方案**：测试开源的后处理库
- **集成测试**：测试与 SGLang-Diffusion 的集成
- **文档**：写使用指南

---

### ⭐⭐ 低优先级（需要更多时间）

#### 6. **Parallel VAE decoding**

**目标**：VAE 解码并行化

**你能做什么**：
- **研究现有实现**：看看其他框架是怎么做的
- **Profiling**：分析 VAE 解码是否是瓶颈
- **原型实现**：尝试简单的并行化方案
- **测试**：验证效果

#### 7. **Consumer-level GPU optimizations**

**目标**：让消费级 GPU（如 3060, 4060）也能高效运行

**你能做什么**：
- **测试不同 GPU**：在不同消费级 GPU 上测试
- **找出瓶颈**：分析消费级 GPU 的特定问题
- **优化建议**：提出针对性的优化方案
- **文档**：写消费级 GPU 优化指南

---

### ⭐ 需要特殊资源

#### 8. **B200 Kernel Optimization**

- 需要 B200 GPU 硬件
- 如果你有访问权限，可以帮忙测试

#### 9. **MacOS Support**

- 需要 Mac 设备
- 如果你有 Mac，可以帮忙测试和调试

---

## 三、推荐你优先做的任务

### 第一优先级：继续你的 bug 18077 工作（SGLang-D 优化）

**理由**：
- 你已经在做了，有基础
- 这是 roadmap 中 "Performance Improvements - Kernel optimizations" 的核心任务
- 有明确的产出（benchmark 报告、kernel 优化、SP 集成）

**你的核心任务**：
1. **Profiling SGLang-D**：找出 SGLang-D 对 GLM-Image 的瓶颈
2. **Kernel 优化**：优化 SGLang-D 的 kernel（QKV, Norm, RoPE, Weight Fusion 等）
3. **Sequence Parallelism**：为 GLM-Image 集成 SP 支持
4. **多GPU 优化**：改进多GPU 下的性能（解决 P99 问题）
5. **写总结报告**：记录所有优化和效果

**注意**：你优化的是 SGLang-D，不是 diffusers backend

### 第二优先级：CUDA graph for GLM-Image

**理由**：
- GLM-Image 是"small model"，正好符合
- 可以复用你现有的 benchmark 基础设施
- 相对独立，容易验证效果

**具体行动**：
```bash
# 1. 研究 CUDA graph 的启用方式
# 2. 为 GLM-Image 启用
# 3. Benchmark 对比
# 4. 写报告
```

### 第三优先级：Norm kernel 测试和适配

**理由**：
- @yingluosanqian 已经有实现
- 你可以帮忙测试和适配到 GLM-Image
- 贡献明确，容易得到认可

**具体行动**：
```bash
# 1. 联系 @yingluosanqian 或 @qimcis
# 2. 获取代码
# 3. 测试和适配
# 4. 贡献测试结果
```

---

## 四、如何贡献

### 1. 选择任务

- 从上面列表中选择 1-2 个任务
- 优先选择你已经在做的（bug 18077）
- 选择你有资源能完成的（比如有对应的硬件）

### 2. 联系负责人

- 在 GitHub Issue 中留言
- 在 Slack #diffusion 频道询问
- 说明你想做什么，需要什么帮助

### 3. 小步迭代

- 不要一开始就做大改动
- 先做小测试、小优化
- 验证效果后再继续

### 4. 记录和分享

- 记录所有测试结果
- 写文档说明你的发现
- 在 Issue 或 Slack 中分享

---

## 五、核心目标总结

**Roadmap 26Q1 的核心目标**：

1. **性能**：让所有模型推理更快（2-3x 提升）
2. **量化**：支持量化，让消费级 GPU 也能用
3. **功能**：增加后处理、更好的 LoRA 支持
4. **平台**：支持更多平台和硬件

**你的位置**：
- ✅ 你已经在做 "Performance Improvements - Kernel optimizations"（核心任务之一）
- ✅ 你的 GLM-Image 工作聚焦在优化 SGLang-D 的性能
- ✅ 你的工作可以扩展到其他 kernel 优化任务
- ✅ 你可以帮忙测试、验证、文档等工作

**注意区分**：
- ❌ **不是** "Improve diffusers backend"（那是 @DefTruth 的工作）
- ✅ **是** "Performance Improvements - Kernel optimizations"（你的工作）
- ✅ 你对比 Diffusers 只是为了建立性能 baseline，实际优化的是 SGLang-D

**建议**：
1. **继续深化 bug 18077**：这是你的主要贡献
2. **扩展相关优化**：CUDA graph、Norm kernel 等
3. **帮助测试和验证**：这是最容易的贡献方式
4. **写文档和报告**：记录你的发现，帮助其他人

---

## 六、具体行动建议

### 本周可以做的

1. **完成 GLM-Image profiling**
   - 这是你当前任务的一部分
   - 完成后可以分享给团队

2. **测试 CUDA graph**
   - 研究如何为 GLM-Image 启用
   - 测试效果

3. **联系 Norm kernel 负责人**
   - 询问能否帮忙测试
   - 看看能否应用到 GLM-Image

### 本月可以做的

1. **完成 GLM-Image 的 SGLang-D 优化**
   - 实现一些 kernel 优化（QKV, Norm, RoPE 等）
   - 集成 SP（如果可能）
   - 优化多GPU 性能

2. **扩展到其他 kernel 优化**
   - 用同样的方法分析其他模型
   - 找出 SGLang-D kernel 的通用优化点
   - 贡献到其他 kernel 优化任务

3. **贡献文档**
   - 写 profiling 指南
   - 写 kernel 优化经验总结
   - 记录 GLM-Image 优化过程

---

记住：**小贡献也是贡献**！测试、验证、文档、bug 报告都是有价值的。
