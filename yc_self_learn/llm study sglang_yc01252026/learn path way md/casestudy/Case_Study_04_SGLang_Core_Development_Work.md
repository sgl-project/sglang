# Case Study 04: SGLang 核心开发工作总览

## 📚 文档信息

**目的**：理解 SGLang 除了新模型支持之外的核心开发工作  
**适用场景**：理解 SGLang 的开发方向、选择贡献方向、规划职业发展

---

## 🎯 核心问题

**除了新模型支持（Day 0/Day 1 Support）之外，SGLang 还有哪些核心开发工作？**

---

## 📋 SGLang 核心开发工作分类

### 1. 性能优化与架构改进 ⚡ **最核心**

**目标**：持续提升推理性能，降低延迟和成本

#### 1.1 RadixAttention - 前缀缓存技术

**工作内容**：
- 优化 Radix Tree 数据结构
- 提升缓存命中率
- 支持更复杂的缓存策略
- 优化内存使用

**影响**：提供高达 **5x 的推理加速**

**相关 Issue/PR**：
- 优化 RadixAttention 性能
- 支持新的缓存策略
- 修复缓存相关 bug

#### 1.2 调度器优化

**工作内容**：
- 零开销 CPU 调度器优化
- 连续批处理（Continuous Batching）改进
- 请求优先级管理
- Prefill-Decode 分离调度优化

**影响**：最大化 GPU 利用率，减少延迟

**你的经验**：你修过 router 高并发调度瓶颈，这是这个方向的工作

#### 1.3 推测解码（Speculative Decoding）

**工作内容**：
- Overlap Spec Support（重叠推测解码）
- 优化推测解码性能
- 支持新的推测策略

**影响**：显著提升吞吐量

#### 1.4 Attention 优化

**工作内容**：
- Flash Attention 集成和优化
- 新的 Attention kernel 实现
- 多 GPU Attention 优化
- SageAttention 优化（Blackwell GPU）

**影响**：降低 Attention 计算时间

#### 1.5 内存管理优化

**工作内容**：
- Paged Attention 优化
- KV Cache 管理优化
- 内存碎片减少
- 支持更长序列

**影响**：支持更大 batch size，减少 OOM

---

### 2. 硬件兼容性扩展 🔧

**目标**：支持更多硬件平台，让 SGLang 能在不同硬件上高效运行

#### 2.1 新硬件支持

**工作内容**：
- **华为昇腾（Ascend）支持**
  - Issue #16360: Triton 编译错误修复
  - 原生支持 Ascend 910B4-1 硬件
- **NPU（Neural Processing Unit）支持**
  - Issue #16498: NPU Runtime 错误修复
  - 支持推理加速芯片
- **AMD GPU 优化**
  - 针对 AMD Instinct MI300X 优化
  - ROCm 支持优化
- **NVIDIA Blackwell (GB200) 支持**
  - Issue #16302: SageAttention3 在 GB10 上的问题修复
  - Issue #16322: GB300 精度问题
  - GB200 NVL72 部署优化

**为什么重要**：
- 企业客户使用不同的硬件平台
- 需要最大化硬件利用率
- 降低硬件锁定（vendor lock-in）风险

#### 2.2 硬件特定优化

**工作内容**：
- 针对特定硬件的 kernel 优化
- 硬件特定性能调优
- 兼容性测试和验证

---

### 3. 功能增强与扩展 🚀

**目标**：添加新功能，满足不同用户需求

#### 3.1 结构化输出增强

**工作内容**：
- JSON Schema 支持优化
- Regex 约束优化
- EBNF 支持
- 压缩 FSM 优化（3x 更快的 JSON 解码）

**影响**：提升结构化输出的性能和易用性

#### 3.2 多模态支持

**工作内容**：
- 图像模型支持（VLM）
- 视频模型支持
- 音频模型支持
- 多模态推理优化

**你的经验**：你做过 GLM-Image/Diffusion 相关的工作，这是这个方向

#### 3.3 LoRA 支持

**工作内容**：
- 多 LoRA 批处理
- LoRA 加载优化
- LoRA 切换优化

**影响**：支持更灵活的模型适配

#### 3.4 推理模型（Reasoning Models）支持

**工作内容**：
- DeepSeek V3/R1 支持
- Qwen3-Thinking 支持
- Reasoning 相关优化

**影响**：支持需要"思考"过程的模型

---

### 4. 系统架构改进 🏗️

**目标**：改进系统架构，提升可扩展性和可维护性

#### 4.1 Prefill-Decode 分离（Disaggregation）

**工作内容**：
- 优化 Prefill 和 Decode 的分离架构
- 独立扩展不同阶段
- 资源分配优化

**影响**：允许针对性地优化不同阶段

#### 4.2 并行策略优化

**工作内容**：
- Tensor Parallelism (TP) 优化
- Sequence Parallelism (SP) 优化
- Pipeline Parallelism (PP) 优化
- Expert Parallelism (EP) 优化

**你的经验**：你做过 GLM-Image SP support，这是这个方向

#### 4.3 分布式部署优化

**工作内容**：
- 多节点部署优化
- 通信优化（all2all, allreduce）
- 负载均衡优化

**影响**：支持更大规模的部署

---

### 5. Bug 修复与稳定性 🐛

**目标**：修复已知问题，提升系统稳定性

#### 5.1 模型特定 Bug

**工作内容**：
- Issue #16289: GLM 4.6/4.7 只输出 "!"
- Issue #16501: DeepSeek V3 continue_final_message 错误
- Issue #16461: DualChunkRotaryEmbedding 参数错误

**你的经验**：你修过很多 bug，这是这个方向的工作

#### 5.2 内存相关问题

**工作内容**：
- Issue #16439: Swapped Pool 内存未注册
- Issue #16322: GB300 精度问题
- 内存泄漏修复

**你的经验**：你修过 FD leak，这是这个方向

#### 5.3 并发和死锁问题

**工作内容**：
- 并发死锁修复
- 线程安全问题修复
- 资源竞争问题修复

**你的经验**：你定位过 "2 concurrency deadlocks"，这是这个方向

---

### 6. 工具链与开发体验 🛠️

**目标**：提升开发体验，降低开发门槛

#### 6.1 Benchmark 和 Profiling 工具

**工作内容**：
- Benchmark 工具优化
- Profiling 工具增强
- 性能分析工具改进

**你的经验**：你做过 benchmark 系列（TTFT/ITL/p99/VRAM），这是这个方向

#### 6.2 CI/CD 系统

**工作内容**：
- CI 流程优化
- 测试框架改进
- 自动化测试增强

**你的经验**：你做过 regression 定位，这是这个方向

#### 6.3 文档和示例

**工作内容**：
- 文档完善
- 示例代码更新
- 教程改进

---

### 7. 前端语言与 API 改进 🌐

**目标**：提升易用性和灵活性

#### 7.1 前端语言（Frontend Language）

**工作内容**：
- 前端语言功能增强
- 控制流优化
- 并行处理优化

**影响**：让用户更容易编写复杂的生成逻辑

#### 7.2 API 改进

**工作内容**：
- OpenAI-compatible API 优化
- 新 API 端点添加
- API 性能优化

**影响**：提升 API 易用性和性能

---

## 📊 开发工作优先级

### 第一优先级（核心性能）⭐⭐⭐

1. **性能优化**：RadixAttention、调度器、Attention 优化
2. **硬件兼容性**：新硬件支持、硬件特定优化
3. **系统架构**：Prefill-Decode 分离、并行策略优化

**为什么重要**：
- 直接影响 SGLang 的核心竞争力
- 用户最关心的性能问题
- 技术难度高，影响面大

### 第二优先级（功能扩展）⭐⭐

1. **新模型支持**：Day-0/Day-1 支持
2. **功能增强**：结构化输出、多模态、LoRA
3. **推理模型支持

**为什么重要**：
- 满足用户需求
- 扩大应用场景
- 保持竞争力

### 第三优先级（稳定性与体验）⭐

1. **Bug 修复**：模型特定 bug、内存问题、并发问题
2. **工具链**：Benchmark、Profiling、CI/CD
3. **文档和示例**：提升易用性

**为什么重要**：
- 提升系统稳定性
- 改善开发体验
- 降低使用门槛

---

## 🎯 基于你的经验的工作方向

### 你已经做过的方向

1. ✅ **新模型支持**：GLM-Image/Diffusion SP support
2. ✅ **性能优化**：Router 调度瓶颈修复
3. ✅ **Bug 修复**：FD leak、diffusion extras 依赖
4. ✅ **工具链**：Benchmark 系列、regression 定位

### 你可以继续深入的方向

#### 方向 1：性能优化（推荐）⭐⭐⭐

**为什么适合你**：
- 你修过 router 调度瓶颈
- 你做过 benchmark 和 regression 定位
- 性能优化是 SGLang 的核心工作

**具体工作**：
- RadixAttention 优化
- 调度器进一步优化
- Attention kernel 优化
- 内存管理优化

#### 方向 2：系统架构改进（推荐）⭐⭐⭐

**为什么适合你**：
- 你做过 SP support（并行策略）
- 你理解系统架构（从你的问题可以看出）
- 架构改进影响面大

**具体工作**：
- Prefill-Decode 分离优化
- 并行策略进一步优化
- 分布式部署优化

#### 方向 3：硬件兼容性（可选）⭐⭐

**为什么适合你**：
- 你理解底层实现（从你的问题可以看出）
- 硬件适配是重要方向

**具体工作**：
- 新硬件支持（NPU、昇腾等）
- 硬件特定优化

---

## 💡 核心开发工作总结

### 除了新模型支持，SGLang 的核心工作包括：

1. **性能优化** ⚡（最核心）
   - RadixAttention、调度器、Attention、内存管理

2. **硬件兼容性** 🔧
   - 新硬件支持、硬件特定优化

3. **功能增强** 🚀
   - 结构化输出、多模态、LoRA、推理模型

4. **系统架构** 🏗️
   - Prefill-Decode 分离、并行策略、分布式部署

5. **Bug 修复** 🐛
   - 模型特定 bug、内存问题、并发问题

6. **工具链** 🛠️
   - Benchmark、Profiling、CI/CD

7. **前端语言与 API** 🌐
   - 前端语言增强、API 改进

### 开发流程

**通常采用分阶段开发**：
1. **基础支持**：让功能能跑起来
2. **性能优化**：逐步优化性能
3. **功能增强**：添加新功能
4. **稳定性**：修复 bug，提升稳定性

---

## 🔗 相关资源

### 官方文档
- [SGLang 官方文档](https://docs.sglang.ai/) ⭐⭐⭐
- [SGLang GitHub](https://github.com/sgl-project/sglang) ⭐⭐⭐
- [SGLang Issues](https://github.com/sgl-project/sglang/issues) ⭐⭐

### 相关 Case Study
- [Case Study 01: Speed Up SGL-Kernel Build](./Case_Study_01_Speed_Up_SGL_Kernel_Build_PR18586.md)
- [Case Study 02: SGLang Request Processing Flow](./Case_Study_02_SGLang_Request_Processing_Flow.md)
- [Case Study 03: Model Definition and Loading](./Case_Study_03_SGLang_Model_Definition_and_Loading.md)

### 相关学习文档
- [SGLang 开发重点与技术创新](../md/13_02_SGLang_开发重点与技术创新.md) ⭐⭐
- [SGLang 当前工作与 AI Infra 工程师主流工作](../md/11_SGLang当前工作与AI_Infra工程师主流工作.md) ⭐⭐

---

## ✅ 总结

**核心答案**：

除了新模型支持（Day 0/Day 1 Support）之外，SGLang 的核心开发工作包括：

1. **性能优化**（最核心）：RadixAttention、调度器、Attention、内存管理
2. **硬件兼容性**：新硬件支持、硬件特定优化
3. **功能增强**：结构化输出、多模态、LoRA、推理模型
4. **系统架构**：Prefill-Decode 分离、并行策略、分布式部署
5. **Bug 修复**：模型特定 bug、内存问题、并发问题
6. **工具链**：Benchmark、Profiling、CI/CD
7. **前端语言与 API**：前端语言增强、API 改进

**基于你的经验，推荐继续深入**：
- **性能优化**：你修过 router 调度瓶颈，做过 benchmark
- **系统架构**：你做过 SP support，理解系统架构

---

**最后更新**: 2025年1月
