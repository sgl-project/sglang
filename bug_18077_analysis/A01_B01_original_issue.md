# A01_B01: Issue #18077 原始内容

## 相关文档
- [A01: GLM-Image 推理效率优化详解](./A01_glm_image_performance.md) - 了解整体问题
- [A01_B02: 性能分析](./A01_B02_performance_analysis.md) - 性能基准测试结果
- [A01_B03: 代码分析](./A01_B03_code_analysis.md) - 代码实现分析

---

## Issue 链接
https://github.com/sgl-project/sglang/issues/18077

## 问题标题
[Feature] Benchmark and Optimize GLM-Image Inference Efficiency (SGLang-D vs. Diffusers)

## 提交者
@zhaochenyang20

## 问题描述

### Description
We are looking to evaluate the current inference performance of `zai-org/GLM-Image` when running on the sglang-diffusion engine compared to the baseline Diffusers implementation.

Preliminary observations suggest that the current implementation for GLM-Image within our stack may be under-optimized. Specifically, it appears to lack support for Sequence Parallelism (SP), which is crucial for handling high-resolution image generation efficiently. Improving this will not only boost GLM-Image performance but also provide architectural insights for the broader SGLang-D project.

### 关键问题：是否必须使用 GLM-Image？

**重要发现**：根据 Issue 评论中的实际测试数据，这个问题**并不局限于 GLM-Image**。

#### 原始 Issue 目标
- **目标模型**: `zai-org/GLM-Image`
- **目标**: 评估 GLM-Image 在 SGLang-D 上的性能

#### 实际测试发现（来自 @haojin2 的评论）
- **测试模型**: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`（**不是 GLM-Image**）
- **测试结果**: 同样存在严重的性能问题
- **关键结论**: "Can confirm this is an issue even on other models, maybe not specific to GLM?"

#### 问题通用性分析

**结论：这个问题是通用的，不限于 GLM-Image**

1. **问题本质**：
   - 不是 GLM-Image 特有的问题
   - 而是 SGLang-D 引擎对 **DiT (Diffusion Transformer) 架构模型**的通用性能问题
   - 影响所有使用类似架构的模型

2. **受影响的模型类型**：
   - ✅ GLM-Image (`zai-org/GLM-Image`)
   - ✅ Wan2.1-T2V (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`)
   - ⚠️ 其他 DiT 架构的扩散模型（如 Z-Image-Turbo 等）

3. **为什么选择 GLM-Image 作为示例**：
   - GLM-Image 是典型的 DiT 架构模型
   - 具有良好的代表性
   - 优化 GLM-Image 可以推广到其他类似模型

4. **测试模型的选择**：
   - Issue 中提到的测试数据来自 `Wan2.1-T2V`，而不是 GLM-Image
   - 这说明问题确实具有通用性
   - 可以用任何 DiT 架构的模型来复现和测试这个问题

### Goals
1. **Benchmarking**: Establish a performance baseline (latency, throughput, and VRAM usage) for GLM-Image using both sglang-diffusion and diffusers.
2. **Profiling**: Identify bottlenecks in the current sglang-diffusion path for this model (e.g., attention kernels, memory overhead).
3. **Optimization (Optional/Bonus)**: Propose or implement initial optimizations, such as enabling Sequence Parallelism or improving memory management.

### Technical Tasks
- [ ] Set up a reproducible benchmarking script for GLM-Image
- [ ] Compare inference latency across different batch sizes and resolutions
- [ ] Analyze if and where Sequence Parallelism can be integrated into the current GLM-Image wrapper
- [ ] Document the findings in a detailed report or table within this issue

### Reference
- [SGLang-D Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/sgl_diffusion_en.md)

---

## 初步观察

### 性能对比数据（来自 Issue 评论）

#### Diffusers Backend
```
================= Serving Benchmark Result =================
Task:                                    text-to-video  
Model:                                   Wan-AI/Wan2.1-T2V-1.3B-Diffusers
Dataset:                                 vbench         
--------------------------------------------------
Benchmark duration (s):                  20.06          
Request rate:                            inf            
Max request concurrency:                 1              
Successful requests:                     10/10             
--------------------------------------------------
Request throughput (req/s):              0.50           
Latency Mean (s):                        2.0054         
Latency Median (s):                      2.0053         
Latency P99 (s):                         2.0067         
--------------------------------------------------
Peak Memory Max (MB):                    14026.38       
Peak Memory Mean (MB):                   14026.38       
Peak Memory Median (MB):                 14026.38       
============================================================
```

#### SGLang Backend
```
================= Serving Benchmark Result =================
Task:                                    text-to-video  
Model:                                   Wan-AI/Wan2.1-T2V-1.3B-Diffusers
Dataset:                                 vbench         
--------------------------------------------------
Benchmark duration (s):                  92.22          
Request rate:                            inf            
Max request concurrency:                 1              
Successful requests:                     10/10             
--------------------------------------------------
Request throughput (req/s):              0.11           
Latency Mean (s):                        9.2222         
Latency Median (s):                        9.0214         
Latency P99 (s):                         10.8502        
--------------------------------------------------
Peak Memory Max (MB):                    8170.82        
Peak Memory Mean (MB):                   8170.44       
Peak Memory Median (MB):                 8170.58       
============================================================
```

### 性能差距分析
- **吞吐量**: SGLang (0.11 req/s) vs Diffusers (0.50 req/s) - **约 4.5x 差距**
- **延迟**: SGLang (9.2s) vs Diffusers (2.0s) - **约 4.6x 差距**
- **内存使用**: SGLang (8.2GB) vs Diffusers (14.0GB) - SGLang 更节省内存

### 关键发现

#### 1. 性能问题
- **SGLang 后端的延迟和吞吐量明显低于 Diffusers**
- 延迟差距：约 4.6x（9.2s vs 2.0s）
- 吞吐量差距：约 4.5x（0.11 req/s vs 0.50 req/s）

#### 2. 内存优势
- SGLang 在内存使用上更高效（8.2GB vs 14.0GB）
- 但这是以性能为代价的

#### 3. **通用性问题（重要）** ⚠️
这个问题**不仅限于 GLM-Image**，而是影响所有 DiT 架构的扩散模型：

- ✅ **GLM-Image** (`zai-org/GLM-Image`) - Issue 原始目标
- ✅ **Wan2.1-T2V** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) - 实际测试模型，同样存在性能问题
- ⚠️ **其他 DiT 模型** - 可能也受影响（如 Z-Image-Turbo 等）

**关键证据**：
- @haojin2 的测试使用的是 `Wan2.1-T2V`，而不是 GLM-Image
- 测试结果显示相同的性能问题模式
- 评论明确说明："Can confirm this is an issue even on other models, maybe not specific to GLM?"

**结论**：
- ❌ **不需要必须使用 GLM-Image** 来复现这个问题
- ✅ 可以使用任何 DiT 架构的扩散模型（如 Wan2.1-T2V）来测试
- ✅ 优化方案应该针对 DiT 架构的通用问题，而不是 GLM-Image 特定问题

---

## 参与者

- @zhaochenyang20 - Issue 创建者
- @haojin2 - 贡献者（已确认问题并提供了基准测试数据）

---

## 标签
- `Good Pro Issue` - 需要深入了解 SGLang 内部机制
- `diffusion` - 与 SGLang Diffusion 相关
- `SGLang Diffusion` - SGLang Diffusion 模块

---

## 下一步行动

### 基准测试计划

#### 模型选择（不限于 GLM-Image）
1. **主要测试模型**：
   - `zai-org/GLM-Image` - Issue 原始目标
   - `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` - 已验证存在相同问题
   - 其他 DiT 架构模型（可选）

2. **测试目标**：
   - 设置可复现的基准测试脚本
   - 在不同批次大小和分辨率下进行测试
   - 分析 Sequence Parallelism 的集成点
   - 生成详细的性能报告

#### 优化方向
- 针对 **DiT 架构的通用优化**，而不是 GLM-Image 特定优化
- 重点解决 Sequence Parallelism 支持问题
- 优化内存管理和内核性能

### 重要说明

**关于模型选择**：
- ✅ 可以使用 `Wan2.1-T2V` 或其他 DiT 模型来测试和验证
- ✅ 不需要强制使用 GLM-Image
- ✅ 优化方案应该具有通用性，适用于所有 DiT 架构模型
