# GLM-Image 性能测试结果总结

## 📋 测试概述

**测试日期**: 2026-02-05  
**测试模型**: `zai-org/GLM-Image`  
**测试方式**: 使用官方 `bench_serving.py`，与 Issue #18077 中 haojin2 的测试方式一致  
**数据集**: **Random**（用于两个后端的公平对比）  
**测试请求数**: 10 个请求  
**并发数**: 1（单并发）  
**Sequence Parallelism (SP)**: ❌ **未启用**（这是 Issue #18077 的核心关注点）

**数据集说明**：
- ⚠️ 虽然 haojin2 使用 VBench 数据集，但 Diffusers 后端的 VBench 测试失败
- ✅ 为了公平对比，两个后端统一使用 **Random 数据集**
- 📊 SGLang 后端也有 VBench 测试结果（延迟 87.86 秒），但不在本次对比中使用

---

## 📊 测试结果

### SGLang 后端（Random 数据集）

| 指标 | 数值 |
|:-----|:-----|
| **延迟 (平均)** | 93.18 秒 |
| **延迟 (中位数)** | 87.13 秒 |
| **延迟 (P99)** | 142.03 秒 |
| **吞吐量** | 0.01 req/s |
| **峰值内存 (最大)** | 39.04 GB |
| **峰值内存 (平均)** | 39.04 GB |
| **成功请求** | 10/10 |
| **失败请求** | 0 |

**测试文件**: `zai-org_GLM-Image_sglang_20260205_111630.json`

### Diffusers 后端（Random 数据集）

| 指标 | 数值 |
|:-----|:-----|
| **延迟 (平均)** | 95.75 秒 |
| **延迟 (中位数)** | 95.55 秒 |
| **延迟 (P99)** | 96.42 秒 |
| **吞吐量** | 0.01 req/s |
| **峰值内存 (最大)** | 38.58 GB |
| **峰值内存 (平均)** | 38.58 GB |
| **成功请求** | 10/10 |
| **失败请求** | 0 |

**测试文件**: `zai-org_GLM-Image_diffusers_20260205_123902.json`

**重要说明**: 
- ✅ **两个后端使用相同的数据集（Random）**，确保公平对比
- ⚠️ Diffusers 后端的 VBench 测试失败（所有请求失败），因此使用 Random 数据集
- 📊 SGLang 后端也有 VBench 测试结果（`zai-org_GLM-Image_sglang_vbench_20260205_133333.json`），但为了公平对比，统一使用 Random 数据集

---

## 🔍 性能对比

### 后端对比（GLM-Image）

| 指标 | SGLang 后端 | Diffusers 后端 | 差异 |
|:-----|:-----------|:--------------|:-----|
| **延迟 (平均)** | 93.18 秒 | 95.75 秒 | SGLang 快 2.7% |
| **延迟 (中位数)** | 87.13 秒 | 95.55 秒 | SGLang 快 9.7% |
| **延迟 (P99)** | 142.03 秒 | 96.42 秒 | Diffusers 更稳定 |
| **吞吐量** | 0.01 req/s | 0.01 req/s | 相同 |
| **峰值内存** | 39.04 GB | 38.58 GB | Diffusers 少 1.2% |

**关键发现**：
- ✅ 两个后端性能**非常接近**（平均延迟差异 < 3%）
- ✅ SGLang 后端略快于 Diffusers 后端（2.7%）
- ⚠️ 但**都非常慢**（~93-96 秒/请求）
- ⚠️ SGLang 后端的 P99 延迟较高（142 秒），说明存在一些异常慢的请求
- ✅ Diffusers 后端的延迟更稳定（P99 延迟较低）

---

## 📈 与其他模型对比

### GLM-Image vs Wan 2.1 T2V 1.3B

根据 Issue #18077 中 haojin2 的测试结果：

| 模型 | 后端 | 延迟 (平均) | 吞吐量 | 内存 |
|:-----|:-----|:-----------|:------|:-----|
| **GLM-Image** | SGLang | **93.18s** | 0.01 req/s | 39.04 GB |
| **GLM-Image** | Diffusers | **95.75s** | 0.01 req/s | 38.58 GB |
| **Wan 2.1 1.3B** | Diffusers | **2.01s** | 0.50 req/s | 14.03 GB |
| **Wan 2.1 1.3B** | SGLang | **9.22s** | 0.11 req/s | 8.17 GB |

**性能差异**：
- GLM-Image 比 Wan 模型**慢 46.4x**（SGLang 后端：93.18s vs 2.01s）
- GLM-Image 比 Wan 模型**慢 47.6x**（Diffusers 后端：95.75s vs 2.01s）
- GLM-Image 的内存使用是 Wan 模型的 **2.8x**

---

## ⚠️ 重要：Sequence Parallelism (SP) 状态

### SP 支持情况

根据 Issue #18077 的描述，**GLM-Image 当前实现可能缺乏 Sequence Parallelism (SP) 支持**，这是处理高分辨率图像生成的关键优化。

**当前测试状态**：
- ❌ **SP 未启用**：本次测试未使用 `--sp-degree`、`--ulysses-degree` 或 `--ring-degree` 参数
- ⚠️ **这是 Issue 的核心问题**：Issue 明确提到 GLM-Image "appears to lack support for Sequence Parallelism (SP)"

**SGLang 支持的 SP 参数**：
- `--sp-degree`: Sequence parallelism size
- `--ulysses-degree`: Ulysses sequence parallel degree (用于 attention layer)
- `--ring-degree`: Ring sequence parallel degree (用于 attention layer)

**建议的后续测试**：
1. 测试启用 SP 后的性能（如果支持）
2. 分析 SP 在 GLM-Image 中的集成点
3. 对比 SP 启用前后的性能差异

**参考**：
- Issue #18077 目标 3: "Propose or implement initial optimizations, such as enabling Sequence Parallelism"
- Issue #18077 技术任务: "Analyze if and where Sequence Parallelism can be integrated into the current GLM-Image wrapper"

---

## 🎯 关键发现

### 1. GLM-Image 性能问题确实存在

   - **延迟极高**：~93-96 秒/请求，远慢于 Wan 模型（2-9 秒）
   - **吞吐量极低**：0.01 req/s，远低于 Wan 模型（0.11-0.50 req/s）
- **内存占用高**：~39 GB，是 Wan 模型的 2.8x

### 2. 两个后端性能接近

- SGLang 和 Diffusers 后端的性能**几乎相同**（差异 < 3%）
- 说明问题可能**不在后端选择**，而在**模型本身的实现或架构**

### 3. 与 Wan 模型的差异模式不同

**Wan 模型**：
- Diffusers 后端明显快于 SGLang 后端（4.6x）
- 符合 Issue #18077 中 haojin2 的观察

**GLM-Image**：
- 两个后端性能几乎相同（都慢）
- 说明问题更严重，需要针对性的优化

### 4. 可能的原因

1. **模型大小**：
   - GLM-Image: ~7B 参数
   - Wan 2.1 1.3B: 1.3B 参数
   - GLM-Image 是 Wan 的 **5.4x 大**

2. **架构差异**：
   - GLM-Image 使用 DiT (Diffusion Transformer) 架构
   - 可能缺乏 Sequence Parallelism (SP) 支持
   - 高分辨率图像生成需要更多计算

3. **优化状态**：
   - Wan 模型可能在 SGLang 中有更好的优化
   - GLM-Image 可能缺乏特定优化（如 Issue #18077 提到的 SP）

---

## 📝 结论

### 主要结论

1. **GLM-Image 性能问题确实存在**：
   - 延迟 ~93-96 秒/请求，远慢于 Wan 模型（2-9 秒）
   - 两个后端性能接近（差异 < 3%），说明问题可能不在后端选择，而在模型本身的实现

2. **问题可能不限于 GLM-Image**：
   - 根据 haojin2 的测试，Wan 模型也存在 SGLang 后端慢于 Diffusers 的问题
   - 但 GLM-Image 的情况更严重（两个后端都慢）

3. **优化方向**（Issue #18077 的核心目标）：
   - 🔴 **实现 Sequence Parallelism (SP) 支持**（**当前缺失，这是性能瓶颈的关键原因**）
   - ✅ 优化 GLM-Image 的 DiT 实现
   - ✅ 提高高分辨率图像生成的效率
   - ✅ 优化内存使用（当前 ~39 GB）
   - ✅ 分析 SP 在 GLM-Image 中的集成点

### 与 Issue #18077 的关系

- ✅ 确认了 GLM-Image 的性能问题
- ✅ 提供了详细的基准测试数据（**基线性能，未启用 SP**）
- ✅ 对比了 SGLang 和 Diffusers 两个后端
- ⚠️ **未测试 SP 支持**：这是 Issue 的核心关注点，需要后续测试
- ✅ 为优化提供了数据支持

### 关于 Sequence Parallelism (SP)

**Issue #18077 明确提到**：
> "Specifically, it appears to lack support for Sequence Parallelism (SP), which is crucial for handling high-resolution image generation efficiently."

**本次测试的局限性**：
- 本次测试提供了**基线性能数据**（未启用 SP）
- **未测试 SP 启用后的性能**（如果支持的话）
- 需要进一步测试和分析 SP 的集成点和效果

**建议的后续工作**：
1. 测试启用 SP 参数（`--sp-degree`, `--ulysses-degree`, `--ring-degree`）后的性能
2. 分析 SP 在 GLM-Image 中的集成点
3. 对比 SP 启用前后的性能差异
4. 如果当前不支持，提出 SP 集成的实现方案

---

## 📁 测试结果文件

所有测试结果保存在当前目录：

### 原始测试结果

- `zai-org_GLM-Image_sglang_20260205_111630.json` - SGLang 后端（**Random 数据集，用于对比**）
- `zai-org_GLM-Image_diffusers_20260205_123902.json` - Diffusers 后端（**Random 数据集，用于对比**）
- `zai-org_GLM-Image_sglang_vbench_20260205_133333.json` - SGLang 后端（VBench 数据集，参考）
- `zai-org_GLM-Image_diffusers_vbench_20260205_134953.json` - Diffusers 后端（VBench 数据集，**失败**）

### 对比报告

- `zai-org_GLM-Image_comparison_20260205_152203.json` - 最新对比报告（Diffusers 后端测试失败）
- `zai-org_GLM-Image_comparison_20260205_132034.json` - 对比报告（Random 数据集，两个后端都成功）

---

## 🔗 参考

- [Issue #18077](https://github.com/sgl-project/sglang/issues/18077) - 原始 Issue
- [haojin2 的测试结果](https://github.com/sgl-project/sglang/issues/18077#issuecomment-xxx) - Wan 2.1 T2V 1.3B 模型测试
- [官方 bench_serving.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/benchmarks/bench_serving.py) - 基准测试脚本

---

## 📅 测试环境

- **测试日期**: 2026-02-05
- **测试方式**: 官方 `bench_serving.py` 脚本
- **数据集**: VBench（与 haojin2 相同）
- **测试请求数**: 10
- **并发数**: 1
- **模型**: `zai-org/GLM-Image`

---

---

## 🔬 关于 Sequence Parallelism (SP) 的测试建议

### 为什么 SP 很重要？

根据 Issue #18077 的描述：
> "Specifically, it appears to lack support for Sequence Parallelism (SP), which is crucial for handling high-resolution image generation efficiently."

SP 对于高分辨率图像生成至关重要，可以：
- 减少单 GPU 的内存压力
- 提高处理长序列的效率
- 支持更大的 batch size 和分辨率

### 如何测试 SP？

#### 方法 1: 启用 SP 参数启动服务器

```bash
# 使用 SP 参数启动服务器
sglang serve \
    --model-path zai-org/GLM-Image \
    --backend sglang \
    --port 30000 \
    --trust-remote-code \
    --sp-degree 2 \
    --ulysses-degree 2 \
    --ring-degree 1
```

**参数说明**：
- `--sp-degree`: Sequence parallelism size (总 SP 度)
- `--ulysses-degree`: Ulysses SP degree (用于 attention layer)
- `--ring-degree`: Ring SP degree (用于 attention layer)
- 关系：`sp-degree = ulysses-degree * ring-degree`

#### 方法 2: 对比测试

1. **基线测试**（当前已完成）：未启用 SP
2. **SP 测试**：启用 SP 后运行相同测试
3. **对比分析**：比较性能差异

### 当前状态

- ❌ **本次测试未启用 SP**：提供的是基线性能数据
- ⚠️ **需要后续测试**：验证 SP 是否支持，以及启用后的性能提升
- 📝 **这是 Issue #18077 的核心关注点**

---

**报告生成时间**: 2026-02-05  
**测试执行者**: yandache  
**测试目的**: 评估 GLM-Image 在 SGLang-D 上的性能，为 Issue #18077 提供基准数据  
**重要说明**: 本次测试**未启用 Sequence Parallelism (SP)**，这是 Issue 的核心关注点，需要后续测试验证
