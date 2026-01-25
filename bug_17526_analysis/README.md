# Bug #17526 分析文档 - GLM Blackwell性能优化

## 📋 文档结构

### A01: GLM Blackwell性能优化详解
- **[A01_glm_blackwell_optimization.md](./A01_glm_blackwell_optimization.md)** ⭐ **主文档**
  - GLM 4.7 FP4在Blackwell上的性能优化
  - 优化项汇总
  - 性能对比分析

### A01_Bxx: 平行文档（与 A01 同级）

- **[A01_B01_original_issue.md](./A01_B01_original_issue.md)** ⭐ **原始 Issue 内容**
  - Issue 链接和标题
  - 优化项列表
  - 性能测试结果
  - 配置文件

- **[A01_B02_optimization_items.md](./A01_B02_optimization_items.md)** ⭐ **优化项详解**
  - 每个优化项的详细说明
  - 预期性能提升
  - 实现状态

- **[A01_B03_performance_analysis.md](./A01_B03_performance_analysis.md)** ⭐ **性能分析**
  - FP8 vs BF16性能对比
  - MTP性能分析
  - 瓶颈识别

- **[A01_B04_blackwell_architecture.md](./A01_B04_blackwell_architecture.md)** ⭐ **Blackwell架构分析**
  - Blackwell GPU特性
  - SM100/SM120架构
  - 性能优化机会

- **[A01_B05_quantization_analysis.md](./A01_B05_quantization_analysis.md)** ⭐ **量化分析**
  - FP4量化原理
  - FP8量化原理
  - KV Cache量化

- **[A01_B06_learning_path.md](./A01_B06_learning_path.md)** 📚 **学习路径**
  - 从零开始学习性能优化
  - 相关文档链接
  - 实践步骤

- **[A01_B07_complete_learning_path.md](./A01_B07_complete_learning_path.md)** 🎓 **完整学习流程** ⭐ **推荐**
  - 从零开始的12周完整学习计划
  - 每天的学习任务和实践
  - 循序渐进的学习路径
  - 最终能够解决性能优化问题

---

## 🎯 快速导航

1. **想了解问题详情** → [A01_B01_original_issue.md](./A01_B01_original_issue.md) ⭐ **从这里开始**
2. **想从零开始学习** → [A01_B07_complete_learning_path.md](./A01_B07_complete_learning_path.md) 🎓 **完整学习流程** ⭐ **强烈推荐**
3. **想了解优化项** → [A01_B02_optimization_items.md](./A01_B02_optimization_items.md)
4. **想分析性能** → [A01_B03_performance_analysis.md](./A01_B03_performance_analysis.md)
5. **想了解Blackwell架构** → [A01_B04_blackwell_architecture.md](./A01_B04_blackwell_architecture.md)
6. **想学习量化** → [A01_B05_quantization_analysis.md](./A01_B05_quantization_analysis.md)
7. **想制定学习路径** → [A01_B06_learning_path.md](./A01_B06_learning_path.md)
8. **想了解整体问题** → [A01_glm_blackwell_optimization.md](./A01_glm_blackwell_optimization.md) ⭐ **主文档**

---

## 📁 其他文件

### 代码相关
- `code/` - 优化代码示例（如需要）

### 测试相关
- `test/` - 性能测试脚本（如需要）

---

## 📝 文档命名规则

- **A01_xxx.md**: 主文档（A01 系列）
- **A01_B01_xxx.md**: A01 的平行文档（A01_B01 系列）

---

## 🔗 Issue 链接

https://github.com/sgl-project/sglang/issues/17526

## 🎯 优化目标

**主要目标**: GLM 4.7 FP4在Blackwell GPU上的性能优化，重点关注延迟场景。

**关键优化项**:
1. GLM 4.7 + NVFP4 + MTP (#17166) - 10%性能提升
2. Auto-enable TRT-LLM MHA (#16755) - 10%性能提升
3. FP8 KV buffer kernel融合 - 3%性能提升
4. 改进scaled_fp4_quant - 1-2%性能提升
5. 更好的scale layout - 5-10%性能提升
6. FlashinferFP4MoE融合 - 待评估

## 📊 性能对比

### FP8 vs BF16
- **FP8 attention + FP8 KV cache**: 4886 token/s, Accuracy: 96.4%
- **BF16 attention + BF16 KV cache**: 6017 token/s, Accuracy: 97.0%

### MTP性能
- **MTP + FP8 KV cache**: 272.60 token/s
- **MTP + BF16 KV cache**: 283.93 token/s

### 瓶颈分析
FP8 KV cache的额外量化操作导致约23us的开销，包括：
- DeviceGemmFp4GemmSm100
- cvt_fp16_to_fp4
- float8_copy_kernel_cuda
- _fused_fp8_set_kv_buffer_kernel
