# Issue #18077 分析文档

## 📋 文档结构

### A01: GLM-Image 推理效率优化详解
- **[A01_glm_image_performance.md](./A01_glm_image_performance.md)** ⭐ **主文档**
  - GLM-Image 在 SGLang-D 中的性能问题
  - 与 Diffusers 的性能对比
  - 优化方向分析

### A02: FLUX.2-Klein 测试设计方案
- **[A02_flux_klein_test_design.md](./A02_flux_klein_test_design.md)** 🧪 **测试设计**
  - 测试模型确定（FLUX.2-Klein）
  - 完整的测试设计方案
  - 测试脚本和报告结构

### A01_Bxx: 平行文档（与 A01 同级）

- **[A01_B01_original_issue.md](./A01_B01_original_issue.md)** ⭐ **原始 Issue 内容**
  - Issue 链接和标题
  - 问题描述和性能数据
  - 目标和任务清单

- **[A01_B02_performance_analysis.md](./A01_B02_performance_analysis.md)** ⭐ **性能分析**
  - 性能基准测试结果
  - SGLang vs Diffusers 对比
  - 瓶颈识别

- **[A01_B03_code_analysis.md](./A01_B03_code_analysis.md)** ⭐ **代码分析**
  - GLM-Image 在 SGLang-D 中的实现
  - 关键代码路径分析
  - Sequence Parallelism 支持情况

- **[A01_B04_optimization_proposals.md](./A01_B04_optimization_proposals.md)** ⭐ **优化方案** 🔍 **重点**
  - Sequence Parallelism 集成方案
  - 内存管理优化
  - 内核优化建议

- **[A01_B05_benchmark_setup.md](./A01_B05_benchmark_setup.md)** ⭐ **基准测试设置**
  - 可复现的基准测试脚本
  - 测试配置和环境
  - 结果收集方法

- **[A01_B06_documentation_links.md](./A01_B06_documentation_links.md)** 📚 **相关文档链接**
  - SGLang-D 文档
  - GLM-Image 相关代码
  - 参考资源

- **[A01_B07_model_size_analysis.md](./A01_B07_model_size_analysis.md)** 💾 **模型大小分析**
  - Wan2.1-T2V-1.3B 显存需求
  - RTX 4090 运行可行性
  - 优化建议

- **[A01_B08_supported_models_and_minimum_gpu.md](./A01_B08_supported_models_and_minimum_gpu.md)** 📋 **支持的模型和最小 GPU 需求**
  - SGLang-D 官方支持的 Diffusion 模型列表
  - 最小 GPU 需求的 DiT 模型分析
  - 显存需求对比

---

## 🎯 快速导航

1. **想了解问题详情** → [A01_B01_original_issue.md](./A01_B01_original_issue.md) ⭐ **从这里开始**
2. **想阅读相关文档** → [A01_B06_documentation_links.md](./A01_B06_documentation_links.md) 📚 **文档链接**
3. **想了解性能分析** → [A01_B02_performance_analysis.md](./A01_B02_performance_analysis.md) 🔍 **重点推荐**
4. **想查看代码实现** → [A01_B03_code_analysis.md](./A01_B03_code_analysis.md)
5. **想了解优化方案** → [A01_B04_optimization_proposals.md](./A01_B04_optimization_proposals.md) 🎓 **核心推荐**
6. **想设置基准测试** → [A01_B05_benchmark_setup.md](./A01_B05_benchmark_setup.md)
7. **想了解模型大小** → [A01_B07_model_size_analysis.md](./A01_B07_model_size_analysis.md) 💾 **硬件需求**
8. **想了解支持的模型** → [A01_B08_supported_models_and_minimum_gpu.md](./A01_B08_supported_models_and_minimum_gpu.md) 📋 **模型列表**
9. **想了解测试设计** → [A02_flux_klein_test_design.md](./A02_flux_klein_test_design.md) 🧪 **测试方案**
10. **想了解整体问题** → [A01_glm_image_performance.md](./A01_glm_image_performance.md) ⭐ **主文档**

---

## 📁 其他文件

### 代码相关
- `code/` - 基准测试脚本和优化代码

### 测试相关
- `benchmark/` - 基准测试结果

---

## 📝 文档命名规则

- **A01_xxx.md**: 主文档（A01 系列）
- **A01_B01_xxx.md**: A01 的平行文档（A01_B01 系列）

---

## 🔗 Issue 链接

https://github.com/sgl-project/sglang/issues/18077

## 🎯 Issue 摘要

**问题**: GLM-Image 在 SGLang-D 引擎上的推理性能相比 Diffusers 实现存在显著差距。

**性能对比**:
- **Diffusers backend**: ~0.50 req/s, ~2.0s latency
- **SGLang backend**: ~0.11 req/s, ~9.2s latency

**主要问题**:
- 缺乏 Sequence Parallelism (SP) 支持
- 高分辨率图像生成效率低下
- 可能的内存管理问题

**目标**:
1. 建立性能基准（延迟、吞吐量、VRAM使用）
2. 识别瓶颈（注意力内核、内存开销等）
3. 提出优化方案（SP集成、内存管理等）
