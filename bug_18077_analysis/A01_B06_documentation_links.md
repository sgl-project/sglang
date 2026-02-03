# A01_B06: 相关文档链接

## 📚 SGLang 官方文档

### 主要文档网站
- **SGLang 官方文档**: https://docs.sglang.ai/
- **GitHub 仓库**: https://github.com/sgl-project/sglang

### SGLang-D (Diffusion) 相关文档

#### 开发指南
- [SGLang-D Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/sgl_diffusion_en.md) ⭐ **重要参考**
- [Contributing to SGLang Diffusion](../python/sglang/multimodal_gen/docs/contributing.md)

#### 模型支持
- [Diffusion Models Documentation](../docs/supported_models/diffusion_models.md)
- GLM-Image 相关配置和实现

### 性能分析文档
- [Benchmark and Profiling](../docs/developer_guide/benchmark_and_profiling.md)
- [Bench Serving Guide](../docs/developer_guide/bench_serving.md)

## 🔍 代码位置

### GLM-Image 实现

#### Pipeline 配置
- `python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py`
  - `GlmImagePipelineConfig` - Pipeline 配置类

#### Pipeline 实现
- `python/sglang/multimodal_gen/runtime/pipelines/glm_image.py`
  - `GlmImagePipeline` - 主要 pipeline 实现

#### Transformer 模型
- `python/sglang/multimodal_gen/runtime/models/dits/glm_image.py`
  - `GlmImageTransformer2DModel` - DiT 实现

#### 模型阶段
- `python/sglang/multimodal_gen/runtime/models/model_stages/glm_image.py`
  - `GlmImageBeforeDenoisingStage` - 去噪前处理阶段

### 相关模块

#### VAE
- `python/sglang/multimodal_gen/runtime/models/vaes/`
- `python/sglang/multimodal_gen/configs/models/vaes/glmimage.py`

#### 调度器
- `python/sglang/multimodal_gen/runtime/schedulers/`

#### 注意力
- `python/sglang/srt/layers/dp_attention/`

### 基准测试工具
- `python/sglang/bench_serving.py` - 服务基准测试
- `python/sglang/multimodal_gen/benchmarks/` - Diffusion 基准测试

## 🔗 外部资源

### GLM-Image 相关
- **Hugging Face**: https://huggingface.co/zai-org/GLM-Image
- **论文/文档**: 需要查找 GLM-Image 的官方文档

### Sequence Parallelism
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
- **相关论文**: 查找 SP 相关研究论文

### 性能优化
- **Flash Attention**: https://github.com/Dao-AILab/flash-attention
- **PyTorch Profiler**: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

## 📖 推荐阅读顺序

### 1. 入门 (了解问题)
1. [A01_B01: 原始 Issue](./A01_B01_original_issue.md) - 了解问题背景
2. [A01: GLM-Image 性能问题详解](./A01_glm_image_performance.md) - 整体问题分析

### 2. 深入 (理解实现)
1. [SGLang-D Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/sgl_diffusion_en.md) - 理解 SGLang-D 架构
2. [A01_B03: 代码分析](./A01_B03_code_analysis.md) - GLM-Image 代码实现

### 3. 分析 (性能分析)
1. [A01_B02: 性能分析](./A01_B02_performance_analysis.md) - 性能基准测试
2. [Benchmark and Profiling](../docs/developer_guide/benchmark_and_profiling.md) - 性能分析工具

### 4. 优化 (实施方案)
1. [A01_B04: 优化方案](./A01_B04_optimization_proposals.md) - 优化建议
2. [A01_B05: 基准测试设置](./A01_B05_benchmark_setup.md) - 测试脚本

## 🛠️ 工具和资源

### 开发工具
- **PyTorch Profiler**: 性能分析
- **nvidia-smi**: GPU 监控
- **tensorboard**: 可视化（如果适用）

### 参考实现
- **Diffusers GLM-Image**: Hugging Face Diffusers 库中的实现
- **其他 SGLang-D 模型**: 查看其他已优化的模型实现

## 📝 Issue 和 PR

### 相关 Issue
- [Issue #18077](https://github.com/sgl-project/sglang/issues/18077) - 当前 issue

### 相关 PR
- 待添加相关 PR 链接

---

**最后更新**: 创建文档时
