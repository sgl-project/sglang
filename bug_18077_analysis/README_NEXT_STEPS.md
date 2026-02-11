# Issue #18077 下一步行动快速参考

## 📋 当前状态

### ✅ 已完成
- [x] 基准测试脚本开发
- [x] SGLang-D vs Diffusers 性能对比
- [x] 单 GPU vs 多 GPU 性能对比
- [x] 问题定位：SP 支持缺失

### 🔧 进行中
- [ ] 基准测试结果 PR

### ⏳ 待开始
- [ ] GLM-Image SP 支持实现

---

## 🎯 关键发现

### 性能表现
- **SGLang-D 比 Diffusers 快 8-13%** ✅
- **多 GPU 扩展性良好**（2 GPU 加速 1.19x-1.27x）✅
- **并发性能瓶颈**：concurrency=1→2 时吞吐量不增加 ⚠️

### 核心问题
- **SP 支持缺失**：GLM-Image 无法使用 Sequence Parallelism
- **错误原因**：Latent 格式不匹配
  - GLM-Image: `[B, C, H, W]` (4D)
  - 基类期望: `[B, H*W, C]` (3D)

---

## 📝 下一步行动

### 阶段一：基准测试 PR（1-2 天）

**任务**：
1. 整理测试报告（Markdown）
2. 提交 PR 展示性能数据

**文件**：
- 测试结果：`benchmark/results/`
- 报告模板：参考 Issue 中的格式

---

### 阶段二：SP 支持实现（3-5 天）

#### 任务 1: 实现 `shard_latents_for_sp()` (1-2 天)

**文件**：`python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py`

**要点**：
- 处理 4D 张量 `[B, C, H, W]`
- 在 H 或 W 维度上切分（选择较大的）
- 实现 padding 逻辑
- 存储元数据到 batch

**参考**：
- ZImage 实现：`zimage.py:174`
- 详细指南：`C02_SP_implementation_guide.md`

#### 任务 2: 实现 `gather_latents_for_sp()` (0.5 天)

**要点**：
- 在正确的维度上 gather
- 恢复完整形状

#### 任务 3: 测试验证 (1-2 天)

**测试场景**：
- [ ] 单 GPU（SP disabled）- 功能验证
- [ ] 多 GPU（SP enabled）- 不同分辨率/SP degree
- [ ] 正确性验证 - SP enabled/disabled 输出对比
- [ ] 性能验证 - 测量加速比

**测试命令**：
```bash
# Single GPU
sglang serve --model-path zai-org/GLM-Image --backend sglang

# Multi-GPU with SP
sglang serve --model-path zai-org/GLM-Image --backend sglang \
    --num-gpus 2 --sp-degree 2

# Benchmark
python bench_serving.py --resolution 1024x1024
```

#### 任务 4: 文档和 PR (0.5-1 天)

- [ ] 代码注释
- [ ] 更新文档
- [ ] 提交 PR

---

## 📚 相关文档

### 主要文档
- **C01**: [当前状态与下一步计划](./C01_current_status_and_next_steps.md) - 完整状态总结
- **C02**: [SP 实现技术指南](./C02_SP_implementation_guide.md) - 详细实现指南
- **B01**: [需求分析](./B01_original_issue_需求分析.md) - 原始需求拆解

### 代码位置
- GLM-Image Config: `python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py`
- 基类实现: `python/sglang/multimodal_gen/configs/pipeline_configs/base.py:721`
- SP 调用: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:781`
- 参考实现: `python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py:174`

### 外部资源
- [Issue #18077](https://github.com/sgl-project/sglang/issues/18077)
- [SGLang-D Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/sgl_diffusion_en.md)

---

## 🔍 技术要点速查

### GLM-Image Latent 格式
```
[B, C, H, W]
- B: Batch size
- C: Channels (16)
- H: Height (image_height // vae_scale_factor)
- W: Width (image_width // vae_scale_factor)
```

### SP 切分策略
1. 选择较大的维度（H 或 W）进行切分
2. Padding 使其可被 SP degree 整除
3. 在最终输出前移除 padding

### 实现检查清单
- [ ] `shard_latents_for_sp()` - 处理 4D 张量，实现切分和 padding
- [ ] `gather_latents_for_sp()` - 实现 gather 逻辑
- [ ] `post_denoising_loop()` - 处理 padding 移除（如需要）
- [ ] 单 GPU 测试 - 功能正确性
- [ ] 多 GPU 测试 - 不同分辨率/SP degree
- [ ] 正确性验证 - SP enabled/disabled 输出对比
- [ ] 性能验证 - 测量加速比

---

## ⚠️ 注意事项

1. **形状处理**：确保正确处理 4D 张量的切分和重组
2. **Padding 逻辑**：确保 padding/unpadding 正确，不影响最终输出
3. **正确性验证**：SP enabled/disabled 的输出应该一致（允许小的数值误差）
4. **性能影响**：验证 SP 通信开销是否合理，加速比是否达到预期

---

**最后更新**：2026-02-05
