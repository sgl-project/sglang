# C01: Issue #18077 当前状态与下一步计划

## 文档说明

本文档整理 Issue #18077 的**当前进展状态**、**已发现的问题**和**下一步行动计划**。

**相关文档**：
- [B01: 需求分析](./B01_original_issue_需求分析.md) - 原始需求拆解
- [B02: 测试设计](./B02_test_design.md) - 测试脚本设计
- [Issue #18077](https://github.com/sgl-project/sglang/issues/18077) - GitHub Issue

---

## 一、当前状态总结

### 1.1 基准测试结果 ✅

**已完成的工作**：
- ✅ 建立了可复现的基准测试脚本（`bench_serving.py`）
- ✅ 完成了 SGLang-D vs Diffusers 的性能对比测试
- ✅ 完成了单 GPU vs 多 GPU 的性能对比测试

**关键发现**：
1. **SGLang-D 实际上比 Diffusers 更快**（8-13% 的性能提升）
   - 这与原始假设（SGLang-D 更慢）相反
   - 说明 SGLang-D 的基础实现已经相当优化

2. **多 GPU 扩展性良好**
   - 2 GPU 相比单 GPU 有 1.19x - 1.27x 的加速
   - 显存使用合理

3. **并发性能瓶颈**
   - 从 concurrency=1 → 2 时，吞吐量几乎不增加
   - 延迟大致翻倍，说明请求被串行化处理
   - 可能的原因：缺少批处理或 GPU 利用率已饱和

### 1.2 测试结果数据

#### SGLang-D vs Diffusers (单 GPU)

| 分辨率 | 并发数 | Latency Mean (s) | Throughput (req/s) | Memory (GB) |
|--------|--------|------------------|-------------------|-------------|
|        |        | SGLang | Diffusers | SGLang | Diffusers | SGLang | Diffusers |
| 1024×1024 | 1 | 85.08 | 96.04 | 0.12 | 0.10 | - | - |
| 1024×1024 | 2 | 162.33 | 182.80 | 0.12 | 0.11 | - | - |
| 512×512 | 1 | 29.53 | 31.97 | 0.34 | 0.31 | - | - |
| 512×512 | 2 | 55.78 | 60.81 | 0.36 | 0.33 | - | - |

**结论**：SGLang-D 在所有测试场景下都优于 Diffusers。

#### 单 GPU vs 多 GPU (SGLang-D)

| 分辨率 | 并发数 | Latency Mean (s) | Throughput (req/s) |
|--------|--------|------------------|-------------------|
|        |        | Single | Multi (2 GPU) | Single | Multi (2 GPU) |
| 1024×1024 | 1 | 85.08 | 76.87 | 0.12 | 0.13 |
| 512×512 | 1 | 29.53 | 24.86 | 0.34 | 0.40 |

**结论**：多 GPU 扩展性良好，但仍有优化空间。

---

## 二、发现的关键问题

### 2.1 Sequence Parallelism (SP) 支持缺失 ❌

**问题描述**：
- GLM-Image 目前**不支持 Sequence Parallelism (SP)**
- 当尝试使用 `sp_degree=2` 时，会触发错误

**错误信息**：
```
einops.EinopsError: Error while processing rearrange-reduction pattern "b (n s) d -> b n s d".
 Input tensor shape: torch.Size([1, 16, 64, 64]). Additional info: {'n': 2}.
 Wrong shape: expected 3 dims. Received 4-dim tensor.
```

**根本原因**：
1. **Latent 格式不匹配**：
   - GLM-Image 使用 `[B, C, H, W]` 格式（4D 张量）
   - 基类 `ImagePipelineConfig.shard_latents_for_sp()` 期望 `[B, H*W, C]` 格式（3D 张量）

2. **缺少专门的实现**：
   - `GlmImagePipelineConfig` 继承自 `ImagePipelineConfig`
   - 但没有重写 `shard_latents_for_sp()` 方法
   - 因此回退到基类实现，导致形状不匹配

**代码位置**：
- 基类实现：`python/sglang/multimodal_gen/configs/pipeline_configs/base.py:721`
- GLM-Image 配置：`python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py:16`
- 调用位置：`python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:790`

**影响**：
- 无法使用 SP 进行高分辨率图像生成
- 限制了 GLM-Image 在高分辨率场景下的可扩展性
- 这是原始 Issue 中提到的核心问题之一

### 2.2 参考实现

其他模型已经实现了专门的 SP 支持：
- **ZImage**：`python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py:174`
  - 处理 5D 张量 `[B, C, T, H, W]`
  - 在空间维度（H 或 W）上进行切分
- **LTX-2**：`python/sglang/multimodal_gen/configs/pipeline_configs/ltx_2.py:287`
  - 处理 3D 打包 token 格式 `[B, S, D]`
  - 在时间维度（帧）上进行切分

---

## 三、下一步行动计划

根据 @zhaochenyang20 的建议，分为两个阶段：

### 阶段一：合并基准测试结果 PR ✅ (进行中)

**目标**：
- 将基准测试结果整理成报告
- 提交 PR 展示性能对比数据

**任务清单**：
- [x] 完成基准测试脚本
- [x] 收集性能数据
- [ ] 整理测试报告（Markdown 格式）
- [ ] 提交 PR 到主仓库

**预期交付物**：
- 详细的性能对比报告
- 可复现的测试脚本
- 测试结果数据（JSON/Markdown）

### 阶段二：修复 SP 支持问题 🔧 (待开始)

**目标**：
- 为 GLM-Image 实现正确的 SP 支持
- 支持高分辨率图像生成

**任务清单**：

#### 3.1 实现 `shard_latents_for_sp()` 方法

**需要实现的功能**：
1. 处理 4D 张量 `[B, C, H, W]`
2. 在空间维度（H 或 W）上进行切分
3. 处理 padding 以确保可被 SP degree 整除
4. 返回切分后的张量和标志位

**实现思路**：
```python
def shard_latents_for_sp(self, batch, latents):
    """
    Shard GLM-Image latents [B, C, H, W] for Sequence Parallelism.
    
    Args:
        batch: Request batch
        latents: Tensor of shape [B, C, H, W]
    
    Returns:
        sharded_latents: Sharded tensor
        did_shard: bool indicating if sharding was performed
    """
    sp_world_size = get_sp_world_size()
    if sp_world_size <= 1:
        return latents, False
    
    # GLM-Image latents are [B, C, H, W]
    B, C, H, W = latents.shape
    
    # Decide which dimension to shard (H or W)
    # Typically shard on the larger dimension for better load balance
    if H >= W:
        # Shard on H dimension
        # Pad H to be divisible by sp_world_size
        if H % sp_world_size != 0:
            pad_h = sp_world_size - (H % sp_world_size)
            pad = torch.zeros(B, C, pad_h, W, dtype=latents.dtype, device=latents.device)
            latents = torch.cat([latents, pad], dim=2)
            H = H + pad_h
        
        # Shard: [B, C, H, W] -> [B, C, H//sp, W] per rank
        rank = get_sp_parallel_rank()
        h_per_rank = H // sp_world_size
        h_start = rank * h_per_rank
        h_end = (rank + 1) * h_per_rank
        sharded = latents[:, :, h_start:h_end, :]
        
        # Store original shape for unpad
        batch.raw_latent_shape = (B, C, H, W)
        batch.sp_shard_dim = 'H'
        
    else:
        # Shard on W dimension (similar logic)
        ...
    
    return sharded, True
```

#### 3.2 实现 `gather_latents_for_sp()` 方法

**需要实现的功能**：
1. 收集所有 SP rank 的切分结果
2. 恢复原始形状 `[B, C, H, W]`
3. 移除 padding

**实现思路**：
```python
def gather_latents_for_sp(self, latents):
    """
    Gather sharded GLM-Image latents from all SP ranks.
    
    Args:
        latents: Sharded tensor from current rank
    
    Returns:
        gathered_latents: Full tensor [B, C, H, W]
    """
    sp_world_size = get_sp_world_size()
    if sp_world_size <= 1:
        return latents
    
    # Gather along the sharded dimension
    # Use sequence_model_parallel_all_gather with appropriate dim
    gathered = sequence_model_parallel_all_gather(latents, dim=2)  # or dim=3 for W
    
    return gathered
```

#### 3.3 更新 `post_denoising_loop()` 方法

**需要处理**：
- 在 denoising 后移除 SP padding
- 确保最终输出形状正确

#### 3.4 测试验证

**测试场景**：
1. **单 GPU 测试**（SP disabled）：
   - 验证功能正确性
   - 确保性能不受影响

2. **多 GPU 测试**（SP enabled）：
   - 测试不同分辨率：512×512, 1024×1024, 2048×2048
   - 测试不同 SP degree：2, 4
   - 验证输出图像质量（与单 GPU 结果对比）

3. **正确性验证**：
   - 对比 SP enabled/disabled 的输出结果
   - 确保数值精度一致

4. **性能验证**：
   - 测量 SP 带来的加速比
   - 验证高分辨率场景下的扩展性

**测试命令示例**：
```bash
# Single GPU (baseline)
sglang serve --model-path zai-org/GLM-Image --backend sglang

# Multi-GPU with SP
sglang serve --model-path zai-org/GLM-Image --backend sglang \
    --num-gpus 2 --sp-degree 2

# Benchmark
python bench_serving.py --resolution 1024x1024 --concurrency 1
```

---

## 四、技术细节

### 4.1 GLM-Image Latent 格式

**格式**：`[B, C, H, W]`
- `B`: Batch size
- `C`: Channel (16 for GLM-Image)
- `H`: Height (image_height // vae_scale_factor)
- `W`: Width (image_width // vae_scale_factor)

**来源**：
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/glm_image.py:547-552`

### 4.2 SP 切分策略

**选择切分维度**：
- 优先选择较大的维度（H 或 W）进行切分
- 这样可以更好地平衡各 GPU 的负载

**Padding 策略**：
- 确保切分维度可以被 SP degree 整除
- 在切分维度末尾添加 padding
- 在最终输出前移除 padding

### 4.3 与其他模型的对比

| 模型 | Latent 格式 | SP 切分维度 | 实现位置 |
|------|------------|------------|---------|
| GLM-Image | `[B, C, H, W]` | H 或 W | **待实现** |
| ZImage | `[B, C, T, H, W]` | H 或 W (空间) | `zimage.py:174` |
| LTX-2 | `[B, S, D]` (packed) | 时间（帧） | `ltx_2.py:287` |
| QwenImage | `[B, H*W, C]` (packed) | 序列 | `base.py:721` |

---

## 五、时间线估计

### 阶段一：基准测试 PR
- **预计时间**：1-2 天
- **状态**：进行中

### 阶段二：SP 支持实现
- **预计时间**：3-5 天
  - 实现 `shard_latents_for_sp()`: 1-2 天
  - 实现 `gather_latents_for_sp()`: 0.5 天
  - 测试和调试: 1-2 天
  - 文档和 PR: 0.5-1 天

---

## 六、风险与挑战

### 6.1 技术风险

1. **形状处理复杂性**：
   - 需要正确处理 4D 张量的切分和重组
   - 需要确保 padding/unpadding 逻辑正确

2. **性能影响**：
   - SP 通信开销可能影响性能
   - 需要验证多 GPU 加速比是否合理

3. **正确性验证**：
   - 需要确保 SP enabled/disabled 的输出一致
   - 可能需要数值精度对比

### 6.2 缓解措施

1. **参考现有实现**：
   - 学习 ZImage 和 LTX-2 的实现
   - 复用成熟的切分和收集逻辑

2. **充分测试**：
   - 覆盖不同分辨率和 SP degree
   - 进行正确性和性能双重验证

3. **渐进式实现**：
   - 先实现基本功能
   - 再优化性能和边界情况

---

## 七、相关资源

### 7.1 代码位置

- GLM-Image Pipeline Config: `python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py`
- Base ImagePipelineConfig: `python/sglang/multimodal_gen/configs/pipeline_configs/base.py:709`
- SP 调用位置: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:781`
- 参考实现（ZImage）: `python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py:174`

### 7.2 文档资源

- [Issue #18077](https://github.com/sgl-project/sglang/issues/18077)
- [SGLang-D Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/code-walk-through/sgl_diffusion_en.md)
- [SGLang-Diffusion Blog Post](https://lmsys.org/blog/2026-01-16-sglang-diffusion/)

### 7.3 相关 Issue/PR

- [PR #18154](https://github.com/sgl-project/sglang/pull/18154): Add offline throughput benchmark script for multi-modal models

---

## 八、总结

### 8.1 当前状态

✅ **已完成**：
- 基准测试脚本和性能对比
- 问题定位（SP 支持缺失）

🔧 **进行中**：
- 基准测试结果 PR

⏳ **待开始**：
- SP 支持实现

### 8.2 关键发现

1. **性能表现**：SGLang-D 已经比 Diffusers 更快，说明基础实现良好
2. **核心问题**：SP 支持缺失限制了高分辨率扩展性
3. **解决方案**：需要为 GLM-Image 实现专门的 SP 切分逻辑

### 8.3 下一步行动

1. **立即行动**：完成基准测试结果 PR
2. **短期目标**：实现 GLM-Image SP 支持
3. **长期目标**：验证高分辨率场景下的性能和扩展性

---

**最后更新**：2026-02-05  
**维护者**：@haojin2, @Nickcp39, @zhaochenyang20
