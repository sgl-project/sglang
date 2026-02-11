# C02: GLM-Image SP 实现技术指南

## 文档说明

本文档提供 GLM-Image Sequence Parallelism (SP) 实现的**详细技术指南**，包括代码示例、实现要点和测试方法。

**目标读者**：实现 SP 支持的开发者

---

## 一、问题分析

### 1.1 错误信息

```
einops.EinopsError: Error while processing rearrange-reduction pattern "b (n s) d -> b n s d".
 Input tensor shape: torch.Size([1, 16, 64, 64]). Additional info: {'n': 2}.
 Wrong shape: expected 3 dims. Received 4-dim tensor.
```

### 1.2 根本原因

- **GLM-Image latents**: `[B, C, H, W]` (4D)
- **基类期望**: `[B, H*W, C]` (3D)
- **当前行为**: 回退到 `ImagePipelineConfig.shard_latents_for_sp()`，导致形状不匹配

### 1.3 解决方案

在 `GlmImagePipelineConfig` 中实现专门的 `shard_latents_for_sp()` 和 `gather_latents_for_sp()` 方法。

---

## 二、实现方案

### 2.1 文件位置

**需要修改的文件**：
```
python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py
```

**当前类结构**：
```python
@dataclass
class GlmImagePipelineConfig(ImagePipelineConfig):
    # ... existing fields ...
    
    # 需要添加的方法：
    # 1. shard_latents_for_sp()
    # 2. gather_latents_for_sp()
    # 3. 可能需要更新 post_denoising_loop() 来处理 padding
```

### 2.2 实现 `shard_latents_for_sp()`

#### 2.2.1 方法签名

```python
def shard_latents_for_sp(self, batch, latents):
    """
    Shard GLM-Image latents [B, C, H, W] for Sequence Parallelism.
    
    GLM-Image uses spatial latents format [B, C, H, W] instead of 
    token format [B, H*W, C] used by other image models.
    
    Args:
        batch: Request batch object
        latents: Tensor of shape [B, C, H, W]
    
    Returns:
        tuple: (sharded_latents, did_shard)
            - sharded_latents: Sharded tensor for current SP rank
            - did_shard: bool indicating if sharding was performed
    """
```

#### 2.2.2 实现逻辑

```python
def shard_latents_for_sp(self, batch, latents):
    from sglang.multimodal_gen.runtime.utils.parallel_utils import (
        get_sp_world_size,
        get_sp_parallel_rank,
    )
    
    sp_world_size = get_sp_world_size()
    if sp_world_size <= 1:
        return latents, False
    
    # GLM-Image latents are [B, C, H, W]
    if latents.dim() != 4:
        return latents, False
    
    B, C, H, W = latents.shape
    
    # Store original shape for later unpad
    batch.raw_latent_shape = (B, C, H, W)
    
    # Decide which dimension to shard (H or W)
    # Shard on the larger dimension for better load balance
    if H >= W:
        # Shard on H dimension
        shard_dim = 2  # H dimension
        dim_size = H
        
        # Pad H to be divisible by sp_world_size
        if H % sp_world_size != 0:
            pad_h = sp_world_size - (H % sp_world_size)
            pad = torch.zeros(
                B, C, pad_h, W,
                dtype=latents.dtype,
                device=latents.device
            )
            latents = torch.cat([latents, pad], dim=2)
            H = H + pad_h
            batch.raw_latent_shape = (B, C, H - pad_h, W)  # Store original H
        
        # Shard: [B, C, H, W] -> [B, C, H//sp, W] per rank
        rank = get_sp_parallel_rank()
        h_per_rank = H // sp_world_size
        h_start = rank * h_per_rank
        h_end = (rank + 1) * h_per_rank
        sharded = latents[:, :, h_start:h_end, :]
        
        # Store metadata for gather
        batch.sp_shard_dim = 'H'
        batch.sp_shard_size = h_per_rank
        
    else:
        # Shard on W dimension
        shard_dim = 3  # W dimension
        dim_size = W
        
        # Pad W to be divisible by sp_world_size
        if W % sp_world_size != 0:
            pad_w = sp_world_size - (W % sp_world_size)
            pad = torch.zeros(
                B, C, H, pad_w,
                dtype=latents.dtype,
                device=latents.device
            )
            latents = torch.cat([latents, pad], dim=3)
            W = W + pad_w
            batch.raw_latent_shape = (B, C, H, W - pad_w)  # Store original W
        
        # Shard: [B, C, H, W] -> [B, C, H, W//sp] per rank
        rank = get_sp_parallel_rank()
        w_per_rank = W // sp_world_size
        w_start = rank * w_per_rank
        w_end = (rank + 1) * w_per_rank
        sharded = latents[:, :, :, w_start:w_end]
        
        # Store metadata for gather
        batch.sp_shard_dim = 'W'
        batch.sp_shard_size = w_per_rank
    
    return sharded, True
```

#### 2.2.3 关键要点

1. **维度选择**：选择 H 或 W 中较大的维度进行切分，以平衡负载
2. **Padding**：确保切分维度可以被 SP degree 整除
3. **元数据存储**：在 batch 中存储原始形状和切分信息，用于后续的 gather 和 unpad

### 2.3 实现 `gather_latents_for_sp()`

#### 2.3.1 方法签名

```python
def gather_latents_for_sp(self, latents):
    """
    Gather sharded GLM-Image latents from all SP ranks.
    
    Args:
        latents: Sharded tensor from current rank [B, C, H_local, W] or [B, C, H, W_local]
    
    Returns:
        gathered_latents: Full tensor [B, C, H, W]
    """
```

#### 2.3.2 实现逻辑

```python
def gather_latents_for_sp(self, latents):
    from sglang.multimodal_gen.runtime.utils.parallel_utils import (
        get_sp_world_size,
        sequence_model_parallel_all_gather,
    )
    
    sp_world_size = get_sp_world_size()
    if sp_world_size <= 1:
        return latents
    
    # Gather along the sharded dimension
    # If sharded on H (dim=2), gather on dim=2
    # If sharded on W (dim=3), gather on dim=3
    # We need to determine which dimension was sharded
    # This information should be stored in batch, but we can infer from shape
    
    # For now, assume we gather on the dimension that was sharded
    # The actual dimension depends on the sharding strategy used in shard_latents_for_sp
    
    # Since we don't have direct access to batch here, we'll need to pass it
    # Or store the shard_dim in a way that's accessible
    
    # Option 1: Gather on dim=2 (H) - most common case
    if latents.dim() == 4:
        # Try to infer: if H is small relative to W, we likely sharded on W
        B, C, H, W = latents.shape
        # This is a heuristic - ideally we'd store shard_dim in batch
        # For now, gather on both dimensions and let the caller handle it
        # Or we can make this method take batch as parameter
        
        # Actually, looking at other implementations, gather is typically called
        # after we know which dimension was sharded. Let's make it explicit:
        gathered = sequence_model_parallel_all_gather(latents, dim=2)
        # If this doesn't work, we may need to gather on dim=3 instead
        # Or make the method signature take shard_dim as parameter
    
    return gathered
```

#### 2.3.3 改进版本（需要 batch 信息）

由于 `gather_latents_for_sp()` 需要知道切分维度，但当前签名没有 batch 参数，我们需要：

**方案 A**：修改方法签名（推荐）
```python
def gather_latents_for_sp(self, latents, batch=None):
    """
    Gather sharded GLM-Image latents from all SP ranks.
    
    Args:
        latents: Sharded tensor
        batch: Optional batch object containing shard metadata
    """
    sp_world_size = get_sp_world_size()
    if sp_world_size <= 1:
        return latents
    
    # Determine shard dimension from batch metadata
    if batch is not None and hasattr(batch, 'sp_shard_dim'):
        if batch.sp_shard_dim == 'H':
            gather_dim = 2
        else:  # 'W'
            gather_dim = 3
    else:
        # Fallback: infer from shape (heuristic)
        B, C, H, W = latents.shape
        gather_dim = 2 if H < W else 3  # Shard on larger dim, gather on same
    
    gathered = sequence_model_parallel_all_gather(latents, dim=gather_dim)
    return gathered
```

**方案 B**：在调用处处理（如果无法修改签名）
- 在 `denoising.py` 的 `_postprocess_sp_latents()` 中根据 batch 信息选择 gather 维度

### 2.4 更新 `post_denoising_loop()`

可能需要更新以处理 SP padding：

```python
def post_denoising_loop(self, latents, batch):
    # Clear KV caches if present
    if getattr(batch, "kv_caches", None) is not None:
        batch.kv_caches.clear()
    
    # Unpad if SP was used
    if getattr(batch, "did_sp_shard_latents", False) and hasattr(batch, "raw_latent_shape"):
        from sglang.multimodal_gen.configs.pipeline_configs.base import maybe_unpad_latents
        latents = maybe_unpad_latents(latents, batch)
    
    return latents.bfloat16()
```

---

## 三、参考实现

### 3.1 ZImage 实现（最相似）

**文件**：`python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py:174`

**特点**：
- 处理 5D 张量 `[B, C, T, H, W]`
- 在空间维度（H 或 W）上切分
- 使用 patch-based 切分策略

**关键代码片段**：
```python
def shard_latents_for_sp(self, batch, latents):
    sp_size = get_sp_world_size()
    if sp_size <= 1 or latents.dim() != 5:
        return latents, False

    plan = self._get_zimage_sp_plan(batch)
    
    # Layout: [B, C, T, H, W]. Always shard on dim=3 by optionally swapping H/W.
    if plan["swap_hw"]:
        latents = latents.transpose(3, 4).contiguous()
    
    # Pad on effective-H so that H_tok is divisible by sp.
    H_eff = latents.size(3)
    H_tok = H_eff // self.PATCH_SIZE
    pad_tok = plan["H_tok_pad"] - H_tok
    pad_lat = pad_tok * self.PATCH_SIZE
    if pad_lat > 0:
        pad = latents[:, :, :, -1:, :].repeat(1, 1, 1, pad_lat, 1)
        latents = torch.cat([latents, pad], dim=3)
    
    h0 = plan["h0_tok"] * self.PATCH_SIZE
    h1 = (plan["h0_tok"] + plan["H_tok_local"]) * self.PATCH_SIZE
    latents = latents[:, :, :, h0:h1, :]
    
    batch._zimage_sp_swap_hw = plan["swap_hw"]
    return latents, True
```

### 3.2 LTX-2 实现（处理打包格式）

**文件**：`python/sglang/multimodal_gen/configs/pipeline_configs/ltx_2.py:287`

**特点**：
- 处理 3D 打包 token 格式 `[B, S, D]`
- 在时间维度（帧）上切分

---

## 四、测试策略

### 4.1 单元测试

```python
def test_glm_image_sp_shard():
    """Test shard_latents_for_sp for GLM-Image"""
    from sglang.multimodal_gen.configs.pipeline_configs.glm_image import GlmImagePipelineConfig
    import torch
    
    config = GlmImagePipelineConfig()
    
    # Create mock latents [B=1, C=16, H=64, W=64]
    latents = torch.randn(1, 16, 64, 64)
    
    # Mock batch
    class MockBatch:
        pass
    batch = MockBatch()
    
    # Test with SP disabled
    sharded, did_shard = config.shard_latents_for_sp(batch, latents)
    assert not did_shard
    assert torch.equal(sharded, latents)
    
    # Test with SP enabled (requires actual SP setup)
    # This would need to be tested in a multi-GPU environment
```

### 4.2 集成测试

```bash
# Test 1: Single GPU (baseline)
sglang serve --model-path zai-org/GLM-Image --backend sglang \
    --resolution 512x512

# Test 2: Multi-GPU with SP
sglang serve --model-path zai-org/GLM-Image --backend sglang \
    --num-gpus 2 --sp-degree 2 \
    --resolution 1024x1024

# Test 3: High resolution with SP
sglang serve --model-path zai-org/GLM-Image --backend sglang \
    --num-gpus 4 --sp-degree 4 \
    --resolution 2048x2048
```

### 4.3 正确性验证

```python
# Compare outputs with SP enabled vs disabled
# They should produce identical (or very similar) results

def test_sp_correctness():
    # Generate image with SP disabled
    image_no_sp = generate_image(sp_degree=1)
    
    # Generate image with SP enabled
    image_with_sp = generate_image(sp_degree=2)
    
    # Compare (allowing for small numerical differences)
    assert torch.allclose(image_no_sp, image_with_sp, atol=1e-5)
```

### 4.4 性能测试

```bash
# Benchmark with different SP degrees
python bench_serving.py \
    --model zai-org/GLM-Image \
    --resolution 1024x1024 \
    --sp-degree 1,2,4 \
    --num-gpus 1,2,4
```

---

## 五、实现检查清单

### 5.1 代码实现

- [ ] 实现 `shard_latents_for_sp()` 方法
  - [ ] 处理 4D 张量 `[B, C, H, W]`
  - [ ] 选择切分维度（H 或 W）
  - [ ] 实现 padding 逻辑
  - [ ] 存储元数据到 batch

- [ ] 实现 `gather_latents_for_sp()` 方法
  - [ ] 在正确的维度上 gather
  - [ ] 处理元数据（如果方法签名允许）

- [ ] 更新 `post_denoising_loop()`（如需要）
  - [ ] 处理 SP padding 的移除

### 5.2 测试验证

- [ ] 单 GPU 测试（SP disabled）
  - [ ] 功能正确性
  - [ ] 性能不受影响

- [ ] 多 GPU 测试（SP enabled）
  - [ ] 不同分辨率：512×512, 1024×1024, 2048×2048
  - [ ] 不同 SP degree：2, 4
  - [ ] 输出图像质量验证

- [ ] 正确性验证
  - [ ] SP enabled/disabled 输出对比
  - [ ] 数值精度验证

- [ ] 性能验证
  - [ ] 测量加速比
  - [ ] 验证扩展性

### 5.3 文档和 PR

- [ ] 代码注释
- [ ] 更新相关文档
- [ ] 提交 PR 并添加测试结果

---

## 六、常见问题

### Q1: 如何选择切分维度（H 还是 W）？

**A**: 选择较大的维度进行切分，这样可以更好地平衡各 GPU 的负载。如果 H >= W，切分 H；否则切分 W。

### Q2: Padding 策略是什么？

**A**: 在切分维度末尾添加 padding，使其可以被 SP degree 整除。在最终输出前需要移除这些 padding。

### Q3: 如何验证实现的正确性？

**A**: 
1. 对比 SP enabled/disabled 的输出图像
2. 验证数值精度（允许小的浮点误差）
3. 测试不同分辨率和 SP degree

### Q4: 性能如何优化？

**A**:
1. 选择最优的切分维度（负载均衡）
2. 最小化 padding（减少通信开销）
3. 使用高效的 gather 操作

---

## 七、相关资源

- **基类实现**: `python/sglang/multimodal_gen/configs/pipeline_configs/base.py:721`
- **ZImage 参考**: `python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py:174`
- **调用位置**: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py:781`
- **并行工具**: `python/sglang/multimodal_gen/runtime/utils/parallel_utils.py`

---

**最后更新**：2026-02-05  
**维护者**：待实现者
