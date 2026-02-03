# A01_B03: 代码分析

## 📁 GLM-Image 在 SGLang-D 中的实现

### 关键文件位置

#### 1. Pipeline 配置
- **文件**: `python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py`
- **类**: `GlmImagePipelineConfig`
- **功能**: GLM-Image pipeline 的配置，包括 VAE、DiT 等组件配置

#### 2. Pipeline 实现
- **文件**: `python/sglang/multimodal_gen/runtime/pipelines/glm_image.py`
- **类**: `GlmImagePipeline`
- **功能**: GLM-Image 的主要 pipeline 实现

#### 3. Transformer 模型
- **文件**: `python/sglang/multimodal_gen/runtime/models/dits/glm_image.py`
- **类**: `GlmImageTransformer2DModel`
- **功能**: GLM-Image 的 DiT (Diffusion Transformer) 实现

#### 4. 模型阶段
- **文件**: `python/sglang/multimodal_gen/runtime/models/model_stages/glm_image.py`
- **类**: `GlmImageBeforeDenoisingStage`
- **功能**: 去噪前的处理阶段

## 🔍 关键代码路径分析

### 1. Pipeline 流程

```python
# GlmImagePipeline 的主要阶段
1. GlmImageBeforeDenoisingStage
   - VAE 编码
   - 文本编码
   - 视觉语言编码
   
2. DenoisingStage
   - Transformer 前向传播
   - 调度器步进
   
3. DecodingStage
   - VAE 解码
```

### 2. Sequence Parallelism 支持情况

#### 当前配置
```python
# GlmImagePipelineConfig
vae_sp: bool = False  # VAE Sequence Parallelism 默认关闭
```

**问题**: `vae_sp` 默认为 `False`，这意味着 VAE 可能没有启用 Sequence Parallelism。

#### 需要检查的地方
1. **VAE 编码/解码**: 是否支持 SP
2. **Transformer 注意力**: 是否支持 SP
3. **图像 patch 处理**: 是否可以利用 SP

### 3. Transformer 前向传播

#### 关键代码 (GlmImageTransformer2DModel.forward)
```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    # ... 其他参数
):
    # 1. RoPE
    image_rotary_emb = freqs_cis
    
    # 2. Patch & Timestep embeddings
    hidden_states = self.image_projector(hidden_states)
    encoder_hidden_states = self.glyph_projector(encoder_hidden_states)
    
    # 3. Transformer blocks
    for block in self.blocks:
        hidden_states, encoder_hidden_states = block(...)
    
    # 4. Output norm & projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)
    
    # 5. Unpatchify
    return output
```

#### 潜在瓶颈
1. **注意力计算**: 每个 transformer block 中的注意力计算
2. **内存分配**: 中间激活的内存分配
3. **同步操作**: 跨 GPU 的同步操作（如果使用 TP）

### 4. 注意力实现

需要检查：
- 是否使用了优化的注意力内核（如 Flash Attention）
- 是否支持 Sequence Parallelism
- 内存访问模式是否高效

## 🔧 Sequence Parallelism 集成点

### 1. VAE 编码/解码
- **位置**: `GlmImageBeforeDenoisingStage` 和 `DecodingStage`
- **方法**: 将图像的空间维度分割到多个 GPU
- **挑战**: 需要处理 patch 级别的分割

### 2. Transformer 注意力
- **位置**: `GlmImageTransformer2DModel` 的注意力层
- **方法**: 将序列维度（patch 序列）分割到多个 GPU
- **挑战**: 需要处理 RoPE 和位置编码

### 3. 图像 Patch 处理
- **位置**: Patch embedding 和 unpatchify
- **方法**: 在 patch 级别进行分割
- **挑战**: 需要保持空间局部性

## 📊 性能优化机会

### 1. 内存管理
- **当前**: 可能使用了过多的中间激活
- **优化**: 使用梯度检查点或激活重计算

### 2. 内核优化
- **当前**: 可能使用了通用的 PyTorch 操作
- **优化**: 使用自定义 CUDA 内核或优化的库（如 Flash Attention）

### 3. 批处理优化
- **当前**: 批处理可能不够高效
- **优化**: 改进批处理策略，减少填充开销

## 🔗 相关代码

### 需要深入分析的模块
1. **注意力实现**: `sglang/srt/layers/dp_attention/`
2. **VAE 实现**: `sglang/multimodal_gen/runtime/models/vaes/`
3. **调度器**: `sglang/multimodal_gen/runtime/schedulers/`

### 参考实现
- Diffusers 的 GLM-Image 实现
- 其他支持 SP 的扩散模型实现

---

**状态**: 🚧 进行中 - 需要进一步代码审查和性能分析
