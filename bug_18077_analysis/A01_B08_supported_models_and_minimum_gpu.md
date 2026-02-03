# A01_B08: 支持的 Diffusion 模型和最小 GPU 需求

## 📋 SGLang-D 官方支持的 Diffusion 模型

根据 `docs/supported_models/diffusion_models.md` 官方文档，SGLang-D 支持以下模型：

### Image Generation Models (图像生成)

| Model Name | HuggingFace Model ID | 参数量 | 架构类型 | 分辨率支持 |
|:-----------|:---------------------|:------|:---------|:----------|
| **FLUX.1-dev** | `black-forest-labs/FLUX.1-dev` | ~12B | DiT | Any resolution |
| **FLUX.2-dev** | `black-forest-labs/FLUX.2-dev` | ~12B | DiT | Any resolution |
| **FLUX.2-Klein** | `black-forest-labs/FLUX.2-klein-4B` | **4B** | DiT | Any resolution |
| **Z-Image-Turbo** | `Tongyi-MAI/Z-Image-Turbo` | ~7B | DiT | Any resolution |
| **GLM-Image** | `zai-org/GLM-Image` | ~7B | DiT | Any resolution |
| **Qwen Image** | `Qwen/Qwen-Image` | ~7B | DiT | Any resolution |
| **Qwen Image 2512** | `Qwen/Qwen-Image-2512` | ~7B | DiT | Any resolution |
| **Qwen Image Edit** | `Qwen/Qwen-Image-Edit` | ~7B | DiT | Any resolution |

### Video Generation Models (视频生成)

| Model Name | HuggingFace Model ID | 参数量 | 架构类型 | 分辨率支持 |
|:-----------|:---------------------|:------|:---------|:----------|
| **FastWan2.1 T2V 1.3B** | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | **1.3B** | DiT | 480p |
| **Wan2.1 T2V 1.3B** | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | **1.3B** | DiT | 480p |
| **Wan2.1 T2V 14B** | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 14B | DiT | 480p, 720p |
| **Wan2.1 I2V 480P** | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | 14B | DiT | 480p |
| **Wan2.1 I2V 720P** | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | 14B | DiT | 720p |
| **FastWan2.2 TI2V 5B** | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 5B | DiT | 720p |
| **Wan2.2 TI2V 5B** | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 5B | DiT | 720p |
| **Wan2.2 T2V A14B** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 14B | DiT | 480p, 720p |
| **Wan2.2 I2V A14B** | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 14B | DiT | 480p, 720p |
| **HunyuanVideo** | `hunyuanvideo-community/HunyuanVideo` | ~7B | DiT | 720×1280, 544×960 |
| **FastHunyuan** | `FastVideo/FastHunyuan-diffusers` | ~7B | DiT | 720×1280, 544×960 |

---

## 🎯 最小 GPU 需求的 DiT 模型分析

### 参数量对比

#### 图像生成模型（参数量从小到大）
1. **FLUX.2-Klein**: **4B** ⭐ **最小参数量**
2. Z-Image-Turbo: ~7B
3. GLM-Image: ~7B
4. Qwen Image 系列: ~7B
5. FLUX.1-dev / FLUX.2-dev: ~12B

#### 视频生成模型（参数量从小到大）
1. **FastWan2.1 T2V 1.3B**: **1.3B** ⭐ **最小参数量**
2. **Wan2.1 T2V 1.3B**: **1.3B** ⭐ **最小参数量**
3. FastWan2.2 TI2V 5B: 5B
4. Wan2.2 TI2V 5B: 5B
5. HunyuanVideo / FastHunyuan: ~7B
6. Wan2.1/Wan2.2 14B 系列: 14B

### 实际显存需求（基于测试数据）

#### 已知测试数据

| 模型 | 参数量 | 实际显存需求 | 测试环境 | 数据来源 |
|:-----|:-------|:------------|:---------|:---------|
| **Wan2.1-T2V-1.3B** | 1.3B | **> 23 GB** | RTX 4090 (24GB) | bug_17671_analysis |
| **Z-Image-Turbo** | ~7B | **接近 24GB** | RTX 4090 (24GB) | bug_17671_analysis |

#### 关键发现

⚠️ **重要**: 参数量小 ≠ 显存需求小

- **Wan2.1-T2V-1.3B** (1.3B 参数) 需要 **> 23GB 显存**
- 原因：DiT + T5 文本编码器组合，T5 编码器占用大量显存
- 即使参数量只有 1.3B，但由于架构特点，显存需求仍然很高

### 最小 GPU 需求的 DiT 模型推荐

#### 图像生成：FLUX.2-Klein (4B)

**推荐理由**：
- ✅ 参数量最小（4B，在图像生成模型中）
- ✅ 专门优化的轻量版本
- ⚠️ 但实际显存需求需要测试验证

**预估显存需求**：
- 由于是 DiT 架构，预计需要 **16-20GB** 显存（需要实际测试验证）
- 可能比 Wan2.1-T2V-1.3B 更节省显存（因为没有 T5 编码器）

#### 视频生成：FastWan2.1 T2V 1.3B 或 Wan2.1 T2V 1.3B

**推荐理由**：
- ✅ 参数量最小（1.3B）
- ⚠️ 但实际测试显示需要 **> 23GB 显存**

**实际显存需求**：
- **> 23GB**（基于实际测试）
- 需要 RTX 4090 (24GB) 或更高

---

## 💡 关于最小 GPU 需求的重要说明

### 为什么参数量小的模型显存需求仍然很大？

1. **DiT 架构特点**：
   - Diffusion Transformer 需要存储中间激活
   - 注意力机制需要大量显存
   - 高分辨率图像/视频需要更多显存

2. **文本编码器开销**：
   - T5 等大型文本编码器占用大量显存
   - 即使主模型参数量小，编码器可能很大

3. **VAE 编码/解码**：
   - VAE 也需要显存
   - 高分辨率时显存需求增加

### 实际建议

#### 对于 RTX 4090 (24GB) 用户

**可以尝试的模型**（按显存需求从低到高）：
1. **FLUX.2-Klein (4B)** - 图像生成，预估 16-20GB ⭐ **推荐**
2. **FastWan2.1 T2V 1.3B** - 视频生成，需要 > 23GB ⚠️ **紧张**
3. **Wan2.1-T2V-1.3B** - 视频生成，需要 > 23GB ⚠️ **紧张**

**不推荐的模型**（显存需求过高）：
- FLUX.1-dev / FLUX.2-dev (~12B)
- Wan2.1/Wan2.2 14B 系列
- Qwen Image 系列（可能需要更多显存）

#### 对于更小 GPU 的用户（< 24GB）

**建议**：
- 使用量化版本（如果有）
- 启用 CPU offload
- 使用较小的分辨率
- 考虑使用 UNet 架构的模型（如 SD-Turbo），而不是 DiT

---

## 📊 总结表格

### 最小 GPU 需求的 DiT 模型

| 任务类型 | 模型 | 参数量 | 预估显存需求 | RTX 4090 可行性 |
|:--------|:-----|:------|:------------|:---------------|
| **图像生成** | **FLUX.2-Klein** | **4B** | **16-20GB (预估)** | ✅ **推荐** |
| 图像生成 | Z-Image-Turbo | ~7B | ~24GB | ⚠️ 紧张 |
| 图像生成 | GLM-Image | ~7B | ~24GB (预估) | ⚠️ 紧张 |
| **视频生成** | **FastWan2.1 T2V 1.3B** | **1.3B** | **> 23GB** | ⚠️ **紧张** |
| 视频生成 | Wan2.1 T2V 1.3B | 1.3B | > 23GB | ⚠️ 紧张 |

### 关键结论

1. **参数量最小的 DiT 模型**：
   - 图像生成：**FLUX.2-Klein (4B)**
   - 视频生成：**FastWan2.1 T2V 1.3B** 或 **Wan2.1 T2V 1.3B (1.3B)**

2. **但显存需求最小的模型**：
   - 需要实际测试验证
   - **FLUX.2-Klein** 可能是最佳选择（没有 T5 编码器）

3. **对于 Issue #18077 的测试**：
   - 可以使用 **FLUX.2-Klein** 作为测试模型（如果关注图像生成）
   - 或使用 **FastWan2.1 T2V 1.3B**（如果关注视频生成）
   - 不需要强制使用 GLM-Image

---

## 🔗 相关文档

- [A01_B01: 原始 Issue](./A01_B01_original_issue.md) - Issue 详细内容
- [A01_B07: 模型大小分析](./A01_B07_model_size_analysis.md) - Wan2.1-T2V 显存需求
- [官方文档: Diffusion Models](../docs/supported_models/diffusion_models.md) - 完整模型列表

---

**最后更新**: 基于官方文档 `docs/supported_models/diffusion_models.md`
