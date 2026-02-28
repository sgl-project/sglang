# A01_B02: FLUX SP/TP 流程与 GLM 设计借鉴

## 相关文档
- [A01_B01: 原始 Issue 内容](./A01_B01_original_issue.md)
- [C02: GLM-Image SP 实现技术指南](./C02_SP_implementation_guide.md)
- 官方文档：[How to Support New Diffusion Models](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/how_to_support_new_diffusion_models.md)

---

## 一、与官方架构的对应关系

按「如何支持新 Diffusion 模型」的架构，SP/TP 落在不同层级：

| 官方组件 | SP/TP 职责 |
|----------|------------|
| **PipelineConfig** | `shard_latents_for_sp` / `gather_latents_for_sp`，`get_freqs_cis` 中的 RoPE 切分 |
| **ComposedPipeline** | 不改，沿用现有 stage 顺序 |
| **PipelineStage** | `DenoisingStage` 已按 `pipeline_config` 调用 shard/gather，无需改 stage 定义 |
| **Modules (dit/transformer)** | TP：`ColumnParallelLinear` 等；SP：`USPAttention` |

**结论**：GLM 的 SP 支持主要在 **PipelineConfig** 上做 override，无需改 Pipeline、Stage 或 Registry。

---

## 二、FLUX SP/TP 流程（在架构中的位置）

```
ComposedPipeline (FluxPipeline)
    │
    ├─ LatentPreparationStage → latents [B, S, D]
    │
    ├─ DenoisingStage
    │     ├─ _preprocess_sp_latents
    │     │     └─ pipeline_config.shard_latents_for_sp(batch, latents)  ← PipelineConfig
    │     ├─ transformer forward
    │     │     ├─ RoPE: shard_rotary_emb_for_sp (在 get_freqs_cis 中)   ← PipelineConfig
    │     │     ├─ Linear: ColumnParallelLinear (TP)                      ← Module
    │     │     └─ Attention: USPAttention (SP)                           ← Module
    │     └─ _postprocess_sp_latents
    │           └─ pipeline_config.gather_latents_for_sp(latents)         ← PipelineConfig
    │
    └─ DecodingStage
```

---

## 三、GLM 设计：FLUX vs GLM 差异与实现范围

| 项目 | FLUX | GLM-Image |
|------|------|-----------|
| **Config** | ImagePipelineConfig | SpatialImagePipelineConfig |
| **Latents** | [B, S, D] token | [B, C, H, W] 空间 |
| **shard** | 序列维 `rearrange("b (n s) d -> b n s d")` | 沿 H 维切分（SpatialImage 已有） |
| **DiT** | `ColumnParallelLinear` + USPAttention | `nn.Linear` + `GlmImageAttention`（无 TP/SP） |

**GLM 需补齐**（均在 PipelineConfig 层）：
1. 在 `GlmImagePipelineConfig` 中正确覆盖 `shard_latents_for_sp` / `gather_latents_for_sp`，保证 4D `[B,C,H,W]` 与 `SpatialImagePipelineConfig` 的 gather 维度一致。
2. 在 `get_freqs_cis` 中按 SP 切分 RoPE（GLM 返回 4D `[1,1,H,W]`，需与 latent shard 对齐）。

**可选**（Module 层）：
- TP：需在 `GlmImageTransformer2DModel` 中引入 `ColumnParallelLinear`（参考 FLUX DiT）。
- SP Attention：当前 `GlmImageAttention` 非 USPAttention，若要 head/sequence 维度 SP，需替换或扩展。

---

## 四、实现步骤（精炼）

1. **PipelineConfig**：在 `glm_image.py` 中覆盖 `shard_latents_for_sp` / `gather_latents_for_sp`（参考 `SpatialImagePipelineConfig` L773–806），并确保 `get_freqs_cis` 返回的 RoPE 按 SP 切分。
2. **不改**：`GlmImagePipeline`、`DenoisingStage`、Registry 均沿用现有逻辑。
3. **可选**：若需 TP，在 `runtime/models/dits/glm_image.py` 中把线性层替换为 `ColumnParallelLinear`。

---

## 五、参考位置

- **SP shard/gather**：`base.py` → `SpatialImagePipelineConfig` (L773–806)
- **Denoising 调用**：`denoising.py` → `_preprocess_sp_latents` / `_postprocess_sp_latents`
- **RoPE 切分**：`base.py` → `shard_rotary_emb_for_sp`
- **TP 线性层**：`linear.py` → `ColumnParallelLinear`
