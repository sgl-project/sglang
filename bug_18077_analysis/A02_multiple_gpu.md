# A02 多卡 SP=2 心得与分步测试

## 相关文档
- [A01_B02: FLUX SP/TP 与 GLM 设计借鉴](./A01_B02_flux_sp_tp_and_glm_design.md)
- [C02: GLM-Image SP 实现技术指南](./C02_SP_implementation_guide.md)

---

## 一、单卡阶段小结（已跑通）

- **形状**：单卡下整条链路为 **4D** `[B, C, H, W]`。  
  - `GlmImageBeforeDenoisingStage.prepare_latents` → `(1, 16, 64, 64)`（512×512 时）  
  - `DenoisingStage` → `latent_model_input.shape=(1, 16, 64, 64)`  
  - `GlmImageTransformer2DModel.forward` → `hidden_states.shape=(1, 16, 64, 64) dim=4`
- **验证方式**：通过 A03 的 `[A03_GLM]` 分层日志在服务端 stdout 逐层确认，请求返回 **HTTP 200** 且能保存 `.jpg`。
- **结论**：单卡下 `SpatialImagePipelineConfig` 的 4D 逻辑与 GLM DiT 一致，无需改 3D 序列。

---

## 二、多卡 SP=2 目标与检查点

- **目标**：在 **tp=1、sp=2** 下，用与 A03 相同的「一步一步确认」方式，验证 SP 切分与 gather 正确。
- **预期**（以 512×512 为例，latent 高 64、宽 64）：
  - **Shard 后**（每个 SP rank）：`batch.latents` / `latent_model_input` / `hidden_states` 为 `(1, 16, 32, 64)`（沿 H 切一半）。
  - **Gather 后**：恢复为 `(1, 16, 64, 64)`，再进 VAE 解码。
- **检查方式**：多卡服务端 stdout 中搜 **`[A03_GLM]`**，与单卡对比：
  - `GlmImageBeforeDenoisingStage` 出口：单卡 `(1,16,64,64)` → 多卡 **先** 仍是 `(1,16,64,64)`（BeforeDenoising 在 rank0 跑，latents 尚未按 SP 切分）。
  - `DenoisingStage._preprocess_sp_latents AFTER`：应出现 **per-rank** 的 `(1, 16, 32, 64)`（或带 padding 的等价形状）。
  - `_predict_noise_with_cfg` / `GlmImageTransformer2DModel.forward`：每 rank 的 `hidden_states.shape` 为 `(1, 16, 32, 64)`。
- **常见问题**：
  - 若出现 **500 + `rearrange` 相关报错**：多半是某处仍按 3D 序列切分，需确认走的是 `SpatialImagePipelineConfig` 的 4D 分支。
  - 若 **出图花屏/全噪声**：可能是 RoPE（`get_freqs_cis`）未按 SP 切分，或 gather 维度不一致。
  - 若 **VAE 解码报错**：检查 `maybe_unpad_latents` 是否在 gather 后正确去掉 padding。

---

## 三、多卡测试流程（与 A03 一致：一步一步）

1. **启动多卡服务**（2 GPU，tp=1，sp=2）  
   ```bash
   ./A01_multiple_gpu.sh
   ```
2. **另一终端跑分步请求**  
   ```bash
   ./A03_multiple_gpu.sh
   ```
3. **看服务端 stdout**  
   - 搜 **`[A03_GLM]`**，按顺序确认：BeforeDenoising → Denoising START → _preprocess_sp_latents BEFORE/AFTER → _predict_noise_with_cfg → GlmImageTransformer2DModel.forward。  
   - 多卡下每个 rank 都会打日志，同一时刻两条 `[A03_GLM]` 可能来自 rank0 和 rank1；关注 **shape 是否为 (1,16,32,64)**。  
4. **若 500**：最后一条 `[A03_GLM]` 之后的 traceback 即出错层；再结合 C02 / A01_B02 对照 shard/gather 与 RoPE。

---

## 四、脚本对应关系

| 脚本 | 作用 |
|------|------|
| **A01_multiple_gpu.sh** | 启动 2 卡服务，`--tp-size 1 --sp-degree 2`，端口 30000 |
| **A03_multiple_gpu.sh** | 与 A03 同结构：health + pipeline 校验 + 单次 `/v1/images/generations`，便于在服务端用 `[A03_GLM]` 逐层确认 |

单卡仍用：`A01_single_gpu.sh` + `A03_test_step_by_step.sh`。
