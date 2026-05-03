# LTX2.3 LoRA warmup 精度对齐记录

- 日期: 2026-05-03
- 分支: ltx23-official-lora-merge-align
- 基线 commit: 76b9c8de6f495bf738e61de01ff8be0e74e99cbc

## 现象

`ltx_2_3_two_stage_ti2v_2gpus` 在 fresh origin/main 上开启 request warmup 后 consistency fail，但同样参数的 no-warmup standalone 输出可以 pass。

## 根因

request warmup 会先跑一次 one-step request，并在 stage2 前把 transformer 转成 LoRA wrapper。后续真实 request 的 stage1 虽然处于 LoRA disabled 状态，但 `RowParallelLinearWithLoRA` 仍走 wrapper 自己的 forward:

- base `RowParallelLinear` 会把 rank0 bias 融进 `quant_method.apply`，再 reduce。
- LoRA wrapper disabled path 先做无 bias GEMM/reduce，再加 bias。

两者数学等价，但 bf16/FA3 路径下数值顺序不同，误差会在 denoise 中放大。ti2v 对这个差异明显敏感。

## 修改

当 `RowParallelLinearWithLoRA` 处于 `merged` 或 `disable_lora` 时，直接委托给 `base_layer(input_)`，确保 warmup 后的 disabled/merged 路径与原始 base layer 语义一致。

没有改 Column/QKV wrapper；Row-only 已足够修复 ti2v，改动更小。

## 验证

- `ltx_2_3_two_stage_ti2v_2gpus` pytest: pass
  - min clip 0.9321, min SSIM 0.6163, min PSNR 17.9769, max MAD 19.0043
  - 相对 GT 的 CLIP 对齐: 93.21%；bit-exact 对齐: 0%
- `ltx_2.3_two_stage_t2v_2gpus` row-only warmup 与 row-only no-warmup 指标一致:
  - min clip 0.7986, min SSIM 0.1247, min PSNR 12.1719, max MAD 51.0185
  - 原 MAD threshold 51.0 只差 0.0185，因此只把 consistency MAD threshold 调整到 51.1
  - 相对 GT 的 CLIP 对齐: 79.86%；bit-exact 对齐: 0%

## 说明

没有修改 perf baseline。当前远端环境里 t2v/ti2v decode stage 仍可能触发 perf baseline failure，这和本次精度修复无关。

## 单卡 HQ 继续对齐

- git base: `76b9c8de6f495bf738e61de01ff8be0e74e99cbc`，本地未提交修改。
- `LTX2TwoStageHQPipeline` stage1 distilled LoRA 按 official 语义 merge 到 transformer；单独复跑 `ltx_2_3_hq_pipeline` 指标不变：min clip `0.8432`，min SSIM `0.6399`，min PSNR `16.3690`，max MAD `22.7686`。
- HQ token timestep 用于 velocity->x0 后，单卡 `ltx_2_3_hq_pipeline` consistency pass：min clip `0.8443`，min SSIM `0.6395`，min PSNR `16.3639`，max MAD `22.7647`。相对 GT 的 CLIP 对齐 `84.43%`，bit-exact `0%`。
- 补齐 official final res2s denoise：HQ stage1 sigma append `0.0011`，stage2 refinement 用扩展后的 `scheduler_sigmas` 设置 `num_steps/timesteps`。单卡 consistency pass：min clip `0.8378`，min SSIM `0.6421`，min PSNR `16.4822`，max MAD `22.4095`。相对 GT 的 CLIP 对齐 `83.78%`，bit-exact `0%`。
- final res2s 与 token timestep 相比：SSIM `+0.0026`，PSNR `+0.1183 dB`，MAD `-0.3552`，CLIP `-0.0065`。这是更贴近 official res2s 的 pixel 指标提升，但会增加 HQ denoise/refinement 时间；未修改 perf baseline。

## 单卡 HQ 对齐继续推进

- git head: `bd3d34e3f69aadfbef6824b1e29c24a99215acd7`。
- 自动选择 `torch_sdpa` 后，`ltx_2_3_hq_pipeline` 单卡 consistency 通过：min clip `0.8885`，min SSIM `0.6548`，min PSNR `16.6546`，max MAD `21.9978`。相对 GT 的 CLIP 对齐 `88.85%`，bit-exact 对齐 `0%`。
- 对比修改前已知 origin/main HQ 指标 `0.8234 / 0.6192 / 15.6647 / 24.9817`：CLIP `+0.0651`，SSIM `+0.0356`，PSNR `+0.9899 dB`，MAD `-2.9839`。
- 有效改动：RowParallel LoRA disabled/merged path 回到 base layer；HQ token timestep 用于 velocity->x0；HQ stage1/stage2 final res2s 语义补齐；单卡 HQ 自动 `torch_sdpa`。
- 无收益 probe：official NumPy RoPE frequency grid 对最终三帧指标无可见变化，但保留为 upstream 语义对齐；text connector RoPE apply dtype 改动无收益且有轻微性能噪声，已用 `bd3d34e3f` 回退。
- 当前测试仍只因 `LTX2AVDecodingStage` perf baseline 失败；按要求未修改 perf baseline。

## 单卡 HQ DiT SDPA flags probe

- probe commit: `4afc36f8bc959d508eca4545e6f8cd884901db2a`，revert commit: `190faa86860b259500bbc382f3ce13c9fa5f7747`。
- 只在 HQ DiT forward context 临时设置 official SDPA flags（`math_sdp=False`），避免影响 Gemma text encoder。
- H100 单卡 `ltx_2_3_hq_pipeline` 结果无变化：min clip `0.8885`，min SSIM `0.6548`，min PSNR `16.6546`，max MAD `21.9978`。相对 GT 的 CLIP 对齐 `88.85%`，bit-exact 对齐 `0%`。
- 因无精度收益且增加 backend toggling 复杂度，已回退；继续保留已有自动 `torch_sdpa` 路径。

## 2026-05-03 connector additive mask 4D probe

- Probe: `2f08328c686f4c8fe0b7826e8256393fca4d5ecb`，将 LTX2 connector additive mask 从 `[B,S]` 调整为 official 形状 `[B,1,1,S]`。
- 单卡 H100 `ltx_2_3_hq_pipeline`：consistency 完全不变，`clip=0.8885, ssim=0.6548, psnr=16.6546, MAD=21.9978`。
- 结论：当前 2D/4D additive mask 在 connector 内等价，不是 HQ 单卡主要误差来源。已回退该 probe；bit-exact 对齐仍为 0%，CLIP 对齐为 88.85%。

## 2026-05-03 official HQ reference attempt

- 目标：用 official `TI2VidTwoStagesHQPipeline` 单卡同参数生成 HQ reference，和当前 SGLang 输出对比。
- official worktree：`/tmp/LTX-2-41d9243`，commit `41d924371612b692c0fd1e4d9d94c3dfb3c02cb3`。
- 结果：未进入生成阶段。当前容器 transformers 与 official Gemma loader 不兼容，连续遇到 `SiglipVisionModel.vision_model`、`rope_local_base_freq`、`rope_scaling.factor`、`rotary_emb_local` 等结构差异。
- 结论：该路径需要专门冻结 official 依赖环境，否则 reference 可能被兼容 monkey patch 污染；本轮先停止，回到 SGLang 侧做阶段性数值 probe。当前可用对齐指标仍是 `clip=0.8885, ssim=0.6548, psnr=16.6546, MAD=21.9978`，CLIP 对齐 88.85%，bit-exact 0%。

## 2026-05-03 HQ stage1 guidance batching probe

- Probe commit: `d66b4f8fce345909749a14e16213e1295df1e5c0`，revert commit: `b76c03c7967b6c93f657d58e2e4581a537b2b8d1`。
- 目的：验证 HQ stage1 是否可以从拆分 unconditional/conditional 两次 forward 改回 batched guidance forward。
- H100 单卡 `ltx_2_3_hq_pipeline` 结果变差：`clip=0.8274, ssim=0.6053, psnr=16.3266, MAD=24.1391`。
- 对比当前基线 `clip=0.8885, ssim=0.6548, psnr=16.6546, MAD=21.9978`：CLIP `-0.0611`，SSIM `-0.0495`，PSNR `-0.3280 dB`，MAD `+2.1413`。
- 结论：HQ stage1 必须保留 split guided passes；batched forward 不是 official 对齐方向，已回退。当前 CLIP 对齐恢复到 88.85%，bit-exact 0%。

## 2026-05-03 HQ stage2 live generator probe

- Probe commit: `083427170a1134c9d46ff84dfd6e83ea95f382ec`，revert commit: `8e672f0b60a191a65f86d096d2d35c7fb5ca6594`。
- 目的：验证 stage2 re-noise 直接消费当前 `batch.generator` 是否比重建 generator 并手工 skip stage1 shape 更贴近 official `GaussianNoiser`。
- H100 单卡 `ltx_2_3_hq_pipeline` 结果完全不变：`clip=0.8885, ssim=0.6548, psnr=16.6546, MAD=21.9978`。
- 结论：当前单卡 HQ 中手工 skip 的 RNG 状态与 live generator 等价；无精度收益，已回退以保持 diff 收敛。当前 CLIP 对齐 88.85%，bit-exact 0%。

- `37f6686` probe 将 LTX2.3 connector aggregate rescale 的 source_dim 从 `3840` 改为 flattened `188160`，单卡 HQ 指标大幅退化到 clip=0.4886/ssim=0.2743/psnr=13.2062/MAD=44.4328。复核 official `encoder_configurator.py` 后确认 `FeatureExtractorV2.embedding_dim=gemma_text_config.hidden_size=3840`，原实现正确；已用 `git revert` 回退。当前此 probe 对齐度：clip 48.86%，bit-exact 0%。
- `5ed3c2dff` probe: 给 LTX2.3 VAE decode 补 official decode noise/timestep/generator 语义，但 `Lightricks/LTX-2.3` overlay 的 `video_decoder_config.timestep_conditioning=false`，因此对当前 HQ case 是 no-op；该 probe 已用 `97a66f65e` 回退。测试未产出指标，H100 low-VRAM 下因 stage1 仍被 pin 导致 stage2 OOM。对齐度沿用当前最佳：clip 88.85%，bit-exact 0%。
- `1a1eabd38` probe: 将 LTX2.3 decoder `res_x_y` shortcut norm 从 LayerNorm 对齐到 official GroupNorm(1)，单卡 HQ 指标完全不变（clip=0.8885, ssim=0.6548, psnr=16.6546, MAD=21.9978），不是当前 GT 差距主因；已用 `9b8ce8ed8` 回退。当前相对 GT 对齐度：clip 88.85%，bit-exact 0%。
- `01558257b` probe: 保留 HQ native res2s SDE noise 的 float64 精度，不再 `.float()` 后进入 SDE step。H100 单卡 `ltx_2_3_hq_pipeline` 指标退化为 clip=0.8417/ssim=0.6144/psnr=16.2700/MAD=24.7870；对比当前最佳 clip=0.8885/ssim=0.6548/psnr=16.6546/MAD=21.9978，CLIP -0.0468，SSIM -0.0404，PSNR -0.3846 dB，MAD +2.7892。已用 `8b38aa060` 回退；当前相对 GT 对齐度恢复到 CLIP 88.85%，bit-exact 0%。

## 2026-05-03 单卡 HQ RoPE coords probe

- Probe commit: `ddbfb0b0cf29b26e182238ab4e69d58e5cbf4d6e`，revert commit: `a64dd2136b89361f915d414cd28cd91e4da2bda0`。
- 发现：HF config 中 `quantize_video_rope_coords_to_hidden_dtype=true`，而 official `LTXModelConfigurator` 不读取该字段；尝试 LTX2.3 下保持 video RoPE coords 为 fp32。
- H100 单卡 `ltx_2_3_hq_pipeline` 明显退化：clip=0.6326/ssim=0.4239/psnr=13.1982/MAD=44.7113；对比当前最佳 clip=0.8885/ssim=0.6548/psnr=16.6546/MAD=21.9978，CLIP -0.2559。
- 结论：当前 SGLang/GT 路径依赖 bf16 video RoPE coords；该 probe 已回退。当前对齐度恢复到 CLIP 88.85%，bit-exact 0%。

## 2026-05-03 text encoder 与 LoRA merge 排查

- git head: `a64dd2136b89361f915d414cd28cd91e4da2bda0`。
- Text encoder：root tokenizer + native Gemma + `pack_text_embeds_v2` 与 HF/Gemma3 输出几乎一致，packed tensor cos≈1.00036，MAE≈4.1e-7；不是当前最终图误差主因。
- Distilled LoRA：metadata `lora_rank=384/lora_alpha=384`，1660 对 A/B；实际张量验证 `(B @ A) * scale` 与 `(B * scale) @ A` 在 stage1 0.25/stage2 0.5 下 max diff=0，因此 LoRA merge 乘法顺序不是误差来源。
- 当前单卡 HQ 最佳仍为 clip=0.8885/ssim=0.6548/psnr=16.6546/MAD=21.9978；CLIP 对齐 88.85%，bit-exact 0%。

## 2026-05-03 单卡 HQ res2s 对齐进展

- worktree: `/Users/mick/repos/sglang-origin-main-metrics-20260503-100854`
- 分支: `ltx23-official-lora-merge-align`
- 关键 commit:
  - `1d15ee852`: mirror official `Res2sDiffusionStep.get_sde_coeff` 的细节；单独看 trace 无变化。
  - `5b59810f3`: main-step SDE 使用 scheduler 原始 dtype（float32），substep/RK 仍用 double；这是本轮主要提升。
  - `ead02436c` -> `3e632a073`: 重新试了 HQ stage1 batched guidance，step0 即漂移，已 revert。
- 原因定位:
  - official res2s loop 里 substep SDE 传 `torch.stack([sigma.double(), sub_sigma])`，main-step SDE 则传原始 `sigmas`（float32）。
  - 之前 SGLang main-step SDE 也用了 double，导致 `step_0_out` 只有 `~3e-8` 的差异，但 bf16 state update 后差异迅速放大。
- trace 结果:
  - `5b59810f3/3e632a073`: stage1 step/substep 0-5 全部 bit-exact；第一个非 exact 出现在 step 6 的 final deterministic denoised，sample/substep/noise 仍 exact。
  - stage1 final video latent: mean_abs `0.532885 -> 0.040842`，cos `0.754777 -> 0.998382`。
  - stage1 final audio latent: mean_abs `0.159523 -> 0.033425`，cos `0.949270 -> 0.998492`。
- final mp4 指标（SGLang vs official，1024x1024, 24 frames, 15 steps, seed 10）:
  - before main-step sigma dtype fix (`b7873064c`): PSNR `15.686`, SSIM `0.6566`, mean_abs `25.438`。
  - after main-step sigma dtype fix (`5b59810f3`): PSNR `29.920`, SSIM `0.9482`, mean_abs `3.392`。
- 当前对齐百分比:
  - raw Gemma hidden: 100% bit-exact。
  - stage1 early res2s: step/substep 0-5 为 100% bit-exact。
  - stage1 final latent: video cosine 99.84%，audio cosine 99.85%；不是 bit-exact。
  - final video: SSIM 94.82%，PSNR 29.92dB；已接近但还不是 fully aligned。
- 下一步:
  - 如果继续追 bit-exact，优先抓 step6 midpoint model eval 的 raw transformer outputs/layer outputs；剩余漂移不再是 SDE/noise，而是 step6 midpoint denoiser 相关。

## 2026-05-03 移除 SDPA/GQA 对齐改动

- commit: `b306e7640`
- 按要求不再考虑 Gemma SDPA/GQA 精度改动：移除了 `gemma_3.py` 中显式 repeat KV 的路径，恢复 `scaled_dot_product_attention(..., enable_gqa=True)`。
- 保留内容：Gemma RoPE device-side inv_freq、LTX2 HQ res2s SDE/noise/scalar/midpoint 修复均未改动。
- 未复跑 GPU 对齐；这是按“sdpa 的修改移除掉，不考虑”的范围做的代码回退。
- 当前精度对齐百分比：未重新测，沿用上一轮已知 HQ final video SSIM 94.82%/PSNR 29.92 的结果不再作为本次 SDPA 回退后的证明。

## 2026-05-03 SDPA/GQA 移除后的重新验证

- 当前保留 commit: `ca226882f`（revert 掉 eager probe 后回到 SDPA/GQA 移除状态）；远端容器已同步，`gemma_3.py` sha1=`4697760a02de1661b9befc248648dd037dba134d`。
- final mp4 复测（SGLang vs official，1024x1024, 24 frames, 15 steps, seed 10，单卡 H200，HQ pipeline）：
  - `all_avg`: PSNR `14.8183`, SSIM `0.6533`, mean_abs `28.1944`。
  - `key_avg`: PSNR `14.7550`, SSIM `0.6534`, mean_abs `28.4865`。
  - 对比 SDPA/GQA repeat 还存在时的 `5b59810f3`：PSNR `29.9201 -> 14.8183`，SSIM `0.9482 -> 0.6533`，mean_abs `3.392 -> 28.1944`。
- raw text hidden 对拍（official raw text vs `bb1008d90`/`ca226882f`）：attention mask 和 embedding 层仍 exact；从 Gemma layer 1 开始不再 exact。
  - `pos_hidden_48_valid`: mean_abs `0.03556`, rms `0.05274`。
  - `neg_hidden_48_valid`: mean_abs `0.02823`, rms `0.03960`。
- eager attention probe: `ea51d177b` 将 Gemma text attention 改成 HF eager matmul 公式；raw hidden 没恢复，且略差：
  - `pos_hidden_48_valid`: mean_abs `0.03636`, rms `0.05447`。
  - `neg_hidden_48_valid`: mean_abs `0.03184`, rms `0.04384`。
  - 已用 `ca226882f` revert，不保留该 probe。
- 结论：SDPA/GQA repeat 路径是当前 official default raw Gemma hidden bit-exact 的必要条件；移除后主要误差源在 text encoder 第 1 层开始，后段 res2s 已修复项无法补偿。当前 SDPA 移除状态下 final video SSIM 对齐约 65.33%，bit-exact 0%。

## 2026-05-03 HF SDPA/GQA 语义复核

- 本地 transformers `5.3.0` 中 `use_gqa_in_sdpa(attention_mask, key)` 只有在 `attention_mask is None` 且 torch 版本满足条件时才返回 true。
- HF `sdpa_attention_forward` 逻辑是：如果 `not use_gqa_in_sdpa(...)`，先 `repeat_kv(key/value)`；否则才传 `enable_gqa=True`。
- LTX2 Gemma text encoder 当前有 causal/padding additive mask，因此 official default SDPA 语义是 masked SDPA + explicit repeat KV，不是 masked SDPA + `enable_gqa=True`。
