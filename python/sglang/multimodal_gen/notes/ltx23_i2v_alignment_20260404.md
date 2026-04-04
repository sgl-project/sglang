## 2026-04-04 LTX-2.3 I2V Native 对齐

- 目标：
  - 官方侧固定使用 `python -m ltx_pipelines.ti2vid_one_stage`
  - SGLang 侧固定使用 `sglang generate --model-path Lightricks/LTX-2.3 --image-path ...`
  - 先做 native black-box 对拍，再决定是否进入 stage 二分
- 固定输入：
  - prompt: `A beautiful sunset over the ocean`
  - input image: `tmp/ltx23_videos/ltx23_i2v_input_sunset.png`
  - input image sha1: `1a6ec0d81f47d86e4da68463d7bd7eff0a81c217`
- 约束：
  - 不再以 overlay helper / 手写 probe 作为主路径
  - 官方和 SGLang 都保留各自 native example 语义
  - 初轮不额外覆盖 `seed / negative_prompt / guidance_scale / num_inference_steps`
- 当前已确认的输入语义：
  - SGLang 里 `width/height` 表示输出视频尺寸
  - 对 `LTX2I2V`，condition image 会先按输出尺寸语义做 resize/crop，再进入后续 stage
  - 因此 black-box 首轮优先不手工覆写尺寸，先记录双方 native 默认实际产物
- 当前脚本：
  - `notes/ltx23_alignment_20260404/h100_setup_official_i2v_env.sh`
  - `notes/ltx23_alignment_20260404/h100_run_official_one_stage_i2v.sh`
  - `notes/ltx23_alignment_20260404/h100_run_sglang_i2v.sh`
  - `notes/ltx23_alignment_20260404/compare_ltx23_i2v_videos.py`

## 2026-04-04 H100 native black-box 首轮结果

- 环境：
  - 机器：H100
  - 容器：`sglang_mick`
  - 官方 repo：`/tmp/LTX-2`
  - SGLang validation worktree：`/tmp/sglang_ltx23_validation`
- 官方 native i2v 已跑通：
  - 命令入口：`python -m ltx_pipelines.ti2vid_one_stage`
  - 输出：`/tmp/ltx23_official_i2v_blackbox.mp4`
  - 结果：稳定生成视频
- SGLang native i2v 已跑通：
  - 命令入口：`python -m sglang.multimodal_gen.runtime.entrypoints.cli.main generate --model-path Lightricks/LTX-2.3 --image-path ...`
  - 输出：`/tmp/ltx23_sglang_i2v_blackbox/sglang_i2v.mp4`
  - 结果：`Pixel data generated successfully in 85.98 seconds`
- 黑盒 compare 结果：
  - 两边输出元数据已对齐：
    - `768x512`
    - `121` 帧
    - `24 fps`
  - video hash：
    - official: `6d39c4806c155a756618787ff14497a00ebcebd3863b6b001d9c08703f9f25f5`
    - sglang: `0a6f78a3603f42ac80721a23036336e70b072d00a9447b8d846ef801409c74f5`
  - 逐帧像素差异仍然明显：
    - aggregate `mean_abs = 35.73`
    - aggregate `psnr = 14.75`
    - frame0 `mean_abs = 6.14`
    - frame60 `mean_abs = 37.25`
    - frame120 `mean_abs = 39.60`

## 2026-04-04 已修掉的真实 i2v blocker

- orchestration bug：
  - `h100_run_sglang_i2v.sh` 最初把 `PYTHONPATH` 写死到 `/sgl-workspace/sglang/python`
  - 导致远端没有真正使用 `/tmp/sglang_ltx23_validation`，而是直接掉回没有 `LTX-2.3` overlay 的旧代码路径
  - 修复后，native CLI 已正确命中 `MickJ/LTX-2.3-overlay`
- model-side bug：
  - 第一处真实失败点在 `runtime/models/dits/ltx_2.py` 的 `cross_attention_adaln` prompt modulation
  - 现象：i2v 在 `LTX2AVDenoisingStage` step0 报 `encoder_hidden_states` 与 `prompt_scale_shift` 维度语义不一致
  - 原因：SGLang 直接把 i2v tokenwise masked `timestep_video` 喂给了 `prompt_adaln_single`；这不符合官方 native 路径里 prompt modulation 使用独立 batch-level sigma 语义的做法
  - 修复：
    - `c8ba59540 Fix LTX-2.3 I2V prompt AdaLN timestep`
    - `2b6c4924e Fix LTX-2.3 I2V prompt AdaLN dtype`
  - 修复后，SGLang native i2v 已能完整 end-to-end 出视频

- 当前精度对齐进度：25%
  - 已完全对齐到的 stage：native example 跑法、输入素材、输出视频元数据、end-to-end 可运行性
  - 当前首个未对齐的大阶段：denoise loop 及其 image-conditioning 生效语义

## 2026-04-04 二分结论 1: step 后 TI2V reinject 不符合官方 native 语义

- 源码对读范围：
  - 官方：`ltx_pipelines/utils/samplers.py`
  - 官方：`ltx_pipelines/utils/helpers.py`
  - 官方：`ltx_core/conditioning/types/latent_cond.py`
  - SGLang：`runtime/pipelines_core/stages/denoising_av.py`
- 当前确认的官方 one-stage i2v 语义：
  - `VideoConditionByLatentIndex` 只在初始 `LatentState` 上写入 image latent token，并把对应 `denoise_mask` 设为 `1 - strength`
  - `GaussianNoiser` 在 step0 前用 `latent = noise * denoise_mask + latent * (1 - denoise_mask)` 保留这些 conditioned token
  - 每步 denoise 后只做 `post_process_latent(denoised, denoise_mask, clean_latent)`
  - sampler step 后不会再次把 image latent token 手动写回 `latent`
- SGLang 当前额外差异：
  - `runtime/pipelines_core/stages/denoising_av.py` 在两条 LTX2 i2v 分支里，step 后都额外执行了一次
    `latents[:, :num_img_tokens, :] = batch.image_latent[...]`
  - 这一步在官方 native i2v 路径里不存在，属于额外 conditioning 语义
- 已做修复：
  - 删除了这两处 step 后的 TI2V reinject
  - 保留 `denoised_video * denoise_mask + clean_latent * (1 - denoise_mask)` 这条官方一致的 mask blend 路径
- 当前精度对齐进度：35%
  - 已完全对齐到的 stage：image preprocess、initial conditioning token 注入、高层 sampler 里的 conditioned-token 保持语义
  - 下一步首个待验证的大阶段：移除 reinject 之后的 native black-box 是否明显收敛；若仍然偏离，再继续查 sigma/timestep 在 transformer API 上的映射

## 2026-04-04 二分结果更新

- commit `2cf3efc31 Align LTX-2.3 I2V latent retention`
  - 动作：删除 `runtime/pipelines_core/stages/denoising_av.py` 里两处 step 后 TI2V reinject
  - 远端 H100 黑盒结果：
    - official sha256: `6d39c4806c155a756618787ff14497a00ebcebd3863b6b001d9c08703f9f25f5`
    - sglang sha256: `5343a55cebc0a0d8c208ed51d36cb9f4c06b0433c088d29e325b6717b3dc4ff3`
    - aggregate `mean_abs = 35.728878021240234`
    - aggregate `psnr = 14.752602149730016`
  - 结论：
    - 视频内容已经变化，但逐帧指标和上一轮完全一致
    - 说明这两处 reinject 不是当前黑盒误差的主因

- commit `ca490a2dc Align LTX-2.3 I2V cross-modality sigma`
  - 动作：把 `runtime/models/dits/ltx_2.py` 里 cross-modality AdaLN timestep 来源改成对端模态 sigma 语义
  - 远端 H100 黑盒结果：
    - sglang sha256: `d32baa88f538b744c3b4eb1c3dd6683eb8cecdd0c9f88237f4176f6e4405a1e8`
    - aggregate `mean_abs = 35.90249252319336`
    - aggregate `psnr = 14.711861218572878`
  - 结论：
    - 指标略微变差，不是正确方向
    - 已用 commit `21234b471 Revert "Align LTX-2.3 I2V cross-modality sigma"` 撤回

- 到目前为止已排除的两类主因：
  - step 后 image-latent reinject
  - cross-modality AdaLN 的 sigma 来源

- 当前新的高优先级怀疑点：
  - request / example 层的 i2v conditioning 语义是否真的完全一致
  - 进入 step dump 二分前，需要先确认官方 `--image PATH 0 1.0` 和 sglang `--image-path PATH` 在 `frame_idx / strength / crf` 语义上是否完全等价

## 2026-04-04 二分结果更新 2：LTX2 默认 CPU generator 是 step0 主因

- commit `22f9a8ffd Align LTX2 packed noise sampling`
  - 动作：
    - 把 `runtime/pipelines_core/stages/latent_preparation_av.py` 里的 video/audio 初始噪声改成直接在 packed token shape 上采样
  - 远端 H100 step0 compare 结果：
    - `video_latent_before cosine = 0.00268407235853374`
    - `audio_latent_before cosine = -0.0030366359278559685`
  - 结论：
    - packed vs unpacked shape 不是主因
    - step0 初始 latent 仍然几乎不相关

- commit `8b3a86248 Fix LTX2 audio packed noise length`
  - 动作：
    - 修正 packed audio 试验里错误使用 latent video frames `17` 作为 audio latent length 的问题
  - 结论：
    - 只是修复实验路径的 audio shape，真正 root cause 仍未命中

- 重新核对后的关键发现：
  - 官方 native `TI2VidOneStagePipeline` 在 `ti2vid_one_stage.py` 里使用：
    - `generator = torch.Generator(device=self.device).manual_seed(seed)`
  - SGLang 的 `LTX2SamplingParams` / `LTX2PipelineConfig` 默认此前是：
    - `generator_device = "cpu"`
  - 之前那轮 `cuda generator` 试验不够可信，所以重新在当前代码上复验

- 显式 `--generator-device cuda` 的远端 H100 step0 compare：
  - `video_latent_before mean_abs = 0.06008164957165718`
  - `video_latent_before cosine = 0.9547939300537109`
  - `audio_latent_before mean_abs = 0.001131223514676094`
  - `audio_latent_before cosine = 1.0000007152557373`
  - `video_latent_after cosine = 0.9544340968132019`
  - `audio_latent_after cosine = 0.9999838471412659`
  - 结论：
    - 初始 latent 已从“几乎完全不相关”收敛到“audio 几乎完全一致、video 高度一致”
    - root cause 就是 LTX2 默认把 generator 建在 CPU，而官方 native path 用的是 CUDA generator

- commit `d24a47077 Default LTX2 generator to CUDA`
  - 动作：
    - 把 `configs/sample/ltx_2.py` 和 `configs/pipeline_configs/ltx_2.py` 的默认 `generator_device` 从 `cpu` 改成 `cuda`
    - 补最小单测锁定 `Lightricks/LTX-2.3` 默认使用 CUDA generator
  - 远端 H100 验证：
    - 在不显式传 `--generator-device` 的 native CLI 路径下，step0 compare 结果与显式 `cuda` 完全一致
    - 说明修复已经进入产品默认路径，而不是实验脚本特供

- 修复后的 native black-box compare：
  - official sha256: `6d39c4806c155a756618787ff14497a00ebcebd3863b6b001d9c08703f9f25f5`
  - sglang sha256: `68443310150b44f0fffaa1b8f9648e0dc273ff00642b1d8ebf59c3b7ed6fe08b`
  - aggregate `mean_abs = 35.03176498413086`
  - aggregate `psnr = 14.855198436765676`
  - frame0 `mean_abs = 7.719090938568115`
  - frame60 `mean_abs = 35.95792770385742`
  - frame120 `mean_abs = 39.51141357421875`
  - 结论：
    - 黑盒确实有改善，但还远未对齐
    - `latent_before` 已不是主因，当前主发散点已经推进到 denoiser / transformer forward 之后

- 当前精度对齐进度：55%
  - 已完全对齐到的 stage：
    - native example 输入语义
    - prompt/image 请求层
    - initial latent construction / step0 latent-before
  - 当前首个未对齐的大阶段：
    - official `FactoryGuidedDenoiser` / SGLang `ltx2_stage1_guider_params` 驱动下的 transformer / guider 输出
  - 下一步：
    - 对读官方 `ltx_pipelines/utils/denoisers.py` 与 SGLang `runtime/pipelines_core/stages/denoising_av.py`
    - 继续二分 `cond / uncond / ptb / mod` 这几条 pass 或其 guider combine 前后的输出

## 2026-04-04 二分结果更新 3：cross-modality sigma 重新复验后仍然更差

- 背景：
  - 之前在 CPU generator 仍未对齐时，曾尝试把 cross-modality AdaLN 改成使用对端模态 sigma，黑盒略微变差
  - 在 `generator_device = cuda` 已成为默认路径之后，再按官方 `transformer_args.py` 重新核对一次

- 官方源码确认：
  - `ltx_core/model/transformer/transformer_args.py`
  - `MultiModalTransformerArgsPreprocessor.prepare(...)` 的 cross-attention modulation 使用的是 `cross_modality.sigma`
  - 即：
    - video cross-attn modulation 用 audio sigma
    - audio cross-attn modulation 用 video sigma

- 试验 commit：
  - `e83b234f3 Align LTX2 cross-modality sigma`

- 远端 H100 step0 compare：
  - 基线（仅 generator 对齐，commit `d24a47077`）：
    - `video_denoised mean_abs = 0.5878117680549622`
    - `video_denoised cosine = 0.5328061580657959`
    - `audio_denoised mean_abs = 1.27132248878479`
    - `audio_denoised cosine = -0.7826241850852966`
  - cross-modality sigma 复验（commit `e83b234f3`）：
    - `video_denoised mean_abs = 0.5880815386772156`
    - `video_denoised cosine = 0.5293195843696594`
  - `audio_denoised mean_abs = 1.307675838470459`

## 2026-04-04 二分结果更新 4：image latent 根因已收窄到 2.3 官方 VideoEncoder

- 触发点：
  - 在 `step0` clean-state dump 里，`video_denoise_mask cosine = 1.0`
  - 但 `video_clean_latent cosine = 0.0080`
  - `image_latent cosine = 0.0370`
  - 说明 mask 语义已经对齐，真正没对齐的是 conditioning image encode 本身

- 官方源码确认：
  - `ltx_pipelines/utils/helpers.py`
    - `combined_image_conditionings(...)` 直接走 `encoded_image = video_encoder(image)`
  - `ltx_core/model/video_vae/video_vae.py`
    - `VideoEncoder.forward()` 最后返回的是 `per_channel_statistics.normalize(means)`
    - 不是 diffusers `AutoencoderKL.encode(...).latent_dist.mode()/sample()` 那条旧路径

- 与当前 SGLang 的差异：
  - `runtime/pipelines_core/stages/denoising_av.py::_prepare_ltx2_image_latent()`
  - 当前仍然走：
    - `self.vae.encode(video_condition)`
    - `latent_dist.mode()/sample()`
    - 手动 `(latent - latents_mean) / latents_std`
  - 这条逻辑更接近旧 `LTX-2` diffusers VAE，不是 2.3 官方 native `VideoEncoder`

- 进一步核对 HF donor 后的结论：
  - `FastVideo/LTX-2.3-Distilled-Diffusers/vae/model.safetensors`
    - 确实包含 top-level `per_channel_statistics.mean-of-means/std-of-means`
    - 但主体 VAE key-space 不是旧 `LTX-2` 那套
  - 结构性差异例子：
    - 旧 `LTX-2`：
      - `encoder.down_blocks.{0..3}.resnets`
      - `encoder.down_blocks.{0..3}.downsamplers`
      - `encoder.mid_block`
      - `encoder.conv_out` 输入通道是 `2048`
    - 2.3 / FastVideo：
      - `encoder.down_blocks.{0..8}` 交替出现 `res_blocks` 和单独 `conv`
      - 没有旧语义上的 `mid_block`
      - `encoder.conv_out` 输入通道是 `1024`

- 结论：
  - “把 2.3 整套 `vae` donor 直接塞进当前 native `AutoencoderKLLTX2Video`”不是可行路径
  - 当前首个未对齐 stage 已进一步收窄到：
    - official `VideoEncoder(image)` / SGLang `condition image latent encode`
  - 下一条正确路线应是：
    - 为 `LTX-2.3` i2v 增加 dedicated native `VideoEncoder`
    - 或在现有 native VAE 侧补 2.3 encoder block schema
  - 在这之前，继续尝试整套 2.3 `vae` donor 替换没有意义

- 当前精度对齐进度：60%
  - 已完全对齐到的 stage：
    - native example 输入语义
    - initial latent construction / step0 latent-before
    - i2v denoise mask / clean-state 的高层 conditioning 语义
  - 当前首个未对齐的大阶段：
    - condition image -> official `VideoEncoder` -> image latent
    - `audio_denoised cosine = -0.7909688949584961`

- 结论：
  - 在 latent-before 已对齐的前提下，这个 patch 依然更差
  - 它不是当前主发散点
  - 已用 commit `2bbd476fc Revert "Align LTX2 cross-modality sigma"` 回滚，保持分支干净

- 当前最可信的下一处主因：
  - official `FactoryGuidedDenoiser` / `MultiModalGuider.calculate()` 所驱动的 `cond / uncond / ptb / mod` pass 输出本身
  - 需要继续对拍 first forward 的 pass-level outputs，而不是继续改 timestep 来源

## 2026-04-04 二分结果更新 4：first divergence 在 video cond pass，不在 guider combine

- commit `ebe4d99aa Revert "Align LTX2 context mask handling"`
  - 动作：
    - 回滚 `encoder_attention_mask = None` 这轮实验
  - 依据：
    - 远端 H100 step0 compare 只带来边际变化：
      - `video_denoised cosine: 0.5328 -> 0.5395`
      - `audio_denoised cosine: -0.7826 -> -0.7834`
    - 不是值得继续保留的主修复

- commit `dd2aaf760 Add LTX2 I2V pass-level step dumps`
  - 动作：
    - 扩展 SGLang `runtime/pipelines_core/stages/denoising_av.py` 的 `step_dump`，额外记录
      - `video/audio cond`
      - `video/audio uncond`
      - `video/audio ptb`
      - `video/audio mod`
    - 同时更新官方 native `official_i2v_step_dump.py` helper，在不改 upstream repo 的前提下，给 `FactoryGuidedDenoiser` 增加同样的 pass-level dump
  - 远端 H100 pass-level compare 结果：
    - `video_cond cosine = 0.3457`
    - `video_uncond cosine = 0.2064`
    - `video_ptb cosine = 0.2849`
    - `video_mod cosine = 0.3415`
    - `video_denoised cosine = 0.5328`
    - `audio_cond cosine = 0.1390`
    - `audio_uncond cosine = 0.6641`
    - `audio_ptb cosine = 0.6437`
    - `audio_mod cosine = 0.99965`
    - `audio_denoised cosine = -0.7826`
  - 结论：
    - guider 的 combine 公式不是首个主因
    - `video cond` 自身已经明显发散，说明第一次 conditioned forward 就不对
    - `audio mod` 几乎完全对齐，说明 audio 单模态路径基本是对的；audio 的大偏差主要来自 cross-modality coupling
    - 但 `video mod` 仍然明显发散，说明真正更靠前的 root cause 仍在 video 路径本身，而不是 only cross-modality guider/cross-attn combine

- 当前精度对齐进度：62%
  - 已完全对齐到的 stage：
    - native example 输入语义
    - prompt/image 请求层
    - initial latent construction / step0 latent-before
    - step0 guider pass-level 输出的“定位层级”
  - 当前首个未对齐的大阶段：
    - step0 第一次 video conditioned forward
  - 下一步：
    - 优先对读官方 `Modality(timesteps/positions/context_mask)` 语义和 SGLang video 路径的 `timestep_video / coords / i2v denoise_mask`
    - 如有必要，再把 native dump 向前推进到 video transformer block0 的 cond/mod 路径

## 2026-04-04 二分结果更新 5：timestep-scale 是一条 false lead，原因是 SGLang `timesteps` 已经是 1000 标度

- 触发原因：
  - 对读官方 `ltx_core/model/transformer/transformer_args.py` 时，看到 `_prepare_timestep()` 会把 `timestep * timestep_scale_multiplier` 再送进 AdaLN
  - 直觉上像是 SGLang `runtime/models/dits/ltx_2.py` 漏掉了这层 scaling

- 试验 commit：
  - `7c1de964d Align LTX2 timestep scaling with official`
  - 动作：
    - 给 main/prompt/cross-scale-shift 的 AdaLN 输入补 `self.timestep_scale_multiplier`

- 远端 H100 pass-level compare（错误 patch）：
  - `video_cond cosine = 0.2156`，比基线 `0.3457` 更差
  - `audio_mod cosine = 0.3029`，比基线 `0.99965` 大幅变差
  - 虽然 `video_denoised cosine = 0.5879`、`audio_denoised cosine = -0.1074` 表面上变好，但 first-divergence 的 pass-level 指标明显恶化
  - 结论：
    - 这不是 source-aligned 修复，更像偶然补偿

- 根因澄清：
  - 本地直接验证 `FlowMatchEulerDiscreteScheduler`：
    - `timesteps[0] = 1000.0`
    - `sigmas[0] = 1.0`
  - 也就是说：
    - 官方 native 路径是 `raw sigma -> *1000 -> AdaLN`
    - SGLang 当前传进 model 的 `timestep` 已经是 scheduler 的 `timesteps`，本来就是 1000 标度
    - 所以上述 patch 实际上造成了 double-scale

- 回滚：
  - `e8d48791a Revert "Align LTX2 timestep scaling with official"`

- 当前最可信的下一处主因：
  - 不是 timestep 是否乘 1000
  - 仍然是 step0 第一次 `video cond` forward 里的 video-path 语义
  - 下一刀应继续看：
    - i2v video `timesteps / denoise_mask / positions`
    - 或 native block0 `cond/mod` 内部输出

## 2026-04-04 二分结果更新 6：`clean_latent` 非条件 token 不该继承初始噪声，只对 `LTX-2.3` official-image-encoder 分支置零

- 触发原因：
  - 读官方 `ltx_core/tools.py::VideoLatentTools.create_initial_state()` 和 `ltx_core/conditioning/types/latent_cond.py::VideoConditionByLatentIndex.apply_to()`
  - 官方语义是：
    - 初始 `clean_latent` 全 0
    - 然后只把 image latent 写进目标 token
    - `GaussianNoiser` 再用 `noise * denoise_mask + latent * (1 - denoise_mask)` 生成 `latent_before`
  - SGLang 之前的 `runtime/pipelines_core/stages/denoising_av.py`
    - 用 `clean_latent = latents.detach().clone()`
    - 这会把非 image token 的初始噪声错误带进 `clean_latent`

- 修复 commit：
  - `e4c7def18 Fix LTX2.3 TI2V clean latent state`

- 动作：
  - 新增一个很小的局部 helper：`LTX2AVDenoisingStage._prepare_ltx2_ti2v_clean_state(...)`
  - 仅当 `vae_config.arch_config.use_official_image_encoder == true` 时，走 `zero_clean_latent=True`
    - 也就是只影响 `LTX-2.3` overlay 产物
  - 旧 `LTX2` 路径继续保留原有 background 语义，不动默认行为
  - 补两条最小单测：
    - `LTX-2.3` 走 zero clean latent
    - legacy `LTX2` 仍保留旧 background 语义

- 本地校验：
  - `python -m py_compile runtime/pipelines_core/stages/denoising_av.py test/unit/test_model_overlay_ltx23.py`
  - 通过

- 远端 H100 step0 compare：
  - 代码同步到 `mick/ltx-2.3@e4c7def18`
  - 关键文件 hash：
    - `denoising_av.py = ae209fb28913f74a9a6f940ee27bd60af57b2b4d`
    - `test_model_overlay_ltx23.py = 1ff54890a7ce04ef9deda678cbf534ebdf644048`
  - 运行结果：
    - `video_clean_latent cosine = 0.9998642`
    - `image_latent cosine = 0.9998641`
    - `video_latent_before cosine = 0.9999685`
    - `video_cond cosine = 0.9607990`
    - `video_denoised cosine = 0.9915146`
    - `video_latent_after cosine = 0.9999637`
  - 结论：
    - `condition image -> image latent -> clean_state -> step0 video latent_after` 这一整段现在已经基本对齐
    - 之前 `video_clean_latent cosine ≈ 0.2158` 的问题已经解决

- 黑盒视频 compare：
  - official: `/tmp/ltx23_official_i2v_blackbox.mp4`
  - sglang: `/tmp/ltx23_sglang_i2v_step_dump_native23/sglang_i2v.mp4`
  - 当前结果仍然很差：
    - `aggregate mean_abs = 36.8914`
    - `aggregate psnr = 14.4569`
    - `frame0 mean_abs = 35.7956`
    - `frame60 mean_abs = 36.1781`
    - `frame120 mean_abs = 39.9998`
  - 解释：
    - step0 clean-state 已对齐，但最终视频没有跟着收敛
    - 这说明主发散点已经后移到 `step1+` 的 denoise loop / guider state update，而不是 image encoder 或 initial clean-state

- 当前精度对齐进度：74%
  - 已完全对齐到的 stage：
    - native example 输入语义
    - initial latent construction / step0 latent-before
    - `condition image -> image latent`
    - `step0 clean_latent / denoise_mask`
    - `step0 video denoised / latent_after`
  - 当前首个未对齐的大阶段：
    - `step1+` denoise loop 的状态推进
  - 下一步：
    - 优先继续 native step dump，把 compare 从 `step0` 扩到 `step1`
    - 若 `step1 latent_before` 已开始发散，就直接查 scheduler step / guider update / `post_process_latent` 之后的状态推进

## 2026-04-04 二分结果更新 7：`step1` 的 sigma schedule 和 TI2V x0 还原语义都修正了，视频 pass-level 已接近对齐

- 触发原因：
  - `step1` compare 首先暴露两边 sigma 不一致：
    - official `0.9949570`
    - sglang `0.9970173`
  - 随后继续对读官方 `X0Model`，发现它是按 `video.timesteps = sigma * denoise_mask` 做 tokenwise x0 还原；
    我们此前在 TI2V 路径里还在用标量 `sigma` 手工还原 `video_cond/uncond/ptb/mod`

- 修复 1：对齐官方 sigma schedule，但不破坏 legacy `LTX2`
  - commit `366b3ed90 Align LTX2.3 sigma schedule with official`
  - commit `0ac898b61 Avoid double-shifting LTX2.3 sigma schedule`
  - 动作：
    - 在 [ltx_2_pipeline.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/pipelines/ltx_2_pipeline.py) 新增 `build_official_ltx2_sigmas(...)`
    - 仅当 `use_official_image_encoder == true` 时，`LTX2SigmaPreparationStage` 使用官方 `LTX2Scheduler.execute(...)` 等价 schedule
    - 同时对同一路径返回 `("mu", None)`，避免把自定义 sigma 再做一次 mu-shift
    - legacy `LTX2` 仍保留旧线性 schedule
  - 结果：
    - `step1 sigma` 现已完全一致：
      - official `0.9949570298`
      - sglang `0.9949570298`

- 修复 2：TI2V video x0 还原改为 tokenwise sigma
  - commit `8f34ef620 Use tokenwise sigma for LTX2.3 TI2V x0`
  - 动作：
    - 在 [denoising_av.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py) 新增 `_ltx2_velocity_to_x0(...)`
    - 对 TI2V video 分支，把 `x0 = x - sigma * v` 改成按 `sigma * denoise_mask` 的 tokenwise 形式
    - audio 仍是标量 sigma，不改 legacy 路径
  - 远端 H100 `step1` compare（修完后）：
    - `video_cond cosine = 0.996356`
    - `video_uncond cosine = 0.996297`
    - `video_ptb cosine = 0.996151`
    - `video_mod cosine = 0.998290`
    - `video_latent_before cosine = 0.999968`
    - `video_latent_after cosine = 0.999965`
  - 结论：
    - `step1` 的 video transformer pass-level 输出现在已经基本对齐
    - 说明 scheduler 和 TI2V x0 语义这两层 root cause 已经修正

- 当前剩余主点：
  - `video_denoised cosine` 仍只有 `0.939495`
  - 但各个 pass-level 输入已经接近 `0.996+`
  - 所以剩余首个主发散点已经收缩到：
    - guider combine / rescale
    - 或者 combine 后到 final `denoised_video` 的这条极窄路径

- 黑盒现状：
  - 最新黑盒视频还没有明显改善，说明当前 residual 虽然范围很小，但放大很快
  - 下一步应直接验证：
    - 用官方 guider 公式在两边 pass dump 上离线重算一次 `video_denoised`
    - 确认是 combine 公式、rescale、还是 dump 对比口径的问题

## 2026-04-04 二分结果更新 8：post-denoise injected run 仍然是噪点，问题已经落到 `denoising_av` 之后

- 动作：
  - 在 [denoising_av.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py) 增加
    `SGLANG_DIFFUSION_LTX2_POST_DENOISE_INJECT_PATH`
    - 只在 `_post_denoising_loop(...)` 开头覆盖 `latents/audio_latents`
    - 不改 denoise loop 本身
  - 新增官方 native final-latent dump 脚本：
    - [official_i2v_final_latents_dump.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/notes/ltx23_alignment_20260404/official_i2v_final_latents_dump.py)
    - [h100_run_official_i2v_final_latents_dump.sh](/Users/mick/repos/sglang/python/sglang/multimodal_gen/notes/ltx23_alignment_20260404/h100_run_official_i2v_final_latents_dump.sh)
  - 新增 injected native run 脚本：
    - [h100_run_sglang_i2v_post_denoise_injected.sh](/Users/mick/repos/sglang/python/sglang/multimodal_gen/notes/ltx23_alignment_20260404/h100_run_sglang_i2v_post_denoise_injected.sh)

- 远端 H100 实验：
  - official native dump：
    - `/tmp/ltx23_official_i2v_final.pt`
  - sglang injected video：
    - `/tmp/ltx23_sglang_i2v_injected_final/sglang_i2v_injected.mp4`
  - runtime 日志确认 injected hook 生效：
    - `Injecting post-denoising LTX2 latents from /tmp/ltx23_official_i2v_final.pt`

- injected video compare：
  - official:
    - `/tmp/ltx23_official_i2v_blackbox.mp4`
  - injected:
    - `/tmp/ltx23_sglang_i2v_injected_final/sglang_i2v_injected.mp4`
  - 本地 compare：
    - `aggregate mean_abs = 36.8703`
    - `aggregate psnr = 14.4899`
    - `frame0 mean_abs = 36.0183`
    - `frame60 mean_abs = 36.0531`
    - `frame120 mean_abs = 40.0042`

- 结论：
  - 即使在 `denoising_av` 结束后直接注入 official final latents，最终视频仍然和 official 相差很大
  - 所以当前“全部是噪点”的主因已经不在 denoise loop 主线，而是在 `denoising_av` 之后：
    - `_post_denoising_loop`
    - `_unpad_and_unpack_latents`
    - 或 `LTX2AVDecodingStage` / VAE decode

- 结合 overlay 现状的进一步判断：
  - [materialize.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/model_overlays/ltx_2_3/_overlay/materialize.py)
    现在仍然把 `vae/**` 从 `Lightricks/LTX-2` 整套拷过来
  - `FastVideo/LTX-2.3-Distilled-Diffusers` 只被拿来做：
    - `transformer/config.json`
    - `text_encoder/config.json`
    - `vae/ltx23_image_encoder/*`
  - injected run 的结果说明：
    - 仅修 image encoder 不够
    - `LTX-2` donor VAE decode / latent layout 很可能仍然不符合 `LTX-2.3`

- 当前精度对齐进度：75%
  - 已完全排除：
    - `denoising_av` 之前的输入层
    - `step0/step1` 的主要 denoise 主线
  - 当前首个未对齐的大阶段：
    - `post-denoising latent -> unpack -> decode`
  - 下一步：
    - 优先对读官方 `LTX-2.3` decode 路径与当前 `LTX2AVDecodingStage`
    - 重点核查 `vae` donor、latent unpack 布局、decode 前 denorm 语义

## 2026-04-04 二分结果更新 9：把 `LTX-2.3` root `vae` 从旧 donor 改成“旧 encoder + 2.3 decoder + 2.3 stats”

- 这轮源码排查结论：
  - `_unpack_latents/_unpack_audio_latents` 和官方 patchifier 已经完全一致
  - injected run 证明问题不在 `denoising_av` 主线，而在 decode 侧
  - 继续对读官方 `VideoDecoder` 后，确认当前最大结构性问题是：
    - overlay root `vae` 仍然整套复用 `LTX-2`
    - 而官方 `LTX-2.3` decoder 走的是 `decoder + per_channel_statistics.un_normalize(...)`
    - 当前 [decoding_av.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding_av.py) 还会在 decode 前额外做一次外部 denorm

- 本地代码修复：
  - [ltx_2_vae.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/models/vaes/ltx_2_vae.py)
    - 新增 `LTX23VideoDecoder3d`
    - 只在 `use_official_video_decoder == true` 时启用
    - decoder block schema 对齐官方 `decoder_blocks`
    - 内部使用 `per_channel_statistics.un_normalize(...)`
    - 原来的 `LTX2` decoder 路径保持不变
  - [decoding_av.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding_av.py)
    - 新增 `use_official_video_decoder` 分支
    - `LTX-2.3` decode 前跳过外部 `latents_mean/std` denorm
    - 原来的 `LTX2` decode 路径保持不变
  - [materialize.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/model_overlays/ltx_2_3/_overlay/materialize.py)
    - 不再把 root `vae/**` 从 `LTX-2` 整套拷过来
    - 改为：
      - root `vae/config.json` 写入 `use_official_video_decoder` 和 donor `official_vae_config`
      - root `vae/model.safetensors` = `LTX-2 encoder.*` + `LTX-2.3 decoder.*` + `LTX-2.3 per_channel_statistics`
    - `ltx23_image_encoder` 子目录继续保留现有官方 image encoder 路径

- 轻量校验：
  - 本地 `py_compile` 通过：
    - `runtime/models/vaes/ltx_2_vae.py`
    - `runtime/pipelines_core/stages/decoding_av.py`
    - `model_overlays/ltx_2_3/_overlay/materialize.py`
    - `test/unit/test_model_overlay_ltx23.py`
  - 新增最小单测覆盖：
    - overlay `vae/config.json` 现在带 `use_official_video_decoder`
    - 2.3 decoder repack 会生成 `decoder.per_channel_statistics.*` 和 top-level `latents_mean/std`
    - `LTX-2.3` decode 前会跳过外部 denorm，而 legacy `LTX2` 仍保留原行为

- 当前精度对齐进度：80%
  - 已完全对齐到：
    - `denoising_av` 结束后的 packed latents
    - `token -> unpack`
    - `decode` 前的语义建模路径
  - 当前待验证的首个阶段：
    - fresh materialize 后的真实 `raw_video`
  - 下一步：
    - 远端 fresh rematerialize `LTX-2.3`
    - 直接 compare `official_i2v_decode.pt` vs `sglang_i2v_decode.pt`
    - 如果 `raw_video` 明显收敛，再复跑黑盒视频
