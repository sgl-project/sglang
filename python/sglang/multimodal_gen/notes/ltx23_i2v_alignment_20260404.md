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
