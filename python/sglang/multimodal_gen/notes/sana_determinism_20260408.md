# Sana Determinism 排查记录 2026-04-08

## 背景

- 目标 case: `sana_image_t2i`
- 现象: 同一提交、同一 seed、同一 prompt，在 CI rerun 中出现一次 `PSNR=20.98`、一次 exact match
- 进一步确认后，CI 两次 job 实际落到了不同 runner:
  - failed: H100
  - rerun success: H200

## 远端复现结论

- 在同一台机器、同一个 server 进程内，重复相同请求，结果稳定
- 但只要重启 server 进程，即使同机同卡同提交，最终图片 hash 也会变化
- 这个现象在 H200 上稳定可复现；H100 上也看到同趋势

## 逐层定位

### 已对齐 / 稳定的阶段

- text prompt embeds: 稳定
- step00 latent input: 稳定
- Sana forward 输入侧:
  - `patch_embed["proj"]` 后的 `pre_block_hidden_states`: 稳定
  - `timestep_emb`: 稳定
  - `caption_projection + caption_norm` 后的 `encoder_hidden_states`: 稳定
- transformer:
  - `block00_out`: 稳定
  - `block01_out`: 稳定

### 最新根因定位

- 在 no-offload 条件下，对整条 transformer 重新做冷启动全量 probe：
  - `text_prompt_embeds`: 稳定
  - `pre_block_hidden_states`: 稳定
  - `timestep_emb`: 稳定
  - `encoder_hidden_states_proj`: 稳定
  - `block00_out` ~ `block14_out`: 稳定
  - `block15_out`: 第一个开始分叉
- 继续把 `block15` 拆成子阶段后发现：
  - `block15.timestep`: 稳定
  - `block15.scale_shift_table`: 跨进程重启后不稳定
  - 所以后续 `norm1_modulated / attn1 / attn2 / ff` 的分叉只是这个参数失配的下游结果

### 决定性证据

- 两次冷启动中，live `block15.scale_shift_table` 的 hash 分别是：
  - `edb4d0604ff6cc3b03347ec410938c73a21feac0`
  - `3f3b5c5bc49168f8bb04da0dedd54b278981b944`
- 把 checkpoint 中同一 key 做 `bf16 -> float32` 后比较，发现它们分别精确匹配：
  - `diffusion_pytorch_model.safetensors` -> `edb4d0604ff6cc3b03347ec410938c73a21feac0`
  - `diffusion_pytorch_model.fp16.safetensors` -> `3f3b5c5bc49168f8bb04da0dedd54b278981b944`
- 说明同一次 Sana 启动里，transformer 有时拿到 full 权重版本，有时拿到 fp16 权重版本

### 真正根因

- `TransformerLoader` 会直接把 component 目录下所有 `*.safetensors` 都交给 transformer loader：
  - [runtime/loader/component_loaders/transformer_loader.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py)
  - [runtime/loader/transformer_load_utils.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py)
  - [runtime/loader/utils.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/loader/utils.py)
- Sana transformer 目录同时存在两套重复权重文件：
  - `diffusion_pytorch_model.safetensors`
  - `diffusion_pytorch_model.fp16.safetensors`
- `safetensors_weights_iterator()` 默认启用 `RunAI model streamer`：
  - [envs.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/envs.py)
  - [runtime/loader/weight_utils.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/loader/weight_utils.py)
- `RunAI SafetensorsStreamer.get_tensors()` 是按 chunk ready 顺序吐 tensor，而不是严格按输入文件顺序串行遍历；在存在重复 key 的前提下，后到达的同名参数会覆盖先到达的参数
- `hf_to_custom_state_dict()` 对重复 key 的处理是“后者覆盖前者”，因此最终 live 参数会在 full / fp16 两套值之间摇摆：
  - [runtime/loader/utils.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/loader/utils.py)

### 最终确认实验

- 保持同机同卡同提交不变，只设置：
  - `SGLANG_USE_RUNAI_MODEL_STREAMER=false`
- 之后连续两次冷启动结果完全一致：
  - 最终图片 hash 一致
  - `block15.scale_shift_table` 一致
  - 整个 probe 输出一致
- 这证明根因不在 Sana kernel / attention / LayerNorm / Conv，而在 transformer 权重加载路径对重复 safetensors 文件的非确定性选择

### 已排除 / 低优先级嫌疑

- 单独把 Sana 中的 `LayerNorm` 强制改成 FP32 计算：
  - 同一进程内仍稳定
  - 跨进程重启仍然漂
  - 说明 `LayerNorm` 不是唯一主因
- 单独把 `SanaLinearAttention` 的核心 `matmul` 路径强制改成 FP32：
  - 同一进程内仍稳定
  - 跨进程重启仍然漂
  - 说明 self-attention 也不是唯一主因
- 在 `h200` alias 对应机器 (`124.158.103.2`) 上，把常见 deterministic 开关全部打开：
  - `torch.use_deterministic_algorithms(True, warn_only=True)`
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  - 关闭 TF32
  - `cudnn.deterministic=True`, `benchmark=False`
  结果仍然是：
  - 同一进程内一致
  - 跨进程重启后输出 hash 继续变化
  - 说明不是“少开了几个 deterministic 开关”这么简单

对应代码位置:

- [runtime/models/dits/sana.py](/Users/mick/repos/sglang/python/sglang/multimodal_gen/runtime/models/dits/sana.py)
  - `self.norm1 = nn.LayerNorm(...)`
  - `norm_hidden = self.norm1(hidden_states)`
  - `norm_hidden = norm_hidden * (1 + scale_msa) + shift_msa`

## 当前判断

- `sana` flaky 的直接根因不是模型前向 kernel 本身
- 真正根因是：
  - transformer loader 同时读取了 `full` 和 `fp16` 两套重复 safetensors
  - 在默认启用的 RunAI streamer 路径下，同名 key 的到达顺序不稳定
  - `hf_to_custom_state_dict()` 对重复 key 采用“最后一个覆盖前一个”的语义
  - 导致同一参数在不同进程启动里有时来自 full，有时来自 fp16
- 这类权重差异随后在 denoising 中被放大，最终表现为 Sana consistency 波动

## 修复方向

- 最小修复：
  - 对 transformer loader 也像 text encoder loader 一样，过滤掉重复 safetensors，只保留一套权重文件
- 兜底修复：
  - 在存在重复权重文件时，禁用 RunAI streamer，退回严格按文件顺序读取
