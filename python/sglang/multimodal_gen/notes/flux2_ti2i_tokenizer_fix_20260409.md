# FLUX.2 TI2I tokenizer 对齐记录（2026-04-09）

目标：
- 修复 `flux_2_ti2i` 在 `origin/main` 上相对 diffusers / official baseline 的明显精度漂移。
- 本次只在独立 worktree 上修改，并优先确认 text 输入链路是否回到官方默认路径。

环境：
- 本地 worktree:
  - `/private/tmp/sglang-flux2-ti2i-fix-20260409`
  - branch: `codex/flux2-ti2i-fix-20260409`
  - commit: `143d6740f3f55b1c5bf0ae15b6c1e6f264fc1765`
- 远端机器：
  - `85.234.79.62`
  - 单卡 H100
  - container: `sglang_mick`
  - 远端 worktree: `/tmp/sglang-flux2-ti2i-fix-20260409`

修改：
- `runtime/loader/component_loaders/component_loader.py`
  - 把 `TokenizerLoader` 里 FLUX.2 special-case 从
    - `AutoProcessor.from_pretrained(component_model_path)`
  - 改为
    - `AutoTokenizer.from_pretrained(component_model_path)`
- 新增回归测试：
  - `test/unit/test_flux_2_tokenizer_loader.py`
  - 目的：钉住 FLUX.2 tokenizer 必须走默认 `AutoTokenizer` 路径，其他模型仍保持 `padding_side="right", use_fast=True`

定位结论：
- `origin/main` 当前的 FLUX.2 tokenizer special-case 使用了 `AutoProcessor`。
- 这和之前的 bisect 结论不一致；已知正确方向是回到官方默认 tokenizer 行为，而不是 processor。
- 在本地环境里，`AutoProcessor.from_pretrained("black-forest-labs/FLUX.2-dev")` 还会直接报 `Unrecognized processing class`，进一步说明这条路径不稳。

远端验证：
- 先用一次 probe 故意走“非 FLUX.2 special-case”路径，对比官方默认 tokenizer：
  - `same_input_ids=false`
  - `same_attention_mask=false`
  - 说明 generic `padding_side/use_fast` 路径确实会把 tokenization 带偏
- 再在修复分支上强制命中 FLUX.2 special-case，对比官方默认 `AutoTokenizer.from_pretrained(...)`：
  - `same_input_ids=true`
  - `same_attention_mask=true`
  - `input_ids_sha256=36555b16a769143b55ba5b083ffcaebe795c6698be1efad3674a5d1d97c414db`
  - `attention_mask_sha256=56d2a8fad646ed814fbd04e06889129f6ea00d35e8accecb81edbcff39b28025`

当前精度对齐进度：
- 已完全对齐到 `text 输入 / tokenizer` stage。
- 即：`build_flux2_text_messages -> tokenizer.apply_chat_template -> input_ids / attention_mask`
  这一段已经和官方默认 tokenizer 路径一致。
- `text encoder hidden states / prompt_embeds / denoise / 最终图片` 本 turn 没有继续完整复拍。

残留问题：
- 远端 `python -m pytest ...` 受环境问题影响，未能作为最终验证依据：
  - 本地：缺 `triton.compiler`
  - 远端：`flashinfer-jit-cache` 版本检查需要 `FLASHINFER_DISABLE_VERSION_CHECK=1`
- `sglang generate` 的 `flux_2_ti2i` CLI probe 这次返回码是 0，但指定 `output_path` 下没有落文件。
  - 这更像 CLI 落盘/输出路径问题，不像 tokenizer 精度问题。
  - 由于 tokenizer probe 已证明 text 输入链路恢复到官方路径，本次先把 tokenizer 修复落地；若后续还要继续追最终图片差异，可从 CLI 输出落盘或 prompt_embeds dump 继续。

## 2026-04-09 12:16 CST 补充：默认尺寸语义

新发现：
- 当前 `Flux2SamplingParams` 仍然有 `width=1024`、`height=1024` 默认值。
- 对 `TI2I` 来说，这会让 `InputValidationStage.preprocess_condition_image(...)` 永远看见“已有输出尺寸”，从而不会像官方 diffusers 那样在默认路径上回退到输入图尺寸。
- 官方 `Flux2Pipeline.__call__` 的默认行为是：
  - 若用户没有显式传 `width/height`，则输出尺寸使用 condition image 经过 preprocess 后的尺寸。

这次修复：
- `configs/sample/sampling_params.py`
  - 把 CLI/API 路径里用户真正显式传入的字段记录到 `req.extra["explicit_fields"]`
- `runtime/pipelines_core/stages/input_validation.py`
  - 在 `TI2I` 分支中，只有 `width/height` 被用户显式传入时才保留它们；
  - 否则回退到 `prepare_calculated_size(final_image)`，再按 VAE multiple 对齐。

新增回归测试：
- `test/unit/test_sampling_params.py`
  - 钉住 CLI/API 路径会正确区分“默认尺寸”和“显式尺寸”
- `test/unit/test_input_validation.py`
  - 钉住 `TI2I` 未显式传尺寸时使用输入图尺寸，显式传尺寸时保留用户尺寸

当前精度对齐进度（更新）：
- 已完全对齐到 `text 输入 / tokenizer` stage。
- 已修复 `TI2I 默认输出尺寸语义`，使其与官方默认路径一致。
- 还没有完成对 `text encoder hidden states / prompt_embeds / image_latent / one-step noise_pred / 最终图片` 的新一轮默认参数复拍，所以当前最终图片精度仍视为“未对齐完成”。

## 2026-04-09 16:52 CST 补充：tokenizer 结论更正

之前错误结论：
- 之前把 FLUX.2 tokenizer special-case 从 `AutoProcessor` 改到 `AutoTokenizer`，并一度以为这和官方 diffusers 对齐。
- 这一步的对照对象拿错了，只对到了 `AutoTokenizer.apply_chat_template(...)`，没有对到官方 `Flux2Pipeline.encode_prompt()` 的真实输入路径。

重新 probe 后的事实：
- 官方 diffusers 当前真正走的是：
  - `PixtralProcessor.apply_chat_template(...)`
- 在远端容器中直接验证：
  - `AutoProcessor.from_pretrained(<...>/tokenizer)` 会返回 `PixtralProcessor`
  - 且具备 `apply_chat_template`
- 用同一份 stage probe 对拍官方 `encode_prompt()` 与 SGLang 当前路径时：
  - `input_ids` 不同
  - `attention_mask` 不同
  - `prompt_embeds` 随之不同

因此本次更正：
- FLUX.2 tokenizer special-case 应恢复为：
  - `AutoProcessor.from_pretrained(component_model_path)`
- 之前“tokenizer 已完全对齐”的结论作废，需要以新的官方 stage probe 为准继续推进。
