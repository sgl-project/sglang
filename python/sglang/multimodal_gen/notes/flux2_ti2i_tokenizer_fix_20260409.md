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
