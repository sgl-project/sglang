---
doc_type: feature-design
feature: 2026-04-30-ug-sampling-cfg-parity
requirement:
roadmap: ug-official-alignment
roadmap_item: ug-sampling-cfg-parity
status: approved
summary: 对齐 BAGEL G denoise 的采样与 CFG 语义，先证明 timestep shift、CFG interval、text/img CFG gating 和 renorm 公式与官方一致
tags: [ug, bagel, sampling, cfg, official-parity]
---

# ug-sampling-cfg-parity design

## 0. 术语约定

- **denoise schedule**：BAGEL 官方 `generate_image` 使用的 shifted timestep 序列和相邻 `dt`。
- **effective CFG scales**：每个 timestep 上经过 `cfg_interval` gating 后真正传给 velocity forward 的 text/img CFG scale。
- **CFG branch semantics**：full/text-cfg/img-cfg 三条 velocity branch 的组合公式；按官方实现，`cfg_text_scale <= 1.0` 时 image CFG 不改变输出。

## 1. 决策与约束

### 需求摘要

Phase 4 的第一项不直接宣称最终 T2I/Edit 图像对齐，而是先把 G denoise 的采样语义固定下来：

- `timestep_shift` 公式与官方 `generate_image` 一致。
- `cfg_interval` 使用官方边界：`t > start and t <= end`。
- native SRT BAGEL G executor 的 full/text/img CFG branch 和 `global`、`channel`、`text_channel` renorm 公式与官方一致。
- opt-in G official parity harness 复用同一套 SGLang sampling helper，避免测试和 runtime 各自复制公式。

本 feature 不做：

- 不扩大到最终 T2I/Edit 图像验收；那分别由 `ug-t2i-official-parity` 和 `ug-edit-official-parity` 关闭。
- 不改变 SRT-owned KV 边界；SGLD 仍只请求 velocity，不接触 KV allocator/page/slot。
- 不打开多卡、CFG parallel 或 batching 产品化。

### 挂载点清单

- `python/sglang/srt/ug/sampling.py` — 新增 BAGEL denoise schedule 和 effective CFG scale helper。
- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/ug.py` — `UGDenoiseStage` 改为使用共享 schedule helper。
- `python/sglang/srt/ug/bagel.py` — native BAGEL executor 改为使用共享 CFG gating helper。
- `python/sglang/multimodal_gen/configs/sample/ug.py` — 拒绝 `timestep_shift <= 0`，避免官方公式首步 NaN。
- `test/registered/scheduler/test_bagel_g_official_parity.py` — opt-in official-vs-SGLang G harness 使用同一个 schedule helper。

## 2. 验收闭环

最小闭环：

1. CPU 单测证明 shifted timesteps/dts 与官方公式一致。
2. CPU 单测证明 `cfg_interval` 的左开右闭边界一致。
3. CPU 单测证明 native SRT BAGEL executor 的 CFG branch、image CFG gating 和 `text_channel` renorm 公式一致。
4. CPU 单测证明 UGPipeline G denoise stage 使用该 schedule，不再内联另一份公式。

Stop signal：

- 如果对齐 CFG 需要让 SGLD 读取/持有 KV page 或 slot，暂停重审。
- 如果 CFG branch 需要绕过 SRT ModelRunner 调 official BAGEL Python runtime，暂停重审。
- 如果最终图像 parity 失败但 sampling/CFG 单元语义已闭环，不在本 feature 内继续扩大范围，转入后续 T2I/Edit item。
