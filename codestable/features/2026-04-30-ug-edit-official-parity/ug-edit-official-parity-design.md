---
doc_type: feature-design
feature: 2026-04-30-ug-edit-official-parity
requirement:
roadmap: ug-official-alignment
roadmap_item: ug-edit-official-parity
status: approved
summary: 对齐 BAGEL image-edit 路径，证明输入图像条件、编辑 prompt、G denoise 和最终图像与官方框架一致
tags: [ug, bagel, image-edit, official-parity]
---

# ug-edit-official-parity design

## 0. 术语约定

- **Edit parity**：同 checkpoint、seed、输入图、编辑 prompt、sampling params 下，官方 BAGEL 与 SGLang UG 对齐 image-edit 的中间张量统计和最终图像 metadata/pixel summary。
- **cfg_img sidecar**：BAGEL image-edit 需要额外的 image-conditioned CFG branch。SGLang 侧必须通过 SRT-owned sidecar session/request 准备它，不能让 SGLD 或外部官方 Python 持有 KV。
- **SRT-owned edit session**：主 image+text context 和 cfg_img context 都由 SRT session/paged KV 管理；G velocity 仍只通过 native SRT executor 回调 ModelRunner。

## 1. 决策与约束

### 需求摘要

`ug-t2i-official-parity` 已经关闭纯文生图路径。`ug-edit-official-parity` 只关闭 image-edit：

- image-edit case 可以独立运行，不强绑 T2I。
- 输入图像走 SGLang UG/SRT native image prefill，不回退到 runtime import 官方 BAGEL helper。
- SGLang artifact 必须记录并断言 `srt_sidecar_request_count == 1`，证明 cfg_img branch 真由 SRT 准备。
- 真权重验收至少跑一次多步 edit quality probe，并把 official/SGLang 输出图拉到桌面人工查看。

本 feature 不做：

- 不进入多轮 U-G-U interleaved；那是 `ug-interleaved-official-parity`。
- 不产品化 batching。
- 不扩大 OpenAI 兼容 API。

### 挂载点清单

- `test/registered/scheduler/test_bagel_g_official_parity.py` — 补 image_edit 专用 artifact/counter 断言，允许 edit prompt env 覆盖。
- `codestable/roadmap/ug-official-alignment/*` — 验收后回写 `ug-edit-official-parity` 状态。

## 2. 验收闭环

最小闭环：

1. CPU/static 层面：harness 可只执行 `image_edit` case。
2. SGLang counter 证明：`prefill_count == 1`、`velocity_count == num_steps - 1`、`srt_sidecar_request_count == 1`、`append_image_count == 0`。
3. 真权重 image-edit quality probe：官方和 SGLang artifact report passed。
4. 输出图片落盘，能人工查看 SGLang 与 official 对照图。

Stop signal：

- 如果 `srt_sidecar_request_count == 0`，说明 image condition 没有进入 shared SRT context，暂停。
- 如果 edit 只能退化成 T2I 或重新 encode prompt/image，暂停。
- 如果 official 清晰但 SGLang 图像明显错位/噪声，暂停。
