---
doc_type: feature-design
feature: 2026-04-30-ug-t2i-official-parity
requirement:
roadmap: ug-official-alignment
roadmap_item: ug-t2i-official-parity
status: approved
summary: 对齐 BAGEL text-to-image 路径，证明 SGLang UG 的 init latent、多步 velocity、VAE decode 和最终图像与官方框架一致
tags: [ug, bagel, t2i, official-parity]
---

# ug-t2i-official-parity design

## 0. 术语约定

- **T2I parity**：同 checkpoint、seed、prompt、sampling params 下，官方 BAGEL 与 SGLang UG 对齐 text-to-image 的中间张量统计和最终图像 metadata/pixel summary。
- **quality probe**：用足够多 denoise steps 的真权重 T2I 运行，确认不是短步数噪声 smoke。quality probe 仍然以 official-vs-SGLang 对齐为准，不把审美质量当作单独指标。
- **SRT-owned session**：T2I 的 prompt/U context 必须由 SRT session/paged KV 管理，G velocity 只能通过 native SRT executor 调 ModelRunner。

## 1. 决策与约束

### 需求摘要

`ug-sampling-cfg-parity` 已经证明 BAGEL denoise schedule 和 CFG 语义对齐，但当时 2-step probe 的图像仍然是噪声态，不能算正式生图结果闭环。本 feature 只关闭 text-to-image：

- 从现有 official-vs-SGLang G parity harness 中拆出可单独运行的 T2I case。
- T2I case 记录 `init_noise`、`velocity_00`、`final_latents`、`generated_image_pixels`。
- SGLang artifact 必须带 debug counters，证明 `prefill_count == 1`、`velocity_count == num_steps - 1`，且没有 append/edit 路径混入。
- 真权重验收至少跑一次更接近质量验证的 T2I 参数，而不是只跑 2-step 噪声 smoke。

本 feature 不做：

- 不关闭 image-edit；edit 由 `ug-edit-official-parity` 单独处理。
- 不做 interleaved U-G-U；那是 Phase 5。
- 不把 official BAGEL/seed 函数 import 到 runtime。official code 仍只允许在 opt-in test runner 里作为 reference。

### 挂载点清单

- `test/registered/scheduler/test_bagel_g_official_parity.py` — 拆出 task filter、T2I case builder、T2I 专用 counter/assertion。
- `codestable/roadmap/ug-official-alignment/*` — 验收后回写 `ug-t2i-official-parity` 状态。

## 2. 验收闭环

最小闭环：

1. CPU/static 层面：harness 可只构造并执行 `text_to_image` case，不再强绑 edit。
2. 真权重 T2I low-res/quality probe：官方和 SGLang artifact report passed。
3. SGLang counter 证明：同一 session、`prefill_count == 1`、`velocity_count == num_steps - 1`、`append_image_count == 0`。
4. 输出图片落盘，能人工查看 SGLang 与 official 对照图。

Stop signal：

- 如果 official 清晰但 SGLang 噪声/模糊，暂停，不进入 edit。
- 如果 T2I 需要每步重新 prefill prompt，暂停。
- 如果 SGLD 需要拿 KV allocator/page/slot，暂停。
