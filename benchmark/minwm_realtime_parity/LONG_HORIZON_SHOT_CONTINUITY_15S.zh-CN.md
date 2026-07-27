# MinWM 15 秒镜头连续性评测

日期：2026-07-27

## 结论

在本轮 3 个 `15.375s` case 中，**没有发现无指令自动切换镜头**：

- idle pottery：主体、机位和工作室背景连续；
- forward forest：持续向前运动，树干和枝叶近距离掠过时有遮挡与运动模糊，但没有
  跳到另一镜头；
- yaw-right street：持续右转，建筑、道路和光照的空间关系连续。

这个结论只覆盖本轮 3 个 seed=42 case，不能证明所有 prompt、seed 和更长时长都不会
切镜。它能说明当前 `sink=4/window=20` serving preset 在约 15 秒、KV 已多次滚动后，
没有稳定复现此前担心的自动切镜问题。

本地三路播放器：

`results/0724-sink4-window20-15s-shot-continuity-b200/player/index.html`

## 实验合同

| 字段 | 值 |
| --- | --- |
| checkpoint | 0724 `global_step_011000/generator/model.pt` |
| SGLang runtime commit | `58ba27529903c4b62f3a32a80b50878c8b674922` |
| GPU | 单张 NVIDIA B200 |
| resolution / FPS | 832×480 / 24 |
| action type | `primitive_token_residual`，label 81 |
| DMD | 4 steps，CFG 关闭 |
| performance lane | dense attention、whole-DiT `torch.compile` |
| causal KV | sink=4，window=20 |
| seed | 42 |
| 每 case | 23 chunks，368 generated + 1 reference = 369 帧 |
| 编码时长 | 369 / 24 = 15.375 秒 |

MinWM 每个 chunk 生成 16 个 pixel frame，因此不能在不裁帧的情况下恰好生成 360 帧；
这里选择 23 chunks，实际覆盖的生成 horizon 为 15.33 秒，比 15 秒略长。

## 检查方法

每个 case 同时保留 MP4 和 lossless RGB NPY，并做三层检查：

1. FFmpeg scene filter 使用 `scene > 0.20` 检查硬切，三个视频均为 0 个命中；
2. FFmpeg `scdet` 逐帧计算，常用默认切镜阈值为 10，本轮最大值分别为
   `3.785 / 6.534 / 5.868`；
3. 对 lossless RGB 计算相邻帧 MAE、HSV histogram Bhattacharyya distance 和灰度
   SSIM，并人工查看逐秒 contact sheet 和所有高差分帧对。

| case | 最大 MAE | 最大 histogram distance | 最小 SSIM | 最大 scdet | 自动切镜 |
| --- | ---: | ---: | ---: | ---: | --- |
| idle pottery | 0.0419 | 0.1336 | 0.6411 | 3.785 | 未发现 |
| forward forest | 0.0872 | 0.2587 | 0.3123 | 6.534 | 未发现 |
| yaw-right street | 0.0624 | 0.1218 | 0.4281 | 5.868 | 未发现 |

forest 的最小 SSIM 出现在 frame 278（11.583s）。相邻帧显示枝叶快速扫过画面：
这是连续相机运动下的近景遮挡，不是场景替换。street 的低 SSIM 与持续 yaw 相符；
pottery 的峰值来自手部快速动作。

## Chunk 边界

如果问题来自 causal KV 滚动或 chunk 拼接，突变容易集中在 frame
`17, 33, 49, ...`。本轮边界相邻帧 MAE 如下：

| case | boundary median | non-boundary median | boundary max |
| --- | ---: | ---: | ---: |
| idle pottery | 0.0122 | 0.0093 | 0.0293 |
| forward forest | 0.0252 | 0.0244 | 0.0574 |
| yaw-right street | 0.0192 | 0.0324 | 0.0593 |

边界没有形成跨 case 一致的异常峰值；yaw-right 的边界中位数反而低于普通帧。这轮没有
看到由 `window=20` 滚动直接触发的可见切镜或固定周期跳变。

## 性能旁证

排除首 chunk 编译后，三个 case 的 scheduler p50 分别为
`400 / 395 / 394 ms/chunk`，对应 `40.0 / 40.5 / 40.6 FPS` 的模型侧生成速率。
本轮客户端通过 `kubectl port-forward` 传输每 chunk 约 19 MiB raw RGB，传输间隔不能
代表 NLB/WebP 产品吞吐，因此不把本机端到端 FPS 作为性能结论。

## 证据

- case manifest：`cases_long_horizon_15s_832x480.json`
- lossless frames、MP4、逐 case stats：
  `results/0724-sink4-window20-15s-shot-continuity-b200/cases/`
- 逐秒与高差分 contact sheet：
  `results/0724-sink4-window20-15s-shot-continuity-b200/analysis/`
- 平铺同步播放器：
  `results/0724-sink4-window20-15s-shot-continuity-b200/player/index.html`
