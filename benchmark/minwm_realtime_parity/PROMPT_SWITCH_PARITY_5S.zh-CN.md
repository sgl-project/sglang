# MinWM realtime prompt switch 5 秒 parity

更新时间：2026-07-27

## 结论

本轮补齐了此前缺失的 prompt switch 验收：不是给不同 session 传不同的初始
prompt，而是在同一个 WebSocket session 中先发送 `init`，再发送
`kind=prompt` 的 realtime event，并要求它从指定 causal chunk 开始生效。

单张 NVIDIA B200 上共跑 5 个 832×480、24 FPS、129 帧（5.375 秒）case：

- 1 个静态 prompt control；
- 4 个在 chunk 1 切换到 snowy night、neon rain、overgrown noon、sandstorm
  prompt 的 case。

最终 5/5 通过：

- MinWM V3 baseline 与 SGLang realtime API 的全部 lossless RGB 生成帧
  bitwise exact；
- `max_abs=0`、`RMSE=0`、`SSIM=1.0`；
- 4/4 prompt event 都在 frame header 和 chunk stats 中首次命中 chunk 1；
- 4/4 case 在切换前与静态 control bitwise exact；
- 4/4 case 在切换后都与静态 control 产生了明显差异。

本地双路同步播放器：

`results/0724-window18-sink9-5s-prompt-switch-b200-bitwise/player/index.html`

## 实验合同

- checkpoint：0724
  `global_step_011000/generator/model.pt`，大小 `20,014,135,255` bytes；
- MinWM main：`0796bc201fae4c86f100620cb23402ae21c8f3b5`；
- SGLang harness/runtime：`294c224b487049d4c57951f2f62b03951b26043f`；
- Kubernetes Job：`minwm-0724-prompt-switch-20260727-01`；
- run id：
  `0724-window18-sink9-5s-prompt-switch-b200-attempt01`；
- seed：42；
- action type：`primitive_token_residual`，本轮使用 idle label 0，避免把
  action 变化误认为 prompt 变化；
- MinWM V3：`local_attn_size=18`、`sink_size=9`；
- SGLang runtime：`window=18`、`sink=9`；
- CFG 关闭；
- whole-DiT `torch.compile` 关闭；
- 两边使用相同的确定性 packed attention 路径。

## 边界如何对齐

一个 SGLang realtime chunk 包含 4 个生成 latent frame。Wan VAE 的 temporal
factor 为 4，且视频最前面还有 1 个 reference frame，因此 chunk 1 的起点是：

```text
pixel boundary = 1 reference + 1 completed chunk × 4 latent/chunk × 4 pixel/latent
               = 17
latent boundary = 1 reference latent + 1 × 4
                = 5
```

baseline 通过 MinWM `text_prompt_interval` 构造两段：

```text
[0, 17)   initial prompt
[17, 129) switched prompt
```

MinWM processor 必须把它映射成 `prompt_seqlens=[5, 28]`。harness 在 GPU
推理前对 prompt 文本和这两个 latent 长度做硬断言。

SGLang 客户端发送 `init` 后立即把 prompt event 排入同一个 WebSocket。MinWM
realtime adapter 明确不在 chunk 0 采样 prompt event，因此排队事件的第一个合法
消费点是 chunk 1。客户端不依赖 `sleep`，并在会话结束时同时验证：

```text
first chunk_stats.event_id == target event id -> chunk 1
first frame_batch.event_id == target event id -> chunk 1
```

如果任意一条证据首次出现在其他 chunk，case 会在保存为成功结果前直接失败。

## 为什么还需要静态 control

只验证 baseline 与 SGLang 相同还不够：两边都忽略新 prompt 时也可能“相同”。
因此四个切换 case 都引用同一个静态 control，并增加两条正交检查：

1. `[0, 17)` 必须与 control bitwise exact，证明 event 没有提前影响 chunk 0；
2. `[17, 129)` 必须与 control 不同，证明新 prompt 确实影响了生成。

| 目标 prompt | 切换前 bitwise | 切换后 changed value fraction | 切换后 SSIM vs control |
| --- | --- | ---: | ---: |
| snowy night | true | 0.8193 | 0.7019 |
| neon rain | true | 0.9292 | 0.5482 |
| overgrown noon | true | 0.8179 | 0.6575 |
| sandstorm | true | 0.9323 | 0.4569 |

这些指标比较的是“切换输出 vs 静态 control”，用于证明 prompt 有效果；它们不
是 baseline/SGLang parity 误差。baseline 与 SGLang 之间仍是逐位相同。

## 性能观察

静态 control 的 chunk 1 scheduler 时间为 669 ms。四次 prompt switch 的
chunk 1 分别为 753/751/749/751 ms，平均 751 ms：

- 切换块一次性增加约 82 ms；
- 相对静态块约增加 12.3%；
- 原因是新 prompt 需要编码，并重建 cross-attention condition KV；
- chunk 2–7 恢复为平均 660.125 ms / 16 帧，即 24.238 FPS；
- 包含一次切换块时，chunk 1–7 平均为 23.770 FPS；
- 切换后的 chunk 2–7 WebSocket payload 到达率为 24.264 FPS；
- 排除首次 compile session 后，平均 TTFF 为 1,004.2 ms。

所以 prompt switch 有一个可测的一次性 cutover 成本，但不会持续拉低后续
steady-state 吞吐。

## 重大实现决策

1. manifest 新增可选 `prompt_switch`，包含 `target_chunk`、`event_id`、
   `prompt` 和 `control_case_id`。原有静态 prompt/action case 不受影响。
2. baseline 与 API 共用同一份 manifest，而不是维护两份手写切换时间，避免
   pixel/latent/chunk 边界漂移。
3. API 客户端同时保存事件发送时间、frame header event id 和 chunk stats
   event id，使结果可以离线审计。
4. report 把 parity、事件边界和 prompt 实际效果作为三个独立的通过条件。
5. 播放器保持全部 case 平铺；切换 case 显示 chunk/frame 边界，并提供
   `Jump to switch`。

## 与预期不同或仍有限制的地方

- 之前的 realtime control 调查已证明 prompt event 会改变输出，也修复了
  event id 关联，但那组证据使用 WebP payload，且没有与当前
  window=18/sink=9 的 MinWM V3 raw RGB baseline 做 5 秒 parity。本轮才补齐这
  个验收缺口。
- prompt event 不是“收到后立刻在任意 frame 生效”，而是只能在 causal block
  边界消费；本轮的可见切点是 frame 17，不是 event 发出时的墙钟时刻。
- 当前 parity harness 有意只允许 `target_chunk=1`。如果客户端等收到 chunk N
  才发送事件，server 可能已经开始准备 chunk N+1，会把网络/调度竞态混进模型
  parity。要可靠测试更晚的任意 chunk，应给协议增加服务端
  `apply_at_chunk`/ack，而不是用 `sleep` 猜时序。
- prompt switch 块不是零成本；本轮实测有约 82 ms 的一次性额外开销。这与
  “切换后 steady-state 仍约 24 FPS”并不冲突。

## 证据位置

- 本地播放器和精简证据：
  `results/0724-window18-sink9-5s-prompt-switch-b200-bitwise/`
- 本地 machine-readable report：
  `results/0724-window18-sink9-5s-prompt-switch-b200-bitwise/report.json`
- 集群完整结果（含 lossless NPY、latent 和逐 forward dump）：
  `/work/parity-results/0724-window18-sink9-5s-prompt-switch-b200-attempt01`
