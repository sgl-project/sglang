# Dragon Ride 60 秒 bitwise 对齐报告

## 结论

在单卡 NVIDIA B200 上，minWM `main` baseline 与 SGLang realtime API 对 Dragon Ride 60 秒 case 的全部 1440 个生成帧实现了逐像素 bitwise 一致：

| 指标 | 结果 |
| --- | ---: |
| 生成分辨率 / 帧率 | 832×480 / 24 FPS |
| 参考帧 / 生成帧 | 1 / 1440 |
| latent 帧 / causal chunks | 360 / 90 |
| seed | 42 |
| 生成帧 bitwise | `true` |
| 最大绝对像素误差 | `0` |
| 最大 RMSE | `0.0` |
| 最低 SSIM | `1.0` |
| 两路 frame SHA256 | `c2ee85f8253b313ebee76c706d0c60844215f6d8e94c88c694c603c63d2fa82d` |
| 两路 MP4 SHA256 | `f83d8f2b5945f1aa450b626a2d1fe5b4c52ae3171356e4c6e44c9bb8eb95e1d4` |

结果页位于
`results/0724-dragon-ride-60s-main-unipc-native-bitwise/player/index.html`，支持一键同时播放/暂停、同步拖动、逐帧前后移动和变速播放。页面也显示本次 action timeline。

## 固定输入

### 代码、配置和权重

- minWM baseline：`main@ae2fd81ead16e05f79a57a2f47a4f3b8ea2391bc`
- baseline 配置：
  `/Users/chenshengdong/workspace/minWM/Wan21/configs/eval/wan22_5b_varlen60s_artf_v2.yaml`
- 配置 SHA256：
  `a5eac8a1478a5c9da63494439302f26671636b44445b0b9fd1496d59d8d2546d`
- checkpoint：
  `s3://leap-world-us-west-2/world-model/minwm/checkpoints/run-archive/rolling/Wan21/Action2V/bidirectional/wan22-5B-varlen-multishot-texiao-addsplithq-da25148-dmd-0724-5eba381389f-merge/global_step_011000/generator/model.pt`
- checkpoint version：
  `Byk70ZwuVy96DMkNuwe_dAHTtvICQCcC`
- checkpoint ETag：
  `d76d7a982b7eca7908e4b2a0fb4a4f6b-2386`
- checkpoint 大小：`20,014,135,255` bytes
- checkpoint CRC64：`8KS+pbHYSjY=`
- 首帧 SHA256：
  `efd7bc6bf4ef5b106a5e31b8cc78cf3a8f80e99d848aa225abb57d201e01f919`

### Action 时序

动作类型是 `primitive_token_residual`，输出格式是 `label_81`。像素帧边界是左闭右开：

| 时间 | 像素帧 | latent 帧 | 输入 |
| --- | ---: | ---: | --- |
| 0–5s | `[0, 120)` | `[0, 30)` | W |
| 5–10s | `[120, 240)` | `[30, 60)` | S |
| 10–15s | `[240, 360)` | `[60, 90)` | idle |
| 15–20s | `[360, 480)` | `[90, 120)` | W |
| 20–25s | `[480, 600)` | `[120, 150)` | S |
| 25–60s | `[600, 1440)` | `[150, 360)` | idle |

实际发给 API 的 action labels 是：

- W：label `9`，30 个 latent 帧
- S：label `18`，30 个 latent 帧
- idle：label `0`
- 完整序列：`W×30, S×30, idle×30, W×30, S×30, idle×210`

## 为什么最初没有对齐

指定的 `_v2.yaml` 使用：

```yaml
sample_solver: unipc
sampling_steps: 4
shift: 5
```

原来的 SGLang MinWM realtime 管线固定使用 DMD timestep 和 DMD/renoise 更新。第一次 60 秒比较实际上是 “minWM UniPC baseline vs SGLang DMD”，首个生成帧就已经明显不同，因此不能归因于 KV cache 的长序列误差。

增加 `MinWMCausalUniPCPipeline` 后，首个生成帧 RMSE 从 `6.797` 降到 `0.566`，但仍然没有 bitwise。进一步检查发现，SGLang 的通用 `FlowUniPCMultistepScheduler` 做了两项合理的吞吐优化：

1. 把 sigma 和标量运算留在 GPU，避免 CPU/GPU 同步。
2. 对二阶 corrector 使用手写 2×2 闭式解，替代 `torch.linalg.solve`。

这些运算在数学上等价，但浮点运算顺序不同。小误差进入 causal KV 后会在 90 个 chunk 中不断累积，最终全程最大 RMSE 达到 `41.355`。

## 最终实现

### 1. 独立的 realtime UniPC 管线

`MinWMCausalUniPCPipeline` 保留现有实时输入、T5、首帧 VAE、persistent KV、action history 和流式 VAE 路径，只替换 timestep preparation 与每个 chunk 的四步 scheduler 更新。

关键约束：

- scheduler 构造时 `shift=1`
- `set_timesteps(4, shift=5)` 时再应用配置 shift
- timestep 必须是 `[999, 936, 832, 624]`
- scheduler 边界保持 baseline 使用的 BFCHW layout
- transformer 内部继续使用 SGLang 的 BCFHW layout

如果在构造和 `set_timesteps` 时各应用一次 shift，四个 timestep 会全部错误。

### 2. MinWM 专用 native-numerics scheduler

`MinWMFlowUniPCParityScheduler` 只服务于 MinWM bitwise 管线。它精确保留 minWM main 的：

- CPU sigma 存储
- `torch.tensor(list_of_scalars, device=...)` 标量物化
- `torch.linalg.solve` corrector
- predictor/corrector 运算顺序

通用 `FlowUniPCMultistepScheduler` 没有被回退，其他模型仍可使用原来的 GPU 优化实现。

在 B200 上用 float32 和 bfloat16 合成输入逐步比较，4 个 scheduler step 都是 bitwise，所有 step 的 `max_abs=0`。

### 3. KV/window 语义

指定 `_v2.yaml` 的有效 generator 配置是：

- `local_attn_size=-1`
- `window_size=20`
- `sink_size=4`

但 minWM main 当前 causal inference 只消费 `local_attn_size`。当它为 `-1` 时，实际是无界 attention；`window_size=20` 不会变成 runtime sliding window，`sink_size=4` 也不会触发淘汰。

因此本次 SGLang 请求使用：

- `realtime_causal_window_size=null`
- `realtime_causal_sink_size=4`

这与 baseline 的实际 cache 语义一致。不能仅根据 YAML 中出现 `window_size=20` 就把 SGLang runtime window 设为 20。

## 与预期不同的地方

1. `_v2.yaml` 在最新 main 中没有同名 exp 配置。测试任务创建
   `configs/exp/wan22_5b_varlen60s_artf_v2.yaml -> wan22_5b_varlen60s_artf.yaml`
   符号链接，才能保持 eval 配置名不变并继承正确 exp。
2. checkpoint 名字含 `dmd`，但本次应该使用哪种推理 solver 由 eval YAML 决定；不能根据 checkpoint 路径名强制选择 DMD runtime。
3. 通用 UniPC 的 GPU 优化有可测吞吐价值，却不能用于 bitwise 模式。
4. 60 秒无界 attention 的 chunk 延迟不是常数：后段序列更长，单 chunk 从约 630ms 增长到约 900ms。
5. baseline 与最终 SGLang 生成的 MP4 文件也完全相同，不只是解码后的像素相同。

## 性能

环境：单卡 B200、PyTorch 2.11.0+cu130、CUDA 13.0、832×480、4-step UniPC、无界 attention。

| 指标 | 数值 |
| --- | ---: |
| baseline 生成耗时 | 92.985s |
| baseline 等效生成吞吐 | 15.49 FPS |
| SGLang TTFF（含首次 segment compile） | 13.40s |
| SGLang steady chunk p50 | 653.49ms / 16 帧 |
| SGLang steady scheduler 吞吐 | 22.70 FPS |
| SGLang 后 10 个 chunk | 约 17.7–18.6 FPS |
| SGLang 峰值显存 | 约 84,502 MiB |

专用 native-numerics scheduler 相比通用 GPU 优化 scheduler 的 steady p50：

- GPU 优化：661.24ms
- native numerics：653.49ms

这次测量中没有观察到吞吐下降（约 `+1.2%`，属于单次运行波动）。原因是 5B DiT forward 和随序列增长的 attention 占主导，四步 scheduler 的标量同步成本相对很小。这个结果不应外推到更小模型或更高并发场景。

24 FPS 的实时门槛在本次 60 秒无界 attention 测试中尚未达到：全程 steady 汇总为 22.70 FPS，而且尾段降到约 18 FPS。若需要稳定 24 FPS，应先让 baseline 明确定义并消费 bounded local attention，再在两端用相同 window/sink 重测。

## 复现与产物

- case：
  `benchmark/minwm_realtime_parity/cases_dragon_ride_60s_832x480.json`
- B200 job：
  `benchmark/minwm_realtime_parity/k8s/minwm_0724_dragon_ride_60s_bitwise_spot.yaml`
- scheduler 单测 job：
  `benchmark/minwm_realtime_parity/k8s/minwm_unipc_scheduler_parity_smoke_spot.yaml`
- 一键同步播放：
  `benchmark/minwm_realtime_parity/results/0724-dragon-ride-60s-main-unipc-native-bitwise/player/index.html`
- 机器可读报告：
  `benchmark/minwm_realtime_parity/results/0724-dragon-ride-60s-main-unipc-native-bitwise/dragon60-summary.json`

## 必须通过的测验

1. 为什么看到 checkpoint 路径里有 `dmd`，仍然不能直接选 DMD scheduler？
2. 本次四个 UniPC timestep 是什么？为什么 shift 不能应用两次？
3. BFCHW 与 BCFHW 分别在哪个边界使用？为什么 layout 也属于 parity contract？
4. 通用 SGLang UniPC 与 minWM 原生 UniPC 的两个主要数值实现差异是什么？
5. 为什么首帧 RMSE 只有 0.566，60 秒后却可以增长到 40 以上？
6. `local_attn_size=-1, window_size=20, sink_size=4` 在当前 minWM main 中的实际 KV 语义是什么？
7. 本次 action 时序在像素帧和 latent 帧上的五个切换边界分别是什么？
8. 怎样用 frame SHA256、MP4 SHA256、max_abs、RMSE、SSIM 共同证明 bitwise？
9. 为什么 steady 22.70 FPS 不能宣称已经稳定达到 24 FPS？
10. 如果以后要启用 bounded window，baseline 和 SGLang 各自必须先确认哪些参数真正被运行时代码消费？

通过标准：第 1–8 题全部正确；第 9–10 题能解释“平均吞吐、尾延迟、配置值和有效运行语义”的区别。
