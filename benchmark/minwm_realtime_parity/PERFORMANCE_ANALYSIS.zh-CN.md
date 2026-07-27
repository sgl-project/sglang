# MinWM 与 LingBot World 2.0 实时推理性能复盘

日期：2026-07-27

## 结论

当前材料不能支持“同硬件、同分辨率、同并行度下，MinWM 比 LingBot World
2.0 慢两倍”这个结论。

本分支里 `24.713 FPS` 对应的不是 LingBot World 2.0 权重，而是：

- 同一份 MinWM 5B 0721 checkpoint；
- 单张 B200；
- `832x480`；
- KV 45；
- 将 MinWM exact packed attention 换成 LingBot-style dense attention 的消融。

`10.822 FPS` 则是单张 H200、`1248x704`、完整历史 KV、bitwise 路径。两次测试
同时改变了 GPU、分辨率、attention 路径和 KV 合同。`1248x704` 的像素数及 MinWM
DiT token 数都是 `832x480` 的 `2.20x`，因此 `24 -> 10 FPS` 的主要原因是工作量和
硬件口径，而不是 5B 实现凭空多出了一倍开销。

真正同口径的 B200、`832x480` 数据是：

| MinWM profile | client FPS | 相对 exact |
| --- | ---: | ---: |
| exact packed deterministic | 23.075 | 1.000x |
| LingBot-style dense、native component | 24.713 | 1.071x |
| dense、SGLang component、eager | 25.541 | 1.107x |
| dense、SGLang component、whole-DiT compile | 32.222 | 1.396x |

所以，在相同 checkpoint、GPU、请求下，能归因到 MinWM exact/parity 实现的已测差距
约为 `7.1%`，而不是 `2x`。解除 bitwise 约束后，现有 whole-DiT compile 性能路径
已经把 `23.075` 提高到 `32.222 FPS`。

## 为什么“5B 应该比 14B 快”不能直接套用

LingBot World 2.0 的公开及 SGLang 部署方式使用 8 卡 Ulysses，并为 VAE 打开 spatial
parallel decode。MinWM 的 `10.822 FPS` 是单卡 H200。比较总参数量时忽略了并行度。

用主线性层 MAC 作为粗略代理：

```text
每层每 token MAC ~= 6*d^2 + 2*d*ffn
每 chunk 有 4 次 DMD forward + 1 次 clean-latent KV commit forward
```

在 `1248x704` 下：

| 模型/并行 | 每个 chunk 每卡的 DiT MAC 代理 |
| --- | ---: |
| MinWM 5B，SP1 | 74.49 TMAC |
| LingBot 14B，SP8 | 76.92 TMAC |

两者每卡工作量实际上几乎相同。原因是 LingBot 的 14B 参数和更长 token sequence 被
8 卡拆分了。这个估算没有包含 attention 二次项、通信、VAE 和输出处理，因此不能代替
实测，但足以否定“14B 对 5B，所以单流一定更慢”的推理。

两者也不是相同的执行形状：

| 项目 | MinWM 5B | LingBot World 2.0 14B |
| --- | ---: | ---: |
| hidden size / layers / heads | 3072 / 30 / 24 | 5120 / 40 / 40 |
| FFN | 14336 | 13824 |
| latent frames/chunk | 4 | 3 |
| 720p patch tokens/chunk | 3432 | 10296 |
| SP8 后 tokens/rank | 429 | 1287 |
| SP8 后 heads/rank | 3 | 5 |
| VAE z / decoder base width | 48 / 256 | 16 / 96 |
| VAE | residual Wan2.2，spatial 16 | Wan，spatial 8 |

MinWM 的序列太短、head 太少，SP8 会把 GEMM 切得过小，同时每层仍需 all-to-all。
这解释了实测 B300 上 SP2/SP4/SP8 都比 SP1 慢约 4–5%。LingBot 的执行形状对 SP8
明显更友好。

## 已确认的性能瓶颈

### 1. parity attention 与算子边界

MinWM exact 路径使用 source-shaped packed varlen attention、确定性设置和为 bitwise
保留的 FP32/FP64、stride/materialization 边界。相同 MinWM 权重换成 dense attention
已经实测提升 `7.1%`。既然不再要求 bitwise，应删除性能 lane 对这些边界的依赖。

### 2. whole-DiT compile 尚未吃满

whole-DiT compile 的收益已实测：

- B200 `832x480`：`25.541 -> 32.222 FPS`，提升 `26.2%`；
- H200 `1248x704`：`10.853 -> 12.711 FPS`，提升 `17.1%`。

但编译日志仍在 KV cursor 的 `Tensor.item()` 处 graph break。LingBot 从初始化起就使用
Python integer cursor；MinWM 单卡路径只在 sequence shard 时启用 integer cursor。
这会妨碍 fullgraph/CUDA Graph 和稳定 shape 优化。

### 3. LingBot fast path 没有完整移植到 MinWM

LingBot 实现已有而 MinWM 尚缺：

- self-attention Q/K/V 合并为一次 `F.linear`；
- RoPE 按 shape/start-frame 缓存；
- 4 个固定 DMD timestep 的 embedding 缓存；
- cross-attention cache 初始化后跳过重复 text projection；
- 第五次 clean-latent cache commit 在最后一层只更新 K/V，跳过该层尾部和最终
  output projection。

MinWM 还会在 5 次 transformer forward 中重复执行 action encoder，并把 action
state 展开/materialize 成整个 token residual。720p 下该 residual 每次约 21 MiB。

### 4. causal VAE 没有进入通用 compile 路径

通用 `DecodingStage` 会编译 `vae.decode`；realtime causal VAE 为维护跨 chunk feature
cache，直接逐 latent frame 调用 `post_quant_conv` 和 `vae.decoder`，绕开了上述
compiled callable。因此 `--enable-torch-compile` 当前主要优化 DiT，没有编译
steady-state causal decoder。

MinWM 的 residual VAE 也比 LingBot VAE 宽得多。模型都叫 Wan VAE，不代表 decoder
成本相同。

### 5. VAE 尾部与输出物化仍在关键路径

720p 每个 steady chunk 输出 16 帧 raw RGB，共 `42,172,416` bytes。当前实现执行：

```text
GPU tensor
  -> clamp/mul/uint8
  -> NHWC contiguous
  -> blocking CPU numpy
  -> 16 次 frame.tobytes()
  -> WebSocket
```

日志中 `sample_to_frames` 为 H200 `318–360 ms/chunk`、B300 约 `205 ms/chunk`。
注意：默认 stage 计时没有 CUDA synchronize，这个数同时吸收了之前未完成的 VAE
kernel，不能把它全部归因给 uint8/D2H。可靠结论是“VAE 尾部 + 输出物化”是大块关键
路径；纯 D2H、转换 kernel 和 VAE 各占多少必须用 CUDA events/Nsight 重测。

逐帧 `tobytes()` 本身通常只有约 `4–20 ms`。42 MiB 的 PCIe 传输理论上也不应达到
200 ms，因此第一优先级不是微调 Python WebSocket，而是拆清 GPU completion、转换和
同步拷贝。

### 6. KV 与 Ulysses 不是当前 5 秒 case 的主解

B200 `832x480` 上 KV128 与 KV45 的 FPS 仅差约 `0.39%`，但 KV45 明显省显存。短 case
中 projection/FFN 占主导；长 session 中 full-history attention 才会持续恶化。

B300 720p 实测：

| Ulysses | client FPS | 相对 SP1 |
| ---: | ---: | ---: |
| 1 | 15.891 | 1.000x |
| 2 | 15.207 | 0.957x |
| 4 | 15.191 | 0.956x |
| 8 | 15.109 | 0.951x |

因此不能再把“提高 Ulysses degree”当成 MinWM 单流达到 24 FPS 的方案。

## 优化方案

### P0：先建立真正同口径基线

同时跑真实 LingBot checkpoint 和 MinWM checkpoint，而不是把
`LingBot-style dense` 当成 LingBot 模型。固定：

- 同一 GPU 型号和 GPU 数量；
- `1248x704`、4 DMD steps、相同输出时长；
- 相同 warmup/chunk 数；
- 相同 raw 或 encoded transport；
- 分别报告纯 DiT、VAE decode、output materialization、网络写入和端到端 FPS；
- 明确 local attention、sink、KV window 及 Ulysses/VAE parallel degree。

至少提供四组：

1. MinWM SP1；
2. MinWM 8 个独立 DP replica 的 aggregate throughput；
3. LingBot SP8；
4. 同一 MinWM 权重的 exact、dense、compiled 消融。

Nsight 在 20 个 warmup chunk 后，只 capture 10 个 steady chunk。需要查看 GPU active、
Tensor Core/GEMM、kernel launch gap、all-to-all、VAE convolution、D2H 和 CPU 等待。
当前仓库没有 `.nsys-rep`，本机也没有 NVIDIA GPU/`nsys`，所以现阶段没有把
occupancy 或通信百分比伪装成已测数据。

### P1：低风险、立即可做

1. 默认性能 lane 使用 dense LocalAttention 和 SGLang native operators，删除
   packed/parity-only materialization。
2. 打开 whole-DiT `torch.compile`；预热和首次 compile 时间单独报告。
3. MinWM 固定长度 session 预分配 KV，单卡也使用 integer cursor，消除 `.item()`
   graph break。
4. 性能 lane 使用固定 KV45/64；另跑长时质量回归决定 sink 和 window。

现有实测证明 P1 的组合在 B200 480p 可提升 `39.6%`，在 H200 720p 可提升 `17.5%`。

### P2：移植 LingBot transformer fast path

1. fused self-attention QKV；
2. cache RoPE、time embedding、text projection；
3. clean-context 第五次 forward 使用 cache-only final block；
4. action encoder 每 chunk 只算一次 frame states，5 次 forward 复用；
5. 固定 shape 后评估 `torch.compile(fullgraph=True)` 和 CUDA Graph。

预期收益必须通过逐项 A/B 验证。工程目标可设为 DiT 再降 `8–15%`，但这不是当前
实测值。

### P3：重做 causal VAE 和输出关键路径

1. 把 `first_chunk=False` 的单 latent-frame causal decode 封装成显式固定 buffer
   callable，单独 `torch.compile`；
2. 验证 residual VAE 与 spatial parallel decode 的 feature-cache halo 是否正确，
   显式 sweep decoder SP1/2/4；
3. 用一个 fused CUDA kernel 完成 clamp、scale、uint8、NCTHW->NTHWC；
4. 使用 pinned-memory ring buffer 和独立 CUDA stream 做 async D2H；
5. 不再为每帧复制一个 Python `bytes`，以一个 contiguous chunk/memoryview 发送；
6. 产品路径优先 NVENC H.264/AV1；raw RGB 只保留为质量/一致性测试路径。

### P4：多卡不要继续硬套 Ulysses

单流低延迟优先做 denoiser/decoder pipeline：

```text
GPU 0: DiT chunk N ---- DiT chunk N+1 ---- DiT chunk N+2
                  \          \
GPU 1:             VAE/output N ---- VAE/output N+1
```

每个 MinWM 720p latent chunk 只有约 `1.32 MiB` BF16，跨卡传 latent 远比传 42 MiB
RGB 便宜。GPU0 保持 DiT/KV/action state，GPU1 保持 causal VAE state。steady-state
关键路径由“DiT + VAE/output”变为二者最大值加小额传输。

多 session 的总吞吐则使用 DP replicas；Ulysses 只在序列足够长、通信被计算覆盖时
考虑。

### P5：精度换速度

在 P1–P4 后仍未达到目标时：

- DiT 使用 FP8 weight/activation，norm、scheduler、action encoder 和 VAE 保留 BF16；
- B300 可另设 NVFP4 实验 lane，但不作为默认质量路径；
- 每项通过 10 个 prompt/seed/action case 的视频质量回归，而不是 bitwise。

建议门槛：逐帧 SSIM/PSNR/LPIPS、时序一致性、action responsiveness 和人工盲评同时
通过；不能只看单帧 cosine。

## 24 FPS 的现实路径

24 FPS、16 帧/chunk 要求 steady chunk latency 不超过 `666.7 ms`。

- H200 720p compiled 已测为 `12.711 FPS`，单卡 H200 尚远。
- B300 720p exact SP1 已测为 `15.891 FPS`。
- 将 H200 上已测 compile 比例机械乘到 B300，可得约 `18.67 FPS`。这是推断，不是
  B300 compile 实测。
- 要让单张 B300 达到 24 FPS，还需在该推断基础上再降约 `22%` latency。最可能来自
  causal VAE compile、输出异步化、LingBot transformer fast path 和 FP8 的组合。
- 两张 B300/H200 做 DiT 与 VAE/output 流水，是比 Ulysses 更可靠的 24 FPS 路径。

## 验收矩阵

每个优化单独一个 lane，固定 10 个 5 秒、1248x704 case：

| lane | 作用 |
| --- | --- |
| A | 当前 dense eager |
| B | A + whole-DiT compile |
| C | B + LingBot transformer fast path |
| D | C + compiled causal VAE |
| E | D + async output |
| F | E + FP8 |
| G | E + 两卡 DiT/VAE pipeline |

每个 lane 报告：

- cold compile、warm TTFF、steady p50/p95/p99 chunk latency、client/scheduler FPS；
- DiT/VAE/output/transport CUDA event 时间；
- GPU active、Tensor Core、kernel launch gap、collective、D2H；
- peak per-GPU memory 和总显存；
- 10-case 视频质量及 action 响应；
- 单流 FPS与多 session aggregate FPS分开。

性能目标优先级：

1. 单 B300 720p：先达到 `>=20 FPS` 且质量门槛通过；
2. 单 B300 FP8：冲刺 `>=24 FPS`；
3. 双卡 DiT/VAE pipeline：稳定 `>=24 FPS`；
4. 8 卡总吞吐：使用 DP replicas，避免为短序列使用 SP8。
