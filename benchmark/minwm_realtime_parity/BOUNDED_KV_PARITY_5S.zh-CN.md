# MinWM V3 与 SGLang 有界 KV 5 秒 Parity

更新时间：2026-07-27

## 实验合同

本实验只改变 KV attention 历史语义，其余 parity 条件保持一致：

- MinWM V3 baseline：`generator_config.local_attn_size=18`、
  `generator_config.sink_size=9`。
- SGLang：请求下发 `realtime_causal_kv_cache_num_frames=18`、
  `realtime_causal_sink_size=9`。
- `window=18` 是总可见 latent-frame 容量，包含 sink、最近历史和当前 chunk，
  不是 `18 + 9`。当前 chunk 为 4 帧，发生滚动后对应
  `9 sink + 5 recent history + 4 current = 18`。
- case：`cases_action_control_kv_roll_832x480.json`，832×480、24 FPS、
  1 张首帧加 128 张生成帧，共 129 帧 / 5.375 秒、8 chunks。
- checkpoint：0724 `global_step_011000/generator/model.pt`。
- action：`primitive_token_residual`，固定 `l` / label 1。
- seed：42；CFG 关闭；whole-DiT `torch.compile` 关闭。
- attention：两边使用相同的确定性 packed attention 路径。

5 秒 case 的总 latent horizon 为 33 帧；它明显超过 18 帧窗口，因此会真正触发
KV roll。短于窗口的 case 即使错误地实现成 full history 也可能伪装成 parity，
不能用于本实验结论。

## 验收顺序

1. 先检查 baseline 运行记录里的实际 `local_attn_size/sink_size`。
2. 检查 SGLang 请求记录里的实际 runtime `window/sink`。
3. 比较 lossless `baseline.npy` 与 `sglang.npy`：
   先要求全部生成帧 bitwise 相同。
4. 如果 bitwise 失败，保留严格失败结果，再使用预先存在的
   `bf16_backend_candidate` 数值阈值报告 `max_abs/RMSE/SSIM`；不得根据本次结果
   临时放宽阈值。
5. 同时保留双方逐 forward parity dump，用于定位首次分叉发生在 KV roll 前还是
   roll 后。

## 结果

最终结果：**bitwise parity 通过**。

- SGLang immutable SHA：
  `85fbea66e153ac2639a33cc6ef1abc3b14387351`。
- MinWM main immutable SHA：
  `0796bc201fae4c86f100620cb23402ae21c8f3b5`。
- Kubernetes Job：
  `minwm-0724-bounded-parity-5s-20260727-04`。
- GPU：单卡 NVIDIA B200。
- baseline 和 SGLang 均输出 832×480、24 FPS、129 帧、5.375 秒。
- 两路 lossless RGB frame SHA256 均为
  `9b38907753915c7f67ad22d8cbe2e74f94f849d88f69531edbbe6761a37b2537`。
- 两路 MP4 SHA256 均为
  `af04a49604bd0a2c1317a792d354820b0dabb8736e5c5b9e5aabd1c2ea7e3189`。
- 全部帧和生成帧均为：
  `bitwise_equal=true`、`max_abs=0`、`RMSE=0`、`SSIM=1.0`。
- 第一层的 block input、self-attention output、self residual、affine
  LayerNorm output、cross Q 和 block output 逐 tensor 比较也全部 bitwise
  相同。

性能数据不是独立的吞吐 benchmark，但可用于确认实时性：首个 payload
`13.799 s`，其中包含首次 segment compile；后续 7 个 chunk 的 scheduler
时间为 `672/666/667/668/667/668/667 ms`，每个 chunk 16 张视频帧，平均
steady-state 为 `23.96 FPS`。

本机可直接打开的双路同步播放器：

`results/0724-window18-sink9-5s-b200-bitwise/player/index.html`

集群完整证据（包括约 700 MiB 的逐 forward dump）：

`/work/parity-results/0724-window18-sink9-5s-b200-attempt04`

## 首次失败、根因与修复

首轮 `attempt01` 的 18/9 配置实际生效，但生成视频未通过数值阈值：

- `max_abs=230`
- `RMSE=14.898077`
- `SSIM=0.795517`

逐 forward dump 证明误差在第一个 DiT forward、KV 尚未发生滚动时就已出现。
patch、time embedding、block input、self Q/K/V 和 self-attention output 都完全
一致；第一个差异出现在 self-attention residual 后的 affine LayerNorm 输出，
随后经 cross-attention 和自回归状态逐步放大。因此，18/9 KV roll 不是误差源。

MinWM 的加载器会先把整个 generator 转为 BF16，再调用编译后的
`WanLayerNorm._norm`。SGLang 的通用 LayerNorm 容器则可能以 FP32 保存 affine
参数。最初把 `weight/bias -> BF16` 放在编译函数内部，结果没有变化；这是本次
最重要的反预期点。将转换移到 `torch.compile` 图外后，compiled function 收到的
operand dtype 和计算图形状才与 MinWM 完全一致，最终恢复 bitwise parity。

这个修复不关闭 segment compile，也没有打开 whole-DiT compile；它只移动两个
很小的 affine 参数 cast，不改变 attention、KV cache 或视频调度语义。

## 实验设施决策

- `attempt02` 为保留首轮证据而复制了第二份 32 GiB donor 和 19 GiB checkpoint，
  导致转换权重时本地 NVMe 空间不足。该失败属于实验基础设施问题，不计入 parity
  结果。
- 后续 Job 支持复用经过 SHA256 和 pinned metadata 校验的 immutable staged
  checkpoint/donor，也可复用未变化的已转换 SGLang 模型。`attempt04` 的输入
  staging 时间因此从 672 秒降为 0 秒，且 baseline 三个 hash 与首轮完全一致。
