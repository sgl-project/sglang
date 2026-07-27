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

GPU 实验完成后在此补充 immutable code SHA、MinWM main SHA、B200 实例、逐帧误差、
首次分叉位置和产物路径。
