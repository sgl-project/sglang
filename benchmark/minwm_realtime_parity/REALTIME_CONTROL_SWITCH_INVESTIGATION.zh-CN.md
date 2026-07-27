# MinWM 同 session prompt / action 切换调查

更新时间：2026-07-27

## 1. 问题定义

用户报告的核心差异是：

- MinWM `inference_v3.py` 中，prompt switch 和 action switch 都能改变后续生成；
- SGLang MinWM WebSocket session 已经进入 Live 后，中途修改 prompt 或操作
  WASD / IJKL（方向键）看不到预期变化。

这里的验收对象是同一个 session 的动态链路，不是“分别用 idle 和 action 启动两个
独立请求后，视频是否不同”。后者只能证明 checkpoint 有 controllability，不能证明
实时事件生效。

动态链路拆成五段：

```text
Web UI event
  → realtime adapter 采样到某个 chunk
  → condition_inputs / prompt embeddings
  → DiT action residual / cross-attention KV
  → 返回帧 event_id 驱动播放器 cutover
```

## 2. 参考实现版本和语义

本次重新核对的 minWM `main` 是：

```text
commit: 0796bc201fae4c86f100620cb23402ae21c8f3b5
entry:  Wan21/wan_inference.py
class:  UnifiedDiffusionInferencePipelineV3
```

`inference_v3.py` 的动态条件合同是：

- prompt segment 只在新的 causal block 边界切换；
- 切换块第一次 forward 传 `condition_switch="prompt"`；
- KV cache 收到该事件后清空 text cross-attention KV，并在本次 forward 用新
  prompt 重新建立；
- action 按当前 block 的绝对范围做
  `action[:, span_start:span_end]`，并与已经 final 的 action history 拼接；
- action switch 不清空 self-KV 或 cross-KV，它改变当前块的
  `primitive_token_residual`。

SGLang 的对应路径是：

| inference_v3 | SGLang |
| --- | --- |
| prompt segment | `prompt` realtime event + `RealtimeTextEncodingStage` |
| `condition_switch="prompt"` | `minwm_prompt_updated` |
| `cache.reset_cross()` | `_reset_crossattn_cache(...)` |
| 当前 action slice | `minwm_action_labels` / `minwm_action_weights` |
| action history | session-local `minwm_action_history` |

因此设计上两者支持相同的 prompt/action 动态切换；问题在事件时序和可观测结果，
不是“MinWM 不支持实时条件”。

## 3. 已确认的问题

### 3.1 prompt 已被采样，但返回帧携带旧 action event ID

线上 session 的日志给出了可复现证据：

```text
receive prompt event_id=144
next chunk condition_kinds=[minwm_action_labels, minwm_prompt_updated]
next chunk event_id=143
```

`MinWMRealtimeAdapter.get_realtime_event_id()` 只检查 action label、action weight 和
camera state 的已采样 ID，没有检查 prompt queue 的已采样 ID。

Web UI 在发送 prompt/action 后会记住 pending event ID。只有收到
`frame.event_id >= pending_event_id` 时，播放器才切掉切换前已经解码的旧帧。所以上例
中 prompt 已进入服务端 chunk，但播放器永远等不到 `144`，不会执行 prompt cutover。
这会把服务端“已切换”和用户“看不到切换”同时变成真。

修复：

- prompt 被采样时保存 prompt event ID；
- 返回该 chunk 已采样的 prompt/action/camera 事件中的最新 ID；
- 增加“旧 camera event 后切 prompt”的回归测试。

### 3.2 静态 action case 不能证明动态 action 正常

旧调查用固定 label 的独立请求验证了：

- idle 与 `w` 输出像素明显不同；
- idle 与 yaw-right `l` 输出像素明显不同；
- action mapping 与 checkpoint 的 `[w,a,s,d,i,j,k,l]` 一致。

这只能排除“checkpoint 完全忽略 action”和“键位表整体错误”，没有覆盖：

- WebSocket event 是否在目标 chunk 前到达；
- state-mode held key 是否被扩成当前 chunk 的 4 个 latent labels；
- whole-DiT compile 是否把 action tensor 或 cache 状态错误固化；
- 播放器是否显示了 action 后的 chunk。

本次新增的 adapter 回归固定验证：

```text
chunk 0: noop → [0, 0, 0, 0]
event#21: hold l
chunk 1: yaw-right → [1, 1, 1, 1]
returned event_id: 21
```

真实 GPU 输出仍按第 5 节的同 session 实验判定，不能用这条单元测试替代。最终
GPU 结果确认 action event 确实改变模型输出；没有发现 action 被 SGLang 丢弃或
whole-DiT compile 把 action 固化的问题。

因此需要把两个现象分开：

- **prompt switch 确有服务端/UI correlation bug**：模型已切 prompt，但旧实现
  没有把 prompt event ID 带回输出，播放器无法按事件切掉旧缓冲帧；
- **action switch 的服务端模型路径正常**：8 个单键都在下一个 chunk 改变输出。
  若部署后的 UI 仍感觉 WASD 不明显，应继续检查具体场景的 controllability 和
  event-to-render 延迟，不能再归因于 action 没送进 DiT。

### 3.3 当前 live Pod 还有一个独立的编译缓存故障

一个 benchmark Job 与 serving Pod 同时挂载了相同的 RWO EBS PVC。SELinux MCS
重新标记后，运行中的 serving Pod 失去 `/work` 访问权限。已有进程仍能回答 health，
但新请求在 TorchInductor 访问 `/work/torchinductor` 时失败：

```text
PermissionError: [Errno 13] Permission denied: '/work/torchinductor'
```

这不是 prompt/action 的模型语义根因，但它会让新的验证 session 直接失败，并掩盖
真实问题。部署 manifest 已把 `TORCHINDUCTOR_CACHE_DIR` 改成 pod-local
`/tmp/torchinductor`。模型 PVC 不应再被 serving 与 benchmark Pod 同时当作可写工作盘。

## 4. 代码级验证

在 B200 容器中执行：

```text
pytest -q \
  python/sglang/multimodal_gen/test/unit/realtime/test_minwm_realtime.py
```

结果：

```text
66 passed
```

其中 8 条参数化 case 覆盖 WASD/IJKL 全部单键，另两条专门覆盖动态 action 和
prompt event correlation。测试在 B200 serving 容器内执行，并显式移除部署环境中
会覆盖配置默认值的空 `MINWM_NATIVE_COMPONENTS`。

## 5. 同 session GPU 实验

固定条件：

| 字段 | 值 |
| --- | --- |
| checkpoint | 0724 `global_step_011000/generator/model.pt` |
| GPU | 1 × B200 |
| 分辨率 | 832×480 |
| seed | 4242 |
| DMD steps | 4 |
| chunk | 4 latent frames / 16 RGB frames |
| first frame / initial prompt | 三组完全相同 |

三组实验在 chunk 0 已开始执行、chunk 1 尚未采样前发送事件：

1. control：不发送事件；
2. action：发送 state-mode held `w`；
3. prompt：切换为差异很大的 snowy-night prompt。

验收：

- 事件生效前的 chunk hash 应与 control 一致；
- 第一个携带事件 ID / `minwm_prompt_updated` 的 chunk 开始，输出 hash 应与
  control 不同；
- prompt chunk 必须返回 prompt event ID；
- whole-DiT compile off/on 分别执行，排除 compile 固化输入。

### 5.1 whole-DiT compile off

WebP 是 UI 使用的预览传输格式（默认允许有损编码）；这里对完整的逐帧 WebP payload
做 SHA-256。
验收只依赖“相同输入是否得到相同字节”和“事件后是否改变”，不把 WebP hash 当成
latent/像素 bitwise parity 指标。

| chunk | control SHA-256 | held `w` SHA-256 | prompt switch SHA-256 | 事件 ID |
| ---: | --- | --- | --- | --- |
| 0 | `ead5d0ff...aeb0d` | `ead5d0ff...aeb0d` | `ead5d0ff...aeb0d` | 无 |
| 1 | `6ce08160...07e` | `319ed45a...30c5` | `81791f36...f2f1` | `101` / `202` |
| 2 | `7f821fab...fe8c` | `86153a90...6eca` | `7af1b402...ff9f` | `101` / `202` |
| 3 | `928148f7...33b8` | `3aa51f87...607b` | `359e8e06...6bf62` | `101` / `202` |
| 4 | `e7f9f7ad...3311` | `f8173ca2...4b4` | `2a4cf65d...4346` | `101` / `202` |

结果满足：

- 三路事件前的 chunk 0 完全相同；
- action 和 prompt 都从第一个携带对应 event ID 的 chunk 1 开始改变；
- 修复后的 prompt 输出正确回传 `event_id=202`。

另外对 UI state-mode 的 8 个单键分别做两 chunk 实验。所有 case 都满足
`chunk0 == neutral`、`chunk1 != neutral`，并在 chunk 1 回传对应 ID：

| 输入 | label | 动态输出 |
| --- | ---: | --- |
| `w/a/s/d` | `9/27/18/36` | 4/4 均改变 |
| `i/j/k/l` | `3/2/4/1` | 4/4 均改变 |

### 5.2 whole-DiT compile on

`--enable-torch-compile true` 使用
`max-autotune-no-cudagraphs`。首次请求用了约 129 秒编译，且首次编译请求不能作为
可复现基线；必须在编译完成后重新跑 control。热身后的结果：

| chunk | warm control SHA-256 | held `w` SHA-256 | prompt switch SHA-256 | 事件 ID |
| ---: | --- | --- | --- | --- |
| 0 | `3b6a65b7...31b60` | `3b6a65b7...31b60` | `3b6a65b7...31b60` | 无 |
| 1 | `b55455df...d6a79` | `de17e2ce...1b0b` | `2d392568...8cb8` | `411` / `412` |
| 2 | `90b9a78a...690e` | `6de8ed59...6a139` | `e50ce50d...2c0d4` | `411` / `412` |

whole-DiT compile 没有固化 prompt 或 action。prompt switch 第一次会触发额外图形/缓存
路径，因此测试耗时比普通稳态 chunk 长，但后续输出语义正确。

上述实验已经闭环到 WebSocket 返回帧，但没有把“浏览器实际显示出的第一帧”当成
GPU 验收信号。prompt 的播放器 cutover 缺陷已由 event ID 机制直接定位并修复；
action 的服务端动态路径则没有复现“不生效”。如果重新部署后浏览器里 action 仍无
可见响应，需要保留 `keydown → event send → sampled event ID → first rendered frame`
四段时间和对应录屏继续定位，不能把服务端 hash 变化直接等同于交互手感已经通过。

### 5.3 18/9 KV、5 秒 raw RGB 正式 parity

后续正式补跑了 1 个静态 control 加 4 个同 session prompt-switch case。每个 case
均为 832×480、24 FPS、129 帧，MinWM V3 与 SGLang 都使用 window=18/sink=9。
prompt event 从 chunk 1（pixel frame 17 / latent frame 5）生效。

结果为 5/5 bitwise exact，且 4/4 case 同时满足：

- frame header 与 chunk stats 的 event ID 首次命中 chunk 1；
- 切换前与静态 control bitwise exact；
- 切换后与静态 control 产生像素差异。

完整设计、性能和证据路径见
[`PROMPT_SWITCH_PARITY_5S.zh-CN.md`](PROMPT_SWITCH_PARITY_5S.zh-CN.md)。

## 6. 部署影响

需要重新部署。原因有两个：

1. adapter 和 Web UI 运行在服务端进程内；事件 ID 修复不会热加载；
2. 当前 Pod 的 `/work/torchinductor` 已不可写，即使代码不变也需要滚动重建，
   并使用 pod-local compile cache。

调查时线上 Deployment 仍固定在 `8de158c6e9d4fae849cc15cc77b1185db1d20511`，
不包含本文的 prompt event correlation 修复；`/health` 成功不能证明它已更新。

部署后必须用同一 WebSocket session 重跑第 5 节，不应只以 `/health`、静态 action
case 或独立 session 的视频差异作为通过标准。
