# MinWM WASD / 镜头控制问题排查记录

更新时间：2026-07-27

> 2026-07-27 纠正：本文最初用“固定 action 的独立 session 会改变视频”
> 推断 WebSocket 同 session 的动态控制链路也正常，这个推断不成立。静态 case
> 只能证明 checkpoint 和 action conditioner 有响应，不能证明事件在正在运行的
> session 中被正确采样、送入目标 chunk，并被 Web UI 切换显示。动态 prompt/action
> 问题的逐层复现和修复见
> [`REALTIME_CONTROL_SWITCH_INVESTIGATION.zh-CN.md`](./REALTIME_CONTROL_SWITCH_INVESTIGATION.zh-CN.md)。

## 1. 结论

固定 action 的独立 session 没有证据支持“WASD 或 IJKL/方向键映射反了”。同
prompt、首帧、seed、分辨率下的最新 0724 checkpoint 实测显示：

- `i/k` 的 pitch 方向相反且符合约定；
- `j/l` 的 yaw 方向相反且符合约定；
- `w/s/a/d` 没有发现互换，但平移控制明显比旋转弱且不对称，尤其 `a`、`w/s`
  仅靠短片不容易稳定辨认；
- 把幅度统一改成 `0.8` 不是修复：它会明显削弱 yaw，`i=0.8` 在本 case 中几乎
  退化成 idle。

这些结果不解释 Web UI 中途切换不生效的问题。后者必须按同一 WebSocket session
逐 chunk 验证；不能归因成下列体验差异：

1. UI 展示的是 LingBot 的确定性几何步长，例如 `0.05/frame`、`4/6 deg/frame`；
   MinWM 的 primitive action 是 learned checkpoint-relative condition，没有这些
   物理单位。
2. UI 状态事件按 140 ms 合并，MinWM adapter 再按 4 个 latent frame 的 chunk
   采样。短按会折叠成至少一个 latent action；长按最早也只能影响下一个尚未开始
   去噪的 chunk，不会修改正在生成的 chunk。
3. 模型对平移 primitive 的可控性弱于 yaw/pitch。这个现象在绕过 UI、直接以
   固定 label 启动独立 session 时仍存在，但它不能证明动态事件链路正常。

本次没有反转 action mapping。动态链路的根因、事件 ID 缺陷和修复验证独立记录在
上面的调查文档中。

后续同-session GPU 补测已经确认：WASD/IJKL 8 个 state-mode held key 都从首个
携带事件 ID 的 chunk 开始改变输出；whole-DiT compile off/on 都成立。因此本文
关于“平移响应弱”的结论是视觉 controllability 结论，不是“action 未进入模型”的
结论。

## 2. 核对的参考实现

远端最新 seedleap/minWM `main`：

```text
commit   0796bc201fae4c86f100620cb23402ae21c8f3b5
date     2026-07-27T05:05:07Z
entry    Wan21/wan_inference.py
config   Wan21/configs/eval/wan22_5b_varlen_dmd.yaml
pipeline UnifiedDiffusionInferencePipelineV3
model    UnifiedWanModel
```

MinWM 与 SGLang 的 primitive 顺序一致：

```text
[w, a, s, d, i, j, k, l]
```

单键 label 对照如下：

| 输入 | 含义 | label |
| --- | --- | ---: |
| noop | 静止 | 0 |
| `w` | forward | 9 |
| `s` | backward | 18 |
| `a` | strafe left | 27 |
| `d` | strafe right | 36 |
| `i` / ArrowUp | pitch + | 3 |
| `j` / ArrowLeft | yaw - | 2 |
| `k` / ArrowDown | pitch - | 4 |
| `l` / ArrowRight | yaw + | 1 |

MinWM 数据侧从 pose delta 生成 primitive 时也采用同一符号：

```text
local z > 0  -> w        local z < 0  -> s
local x < 0  -> a        local x > 0  -> d
pitch > 0    -> i        pitch < 0    -> k
yaw < 0      -> j        yaw > 0      -> l
```

因此 UI key → SGLang label → MinWM primitive bits 三层没有发现符号翻转。

## 3. Checkpoint 的 action 输入格式

0724 目录中只有：

```text
global_step_011000/generator/model.pt
global_step_011000/generator/model.pt.sha256
global_step_011000/merge_manifest.txt
```

没有 resolved training config。`model.pt` 中的
`primitive_token_residual` 权重同时能接收 label 和连续权重，单靠 tensor shape
无法反推出训练时 `action_output_format`。

当前 MinWM `wan22_5b_varlen_dmd.yaml` 继承的 processor 配置是
`primitive_float`；0724 run 名称也属于 varlen/multishot 配方。这是该 checkpoint
偏向连续窗口输入的有力线索，但不是可审计的训练配置证明。为了避免把推断写成事实，
本次同时测试了：

- Web UI 当前等价路径：`label_81`，单键幅度等价于 1.0；
- 连续路径：每个 RGB-frame window 对应 primitive 幅度 0.8。

## 4. 实验设计

固定合同：

| 字段 | 值 |
| --- | --- |
| checkpoint | 0724 `global_step_011000/generator/model.pt` |
| GPU | 1 × B200 |
| SGLang profile | speed + whole-DiT compile |
| prompt / first frame | 同一条空街 prompt，`p87.png` |
| seed | 42 |
| 分辨率 | 832×480 |
| fps | 24 |
| 每 case | 4 chunks，65 帧（1 reference + 64 generated） |
| KV | exact/default profile，无 request override |

case manifest：

- `cases_action_controls_832x480.json`：idle、8 个单键和 `w+l`，共 10 组；
- `cases_action_controls_080_832x480.json`：8 个单键，逐 RGB 帧幅度 0.8。

结果目录：

```text
benchmark/minwm_realtime_parity/results/action-control-0724-label81/
benchmark/minwm_realtime_parity/results/action-control-0724-primitive-float-080/
```

## 5. 结果

对第 9 帧以后相邻帧做特征点光流，再用 RANSAC 拟合全局
`AffinePartial2D`。`dx/dy` 表示画面内容的位移，不是相机自身位移；例如相机向右
yaw 时，画面应向左移动。

### 5.1 label_81 / 幅度 1.0

| case | median dx | median dy | scale | 判断 |
| --- | ---: | ---: | ---: | --- |
| idle | -0.071 | -0.174 | 1.000071 | 基准漂移 |
| `w` | -0.262 | -0.743 | 1.000467 | 有前向响应，较弱 |
| `s` | -0.312 | -0.744 | 1.000087 | 与 `w` 可分性弱 |
| `a` | -0.007 | -0.085 | 1.000002 | 接近 idle，响应弱 |
| `d` | -0.532 | -0.270 | 0.999930 | 右移响应可见 |
| `i` pitch up | +0.141 | **+1.001** | 0.999125 | 画面下移，方向正确 |
| `k` pitch down | -0.285 | **-1.882** | 1.000187 | 画面上移，方向正确 |
| `j` yaw left | **+1.589** | +0.108 | 1.000077 | 画面右移，方向正确 |
| `l` yaw right | **-2.921** | -0.227 | 1.000291 | 画面左移，方向正确 |
| `w+l` | **-6.241** | -0.815 | 1.001420 | 右转组合响应明显 |

全局 affine 不能可靠衡量带深度视差的 forward/back/strafe，所以平移结论还结合了
末帧人工检查。它足以证明四个 look 方向没有反转，也显示平移控制的响应更弱。

### 5.2 primitive_float / 幅度 0.8

| case | median dx | median dy | 对 1.0 的变化 |
| --- | ---: | ---: | --- |
| `i=0.8` | +0.007 | +0.002 | pitch-up 几乎消失 |
| `k=0.8` | -0.295 | -1.711 | 略减弱 |
| `j=0.8` | +0.946 | -0.044 | yaw-left 减弱 |
| `l=0.8` | -0.584 | -0.056 | yaw-right 大幅减弱 |

因此 UI 按键继续使用离散 label/1.0 更合理。连续幅度适合显式 API 调参或未来模拟
摇杆，不适合作为没有标定依据的全局默认修复。

## 6. Web UI 状态路径的时序

```text
keydown / keyup
  → 140 ms transition batch
  → camera_actions(mode=state)
  → ControlStateQueue
  → 每次生成开始前 sample 4 个 latent action
  → label_81
  → 下一 chunk 的 16 个 RGB frame
```

MinWM 训练数据的 label 路径也会把 4 个 RGB-frame interval 聚合成一个 latent
action；`action_effective_ratio=0` 时，只要窗口中出现有效动作，就保留第一个有效
primitive。因此“短按至少变成一个 latent action”与当前数据处理合同一致，不是
off-by-four。

真正的交互边界是 chunk：事件到达时如果当前 chunk 已开始去噪，只能影响下一
chunk。当前 B200 稳态约 0.42 秒/chunk；浏览器还有 decode/playback queue，不过
播放器会在收到带 event id 的新结果后尝试切掉旧队列。排查手感时应记录四个时间：

```text
keydown → event send → server sampled event id → first rendered affected frame
```

仅看键盘按下时间或服务端 38 FPS 都不能回答控制反馈延迟。

## 7. 本次代码与测试变化

- 扩充 WASD/IJKL 全部单键的 label 回归表；
- 增加 realtime state → 四个 latent label 的方向测试；
- 增加同场景 10 组 label case 和 8 组 0.8 连续幅度 case；
- API runner 新增 `--sink-size/--kv-cache-num-frames`，并移除对 GPU serving package
  的客户端导入，使 macOS CPU 环境也能运行诊断；
- MinWM 专用部署 UI 不再展示 LingBot 的确定性物理步长。

B200 容器内执行完整 `test_minwm_realtime.py`，结果为 64 passed。测试时显式移除
部署 speed profile 的空 `MINWM_NATIVE_COMPONENTS` 环境变量，避免它改变配置默认值。

## 8. 下一步

1. 保持当前键位和 label，不做方向翻转。
2. 在浏览器 telemetry 中补齐 event-to-render 四段时间，再判断是否需要缩短
   transition flush 或主动跳过更多旧帧。
3. 用至少三个不同场景、每个方向 5 秒以上的视频评估平移 controllability；
   当前单场景结果已定位到“模型响应弱”，还不足以给 checkpoint 下总体结论。
4. 若产品需要速度摇杆，单独标定 `primitive_float` 幅度曲线，不要把 0.8 当成
   通用默认。
