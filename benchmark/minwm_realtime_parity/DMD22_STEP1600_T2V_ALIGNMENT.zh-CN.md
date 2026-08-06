# MinWM DMD22 step-1600 T2V 对齐记录

更新时间：2026-07-28

## 目标

把下面的 DMD student 接入 SGLang MinWM realtime WebSocket 推理，并对齐原生
minWM V3 的 T2V 推理：

```text
s3://leap-world-us-west-2/world-model/minwm/checkpoints/run-archive/rolling/Wan21/Action2V/dmd/wan22-5B-stage3-dmd-22-0724-5eba381389f/global_step_001600/ema_student/model.pt
```

对齐优先级是：

1. latent 和 uint8 raw frame bitwise 一致；
2. 若 bitwise 不成立，再报告 `max_abs`、RMSE、PSNR、cosine、SSIM；
3. 编码后的 MP4 只用于观看，不作为 bitwise 判据。

## 固定 provenance

| 项目 | 固定值 |
| --- | --- |
| checkpoint bytes | `10007171771` |
| checkpoint source VersionId | `9u_oXdzep3r4ckmlcp5xCw7JPsq61Lp5` |
| checkpoint source ETag | `f6535c71c5602dcd078e5b36434c0416-191` |
| baseline script SHA-256 | `50ea62839163e224c6ed833c59c9c338f510676947fdde2f3a7066637d532919` |
| baseline YAML SHA-256 | `0e9bf2b8c9ea20b7c484182c88ba5edb676b6cb9f15948f2c17dcf6f2d940efc` |
| minWM main | `c1c8d447a8f195aab4092061bc6461bdecaab841` |
| SGLang 首个 T2V 提交 | `0416cb75e799506754ef303d26dfd6af4959a41a` |

baseline 没有归档运行时 minWM Git SHA，所以 `c1c8d447...` 是按用户要求选取的
当前 `main`，不是对原始运行代码身份的证明。YAML 快照和输入文件有 SHA，可以精确固定。

## baseline 的实际语义

归档脚本看起来是“8 卡推理”，但模型没有使用 8 卡并行：

- `torchrun --nproc_per_node=8`
- `--sp_size 1`
- 每个 rank 通过 distributed sampler 处理不同 prompt
- 每个模型实例只占一张卡

脚本设置了环境变量 `SEED=42`，却没有把 `--seed "$SEED"` 传给
`wan_inference.py`。命令行参数默认值是 `0`，所以 baseline 实际 seed 是 `0`。

trajectory 会覆盖 YAML 的 `num_frames: 181`：

```text
latent_frames = trajectory_step_count(trajectory) + 1
pixel_frames = 1 + 4 * (latent_frames - 1)
```

第一帧 latent 的动作固定为 idle，trajectory 的第一个动作从第二帧 latent 开始生效。

## 两批输入的恢复

归档的 `prompts.txt` 和 `trajectories.txt` 按行一一对应，但它们只保留了后一次
8-video batch。旧 6-video batch 的 trajectory 文件已被覆盖。

后 8 条直接使用归档输入。旧 6 条按用户提供的映射恢复：

| prompt 行（1-based） | 旧 trajectory |
| ---: | --- |
| 1 | `w*181` |
| 2 | `W*60,k*30,w*60,d*31` |
| 3 | `a*80,d*81` |
| 5 | `W*60,a*60,w*61` |
| 6 | `w*181` |
| 8 | `w*181` |

这份映射已固化在
`cases_step1600_t2v_30s_832x480.json`。每条 case 还记录了对应的归档 MP4
文件名。

### 和预期不同的地方

两个 ronin case 的 trajectory 都只有 161 个动作 interval：

- `a*40,l*40,j*40,d*41`
- `a*80,d*81`

因此它们是 162 latent / 645 pixel frames / 26.875 秒，而不是 182 latent /
725 pixel frames / 30.208 秒。14 个归档 MP4 的实际帧数已逐个探测，只有这两条是
645 帧，其余 12 条是 725 帧。评测必须按 case 使用不同的总帧数，不能根据目录名
`30s` 写死。

## T2V 和原有 I2V 路径的关键区别

原来的 SGLang MinWM realtime 路径只支持 I2V：

- 必须有 `first_frame`
- 先编码并提交 1 个 reference latent
- 每个生成 chunk 固定 4 latent
- 初始随机数还要消费一个随后被 reference 覆盖的噪声位置

本 checkpoint 的 baseline 是纯 T2V：

- 没有 `first_frame`
- 第一块是 1 latent
- 中间块是 4 latent
- 最后一块可以是 1 latent remainder
- 所有随机数都对应生成 latent，不存在被丢弃的 reference-noise slot

标准 725-frame case 的块序列是：

```text
[1] + [4] * 45 + [1] = 182 latent frames，共 47 chunks
```

645-frame case 的块序列是：

```text
[1] + [4] * 40 + [1] = 162 latent frames，共 42 chunks
```

## SGLang API 契约

T2V 请求不传 `first_frame`，并显式传最终 pixel-frame 数：

```python
{
    "type": "init",
    "generation_mode": "t2v",
    "prompt": "...",
    "size": "832x480",
    "fps": 24,
    "seed": 0,
    "num_frames": 725,
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "max_chunks": 47,
    "condition_inputs": {
        "action_labels": [0, ...]  # 182 个 latent labels
    }
}
```

`generation_mode` 是新客户端的显式合同。为兼容旧重放脚本，不传时 adapter 仍按
`first_frame` 是否存在推断：有首帧是 I2V，无首帧是 T2V；显式模式与首帧矛盾时
会直接拒绝。`max_chunks` 可以省略，adapter 会从 `num_frames` 精确推导。

adapter 根据 VAE temporal factor=4 把 pixel frames 转成 latent frames，并验证：

```text
num_frames == 1 + N * 4
```

然后使用模型配置中的 `num_frame_first_block=1` 和
`num_frames_per_block=4` 推导 chunk 数与最后的 remainder。

## 具体实现

### 配置

- `MinWMVideoArchConfig.num_frame_first_block = 1`
- converter 把这个字段写入转换后的 `transformer/config.json`
- 新增 request-local condition `minwm_total_latent_frames`

### adapter

`minwm_realtime_adapter.py` 现在同时支持：

- I2V：有 `first_frame`，保持旧行为；
- T2V：无 `first_frame`，要求用 `num_frames` 固定完整 horizon。

T2V 的 chunk size 是 request-local 的，不能把第一块/尾块语义写死在全局 scheduler。

### noise

baseline 在进入 chunk loop 前一次性调用：

```python
torch.randn([B, F, C, H, W], dtype=torch.bfloat16, device="cuda")
```

SGLang 也按完整 horizon 一次性生成同样的 BFCHW tensor，再用 cursor 按
`1/4/.../1` 切片。不能逐 chunk 分别调用 `torch.randn`，因为 CUDA RNG fill order
会改变。

I2V 仍保留 reference-noise slot；T2V 不消费这个 slot。

### KV cache

baseline 配置解析后是：

- `generator_config.local_attn_size = -1`
- `generator_config.sink_size = 0`

所以本次 bitwise 路径使用 full history，不传 runtime sink/window override。对于
725-frame case，KV horizon 是精确的 182 latent；对于 645-frame case是 162。

### VAE

T2V 第一块直接解码生成 latent，不拼接 reference latent。realtime decoder 维持
Wan causal VAE cache，使逐 chunk 解码的时间因果状态连续。

Wan2.2 causal VAE 的 residual cache 不能把单独解码 1 latent 后留下的内部状态直接
接到常规 4-latent block；这样第二块会出现 temporal shape `2 != 4`。SGLang 因此：

1. 第一块先单独解码并发送首帧；
2. 第二块到来时重置 decoder cache，用“首 latent + 当前 4 latent”重新播种；
3. 丢弃重复解出的首帧，只发送新增的 16 帧；
4. 后续 4-latent block 继续使用已经播种好的 cache。

重播种只修复流式边界条件，不改变 baseline 的完整 latent 序列。

### action history 冷启动

原生 V3 的 T2V 第一块以 `action_hist=None` 开始，所以 action causal conv 只接收
当前第一块的 1 个 idle action。原有 SGLang I2V 路径会先放入 1 个 reference/no-op
历史帧；若把这个行为沿用到 T2V，第一块会错误地看到两个 idle frame，causal conv
左边界随即发生变化，之后所有 latent 都无法对齐。

当前实现明确区分：

- T2V：block 0 action history 长度是 0；
- I2V：block 0 action history 长度是 1，对应已经提交的 reference latent；
- 第一块完成后，两条路径都只保留模型配置要求的最近
  `action_history_frames=4` 个 action。

这项差异比“首帧动作都是 idle”更隐蔽：动作值相同并不代表卷积输入长度和边界状态
相同。

## 测试状态

- MinWM realtime unit tests：`71 passed, 1 deselected`（T2V/I2V action 冷启动修复后，
  定向回归 `2 passed, 72 deselected`）
- harness tests：`6 passed`
- Ruff 和 `git diff --check`：通过
- deselected 的测试是当前容器中 diffusers UniPC timestep rounding 漂移；本任务使用
  固定 few-step DMD schedule `[1000, 750, 500, 250]`，不经过 UniPC。
- B200 单 case latent probe `02_explorer_w30j60l60a31`：182 个 BF16 latent
  bitwise 一致，`max_abs=0`。这直接覆盖了 DiT、DMD scheduler、action 和 KV cache。
- 14-case raw-frame parity：使用两张 B200 并行执行 native baseline 和 SGLang API，
  Job `minwm-dmd22-step1600-t2v-full14-20260728-01` 已完成：
  - 14/14 通过 numerical parity；
  - 13/14 的 480×832 uint8 RGB raw frames 完全 bitwise 一致；
  - 只有 `01_explorer_w181` 不是 pixel bitwise：`max_abs=1`、
    RMSE `0.0068532548`、SSIM `0.9999997259`；
  - 该 case 只有 `0.0046967%` 的 uint8 channel values 变化，且变化幅度均为 1。

latent probe 证明 DiT、DMD scheduler、action 和 KV cache 已经 bitwise 对齐；唯一
一条 case 的 pixel 差异来自两条 decode 路径的执行边界：

- baseline 把 182 latent 一次性交给 causal VAE；
- realtime 为了低延迟按 `1/4/.../1` latent 流式解码，并在第二块重播种 cache。

两者输出仅有 uint8 round-to-nearest 边界上的 ±1 差异。本次把它归类为“latent
bitwise、全量 pixel numerical parity，其中 13/14 pixel bitwise”，而不是笼统宣称
14 条 pixel 全部 bitwise。

## B200 性能结果

同一批 14 个 case 共生成 9990 帧。baseline 与 SGLang 分别独占一张 B200，并行启动
但互不共享 GPU；计时包含每个 case 的端到端生成时间：

| 统计口径 | native minWM baseline | SGLang realtime | SGLang 提升 |
| --- | ---: | ---: | ---: |
| 全部 14 case，加权 output FPS | 19.264 | 23.711 | 23.1% |
| 排除首次 compile/cold-start，加权 output FPS | 19.648 | 24.450 | 24.4% |

SGLang warm case 的端到端 FPS 中位数为 24.482，范围为
24.044–24.633；TTFF 中位数为 0.507 秒，首帧之后的 steady-state FPS 中位数为
24.926。第一个 case 包含 compile/cold-start：TTFF 13.371 秒，端到端 17.101 FPS，
但首帧后的 steady-state 已达到 24.946 FPS。baseline warm case 的 FPS 中位数为
19.486，范围为 19.293–20.575。

这里的 output FPS 是“生成出的 pixel frames / 端到端秒数”，不是 WebUI 的 render
FPS。它说明 warm SGLang 已略高于 24 FPS 实时线；浏览器能否稳定消费还取决于
WebSocket payload、解码、排队和页面渲染。

## 结果阅读原则

结果目录是
`results/dmd22-step1600-t2v-full14-b200-numerical/`。平铺页面同时展示 14 个 case，
每个 case 提供 baseline/SGLang 双路和 “Play both” 同步播放。用支持 HTTP Range 的
本地服务器验证后，页面显示 14/14 PASS、14 张卡片 Ready；实际点击第一组同时播放，
8.85 秒位置的双路时间差为 0.0054 秒。数值结论必须读取 `report.json`：

- `bitwise_equal=true` 才代表解码后的 uint8 raw frame 完全相同；
- MP4 文件 SHA 不同不一定是模型不同，编码器 metadata/码率也会改变文件字节；
- 原始归档 MP4 会作为第三份 provenance 媒体保留，用于确认 baseline rerun 没有
  发生明显代码漂移。

## 必须通过的测验

1. 为什么脚本里的 `SEED=42` 不代表这批 baseline 的实际 seed 是 42？
2. 为什么 `torchrun --nproc_per_node=8` 在这里不等于 8 卡 sequence parallel？
3. `w*181` 为什么对应 182 个 latent frame 和 725 个 pixel frame？
4. T2V 的第一块为什么是 1 latent，而原有 I2V realtime 的每个生成块是 4 latent？
5. 为什么 T2V 不能像 I2V 一样先消费一个 reference-noise slot？
6. 为什么要一次性生成完整 BFCHW noise，而不能每个 chunk 单独 `torch.randn`？
7. 两个 ronin case 为什么只有 645 帧？如果把它们强行请求成 725 帧，会破坏哪两项
   输入契约？
8. `local_attn_size=-1, sink_size=0` 在本次对齐中意味着什么？
9. 为什么 MP4 SHA 不相等不能直接说明模型数值不一致？
10. 如果 raw frame 不是 bitwise，但 `max_abs=1`、RMSE 很小，你下一步会优先比较
    latent、VAE 输入还是 MP4 编码结果？为什么？
