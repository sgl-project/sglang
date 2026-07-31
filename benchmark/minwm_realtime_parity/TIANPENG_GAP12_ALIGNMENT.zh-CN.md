# 天鹏 detailmix gap12 模型：SGLang 移植与对齐记录

更新时间：2026-07-31

状态：实现与 CPU 语义门已完成；L40S 全量 1089 帧诊断已完成，但尚未达到
数值对齐；B200/B300 同机原生 baseline 门正在等待 Local Zone Spot 容量。

## 1. 本次对齐对象

算法侧对齐页面：

<https://leap-world-us-east-2.s3.us-east-2.amazonaws.com/world-model/sft/prompt_compare/detailmix_director_gap12_20260729_094145/inference-alignment/index.html>

算法侧代码：

- 用户指定的 MinWM merge commit：`4220c8a`
- 页面产物记录的实验分支 commit：
  `386f8c242704db4c848bae14097b89cfebc5ae4e`
- `4220c8a` 是 PR #171 的 merge commit，包含上述实验分支的推理语义。

模型：

```text
s3://leap-world-us-west-2/world-model/minwm/checkpoints/run-archive/rolling/Wan21/Action2V/bidirectional/wan22-5B-varlen-multishot-texiao-0725detailed-mix-dccb050-dmd-0724-5eba381389f-merge/global_step_010000/generator/model.pt
```

不可变校验：

| 字段 | 值 |
| --- | --- |
| VersionId | `sL6CTylRv4QWY98mVTkuLoe5REbKHlvd` |
| bytes | `20,014,120,667` |
| SHA-256 | `18a48a2709d74b93ce26f0b808f381d191553853aae81dd72d2438430251d379` |

推理合同：

| 项目 | 值 |
| --- | --- |
| 分辨率 | 832×480 |
| FPS | 24 |
| 视频帧 | 1089（45.375 秒） |
| latent 帧 | 273 |
| block | `1 + 68 × 4`，共 69 块 |
| DMD steps | `[1000, 750, 500, 250]` |
| action type | `primitive_token_residual` |
| local attention | 32 latent 帧 |
| sink | 8 latent 帧 |
| RoPE | `block_relative` |
| 最大可见帧 gap | 12 |
| prompt 首帧延迟 pin | 开启 |
| prompt switch chunk | 13、42、50 |
| block seeds | 729001…729069 |

参考视频 SHA-256：
`0295dc25077c550a76ad8cd57a44e4189037443e693127e621bdbbab9d69865c`。

## 2. 架构直觉

普通 Wan 的 causal cache 可以直接保存“已经做过 RoPE 的 K”，因为绝对位置不再
变化。本模型不同：`block_relative` 会在 window 淘汰后，按**当前可见 KV 的时间
顺序**重新压缩位置；因此同一条历史 K 在不同 block 里可能使用不同的有效 RoPE
位置。

所以正确的数据流是：

```text
Q/K linear
  → RMSNorm
  → 缓存 raw K（尚未 RoPE）
  → 选择 sink / prompt pin / tail
  → 对当前可见的完整 K 重新计算 block-relative position
  → 对 Q 和完整可见 K 应用 RoPE
  → attention
```

prompt switch 也不是立刻把新 prompt 首帧永久 pin 住。它先把首帧记为候选；候选
仍处于连续 tail 时不额外占窗口，直到它将要被淘汰，才晋升为动态 pin。窗口的时间
顺序始终是：

```text
global sink → dynamic prompt/scene pin → continuous tail
```

## 3. 具体移植内容

### 3.1 position-aware raw-K cache

新增 `minwm_kv_cache.py`，保存：

- RoPE 前、RMSNorm 后的 K；
- absolute position 与写入时 effective RoPE position；
- 全局 token id；
- prompt 候选首帧、动态 pin 和 scene-cut pin；
- `absolute` / `block_relative` 策略及 `rope_max_frame_gap`；
- noisy denoise forward 对 active chunk 的原位覆盖语义；
- CFG lane 的 committed self-history 复制接口。

### 3.2 self-attention

MinWM cache 路径不再把已旋转 K 写入通用 cache。每次 forward 在完成 window
选择后，对 query 和完整可见 key 重新应用 source-shaped FP32 interleaved RoPE。
无 cache 的训练/双向路径保持原行为。

### 3.3 prompt 与 scene cut

- prompt/scene cut 都只清 text cross-attention cache，保留 latent self history；
- prompt switch 支持延迟首帧 pin；
- scene cut 支持 RoPE temporal offset 和新场景 sink pin；
- `block_relative` 与非零 scene-cut RoPE offset 的非法组合会 fail fast；
- realtime init 支持按 chunk 写死的 `minwm_prompt_schedule`，避免长视频对齐依赖
  浏览器/网络发送时机；
- live WebSocket 仍支持 `prompt`，并新增 `scene_cut` 事件。

### 3.4 action

- 目标 checkpoint 继续使用 `primitive_token_residual`；
- 同时移植 4220c8a 已支持的
  `primitive_rope_token_residual`，converter 会根据 checkpoint tensor 自动识别；
- T2V 第 0 个 latent 固定使用 noop action；
- 页面里的 1088 行 action 对应视频帧 1…1088，按每 4 行组成 latent
  1…272 的 action window，首 block 不消费 action queue；
- I2V 仍把 reference latent 作为一个 noop history frame，再消费第一个生成块动作。

### 3.5 director 逐 block seed

这次最容易误实现的地方是 seed。

director 第 N 次生成并不是只调用：

```python
torch.randn([B, current_F, C, H, W])
```

它会从该 block 的 seed 重新执行：

```python
all_noise = torch.randn([B, prefix_F + current_F, C, H, W])
current_noise = all_noise[:, prefix_F:]
```

历史前缀虽然会被 clean latent 覆盖，但它已经消耗 RNG。CUDA 随机数的 fill 顺序
属于 parity 合同。因此 SGLang 的逐块 seed 条件同时携带
`minwm_chunk_seed_prefix_frames`，先抽完整 BFCHW prefix，再取尾部。

### 3.6 bounded window

旧代码把“请求没有 runtime window override”一律视为 unbounded，并根据完整
request horizon 扩大 cache。这会静默覆盖 checkpoint 的 `local_attn_size=32`。

现在只有模型配置明确为 `local_attn_size=-1` 时才扩成完整 horizon；模型配置
为 32 时固定使用 32。converter 也要求 bounded
`sliding_window_num_frames == local_attn_size`，避免转换时写出两套互相冲突的值。

## 4. 与最初预期不同的地方

1. 页面记录的代码 SHA 不是用户给的 `4220c8a`。前者是实验分支 head，后者是
   包含它的 merge commit；实现以 merge commit 为代码基准，以页面 manifest
   为输入/输出基准。
2. 69 个 seed 不是“一个初始 seed 的派生展示”，而是 69 次独立、逐 block 的
   RNG 重放合同。
3. 页面 action 长度是 1088，不是 1089，也不是 `273×4=1092`。原因是 T2V
   第 0 个 latent 没有前置 action interval。
4. 只把 `window=32/sink=8` 写进 transformer config 仍不够；旧 stage 会把它
   扩成完整 273 帧，必须一起修复 cache policy。
5. 参考产物只提供编码后的 MP4，没有 baseline latent、每步 flow 或 cache dump。
   因而页面产物最多建立“解码视频数值对齐”；严格的 latent/cache bitwise 结论
   需要算法侧额外提供 tensor dump，或在同一任务里运行 baseline。

## 5. 已完成的语义门

已直接动态加载 MinWM `4220c8a` 的原始 `CausalKVCache`，使用目标合同跑：

- 69 个 block；
- `1/4/…/4` 变长 block；
- window 32；
- sink 8；
- block-relative gap 12；
- chunk 13、42、50 三次 prompt switch；
- prompt 首帧延迟 pin。

逐 block 比较以下 tensor，全部 exact equal：

- cache 可见 absolute positions；
- attention query RoPE positions；
- attention key RoPE positions；
- sink/pin/tail 淘汰结果。

此外单独验证：

- raw K active chunk 覆盖不追加；
- source `torch.polar(float64) → float32` RoPE 表与 SGLang N-D RoPE 表 bitwise
  equal；
- `primitive_rope_token_residual` 的 label 与重复 binary window 路径 equal；
- director prefix RNG 与 SGLang尾块抽样 equal。

## 6. 一键准备与对齐

对齐脚本会下载并验证页面 manifest、参考视频和 SHA-256，生成完整 realtime
request；正式运行后还会生成 PSNR/SSIM 报告和双路同步播放页。

只准备输入：

```bash
python3 benchmark/minwm_realtime_parity/tianpeng_alignment.py \
  --results benchmark/minwm_realtime_parity/results/tianpeng-gap12-4220c8a \
  --prepare-only
```

调用已启动的 SGLang：

```bash
python3 benchmark/minwm_realtime_parity/tianpeng_alignment.py \
  --results benchmark/minwm_realtime_parity/results/tianpeng-gap12-4220c8a \
  --ws-url ws://127.0.0.1:30000/v1/realtime_video/generate
```

产物：

```text
alignment_contract.json
request.json
baseline.mp4
sglang.npy
sglang.mp4
comparison.json
index.html
```

## 7. 模型转换

```bash
python3 python/sglang/multimodal_gen/tools/convert_minwm_checkpoint.py \
  --minwm-checkpoint /path/to/model.pt \
  --donor-diffusers-dir /path/to/pretrained \
  --output-dir /path/to/sglang-model \
  --link-donor \
  --source-uri 's3://.../global_step_010000/generator/model.pt' \
  --source-version-id sL6CTylRv4QWY98mVTkuLoe5REbKHlvd \
  --action-type auto \
  --local-attn-size 32 \
  --sink-size 8 \
  --sliding-window-num-frames 32 \
  --rope-position-mode block_relative \
  --rope-max-frame-gap 12 \
  --prompt-first-frame-pin-enabled
```

部署时不依赖 `~/workspace/minWM`；需要的推理语义、action conditioner 和 cache
都已在 SGLang 内。donor Diffusers 目录仍用于 text encoder、tokenizer 与 VAE。

## 8. GPU 实测结果

### 8.1 L40S 诊断

已在单张 L40S 上完成 69 chunk / 1089 帧：

| 项目 | 结果 |
| --- | --- |
| PSNR | 20.829439 dB |
| SSIM | 0.580779 |
| 端到端 wall time | 243.277 秒 |
| 原始视频帧吞吐 | 4.48 FPS |
| 稳态 chunk | 约 3.2 秒 / 16 视频帧 |
| DiT | 约 2.08 秒 / chunk |
| peak GPU memory | 约 33.3 GiB |

结果页：

<https://leap-world-us-east-2.s3.us-east-2.amazonaws.com/world-model/sft/prompt_compare/detailmix_director_gap12_20260729_094145/sglang-alignment/20260731-l40s-02/index.html>

这个结果与先前 B300 诊断几乎相同，说明差异不是 L40S/B300 硬件本身造成的。
该运行用完整 `pip install` 启动 SGLang，数值依赖从基线镜像的
Torch 2.12 / Transformers 4.56 / Diffusers 0.35 漂到了
Torch 2.11 / Transformers 5.12 / Diffusers 0.37，因此它只能定位“仍有差异”，
不能作为最终同依赖栈 parity 结论。

### 8.2 同机、同依赖栈强门

新增一个双容器 Job，固定使用与天鹏运行相同的不可变镜像；两个容器各申请一张
GPU，并在同一台 8×B200/B300 节点上串行执行：

1. MinWM `4220c8a` 原生 `DirectedSession` 重放 69 个 block；
2. SGLang 使用同一份 checkpoint、donor、Torch、Transformers、Diffusers 和
   FlashAttention 重放同一请求；
3. 对两路 `uint8` 原始 RGB 逐值比较，再生成“已发布 baseline / 原生 MinWM /
   SGLang”三路同步播放页。

节点资源划分为：

| 用途 | GPU |
| --- | ---: |
| Realtime Studio 临时实例 | 1 |
| 原生 MinWM baseline | 1 |
| SGLang parity | 1 |
| 保留给其他任务 | 5 |

目标是 `us-east-1-atl-2a` 的 Local Zone Spot
`p6-b200.48xlarge / p6-b300.48xlarge`。截至本次记录时间，Karpenter 正在因
`UnfulfillableCapacity` 重试；不能把 Pending 状态误报为已经完成对齐。

## 9. 交给克君部署前的检查

- 必须使用本次提交后的 SGLang commit，而不是复制工作区文件；
- 必须使用 checkpoint SHA-256
  `18a48a…d379`；
- transformer config 必须是 window 32、sink 8、gap 12、prompt pin true；
- action type 必须由 converter 检测为 `primitive_token_residual`；
- 产品实时测试不需要下发逐 block seed 或 prompt schedule；那两项是离线重放接口；
- live prompt/scene-cut、WASDIJKL 和 fractional action weight 继续走同一个 MinWM
  realtime adapter；
- 产品 profile 若打开 torch.compile 或非确定 attention，应明确它是性能模式，
  不承诺跨进程 bitwise。

## 10. 必须通过的测验

1. 为什么 block-relative 模式不能缓存 RoPE 后的 K？
2. prompt 首帧为何要等到离开 tail 时才晋升为 pin？
3. window 32、sink 8 时，可见 KV 的三段顺序是什么？
4. 为什么页面有 1089 个视频帧，却只有 1088 行 action？
5. 为什么第一个 T2V latent 不能消费 action queue 的前 4 行？
6. 为什么 seed 729020 不能只用于抽取当前 4 帧 noise？
7. `local_attn_size=32` 已写进 config 后，旧 stage 为什么仍可能实际使用 273
   帧 window？
8. prompt switch 时哪些 cache 要清，哪些 cache 必须保留？
9. `block_relative` 为什么拒绝非零 scene-cut RoPE offset？
10. 只有 baseline MP4、没有 latent dump 时，为什么不能声称 latent bitwise？
11. 产品实时部署时，哪些参数属于模型合同，哪些输入只用于 parity replay？
12. 如果新 checkpoint 改成 `primitive_rope_token_residual`，converter 和运行时各
    如何识别并加载它？
