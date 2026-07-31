# 天鹏 detailmix gap12 模型：SGLang 移植与对齐记录

更新时间：2026-07-31

状态：实现、CPU 语义门和 Phoenix Local Zone H200 同机强门均已完成。
原生 MinWM、SGLang 和天鹏发布视频的 1089 帧三路结果已经发布；当前结论是
**解码视频数值对齐，未达到 bitwise**。公网 Realtime Studio/API 继续运行，
供产品和算法同事实时体验。

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
6. 部署 manifest 曾把短 SHA `fc9d1e7621` 手工补成错误的完整 SHA，导致第一次
   checkout 失败。部署合同现在只接受直接复制的 `git rev-parse HEAD` 完整
   输出；禁止凭短 SHA 手工补全。
7. baseline 镜像没有与其 Torch ABI 匹配的 `sglang-kernel`。强行安装当前
   SGLang pin 的 wheel 会在 import 时出现 undefined symbol。parity 环境现在
   保留镜像依赖；SGLang RMSNorm 在预编译 kernel 不可用或 ABI 不兼容时回退到
   PyTorch native 实现。
8. SGLang 原实现把 cache inference 的 Q/K RMSNorm 也送入了 segment
   `torch.compile`，而 MinWM `4220c8a` 在 cache 路径中是 eager。BF16 reduction
   的 rounding boundary 因此不同。修复后 cache 路径保持 eager；无 cache 的
   训练/双向路径仍可编译。

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
GPU，并在同一台 8-GPU 节点上串行执行：

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

最初依次尝试了 `us-east-1-atl-2a` Local Zone 和 `us-east-1d` 的
`p6-b200.48xlarge / p6-b300.48xlarge` Spot，并短暂允许 On-Demand 兜底；
EC2 都返回 `UnfulfillableCapacity` / `InsufficientInstanceCapacity`。继续提高
出价无法解决物理容量不足，因此按“尽快可测”的优先级转向 H100。

随后在 `us-east-2b` 短暂获得一台 `p5.48xlarge` Spot：

| 字段 | 值 |
| --- | --- |
| 实例 | `i-059aa7485618a331b` |
| GPU | 8×NVIDIA H100 80GB |
| NodePool | `minwm-test-h100-spot` |
| 任务申请 | 3 GPU |
| 空闲 | 5 GPU |

服务和两路 parity Pod 一度被调度到同一节点，但实例启动数分钟后收到
Spot rebalance recommendation 和 interruption，Karpenter 正常驱逐 Pod，
实例随后被回收。它不能作为产品测试入口。

接着用单卡 `p5.4xlarge` 降低申请粒度，并让 Karpenter 在 Spot 失败后自动尝试
On-Demand；`us-east-2a/b/c` 与 `us-east-1a/b` 的可用 offering 都返回
`UnfulfillableCapacity` / `InsufficientInstanceCapacity`。这证明当时缺的是
物理容量，不是一次申请 8 卡过大。

最后复用 `us-west-2-phx-2a` Local Zone 中刚释放的 `p5e.48xlarge` Spot：

| 字段 | 值 |
| --- | --- |
| 实例 | `i-0128c7ba0be106051` |
| GPU | 8×NVIDIA H200 141 GB |
| NodePool | `minwm-test-phx2-p5e-spot` |
| Realtime Studio/API | 1 GPU |
| 原生 MinWM + SGLang parity | 2 GPU |
| 空闲 | 5 GPU |

CUDA smoke 已确认 Torch 能识别 H200；服务的 MinWM 单测为 84/84 通过，
checkpoint bytes 与 SHA-256 通过。节点有 EKS 报告的少量 PCIe replay 告警，
所以最终可用性仍以真实 WebSocket 视频请求而不是 Node Ready 为准。

Phoenix Local Zone 不能直接作为区域 NLB 的 target。把 Local Zone subnet
交给 Network Load Balancer 时，AWS 明确返回
`You cannot have any Local Zone subnets for load balancers of type 'network'`。
现在使用与 LingBot2 相同的两级拓扑：

```text
区域 NLB（us-west-2a/2b）
  -> 区域 c6a.large 上的 Nginx WebSocket gateway
  -> ClusterIP
  -> Phoenix Local Zone H200 上的 SGLang MinWM
```

两个 target group 都已变为 `healthy`，公网 `/health`、`/v1/models`、WebUI
和真实 WebSocket 请求均已通过。临时产品测试入口：

<http://k8s-default-minwmtia-1301c44132-e45be86dd5db2100.elb.us-west-2.amazonaws.com/>

完成 parity 后，公网实例已经滚动到 SGLang
`8a4e94992e2d3a4ea4b9a30cfc60873830314482`，Deployment annotation 也从
provisional 改成 `numerical-parity-h200-20260731`。WebUI 默认值已核验为
832×480、24 FPS、sink 8、window 32。

最终公网 WebSocket smoke 使用 WebUI 相同的 WebP preview 路径、T2V 两个
chunk，共返回 17 帧；经过 NLB 的端到端 wall time 为 5.280 秒。第一块
scheduler forward 为 0.829 秒，第二块 16 帧为 1.258 秒，即该短样例的稳态
模型块约 12.7 输出帧/秒；两块总 WebSocket payload 约 0.37 MB。对照 smoke
也确认 raw RGB 路径的服务器 compute 相同，但 16 帧原始 832×480 RGB 是
约 19.2 MB，跨公网链路传输明显变慢；因此 WebUI 默认使用 WebP，而离线 parity
才使用 raw。短样例吞吐不等价于完整 45 秒长序列，最终仍以 69-block 运行统计
为准。

首次 Phoenix 启动还暴露了三个与 GPU 无关的部署缺口：

- 服务的 50 GiB ephemeral-storage limit 小于 20 GB checkpoint 加 donor
  实际占用，Pod 被 kubelet 以明确的 `Pod ephemeral local storage usage
  exceeds ... 50Gi` 驱逐；Phoenix overlay 已改为 request 100 GiB、limit
  500 GiB。
- baseline helper 虽已延迟导入 `msgspec`，仍通过 `run_sglang_api` 间接顶层
  导入它；原生镜像因没有该 WebSocket 依赖而失败。现在
  `decode_frames` 也只在真正发起 WebSocket 请求时导入，并新增无 WebSocket
  依赖的 import 回归测试。
- 天鹏镜像固定 Transformers 4.56.0，而当前 SGLang pin 是 5.12.1。原先
  diffusion quantization registry 会在 import 时加载所有量化 backend，
  继而把未使用的 SRT/LLM 配置和 Transformers 5 专属配置带入无量化 MinWM。
  现在 registry 只在真正选中某个量化方法时加载对应 backend；未启用
  `SGLANG_CACHE_DIT_ENABLED` 时也不 import `cache_dit`，避免它要求较新的
  Diffusers；diffusion transformer loader 的类型声明也改回它实际使用的
  diffusion `QuantizationConfig`，不再误导入 SRT 量化 registry。parity lane
  因此无需 patch Transformers/Diffusers，也不安装
  会改变 Transformers 4.56 import 路径的 LLM-only `kernels` 包；其余非数值依赖由
  `install_parity_dependencies.py` 递归补齐，并在启动 server 前再次比较
  Torch、Transformers、Diffusers、FlashAttention 和 TorchVision 版本。

最终 H200 原生 baseline 已完整跑完 69 block：

| 项目 | 结果 |
| --- | ---: |
| generation | 851.410 秒 |
| VAE decode | 27.237 秒 |
| 1089 帧端到端吞吐 | 1.239 FPS |
| 第 1 block | 16.229 秒 |
| 最后 1 block | 16.543 秒 |

原生耗时随历史逐块增长，说明当前 `4220c8a` baseline 的 window attention
执行仍有需要单独剖析的性能问题；它不影响本轮“同输入、同依赖栈”的数值比较。

### 8.3 H200 同机最终数值与性能结果

最终三路同步页：

<https://leap-world-us-east-2.s3.us-east-2.amazonaws.com/world-model/sft/prompt_compare/detailmix_director_gap12_20260729_094145/sglang-alignment/20260731-h200-same-stack-qkfix/index.html>

三路都使用 832×480、24 FPS、1089 帧；原生 MinWM 与 SGLang 位于同一台 H200
主机、使用同一 checkpoint、donor 和数值依赖。SGLang 关闭 whole-DiT
`torch.compile`，保留与 source config 相同的 segment compile；两路 packed
attention 都实际选择 FA2。

| 对比 | PSNR | SSIM |
| --- | ---: | ---: |
| 天鹏发布 MP4 vs 原生 MinWM 重跑 | 21.637946 dB | 0.607780 |
| 天鹏发布 MP4 vs SGLang | 21.406803 dB | 0.591320 |
| 原生 MinWM 重跑 vs SGLang | 20.799225 dB | 0.576889 |

原生重跑本身没有复现天鹏发布 MP4 的 bitwise 结果；两者即使 checkpoint、commit、
配置、action 和 seed 合同相同，编码视频 PSNR 也只有 21.638 dB。SGLang 相对发布
视频为 21.407 dB，比原生重跑低 0.231 dB。因此合理结论是：SGLang 已接近本次
可观测的 baseline 重放精度，但在算法侧没有提供 latent/cache dump 的前提下，
不能宣称 latent 或视频 bitwise parity。

原生 MinWM 和 SGLang 的未编码 `uint8 RGB` 直接比较结果是：

| 项目 | 结果 |
| --- | ---: |
| bitwise equal | false |
| exact frames | 0 / 1089 |
| raw RGB PSNR | 17.509618 dB |
| mean absolute difference | 24.5213 |
| max absolute difference | 255 |

逐张量定位确认 checkpoint 权重、首个 latent input、prompt embedding、timestep、
action、patch embedding、首层 self-attention Q/K/V linear 都能 exact equal。
修复 cache Q/K RMSNorm 的 eager/compile 差异后，首层 self-attention 的
Q/K norm、RoPE 和 attention output 也可 exact equal；剩余误差从后续
compiled AdaLN/cross-attention 的 BF16 rounding 开始累积。调试 hook 本身会改变
Inductor 编译/自动调优路径，因此只用来定位首个算子差异，最终结论始终取无 hook
的完整 69-block 运行。

SGLang 最终端到端 wall time 为 123.049 秒，1089 帧吞吐约 8.85 FPS；稳态
16 帧 chunk 多数为 1.75～1.85 秒。相同 H200 上原生 MinWM 是 878.647 秒
（generation + decode），因此本次 SGLang 实现端到端约快 7.14×。这不是
实时 24 FPS，但已经明确快于当前 `4220c8a` 原生 baseline；进一步达到 24 FPS
需要 SP/compile/kernel 级性能工作，不能靠降低 parity 约束来声称已经实现。

视频、指标和复现 evidence 落盘并校验后，双容器 parity Job 已删除，释放了
它占用的 2 张 H200。当前节点只保留 Realtime Studio/API 的 1 张 GPU，其余
7 张可供同事复用。

部署 overlay：

```bash
kubectl kustomize --load-restrictor LoadRestrictionsNone \
  benchmark/minwm_realtime_parity/k8s/tianpeng_gap12_h200_phx2 \
  | kubectl --context codex-minwm-test-phx2 apply -f -
```

同机 parity overlay：

```bash
kubectl kustomize --load-restrictor LoadRestrictionsNone \
  benchmark/minwm_realtime_parity/k8s/tianpeng_gap12_parity_h200_phx2 \
  | kubectl --context codex-minwm-test-phx2 apply -f -
```

失败的 us-east-1/us-east-2 Deployment、NLB 和本次临时单卡 H100 NodePool 已删除；
没有遗留第二台 GPU 主机。us-east-2 的旧 parity 工作盘暂时保留到本轮结果落盘，
避免在输出确认前做不可恢复清理。

## 8.1 B200 性能优先部署复核（2026-07-31）

在 `us-east-2b` 的单张 B200 上，使用同一个 gap12 checkpoint、832×480、
24 FPS、4 DMD steps、window 32、sink 8 做了三种执行档复核。结果表明旧模型上
有效的 `dense + whole-DiT compile` 不能直接套到本次 T2V checkpoint：

| 执行档 | 完整 chunk scheduler FPS | window=32 饱和 FPS | 结论 |
| --- | ---: | ---: | --- |
| dense + whole-DiT compile | 1.87 | 1.22 | KV 变长后严重退化，禁用 |
| packed FA4 + whole-DiT compile，FA4 graph break | 未跑完 | 约 1.8～2.9 | 正确但仍为负优化，禁用 |
| packed FA4 非确定性 + segment compile | 13.21 | 11.50 | 当前最终性能档 |

whole-DiT compile 直接包住 packed FA4 时，PyTorch 2.11 Inductor 会在 varlen
`cumsum` 元数据上报 `FakeTensor * Node`。SGLang 因此把融合 FA4 边界显式留在
eager，避免请求崩溃；但 69 个 block 上的 graph break 会让整图 compile 比
eager/segment compile 更慢，所以产品部署仍关闭 whole-DiT compile。

最终部署开关如下：

```text
performance_mode                  speed
MINWM_ATTENTION_IMPL              packed
MINWM_PACKED_ATTENTION_DETERMINISTIC false
MINWM_NATIVE_COMPONENTS           <empty; use SGLang components>
MINWM_SEGMENT_COMPILE             true
enable_torch_compile              false
attention backend                 fa（B200 实际选择 FA4）
SP / CFG parallel                 1 / false
```

公网 WebSocket 的同一 5 秒 case 复核为：TTFF 2.43 秒，wall time 12.55 秒，
完整 chunk scheduler 13.21 FPS。另一次边接收边做 WebP 解码和 MP4 写盘的客户端
测量为 121 帧 / 18.52 秒；它包含同步解码/写盘背压，不代表纯传输吞吐。浏览器
传输仍使用 WebP quality 55、560px preview、3 帧/消息、pacing=false。

这次不能把 13.7 FPS 与 7 月 27 日的约 38 FPS 直接归因于加速开关变化：旧结果是
另一 checkpoint/I2V/window 20 合同，本次是 gap12 T2V/window 32；本次在固定新
checkpoint 和请求下的可比结论是，最终档显著快于 dense compile，但单卡仍未达到
24 FPS。上述最终数字取远端运行文件 SHA 与提交 `9ef0696d1c` 对齐后的重跑。

### 8.2 7 月 27 日与 7 月 31 日 B200 吞吐差异的同机归因

2026-07-31 在公网实例所在的同一台 `p6-b200.48xlarge` 上，用空闲 B200 重新运行
了 7 月 27 日 checkpoint、SGLang `8de158c6e9` 和同一容器镜像。两次都是单卡，
PyTorch 都是 `2.11.0+cu130`。因此这里没有把 B300 理论算力、Spot 日期或不同
GPU 型号混入比较；7 月 27 日和本节复现实际都是 B200。

固定 832×480、4 DMD steps、raw transport、10 个 warmup chunks 和 20 个 measured
chunks 后，结果如下：

| checkpoint / runtime | KV 合同 | whole-DiT compile | scheduler ms/chunk | scheduler FPS |
| --- | --- | --- | ---: | ---: |
| 7/27 存档：旧权重 + 旧 runtime | 原配置，full history | 开 | 422.10 | 37.906 |
| 7/31 同机复现：旧权重 + 旧 runtime | 原配置，full history | 开 | 448.20 | 35.698 |
| 新权重 + 旧 absolute-RoPE runtime（仅性能消融） | window 32 | 开 | 416.25 | 38.438 |
| 新权重 + 当前 parity-capable runtime | window 20 | 关；segment compile | 1233.50 | 12.971 |
| 新权重 + 当前 parity-capable runtime | window 32 | 关；segment compile | 1415.80 | 11.301 |

“新权重 + 旧 runtime”会删除 `block_relative/gap12/prompt-first-frame-pin` 配置，缓存
RoPE 后的 K，因此**不能用于视频数值对齐或产品部署**。它的用途只是性能归因：在
相同新权重、相同 B200、相同 I2V case 和 window 32 下，旧执行路径仍能达到
38.44 FPS，说明 5B 权重本身没有让矩阵计算慢三倍。当前正确语义路径的 11.30 FPS
主要是实现问题。

window 32 改为 20 只把当前路径从 1415.8 ms 降到 1233.5 ms，改善 182.3 ms / 12.9%。
以旧 runtime 同权重的 416.25 ms 为参照，总退化是 999.55 ms；window 变大只能解释
其中约 18.2%，不是主因。raw transport 在两条路径都只有约 7～8 ms/chunk，也不是
WebUI、WebSocket 或浏览器造成的 scheduler 差异。

同 checkpoint、同 window 32 的 PyTorch trace 进一步显示：

| 每个完整 chunk | 旧 absolute-RoPE runtime | 当前 gap12 runtime | 倍数 |
| --- | ---: | ---: | ---: |
| CUDA kernels | 5,730 | 32,901 | 5.74× |
| `cudaLaunchKernel` | 1,599 | 30,173 | 18.87× |
| `cudaStreamSynchronize` | 40 | 2,468 | 61.70× |
| D2H memcpy events | 17 | 1,545 | 90.88× |
| GPU kernel 累计时间 | 267.79 ms | 397.70 ms | 1.49× |

trace 开启后 wall time 会被导出和栈采集严重放大，因此表中不使用 profiler wall time；
这里只使用 kernel 时间戳、调用数和同步事件，且不把 Torch profiler 的静态 occupancy
当成 GPU 利用率。当前镜像/主机没有 `nsys`，所以本节没有声称 SM/Tensor Core
hardware counter 数字。

根因是 `4220c8a` baseline 要求的新 cache 语义改变了可编译边界：

1. 旧路径把 RMSNorm/RoPE 后的 K 写入 cache，历史 K 可直接复用；当前路径为支持
   `block_relative`、gap clamp、sink/tail/dynamic pin，必须保存 raw K，并在可见
   window 改变后重新生成整段 key RoPE。
2. 当前每个 chunk 有 5 次 DiT forward，每次 30 层；每层各自做 window 选择、
   position metadata、raw-K gather/copy 和 query/key RoPE。相同的层无关 metadata
   被重复 150 次，并触发大量 `.item()`、D2H 和 stream synchronization。
3. 每层还有 self/cross 两次 packed FA4，共 300 次 attention 调用。PyTorch 2.11
   Inductor 不能 lowering 运行时 `cumsum` 构造的 varlen metadata，FA4 边界必须
   graph break；segment compile 最终只编译许多小 norm/AdaLN 区域，不能像旧
   whole-DiT graph 那样持续喂满 GPU。
4. 强行打开 whole-DiT compile 并没有恢复旧图：动态 KV 长度和有状态 cache 使其
   成为负优化，window 饱和时 dense 路径仅 1.22 FPS。

所以“同为 B200，FPS 差一半”更准确的描述是：在稳态同口径下，当前正确语义路径
是旧 compiled 路径的约 `0.30×`（11.30 vs 38.44 FPS），其中 window 只占小头，
主要损失来自 raw-K/block-relative cache 的逐层重复工作、CPU/GPU 同步和 compile
失效。达到 24 FPS 需要把 16 帧 chunk 压到 666.7 ms；trace 中当前 GPU kernel
累计时间约 398 ms，说明单张 B200 的算力预算原则上足够，优先级应是：把层无关的
window/position metadata 与 RoPE 表移到 chunk 级只算一次、批量更新 30 层 cache、
用常量 cu-seqlens 的 FA4 custom op 消除 graph break，并为 window 饱和形状建立
可复用 compiled/CUDA-graph 路径。每项都必须重新跑 gap12/prompt-switch numerical
parity，不能直接上线旧 absolute-RoPE 快路径。

复现脚本和原始统计保存在：

```text
benchmark/minwm_realtime_parity/k8s/tianpeng_gap12_b200_standalone/
  run_old_0727_diagnosis.sh
benchmark/minwm_realtime_parity/results/tianpeng-gap12-b200-0727-diagnosis/
```

最终服务由 systemd 管理，使用单张 B200（物理 GPU 3），API/UI 分别监听
30060/18060，再由 Nginx 暴露 80 端口。其余 7 张 GPU 已释放。验证产物在：

```text
benchmark/minwm_realtime_parity/results/
  tianpeng-gap12-b200-public-5s-performance-final/
    request.json
    summary.json
    sglang-performance.mp4
```

### 8.3 正确语义路径的 hot-path 优化（2026-07-31）

本轮没有回退到旧 absolute-RoPE cache，也没有改变 checkpoint 的 window 32、
sink 8、gap 12 或 prompt pin 合同。优化只消除当前正确语义路径中的重复工作：

1. 每次 transformer forward 只由第 0 层构造一次 sink/pin/tail 选择和
   block-relative position metadata，30 层共享同一个不可变 plan；同一 chunk 的
   4 次 DMD recompute 也复用选区结果。
2. 单卡路径用 host integer cursor 作为权威位置，去掉逐层 `.item()` 带来的
   device-to-host 同步；position IDs 和 uniform frame indices 按固定 shape 缓存。
3. query/key RoPE 表每个 forward 只生成一次；完成第一次可见窗口 RoPE 后，
   recompute 只旋转并覆盖当前 chunk 的 K，历史 rotated K 直接复用。
4. packed FA4 的固定 batch/sequence `cu_seqlens` 按 shape 缓存，不再在每个
   self/cross attention 边界运行 `full + cumsum + pad`。
5. append 时直接分别 gather 旧 cache 与新 K/V，再拼成可见窗口，避免先构造完整
   history K/V 后再做第二次 gather。

三个容易单独消融的开关默认打开：

```text
MINWM_CACHE_ROTATED_K=true
MINWM_PRECOMPUTE_CACHE_ROPE=true
MINWM_CACHE_PACKED_METADATA=true
```

whole-DiT `torch.compile` 仍保持关闭。这里与最初预期不同：它不是“打开即可加速”的
开关。当前 PyTorch 2.11 + FA4 对有状态、变长 raw-K cache 的整图会产生 graph break，
实测仍远慢于 eager/segment compile；本轮收益来自缩小 eager hot path，而不是强行
扩大编译图。

同一台 B200、同一 checkpoint、832×480、4 steps、window 32、sink 8 的结果：

| 执行路径 | 稳态 scheduler FPS | 相对原正确语义路径 |
| --- | ---: | ---: |
| 优化前 parity-capable runtime | 11.301 | 1.00× |
| 共享 plan/host cursor，但关闭上述三个复用开关 | 14.935 | 1.32× |
| 完整性能优化，10 warmup + 20 measured chunks | 30.289 | 2.68× |
| 完整性能优化，公网 Nginx 5 秒 case | 30.227 | 2.67× |

公网 5 秒 case 共返回 65 帧，后三个完整 16-frame chunk 的
`scheduler_forward_ms` 为 528/531/529 ms，已经低于 24 FPS 所需的
666.7 ms/chunk。服务冷启动后的首次请求 TTFF 约 19.89 秒，主要包含首次
kernel/segment compile；预热后的下一次 5 秒请求 TTFF 为 0.87 秒。TTFF 不应与
稳态生成 FPS 混为一个指标。

性能档不承诺 bitwise。相对优化前、可重复 bitwise 的 SGLang 参考，固定 prompt、
seed、首帧和 action 的 raw uint8 RGB 比较为：首帧 exact，`exact_frames=1/65`、
`mean_abs=1.477/255`、`max_abs=190`、`changed_fraction=0.483`。同一稳定服务进程
连续重跑则是 65/65 帧 bitwise exact；相对参考的 raw RGB PSNR 为 34.736 dB。
跨执行档误差仍比“值完全一致”
宽松得多，因此上线口径必须写成性能优先数值回归，不能写成 parity 档。重大决策是
保留优化前提交作为严格 parity 回滚点；三个开关只用于局部消融，关闭它们仍会经过
共享 cache plan，因此不能冒充 bitwise 档。后续若要恢复严格 parity，需要按
latent 与逐 block dump 定位 split gather/缓存复用改变 FA4 数值路径的首个差异，
而不是只比较最终 MP4。

部署前执行了 140 个 realtime 单测，另有 Ruff、format、`py_compile` 和
`git diff --check`。公网入口经 Nginx WebSocket 重新跑过完整 5 秒 case，而不是只
检查 `/health`。

当前公网 `http://18.221.245.74/` 由 systemd 的
`minwm-tianpeng-perf.service` 管理，物理 GPU 4、API 30120、WebUI 18070，再由
Nginx 统一暴露 80 端口。切换并复核后，旧 GPU 1/2/3 候选与服务均已停止，当前只
占用一张 B200。

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
- 本次 gap12 T2V 性能档只能开启非确定 packed FA4 和 segment compile；不要按旧
  checkpoint 经验开启 whole-DiT compile，除非重新证明饱和 window 吞吐更高。

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
13. 为什么 cache inference 的 Q/K RMSNorm 必须保持 eager，而训练/双向路径仍
    可以 segment compile？
14. 本次为什么把“21.407 dB 接近 21.638 dB”表述为数值对齐证据，而不是
    bitwise parity 证明？
