# MinWM Realtime API 变更阅读指南

这份报告面向代码评审者和后续维护者。它解释这次变更为什么存在、实现背后的直觉、
哪些地方为了数值一致性没有采用 SGLang 的常规快速路径，以及应该按什么顺序阅读代码。
更完整的运行命令、逐项决策日志和实验记录见
[`MinWM-Realtime.zh-CN.mdx`](../../docs_new/cookbook/diffusion/MinWM/MinWM-Realtime.zh-CN.mdx)。

## 1. 一句话结论

这次变更把 minWM `main` 分支的 Wan2.2 5B DMD 模型接入 SGLang realtime
WebSocket API，支持首帧、prompt、seed 和 `primitive_token_residual` action。
在同一运行时、相同输入和确定性 kernel 配置下，正式的十用例 B200 验证达到了
generated frame 逐字节相同；H200 上的三组 1248x704、129 帧长视频也逐字节相同。

当前实现是单卡、batch-one 的 causal session。它**不支持** Ulysses/Ring sequence
parallel；非一的 SP 配置会被显式拒绝，避免多卡重复计算被误报成有效 SP。

## 2. 背景和真正的兼容性目标

目标不是“让一个相似的 Wan 模型可以出视频”，而是复现下面这条完整推理合同：

- minWM 源码固定在 `main`；
- 权重固定为 0721 checkpoint：
  `wan22-5B-stage3-dmd-8-0721-6a531f0e067/global_step_003200/ema_student/model.pt`；
- action type 是 `primitive_token_residual`，不是 camera pose/Plücker ray；
- 输入合同是同 prompt、seed、首帧和逐帧 action；
- 首选 bitwise equality，只有无法达到时才允许使用事先声明的数值误差范围。

最新 checkpoint 的原生训练配置是无首帧 processor 的 causal T2V。这次产品/API
合同需要首帧，因此正式首帧测试明确使用 V3 compatibility path。它证明的是该权重在
指定 API 合同下的兼容性，不应被描述成 checkpoint 原生 eval。

## 3. 从请求到视频的数据流

```mermaid
flowchart LR
    A["WebSocket 请求<br/>prompt/seed/首帧/action"] --> B["MinWM Realtime Adapter"]
    B --> C["HF UMT5<br/>512 位置零填充"]
    B --> D["Diffusers Wan VAE<br/>首帧 latent"]
    B --> E["BFCHW 全 horizon RNG"]
    C --> F["4-step causal DMD DiT"]
    D --> F
    E --> F
    B --> G["primitive token residual<br/>4 帧 action history"]
    G --> F
    F --> H["因果 KV cache<br/>默认完整历史"]
    H --> F
    F --> I["Causal VAE decode"]
    I --> J["frame_batch / MP4"]
```

最重要的状态边界有三个：

1. session 保存 action history，两个 kernel-size-3 causal Conv1d 需要四帧历史；
2. session 保存每层 self-attention K/V，默认合同是 `local_attn_size=-1` 的完整历史；
3. bounded session 会一次性预抽取完整 horizon 的 BFCHW noise，再按 chunk 切 view，
   从而保持与 baseline 相同的随机数消费顺序。

## 4. 为什么不能直接复用 LingBot World 2.0

LingBot World 提供了可复用的 realtime session、WebSocket 控制流和 causal pipeline
骨架，但模型语义不同：

- LingBot 的首帧条件会形成 20-channel I2V condition；MinWM V3 使用一个干净的
  reference latent，生成 chunk 保持 48 channels；
- MinWM 使用 81 类离散 action 或八维连续 primitive 权重；
- MinWM 的 action residual 在 patch embedding 之后注入；
- 两者的 conditioning、cache 和数值边界不能只通过改 config 互换。

所以实现复用了服务框架和 causal Wan 基础设施，同时为 MinWM 增加专用 config、
adapter、action encoder、DiT block/head 和 pipeline stages。

## 5. Action 的直觉与精确定义

一个 action label 是两个九状态轴的笛卡尔积：

- translation primitive 顺序：`[w, a, s, d]`，加 idle 共九种组合状态；
- look primitive 顺序：`[i, j, k, l]`，加 idle 共九种组合状态；
- 总 label 空间是 `9 × 9 = 81`，合法范围 `[0, 80]`；
- label `0` 是 noop。

API 也接受 `[w,a,s,d,i,j,k,l]` 顺序的八维连续权重，范围 `[0,1]`。因此 `w=0.8`
必须作为 `action_weights` 中第一维的 `0.8` 发送；只发送 label 无法表达幅度。
每个 latent frame 对应四个 pixel-frame action，encoder 看到四帧历史和四帧当前窗口，
只把最后四个 encoded state 注入当前 patch token。

## 6. Bitwise 一致性为什么比“公式相同”更难

GPU 数值合同不仅包括值、shape 和 dtype，还包括随机数分配 shape、tensor stride、
materialization、算子边界和 attention backend。以下改动看似细小，却都曾产生可观测差异：

- 随机数先按 baseline 的 `[B,F,C,H,W]` 生成，再转成 SGLang layout；
- UMT5 保留 512 个位置，真实 token 后是零，而不是缩短 sequence；
- 首帧 VAE 保留 BF16→FP16→BF16 的隐藏 wire boundary；
- RMSNorm 必须先把 FP32 normalized value round 到 BF16，再乘 BF16 weight；
- RoPE 使用 baseline 的 FP32 interleaved 实/虚部公式；
- AdaLN、gate 和 residual 按 baseline 的 promotion/rounding 顺序执行；
- patch embed 使用原生 `Conv3d`，action 后用 `torch.cat` 形成相同 contiguous stride；
- self/cross attention 使用与设备匹配的 source-shaped packed-varlen backend：
  Blackwell 优先 FA4，Hopper 按 baseline 回退到 FA2；
- MinWM 禁用通用 autocast，只保留 baseline 本身使用的局部 compiled fused segments。

一个很有用的排错原则是：先证明某个 operator 的输入在 value、dtype、shape 和 stride
上完全相同，再讨论替换它的 kernel。否则下游误差抵消可能让错误实现暂时看起来更接近。

## 7. Realtime、KV 和 TTFF

每个模型 chunk 生成 16 个 RGB frame；第一个 chunk 还返回一张 reference frame。
`frame_batch` 是传输分包，不等同于模型 chunk，所以延迟统计必须等到该 chunk 的
`is_final` batch 才结束。

TTFF（Time To First Frame）是请求开始到收到第一批可显示帧的时间。第一次 shape
会包含局部 fused graph 的编译和 cache 初始化，因此应分别报告 cold TTFF、warm TTFF
和 steady-state chunk throughput。

MinWM `main` 的 KV 合同是 `local_attn_size=-1, sink_size=0`，也就是完整历史。
bounded 请求按完整 horizon 预分配；unbounded 请求允许 cache 增长。历史测试中的
45/128-frame window 只能作为显式性能实验，不能静默替换默认模型语义。

## 8. 性能结果应该怎样读

所有 FPS 都必须和 GPU、分辨率、chunk、KV 合同、编译 profile 绑定：

| 设备和 profile | 分辨率 | 结果 | 一致性含义 |
| --- | ---: | ---: | --- |
| B200 exact | 832x480 | 23.075 FPS | 正式 bitwise 路径 |
| B200 dense-native | 832x480 | 24.713 FPS | attention ablation |
| B200 optimized components | 832x480 | 25.541 FPS | 不再保持同一 native boundary |
| B200 whole-DiT compile | 832x480 | 32.222 FPS | 非 parity 性能上限 |
| H200 exact、完整历史 KV | 1248x704 | 10.393 client FPS | 720p 长视频正式 bitwise 路径 |

在相同 720p case 的 warm full-case 口径下，minWM baseline 约为 8.20 generated FPS，
SGLang API 约为 10.095 generated FPS，API 快约 23%。不要把 832x480 的 23 FPS
和 1248x704 的 10 FPS 直接比较：像素数、token 数、GPU 和 KV 历史都不同。

为了 bitwise 所付的正式 B200 税可拆为：

- source-shaped packed attention 相对 dense attention：约 `6.63%`；
- native VAE/T5 boundary：约 `3.24%`；
- exact profile 相对已测非 exact 优化组合：合计约 `9.65%`。

Whole-DiT compile 目前不能默认打开。1248x704 首图在 Torch 2.11 Inductor
post-grad 阶段触发 `FakeTensor * Node`，没有产生可做 parity 比较的输出。

## 9. Sequence Parallel 审计与 24 FPS 上限

当前 MinWM 路径不支持 Ulysses SP，原因不是缺一个启动参数，而是下面四个合同都尚未
实现分片通信：

1. MinWM 的 source-shaped packed-varlen attention 没有 Ulysses all-to-all；
2. causal K/V cache 没有按 sequence/head rank 定义所有权；
3. action 的四帧历史和 chunk 边界没有跨 rank 交换；
4. absolute RoPE position 和 output gather 没有经过多卡 bitwise 验证。

因此 PR 会拒绝所有非一的 `sp_degree`、`ulysses_degree` 或 `ring_degree`。生产扩吞吐时，
目前正确的方法是每卡一个独立 replica，让负载均衡器按 session 路由。

即使将来接通 Ulysses，单靠 SP 也无法让当前 720p 串行 pipeline 到 24 FPS。H200
24-chunk profile 的中位时间约为：

- DiT：637.35 ms；
- VAE decode：474.10 ms；
- RGB/output：314.44 ms；
- 其他串行开销：187.80 ms。

假设 Ulysses 对 DiT **理想线性加速且通信为零**，乐观上限为：

| Ulysses degree | 估算 chunk 时间 | 乐观 FPS |
| ---: | ---: | ---: |
| 1 | 1613.69 ms | 9.92 |
| 2 | 1294.99 ms | 12.36 |
| 4 | 1135.68 ms | 14.09 |
| 8 | 1056.01 ms | 15.15 |
| 无限 | 976.34 ms | 16.39 |

24 FPS 要求 16-frame chunk 不超过 666.67 ms，但现有非 DiT 串行部分已经约
976.34 ms。因此答案是：**没有任何 Ulysses SP degree 能让当前结构达到 24 FPS**；
实际 all-to-all 开销只会让表中数字更低。要达到目标，至少还要并行化/流水化 VAE，
减少 RGB materialization/传输，重叠 decode 与下一 chunk 的 DiT，并在此基础上再评估
SP2/SP4。MinWM 有 24 个 attention heads；未来 Ulysses degree 还必须满足 head
可分以及 token padding 条件，但这只是必要条件，不是性能保证。

## 10. 代码阅读顺序

建议按下面顺序阅读，每一步都回答一个明确问题：

1. `configs/pipeline_configs/minwm.py`：模型合同是什么，哪些 component 必须 native？
2. `configs/sample/minwm.py`：API 默认分辨率、步数和 realtime 参数是什么？
3. `runtime/entrypoints/openai/realtime/adapters/minwm_realtime_adapter.py`：
   WebSocket action 怎样变成 chunk condition？
4. `runtime/models/dits/minwm_action.py`：81 类 label 和连续权重怎样形成 token residual？
5. `runtime/pipelines_core/stages/model_specific_stages/minwm/`：
   RNG、cache、DMD 和 VAE chunk 状态怎样维护？
6. `runtime/models/dits/minwm.py`：为了 parity 覆盖了哪些 Wan operator boundary？
7. `runtime/pipelines/minwm_causal_dmd_pipeline.py`：这些 stage 怎样串起来，哪些并行参数被拒绝？
8. `tools/convert_minwm_checkpoint.py`：原始 `model.pt` 怎样变成自包含 serving tree？
9. `test/unit/realtime/test_minwm_realtime.py`：每条关键合同由什么回归测试保护？
10. `benchmark/minwm_realtime_parity/`：怎样生成 baseline/API、比较 lossless arrays、
    汇总吞吐并生成同步播放器？

## 11. 变更边界和后续工作

这次 PR 已覆盖 checkpoint 转换、模型注册、realtime adapter、action conditioning、
causal KV、数值 parity、benchmark、同步播放器和中英文文档。它没有声称完成：

- Ulysses/Ring sequence parallel；
- 多 session dynamic batching；
- 720p 24 FPS；
- whole-DiT compile 的 strict parity；
- 0721 checkpoint 的原生首帧训练合同。

推荐的下一阶段顺序是：先 profile VAE/RGB，做 decode/DiT pipeline overlap；再实现
MinWM packed attention 的 Ulysses all-to-all 和 KV ownership；最后在相同
prompt/seed/首帧/action 的多卡环境中重新建立 bitwise 或预声明 tolerance 的验收矩阵。

## 必须通过的测验

通过标准：总分至少 `17/20`，并且标有“关键”的第 2、5、8、12、15、17、19 题必须
全部正确。回答时必须用自己的话解释；只写“是/否”不得分。评审者可以要求指出对应
代码或实验字段。

1. 这次实现复用了 LingBot World 的哪些部分，又为什么不能直接复用它的 conditioning？
2. **关键：** 最新 0721 checkpoint 的原生 eval 合同与本次首帧 API 合同有什么区别？
3. 为什么默认分辨率是 832x480，而 720p 测试使用 1248x704 而不是 1280x720？
4. `primitive_token_residual` 的 primitive 顺序是什么？label 合法范围是什么？
5. **关键：** 要表达 `w=0.8` 应发送 label 还是 `action_weights`？给出具体八维 row。
6. 为什么 action encoder 每个 chunk 既需要历史四帧，也需要当前四帧？
7. 同一个 seed 下，把一次 BFCHW 抽样拆成多个 chunk 抽样为什么可能得到不同视频？
8. **关键：** bounded session 怎样复现 baseline 的随机数消费顺序？
9. 为什么 512-position zero-padded context 不能简单裁剪到真实 prompt 长度？
10. BF16→FP16→BF16 的 reference-latent boundary 为什么不能因“最终仍是 BF16”而删掉？
11. 为什么 value/shape/dtype 相等仍不足以推出两个 CUDA 路径 bitwise 相等？
12. **关键：** 排查第一个数值漂移时，为什么要先比较 operator input 的 stride？
13. H200 为什么使用 FA2，而 B200 可以使用 FA4？这对 parity 有什么影响？
14. TTFF、warm chunk latency 和 steady-state FPS 分别回答什么问题？
15. **关键：** MinWM 默认 KV window 是多少？bounded 和 unbounded session 怎样分配 cache？
16. 为什么 `frame_batch` 不能直接作为模型 chunk 的延迟完成条件？
17. **关键：** 当前设置 `ulysses_degree=2` 会发生什么？PR 为什么选择 fail-fast？
18. 在“DiT 理想无限加速”的假设下，720p pipeline 的 FPS 上限约是多少？
19. **关键：** 为什么任何有限 Ulysses degree 都不能单独达到 720p 24 FPS？
20. 如果由你负责下一轮优化，请列出三个按优先级排序的工作，并说明每项如何验证性能
    和数值一致性没有回退。

答题模板：

```text
1. ...
2. ...
...
20. ...
```
