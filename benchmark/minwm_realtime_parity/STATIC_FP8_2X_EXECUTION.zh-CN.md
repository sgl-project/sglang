# MinWM static FP8 2× 调查与执行记录

更新时间：2026-08-06

## 1. 验收合同

- 正式硬件：B200 Spot；只使用 AWS `spot` profile / Kubernetes `minwm-spot` context。
- 输入：固定 `1248x704`、4 DMD steps、KV45、每 chunk 4 latent / 16 pixel frames。
- 测量：20 个 warmup chunks + 200 个 measured chunks。
- 对照：同 checkpoint、同 commit、同输入、同服务参数；唯一实验变量必须被显式记录。
- 正确性：不允许静默 fallback；输出帧数、分辨率、协议和量化覆盖率必须满足相同合同。
- 目标：static FP8 的客户端端到端稳态 FPS 至少为 BF16 的 `2.0x`。

## 2. 第一轮事实

固定 720p B200 Spot 第一轮结果：BF16 `14.079` client FPS，online FP8
`14.122`，static FP8 `14.158`。static FP8 只提高 `0.56%`，没有达到目标。

同一轮 server trace 的全部 200 个 measured chunks 聚合（不是挑单个最优 chunk）：

| 阶段 | BF16 mean / p50 | static FP8 mean / p50 | static 相对 BF16 |
|---|---:|---:|---:|
| DiT denoise | 566.045 / 565.869 ms | 569.782 / 569.182 ms | `-0.66%`（变慢） |
| VAE decode pipeline | 468.655 / 468.592 ms | 466.961 / 467.032 ms | `+0.36%` |
| scheduler total | 1135.394 ms | 1129.063 ms | `+0.56%` |
| chunk total | 1173.071 ms | 1160.667 ms | `+1.07%` |

BF16 denoise 的 min/max 为 `560.364/579.700 ms`，static 为 `566.532/574.714 ms`；两组分布
整体重叠但 static 均值更慢。客户端的 `+0.56%` 不能归因于 FP8 DiT：它来自 VAE、raw frame
构造、WebSocket 写入及运行间噪声的合计差异。代表性 chunk 219（BF16/static 的 denoise
`568.381/571.671 ms`）与全量聚合结论一致。

这排除了“客户端传输吞掉了一个很大的 DiT 收益”：旧 static FP8 在模型层就没有拿到收益。

## 3. 与“调研 minWM 量化吞吐方案”任务的差异

隔壁任务旧 B200 矩阵为 `832x480`：BF16 `23.183`、online FP8 `26.653`、
static FP8 `24.859` client FPS。它与本轮不能直接横比：

1. `1248x704` 的像素数和 DiT token 数约为 `832x480` 的 `2.20x`。
2. 旧矩阵的 online FP8 与 static FP8 使用不同 SGLang SHA；Job annotation 还出现第三个 ref，
   因而不是只改变 activation scale 来源的同构 A/B。
3. 旧矩阵使用 step-3200 checkpoint 和默认 480p case；本轮固定 720p 的正式矩阵使用另一套
   checkpoint/case 合同。
4. 两轮 static checkpoint 的配置都是 `quant_method=fp8`；量化 registry 实际选择的是
   `Fp8LinearMethod`，而不是键为 `modelopt_fp8` 的 `ModelOptFp8LinearMethod`。旧
   `Fp8LinearMethod` 在 SM100 上仍把 serialized scalar weight scale 展开为 per-channel，
   并调用 generic `apply_fp8_linear`。因此 `+7.2%` 与 `+0.56%` 都不是修复后路径的上限。

隔壁任务还有一个容易造成“static 看起来很差/online 看起来很好”的计时事实：在它的 480p
实验里，online/static 的 denoise 分别为 `357.15/402.55 ms`，VAE 为
`219.97/219.26 ms`。约 30 blocks × 10 Linear × 4 steps，即约 1200 次 Linear/chunk；
`45.4 ms / 1200 ≈ 37.8 µs/Linear`，正好符合旧 static 路径每层额外 scale layout、Triton
static quant 和 generic scaled-mm dispatch 开销的量级，而不是 FP8 Tensor Core 的硬件极限。

另外，隔壁 A/B 的 online SHA 为 `47a8bcbf...`，static 实际 SHA 为 `4a66501d...`，Job
annotation 还记录 `afbee9c...`；static builder、权重量化尺度、calibration 分辨率和吞吐
case 也没有完全对齐。它适合发现候选方向，不足以给出“量化方法本身”的因果排序。本轮差异大，
首先来自口径（720p vs 480p）和实验非同构；更关键的是，早期本轮 micro 一度测了正式 E2E
根本不会调用的 `ModelOptFp8LinearMethod`。隔壁旧 static 和本轮旧 E2E 实际都走
`Fp8LinearMethod`，所以两边旧数字并不矛盾，也都不能代表修复后的上限。

最终已在同一 immutable SHA、同一 B200 Spot 节点、同一输入上完成四泳道
分层 A/B，见执行日志 `-06`；旧矩阵只用于解释历史偏差，没有被用来外推本轮上限。

## 4. 根因与修复决策

最初的代码审计发现，主 SRT 的 `ModelOptFp8LinearMethod` 已在 SM100 使用
`apply_fp8_linear_bmm_flashinfer`：保留 scalar weight/input scale，并调用 FlashInfer per-tensor
FP8 BMM；diffusion 的同名复制版则落后。早期修复和 micro 因此集中在这个类上。

随后把 builder 配置、量化 registry 和正式日志三者串起来，发现上述审计漏掉了运行时身份：
minWM static builder 写出的是 `quant_method=fp8`，registry 选择
`multimodal_gen/runtime/layers/quantization/fp8.py::Fp8LinearMethod`；正式日志也明确打印
`Detected fp8 checkpoint`。只有配置写 `modelopt_fp8` 才会进入此前修过的
`ModelOptFp8LinearMethod`。这意味着早期 helper micro 与正式 E2E 测的是两个不同实现。

真实旧 `Fp8LinearMethod` 在 serialized static SM100 上会把 scalar weight scale 展成
per-channel，并进入 generic `apply_fp8_linear`；static activation 还经旧通用 quant/scale
layout。online FP8 使用 CUDA per-token quant，所以旧 static 并不天然比 online 少开销。
这解释了为什么 helper micro 可见 `1.3x--1.58x`，而完整 DiT 约为 `1.0x`：不是 Amdahl
定律把明显的 DiT 收益吞掉，而是生产路径压根没有调用被测 helper。

最初假设是让 diffusion static FP8 在 SM100 + FlashInfer 可用时复用主 SRT 的 per-tensor
`bmm_fp8` 路径。但 `-08` 的真实 B200 数据否定了“主 SRT 路径也适合 DiT 大 M 矩阵”这一
假设：数值正确不等于性能正确。随后在完全相同的已量化输入、权重和 scalar scale 上增加
`torch._scaled_mm` 对照；`-09` 证明 cuBLASLt 才是当前 minWM 形状的较优后端。因此正式修复改为：

- SM100 保留 scalar weight/input scale；
- static activation quant 不再物化 `(M, 1)` scale；
- 对 16 对齐的 minWM 主矩阵调用 native `torch._scaled_mm`；
- 对其他 diffusion 模型可能出现的非 16 对齐 `K/N`，仅在该分支 pad 到 16 后裁剪输出；
- 其他硬件仍保留原 generic CUTLASS 路径。

最终修复 `f32df1c9b8...` 把同一组 SM100 scalar/native scaled-mm 规则接入真正的
`Fp8LinearMethod`：serialized static FP8 保留 scalar scale、构造 column-major transposed
weight，并在 `apply()` 直接路由 native scaled-mm；dynamic/online FP8 和旧 GPU 保持原路径。
测试使用真实 `Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="static")`，并规定
generic helper 一旦被调用就失败，从而防止再次出现“测试通过但生产 registry 不会调用”的假阳性。

验证分三层：

1. 单元/正确性：scalar scale、快路径路由、数值误差。
2. Spot GEMM/DiT：证明不是 silent fallback，并量化 Linear 与完整 denoise 的收益。
3. Spot 端到端：复用固定 720p 的 20+200 合同。

## 5. Amdahl 上限与最终归因

按全部 200 chunks 的 BF16 scheduler 均值，DiT 为 `566.045 ms`、VAE 为 `468.655 ms`、
其余调度约 `100.695 ms`。客户端基线 `14.079 FPS` 的 2× 目标要求 chunk interval 不超过
`16 / 28.158 = 568.22 ms`。当前顺序执行即使把 DiT 降到零，VAE + 其他调度仍约
`569.35 ms`，已略高于目标。因此只修 static FP8 GEMM 在物理上无法稳定满足客户端端到端 2×。

若把 VAE 放到同一台 p6 主机的另一张 GPU，并与下一 chunk 的 DiT 重叠，稳态 interval 不是把
VAE 再加到 DiT 上，而是近似
`max(DiT + scheduler CPU, VAE decode + raw transport + client delivery)`。早期把约 100 ms 的
“其它调度”全部加到 DiT 上、推导出 `DiT < 467.53 ms`，属于重复记账；正式 overlap trace 已纠正
这个模型。真正的进入条件是两个分支都低于 `16 / (2 * BF16 client FPS)`：一边看
`scheduler_forward/client`，另一边必须把远端 VAE 的解码、42.2 MB raw RGB 搬运和 WebSocket
交付都算进去。两项收益仍必须隔离报告，不能把两张 GPU 流水化全部归因给 static FP8。

这不是提前降低验收标准。先实测快路径后的 DiT/端到端数字；若 Amdahl 瓶颈按预期转移到
VAE/调度，则继续做不降低画质的流水化或阶段并行，并分别报告：

- static FP8 本身的隔离收益；
- 后续流水线优化的隔离收益；
- 组合方案相对当前 BF16 生产基线的端到端收益。

最终 `-06` 与上述模型一致：BF16-overlap 的 remote VAE 分支均值约
`478.398 ms`，低于 BF16 计算分支的 `581.595 ms`；static-overlap 的 remote VAE
分支约 `474.673 ms`，低于 static 计算分支的 `560.105 ms`。因此稳态间隔由
GPU0 计算分支决定，shared-memory 回传没有再成为新的隐性串行项。

## 6. 执行日志

### 2026-08-05：代码审计

- 发现 diffusion 与主 SRT 的 ModelOpt FP8 实现漂移。
- 决策：先同步 SM100 per-tensor FlashInfer 路径，不扩大到 NVFP4 或 Attention。
- 偏离预期：第一轮 static FP8 的 DiT 比 BF16 略慢；问题不是动态 scale 开销，而是 GEMM
  backend/scale layout 选择错误。
- 新增对照要求：所有结论必须同时标注分辨率、checkpoint、SGLang SHA 和计时边界。

### 2026-08-05：本地验证

- `py_compile`、`ruff check`、`ruff format --check` 和 `git diff --check` 通过。
- 新增 CPU 路由测试：mock SM100 + FlashInfer 时必须调用 per-tensor BMM，generic FP8 路径若被调用则失败。
- 更新原有 CUDA correctness test：SM100 快路径必须保留 scalar weight scale；其他 CUTLASS 路径仍验证
  per-channel scale。
- 偏离预期：本机默认 Python 3.9 不满足仓库 3.10+ 类型语法；本机 Python 3.11 的 Torch/Triton
  组合又在 pytest collection 阶段发生模块/类型冲突。它们发生在目标测试执行前，因此不能作为测试通过证据；
  完整 correctness test 移到固定镜像的 B200 Spot 上执行。

### 2026-08-05：B200 Spot 微基准 `-01`

- profile/context：`spot` / `minwm-spot`；Pod 落到已有 `minwm-test-b200-spot` 的
  `p6-b200.48xlarge`，没有落到 on-demand。
- 结果：pytest collection 在导入 SGLang 时因镜像缺少 `orjson` 失败；CUDA test 与 microbenchmark
  均未开始，不能用于判断修复有效性。
- 决策：保留 `-01` Job 和 S3 日志；新建 `-02`，复用首轮正式入口的 SGLang diffusion extra 与
  FlashInfer JIT cache 安装方式，不覆盖旧 run id。

### 2026-08-05：B200 Spot 微基准 `-02`

- 结果：原有 B200 Spot 节点在提交窗口内被回收；Pod 未进入 Running，等待 30 分钟后触发
  `DeadlineExceeded`。因此依赖安装、CUDA test 和 microbenchmark 都未执行。
- 证据：NodePool `minwm-test-b200-spot` 仍 Ready，但节点数已变为 0；Job event 是
  `FailedScheduling`，不是容器错误。
- 决策：保留 `-02`，以 `-03` 重提并把 deadline 扩为 60 分钟，给 Spot 补充容量更充分的时间；
  仍不切到 `aws03` 或 on-demand。

### 2026-08-05：B200 Spot 微基准 `-03`

- 结果：Karpenter 先后尝试多个 NodeClaim，最终在 east2b 创建 B200 Spot；完整依赖安装完成，
  但 pytest collection 因残留 `peft==0.17.0` 导入已从 `transformers==5.12.1` 移除的
  `HybridCache` 而失败。CUDA test 与 microbenchmark 未执行。
- 证据纠偏：Pod 消失后只看 Kubernetes 事件，曾把 `TaintManagerEviction` 误判为运行主因；S3 中实际
  留有 provenance 和 `pytest.log`，证明主因是 PEFT/Transformers API 不兼容。最终判断以 S3 日志为准。
- 决策：暂停尚未拿到节点的 `-04`，避免它用同样环境失败；`-05` 沿用短安装路径并显式
  `pip uninstall -y peft`。保留 suspended `-04`，不删除旧 Job。

### 2026-08-06：B200 Spot 微基准 `-05`

- profile/context：仍严格为 `spot` / `minwm-spot`；代码固定在 `761b76f520...`，镜像固定为
  `sha256:bedc07ea...`。
- 结果：多轮 `UnfulfillableCapacity` 后，NodeClaim `minwm-test-b200-spot-7hsjl` 在 east2b
  成功启动 `p6-b200.48xlarge` Spot。依赖安装继续推进，但 pytest collection 在导入 IPython 时
  因缺少 `traitlets` 失败；CUDA test 与 microbenchmark 均未执行，不能作为性能证据。
- 现场变量：同集群同时存在另一个请求 B200 的任务，会影响获取节点的等待时间；它不会共享本 Pod
  的 GPU，也不会进入正式计时窗口，因此节点成功独占后不影响测量合同。
- 决策：Pending 不产生 GPU 费用；为避免低 placement-score 时被 1 小时 deadline 人为截断，
  把 `activeDeadlineSeconds` 延长到 4 小时。此次变更不改变 profile、实例类型、镜像、代码或计时合同。
- 决策：保留 `-05` S3 日志；`-06` 使用新 run id，在最小依赖后让 pip 补齐 IPython 的传递依赖，
  并升级到含 quant/GEMM 分解计时的 immutable SHA `aae6435e9160...`。复用当前 Spot 节点，不覆盖旧结果。

### 2026-08-06：B200 Spot 微基准 `-06` / `-07`

- `-06` 补齐 IPython 后，pytest collection 继续暴露 `transformers==5.12.1` 与镜像原生
  `huggingface_hub` 的 API 不兼容（缺少 `is_offline_mode`）；CUDA test 与 microbenchmark
  仍未执行。
- 偏离预期与纠偏：顶层包逐个 `--no-deps` 升级破坏了固定镜像原有的一致环境。停止继续追补依赖；
  `-07` 回到镜像原生 Transformers/Diffusers/PEFT，只安装 `-01` 已确认缺失的 `orjson` 和固定
  FlashInfer JIT cache。新 run id 保留 `-06` 失败证据，并继续复用已启动的 B200 Spot 节点。
- `-07` 进一步证明训练镜像并不是完整 SGLang 开发环境：补完 `orjson` 后，collection 下一个缺失项是
  `pybase64`。因此 `-08` 不再逐项试错，恢复 `-03` 已经验证能完整安装的
  `pip install -e python[diffusion]`，并在安装结束后立即卸载导致 `HybridCache` 冲突的 PEFT；这是目前
  唯一同时覆盖完整依赖和已知兼容性修正的配方。

### 2026-08-06：B200 Spot 微基准 `-08` / `-09`

- `-08` 首次跑通完整证据链：固定镜像和 SHA、B200 SM100、5/5 correctness tests、四个
  minWM Linear 形状以及 S3 JSON 全部成功。
- 偏离预期：FlashInfer `bmm_fp8(backend="cublas")` 并不是这些单矩阵 DiT 形状的快路径。
  对 `(M,N,K)=(3432,3072,3072)` 与 `(13728,3072,3072)`，完整 static Linear 相对 BF16
  只有 `0.317x/0.728x`；两个大 MLP 形状也只有 `1.309x/1.245x`。纯 FP8 GEMM 已经偏慢，
  因此不能继续把问题只归因于 static quant 或 repeated scale。
- `-09` 在同一输入上加入 scalar `torch._scaled_mm`：四个形状的完整 Linear 相对 BF16 p50
  分别为 `0.804x/1.382x/1.482x/1.334x`。在 720p 4-frame 主形状上，它稳定优于 FlashInfer；
  输出相对 legacy 的 L2 差异只有 `1.28e-5` 到 `1.54e-5`。
- 决策：撤销 diffusion 对 FlashInfer BMM 的选择，改为 SM100 scalar cuBLASLt scaled-mm；
  FlashInfer helper 保留给主 SRT，不扩大改动范围。

### 2026-08-06：B200 Spot 微基准 `-10` / `-11`

- `-10` 的路由和两个 aligned correctness case 通过，但原有非对齐投影用例
  `(M,N,K)=(19,150,80)` 被 cuBLASLt 拒绝：`mat2 shape (80x150) must be divisible by 16`；
  总结果为 4 passed / 1 failed，因此没有进入 microbenchmark，也不能验收。
- 纠偏：只在 `K` 或 `N` 非 16 对齐时构造 column-major padded weight 和 padded activation，
  scaled-mm 后裁剪到原输出宽度；minWM 全部主矩阵是 16 对齐，正式热路径不承担 padding 成本。
- `-11` 使用 immutable SHA `3c910b87bc...` 在同一台 B200 Spot 上完成复测：5/5 correctness
  全部通过，包括 `-10` 失败的非 16 对齐投影；没有进入 generic FP8 fallback。
- 修复后实际 helper 相对 BF16 的 p50 speedup，按 `(M,N,K)` 分别为：
  `(3432,3072,3072) 0.743x`、`(13728,3072,3072) 1.301x`、
  `(13728,13824,3072) 1.578x`、`(13728,3072,13824) 1.268x`。输出相对 legacy
  static FP8 的 L2 差异为 `1.28e-5` 到 `1.54e-5`。
- 结论纠偏：cuBLASLt 修复显著抬高了三个大 M 热点形状的上限，但小 M 形状仍慢于 BF16；
  因此不能从单个 MLP 的 `1.578x` 外推整个 DiT，更不能外推端到端 `2x`。
- 完整产物：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/static-fp8-2x/minwm-static-fp8-fastpath-b200-20260806-11/`。

### 2026-08-06：固定 720p 成对 E2E `-02`

- 已提交 Job `minwm-static-fp8-fastpath-e2e-b200-20260806-02`；AWS profile/context 为
  `spot` / `minwm-spot`，node selector 强制 `p6-b200.48xlarge` + Spot。
- BF16 和 static FP8 在同一 Pod 顺序运行，固定 SGLang SHA `3c910b87bc...`、minWM SHA
  `2efc6485f6...`、checkpoint version、输入 case、20 warmup + 200 measured chunks；static lane
  复用 BF16 lane 的输入和转换后模型，避免重新取样或模型转换差异。
- Job 成功完成，两个 lane 都各收集 200 个 measured chunks。客户端 BF16/static 分别为
  `13.9848/13.9918 FPS`，static 仅为 `1.00050x`，未达到 `2x`。
- 全量 server trace 显示 BF16/static 的 DiT denoise mean 为 `570.829/575.734 ms`，即修复后的
  旧 static helper 在完整 DiT 中仍慢 `0.85%`；VAE decode 为 `470.452/468.994 ms`，scheduler
  total 为 `1143.043/1142.478 ms`。客户端的微小正差全部可由非 DiT 阶段与运行噪声解释，不能
  归因成 FP8 收益。
- 偏离预期：static 在最初 4 个非首 chunk 的 DiT 是 `423.425 ms`，看起来明显快于 BF16 的
  `446.304 ms`；但 KV45 尚未填满，二者在 chunk 10--19 已升到 `569.613/565.110 ms`，进入
  measured window 后稳定为上述结论。这证明必须保留 20-chunk warmup，不能引用短 smoke 的
  冷 KV 数字作为吞吐上限。
- 根因进一步收敛为两个独立层次：`-11` 已证明大 M Linear 的 native FP8 GEMM 有
  `1.27x--1.58x` 收益，但旧 Triton static activation quant 是逐行 program，约 1200 次
  Linear/chunk 的量化与 dispatch 开销抵消了 GEMM 收益；即使 DiT 修好，未量化且顺序执行的
  VAE 仍占 scheduler 的约 `41.2%`，按 Amdahl 上限无法单靠 static FP8 达到端到端 `2x`。
- 纠偏实现 `711537c535...` 把 activation quant 改为仓库已有的扁平 vectorized CUDA
  per-tensor static quant custom op；它跳过动态 absmax，并避免旧 Triton inline-asm 进入
  TorchInductor。B200 Spot 微基准 `-12` 将分别测旧 quant、新 quant、纯 scaled-mm 和完整
  helper，结果不与本次 `3c910b87bc...` 的 E2E 混写。
- 结果根目录：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/static-fp8-2x/e2e-paired-02/`。

### 2026-08-06：优化 activation quant 与精确 VAE overlap

- immutable SHA `711537c535...` 将 static per-tensor activation quant 改为已注册的扁平
  vectorized CUDA custom op，并继续使用 SM100 native `torch._scaled_mm`。它不是动态 quant：
  scale 仍来自离线 calibration，只把逐行 Triton program 和旧 inline-asm 路径替换掉。
- 微基准 Job `minwm-static-fp8-fastpath-b200-20260806-12` 会在同一输入上分别记录旧 quant、
  新 JIT quant、纯 scaled-mm 与实际 helper；只有 B200 测试和 JSON 完整后才能判断优化是否有效，
  不能从代码形态直接推断收益。
- 以当前 strict BF16 `13.9848408 FPS` 重算，2x 门槛为 `27.9696816 FPS`，对应每 16 帧 chunk
  间隔不超过 `572.05 ms`。当前 BF16 scheduler `1143.043 ms` 中 DiT 为 `570.829 ms`；即使
  DiT 变成 0，剩余约 `572.214 ms`，已经略高于门槛。这是“单 GPU 串行 static FP8 不可能
  稳定过线”的实测 Amdahl 证据，不是降低验收标准。
- 为测联合上限，immutable SHA `0b695a919c...` 实现第二张 GPU 上的原始完整 causal VAE decode。
  GPU0 返回已经完成 MinWM 预处理的 latent，API 层在发送 chunk N 时异步调用 GPU1，同时 GPU0
  计算 chunk N+1 的 DiT。协议保持 request/event/chunk index 一一对应；T2V 首 latent 重播后裁掉
  重复首帧，final chunk 只关闭一次 session。没有使用集群已有的 L4 + TAEHV 服务，因为 TAEHV
  是近似 decoder，会改变画质和验收语义。
- 四 lane Job `minwm-static-fp8-exact-vae-overlap-b200-20260806-01` 固定同一 setup/checkpoint/
  input，依次测 BF16-local、optimized static-local、BF16-exact-overlap、static-exact-overlap。
  其中 static-local 隔离量化收益，BF16-overlap 隔离拓扑收益，static-overlap 才是联合上限；
  最终不能把两张 GPU 流水化的全部收益归因给 static FP8。

### 2026-08-06：Spot 容量等待（偏离预期）

- 两个正式 Job 均使用 `AWS_PROFILE=spot`、context `minwm-spot`，node selector 强制
  `karpenter.sh/capacity-type=spot` 和 `p6-b200.48xlarge`；没有调用 `aws03`。
- 提交后 Pod 均停留在 `Pending`，NodePool `minwm-test-b200-spot` 自身为 `Ready`、nodes=0。
  Karpenter 多次创建并提名 NodeClaim，但 AWS Fleet 返回 `InsufficientCapacityError` /
  `UnfulfillableCapacity`，其中包含 east2b 的 p6-b200 容量不足。该状态不产生 B200 GPU 运行费，
  也不能用于判断代码正确性或性能。
- 决策：保留 Job 等待优惠 Spot 容量，不切换到 `aws03` 或 on-demand。容量到达后先执行固定镜像
  pytest 门禁和 `-12` micro；失败必须使用新 run id 保全 S3 证据，不能覆盖既有结果。

### 2026-08-06：activation quant micro `-12` / `-13`

- `-12` 的新增单测错误地用 `reshape()` 后张量对象是否同一来判断是否发生复制；即使连续内存
  reshape 不复制，Python Tensor wrapper 也不保证 identity。它是测试断言错误，不是 CUDA
  quant correctness 或性能失败；修复为比较 storage/data pointer 与输出数值。
- `-13` 在 B200 Spot 上完成 6/6 tests。四个 `(M,N,K)` 代表形状的完整 helper 相对 BF16 p50
  分别为 `0.8525x`、`1.3277x`、`1.5854x`、`1.2776x`。旧/新 activation quant p50 分别为
  `33.888/22.912 us`、`46.064/41.584 us`、`50.800/47.504 us`、`122.848/136.880 us`。
- 偏离预期：vectorized quant 不是所有大形状都更快；down projection 反而变慢。更重要的是，
  这些数字后来被确认属于 `ModelOptFp8LinearMethod` helper，不能证明正式 minWM static E2E
  已走这条路。产物：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/`
  `static-fp8-2x/minwm-static-fp8-fastpath-b200-20260806-13/`。

### 2026-08-06：真实 static FP8 路由根因与门禁 `-14` / `-15` / `-16`

- builder 生成的量化配置、registry 和正式日志共同确认：实际配置键是 `fp8`，实例是
  `Fp8LinearMethod`；此前优化的是键 `modelopt_fp8` 对应的另一个类。旧正式 E2E 因而从未执行
  此前的 fastpath。这是“micro 明显改善、whole DiT 完全不动”的首要原因。
- immutable SHA `f32df1c9b8...` 把 scalar/native scaled-mm fastpath 接入真实
  `Fp8LinearMethod`；测试用真实 static serialized config，并禁止 generic fallback。
- `-14` 在 7/8 tests 后失败：新增 weight-processing test 没初始化 tensor-parallel group；
  `-15` mock TP 后仍在 7/8 失败，因为测试层还在 CPU，而 `requantize_with_max_scale` 是 CUDA
  custom op。两次都发生在测试搭建，micro 未执行，不能当作 kernel 或实现失败。
- `-16` 把被测层移到 CUDA 后重跑完整 8-test 门禁和同形状 micro。正式四-lane E2E 必须等该
  门禁成功后才独占提交，避免把失败实现或并发 GPU 负载带入验收窗口。
- `-16` 最终在 B200 Spot 上 8/8 tests 通过，Job 正常 Completed。native scaled-mm helper
  相对 BF16 的四个 p50 speedup 为 `0.851x/1.323x/1.631x/1.286x`；该 micro 仍只用于确认
  backend/形状上限，生产覆盖证据来自真实 config 的路由与 weight-layout 测试，最终收益必须看
  whole-DiT/E2E。产物：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/`
  `static-fp8-2x/minwm-static-fp8-fastpath-b200-20260806-16/`。

### 2026-08-06：精确 VAE overlap `-01` / `-02` 的协议与实现身份纠偏

- `-01` 的 BF16-local/static-local 仍约为 `14.055/14.107 FPS`，这是尚未接入真实
  `Fp8LinearMethod` 的旧路径。首次 remote lane 在客户端调用
  `requests.Session.post(content=...)` 时失败；Requests 接口应使用 `data=`。修复仅改变 HTTP
  body 传递，不改变 payload 或解码语义。
- `-02` 修正 HTTP 后，BF16/static local 分别为 `14.004/14.074 FPS`；remote server 首个
  chunk 成功，第二个 chunk 返回 500。服务端日志显示 temporal tensor 大小 `2` 与 `4` 冲突。
- 根因不是网络：主 lane 的 optimized profile 显式 `MINWM_NATIVE_COMPONENTS=''`，加载日志为
  `AutoencoderKLWan (sgl-diffusion version)`；remote server 继承默认
  `MINWM_NATIVE_COMPONENTS=text_encoder,vae`，实际加载
  `AutoencoderKLWan (native-required version)`。两者即使模型目录和类短名相同，persistent
  causal cache 语义也不同；native 版本冷态首 chunk 约 `545 ms decode + 274 ms raw`，且第二
  chunk cache 失配，不能作为“精确 overlap”证据。
- `-03` 固定 remote server 的 `MINWM_NATIVE_COMPONENTS=`，并在健康检查后强制日志必须包含
  `Loaded vae: AutoencoderKLWan (sgl-diffusion version)`，否则正式 lane 立即失败。这样 remote
  与 optimized local 使用完全相同的 VAE 实现/权重/预后处理，再比较流水化收益。

### 2026-08-06：正式 `-03` 与 Nsight Systems 根因定位

- `-03` 在同一台 Spot B200 上完成 BF16-local/static-local；客户端分别为
  `14.0761/13.8870 FPS`，static 为 `0.9866x`。CUDA trace 中 BF16/static denoise p50 为
  `567.172/589.371 ms`，即真实 all-linear static FP8 反而慢 `3.94%`；VAE p50
  `466.727/463.526 ms`，不能解释 DiT 的退化。static checkpoint 日志明确为
  `Detected fp8 checkpoint`，builder 也记录 300 个量化权重，故不能再用 silent fallback 解释。
- `-03` 在进入 overlap lane 前失败，原因是进程仍把 VAE 日志文件打开在 S3 CSI 上时，shell
  同时读取该文件，返回 `Operation not permitted`；VAE 进程退出后远端日志实际存在，且包含
  预期的 `AutoencoderKLWan (sgl-diffusion version)`。`-04` 改为先写本地 `/work`，健康检查和
  实现身份检查都读本地文件，cleanup 在 kill/wait 后再复制到 S3。它不是 VAE 模型或 cache 失败。
- 依照 Nsight Systems 服务抓取流程，Job
  `minwm-static-fp8-nsys-b200-20260806-01` 在同一台 Spot B200 上分别启动 BF16/static server，
  每个 lane 先完成一次不计入结果的完整 warmup，再用 `nsys start/stop` 抓稳态窗口；trace 固定
  `cuda,nvtx`、fork tracing 与 CUDA graph node tracing，没有混用 PyTorch profiler。
- 原始 trace 因捕获边界包含不同数量的 block invocation，以下均用代表性的 FFN GEMM instance
  数 `930/997` 归一化。BF16/static 的 GPU kernel 总时间为 `6.797/6.387 ms`，说明 FP8 GPU
  计算本体其实快约 `6.0%`；但 capture wall time为 `8.159/8.466 ms`，static 仍慢 `3.76%`。
  这与 20+200 whole-DiT 的方向一致。
- 线性 GEMM 本体从 BF16 的约 `590.3 us` 降到 static FP8 的 `310.6 us`；输入量化增加
  `77.8 us`。真正异常是严格确定性模式：static 每个归一化 block 的 kernel 数从 `97.1`
  增至 `127.6`，`cudaLaunchKernel` 类 API 时间从 `1.787` 增至 `1.872 ms`；同时出现
  `per_tensor_quant_fp8` 7970 次、Float8 poison-fill 7970 次，并比 BF16 多约 16 次 BF16
  fill/block。根因是 `torch.use_deterministic_algorithms(True)` 默认给 `torch.empty` 输出插入
  uninitialized-memory fill，而 quant custom op 和 `_scaled_mm` 随后又完整覆写这些输出。
- 决策：保留确定性算法，只在 static FP8 quant + scaled-mm 两个“完整覆写输出”的操作范围内暂时
  关闭 poison fill，并用 `finally` 恢复全局设置；不关闭 deterministic attention，也不改变 scale、
  权重、输出 dtype 或数值路径。提交 `417d47cace...` 增加了作用域恢复测试与 deterministic B200
  micro。若 whole-DiT 仍不足，再采用已实现但尚未启用的 FFN-only scope，避免用无证据的层级猜测
  替换 kernel 证据。
- Nsight 产物：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/`
  `static-fp8-2x/nsys-static-fp8-01/minwm-static-fp8-nsys-b200-20260806-01-profile/`，每个 lane
  均保留 `.nsys-rep`、SQLite、stats、server/client log 与 throughput JSON。

### 2026-08-06：deterministic fill 修复复测 `-17` 与正式 all-linear `-04`

- deterministic B200 micro `-17` 为 9/9 tests；作用域退出后
  `deterministic_algorithms=true`、`fill_uninitialized_memory=true`，证明异常和正常路径都恢复全局设置。
  四个代表形状的完整 helper speedup 为 `0.820x/1.308x/1.709x/1.304x`：两个 FFN 大矩阵
  明显获益，小 M attention 仍亏损。
- 正式 `-04` 固定同一 Spot B200、同 checkpoint/case、20+200。BF16-local/all-static-local 为
  `14.0331/14.1382 FPS`，BF16-overlap/all-static-overlap 为 `26.3780/26.4198 FPS`；联合结果相对
  BF16-local 只有 `1.8827x`，未过线。
- all-static 的 local denoise CUDA mean 为 `571.623 ms`，BF16 为 `566.224 ms`；去掉 poison fill
  已把上一版 static 的约 `589 ms` 明显拉回，但全量 attention FP8 仍使 whole-DiT 慢约 `5.4 ms`。
  这证明 fill 是重要根因之一，但不是唯一根因。
- exact VAE server 日志最终从本地 `/work` 复制到 S3，且强制包含
  `AutoencoderKLWan (sgl-diffusion version)`；`-03` 的 S3 CSI open-file 权限问题不再出现。
- 产物：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/`
  `static-fp8-2x/e2e-exact-vae-overlap-04/`。

### 2026-08-06：FFN-only static FP8 `-05`

- Nsight 与 micro 均显示“并非所有 Linear 都该量化”。离线 builder 新增 `module_scope=ffn`：只把
  30 blocks × 2 FFN weights（共 60 个）写成 FP8；self/cross attention 仍只保存 BF16 权重，
  不保留同一层的第二份权重副本。输出 transformer 为 `7,364,523,872` bytes。
- 偏离预期：minWM 非 TP block 原先没有把完整 module prefix 传给 Linear，`ignored_layers` 无法在
  运行时区分 attention 和 FFN。修复为 self projection、`attn2`、`ffn` 都传真实 prefix；门禁用
  生产 `Fp8Config` 验证 attention 为 `UnquantizedLinearMethod`、FFN 为 `Fp8LinearMethod`。
- 目标镜像上 150 tests 全通过；转换 manifest 为 `module_scope=ffn`、`quantized_weights=60`、
  ignored `to_q/to_k/to_v/to_out/attn2`，排除了全量误量化和 silent fallback。
- 四 lane client FPS：BF16-local `14.12754`、FFN-static-local `14.41120`、BF16-overlap
  `26.07334`、FFN-static-overlap `26.53392`。FFN-only 隔离收益为 `1.02008x`，联合收益为
  `1.87815x`，仍未达到 client 2×。
- whole-DiT 根因被进一步隔离：local BF16/FFN-only denoise CUDA mean 为
  `565.985/545.165 ms`，FFN-only 实际省 `20.820 ms`。overlap lane 为 `569.503/543.853 ms`。
  static-overlap 的 scheduler 已到 `28.25582 FPS`，相对同作业 BF16-local 恰为 `2.00005x`；
  但 client 只有 `26.53392 FPS`，因为 scheduler `566.255 ms` 后，输出关键路径仍有
  `14.73 ms` raw payload join、`17.32 ms` WebSocket write 以及 42.2 MB HTTP MessagePack 搬运，
  client interarrival 被拉到 `603.002 ms`。
- 决策：计算路径已经过线，不继续无目标地改 GEMM。loopback exact VAE 对 raw 输出改为 GPU1 先
  拼 16 帧 transport batch，再通过 `/dev/shm` 返回小句柄；API 只读一次并删除文件。仅
  `127.0.0.1/localhost` 选择共享内存，非 loopback 或 WebP/JPEG 保留原 HTTP frame payload。
  同时依据 `-17` 的直接数据，大 M（`M>=8192` 且 contiguous）static quant 选择更快的 row-wise
  Triton kernel，小 M 继续使用 flat JIT kernel，给 2× 边界留出真实计算余量。
- 产物：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/`
  `static-fp8-2x/e2e-exact-vae-overlap-05/`。

### 2026-08-06：shared-memory 最终验收 `-06`

- 正式 Job `minwm-static-fp8-exact-vae-overlap-b200-20260806-06` 仅使用
  `AWS_PROFILE=spot` / context `minwm-spot`；Pod 调度到
  `ip-172-31-68-105.us-east-2.compute.internal`，node selector 为
  `capacity-type=spot` / `p6-b200.48xlarge`，申请 2 张 B200。Job 终态为
  `SuccessCriteriaMet=True`，没有调用 `aws03`。
- 运行时 SGLang SHA 为 `1ad71533009d72c04b4fc20c16ad73c1cd3d9540`，MinWM SHA 为
  `2efc6485f65e8fcab506665efde79bc41406385e`。目标 B200 镜像上 154/154 tests 通过；
  static manifest 仍为 `module_scope=ffn`、60 个 FP8 weights、attention ignored，排除了
  路由回退或全量误量化。
- 四泳道都使用同一 checkpoint/case、KV45、4 steps、20 warmup + 200 measured
  chunks：

| lane | client FPS | scheduler FPS | 相对 BF16-local |
|---|---:|---:|---:|
| BF16-local | `13.962218` | `13.975386` | `1.000000x` |
| FFN-static-local | `14.337702` | `14.351320` | `1.026893x` |
| BF16-exact-VAE-overlap | `27.455537` | `27.510553` | `1.966417x` |
| FFN-static-exact-VAE-overlap | `28.513450` | `28.566072` | **`2.042186x`** |

- 严格客户端验收线为 `2 * 13.962218 = 27.924437 FPS`，组合方案高出
  `0.589014 FPS`，即相对 2x 线有 `2.109%` 余量；客户端稳态间隔为
  `561.139 ms`，低于门槛 `572.975 ms`。因此结论不依赖 scheduler 内部口径。
- 隔离归因：FFN-only static 本地收益为 `1.026893x`；BF16 VAE overlap
  收益为 `1.966417x`；在 overlap 拓扑上再加 static 为
  `28.513450 / 27.455537 = 1.038532x`。对 BF16-local 的 `2.042186x` 是联合收益，
  不归因为“static FP8 单独 2x”。
- 关键输出瓶颈已解决：BF16/static overlap 的 `raw_payload_build_ms.mean`
  均为 `0.0 ms`，而 `-05` static 为 `14.73 ms`；远端完整 SGL VAE 的 200-chunk
  均值分别为 `decode/raw/total = 213.546/239.562/478.398 ms` 与
  `213.318/238.656/474.673 ms`，低于对应 GPU0 scheduler forward。运行中抽查
  `/dev/shm/sglang-realtime-vae` 未见遗留文件，API 读取后删除的契约生效。
- 本轮对“不到 2x”的完整解决链是：修复生产 registry 路由 -> 用 Nsight
  定位 deterministic poison fill / launch 税 -> 根据真实形状仅量化 FFN -> 大 M 选择
  更快 static quant kernel -> 用 GPU1 重叠原始完整 causal VAE -> 用本机共享内存
  移除 42.2 MB/chunk HTTP MessagePack 回传。每一步都有独立 lane 或 kernel/trace
  证据，没有用降低画质的近似 VAE 换取验收。
- S3 根目录：`s3://leap-world-us-east-2/world-model/evals/minwm/quantization/20260805/`
  `static-fp8-2x/e2e-exact-vae-overlap-06/`。其中保留 provenance、Job 日志、
  154-test 日志、四份 throughput JSON/server log、static manifest 和 exact-VAE server log。
