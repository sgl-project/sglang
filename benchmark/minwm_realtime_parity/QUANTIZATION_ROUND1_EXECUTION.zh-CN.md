# minWM SGLang 量化第一轮执行记录

更新时间：2026-08-05（Asia/Shanghai）

## 目标与边界

本轮只回答一个问题：在固定 720p MinWM realtime 请求上，现有量化路径与 whole-DiT compile 组合能把吞吐上限推到哪里。

本轮不是质量放行实验，不做多 case 视觉回归，也不据此直接决定生产默认值。任何性能胜出项都必须进入后续质量轮才能放行。

## 固定实验合同

| 项目 | 固定值 |
|---|---|
| 硬件 | AWS Spot `p6-b200.48xlarge`，单 Pod、单 GPU、所有 lane 串行 |
| 集群 / 可用区 | `minwm-spot`（`leap-world` east2）/ 实际 `us-east-2a` |
| NodePool | `minwm-test-b200-spot` |
| 模型代码 | `seedleap/minWM@2efc6485f65e8fcab506665efde79bc41406385e` |
| checkpoint | `global_step_003200/ema_student/model.pt`，S3 VersionId `wduScksw2f3yPErnG9lBioOuE2AToyAP` |
| 请求 case | `cases_720p_compile_smoke.json` / `00_forward_080_pottery_720p` |
| 分辨率 | `1248x704`（项目中的 720p 合同） |
| realtime 参数 | 4 latent frames/chunk，KV cache 45 frames，4 denoise steps |
| 测量窗口 | warmup 20 chunks + measured 200 chunks |
| attention / components | dense、SGLang optimized components |
| 主指标 | client steady received FPS（ratio of sums） |
| 次指标 | scheduler forward FPS、chunk p50/p95/p99、失败类型、峰值显存 |

公平性规则：除量化方法、NVFP4 kernel backend、whole-DiT compile 开关外，其余输入和服务参数保持不变。每个量化配置先测 eager，再测 compile。

## 第一轮矩阵

| # | 权重 / 激活 | kernel backend | compile | 状态 | client FPS | scheduler FPS | 相对 BF16 同 compile |
|---:|---|---|---|---|---:|---:|---:|
| 1 | BF16 | BF16 GEMM | off | 完成 | 14.079 | 14.092 | 基线 |
| 2 | BF16 | BF16 GEMM | on | `cold-compile-timeout` | - | - | 失败基线，不计 FPS |
| 3 | online FP8 W8A8 dynamic | SGLang FP8 | off | 完成 | 14.122 | 14.135 | +0.30% |
| 4 | online FP8 W8A8 dynamic | SGLang FP8 | on | `cold-compile-timeout` | - | - | 失败项不计 FPS |
| 5 | calibrated FP8 W8A8 static | ModelOpt static FP8 | off | 完成 | 14.158 | 14.171 | +0.56% |
| 6 | calibrated FP8 W8A8 static | ModelOpt static FP8 | on | Triton/Inductor 编译失败 | - | - | 失败项不计 FPS |
| 7 | ModelOpt NVFP4 W4A4 | `flashinfer_trtllm` | off | 完成 | 13.706 | 13.718 | -2.65% |
| 8 | ModelOpt NVFP4 W4A4 | `flashinfer_trtllm` | on | `cold-compile-timeout` | - | - | 失败项不计 FPS |
| 9 | ModelOpt NVFP4 W4A4 | `flashinfer_cutlass` | off | 完成 | 13.269 | 13.280 | -5.76% |
| 10 | ModelOpt NVFP4 W4A4 | `flashinfer_cutlass` | on | `cold-compile-timeout` | - | - | 失败项不计 FPS |
| 11 | ModelOpt NVFP4 W4A4 | `flashinfer_cudnn` | off | 完成 | 12.202 | 12.211 | -13.34% |
| 12 | ModelOpt NVFP4 W4A4 | `flashinfer_cudnn` | on | FlashInfer / stream API 失败 | - | - | 失败项不计 FPS |

## 预期与判读

- BF16 compile 是当前已知强基线；量化必须与相同 compile 状态比较，不能把 compile 收益误归因给量化。
- online FP8 的动态 activation scale 有运行时成本，可能降低理论收益；它的价值是零离线校准成本。
- calibrated static FP8 消除动态 activation scale，预期比 online FP8 更接近 FP8 kernel 上限。
- NVFP4 显著降低 DiT 线性层带宽，但 realtime 端到端仍包含 VAE、调度和传输，因此 DiT kernel 收益不会等比例变成 client FPS。
- 三个 NVFP4 backend 先全部试跑；backend 不支持、数值或 compile 失败也属于本轮有效结论，不做静默回退。

## 已知限制

- MinWM NVFP4 builder 当前只使用校准 dump 的 `forward_000.pt`。这足够用于本轮固定 case 性能探索，但不能代表多状态质量覆盖。
- 静态 FP8 与 NVFP4 只量化 30 个 transformer block 中的大型 linear（共 300 个目标层）；action encoder、embedding、norm、scheduler、VAE 保持 BF16。
- 单次测量主要用于找上限。胜出组合后续仍需重复测量、误差条和多 case 质量回归。

## 执行日志、问题与决策

### 2026-08-05：运行前盘点

1. 发现仓库旧 B200 量化 YAML 使用 `ray` namespace、`minwm-test-b200-spot` NodePool、`minwm-test-b200-karpenter` capacity label 和 `seedleap.ai/workload=wan22-ti2v` toleration。
2. 当前集群没有 `ray` namespace；可用输入 PVC、GitHub secret 和 ServiceAccount 位于 `default` namespace。旧 NodePool 也不存在。
3. 当前可用 B200 Spot NodePool 是 `minwm-test-atl2-p6-spot`，固定 `us-east-1-atl-2a`，capacity label / taint 为 `minwm-test-atl2-karpenter`，从零扩容，8 小时到期。
4. 决策：新建第一轮专用清单，不修改或误用旧集群清单；namespace 改为 `default`，使用当前 NodePool 与 taint，输入只读挂载 `s3-claim`，结果写入专用 EBS PVC。
5. 发现 throughput client 默认读取 `cases.json` 的 832x480 case；而静态 FP8 / NVFP4 校准使用 1248x704 case。若直接复用旧入口，会产生“720p 校准、480p 测速”的不可比结果。
6. 决策：入口新增显式 throughput cases/case/warmup/measured 参数。本轮校准和测速统一为同一个 1248x704 case。
7. 决策：所有 lane 放在同一 Pod 串行运行，请求 1 张 GPU。BF16 首次完成 staging、依赖安装和模型转换；后续 lane 复用 Pod 内输入与 BF16 模型，避免重复 S3 staging。NVFP4 只导出一次，三个 backend 复用同一份量化权重。

### 2026-08-05：首次提交与 Spot 调度

1. `kubectl apply --dry-run=client`、`kubectl apply --dry-run=server` 和清单内嵌脚本的 `bash -n` 均通过后，创建专用 PVC 与 Job。
2. Karpenter 正确为 Pod 提名 `minwm-test-atl2-p6-spot` NodeClaim，证明 namespace、selector、toleration 和资源请求已经进入预期调度路径。
3. AWS Fleet 随后连续返回 `UnfulfillableCapacity` / `InsufficientCapacityError`，失败发生在创建 `p6-b200.48xlarge` Spot 实例阶段。PVC 的 `WaitForFirstConsumer` 是该 Pending 状态的结果，不是独立存储故障。
4. 决策：先保留 Pending Job 让 Karpenter 自动重试，不改成 B300 或 On-Demand。否则硬件变化会使本轮与既定 B200 合同不可比。若最终决定切硬件，必须新建单独矩阵并明确标注，不能覆盖本轮身份。
5. B200 在容量失败缓存窗口后仍无可行 offering。为了让第一轮量化方法探索继续推进，决策新增独立的 `p6-b300.48xlarge` Spot fallback Job/PVC；它保持相同 12-lane 请求合同，但使用独立 matrix id，数据不得标为 B200。
6. B200 Job 先保留 Pending 继续争取容量；如果 B300 已 Running，则暂停 B200 Job，避免同一第一轮在容量恢复后意外双跑。暂停是可恢复状态，不删除已有 Job/PVC 或诊断事件。

### 2026-08-05：识别到正确的 east2 Spot 控制面

1. use1 的 B300 fallback 也连续收到 `UnfulfillableCapacity`，说明 ATL2 单可用区 P6 池当时同时缺 B200 与 B300。
2. 进一步检查本机 kubeconfig 后发现 `minwm-spot` context 指向 us-east-2 的 `leap-world` 集群。这里正好存在旧量化清单引用的 `ray` namespace、`minwm-test-b200-spot`、`s3-claim` 和 `github-token`。
3. east2 的 B200 Spot NodePool 跨常规三 AZ，且当天已有 static-FP8 与 NVFP4 Job 成功完成，说明它比 ATL2 单可用区池更符合本轮运行预期。
4. 决策：暂停 use1 的 B200 / B300 Pending Job，保留 Job、PVC 和事件用于审计；没有创建 GPU、没有结果数据被中止。第一轮转到 east2 的 B200 Spot 池，结果写到独立 S3 前缀。

### 2026-08-05：读取旧量化结果与 compile 失败证据

1. 旧结果合同明确是 `832x480 / 00_forward_pottery`，不是本轮固定的 `1248x704`。eager client FPS 分别为 BF16 `23.183`、online FP8 `26.653`、static FP8 `24.859`、NVFP4 `20.173`。
2. 因此旧数据只能形成先验：online FP8 约比 BF16 快 `15.0%`，static FP8 约快 `7.2%`，该次 NVFP4 反而慢约 `13.0%`；不能把它当作本轮 720p 结论。
3. 旧 BF16 / online FP8 compile lane 均失败。日志显示 whole-DiT 在 KV 长度增长阶段反复产生冷编译：BF16 后段单 chunk 曾达到 `337–391 s`，online FP8 首 chunk 约 `150 s`；client 最终收到 WebSocket `1012 service restart`。这更像“长冷编译暴露在 Spot / Job 生命周期内”而不是稳态 FPS 失败。
4. 决策：本轮仍保留 20 个 warmup chunk，让 KV45 前的形状编译不进入 measured 200 chunks；east2 Job deadline 从 4 小时延长到 7 小时。各 lane 结果直接写独立 S3 目录，即使 Spot 中断也保留已完成证据，后续只补缺失 lane。
5. 决策：将运行顺序从“每种量化 eager+compile 连跑”改为“先完成 6 个 eager，再运行 6 个 compile”，且 compile 阶段按旧先验把 online FP8 放在最前。这样长冷编译或 Spot 中断不会阻止其余量化方法先拿到 eager 上限。

### 2026-08-05：增加 west2d 独立 P6 Spot 供给路径

1. east2 B200 Job 持续收到 `InsufficientInstanceCapacity`；NodePool 的另一名称虽然不同，但与现有池使用相同的三组 subnet、security group 和 AMI，不构成新的 AWS 容量路径，因此没有并行提交重复 Job。
2. `codex-minwm-test-phx2` 集群存在 `minwm-sp12-usw2d-p6-spot`，位于 `us-west-2d`，允许 `p6-b200.48xlarge` 与 `p6-b300.48xlarge` Spot，是真正独立于 east2 / ATL2 的供给路径。另一个名称相似的 `minwm-test-usw2d-p6-spot` 当前 GPU limit 已为 0，不能使用。
3. 该集群的 `s3-claim` 是只读卷，不能像 east2 一样把每条 lane 直接写回 S3。决策：结果写入专用 `minwm-dmd-0724-p6-gp3` RWO PVC；矩阵末尾仍把完整 summary 打到 Job log，PVC 保留逐 lane 原始数据。
4. 同集群刚完成的 H200 BF16 / online FP8 作业使用旧 SHA `7cb482cc...`、832x480 默认 case 且仅含 eager，因此只作为供给与执行链路旁证：client FPS 分别为 `18.958` 和 `21.271`（FP8 `+12.2%`），不能并入固定 720p 正式矩阵。
5. 决策：west2d Job 不锁死 B200 或 B300，让 Karpenter 在该 NodePool 的合法 P6 offering 中选择实际可获得的 Spot 型号；最终结果必须绑定实际 node / instance type / GPU 名称，不与 B200 基线混写。east2 B200 在 west2d Job 获得 GPU 后暂停，避免两个完整矩阵意外同时运行。

### 2026-08-05：发现 AWS03 跨三 AZ 的托管 B200 Spot 节点组

1. west2d Karpenter Job 首次请求也得到 `UnfulfillableCapacity`。继续检查 kubeconfig 中非 Karpenter 集群后，发现 AWS03 账户的 us-west-2 集群存在 `minwm-spot-p6-b200-0703` 托管节点组：`capacityType=SPOT`、B200、跨三个 subnet、`desired=0 / max=20`。
2. 同集群现有 8 台 B300 节点属于 `wan22-cb-*` Capacity Block 节点组，不是 Spot。决策：不使用这些现成节点来替代 Spot，也不把 Capacity Block 数据混入本轮。
3. AWS03 west 集群有 `github-token`，并将 east2 的 S3 bucket 以 RWX `s3-claim` 挂载；旧 MinWM west Spot 清单也验证过相同 nodeSelector、taint 和路径。它不需要专用 EBS PVC，可继续逐 lane 直写 S3。
4. 集群没有 Cluster Autoscaler deployment。决策：先提交专用 Job，再将 `minwm-spot-p6-b200-0703` 的 desired size 从 0 临时调到 1；实验终止后恢复为 0。这个节点组本身明确是 Spot，且跨三个 AZ，比单 AZ Karpenter 路径更有机会拿到容量。
5. B200 ASG 的第一次实例启动仍返回 `UnfulfillableCapacity`。为避免上限探索只押一个 SKU，新增同集群 `minwm-spot-p6-b300-0703` 的独立固定 720p 矩阵，并将 desired size 临时调到 1。B300 是独立硬件结果；B200 / B300 谁先获得真机，就暂停另一个尚未获得 GPU 的完整矩阵并恢复其 desired size 为 0，避免并行消耗两台 P6。
6. 多条 P6 路径均持续等待容量时，phx2 集群已有 H200 Spot 节点且存在空闲 GPU。决策：并行运行固定 720p 的 BF16、online FP8、static FP8 eager/compile 子矩阵，用它回答 FP8 路径与 compile 的趋势；Hopper 不具备本轮 NVFP4 硬件路径，因此 6 条 NVFP4 lane 显式标记 `hardware-not-supported`，不尝试后静默失败，也不把 H200 数据与 Blackwell 上限合并。

### 2026-08-05：正式 B200 Spot 获得容量并完成 eager 矩阵

1. east2 Job 在多轮 `InsufficientInstanceCapacity` 后获得 `p6-b200.48xlarge` Spot 节点 `ip-172-31-108-150.us-east-2.compute.internal`，实际 AZ 为 `us-east-2a`。Pod `minwm-quant-r1-b200-20260805-01-use2-v2-52vl7` 请求 1/8 GPU，镜像 digest 与 SHA 均符合合同。
2. 节点获得后立即暂停 west2d Karpenter Job 与 AWS03 B200/B300 托管节点组 Job；AWS03 两个 ASG 都恢复 `desired=0`、实例数 0。没有删除 Job/PVC/事件。
3. H200 fallback 确实新建了 `p5e.48xlarge` Spot 节点并完成三条 fixed-720p eager：BF16 `9.122`、online FP8 `9.421`（`+3.28%`）、static FP8 `9.417`（`+3.23%`）。它的 online-FP8 compile 首 chunk 长时间停在 CPU Dynamo/Inductor 冷编译；正式 B200 稳定后暂停 H200 Job，保留 Bound PVC 和 eager 原始数据，不继续占用整台 H200。
4. 两个集群的 pip 安装都打印了 `ERROR: pip's dependency resolver...`，但随后成功完成安装且 lane status 为 0。决策：将其记录为非致命依赖提示，不能仅按日志关键字判失败；判定以命令退出码与 lane end marker 为准。
5. H200 新节点注册初期，EBS provisioner 曾报告缺少 topology key，S3 CSI 也曾短暂未注册。节点标签与 CSI DaemonSet 就绪后 PVC 自动 Bound、Pod 自动启动；没有通过重建资源掩盖这个时序问题。
6. B200 eager client FPS 依次为 BF16 `14.079`、online FP8 `14.122`、static FP8 `14.158`、NVFP4 TRT-LLM `13.706`、NVFP4 Cutlass `13.269`、NVFP4 cuDNN `12.202`。static FP8 暂居第一但只比 BF16 快 `0.56%`；NVFP4 三个 backend 都是负收益。
7. 旧 480p online FP8 的约 `+15%` 没有迁移到本轮 720p B200（仅 `+0.30%`）。这是本轮最重要的预期偏离，说明分辨率、VAE/传输占比或非 GEMM 部分足以吞掉量化 kernel 收益；后续不能用旧 480p 数据外推 720p。
8. 成本约束补充：后续只使用 `spot` profile 对应的 `minwm-spot` 路径，不再使用 `aws03` profile；前者有优惠、更便宜。AWS03 路径本轮只做过供给探测，B200/B300 都始终为 0 实例，现已保持 Job suspended、ASG desired=0，不计入结果。
9. `online-fp8-compile` 于 `11:24:13Z` 启动后，30 分钟内仍无法完成 20 个 warmup chunk。运行中从 stdout 看不到 chunk 级进度，一度误判为“首 chunk 未返回”；lane 关闭后复核 S3 server trace，确认实际完成了 `chunk_index=0..4`，各 chunk 总耗时约 `234 / 177 / 278 / 399 / 528 s`。这说明问题不是单次首编译，而是 KV / shape 推进过程中反复发生长编译。期间 32 个 Inductor worker 持续换批、服务进程存活、显存约 `57.3 GiB`，GPU 大部分时间利用率为 0。为控制 Spot 成本并覆盖剩余方法，决策将各 compile lane 的完整 warmup 等待上限设为 30 分钟；该 lane 在 `11:56:58Z` 结束，记录为 `cold-compile-timeout`，不补零 FPS。服务进程收到 TERM 后 30 秒未退出，因此最终使用 KILL；client 如实留下 WebSocket `1012 service restart`，矩阵随后进入 `bf16-compile`。
10. `bf16-compile` 也在 throughput client 运行 `1804 s` 后未完成 warmup、未产出 throughput 文件。事后 trace 同样只完成 `chunk_index=0..4`，耗时约 `185 / 166 / 271 / 390 / 518 s`；scheduler 进程约占用一个 CPU core，显存约 `62.0 GiB`，GPU 大部分时间为 0。它于 `12:29:59Z` 按同一规则结束并进入 `static-fp8-compile`。首次 watchdog 虽打印了 timeout marker，但因远端 awk 引号错误没有取到服务 PID；只读进程复核发现偏差后，改用精确进程匹配执行 TERM/KILL，并修正后续 watchdog。最终状态以 Pod 内进程、server trace 和 lane end marker 为准，不以 watchdog 自身的日志声明为准。
11. `static-fp8-compile` 在首个请求约 28 秒后主动失败，并非超时。底层是名为 `_static_quant_fp8` 的 Inductor/Triton kernel 编译失败：TTIR 分析无法处理 `tt.elementwise_inline_asm`，随后 `RuntimeError: PassManager::run failed`。因此该 lane 标记为明确的 compiler incompatibility，不归因于 Spot，也不允许回退到 eager 后冒充 compile 结果。
12. `nvfp4-trtllm-compile` 同样只完成 `chunk_index=0..4`（约 `192 / 171 / 274 / 400 / 529 s`），没有完成 warmup，也没有 throughput 结果；最终于 `13:10:17Z` 结束并标记 `cold-compile-timeout`。`nvfp4-cutlass-compile` 的 client 超过 30 分钟后，watchdog 曾瞬时查询不到服务 PID，却错误地把该 lane 加入“已处理”集合，导致它继续运行到 `14:05:10Z` 才由人工复核强制结束，实际 lane 时长约 55 分钟，并因此多完成到 `chunk_index=6`（后两 chunk 约 `687 / 719 s`），仍远未完成 20 个 warmup。这是成本控制偏离预期；最后一条 cuDNN lane 改用单 lane watchdog：每 30 秒确认 client 仍是 cuDNN，30 分钟后必须解析并验证服务 PID，只有 client 消失或进程确认结束才退出。
13. `nvfp4-cudnn-compile` 在首请求约 29 秒后明确失败：FlashInfer cuDNN FP4 路径把 `torch.Stream` 当作带有 `cuda_stream` 属性的对象，触发 `AttributeError: 'torch.Stream' object has no attribute 'cuda_stream'`。同一 cuDNN backend 的 eager lane 能完整运行，因此这是 cuDNN FP4 与 torch.compile 组合的 API / graph integration 失败，不是 backend 在 B200 上完全不可用。矩阵最终因 6 条 compile lane 非零退出而以 `BackoffLimitExceeded` 标记 Job Failed；这是聚合脚本的预期失败语义，6 条 eager summary 和全部失败日志均已写入 S3。

## 作业证据

- SGLang immutable SHA：`ca2509f03432d07f183f8f1816c1ae1f218ec6a0`
- Job：`ray/minwm-quant-r1-b200-20260805-01-use2-v2`
- Matrix ID：`minwm-quant-r1-b200-use2-20260805-02`
- Pod：`minwm-quant-r1-b200-20260805-01-use2-v2-52vl7`
- Node：`ip-172-31-108-150.us-east-2.compute.internal`；`p6-b200.48xlarge`；Spot；`us-east-2a`
- image digest：`sha256:bedc07ea3ba55059a8c1c569c3b177c4d00d41f37d4fa9105375531534ef5f2a`
- GPU request：1 x B200
- 结果前缀：`/s3/world-model/evals/minwm/quantization/20260805/round1-fixed-720p`
- 汇总文件：上述前缀下的 `minwm-quant-r1-b200-use2-20260805-02-matrix-summary.json`
- 终态：`Failed / BackoffLimitExceeded`；原因是聚合脚本保留 6 条 compile lane 的非零状态，不是 Spot 中断。Pod 已结束，不再占用本轮 GPU request。

## 结果结论

### 本轮可测上限

在固定 B200 Spot、1248x704、KV45、4 steps、20+200 chunks 合同下，本轮拿到的最高有效 client FPS 是 calibrated static FP8 eager 的 `14.158`，相对 BF16 eager `14.079` 只提高 `0.56%`。online FP8 eager 为 `14.122`（`+0.30%`）。单次运行没有误差条，因此这两个提升都应视为“基本持平、static FP8 略占优”，而不是足以直接改生产默认值的显著收益。

NVFP4 三个 backend 均完整跑通 eager，但性能全部下降：TRT-LLM `13.706`（`-2.65%`）、Cutlass `13.269`（`-5.76%`）、cuDNN `12.202`（`-13.34%`）。所以当前实现的 W4A4 带宽优势没有转化为 MinWM 720p realtime 的端到端吞吐优势；最好的 NVFP4 也落后 BF16。

| eager lane | peak GPU memory MiB | 相对 BF16 峰值 |
|---|---:|---:|
| BF16 | 61,960 | 基线 |
| online FP8 | 57,270 | -7.57% |
| static FP8 | 57,618 | -7.01% |
| NVFP4 TRT-LLM | 55,894 | -9.79% |
| NVFP4 Cutlass | 55,894 | -9.79% |
| NVFP4 cuDNN | 55,922 | -9.74% |

量化确实降低峰值显存，但幅度远小于 transformer linear 权重位宽的缩减比例，因为 VAE、attention、KV/cache、激活和未量化模块仍占用显存。若目标是吞吐，NVFP4 当前不值得采用；若目标是显存容量，它约 10% 的端到端峰值节省可作为后续独立课题，不能与加速混为一谈。

### compile 结论

本轮 6 条 compile lane 没有任何一条完成 20+200 合同，因此没有合法的 BF16 compile 基线，也没有量化 + compile 的有效 FPS。不能拿某个 compile lane 的局部 chunk 时间与 eager BF16 比较，也不能把失败项写成 0 FPS。

- online FP8、BF16、NVFP4 TRT-LLM、NVFP4 Cutlass：KV / shape 推进时反复触发分钟级编译，只完成 5–7 个 warmup chunk；单 chunk 最长达到约 `719 s`，在成本上限内无法进入 steady measured window。
- static FP8：`_static_quant_fp8` Triton kernel 的 TTIR 分析无法处理 `tt.elementwise_inline_asm`，`PassManager::run failed`。
- NVFP4 cuDNN：FlashInfer cuDNN 路径对 `torch.Stream.cuda_stream` 的假设与当前 PyTorch API 不兼容。

因此，本轮全局“有效结果”最优仍是 static FP8 eager `14.158 FPS`；whole-DiT compile 的理论稳态上限没有测到。实际证据更接近“当前 compile 形状策略不可部署”，后续若继续，应先解决固定 KV shape 的预编译 / cache 复用与两个明确 compiler/API 兼容问题，再谈稳态收益。

### 端到端归因与放行决策

所有 eager lane 的 scheduler FPS 仅比 client FPS 高约 `0.08%–0.09%`，量化差异基本完整传导到客户端；WebSocket 写出不是吞掉 FP8 收益的主要环节。720p 下只有约 0.3%–0.6% 的 FP8 收益，更可能说明未量化路径和 kernel/runtime 开销主导了 Amdahl 上限，而不是网络层把一个很大的 DiT 加速吃掉。

第一轮到此结束，不修改生产默认值。static FP8 若要继续，必须进入重复测量与多 case 质量轮；online FP8 可作为零离线校准成本的低风险候选，但当前性能收益接近噪声；NVFP4 加速方向暂不推进，除非后续 profile 证明可修复的特定 kernel 开销。
