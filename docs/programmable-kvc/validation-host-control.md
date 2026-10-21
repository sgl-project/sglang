# KV 控制与 HiCache 验证记录

本记录覆盖本地 SGLang 实现的控制入口、副本优先级和持久 host 层搬运。默认关闭控制；开启后执行显式命令。本阶段支持单个 TP/PP/CP/DP owner，HiCache 搬运限于 FULL 状态、非 overlap 调度和本地持久 host 池。

## CPU 验证

各文件在独立进程运行，使用 `CUDA_VISIBLE_DEVICES=999` 隐藏 GPU，并显式选择 Python TreeCore。缓存测试使用真实的 tensor 池、allocator、radix tree 和 HybridCacheController。搬运测试只控制 L2 DMA 及其完成通知，实际复制 K/V tensor。

| 测试文件 | 通过项 |
| --- | ---: |
| `managers/test_kv_control.py` | 78 |
| `mem_cache/test_kv_control_cache.py` | 21 |
| `mem_cache/test_kv_control_service.py` | 12 |
| `mem_cache/test_kv_control_transfers.py` | 14 |
| `mem_cache/test_unified_hicache_transfer_lifecycle.py` | 8 |
| `managers/test_prefill_adder.py` | 53 |
| `managers/test_schedule_policy.py` | 11 |
| `spec/test_weight_update_runners.py` | 1 |
| `managers/test_scheduler_weight_version_tracking.py` | 20 |

共 218 项。前 217 项的逐文件输出保存在 `/tmp/sglang-kv-host-cpu-proof.txt`；增加 flush 后资源批准回调的回归测试后，搬运与服务两个文件合计 26 项通过。

覆盖行为包括：共享前缀只计费一次、保留运行中请求的 decode 增长估计、整个可复用前缀就绪前持续持有引用、DMA 期间 radix 分裂、单个与全部等待者取消、部分 MOVE、部分提交失败、失败结果的 ACK/重放、独立重试，以及分配前的范围和结果预算检查。超过目标池总容量的操作返回终态 `deferred`，避免留下无法完成的活跃记录。临时容量不足采用可取消的 `queued` 状态。

缓存 flush 会更换 epoch，同时保留调度器的资源批准回调。权重更新的受控流程已另行实现，见文末记录；低层未启用实际 key 命名空间的调用模式仍拒绝参数修改。

## 单卡验证

使用 GPU 1：NVIDIA A100-SXM4-80GB。目标虚拟环境安装了 PyTorch 2.14.1+cu130、torchvision 0.29.1、sglang-kernel 0.4.10；其他项目的环境和已有进程保持原状。

真实 CUDA 缓存测试通过优先级淘汰、KV 字节保留、请求引用保护、压力下回收、副本重建与重放，以及 allocator 余额检查。独立 HiCache 测试使用 direct I/O，验证 GPU→host MOVE 和 host→GPU COPY。

恢复测试会先将已经释放的 GPU 缓冲区填成错误值，再检查从 host 恢复的非零 K/V 数据。MOVE 后 64 个 token 槽全部可用；COPY 后保留 4 个有效缓存 token，可用槽为 60。结束时请求与 host 引用计数归零，搬运队列和控制任务均已退出。该微型缓存 fixture 的峰值 tensor 分配为 9,728 字节；它不代表服务内存占用或吞吐结果。

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  test/manual/entrypoints/programmable_kv_gpu_smoke.py

PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  test/manual/entrypoints/programmable_kv_hicache_smoke.py
```

`/tmp/sglang-kv-gpu-hicache-poison-smoke.log` 保存了恢复检查结果。微型 fixture 的 16 字节行宽会触发可选 JIT kernel 的不支持提示；实际运行的是显式选择的 direct 后端。

## 服务级 smoke test

本地 OPT-125m 配置配合 `--load-format dummy` 完成了真实 GPU forward、HTTP、typed IPC 与缓存执行。测试请求先通过附带指令设置优先级，再独立 MOVE 整个已提交前缀，检查目标 GPU 副本消失且 host 副本可用，随后 COPY 回 GPU。重复生成命中 3 个缓存 token，输出 token 与原请求一致。恢复副本的优先级为零；flush 更换 epoch，旧操作引用查询返回 HTTP 409。

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
  .venv/bin/python test/manual/entrypoints/programmable_kv_server_smoke.py \
  --model-path LOCAL_OPT_CONFIG --load-format dummy --hicache
```

脚本为每次运行生成独立的 owner 身份和命名空间，保存 `server.log`、`result.json`，只清理本次启动的进程。已通过的结果位于 `/tmp/sglang-kv-opt-hicache-final-smoke/`、`/tmp/sglang-kv-opt-hicache-unique-smoke/` 和 `/tmp/sglang-kv-opt-hicache-rebind-smoke/`。末次运行还检查了 flush 后仍由调度器批准 GPU 预取容量。随机权重的结果用于接口与执行路径验证，不能据此评价模型质量。

tiny Llama 的本地 checkpoint 路径仍未通过启动验证：环境中的 CUTLASS DSL 4.8.0 分发未提供所需的 `cutlass.cute` 模块，现有 DualGemm 在缓存初始化前报错。初始 PyTorch/kernel ABI 和 torchvision 不匹配的失败日志也已保留。未绕过版本检查或替换模型 kernel 来生成通过结果。

## 后续实现范围

真实 L3/linker 搬运、混合状态与多 rank 发布、可组合策略与请求 admission 协作仍需开发。当前 capabilities 会明确报告已支持的方向、粒度和限制。此处没有 SGLang 吞吐提升或训练权重输出质量结论。

## 参数更新与版本发布

参数修改前会隔离旧 epoch 和实际 lookup namespace；多步更新期间，新推理直接返回 503，KV 提交返回 busy。参数工作同步完成后才执行所需的 cache flush，并发布版本对应的新命名空间。`flush_cache=False` 时保留的旧物理副本也无法命中新版本查找。

23 项新增 CPU 测试修改实际 tensor，覆盖 disk/IPC、begin/tensor/distributed/end、Python Engine 真实序列化及 typed IPC、版本声明、回滚与部分失败、同步失败、最终 flush 重试、错误线程和内存所有权。相关 CPU 检查合计 143 项通过，2 项跳过，输出在 `/tmp/sglang-kv-weight-cpu-proof.log`。

GPU 1 上的 OPT smoke test 实际修改了 attention LayerNorm bias 参数：修改前后 checksum 不同，namespace 和 epoch 更新；同一输入在新版本首次生成命中 0 个缓存 token，随后重复生成命中 3 个，输出 token 一致。修改的是随机权重 fixture 中的有限偏置参数；这不构成训练质量或完整训练更新结论。结果保存在 `/tmp/sglang-kv-opt-weights-smoke/result.json`。

现有 `test_weight_checker.py` 按其单卡配置运行得到 63 项通过、1 项失败；失败的 FP8 路径缺少 `deep_gemm`。在隐藏 GPU 环境中误运行该套件的失败记录保留在 `/tmp/sglang-kv-weight-checker-cpu-env-failed.log`，单卡输出在 `/tmp/sglang-kv-weight-checker-gpu-proof.log`。未修改既有测试以掩盖依赖问题。

## 文件存储写入

实现 COPY GPU/Host→Storage、MOVE GPU→Storage，限单 owner、FULL、持久 Host
与 `file` 后端的单个完整前缀。复用实际 HiCache DMA、文件写入线程及 ACK 队列。
完整 hash 链与 Host indices 在提交时固定，普通 radix 分裂不会改变完成覆盖范围。

写入线程会在 ACK 前核对连续前缀，避免文件缓存淘汰已写批次后仍报告完整成功。
短写与核对失败保留 GPU 源副本，并返回实际确认的覆盖范围和失败原因。MOVE 仅在
完整目标就绪后尝试正常回收；其他请求仍持有源引用时返回 PARTIAL/source_busy。
取消须等待实际 worker 退出；一个调用方取消不会撤销其他调用方共享的写入。

新增 CPU 用例 10 项通过；加既有 cache/HiCache 生命周期用例共 39 项、18 个子用例
通过。Serving、命名空间、参数更新与 Host 回归 61 项通过。既有文件路径和 LRU
测试 37 项通过。4 项既有 backfill 测试依赖自动选择 GPU：隐藏 GPU 的尝试在
fixture 初始化失败，改用 1 号卡后全部通过；两次日志均保留。

1 号卡 A100 的 FP16 小型验证通过：真实非零 K/V 经 GPU→Host→文件写入后逐字节
读回一致，完整写入后才释放 GPU 副本；污染空闲 GPU 行再从 Host 恢复，验证数值、
引用与 allocator 余额。微型 fixture 的 GPU 分配峰值为 9,728 字节，使用 direct
I/O。可选 JIT 对该 fixture 的 16 字节行给出不支持提示，未据此宣称 JIT 通过。

日志：`/tmp/sglang-kv-storage-cpu-proof.log`、
`/tmp/sglang-kv-storage-regressions.log`、
`/tmp/sglang-kv-storage-backend-proof.log`、
`/tmp/sglang-kv-storage-backfill-gpu-proof.log`、
`/tmp/sglang-kv-gpu-file-storage-final-smoke.log`。

Storage 读取、跨分支冷存储操作、其他后端与策略/调度协作尚待实现，能力发现接口
保留对应限制。

## 文件读取与服务往返

实现 COPY Storage→Host/GPU，可在 GPU、Host 和对应 radix 节点都已淘汰后，根据
实际命名空间和完整 hash 链恢复。显式读取使用独立的最小长度，普通 prefetch 阈值
及提前结束策略保持原有行为。完整 Host 缺失范围与 GPU 预算须在发起查询前通过，
容量等待不持有引用。Host 发布时取得内部引用，GPU 阶段等待真实 H2D fence。

新增读取 CPU 测试 11 项通过，全部使用编码后的实际命名空间文件键。覆盖只为
缺失后缀计费、GPU 预算不足时无 I/O、部分读取、查询取消、磁盘/DMA 取消、共享
读取与参数命名空间隔离。写入测试补了组合操作失败后取消的场景，共 11 项通过。
两组最终共 22 项通过，日志为 `/tmp/sglang-kv-storage-read-final-cpu-proof.log`。
既有生命周期、命名空间和参数更新回归 43 项通过；既有 FULL_ps4 的真实 GPU L3
写入/读取测试 2 项通过。

1 号卡非零 KV 往返验证通过：淘汰 GPU 和 Host 数据后，污染空闲 GPU 行，再从文件
恢复并检查字节、引用及 allocator。早期探针发现 list 与 array 类型不一致，修正后
通过；失败日志仍保存在 `/tmp/sglang-kv-gpu-file-roundtrip-key-type-failed.log`。

1 号卡 OPT-125m dummy-weight 服务验证通过：生成前缀、独立 MOVE 到文件、清空物理
缓存并取得新 epoch、确认两级本地副本缺失、独立 COPY 回 GPU、再次生成命中
3 个输入 token 且输出 token 一致。日志与结果保存在
`/tmp/sglang-kv-opt-file-storage-roundtrip-smoke/`。该结果用于验证接口与缓存行为，
不代表模型质量或服务吞吐表现。内部 COPY 引用在操作完成后释放，等待未来请求
admission 的保留行为仍需策略层实现。

## 可配置的 retention 策略

默认策略列表为空。启用 retention 后，importance/resume 信号通过只读接口生成有限
优先级提议，组合器处理显式意图与依赖，控制核心登记子操作并执行。父信号返回
解释结果和子操作引用，具体物理结果可单独查询。未消费的 tool_wait 不读取历史、
不构造观察数据。任务结束仅清理对应 incarnation 的策略声明，保留共享任务和
显式声明。策略不会取得 Pin 或改变 admission。

新增 CPU 验证 16 项通过，覆盖真实缓存优先级、重放和 TTL、多策略组合、容量拒绝、
任务清理、缓存重置、启动配置，以及暂不可执行请求中的清理行为。既有相关回归
170 项、18 个子用例通过，Host/文件搬运回归 36 项通过。

1 号卡服务验证通过：策略值 200 与显式值 90 冲突时，有效值为 90；经文件保存、
物理缓存清空和恢复后，新副本接受策略值 55。两个策略子操作均可查询且完成，
后续生成命中 3 个输入 token、输出 token 一致。使用 OPT-125m 配置与 dummy weights，
不据此评价模型质量或吞吐。

日志为 `/tmp/sglang-kv-retention-cpu-proof.log`、
`/tmp/sglang-kv-retention-regressions.log`、
`/tmp/sglang-kv-retention-movement-regressions.log`；服务结果在
`/tmp/sglang-kv-opt-retention-storage-smoke/`。
Tool-wait 成本判断、quota 调度指导和保留至实际 admission 的准备引用仍待实现。

## 配额、tool-wait 与实际 admission

已接通 retention、quota、tool_wait 三种可选策略，默认列表仍为空。配额使用软份额、
空闲容量借用、importance 与等待时间，调度器在原生前缀匹配后读取有界观察并排序。
实际分配继续由 `PrefillAdder` 使用当前匹配、批次预留及运行中请求增长进行检查。
观察缺失或截断不会被视为零需求。共享的可淘汰前缀在整体准备需求中计费一次。

Tool-wait 仅在已观察到引用压力、暂停预测长于配置阈值和搬运成本估计时提出保存。
默认参数为压力 0.7、最小暂停 0.5 秒、初始单程搬运估计 0.1 秒。该初值属于配置估计；
真实控制到就绪耗时用于后续修正，不代表硬件带宽测量。短暂停、未知暂停与低压力
场景不搬运。当前搬运策略限 FULL、单 owner、持久 Host，以及可选的 `file` 后端；
策略目标须为单个 token 分支和页对齐的完整前缀。

恢复会绑定明确的任务 incarnation 和实际请求。准备在 `PrefillAdder` 创建前启动，
并检查完整请求所需余量；发生分配后使用后续新调度轮次，避免沿用旧预算。COPY
完成后保留内部准备引用，等请求取得自己的缓存锁再交接。请求取消、任务结束、
过期、显式覆盖、重置，以及 decode 余量不足都会释放未消费的准备引用。已经完成
的 COPY 状态不能替代当前就绪检查。已有推理预取会先发布 Host 数据，策略不会再
为同一请求重复发起冷查询。

CPU 用例覆盖真实 Host/文件数据、异步取消和 worker 排空、准备引用交接、重复与
共享请求、整请求余量、ACK 后清理、显式优先级、旧预测被真实恢复替换、历史快照、
观察截断及启动配置。既有原生调度、预算、参数更新与缓存测试继续保留。

1 号卡的组合服务验证通过：策略保存后确认原 GPU 副本缺失，恢复请求复用 5 个前缀
token，输出与不同 cache salt 的重新计算对照一致；统计记录 1 次准备消费和 5 个
消费前缀 token。服务使用本地 OPT-125m 配置、FP16 与 dummy weights。底层非零
17/18 K/V 文件往返和引用交接另有真实 CUDA 字节验证。上述证据用于功能与正确性，
未据此宣称模型质量、吞吐提升或训练流程完成。

组合服务日志与结果：`/tmp/sglang-kv-opt-tool-coordination-smoke/`。
最终复验：`/tmp/sglang-kv-opt-tool-final-smoke/`。
CPU 验收：`/tmp/sglang-kv-final-cpu-acceptance.log`。
策略与生命周期专项日志：`/tmp/sglang-kv-tool-acceptance-cpu-proof.log`、
`/tmp/sglang-kv-tool-lifecycle-cpu-proof.log`。

TP/PP/CP/DP 多 owner、直接 linker、混合状态搬运、通用范围/descriptor 搬运、其他
存储后端及性能实验仍在此实现范围之外，接口会返回对应限制。

最终合并 CPU 验收：355 项、79 个子用例全部通过，用时 28.27 秒。
最终 1 号卡组合策略服务复验通过：准备复用 5 个 token，输出与重新计算对照一致。
格式、静态检查和 diff 空白检查通过。当前可执行范围的接口、生命周期、真实搬运、
参数隔离、可组合策略和 admission 引用交接均有对应代码及验证记录。
