# 可恢复 KV Placement 副本：重新实现记录

## 基准与范围

- 任务：`/root/agent-tasks/kv-placement-index-architecture-comparison/TASK.md`。
- 架构：`yangbodong22011/agent-tasks` main `5f3ab8a84c109ffe7fa078a2c32bf1df3898da58` 中的 `kv-placement-index-architecture-comparison/ARCHITECTURE.md`，通过 Git 重新获取。
- 源码：`skajre/sglang` 的 `codex/kv-placement-snapshot`，`ae18033b22a3dc51c5c9da53048b6d9149b2028f`。
- 工作树：`/root/sglang-kv-placement-rebuild`；分支：`codex/kv-placement-rebuild-20260911`。
- 原 `/root/sglang` 存在暂存修改与合并冲突，原样保留；未读取或复用先前实现及报告。

## 任务拆解与验收

1. Worker 协议：版本化 Snapshot v2，包含 namespace/model/stream/hash schema/cache spec、tier/component；保留 v1；Replay 携带原始 epoch/sequence，镜像按 namespace/hash/tier 隔离。
2. Indexer 恢复：按 stream 的恢复会话、epoch/sequence fencing、staging、原子替换、lease 失效、coverage；保留 flat placement 与反向 holdings。
3. Bridge：单进程配置多个 Worker，发现 DP streams；一个 Bridge 仅连接一个 Indexer；先订阅再 snapshot/barrier/catch-up；有界队列、短 gap replay、失败重新 snapshot；支持成员变更、重启与启动抖动。
4. Router：随机选择副本，完整 coverage 校验，失败/部分恢复尝试其他副本，全部不可用时负载降级；动态副本发现；不保存 placement、不订阅 KV events。
5. Load Reporter / Monitor：Worker 到 Router 的 gRPC 报告、HTTP 注册与续租；统一 publisher epoch；按 freshness 与当前 generation 使用负载。
6. 测试：协议与状态机测试，真实本地进程（2 Worker、1 Router、2 Bridge+Indexer）正常调度、故障、Indexer 重启恢复、扩缩容；增加 gap、溢出、Worker 重启、共享 hash 隔离及稳定性覆盖。模型推理与模拟协议测试分别注明。
7. 交付：按功能拆分 commit，统一记录命令、结果、限制与复现方式，上传 GitHub 并核实远端提交。

## 当前事实与决策

- 基线 Worker 提供 v1 Snapshot、barrier 与有界 replay；snapshot 镜像只按 hash 索引，无法保留多 tier。
- 基线 Indexer apply 不检查 sequence，没有恢复/coverage API；Bridge 每进程只接一条流。
- 基线 Router 仅配置单 Indexer；负载通路是 ZMQ，没有目标架构要求的 `/v1/start_reporting` 与 gRPC LoadReport。
- 首先扩展 Worker 协议，随后实现 Indexer 恢复与 Bridge，最后接入 Router/负载与完整集成测试。

## 执行记录

- 2026-09-11：重新拉取任务架构及目标分支，创建独立工作树，完成需求与源码差异核对。
- Worker 第一阶段：新增 Snapshot v2、namespace/hash/tier 镜像、component 与 block size、带 epoch 和完整性边界的 Replay v2；v1 保持可用。Replay socket 发送超时避免慢消费者无限阻塞发布线程。
- 实测：`PYTHONPATH=python python -m pytest test/registered/unit/disaggregation/test_kv_snapshot_v2.py test/registered/unit/disaggregation/test_kv_events.py -q`，26 passed，22 subtests passed；存在基线 pytest 配置/torch 弃用警告。
- Indexer：新增独立 KVReplica 协议与完整 flat placement，stream owner/session、staging chunks、barrier commit、sequence fencing、replay 缺口位置反馈、coverage、lease 和 stream 删除。旧无条件写 API 不可修改可恢复副本状态。
- 实测：`cargo test -p sgl-kv-indexer`，80 library + 4 binary + 10 gRPC contract + 29 memory integration + 6 recovery tests 全部通过。
- Bridge：`KV_BRIDGE_CONFIG` 指向 JSON 配置，包含一个 Indexer endpoint 与多个 Worker URL；定时重读配置、发现 DP streams，按 stream 订阅；256 条/32 MiB live 缓冲与每副本 snapshot 并发限制，overflow 重新恢复。缺口优先 Replay v2，不完整则 snapshot。
- 实测：`PYTHONPATH=python .venv/bin/python -m pytest experimental/sgl-router/sgl-kv-indexer/tests/test_replica_processes.py -q`，1 passed（13.44s）；覆盖 2 production publishers + 2 Rust Bridge + 2 Rust Indexer、初始 snapshot、live 增删、kill/restart Indexer、第三副本加入、Worker epoch 更换与成员移除。此测试未启动 Router 或模型推理，不能代替最终 2 Worker/1 Router 验收。
- Router：随机尝试副本、对 eligible streams 精确校验 coverage/epoch/hash schema，同一 HTTP Worker 的多个 DP rank 取最小命中；总查询 deadline 在副本尝试间分配；支持逗号 endpoint 列表及 `@file` JSON 数组热更新；全部失败产生无缓存信号，保留推理调度。
- Load：嵌入 Router 的 gRPC Load Monitor，HTTP `/v1/start_reporting` 经 Worker 内部通道注册各 scheduler Reporter；同一 publisher epoch、独立 Router target lease、最新样本槽位、有界 freshness；Router 一次捕获 load + generations 后执行查询。添加查询成功、降级、失败尝试指标。
- 实测：`cargo test --workspace --lib --quiet`，82 Indexer + 563 Router 单元测试通过。
- 实测：单独运行 `test_router_random_failover_all_down_restart_and_membership`，1 passed（15.51s）。实际 Router/Bridge/Indexer 进程、生产 Reporter/Publisher、模拟推理响应：两副本随机使用、单副本 kill、全部 kill 时 fresh min-load、重启缓存恢复、新副本加入、旧副本移除、空列表及坏配置保留。
- 实测：Worker KV/Reporter/LoadStat/idle 路径回归，62 passed、41 subtests passed。Reporter 双 target 租约、旧测量不重发、generation 验证通过。
- GPU 环境：两张 A10 的大部分显存被既有 PID 83601 的 TP=2 Qwen2.5-7B 服务占用。未终止该服务，已询问是否可以暂停；并继续准备小型真实 Worker 验证，避免模拟响应代替推理验收。
- 原生 Worker 集成：Scheduler 向 Publisher 提供真实 model/page size/bigram；Unified Cache 发出 FULL/SWA/MAMBA 每页驻留状态，覆盖 auxiliary eviction、load-back、host 写入与 Mamba 边界。Reporter 接入 Scheduler 的实际负载采样；补全 grpcio/Protobuf 运行时依赖和注册错误响应测试。
- 实测：相关 Python 单元测试 63 passed、41 subtests passed（14.26s）；`test_unified_radix_cache_unittest.py -k KVEvents`，9 passed、2 subtests passed（7.12s），包括真实 GPU Unified Cache 的 SWA/Mamba component 更新。
- 扩大回归时运行了整个 Unified Cache 文件（约 2573 个参数化用例），进程在约 17% 时 exit 139，没有完整结果。尚未定位，不能声称整个缓存后端矩阵通过；随后本次修改直接涉及的 KVEvents 选择集独立复跑通过。保留为明确验证限制，不把失败隐去。
- 真实 GPU 集成实测：`KV_REPLICA_GPU_TESTS=1 PYTHONPATH=python .venv/bin/python -m pytest experimental/sgl-router/sgl-kv-indexer/tests/test_replica_processes.py -k real_two_gpu -q`，1 passed（41.66s）。两个真实 SGLang CUDA Worker 使用小型随机初始化 Qwen2，预热缓存早于 Indexer 启动；两副本都从 snapshot 找到原有 prefix，真实 Router 完成正常、单副本故障、全部故障、重启、扩缩容后的推理。不是质量或性能测试。原有 7B 服务未停止。
- GPU 首轮测试修正：5 秒 HTTP 预热超时不足以等待首次 kernel 编译，单独将推理超时设为 120 秒；Router tokenizer 参数须指向 `tokenizer.json` 文件而不是目录。最终副本 RPC 仍是 1 秒断言超时，生产 Router query deadline 不变。
- GPU Router 最终指标：complete queries=16、fallback queries=4、failed attempts=10；请求日志验证故障期间 HTTP 200，恢复后使用新副本。原始日志位于 `/tmp/pytest-of-root/pytest-8/test_real_two_gpu_workers_rout0/`，XML 摘要 `/tmp/kv-rebuild-gpu.xml`。
- 扩展进程测试：3 passed、1 GPU-opt-in skipped（32.02s）；覆盖尾事件丢失且没有后续 PUB 时的 Replay 修复、暂停单 Indexer 触发本地 Bridge overflow/恢复、另一副本持续更新、Worker 热加入、第二 Router 独立注册并共享副本。XML 摘要 `/tmp/kv-rebuild-process.xml`。
- 最终完整进程复跑：`KV_REPLICA_GPU_TESTS=1 PYTHONPATH=python .venv/bin/python -m pytest experimental/sgl-router/sgl-kv-indexer/tests/test_replica_processes.py -q --junitxml=/tmp/kv-rebuild-process-all.xml`，4 passed、0 skipped（70.06s）。GPU 指标 complete=16、fallback=4、failed attempts=11；最终日志 `/tmp/pytest-of-root/pytest-10/test_real_two_gpu_workers_rout0/`。失败尝试数随随机副本选择变化，不作为固定数断言。

## 验收证据

| TASK / 架构要求 | 实现入口 | 验证证据 |
|---|---|---|
| 多 Indexer 随机选择 | `src/fleet.rs`、Router `replica_control.rs` | 30 次 cache route 的日志包含两个副本地址；副本失败改选其余副本 |
| 每 Bridge 配多个 Worker、每副本全量恢复 | `src/replica_bridge.rs`、`KV_BRIDGE_CONFIG` | 两个既有 Worker 缓存早于副本启动；每副本 snapshot 后拥有相同 coverage/prefix |
| Router 横向扩展 | 每 Router 独立 Load Monitor / target lease | 进程测试新增第二 Router，双方 fresh cache routing 成功；移除第二 Router 不影响第一 Router |
| Worker 横向扩展 | Worker discovery + hot Worker list + Router registry | 第三 Worker 预存缓存，两个既有 pair 热发现并恢复；删除成员后该副本不再声明完整 coverage |
| Bridge+Indexer 横向扩展 | 全量 soft-state 副本，Router `@file` 发现 | 启动第三 pair、切换仅第三 endpoint、停止旧 pair，推理继续且新副本有缓存 |
| 本地 2 Worker / 1 Router / 2 Indexer | `tests/test_replica_processes.py` GPU opt-in | 两张 A10 上真实 CUDA Scheduler、KV cache、Publisher、Reporter、Router、Bridge、Indexer 全链路；正常与故障阶段推理成功 |
| Indexer 挂掉时的调度 | random retry + deadline + fresh-load fallback | 单副本 kill 保持 cache affinity；全部 kill 的确定性负载测试选择 idle Worker；真实推理测试全部故障阶段仍成功 |
| Indexer 重启恢复原有缓存 | hidden staging + barrier + Replay + atomic replace | kill/restart 后从仍运行 Worker 恢复，非仅等待新请求重新产生事件 |
| Worker epoch / event fencing | Worker v2、Indexer session/sequence 状态机 | 旧 epoch、重复、乱序、缺口、冲突 session、错误 barrier/offset/count 被隔离；Worker 重启旧 cache 不残留 |
| Ready coverage，不把 not-ready 当 miss | `ReplicaPrefixResponse.coverage` | staging 不可查询、缺失 rank 不完整、同步后无命中为 complete；Router 拒绝 stale/重复/缺失 coverage |
| tier/component v2 | publisher mirror、native event recorder、prefix scanner | namespace/hash/tier 隔离，FULL/SWA/MAMBA 与 parent/size 保留；原生 SWA eviction/Mamba boundary GPU 断言 |
| 同 hash 持有关系隔离 | flat map + per-stream reverse holdings | worker/rank/namespace/tier remove 隔离；过期 GC 不影响活跃 owner；未知 spec 不替换可见旧状态 |
| bounded buffer / replay / 慢副本隔离 | Bridge bounded queue、Worker bounded replay | 丢失末尾 PUB 无后续事件仍恢复且不取新 snapshot；暂停 Indexer 触发本地 overflow，另一副本进度不受阻，恢复后重新完整 |
| 新鲜负载与 placement generation 一致 | `load_reporter.py`、Router `replica_control.rs` | 同 publisher epoch；多目标独立 lease、序列去重、旧 measurement 不刷新、所有 DP rank 必须新鲜 |
| Router 不保存 placement / 不订阅 KV | `main.rs`、route context | indexer 模式不启动 KVEventManager、不向本地 HashTree 填数据；仅保留既有 policy 工厂所需的空结构 |
| 版本与可观测性 | v1/v2、独立 legacy service、metrics | 旧协议回归通过；不支持的 schema fail-closed；complete/fallback/failed-attempt counters 与 READY/recovery 日志 |

最终 Rust 回归命令（工作目录 `experimental/sgl-router`）：

```bash
cargo fmt --all --check
cargo test --workspace --tests --quiet
```

结果：844 passed（82 Indexer unit + 4 server + 10 gRPC contract + 29 memory +
8 replica recovery + 563 Router unit + 4 Router binary + 58 component + 86 proxy），
0 failed。Python 相关选择集最终 63 + 9 passed，另有 41 + 2 subtests passed；
语法/未定义名 Ruff 检查和 `git diff --check` 通过。

复现 Python 相关选择集：

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  test/registered/unit/managers/test_loadstat_wire.py \
  test/registered/unit/managers/test_scheduler_on_idle_load.py \
  test/registered/unit/disaggregation/test_kv_events.py \
  test/registered/unit/disaggregation/test_load_reporter.py \
  test/registered/unit/disaggregation/test_kv_snapshot_v2.py -q
PYTHONPATH=python:test/registered/unit/mem_cache .venv/bin/python -m pytest \
  test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py -k KVEvents -q
```

## 交付与限制

- 工作树基于指定远端分支重新创建，不触碰原工作树的冲突/暂存修改。所有测试启动的子进程均已清理，既有 PID 83601 的服务未停止。
- 功能提交：`e669562dc8` 任务拆解；`58037b4aa1` Worker v2；`131a5e823b` Indexer；`cfdc483993` Bridge；`79c5e606a0` Router/Load；`ab604b356d` 原生 component 与 Scheduler 集成。额外测试和部署/验收文档各自独立提交。
- [README.md](README.md) 给出两副本部署、Worker 配置、每 Router 负载监听地址、热更新/扩缩容、协议边界和复现步骤。
- 当前支持架构要求的 FULL/SWA/MAMBA；不支持的 C128 / 缺少 component event 合同的 hybrid alternative core 会被 schema fencing 排除，不能误报命中。SSD placement 保留，但现有候选扫描只使用 HBM/DRAM。
- 本轮验证是功能与故障恢复，不是吞吐/长时间 soak/多机网络分区或大模型精度基准。GPU 使用随机初始化的两层 Qwen2，不能据此宣称 7B 生产负载的性能。
- 全量 Unified Cache 文件执行曾 exit 139，原因未定位；本次直接涉及的 9 项 KVEvents/GPU 路径已独立通过。该限制不影响上述已执行功能测试的结果，但不应据此宣称所有可选缓存后端均已验证。
- 控制面部署在可信网络；未新增 TLS/租户鉴权。内存随当前 holdings 增长；Snapshot staging 在恢复期间增加峰值内存。生成协议依赖 Protobuf 6，旧客户端只能使用隔离的 legacy API，不能混入 READY 副本数据。
- GitHub 交付分支：`yangbodong22011/sglang` 的 `codex/kv-placement-rebuild-20260911`；上传后通过 `git ls-remote` 核实最终提交。
