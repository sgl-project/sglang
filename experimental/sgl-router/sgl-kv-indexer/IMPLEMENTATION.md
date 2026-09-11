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

## 验收证据

尚未完成；后续逐项补充实测结果，不以构建成功代替架构验收。
