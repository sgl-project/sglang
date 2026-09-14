# Snapshot v2 同步与验证（2026-09-14）

## 来源与交付范围

- 已重新 fetch `skajre/sglang` 的 `codex/kv-placement-snapshot`，远端仍为
  `ae18033b22a3dc51c5c9da53048b6d9149b2028f`；通过 ancestry 检查确认交付分支包含它。
- 已重新 fetch `yangbodong22011/agent-tasks` main，确认
  `kv-placement-index-architecture-comparison/ARCHITECTURE.md` 与本地设计一致。
- 复用 `codex/kv-placement-rebuild-20260911` 的已提交实现（截至 `e4ae4f62ea`），
  在 `codex/kv-placement-snapshot-v2-20260914` 上继续修正并重新验证。
  保留 Worker、Bridge、Indexer 和 Router 的完整 v2 恢复链路。
- 原 `/root/sglang` 的暂存内容和未完成 cherry-pick 保持原状；交付工作树为
  `/root/sglang-kv-placement-rebuild`。

## 本次发现与修正

1. 显式空 component 列表原先被编码成 legacy mask 0，可能误报整块命中。
   Python snapshot mirror 现在删除对应 tier，Rust v2 live decoder 生成 revoke；
   其他 tier 和其他 Worker 对同一 hash 的持有关系保留。
2. Recoverable decoder 原先可能忽略未知事件或 component label，然后继续推进
   sequence。现在拒绝整个 batch，让 Bridge 走恢复路径；legacy API 的容错语义保留。
3. `/server_info` 原先对没有 snapshot endpoint 的 live-only Worker 也声明 v2。
   现在仅在存在可路由的 snapshot endpoint 时发布恢复能力和相关身份字段，保留
   live-only descriptor 的兼容性。

前两项先新增失败测试，再修正实现并确认通过。第三项由现有 server-info 回归测试
发现，同时扩充了开启/关闭 snapshot 时的能力声明断言。新增真实进程测试验证
空组件更新在 live、冷启动 snapshot、Indexer 重启后的查询结果一致。

首轮进程测试还暴露了两处测试问题：新增 component 用例没有声明 WorkerCacheSpec，
按设计被前缀扫描器拒绝，现已补全测试 Worker 的规格；既有 overflow 用例可能在
Bridge 尚未撤销旧 READY 状态时就读取日志，现改为暂停期间等待 overflow，并校验
两个副本均追上最后一个事件的 watermark。双 GPU 推理用例在首轮已通过。
进一步复跑确认：等待中的 RPC 可能先超时，使原有主循环日志只出现 transport timeout。
Bridge 现在在接收队列实际溢出的位置立即记录日志，便于诊断并让测试精确同步故障点。

## 架构要求与验证入口

| 要求 | 实现与证据 |
|---|---|
| v2 header 的 namespace/model/stream/epoch/schema/page/barrier/cache spec | Python `KVSnapshotHeaderV2`、Bridge `WireHeader` 校验、`test_kv_snapshot_v2.py` |
| hash/parent/size/tier/component 与按 namespace/hash/tier 的镜像 | `KVSnapshotBlockV2`、`_snapshot_blocks_v2`、tier/component/parent/empty-component 单测 |
| 原始 Worker epoch/sequence 与 replay 完整性 | `replay-v2`、`decode_recoverable_batch`、`replica_recovery.rs` 的 gap/duplicate/epoch/session 测试 |
| staging 不可查询、原子发布、READY coverage 与持有关系隔离 | `ReplicaService`、`replica_recovery.rs` |
| 一个 Bridge 接全部 Worker streams，并独立恢复一个 Indexer | `replica_bridge.rs`、多 Worker/多副本进程测试 |
| Router failover、全部副本失效时负载降级、扩缩容 | `fleet.rs`、`replica_control.rs`、Router 进程测试 |
| 原生 FULL/SWA/MAMBA 事件与真实推理 | `TestUnifiedRadixCacheKVEvents`、双 GPU 进程测试 |
| v1 兼容和实际服务发现 | 原有 KV events/server-info 测试、新增 v2 descriptor 断言 |

## 复现命令与结果

工作目录为交付工作树根目录：

```bash
cargo fmt --manifest-path experimental/sgl-router/Cargo.toml --all --check
cargo test --manifest-path experimental/sgl-router/Cargo.toml --workspace --tests --quiet
cargo build --manifest-path experimental/sgl-router/Cargo.toml --bins --quiet
```

Rust：846 passed，0 failed；格式检查和二进制构建通过。

```bash
PYTHONPATH=python:test/registered/unit/mem_cache .venv/bin/python -m pytest \
  test/registered/unit/disaggregation/test_kv_snapshot_v2.py \
  test/registered/unit/disaggregation/test_kv_events.py \
  test/registered/unit/disaggregation/test_load_reporter.py \
  test/registered/unit/managers/test_loadstat_wire.py \
  test/registered/unit/managers/test_scheduler_on_idle_load.py \
  test/registered/unit/entrypoints/test_server_info.py \
  test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py::TestUnifiedRadixCacheKVEvents \
  -q --junitxml=/tmp/kv-snapshot-v2-python-20260914.xml
```

Python：104 passed，55 subtests passed；14.90 秒。环境有既有 pytest 配置和
依赖弃用警告。一次早期选择表达式误选了更大的 Unified Cache 集合，已主动中断；
上述完整结果使用明确的 `TestUnifiedRadixCacheKVEvents` 节点。

```bash
KV_REPLICA_GPU_TESTS=1 PYTHONPATH=python .venv/bin/python -m pytest \
  experimental/sgl-router/sgl-kv-indexer/tests/test_replica_processes.py \
  -q --junitxml=/tmp/kv-snapshot-v2-process-20260914.xml
```

进程测试：5 passed，0 failed，0 skipped；70.65 秒，包含两个真实 CUDA Worker 的
Qwen2 推理、单副本/全部副本故障、重启恢复、扩缩容、尾事件回放、慢副本隔离和
新增空组件恢复一致性用例。最终日志位于 `/tmp/pytest-of-root/pytest-14/`。

Python 语法/未定义名 Ruff 检查（`E9,F63,F7,F82`）、`git diff --check` 均通过。
这些结果来自本次工作树实测，不引用 September 11 的测试结果作为本次通过证据。

## 验证边界

本次验证功能、协议兼容和故障恢复，未执行性能基准或完整可选缓存后端矩阵。
真实 GPU 测试使用小型随机初始化 Qwen2；既有 GPU 服务保持运行。
原实现记录中的 C128/部分 alternative core、SSD 候选匹配、TLS/租户鉴权等限制仍适用，
详见 [IMPLEMENTATION.md](IMPLEMENTATION.md) 和 [README.md](README.md)。
