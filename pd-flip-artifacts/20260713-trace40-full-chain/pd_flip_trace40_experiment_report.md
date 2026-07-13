# 多节点 PD 身份切换 40 请求实验报告

实验日期：2026-07-13（Asia/Shanghai）  
主实验：`20260713T082202Z-trace40-full-chain`

## 技术摘要

本次实验确认了链路的核心迁移路径可以真实运行：SLO monitor 触发 D→P 决策后，node2 停止接收新请求，按前 N 个请求选择 50% 的运行批次，将初始 KV 迁移到 node3；随后完成 delta KV、target commit、source finish 和 target activate。node3 接管 2 个请求后继续 decode，并把输出 relay 回 node2。最终 40/40 请求完成，0 请求错误。

完整身份切换仍未闭环。配置的 10 秒观察窗结束后，P 的 SLO 风险仍存在，controller 尝试把 node2 剩余 1 个运行请求放入第二个 migration session，但 source 返回 `conflicting migration session already exists`。因此 node2 保持 decode 身份，最终角色仍为 P,D,D,D。此外，node3 在客户端 workload 完成后触发 pool memory leak invariant，scheduler SIGQUIT，容器以 137 退出（不是 OOM）；服务随后已人工重启恢复。结论应表述为“第一批迁移和目标续算成功；第二批 session rollover/最终 D→P 切换失败；目标节点存在迁移后资源释放泄漏”，不能表述为完整链路成功。

## 实验范围与配置

- 节点：node0=Prefill，node1/2/3=Decode；迁移源 node2，目标 node3。
- Trace：40 条请求，长短请求交错；每条请求带独立 TTFT/TPOT SLO。
- 触发实验把 TTFT SLO 覆盖为 30 ms，以稳定触发 P 风险；TPOT 使用 trace 中的请求级目标。
- 触发阈值：P 达成率低于 0.99 且 D 达成率高于 0.99。
- 首批迁移比例：0.5，选择 running batch 的前 N 个请求；目标容量不足时按设计持续减半。
- 源节点迁移期间停止接收新请求。
- 观察窗：10 秒。

## 关键结果

| 项目 | 结果 |
|---|---:|
| 请求完成 | 40/40 |
| 请求错误 | 0 |
| TTFT 达标 | 6/40（15%） |
| TPOT 平均值达标 | 40/40（100%） |
| TPOT P95 达标 | 40/40（100%） |
| TPOT interval 达标 | 91,735/91,740（99.99455%） |
| monitor 决策时 P 达标 | 3/10 |
| monitor 决策时 D 达标 | 4,755/4,755 |
| 第一批迁移请求 | 2 |
| 第一批最终 owner | node3/target |
| 第二批剩余请求 | 1 |
| 最终身份切换 | 未完成；node2 保持 decode |
| 目标节点收尾稳定性 | 失败；pool leak invariant 后退出 137 |

四轮对照结果见 [experiment_index.csv](experiment_index.csv)。无触发对照轮 40/40 请求的 TTFT、TPOT 均达标；30 ms 触发轮稳定制造了 P 风险，同时 D 的 TPOT 基本保持 100%。

## 第一批迁移链路与耗时

| 阶段 | 耗时 |
|---|---:|
| Router drain source | 0.370 ms |
| Pause source admission | 3.470 ms |
| Source start | 12.327 ms |
| Target prepare | 21.871 ms |
| Target prepared → initial KV first progress | 40.427 ms |
| Initial KV first progress → complete | 60.783 ms |
| Target prepared → initial KV complete | 101.210 ms |
| Delta start 第一次（quiesce pending） | 5.601 ms |
| Delta start 第二次 | 4.706 ms |
| Target delta prepare | 11.252 ms |
| Target commit | 11.294 ms |
| Source finish | 4.583 ms |
| Target activate | 7.013 ms |

逐请求精确进程计时：

| RID（缩写） | P/H/C0/C1 tokens | Initial bytes / 时间 | Delta bytes / 时间 | 结果 |
|---|---:|---:|---:|---|
| fc1d58… | 1974/1974/2283/2283 | 31,260,672 / 42.901 ms | 14,303,232 / 50.841 ms | target |
| 7f86c6… | 189/5/436/436 | 49,250,304 / 95.015 ms | 14,303,232 / 47.530 ms | target |

两条请求的 `stitch_mode=prefix_stitch`、`final_owner=target`、`fallback_attempted=false`。第一条 H=P，远端 prefix 覆盖完整 prefill；第二条仅命中 5 个 prefix token，剩余缺口由 source decode 侧 KV 覆盖。该轮没有进入“拼接失败后全量 source fallback”，因为双源拼接本身成功。

## 目标节点继续计算证据

- `controller_actions.csv` 中 target commit、source finish、target activate 均为 success。
- source finish 返回 `source_released`，pending=0、failed=0，两条请求的 `final_owner=target`。
- node3 在 activate 后 running requests 从 4 增至 6，随后持续 decode。
- node2 日志在本轮时间窗内持续收到来自 node3 的 `/pd_flip/migration/output/relay` HTTP 200。
- workload 最终 40/40 完成、0 error，说明迁移请求的客户端输出没有中断。
- 但 node3 在 08:23:00、仍显示 1 个运行请求和 4,085 tokens 时触发 `pool memory leak detected`：total=598,921、available=499,532、evictable=101,370、protected=0、session_held=0、uncached=0。scheduler 随后 SIGQUIT，容器最终退出 137。这说明“目标续算和客户端完成”成立，但“迁移后资源回收与长期稳定”不成立。

## 观察窗

观察窗配置为 10.0 秒。它发生在第一批 activate 之后、第二批剩余请求迁移之前，期间 node2 继续处理未迁移请求并重新采样 SLO。观察结束时 fresh window 为 P=4/29、D=66,294/66,309，因此 P 风险仍存在，进入第二批迁移。

`migration_stage_durations.csv` 中 `kv_transfer_complete → cleanup_router_undrain = 11.7127 s` 不是纯观察时间；它混入了 delta、commit/activate、10 秒观察、第二批失败及清理，不能直接当作观察窗耗时。

## 未闭环原因

第二批使用新 session id `...-final` 调用 node2 source start，在 7.964 ms 后返回 `conflicting migration session already exists`。随后 controller 用新 id 清理，而 worker 仍持有第一批 id，因此 target/source abort 分别返回 session-id mismatch；controller 最终恢复 admission、undrain router，并保持 node2 为 decode。

当前证据只证明 rollover predicate 中至少一个内部 entry 安全条件未满足；公开 status 没有暴露具体 blocker。第一批的 terminal 状态、计数和 owner 均正常，因此下一步应给 `_pd_flip_can_rollover_session` 增加结构化 blocker 输出，并在迁移请求的 relay 生命周期结束前后分别验证 rollover。不要把报告生成器的“未观察到 commit”当作根因。

另一个独立阻断项是 node3 的 pool 资源回收泄漏。日志显示 invariant 中没有 protected、session-held 或 uncached tokens，但 `available + evictable` 与 total 存在差额；需要按迁移 RID 审计 target adopt、relay 完成、request finish、KV release 和 radix/HiCache ownership 的配对关系。

## 报告脚本局限

`report/migration_link_summary.json` 把本轮标成 `kv_transfer_complete_without_commit_observed`，这是提取器误分类：它只把通用采样时间线抽到 initial KV complete，再用 controller 的最终错误覆盖整个轮次；但原始 controller actions 明确记录 commit、finish、activate 成功。因此：

- 第一批是否成功，以 `controller_actions.csv`、`controller.log` 和 node2/node3 日志为准。
- `request_migration_join_count=0` 只表示现有 join 提取逻辑没关联上，不表示没有请求迁移。
- 后续汇总应按 session id 分段，并纳入 delta/commit/activate/relay 阶段。

## 全流程耗时

- Runner 总耗时：126.784 秒。
- Controller：13.259 秒。
- Workload：48.515 秒。
- Workload 完成后保留测量约 10.152 秒。
- 累积日志收集与 raw 校验：56.905 秒。
- 压缩归档：1.170 秒。

完整明细见 [final_run_timing_summary.csv](final_run_timing_summary.csv)。

## 数据质量与限制

- 四轮均 40/40 完成、0 请求错误，raw validator 通过。
- TTFT 30 ms 是为了触发状态机的实验覆盖值，不代表生产目标。
- 服务日志是累积日志，引用 relay 时只使用本轮 08:22:15 之后的时间窗，不能把整份日志中的 relay 总数归因给本轮。
- 现有汇总器缺少 session-aware 关联，报告已明确纠正其第一批 outcome 误判。
- 本轮没有触发容量持续减半，也没有触发双源拼接失败后的 source-full fallback；这两条分支仍需单独故障注入验证。
- node3 的退出发生在 workload 客户端完成之后；所以请求级成功率不能替代服务稳定性判定。

## 下一步

1. 为 rollover predicate 输出每个 entry 的结构化 blocker，并补真实第一 session → 第二 session 回归测试。
2. 给迁移 target 的 KV/pool ownership 增加逐 RID acquire/release 账本，修复 workload 结束后的 pool leak invariant。
3. 修正汇总器：按 session id 切分，记录 delta、commit、activate、relay、进程退出和最终 role。
4. 清理 session 后重跑同一 trace，验收 node2 最终变为 prefill，且四个 worker 在 workload 后稳定存活。
5. 再做两轮故障注入：目标容量不足持续减半；远端 prefix restore 失败后全量 source-decode fallback。

## 原始证据入口

- 主 raw/log 压缩包：`20260713T082202Z-trace40-full-chain.tar.gz`
- 压缩包校验：`SHA256SUMS`（主包 SHA256=`a5489673445d3f28d1eccdf5adfcf9a9de0bf14f4b445801ecad48a839431007`）
- Controller 动作：`20260713T082202Z-trace40-full-chain/report/controller_actions.csv`
- 状态轨迹：`20260713T082202Z-trace40-full-chain/report/controller_state_trace.csv`
- 逐请求指标：`20260713T082202Z-trace40-full-chain/workload/state_machine/request_metrics.jsonl`
- 四节点日志：`20260713T082202Z-trace40-full-chain/logs/node0.log` 至 `node3.log`
- Router/Mooncake 日志：`20260713T082202Z-trace40-full-chain/logs/router.log`、`mooncake-store.log`
