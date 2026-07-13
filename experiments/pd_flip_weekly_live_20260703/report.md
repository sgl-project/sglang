# PD Flip 四节点实验报告

日期：2026-07-03  
实验目录：`C:\Users\Tianci J\Desktop\sglang\experiments\pd_flip_weekly_live_20260703`

## 结论先行

这次四台机器都能连通，router、node0、node2、node3 能正常工作；node1 所在机器 `cloud-100` 的 8 张 H20 GPU 已被其他容器占用约 132-133 GiB/143 GiB，worker 启动时报 CUDA OOM。因此本轮真实在线实验实际运行在降级拓扑：`node0=prefill`，`node2/node3=decode`，`node1=plain/unhealthy`。

四个实验的结果：

| 实验 | 目标 | 结果 |
|---|---|---|
| 1 | 验证 monitor 能监控 SLO 达成率并指挥切换 | 通过。monitor 触发 D->P，两阶段 KV migration 后 commit，状态为 `safe -> preparing_kv_transfer -> flipping_role -> safe`。 |
| 2 | 验证 KV 能正常迁移，迁移后能继续 decode | 通过。观测到 `kv_transferred_reqs_observed=[0,0,0,0,2,2,2,2]`，失败数全为 0；长请求继续生成 2698 个 completion tokens，`finish_reason=stop`，无 abort 字符串。 |
| 3 | 验证 KV 完成后源 decode 节点等待一段时间确认 SLO 未恢复后才 flip | 未按原设想通过。当前代码是在迁移过程中持续检查 SLO，迁移完成后只再采样一次再 commit，没有显式 post-KV hold window。live artifact 里 `monitor.iterations=1`，状态直接从 preparing 进入 flipping。 |
| 4 | 构造 100 请求 trace，并比较无状态机/有状态机 SLO | trace 已构造并保存。降级 1P2D 拓扑下，无状态机 combined SLO=0.85；有状态机 combined SLO=0.83，且有 17 个 zero-chunk 请求。这个 A/B 是负向结果，不能作为收益证明，只能说明 monitor 能触发，但在 node1 缺失时切走一个 decode 会伤害吞吐。 |

## 对照流程图讲实现

### 1. Monitor

流程图里的 monitor 对应 `scripts/playground/disaggregation/pd_flip_monitor.py` 和 `scripts/playground/disaggregation/pd_flip_controller.py` 的 monitor loop。

普通 metrics 模式的实现是：

1. 每个采样周期，controller 先收集当前节点角色和 router 状态，再调用 `slo_monitor.collect_cluster(...)`。
2. monitor 对每个 worker 调 `/metrics` 和 `/v1/loads?include=all`。
3. `/metrics` 里读取两个 Prometheus histogram：
   - `sglang:time_to_first_token_seconds`，用于 prefill/TTFT SLO。
   - `sglang:inter_token_latency_seconds`，用于 decode/TPOT SLO。
4. monitor 不是自己在请求入口和出口打点，也不是保存完整 per-request 记录；它读的是 worker 已经暴露出来的累计 histogram bucket。
5. 对某个 SLO 阈值，例如 `TPOT_SLO_SECONDS=0.02`，monitor 找到第一个 `le >= 0.02` 的 bucket，把这个 bucket count 当成 good，把 `+Inf` bucket 当成 total。
6. 因为 Prometheus histogram 是累计计数，monitor 会和上一次采样做差分，得到这一个采样窗口新增的 good/total，再放进 `SLOWindow`。所以它不是全生命周期累计，而是滚动窗口统计。
7. prefill 节点的 role attainment 用 TTFT good/total；decode 节点的 role attainment 用 TPOT good/total。
8. 同时 `/v1/loads?include=all` 提供 `num_running_reqs`、`num_waiting_reqs`、`token_usage`，用于判断 queue/load 和选择迁移源。

对应源码：

- `pd_flip_monitor.py:22-23` 定义 TTFT/TPOT metric 名。
- `pd_flip_monitor.py:165-186` 采集 `/metrics` 和 `/v1/loads?include=all`，并做 histogram 差分。
- `pd_flip_monitor.py:195-215` 根据 SLO 阈值从 histogram bucket 算 good/total。
- `pd_flip_monitor.py:80-118` 维护滚动窗口。
- `pd_flip_controller.py:395-448` 是 monitor loop，采样后如果 prefill SLO 低于阈值就触发 D->P。

这和你的理解“大方向一致”，但细节上有两个修正：

1. 普通 monitor 不直接记录 request 进入时间和离开时间；它读 worker 侧已经汇总好的 histogram。
2. TPOT 达成率不是“先算每个请求平均 TPOT，再比较要求”，而是“每个 inter-token latency 观测是否落在 SLO bucket 内”，再算 good/total。

本次实验还启用了 request-level trace ledger 模式，它更接近你说的逐请求记录：

1. 请求 payload 可带 `custom_params.pd_flip_slo`，里面有 `ttft_seconds` 和 `tpot_seconds`。
2. trace probe 记录 `start_time`、`first_token_time`、`last_token_time`、`completion_tokens`、`good_tpot_intervals`、`total_tpot_intervals` 到 JSONL ledger。
3. `TraceSLOMonitor` 读取最近窗口内每个 request 的最新 ledger 记录。
4. TTFT good/total 是请求级别计数；TPOT good/total 是 token interval 级别计数。

对应远端源码：

- `pd_flip_trace_slo.py:52-70` 从 request payload 里解析 request-level SLO。
- `pd_flip_trace_slo.py:87-120` 把 ledger 统计转换成 cluster snapshot。
- `pd_flip_trace_slo.py:125-148` 读取最近窗口内每个 request 的最新记录。
- `pd_flip_trace_slo.py:151-178` 分别计算 TTFT 和 TPOT good/total。

### 2. Safe

`Safe` 是 monitor 正常采样状态。此时只读指标，不改路由，不暂停 admission。live handoff 的第一条状态就是：

```text
safe(reason=monitor_sampled)
```

### 3. preparing_kv_transfer

当 prefill SLO attainment 低于 enter threshold，controller 选择一个 decode 节点作为 source，再选择另一个 decode 节点作为 migration target。

本轮 handoff 实验里：

- source：`node2`
- target：`node3`

动作顺序：

1. router 对 source drain，停止给 source 分配新请求。
2. source worker pause admission。
3. source 调 `/pd_flip/migration/source/start`，生成需要迁移的 KV manifests。
4. target 调 `/pd_flip/migration/target/prepare`，接收 manifests，prepare KV，但还不正式 adopt。
5. controller 轮询 source/target 的 `/pd_flip/migration/status`。

对应源码：

- `pd_flip_controller.py:589-615` 进入 preparing 并 drain/pause source。
- `pd_flip_controller.py:617-638` source start + target prepare。
- `pd_flip_controller.py:825-853` 轮询 migration status，并在迁移过程中继续检查 SLO 是否恢复。

### 4. migrating

流程图里的 migrating 是两阶段迁移的中间态。当前代码里没有单独叫 `migrating` 的 monitor state 字符串，但动作上对应：

- source 已经导出 KV manifests。
- target 正在 prepare/adopt。
- controller 轮询 source 和 target migration status。
- 如果此时 prefill SLO 已恢复到 exit threshold，controller 会 abort migration，恢复 source admission，并回到 safe。
- 如果 source 和 target 都 migration complete，则返回 `"transferred"`。

本轮 handoff 观测：

```text
kv_transferred_reqs_observed = [0,0,0,0,2,2,2,2]
kv_failed_reqs_observed = [0,0,0,0,0,0,0,0]
```

说明迁移过程中真实观测到 2 个请求 KV 成功转移，没有失败请求。

### 5. flipping_role

迁移完成后，controller 再采样一次 SLO：

- 如果 SLO 已恢复，不 commit，abort migration。
- 如果 SLO 仍风险，则进入 `flipping_role`。

进入 flipping 后动作是：

1. target commit migration。
2. source finish migration，释放已迁移 request id。
3. 等 source idle。
4. source 调 `/pd_flip/runtime_role/set` 切成 prefill。
5. router 更新 source role 为 prefill。
6. source resume admission。
7. router undrain source。

对应源码：

- `pd_flip_controller.py:677-704` 迁移完成后 commit 前再采样一次 SLO。
- `pd_flip_controller.py:706-765` commit、finish、wait idle、切 source role、更新 router、恢复 admission。

### 6. Safe(prefill)

切换成功后 source 从 decode 变成 prefill，状态机回到 safe。handoff 实验最终状态：

```text
safe -> preparing_kv_transfer -> flipping_role -> safe
```

## 实验 0：四节点环境状态

四个 host 都能连通：

| 逻辑节点 | Host | IP | 预期角色 | 实际状态 |
|---|---|---|---|---|
| node0 | cloud-099 | 192.168.0.42 | prefill | 正常 |
| node1 | cloud-100 | 192.168.0.40 | prefill | worker OOM，router 里是 plain |
| node2 | cloud-101 | 192.168.0.39 | decode | 正常 |
| node3 | cloud-102 | 192.168.0.41 | decode | 正常 |

node1 阻塞证据：

- `node1_gpu_memory.csv` 显示 8 张 GPU 已占用约 132745-133225 MiB/143771 MiB。
- `node1_worker_oom_tail.log` 中有多条 `torch.OutOfMemoryError: CUDA out of memory`。

最终收尾状态已恢复为：

```text
node0=prefill
node2=decode
node3=decode
node1=plain/unhealthy
```

见 `final_router_workers.json` 和 `final_runtime_statuses.jsonl`。

## 实验 1：monitor 监控 SLO 并指挥切换

执行结果：通过。

核心证据文件：

- `pd_flip_live_20260703_113901/summary.json`
- `pd_flip_live_20260703_113901/monitor.json`
- `pd_flip_live_20260703_113901/monitor.raw`

关键结果：

```text
success = true
monitor_exit = 0
monitor_message = pd flip committed after two-phase migration
state_trace = safe -> preparing_kv_transfer -> flipping_role -> safe
restore_exit = 0
```

这个实验说明 monitor 不只是采样指标，而是成功调用 controller 完成了 D->P 切换。

## 实验 2：KV 正常迁移，迁移后继续 decode

执行结果：通过。

关键结果：

```text
kv_transferred_reqs_observed = [0,0,0,0,2,2,2,2]
kv_failed_reqs_observed = [0,0,0,0,0,0,0,0]
client_finish_reason = stop
client_completion_tokens = 2698
client_content_len = 12289
client_has_abort_literal = false
```

request-level SLO probe 也正常完成：

```text
completion_chunks = 128
finish_reason = length
ledger rows = 129
ttft_seconds = 7.2126
ttft_slo_seconds = 0.001
total_tpot_intervals = 127
good_tpot_intervals = 125
tpot_slo_seconds = 0.02
```

这里 `TTFT_SLO=0.001` 是故意设置得极低，用来触发 monitor；TPOT 部分 127 个 token interval 里有 125 个满足 20ms SLO。

## 实验 3：KV 完成后是否等待一段时间再 flip

执行结果：未按原设想通过，发现实现 gap。

你希望验证的是：

```text
KV 迁移完成 -> 源 decode 继续等待一段时间 -> 确认 SLO 没恢复 -> 再进入 flip
```

当前实现实际是：

```text
KV 迁移过程中循环检查 SLO 是否恢复
KV transferred 后再 collect_cluster 一次
如果仍然低于 commit threshold，立即进入 flipping_role
```

对应源码：

- `pd_flip_controller.py:825-853` 迁移过程中循环检查 SLO recovery。
- `pd_flip_controller.py:677-704` 迁移完成后只额外采样一次。
- `pd_flip_controller.py:706-765` 若仍风险，直接 commit + flip。

live artifact 也支持这个判断：

```text
monitor.iterations = 1
state_trace = safe -> preparing_kv_transfer -> flipping_role -> safe
```

所以第三个实验不能写成“验证通过”。更准确的汇报措辞是：我们验证了当前实现具有“迁移中 recovery abort”和“commit 前再确认一次”的保护，但还没有实现“KV 完成后的显式等待窗口”。这是下一步可以补的状态机能力。

## 实验 4：100 请求 trace A/B

trace 已构造，100 条请求平均分为 5 类：

| category | count |
|---|---:|
| prefill_heavy | 20 |
| long_context | 20 |
| medium | 20 |
| tool_context | 20 |
| short | 20 |

每条请求 `max_tokens=128`。完整 trace：

- 结构索引：`trace_ab_baseline_20260703_1p2d/trace_requests.csv`
- 完整 prompt：`trace_ab_baseline_20260703_1p2d/trace_requests.json`
- 100 条预览表：`trace_requests_preview.md`

A/B 结果：

| 实验 | requests | completed | errors | TTFT attainment | TPOT attainment | combined | avg TTFT | p95 TTFT | avg TPOT | p95 TPOT | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no state machine | 100 | 100 | 0 | 0.85 | 1.00 | 0.85 | 2.0876s | 8.7070s | 0.01202s | 0.01245s | 23.77s |
| state machine | 100 | 83 | 17 | 0.83 | 0.83 | 0.83 | 0.4981s | 1.3099s | 0.01294s | 0.01634s | 608.64s |

解释：

1. baseline 在 1P2D 降级拓扑下跑完 100/100，请求都正常完成，combined SLO=0.85。
2. state-machine run 的 monitor 成功提交了 flip，选择 `node3 decode -> prefill`，migration target 是 `node2`，迁移耗时约 0.854s，总切换耗时约 2.547s。
3. 但由于 node1 缺失，flip 后拓扑相当于从 1P2D 变成 2P1D，decode 容量下降，100 条里有 17 条 zero-chunk 请求，combined SLO 降为 0.83。
4. wrapper 原本按 `SOURCE_NAME=node2` restore，但实际 monitor 选了 `node3`，所以自动 restore 失败；实验结束后已手动把 node3 runtime 和 router 都恢复成 decode。

这部分不能汇报成“状态机提升了 SLO”。更准确的结论是：真实集群上验证了状态机可触发、KV 迁移可提交，但在 node1 OOM 导致只有 1P2D 的情况下，A/B 对比是负向结果。要证明收益，需要先释放 node1 GPU，恢复完整 `2P2D` 拓扑，再重复 A/B。

## 数据文件清单

| 文件/目录 | 内容 |
|---|---|
| `pd_flip_live_20260703_113901/` | monitor-driven handoff 实验原始数据 |
| `pd_flip_live_20260703_113901/summary.json` | handoff 汇总 |
| `pd_flip_live_20260703_113901/monitor.json` | monitor loop 结构化结果 |
| `pd_flip_live_20260703_113901/trace_slo_ledger.jsonl` | request-level SLO ledger |
| `trace_ab_baseline_20260703_1p2d/` | 无状态机 100 请求实验 |
| `trace_ab_state_machine_20260703_1p2d/` | 有状态机 100 请求实验 |
| `trace_requests_preview.md` | 100 条 trace 预览 |
| `live_experiment_summary.csv` | A/B 指标汇总 |
| `node1_gpu_memory.csv` | node1 GPU 占用证据 |
| `node1_worker_oom_tail.log` | node1 worker OOM 日志 |
| `final_router_workers.json` | 实验收尾 router workers 状态 |
| `final_runtime_statuses.jsonl` | 实验收尾 node0/node2/node3 runtime 状态 |

## 汇报建议

可以这样讲：

1. “本周我们把 monitor、KV 两阶段迁移、decode-to-prefill role flip 这条链路在真实集群上跑通了。”
2. “monitor 的常规实现读 Prometheus histogram 和 load endpoint，计算 rolling-window SLO attainment；实验 trace 模式额外支持 request-level ledger。”
3. “KV 迁移不是简单重启，而是 source 导出 manifests，target prepare/adopt，commit 前有 SLO 再确认，迁移失败或 SLO 恢复会 abort。”
4. “目前第三个需求里显式 post-KV hold window 还没实现，live run 也证明状态是直接 preparing 到 flipping，这是一个清晰的下一步。”
5. “100 请求 A/B 在 node1 OOM 的降级拓扑下没有体现收益，反而因为切走一个 decode 造成 17 条 zero-chunk。这个结果说明完整 2P2D 资源是验证收益的前置条件。”
