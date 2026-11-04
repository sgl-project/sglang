# 持续生命周期配对实验

`lifecycle_bench.py` 在同一服务二进制上重放固定数量的模拟 rollout 任务，比较 `on_demand`、`fixed_prefetch`、`lifecycle`。沿用 `stress_bench.py` 的 HTTP/SSE、配置核验与发布能力，保留旧压力脚本及其统计口径。脚本只使用已有服务，不管理 GPU、Mooncake 或 Router 进程。

三组均调用 `task_started`、`request_observed`、`task_finished`，以避免将 Router 关联管理成本只算入候选组。`on_demand` 关闭准备事件，`fixed_prefetch` 在固定提前量直接提交 Engine PREFETCH，`lifecycle` 发送 `tool_wait`、`resume_expected`，由 Router 当前配置选择准备时机和保护。运行清单保存 Engine/Router capabilities 和源码摘要；Router 策略配置与服务启动命令需另行归档。

恢复输入在已发布前缀后追加一个固定的合成工具结果 token，取值为前缀末 token。源输出参考使用同一完整输入，发布/准备目标只包含原始页对齐前缀。这样完整前缀可以实际复用，`repeated_prefill_tokens` 不会把“输入完全相同时为生成 logits 强制重算末页”的工作算成恢复失误。manifest 明确记录此合成输入规则。

每组只在开始时清理 A/B 本地缓存，再在 A 生成和发布完整源前缀。每个源输入执行两次相同的固定输出长度和 seed，第二次暖缓存生成作为恢复输出参考；首次源生成与两次输出是否相同均保留。真实 Qwen3 BF16 对照出现过完整 prefill 与 GPU 暖缓存之间的一位数字差异，暖 GPU、L2 与 L3 恢复则一致，单独的 GPU 测试再核验搬运 K/V 的逐元素一致性。源端两次生成耗时都计入源发布及完整作业口径。B 从冷本地状态开始，组内按自然驻留和淘汰推进。A 的源缓存保留；直接 A 的路由变化实验及自由路由中选择 A 的请求可能命中 GPU，按实际命中层级记录。

`--reuse-published-sources` 可用于模型与上下文保持不变、源工作集可由 L3 保留的实验。每个配对轮次使用新 salt，轮内各组共用 salt 和源参考，只在该轮首组执行 A 的初始生成、暖缓存参考生成及发布；后续组只清 B 本地缓存。每组仍有完整源账本，复用记录以 `reference_only/reused_from` 指向实际较早的发布，历史发布记录不保证当前驻留。L3 容量竞争、保护状态继承或源 A 驻留变化是实验变量时，应保留默认的逐组发布方式。

复用模式下，`source_generation_calls` 和 `source_execution_wall_s` 记录实际执行，后续复用组均为零。测得的公共源准备耗时按该轮组数均分，写入既有 `source_publication_s`，用于各组 `complete_job_wall_s` 的一致比较；各组分摊之和等于实际公共准备耗时。分摊定义、实际阶段耗时 `source_phase_wall_s` 和来源组一并归档。推理工作墙钟、压力配置、任务计划及有效恢复数均按原定义计量。

**计量和控制收尾**

每个任务使用不可变的实际工具完成时刻。预测误差仅改变报告给策略的等待时间；正式生成不为预取延后。控制执行在线程池中运行，运行及排队数量受限。单个推理完成后设置消费结束事件，随即允许客户端处理后续任务；不逐任务 join 无关控制收尾。控制积压时实施背压，并记录提交等待与发送迟到。全部后台操作在组结束时 drain，耗时计入工作墙钟。

准备已完整观测或正式请求开始后，客户端停止主动准备轮询，控制 owner 继续等待消费结束并执行清理。恢复时保留观测年龄；历史 ready 不表示持续驻留，实际复用由请求命中信息验证。完整服务及成本快照保存在逐组文件，console、summary 和控制汇总日志只输出指标，避免重复写入大段元数据。

`effective_tasks_per_s` 为完成、未预定取消且输出匹配源参考的模拟 rollout 任务数除以工作墙钟，包含关联、工具等待、控制背压和期末 drain。`all_completed_tasks_per_s` 保留所有完成响应；错误归属和控制失败不会从延迟分布中删除。另报包含源发布的完整作业吞吐、初始实验 flush、控制状态及严格准备完成数量。未接入 trainer，`trainer_waiting_time` 保持为空。

这是有界并发的固定工作量客户端，不能替代开放到达流量或真实训练实验。每条任务记录发送迟到；压力过大时客户端会减少注入速率，报告必须保留这一限制。配对区间以完整轮次为采样单位。

**输入、压力和元数据**

固定 seed 生成每条 lane 的任务顺序、实际等待、预测误差、取消和路由变化；同一计划用于所有组。`--branches-per-family 4 --shared-prefix-fraction .5` 表示每个家族四个分支共享前半输入与 salt。每个分支使用独立逻辑任务 incarnation，避免并行更新 session revision。`--salt-mode shared` 则令整组共用 salt，覆盖同 salt 不同内容的读取限制。完整 token 材料和家族标识写入清单，可离线重放。

`--host-capacity-tokens` 应填入服务实际观测容量。脚本按 salt 和祖先依赖去重逻辑页，工作集小于该容量时拒绝压力运行；功能测试可显式使用 `--allow-small-working-set`。这些 token 计数不代表物理 I/O 字节。`--demote-under-pressure` 启用每两秒一次的 host 使用量采样；仅接受单个、有效且六秒以内的 `hicache_host_used_tokens/total_tokens` 观测。达到 `--pressure-threshold` 后才向 `tool_wait` 报告压力，并指定 host 源，允许支持该路径的策略执行 L2→L3 DEMOTE。指标缺失、多池歧义或过期时报告无压力，不推断容量已满。

长测前使用专用 Router。生命周期 receipts 和 task tombstones 有明确容量，完成任务并不会自动清空所有历史。累计任务超过 5,000 时，必须提供 `--router-budget-file` 指向实际 Router JSON。脚本只提取并归档 `max_operations`、`limits.max_tombstones`、`limits.max_sessions` 数值，不保存 bearer；核验完整运行事件总数、任务数及并发需求。该检查假设 Router 无其他历史记录。还需配置足够的 active branches、request/content/material、job 和 Engine 保护记录容量，保留启动配置以供核对。

**运行示例**

快速筛选使用真实 `mem_fraction_static=0.8`、无 `max_total_tokens` 上限、TP1 FULL attention、HiCache 和 Mooncake。脚本核验双 Engine 配置一致，自动分配容量差异容许 1%。Router 使用当前候选策略配置；不得在配对中替换服务二进制或调节容量。

```bash
python benchmark/session_kv_control/lifecycle_bench.py \
  --router http://127.0.0.1:12910 \
  --router-control-key-env SGL_ROUTER_CONTROL_BEARER \
  --engine-a http://127.0.0.1:51801 --engine-b http://127.0.0.1:51803 \
  --model /home/youwei/bzh/model/Qwen/Qwen3-8B \
  --output /tmp/lifecycle-c8-screen \
  --host-capacity-tokens 693952 \
  --router-budget-file /path/to/protected-router-config.json \
  --concurrency 8 --cases 64 --passes 2 --blocks 1 \
  --prefix-tokens 16384 --output-tokens 32 --tool-wait-s 3
```

该默认矩阵每组 128 次恢复，加上 64 次源发布；不满足正式长测要求。筛选 C8/C16/C32 后冻结参数，使用独立 seed 正式评估。可从 `--cases 64 --passes 20 --blocks 5 --min-arm-resumes 1000 --min-arm-seconds 720` 起步；每组每轮 1,280 次恢复、五轮累计 6,400 次且至少 60 分钟。运行时长门槛只做验收，不用空闲等待补足；某轮不足 720 秒时返回 2，需在正式实验前根据筛选结果增加固定 passes。采用三组时有效测量至少三小时，源发布和初始准备另计。

共享分支会降低去重工作集，50% 共享、四分支家族可从 `--cases 96` 起步。预测/取消筛选可附加 `--wait-jitter-s 1 --prediction-error-s 2 --cancel-fraction .1`，直接模式再加入 `--route-change-fraction .1`。固定有效目标和计划的取消集合在各组保持一致。`--via-router` 令正式请求经普通 Router 选路，并强制核验 `x-sgl-worker-url`；该模式与显式路由变化比例互斥。

**离线复算和验证**

```bash
python benchmark/session_kv_control/analyze_lifecycle.py \
  --results /tmp/lifecycle-c8-screen --output /tmp/lifecycle-recomputed.json

cd benchmark/session_kv_control
python -W error::ResourceWarning -m unittest -v test_lifecycle_bench
```

原始 HTTP/SSE/控制记录在 `lifecycle-raw.jsonl`，清单保存完整计划，每组 JSON 保存全部任务，summary 保存轮次配对。manifest、组结果和 summary 采用临时文件替换写入；结束时生成文件 SHA256。复算检查计划、输入摘要、源输出参考、固定输出数、异步控制记录和期末 drain 时间范围，不仅依赖已有 summary。返回 2 表示任务、输出、控制或时长门槛未通过；原始结果仍保留。

正式归档应同时保存服务源码/dirty 状态、Router 配置的非秘密副本、模型身份、实际池容量、资源采样、Mooncake segment 状态、全部失败/退回与清理记录。`server_before/after` 保留可用的服务信息；物理 GET/PUT 成本必须使用实现提供的计数或独立观测，不从缓存命中 token 数换算为测得带宽。
