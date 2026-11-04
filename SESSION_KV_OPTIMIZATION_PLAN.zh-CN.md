# Session KV 生命周期优化：方案、实施计划与性能验收

日期：2026-11-04。状态：实施及评估中。本文保留设计目标，已落地接口与提交见第 10 节实施记录。设计候选不等同于已支持的配置；正式性能结果需以冻结配置后的配对评估为准。

代码基线为 `rfc-kvc` 的 `d1a689121c`，保留原分支及上游合入历史。运行代码的长测快照为 `ff764308fb`；其后提交主要归档实验与文档。原有 P0–P8 实现和 P9 配置扩展继续由 [PLAN.md](PLAN.md) 管理，本轮使用 O0–O7 编号。外部语义来源为 [RFC #45189](https://github.com/sgl-project/sglang/issues/45189)。

## 1. 目标、范围与判断依据

优化目标是在有限 L2 容量、L3 保留能力和搬运成本之间做取舍，降低后续请求实际承担的恢复与重复 prefill 开销。提前预取将部分读取移出请求关键路径，同时提前占用 L2 与带宽。错误预测、过早准备和不合适的保留策略都可能损害其他请求。

对外继续使用 `PIN`、`PREFETCH`、`DEMOTE` 三种操作。调度、时机预测、容量预算和候选排序属于策略；Engine 保留实际批准、执行和资源释放权。L2 命中率、提前就绪率、重算 tokens 是解释指标，验收以完整任务延迟、有效吞吐、尾延迟及资源成本为准。

本轮基准配置沿用 FULL attention、TP=1、多独立 Engine、persistent-host HiCache、Mooncake、`mem-fraction-static=0.8`，无显式 `max-total-tokens` 上限。file 用于可控故障与正确性测试。overlap、CUDA graph、TP/PP/DP、Hybrid 和 RDMA 等配置扩展独立评估，避免与策略变化混合。

| 决策 | 本方案采用的方式 |
| --- | --- |
| 主要策略 owner | 单 Router；已有 Engine 可选策略在对照实验中关闭，避免重复处理同一生命周期事件 |
| Engine 执行 | 复用 HiCache、native storage jobs、保护注册表与 owner 完成处理 |
| 新策略上线方式 | 单独开关，默认关闭；纯观测、影子决策、受控启用依次验证 |
| L2 PIN | 有限 TTL 的普通驱逐排除，预算按物理去重范围计算 |
| L3 PIN | 经现有单写入者 backend protection owner 发起 Mooncake soft-pin，保留后端条件 |
| DEMOTE 扩展 | 优先补齐现有可回收范围内的 L2→L3；更广的 host 副本回收后置 |
| 参数选择 | 从预注册的小范围扫描选取；不根据正式测试结果反复调参后复用同一测试集 |
| 提交方式 | 每项功能及必要测试通过后独立提交；不将 Codex 加入 Co-authored-by |

### 已有证据及适用范围

[0.8 长测报告](benchmark/session_kv_control/results/2026-11-04-mem08/FINAL_ACCEPTANCE.zh-CN.md)记录约 137 分钟的执行时段、5,724 次生成调用和 4,656 条跨条件一致输出。执行时段包含准备、重置和切换。目标 B 的 L1/L2 容量为 346,944/693,952 tokens；natural L2 测试观测分配/预留峰值约 99.24%/99.75%。Mooncake v0.3.14 TCP，每 Engine 挂载 128GiB segment。

| 实验 | 平均 TTFT 变化 | 工作吞吐变化 | 需要保留的结论边界 |
| --- | ---: | ---: | --- |
| L2 PIN，natural 热点 | −87.1% | +0.49% | 热点有收益，包含冷计算的混合吞吐约持平 |
| L2 PIN，显式 DEMOTE 口径热点 | −84.2% | −0.13% | 用于 PIN 条件对照，不能独立证明 DEMOTE 策略收益 |
| 优化快照 PREFETCH，C8 | −48.0% | +11.5% | p95 改善 28.3%；两组重复 prefill 均为零 |
| 优化快照 PREFETCH，C16 | −9.9% | −1.1% | 完整工作吞吐约持平 |
| 优化快照 PREFETCH，C32 | −28.2% | +13.9% | p95 增加 16.2%，4 次 `DEFERRED/whole_scope_capacity` |

L2 PIN 两种口径执行于 `f65f0393c4`，表中的优化后 L3 数据执行于 `ff764308fb`；各项均相对同快照的按需组。C8 仅 141/384 次预取在实际请求发送前完成，在途复用也可能贡献收益。C32 重复 prefill 从 1,263,360 降至 295,552 tokens，减少约 76.6%；四次退回占约 1%，尚不能单独解释整体 p95 变化。旧快照的一次失败与新快照的四次退回均保留于全部完成请求统计。

上述吞吐含工具等待与控制收尾，排除每组初始实验 flush。两快照的按需基线自身有变化，跨快照绝对差值不能全部归因于优化。主因果对照应使用同快照内配对，以及后续同实现、不同功能开关的消融实验。

现有测试只有三个配对轮次，使用独立 salt 的合成 16K 前缀、固定 3s 工具等待和 2.5s 提前量。它没有充分验证预测误差、同 salt 共享前缀、L3 PIN 的容量竞争、三动作组合或 RL 训练等待。当前 L3 长测由 Engine 接口驱动；已有 Router 功能基准需要扩展为持续生命周期负载。

## 2. 代码现状与修改边界

下表路径均相对本仓库根目录。标为“拟新增”的模块尚不存在；实现阶段可在不改变职责的前提下调整文件组织，并同步本文。

| 位置 | 已有行为 | 本轮改动 |
| --- | --- | --- |
| `sgl-router/src/orchestration/runtime.rs` | 生命周期输入、延迟提交、全局 ms/token EWMA、操作及 claim 管理、准备位置反馈 | 接入独立策略决策、分组成本估计、预算观测、失效与消费后释放 |
| `sgl-router/src/orchestration/mod.rs` | 严格配置校验、准备信息合入路由 | 新策略开关与有界参数；保留普通路由的最终选择权 |
| `sgl-router/sgl-kv-orchestrator/src/{session_index,target_resolver,operations,observations}.rs` | 会话/分支关联、冻结范围、精确 job 合并、位置观察 | 增加策略输入快照、决策原因、跨事件重新评估；不合并不同权重或内容身份 |
| `sgl-router/sgl-kv-orchestrator/src/preparation.rs`（本轮新增） | 独立、有界的分组准备成本统计 | 纯统计与提前量计算；生命周期与异步 owner 交互保留在现有 runtime |
| `python/sglang/srt/mem_cache/unified_cache/control_storage_reads.py` | 默认最多 4 个 native read；不同有效 salt 流水；同 salt 串行；精确 demand 复用；完整缺口预留；GPU FIFO 屏障 | 字节/容量预算、真实 demand 提升、等待原因、时限调度；后期评估重叠范围共享 |
| `python/sglang/srt/managers/cache_controller.py` | HiCache 查询、GET/写回队列、缓冲分配、完成处理 | 同时覆盖普通恢复与控制读取的观测和调度入口；不假设增加 native job 数会增加 GET worker 数 |
| `python/sglang/srt/mem_cache/unified_cache/{control_storage,storage_execution,control_transfers,unified_tree_core}.py` | storage COPY 与 GPU→storage MOVE；host 叶节点回收；write_back 下另有 FULL host 重复副本回收 | L2→L3 保存后定向回收、作用范围反馈；复用或提取安全 helper |
| `python/sglang/srt/mem_cache/unified_cache/control_protection.py`、`python/sglang/srt/managers/kv_control/protection.py` | L2 claim、TTL、去重预算、共享保护 | 自动策略的短租约和预算选择；保留显式调用者已承诺的保护 |
| `sgl-router/sgl-kv-orchestrator/src/backend_protection.rs` | Mooncake 单写入者 soft-pin、共享 claim、恢复不确定状态 | L3 生命周期选择与成本反馈，不增加第二个保护写入者 |
| `python/sglang/srt/managers/kv_control/{contracts,feedback,service}.py` | 严格 schema、范围结果、请求命中、capabilities | 经协商的观测扩展、能力描述；修改时同步 Rust schema/fixtures |
| `benchmark/session_kv_control/{stress_bench,router_bench,observe_resources,analyze_stress}.py` | Engine 压测、Router B2/B3、资源采样、离线分析 | 动作消融、预测误差、共享前缀、持续 Router/RL 模式及成本分析 |

实现必须保持以下约束：

- `PREFETCH` 只承诺所报告时刻已准备的范围；持续保护需要单独 PIN。PIN 不隐式搬运缺失数据。
- `DEMOTE` 的 `preserved_scope` 与 `reclaimed_scope` 分开报告；仅有 L3 写入凭证时不能报告 L2 已释放。
- 物理数据按 namespace、权重版本、layout、page、salt 和内容身份隔离；任务结束仅清理该任务的兴趣及 claim。
- 所有 radix 更新、slot 释放和引用变化由 owner 提交。取消、超时、RPC 响应丢失不构成物理 quiescence。
- Engine 的容量检查为执行依据。Router 的容量观测具有时间差，只能用于筛选和提交节流。
- 未知保护或位置保持 unknown，经过相应 query/recovery 再使用；控制面故障下普通推理继续使用已有安全恢复路径。
- 自动策略不得提前撤销其他调用者的有效 PIN，也不得通过修改 salt 制造本不合法的并发或缓存共享。

## 3. 决策状态与生命周期

策略按“兼容内容范围 × 目标 Engine/epoch × 调用方需求”维护状态。会话、任务和分支关系用于推断需求；物理 job 与 claim 继续由现有管理器管理。新增状态有明确容量、过期和清理限制。

| 输入 | 来源与处理 |
| --- | --- |
| 当前范围、分支 revision、共享祖先 | 现有 session index 与 target resolver；提交时冻结，分支增长后重新解析 |
| 恢复时间窗口、预测置信度、用途 | 应用事件优先；未知时不编造准确 deadline，采用保守策略 |
| 即将到达与正在等待的真实需求 | 请求入口和 Engine demand 关联；可信真实 demand 优先于推测 |
| L1/L2/L3 位置与保护 | 带 epoch/watermark 的查询、事件和后端凭证；记录新鲜度 |
| 读取排队、读取时间、H2D、发布成本 | O0 的阶段计时，按目标/后端/layout/大小/负载档分组 |
| L2 可用量、预留量、可回收候选 | Engine owner 快照；候选量为有界扫描估计，执行时再次检查 |
| RL 关键任务信号 | 外部 scheduler/trainer 提供的 ready 队列、批次目标和等待状态；未接入时权重使用普通任务默认值 |

状态流程为 `observed → planned → preparing → prepared → consumed/released`。任一阶段可进入 `deferred`、`cancel_requested` 或 `unknown`；这些是内部策略状态，公共 operation 状态继续遵循既有协议。

| 事件或状态 | 决策及操作 | 退出条件 |
| --- | --- | --- |
| 短等待，已在 L2，近期复用价值高 | 预算允许时短期 PIN；有现成副本时跳过 PREFETCH | 消费、预测改变、TTL 或任务结束 |
| 长等待或低近期复用，存在容量压力 | 优先选有 L3 副本的冷目标 DEMOTE；长期保留需求可另发 L3 soft-pin | 保存/回收分别确认；阻塞时等待或退回 |
| 恢复窗口接近，目标 worker 已选 | 提交 PREFETCH；已准备且消费有间隔时申请短期 L2 PIN | 真实请求消费、失效或保护到期 |
| 真实请求早到 | 复用兼容在途读取并提升需求等级；无法推进时安全退回 | 普通请求不等待无用控制操作 |
| 恢复时间推迟 | 延迟未开始 job，重新评估已完成内容的 PIN；必要时释放自有 claim | 防止不断续期长期占满 L2 |
| 分支取消、任务结束、worker 改选 | 撤销该需求，释放自有 claim；共享 job 按剩余兴趣继续 | 在途 work drain 后释放资源，保留已发生的效果 |
| 权重变化、Engine 重启、事件 gap | 失效对应位置/预测；重新查询或按需恢复 | 不使用旧 epoch 的 ready 或保护凭证 |

PREFETCH→PIN 采用两个显式操作。第一版允许两个操作间存在驱逐窗口：PIN 失败就保留缺失/退回结果并重新评估，不宣称连续保护，不循环搬运到成功。请求 admission 期间已有内部引用继续按原生命周期移交。未来若增加有界移交机制，应另行声明能力并测试，不能隐式延长所有 PREFETCH 的驻留时间。

消费后释放 PIN 以 Engine 已建立正常请求引用或已完成实际使用的证据为准；仅有 Router 发出请求不能证明引用已接管。准备操作 terminal 也不代表 speculative 占用消失：未消费内容保留在准备账本中，直到实际使用、回收或观测失效；失效未知时保守处理并有界重查。

冷却时间限制同一范围短时间内反复 DEMOTE/PREFETCH。真实 demand 可以触发重新准备；显式有效 PIN 始终阻止不符合契约的源回收。任务结束不等于删除全部内容，其他任务仍可能共享这些内容。

决策按以下顺序执行；这是拟实现的策略伪代码，公开三动作协议保持独立：

```text
on_event_or_timer(snapshot):
    reconcile task revision, worker epoch, claims and shared interests
    if demand ended:
        withdraw this demand; release its auto claims; drain only unshared work
        return
    if compatibility or location evidence is unknown:
        query within control budget; otherwise use ordinary recovery
        return
    if Engine reports a matching actual waiting request:
        promote the shared read through Engine admission; retain safe fallback
        return
    if confirmed L2 scope exists and near-term value passes the policy threshold:
        acquire or renew this policy's bounded short PIN if budget allows
        return
    if reuse is distant and host pressure exists and cooldown permits:
        request DEMOTE only for a verified compatible reclaim candidate
        optionally retain L3 through a separately confirmed soft-pin claim
        return
    if expected resume window exists and useful preparation is still plausible:
        schedule PREFETCH at the calculated due time
        recheck evidence and budgets at dispatch; defer when admission fails
```

短等待、远期等待和价值阈值由剩余等待与当前准备成本比较得到。默认使用显式应用恢复窗口；没有概率校准数据时，保留“明确需要/可能需要/未知”分档，不伪造精确概率。策略决策保存输入快照摘要、选择原因和未执行原因，便于按同一轨迹离线复算。

## 4. PREFETCH：时机、预算与需求优先级

### 时机与成本估计

恢复准备时间分解为查询/排队、L3 读取、owner 发布；如目标包含 L1，再单列 H2D。当前主路径目标为 L2，H2D 通常仍属于恢复请求成本。分组估计至少区分目标 Engine、后端/layout、缺失大小和负载档，按有界样本维护 EWMA/分位数与样本数。旧全局斜率作为冷启动回退。

建议的提交时刻为：`预计恢复时刻 − 估计准备分位耗时 − 预测误差裕量`。预测不准确时同时限制最大提前量，避免仅靠增加裕量掩盖拥塞。余量和分位水平作为实验参数，不预设“越早越好”。缺失范围按已确认副本扣除，不能按总前缀长度重复计费。

独立记录纯预取、与 demand 合并、缺失/失败/取消样本。失败和超时记录为截断或失败观察，不能当作零耗时，也不能仅用成功快样本训练模型。Router 与 Engine 使用各自 monotonic 计算本地耗时；跨进程关联用 job/operation ID、epoch 和持续时间，禁止直接相减不同进程的 monotonic 时间。

恢复窗口不足不自动意味着毫无价值：已在途读取可以减少剩余等待。新 speculative job 只有在预算允许且预期减少未被其他工作覆盖的等待时才进入队列；普通 demand 到达后的读取继续由正常恢复路径处理。

### 容量与搬运预算

同时保留任务数上限与容量上限。建议将以下字段作为新配置设计候选，具体名称在 O0/O1 固定；它们目前不能直接填入服务配置。

| 候选配置 | 单位与语义 |
| --- | --- |
| `max_speculative_host_bytes` | 无真实 demand 的准备内容与在途缺口的物理去重预算 |
| `max_storage_read_inflight_bytes` | 所有参与该预算的 L3 读取在途容量，普通恢复和 native 控制读取均计入 |
| `demand_host_reserve_bytes` | speculative 准入需留下的余量，供真实恢复与 staging 使用；不预先物理分配 |
| `speculative_read_bytes_per_second` | 可选的后端调用数据量限速；没有 wire 计数时不得命名为网络实测速率 |
| `max_prefetch_lead_ms`、`ready_pin_ttl_ms` | 最大准备提前量、准备完成后的可选短 PIN TTL |
| `auto_pin_budget_bytes`、`migration_cooldown_ms` | 自动策略保护预算与迁移冷却期；不得覆盖显式 PIN 的契约 |

对每个 host pool 使用同一账本：实际分配已从 `available_size()` 扣除；未分配的物理 job 缺口才作为额外预留扣除。shared job 只计一次。取消查询的预留保留至实际分配或 drain；转为 demand 时重新分类，不能“提升优先级”后丢失其占用。

供 O0/O1 起步实验使用的建议值如下，均需写入 manifest，不能当作已经测优的生产默认值。按容量比例换算 bytes 时使用实际 FULL pool/layout，向下对齐 page；范围预算无法容纳一个允许的最大 job 时，应校验报错或显式缩小可准入范围，避免队首永久等待。

| 参数 | 起步值或规则 | 筛选候选 |
| --- | --- | --- |
| native read 个数 | 保持当前上限 4 | 1/2/4；仅在资源证据支持时测试更大值 |
| speculative host 预算 | 实测 L2 bytes 的 10% | 5%/10%/20% |
| 全部读取在途预算 | 实测 L2 bytes 的 20% | 10%/20%/30%，核验最大 scope 可容纳 |
| demand host 余量 | 实测 L2 bytes 的 5% | 5%/10%；真实 demand 本身仍需物理准入 |
| 自动 L2 PIN 预算 | 现有保护上限与 L2 10% 的较小值 | 保持显式 PIN 上限，单独扫描自动预算 |
| 准备耗时分位 | p90，样本不足回退现有成本估计 | p50/p90/p95；估计器按组有界 |
| 最大提前量 | 10s 且不超过剩余工具等待 | 2.5s/5s/10s |
| 完成后短 PIN | 1s，已有可用保护则不重复申请 | 0/0.5s/1s/2s |
| 同范围迁移冷却 | 3s，真实 demand 可要求重新准备 | 1s/3s/10s |
| 后端调用 bytes 限速 | 默认不额外限速，先用容量与并发门控 | O0 有稳定观测后按实际共享服务能力设值 |

预算筛选不并行扫描所有参数。先固定任务数比较容量预算，再固定预算比较时机和 PIN；每组仅改变声明的因素。显式 PIN 已占用的范围应计入可用容量，自动策略不得用自己的预算上限挤占已承诺的保护。

speculative job 的准入需同时满足 job 数、在途 bytes、自有准备预算和需求余量。不得通过释放他人的 claim 或主动驱逐受保护内容取得资格。真实 demand 不受“推测预算”约束，但仍受物理池、共享 I/O 和 GPU 安全约束；无法批准时保留安全计算退回。

不能简单把 PIN bytes、准备 bytes 和在途 bytes 相加作为物理用量：这些集合可能重叠。用物理引用集合去重，另按用途记逻辑分类。可回收估计使用有界扫描或已有索引，避免每个 idle poll 全树遍历。

### 队列调度

第一版按“真实等待需求 → 临近恢复窗口 → 远期预测”分级，组内使用时限和等待年龄，避免后台工作长期饥饿。写回也有独立最小推进机会，防止读优先阻塞源释放。保留 GPU FIFO 屏障和同 salt 串行约束，优先级不允许绕过物理冲突。

需求加入 queued native job 后，更新调度属性，不重复 query/GET。调度属性与不可变物理 job 身份分开，避免改变 deadline 就制造新读取；同 job 的调度等级由所有仍有效兴趣聚合。Router 不能伪造 Engine 已观察到的真实 demand。

O1 先控制入队和尚未发出的工作，不承诺抢占已提交后端的 GET。若 O0 显示真实 demand 仍长期排在已入队大传输后，O1b 再在现有 I/O 批次边界增加有界调度；需保留 partial ACK、buffer 引用、取消与写回推进语义。独立增加 GET worker 不属于本阶段默认方案。

## 5. DEMOTE：L2→L3 保存与定向回收

本轮优先实现 FULL/TP=1/persistent-host 的 L2→L3 DEMOTE。已有 write_back 分支具有 FULL host 重复副本回收 helper，但正式长测使用 write_through。不能据此宣称 write_through 已有通用定向 host 回收，也不能为了获得收益暗中切换写回模式。

O3a 限定为可形成完整合法回收范围的 host-only 叶节点或自末端向前的可回收后缀。一个完整前缀可能因共享子节点或锁只能部分回收；严格报告实际范围。内部节点、仍有 GPU 副本的目标和未对齐的物理边界不能直接套用 H-leaf 删除。

执行步骤：

1. 冻结目标范围和兼容身份，持有源读取所需引用；核验当前 backend capability。
2. 对完整组件/key 集合确认有效 L3 副本。历史 `BlockStored` 或本地标记不足以证明当前存在；后端不支持可靠确认时使用幂等写入并等待完成。
3. 仅为缺失内容安排写入，保存实际成功范围。预先存在的字节与本次写入的字节分开计数。查询/写入均不阻塞 owner 线程。
4. I/O quiescent 后回到 owner，再次检查 PIN、计算引用、H2D/写回、节点关系及当前 epoch。移交或释放本 job 自有 source hold 后才检查可回收性；不得清理其他 job 的引用。
5. 调用受范围约束的回收 helper，释放确实符合条件的 host slots，提交 CPU location removal 及 pool 计数，保留 L3 与其他副本的位置。
6. 返回 `preserved_scope`、`reclaimed_scope`、`blocked_scope` 和原因。源仍有保护/引用时保留数据，按有界重试、deadline 或取消收尾。

后端存在查询和后续回收之间不能提供无限期原子驻留保证。L3 soft-pin 只能提供后端声明的条件；需要保留保护时通过现有 owner 建立 claim 并确认。后端保护未知、写入失败或保留条件不满足时，不报告保证已成立。DEMOTE 成功报告的是已观察到的保存/回收效果，后续 L3 失效仍需正常恢复处理。

回收 helper 必须限制在目标源范围内。若复用 `_evict_host_leaf` 会递归删除祖先或产生范围外回收，需要提取/限制逻辑，不能把这种副作用当作目标回收完成。现有 PIN、引用和共享分支约束在每次提交前检查。结果可以是保存成功且回收部分阻塞；重放相同操作不会重复释放 slots。

O3b 再评估 GPU 驻留时的 FULL host 重复副本定向释放。先复核现有 write_back helper 的前提，再为 write_through 补足正确性证据。O3a 无法处理的目标继续返回明确阻塞原因，不以扩大 GPU 驱逐范围来掩盖 L2 约束。

策略侧优先选择长期不用、L3 已存在、低共享收益且实际可回收的目标。无容量压力时跳过仅为“腾出空间”而产生的搬运；采用冷却期避免短时间往返。验收统计实际释放 bytes、额外写入/再读取 bytes 和由此减少的退回/重算。

## 6. PIN 选择、共享内容与 RL 任务价值

自动 PIN 候选依据复用概率、预期复用次数、未保护时的恢复/重算代价、对任务进度的影响，以及新增物理 bytes 和保留时长排序。第一版采用可解释的分档规则与有界配额，不引入训练型预测模型。

用于候选比较的启发式可写为：`预期减少的未覆盖等待毫秒 / 新增物理 byte-seconds`。不同使用者的收益只统计实际潜在需求；同一物理范围容量计一次。该比值属于待验证的排序方法，统计推断不能当作单请求的精确反事实收益。新增物理占用为零的共享保护单独处理，避免除零或重复扣费。

PIN 不需要改变既有公开语义。自动策略只续期仍有价值的自有 claim；重新排序后可以释放自己的自动 claim，但不能撤销其他调用者的有效契约。续期需有总保留上限、重评估间隔和失败反馈，避免反复续期占满预算。共享祖先的多个 claim 继续独立管理。

L3 soft-pin 适用于较长等待中仍有复用价值的内容。需要在受控 L3 容量竞争下比较存在率、恢复率、重算、pin 调用开销和对其他内容的影响。后端无法观察内部驱逐原因时记录 unknown，不用成功 pin RPC 代替持续驻留证明。

RL 中使用两类实验：固定目标数量的同步 rollout 批次，以及 trainer 消费队列驱动的异步 rollout。应用可报告剩余有效样本、可运行任务数、训练端等待和即将成为慢任务的分支。策略优先减少真实等待训练所需样本的恢复成本；可被其他任务覆盖的等待使用较低优先级。训练等待按 trainer 实际无法推进的墙钟区间计量，不能将多个 rollout 的并行等待相加充当训练等待。

“有效 rollout”由实验前固定的训练接收规则定义，包括版本、完成状态和时效要求，不依赖候选策略的成功与否。随机采样时使用相同输入/seed 调度并记录接受标准，不能靠减少目标数量、输出长度或剔除困难分支提高吞吐。不同权重 namespace 的 KV 不跨版本复用；参数更新需同时处理旧任务、旧位置、claim 与在途工作。

## 7. 共享前缀与较小准备范围

O6 属于后续扩展，不阻塞 O0–O5。当前 native 精确目标可合并，但同有效 salt 串行。实际业务常有同 salt 的独立上下文或分支；下一轮必须包含这类测试，避免只在独立 salt 工作集上评价流水收益。

O6a 可在 Router 选择页对齐的完整祖先前缀 `[0,k)`，优先准备或保护多个分支都会用到的内容。原始完整需求与实际选取范围分别记录，不将缩短目标包装为完整准备。是否减少总时间要同时考虑剩余 prefill、批处理与 H2D。

O6b 才考虑重叠 job 复用和同 salt 不相交范围并行。需要显式范围重叠、祖先依赖和引用管理，不能删掉 salt 串行检查后直接开放并发。FULL attention 的可恢复范围必须包含所需依赖；不同组件/模式另作能力扩展。逻辑请求取消、共享子任务继续和物理字节去重需要单独测试。

## 8. 观测、归因与协议兼容

O0 必须在新策略前完成。观测覆盖普通按需读取、native PREFETCH、DEMOTE 写入及 PIN，避免只测候选路径。热路径使用低基数计数器、采样 histogram 和有界 journal；完整 token/存储 key 不作为指标标签。

| 观测 | 定义与采集位置 |
| --- | --- |
| job 阶段 | 接受、排队、容量批准、query、GET/PUT、发布、首个 demand 加入、请求使用、取消、drain；记录物理 job ID 与所有逻辑兴趣 |
| 容量 | 总容量、实际分配、未分配预留、PIN 去重占用、未消费准备占用、实际回收；分类交集明确 |
| 搬运成本 | 后端 API 提交/成功/重试 bytes 分开；有 wire 观测再单列网络 bytes，缺失保持 null |
| 时间成本 | query/queue/transfer/publish/H2D 分开；重叠时间不直接相加为墙钟时间 |
| 使用与浪费 | 实际消费的范围/目标、部分使用、取消、路由改选、过期未用；期末仍等待消费的样本作为未决，不直接计浪费 |
| 驱逐影响 | 被回收范围后来再次访问、重新搬运/重算的次数与成本；只有配对实验支持时才归因为策略增量伤害 |
| 请求与任务 | HTTP TTFT、完成延迟、完整任务耗时、普通背景流量延迟、请求发送偏差、控制 drain 时间 |
| RL | 有效样本/s、批次完成时间、训练等待墙钟时间、就绪队列、版本失效/废弃样本；模拟与真实 trainer 数据分别标注 |

一个物理 GET 被多个请求共享时，物理 bytes 只计一次，逻辑受益范围可以多次关联。按内容与 job 的交集计算实际使用，重叠请求不能使“物理已使用字节”超过该 job 完成字节。暂时无法确定的归因保持 ambiguous。逻辑 KV 字节估计、后端提交数据量和实际网络传输量分别命名。

L2 byte-seconds 对去重驻留集合积分，报告采样周期、误差与缺口。Engine 运行期间只依赖 `nvidia-smi` 不能证明缓存占用；保留启动容量与可用 token 观测。资源采集本身需测 CPU、内存和延迟开销。

协议采用保守扩展：当前 Python/Rust schema 都严格拒绝未知字段。新 telemetry 可放在独立、版本化的查询结构；新增调度提示需在 capability 协商后才发送，必要时使用新的 action version，但动作仍为原三种。不能直接向旧 `Prefetch`/`Demote` payload 塞入新字段。同步更新 contracts、Rust models/protocol、JSON schemas、golden fixtures 和未知能力回退测试。

新字段的 name、unit、是否可空、来源、epoch 和范围在 O0 固定。时钟、计数器 reset、gap、查询错误都有明确状态。没有物理 byte 观测时允许验证时延与功能，但不得宣称已经降低带宽成本。

## 9. 性能实验与验收

### 对照组和工作负载

使用同一候选二进制、不同开关进行主要对照，保留固定 `ff764308fb` 作为历史锚点。各组共享相同输入、工具轨迹、有效任务目标、资源容量与源发布方式。基础动作已存在时，按需组仅关闭新增自动策略，保留普通 HiCache 恢复。

| 组 | 用途 |
| --- | --- |
| A0：按需恢复 | 同实现的主基线 |
| A1：已有固定时机 PREFETCH | 区分本轮策略与已有提前搬运收益 |
| A2：仅自动 PIN | 检验保留选择与预算 |
| A3：仅生命周期 DEMOTE | 检验实际回收收益和再读取代价 |
| A4：动态 PREFETCH，无额外 PIN/DEMOTE | 检验准入与时机；按需提升能力独立开关消融 |
| A5：动态 PREFETCH + 短 PIN | 检验准备到消费之间的保留价值 |
| A6：PIN + PREFETCH + DEMOTE | 检验组合策略及容量竞争 |
| A7：已知真实未来事件的参考策略 | 只作为相同资源约束下的理想信息参考，不能冒充可部署策略 |

扫描维度按阶段选择，不执行全部笛卡尔积：并发 8/16/32；有效工作集约为实测 L2 的 0.75/1/1.5/2 倍；上下文 2K/8K/16K，长上下文不超过实际模型配置；短/长工具等待和抖动；预测偏早/偏晚、分支取消、路由变化；同 salt 不同内容、共享祖先 0/50/90%；L3 占用与保护压力；有/无普通背景请求。

共享率按实际去重内容与兼容身份验证。工作集比例也按去重物理内容计算，不能仅相加各分支 token 数。L3 压力通过本任务专用内容及后端可观测容量控制，记录 master/segment 状态；禁止清理无关任务数据。

RL 分别覆盖 ready 队列充足、ready 队列不足、同步批次接近完成三种条件。测试工具等待必须来自可重放轨迹或明确标注的模拟分布。只有接入实际 trainer 时才报告真实训练等待收益。

### 长测执行与统计

快速筛选用于发现错误和明显退化；参数选择集与正式评估集分离。入选的关键条件建议累计至少 60 分钟有效测量且至少 5,000 次恢复请求，采用至少五个独立配对轮次，每轮至少十分钟；时间和请求量双门槛都满足后结束。该预算按“条件”计算，正式矩阵可能持续数小时，应按 O7 选定场景分批执行。

每轮随机化 A/B 顺序，重建相同初始状态；轮内自然累积缓存与压力。保留稳态窗口和包含 warmup/源发布的完整作业口径。控制脚本不以每次请求 flush 或强制 DEMOTE 制造低容量条件。RL 多轮任务生命周期模式与原有受控路径基准分别报告。

主任务调度器不逐请求阻塞等待无关控制 drain；使用有界异步收尾、明确背压和期末 drain，并将未完成后台任务计入成本。既有 `stress_bench.py` 的控制收尾口径保留，避免修改旧脚本后无法复现历史结果。

采用固定作业数的完成时间与持续流量的有效吞吐两种视角；到达计划包含时间戳，记录请求发送迟到和队列增长。持续流量测试需避免客户端因阻塞降低注入率后掩盖尾延迟。不得推迟恢复请求来等待 PREFETCH。

以独立轮次为配对/重采样单位报告均值、p50/p95、吞吐比及 95% 区间；缓存状态相关的单请求不视为独立样本。样本充足才讨论 p99。保留所有失败、退回、取消及完成请求，另列正确性和严格路径校验，成功预取子集不能替代总体统计。

### 建议的验收门槛

以下数值为本计划建议，O0 在正式实验前固化到 manifest；变更必须在新评估轮次开始前记录。

| 项目 | 验收条件 |
| --- | --- |
| 正确性 | 确定性恢复输出一致；无越界 ready、无重复释放、无跨版本命中；所有资源上限与共享 claim 契约成立 |
| 功能收尾 | 已接受操作可查询；取消/结束后资源安全 drain；未知结果保留 unknown，不能伪造完成 |
| 观测开销 | 同组开关观测的有效吞吐退化目标不超过 2%；超出时降采样或调整实现 |
| 主要收益 | 预先指定的受压力场景中，有效任务/rollout 吞吐目标提升至少 5%，或固定任务量耗时下降至少 5%；配对区间支持改善 |
| 时延约束 | 主要场景 p95 目标退化不超过 5%；普通背景流量也单独检查 |
| 无收益场景 | 无复用或等待已被其他任务覆盖时，吞吐相对基线下降不超过 3%；区间过宽时结论为不足 |
| 成本预算 | 在途、预留、自动 PIN、控制队列不超配置上限；无用传输和 L2 byte-seconds 完整披露 |
| RL 结论 | TTFT 单独改善不能宣称训练提速；必须有有效 rollout 或实际训练等待证据 |

存在平均收益和尾延迟冲突的配置可以保留为可选实验结果，不能作为普遍推荐默认值。测试未通过则保留失败证据、修复或缩小适用范围；不得降低内存预算、减少有效任务或删除退回样本来达到门槛。

## 10. 实施阶段与提交计划

依赖关系为 `O0 → O1 → O2 → O4 → O5`；`O0 → O3a → O5`。O3b、O6 后置，由观测结果决定是否值得实现。O7 在各功能阶段做小规模验证，在组合策略完成后执行正式矩阵。

| 阶段 | 交付内容 | 必要测试与完成条件 | 建议提交拆分 |
| --- | --- | --- | --- |
| O0：观测与契约 | job 关联、分阶段时间、bytes 层级、容量/预留账本、请求使用；协议与配置草案定稿；影子模式 | 单次物理读取多兴趣只计一次；普通恢复覆盖；reset/gap/unknown；观测开销配对 | `feat(kv-control): expose bounded transfer cost observations`；离线分析另提交 |
| O1：容量与需求调度 | speculative/真实 demand 分类、预算准入、queued demand 提升、等待原因、写回推进 | 长短请求混合、容量耗尽、取消 drain、GPU 屏障、无饥饿；真实 file/Mooncake 验证 | 先预算账本，再队列调度；O1b I/O 批次调度单独提交 |
| O2：动态时机 | 纯策略模块、分组成本模型、误差裕量、最大提前量、模型重置与分组降级 | fake clock：早到/晚到/预测更改/epoch 变化；固定提前量消融；样本不足回退 | `feat(kv-orchestrator): schedule prefetch from observed preparation cost` |
| O3a：受限 host DEMOTE | FULL host-only 范围保存、确认与定向回收；capabilities/result 更新 | 后端缺失/部分写入、共享祖先、PIN、传输/计算引用、范围边界、重放、重启；GPU/Mooncake | 存储保存与回收实现、协议/fixtures 和测试作为同一完整功能提交 |
| O4：自动 PIN | 短租约、物理去重预算、候选排序、消费反馈、续期与 L3 保留选择 | 准备后驱逐窗口、预测推迟、共享调用方、TTL、L3 不确定结果；无用保护负例 | L2 自动保护与 L3 生命周期选择分别提交 |
| O5：组合策略 | 生命周期状态机、DEMOTE 冷却、取消/结束/路由变化、权重更新；Router 持续负载入口 | 三动作消融；输入乱序/重复/响应丢失；普通推理推进；实际 worker 归属核验 | 策略与 Router 集成提交；持续负载 harness 独立提交 |
| O3b：host 重复副本回收 | 经审查的 write_through FULL host duplicate 回收 | GPU 副本仍可用、H2D/写回/PIN 不冲突、radix 重构与事件正确 | 可选扩展提交，不与 O3a 混合 |
| O6：前缀范围优化 | O6a 完整祖先选择；O6b 重叠 job 共享和同 salt 安全并行 | ancestor/descendant、范围拆分、不同权重/salt、共享取消、bytes 去重 | O6a/O6b 分别实现与验收；O6b 需要独立设计审查 |
| O7：正式评估 | 冻结参数与矩阵，长测、RL 对照、离线复算、最终适用范围 | 上节双门槛；每轮全部原始结果、来源和资源记录可核对；负结果保留 | harness/分析经测试提交，再提交报告与可复算证据 |

每个提交只包含对应功能、必要测试和文档；提交前复核 diff，不混入无关改动。性能开关的正式启用依赖其验收，不因单个热点 TTFT 显著下降就整体启用。新阶段完成后更新本表状态、源码提交和证据路径。

关闭策略开关时停止创建新的自动意图，撤回无真实需求的兴趣，释放本策略可释放的 claim；已在途传输继续由原 owner drain，普通 demand 及其他调用者的保护保持各自生命周期。关闭开关不能清空物理账本或直接释放内存。回退旧版本前确认新版本操作收尾与协议兼容；结果未知时记录并按恢复规则处理。

### 实施记录

以下为本轮新增实现，详细审阅与独立复测记录见 [PLAN-REVIEW-LOG.md](PLAN-REVIEW-LOG.md)。这些功能均已分别测试并提交，正式压力性能验收仍在进行。

| 阶段 | 已实现范围与提交 | 保留的边界 |
| --- | --- | --- |
| O0 | `6d102ac4b8`：普通/native query 与 GET 调用、耗时、anchor payload tokens/bytes、native job 和容量 gauge | wire bytes 为 null；没有新增完整物理 PUT、H2D 或跨 job 使用归因账本 |
| O1 | `acc163e3a0`：可选在途 token 预算、speculative 准入余量、真实 demand 优先及有界背景机会 | 普通 demand 保留原恢复路径；预算约束在途读取，不承诺限制所有已经发布且尚未消费的自然缓存；不抢占后端已提交 GET |
| O2 | `f5640888af`：按实际 worker/epoch/兼容布局/backend namespace/大小分组的准备成本分位、有限提前量、早到推进 | 成本包含控制与发布耗时，未单列负载档或学习预测概率；缺少样本时回退既有斜率 |
| O3a | `d4bade6739`：host-only 叶节点/合法后缀保存后定向回收；`0265dcc4b4`：真实 GPU/Mooncake K/V 逐元素恢复证明 | GPU 驻留重复副本、共享祖先、PIN 和在途引用继续阻止回收；不扩大 GPU 驱逐范围 |
| O4 | `8897e68356`：准备后一次有限 L2 PIN 子操作，有界保守预算、未知结果保留、共享兴趣清理 | 默认关闭；Router 重叠范围可能重复预留，Engine 仍按物理范围审批；未新增自动 L3 候选排序，已有显式 backend soft-pin 继续使用原 owner |
| O5 | `7e7962ffa0`：新 wire episode；`92e45056e6`：最小冷层驻留；`e35f78493b`：新鲜观测复用；`d44946b5de`：终态回执复用；`5a683b6667`、`61284379d0`：协商只读批量观察 | 保留当前 scope/claim、epoch 和反馈水位验证；旧 Engine 回退单查询；同 salt 和 GPU FIFO 安全约束保留 |
| O7 | `534f71fd20`、`57c87c2f60`、`ae25af0c99`：固定重放及不可变源复用；`26516932c9`：成本/资源复算；`bbecfd0a23`、`394ef5b481`：尚无回执与显式零命中计量 | 模拟 rollout；未接入真实 trainer，训练等待为 unknown；L3 竞争实验保留逐组发布；原工具异常证据不改写 |
| 工程检查 | `43aa45efaa`：精确 Rust 1.92.0 core 全目标 Clippy 与测试通过 | 不代替完整上游 CI 或未测试硬件/缓存模式的兼容性证明 |

可用配置以代码和对应 README 为准：Engine `cache_limits` 新增 `max_storage_read_inflight_tokens`、`max_speculative_storage_read_tokens`、`demand_host_reserve_tokens`、`storage_demand_priority`，默认均不启用限制/优先策略。前三项按 page 对齐的 anchor tokens 计量，不可填入 byte 数；当前 FULL pool 的 `native_host_bytes_per_token` 用于解释对应 payload 大小。已分配空间与未分配预留分开计数，取消后的预留等到物理 drain 才释放。

Router 新增可选 `preparation_policy`。省略该字段保留旧时机策略；设置后使用有界分组统计，`ready_pin_ttl_ms=0` 默认不申请自动保护。`auto_pin_max_tokens` 是该 Router 的保守保护预留上限，不能替代 Engine 的物理去重审批。`min_cold_residency_ms` 约束生命周期 host DEMOTE，显式三动作调用仍遵循原语义。O3b、O6、O1b 及完整未消费准备账本均未实现，是否扩展由压力证据决定。

只读 `read_batch` 版本 1.0 在单个 Engine owner turn 内按序执行最多八个既有查询，客户端必须先协商能力。`Events` 可推进 owner 状态，批量回复仍按原反馈 ticket、epoch 和水位验证；日志分页未追上或批内发生失效时撤销 readiness 并重新查询。该接口没有合并三动作，也不承诺全局原子快照。

压力筛选另发现预计恢复窗口会被 Router 排队及 worker 发现耗时延长，出现请求已开始恢复后再发 speculative GET 的情况。已实现 Router API 入队时冻结本地绝对窗口，按存活兴趣合并时限，过期且未尝试的准备安全退回；已尝试动作继续原 wire 身份收尾。独立 48 项 HTTP 回归及 50 项核心测试通过，正式性能仍待冻结配置后的评估。普通恢复发布后至 GPU 准入的保留窗口继续作为单独的优化候选；暂不扩大 GPU 或重复 host 副本的驱逐范围。

## 11. 正确性矩阵、复现入口与交付物

| 测试层 | 应覆盖的变化 |
| --- | --- |
| Engine CPU/可控存储 | `test_kv_control_transfers.py`、`test_kv_control_protection.py`、`test_kv_control_cache.py`、`test_kv_control_feedback.py`、`test_kv_control_service.py`，以及 storage lifecycle 用例；加入容量、部分回收、优先级、取消引用测试 |
| Rust 纯模块 | `sgl-kv-orchestrator` 的状态、backend protection 和协议 fixtures；新增策略 fake-clock、预算与模型更新测试 |
| Router 集成 | `sgl-router/tests/proxy/kv_orchestration.rs`；真实路由归属、生命周期、反馈 gap、旧 Engine 回退 |
| 实际 GPU/file/Mooncake | 真实保存、host 回收、恢复、共享 claim、取消和 worker 重启；输出与范围正确 |
| 实验工具 | 现有 `test_stress_bench.py`、`test_analyze_stress.py`、`test_router_bench.py`；新增失效预测、物理计数去重、异步收尾和 RL 有效样本统计 |

具体 Python 用例位于 `test/registered/unit/mem_cache/`，运行时使用仓库已有虚拟环境和设备约束；GPU 与 Mooncake 测试先读取已有测试配置，再使用本任务资源。不同层有针对性地运行；文件纯排版变更不启动 GPU 服务。

已有离线完整性检查可直接复用：

```bash
.venv/bin/python benchmark/session_kv_control/results/2026-11-04-mem08/verify_acceptance.py
```

已有压力分析入口：

```bash
.venv/bin/python benchmark/session_kv_control/analyze_stress.py \
  --results benchmark/session_kv_control/results/2026-11-04-mem08 \
  --output /tmp/session-kv-mem08-plan-recheck
```

第二个命令对历史数据退出码为 2，表示保留的一次 FAILED 和四次 DEFERRED 未满足每次预取均成功的严格路径校验；不能将其改为全成功。历史数据与校验文件保持不变，新结果使用独立目录。

每次正式评估交付：源码 hash 与 dirty 状态、全部开关/预算、模型/tokenizer/权重 namespace 身份、实际容量、Mooncake master/segment 状态、输入轨迹和随机调度、逐操作/请求/job 记录、资源采样、配对分析及不确定范围。保存 GET/PUT bytes 的采集层级和不可观测字段；证据分片与 SHA 校验沿用已有工具。同步保留初始状态准备、源发布、warmup、控制收尾和清理的耗时。

验收报告必须回答：哪个任务因何种动作减少了多少等待；为此增加多少搬运和 L2 保留；哪些场景约持平或退化；是否影响普通流量；是否提高有效 rollout 或减少训练等待。缺少因果证据的分项明确列为观察或假设。

## 12. 实现前需要固定的实验选择

下列事项有可执行默认方式，不构成当前文档交付的阻塞；实施时通过配置和 manifest 固定，并记录调整依据。

| 事项 | 默认实施方式 | 变化条件 |
| --- | --- | --- |
| 预测来源 | 应用显式剩余等待窗口；合成误差分布用于筛选 | 获得可重放真实工具轨迹后替换，保留来源证明 |
| 预算初值 | 以实测容量和典型缺口构造少量候选，影子模式观察后冻结 | O0 显示队列/容量瓶颈与初值不匹配 |
| L3 观测不足 | bytes/驻留缺口标 unknown；后端 API 数据量与 wire 数据分离 | 后端提供明确的计数器和保留能力 |
| 没有真实 trainer | 实现明确标注的 rollout/trainer 消费模拟，报告任务吞吐 | 获得实际训练接口后才能得出真实训练收益结论 |
| host duplicate 回收 | O3a 维持现有回收边界，O3b 独立实现 | O0 证明 GPU 重复副本是主要容量阻塞来源 |
| 同 salt 并行 | 保留串行约束，测试真实分布并量化损失 | O6b 的引用/范围设计与验证完成 |

关联材料：[原实现计划](PLAN.md)、[压力测试入口](benchmark/session_kv_control/STRESS_BENCH.md)、[Router 基准](benchmark/session_kv_control/ROUTER_BENCH.md)、[0.8 长测结果](benchmark/session_kv_control/results/2026-11-04-mem08/FINAL_ACCEPTANCE.zh-CN.md)、[三动作协议说明](sgl-router/sgl-kv-orchestrator/README.md)。
