# SGLang Programmable KV Cache：与现有设计的差异

日期：2026-10-19。本文是设计决策与兼容性说明，未表示对应代码已经实现。代码核对基线为 SGLang `f02458fff1`、vLLM `ec5b1adbcd`；公开提案状态见 [调研记录](/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc/docs/programmable-kvc/research-2026-10-19.md)。

**本轮用户明确确认的三项决定**

1. 沿用较新 vLLM 方向：无公共 Pin 保证，采用 TTL 软优先级；默认执行显式命令，本地策略按需启用。
2. 显式 SET_PRIORITY 优先于 SGLang 既有 session 偏好；session 偏好可作为组合策略参与。在用请求与传输引用仍是不可绕过的正确性约束。
3. 公共基础声明采用副本级 SET_PRIORITY。副本失效后声明结束；长期任务偏好由可选策略重新提交，不把旧逻辑 Retain 一并复制为第二套公共基础语义。

其他接入细节属于 RFC 草稿的建议，不能反推为用户已逐项确认。

| 对照点 | 现有代码或提案 | 本 RFC 的选择与影响 |
| --- | --- | --- |
| 保留保证 | 旧 vLLM 附件与 SGLang #27574 包含 bounded Pin；SGLang #30796 是 Mooncake L3 lease POC | 公共契约只有可被压力淘汰的 priority。后端 lease 可单独声明，不通过 priority=100 或无限 ref 伪装 |
| session 与 priority | #29173 的 FULL 先比较 session_ref，再比较 base priority；SWA/Mamba 使用无 session 引用优先的扫描 | 新 retention profile 中显式 cache priority 优先，session 策略不得覆盖冲突的显式意图。旧模式在未启用新 profile 时保持原行为 |
| 声明寿命 | 新 vLLM 同时有 copy claim 与旧 logical Retain；#27970 在访问后延长 idle TTL | 公共 claim 绑定当前副本的 incarnation，TTL 从生效开始。命中不续期；分裂节点不重启 TTL；淘汰/重载不继承旧 claim |
| 多个声明者 | plain priority 常以一个 max 标量累计；#27970 保存一个较强元组 | 分别保存 claim_id、revision、value、expiry。合并有效优先级；清理一方不消除另一方，不把 A 的高值与 B 的长 TTL 拼接 |
| 推理调度 priority | SGLang 请求 priority 可被调度与缓存路径使用 | 新缓存动作拥有独立语义，不隐式修改请求队列。quota/score 通过显式 scheduler 扩展消费 |
| 公共输入 | #38891 已合入多 action 的 typed kv_hints；#24656/#36409 还有其他字段提案 | 复用 kv_hints，不新增竞争 envelope。原生动作和后端动作的解释归属按 action 注册，禁止同一动作重复执行 |
| 控制调用方 | #38891 主要面向选好 worker 后的可信 orchestrator | 草稿允许 router 和部署授权的直连应用使用独立控制；仍属于可信控制面，session_id 不作为权限凭证 |
| 生命周期 | 已有 request、StreamingSession、session radix reference，各自用途不同 | 长期 task/continuation 元数据独立于 streaming KV 所有权；无生成时也能提交控制，工具等待不结束 task |
| SessionPrefixIndex | 现有 tracker 保存当前组件可复用的 session frontier/leaf；BlockStored 标记触发者 | 复用 tracker 观察，补有界的逻辑 continuation/分支记录。内容身份、逻辑关联、驻留位置分别维护 |
| Prefetch | 现有 HiCache 主要按 Req 触发；#45128 讨论排队期间 SSD→存储 DRAM | 支持请求到达前的独立操作；结果明确准备到了存储 DRAM、引擎 host 还是 GPU。收到命令不能等同于 GPU 可复用 |
| Demote/MOVE | #27574 的概念示例允许冷层保存后让本地可淘汰 | MOVE 明确区分目标保存成功、源是否实际回收。有效引用或树结构约束阻止回收时报告 partial，不伪报释放容量 |
| 数据执行 | vLLM 通过 KV connector；SGLang 同时有 HiCache 和 direct linker | 用 SGLang 自己的 adapter 接入两类路径，保留真实 buffer/lock/worker 完成机制；不移植 vLLM 的 allocator 或 connector 类 |
| 混合模型 | FULL/SWA/MAMBA 的重用边界、池和淘汰规则不同 | 资源观察按池/组件表达；range 的实际覆盖与可恢复 checkpoint 单独报告。只有完整可用的必要状态才能成为命中 |
| Python/Rust | 已有 Rust TreeCore；session-aware 当前回退 Python | 共用 TreeCore 能力契约和语义测试。能力不足时明确回退或 unsupported，避免每个后端各造一套控制逻辑 |
| 事件 | 已有 stored/removed、session 归因、bounded replay；完整快照相关 PR 仍 Open | 分开操作结果、关联变化和驻留变化；重启/缺口恢复继续复用现有事件路径，不能把触发 session 当作完整 owner set |

**显式优先级与 session 偏好的具体例子**

在 FULL 组件当前排序下，A 无 session 引用、priority=90，B 有 session 引用、priority=0；A 会更早成为候选。新 profile 要支持调用方通过显式优先级保留 A。这里比较的是同一资源池内、满足树结构约束的可淘汰候选；正在被请求或 DMA 使用的内容不参与这个比较。

迁移不能只改 node.priority。FULL 的 session tuple 和 SWA/Mamba 的两轮扫描都要接入统一的优先级决策。session 作为可选策略时，其建议要经过 composer；冲突的策略建议不能改变已接受的显式命令。独立显式声明之间继续按最大有效优先级合并。无有效差异的候选使用 LRU，组件之间为维护可复用状态而存在的级联约束继续由缓存实现执行。

此选择会改变启用新 profile 后的淘汰顺序，属于需要公开说明的行为变化。未启用时沿用 SGLang 原有策略。RFC 不声称两种模式在有指令时结果完全相同。

**副本级 claim 的具体例子**

A 对 engine E 的 GPU 副本 C 设置 priority=90、TTL=10s；B 对 C 设置 priority=20、TTL=60s。前 10 秒取 90，之后取 20；A 的到期不清除 B。

若 C 在第 5 秒因压力被淘汰，C 的声明随其 incarnation 结束。第 7 秒从 Mooncake 重载的是新副本 C2，不能自动继承 C 的声明。可选 task policy 可以针对 C2 提交新 claim，并明确新的 TTL；这一步会留下独立的动作和结果。旧 operation 的迟到 completion 或重试不得操作 C2。

树节点切分通常没有改变物理副本。切分后声明按实际覆盖区间分配，保持原 expiry 和 revision；不能因内部结构变化延长寿命。节点合并也不能将不同声明的范围和时限混为一个标量。实现可以使用旁路区间记录，无须强制每个 token 一个节点。

**需由实现和评审进一步确定的细节**

- 新 profile 与旧 session-aware 配置的组合/迁移方式，以及 session 策略如何映射为数值建议。
- page 边界的 exact/clip 规则、SWA window 与 Mamba checkpoint 的目标覆盖报告。
- 独立控制的 HTTP/gRPC 路径、操作记录上限和结果保留期限。
- HiCache、buffer-only、direct linker、Python/Rust、TP/PP/CP 的能力与验证矩阵。
- action 命名与 #36224 既有 kv.* 提案的协商；草稿使用 sglang.* 表示原生解释方，不宣称跨引擎名称已标准化。

以上细节不改变已确认的软优先级、显式命令优先和副本级声明语义。

**本轮展开的设计约束（RFC 草稿建议）**

- envelope 内的 action 列表不隐含原子事务或执行依赖。COPY 完成后再设置 priority 的关系由明确的 plan 表达；不能仅凭列表顺序假定前一个动作已经完成。
- range 解析分别报告请求范围、实际操作范围和恢复所需的 prefix/window/checkpoint 依赖。页边界不足时明确裁剪或返回不支持，不静默扩大保护范围。
- 独立控制能力按 action、方向、组件、并行布局和完成阶段公布。direct linker 能为活跃请求加载 KV，不自动代表它能在无请求时建立可查询的 GPU 缓存副本。
- 操作结果、共享传输任务和物理 buffer 分别拥有生命周期。单个调用方结束或取消后，其他调用方及 DMA 仍可能持有资源；结果记录清理不能提前释放这些资源。
- 资源预留使用既有 pool。已转为真实分配的 reservation 不重复扣减 free capacity；尚未分配的逻辑预留则必须影响后续批准。
- replay 与结果缓存都有界。保留窗口到期后不承诺无限期的恰好一次执行；迟到结果还需经过 engine epoch、操作实例与副本代次检查。

这些补充用于将职责说明落实为可验证的操作契约，具体字段与实现仍需评审。RFC 同时加入 TokenCake 的优化前后实验、消融及资源充足时的退化对照，用于说明哪些接口需求已有实验动机，哪些仍需 SGLang 实现验证。
