# 7天 System Design 强化计划：低风险进化（Resiliency/复杂度管理/ROI决策）

## 核心理念

这不是"背 system design 题库"，而是把你现在很强的 E2E 交付，升级成面试官一听就知道你能当 owner 的那种：**可度量、可回归、可灰度、可回滚、可复盘**。

### 核心框架：Google SRE

- **SLO + Error Budget**：在"发功能 vs 保稳定"之间做控制
- **Canary + Rollback**：降低变更风险
- **可观测性 + Postmortem**：把系统变成"可自愈的免疫系统"

### 训练载体

用你最熟的一个系统（Rad-Linter 或 KYC/抽取）贯穿 7 天。

---

## 总目标（7 天后你要拿到的 6 个"面试可展示交付物"）

1. **三层指标卡（L0/L1/L2）**：稳定性/ROI/长期健康
2. **回归集 + Eval 门禁**：golden set + 通过阈值
3. **可观测性方案**：metrics/logs/traces + 关键 dashboard
4. **灰度发布 + 回滚策略**：feature flags + canary + rollback 条件
5. **保护策略矩阵**：限流/熔断/重试/降级/幂等（触发→动作→验证）
6. **Runbook + Postmortem 模板**：能扛 oncall 的证据

---

## 7天计划概览

| 天数 | 主题 | 交付物 | 关键产出 |
|------|------|--------|----------|
| Day 1 | 指标体系 | METRICS_CARD.md | 三层指标（L0/L1/L2）+ Error Budget Policy |
| Day 2 | 可观测性 | OBSERVABILITY.md | 观测设计 + Dashboard 草图 |
| Day 3 | 回归与门禁 | REGRESSION.md + EVAL_REPORT_TEMPLATE.md | Golden Set + 通过阈值 |
| Day 4 | 发布策略 | ROLLOUT_AND_ROLLBACK.md | Feature Flag + Canary + 回滚条件 |
| Day 5 | 保护与容错 | PROTECTION_MATRIX.md | 限流/熔断/重试/降级/幂等矩阵 |
| Day 6 | 事故响应 | RUNBOOK.md + POSTMORTEM.md | 可执行 runbook + 复盘模板 |
| Day 7 | 面试固化 | INTERVIEW_SCRIPT.md | 30秒/2分钟/5分钟话术 |

---

## 详细计划

### Day 1｜指标体系：把"成功"定义成可打分的三层指标（L0/L1/L2）

**为什么要练**：面试官要的不是"我很稳定"，而是"稳定怎么验收"。SLO/错误预算的意义就是把"可靠性 vs 创新速度"变成一个可执行的控制机制。

**目的**：你能把系统价值翻译成稳定性 + 业务收益 + 长期健康三层指标。

**目标**：产出一页《指标卡》。

**核心指标**：

- **L0 稳定性**（监控/可用/延迟）：
  - 成功率
  - p95/p99
  - 错误率
  - 回退率（fallback）
  - SLO

- **L1 进化收益**（ROI）：
  - 每单节省的人审分钟数
  - 错误拦截率带来的风险降低
  - 吞吐提升带来的成本节省（$ / request 或 tokens / request）

- **L2 长期健康**（可维护/可扩展）：
  - 变更失败率
  - 回滚频率
  - 回归门禁通过率
  - 告警噪音（precision）
  - toil（重复劳动）趋势

**Error Budget Policy**：预算烧太快就冻结发布转稳定性工作。

**交付物**：`METRICS_CARD.md`

---

### Day 2｜可观测性：把"我有监控"升级成"我能定位根因"

**为什么要练**：Senior 的价值不是"少出错"，而是"错了能快速发现/定位/止血"。OpenTelemetry 把 observability 讲成 traces/metrics/logs 三类信号。

**目的**：让你在 SD 面试里自然讲出"可观测闭环"。

**目标**：一页《观测设计 + dashboard 草图》。

**三类信号**：

- **Metrics**：
  - RPS
  - p95/p99
  - error
  - timeout
  - queue depth
  - 429
  - fallback
  - schema-fail

- **Logs**：结构化字段
  - request_id
  - model/prompt version
  - validator_fail_reason
  - latency breakdown

- **Traces**：一次请求分段
  - preprocess → OCR/VLM → LLM → validate → store
  - 能用 trace_id 把日志串起来

**交付物**：`OBSERVABILITY.md`

---

### Day 3｜回归与门禁：把 AI 系统变成"可回测工程"

**为什么要练**：你现在最容易被误判的点就是"能交付，但改动会不会把系统搞坏？"回归集 + 指标门禁是最硬的 senior 信号。

**目的**：让每次 prompt/模型/validator 改动都有 before/after 可证据化。

**目标**：建立最小黄金集 + 通过阈值（release gate）。

**核心内容**：

- 建 50–200 条 golden set（hard cases：模糊/遮挡/版式变化/长尾）
- 门禁指标：
  - schema pass rate
  - 字段级准确率/一致性
  - fallback 比例
  - 成本上限（tokens）
- 输出一页"回归报告模板"（改动必须跑）

**交付物**：`REGRESSION.md` + `EVAL_REPORT_TEMPLATE.md`

---

### Day 4｜发布：Feature Flag + Canary，把"上线"变成可控实验

**为什么要练**：Google SRE 明确讲：变更带来风险，canary 的目标是用小流量获得真实输入下的信心。Feature toggle/flag 的核心是部署与发布解耦，随时可回切。

**目的**：你能说清"怎么低风险进化"。

**目标**：写一份《灰度/回滚策略》（含阈值）。

**核心内容**：

- **Feature Flags**：按 model_version / prompt_version / validator_strictness 放量
- **Canary**：1% → 5% → 25% → 100%
  - 每步观察 L0/L1 指标（p95、error、schema-fail、fallback、$成本）
- **Rollback 条件**：写成明确阈值（比如 schema-fail×2 或 p95 +20% 立即回滚）

**交付物**：`ROLLOUT_AND_ROLLBACK.md`

---

### Day 5｜保护与容错：Engineering for Failure（免疫系统）

**为什么要练**：SRE/大厂不是追求"永不失败"，而是"失败可控、可恢复"。Error budget 也强调"发布速度 vs 稳定性"的现实权衡。

**目的**：把你提过的 rate limit/protection 讲成"策略→触发→动作→验证"。

**目标**：产出一张《保护矩阵》。

**必须覆盖**：

- **限流**：RPS + tokens/min + 并发
- **重试**：哪些可重试、指数退避、最大次数
- **熔断**：上游 OCR/LLM 抖动直接降级
- **幂等**：request_id 去重，避免重复计费/重复写库
- **降级**：小模型/规则/转人工

**如何验证**：压测看 p95、error、429、queue depth 的拐点。

**交付物**：`PROTECTION_MATRIX.md`

---

### Day 6｜事故响应：Runbook + Postmortem，把"出事"变成组织学习

**为什么要练**：Google SRE workbook 把 on-call/incident/postmortem 作为核心实践：出事不可怕，关键是快速止血 + 防复发。

**目的**：证明你是"交给你我能睡得着"的人。

**目标**：一份可执行 runbook + 一份复盘模板，并做一次桌面演练。

**核心内容**：

- **Runbook**：
  - 告警触发 → 先看哪个 dashboard
  - 什么条件回滚 → 什么条件降级
  - 怎么定位（trace → log）

- **Postmortem**：
  - 时间线
  - 根因/触发
  - 影响面
  - 行动项（owner + DDL）

**交付物**：`RUNBOOK.md` + `POSTMORTEM.md`

---

### Day 7｜面试固化：把系统讲成"低风险进化"的评审节奏

**为什么要练**：你不缺技术深度，缺的是"每次都能稳定输出 senior 评审口吻"。

**目的**：把前 6 天的交付物串成 30 秒 / 2 分钟 / 5 分钟三套话术。

**目标**：你能在 5 分钟内覆盖：目标→约束→设计→指标→失败模式→发布回滚→长期演进。

**固定结构**（背下来）：

1. Goal + 三层指标（L0/L1/L2）
2. 关键 trade-off（为什么这样设计）
3. Failure modes + 兜底
4. Flag + canary + rollback + 回归门禁（这就是"低风险进化"）

**交付物**：`INTERVIEW_SCRIPT.md`

---

## 核心理念总结

### SRE 的核心逻辑

**SLO 不要求 100% 满足**，要允许 error budget，用它来决定什么时候继续发布、什么时候停下来还债。

**Canary/Feature Flags 的意义**：让你能在真实流量下试错，但把伤害限制在很小范围，并能快速回滚。

**这一套不是在"学大厂流程"**，而是在把你的深技术变成"可托付的工业能力"。

---

## 开始学习

按照 Day 1 开始，每天完成一个交付物。所有文档都保存在 `system_design/` 目录下。