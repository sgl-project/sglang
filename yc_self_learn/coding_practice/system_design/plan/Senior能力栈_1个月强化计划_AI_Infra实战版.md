# Senior 能力栈：1个月强化计划（AI Infra 实战版）

**目标**：从 L3（0-1 经验）→ L4（1-1000 经验）→ L5（服务级 owner）  
**标准**：大厂 AI Infra / LLMOps / Model Serving 岗位的真实能力栈（L5，8-12年工作经验）  
**载体**：用你的核心项目（KYC、SGLang）实战训练

**你的背景**：创业公司 0-1 的研发和开发经验，缺少 1-1000 的系统化工程能力（测试、可维护性、可观测性等）

---

## 🎯 你的能力评估（基于你的实际情况）

### 你的能力水平：**L3（0-1 经验）→ 目标：L4（1-1000 经验）→ 最终目标：L5（服务级 owner）**

**L3 定义**：能独立完成功能开发，能写代码，能上线，但缺少系统化的工程能力

**L4 定义**：组件级 owner（端到端交付一个功能/模块，能写小 design、能上线、能扛 oncall/指标、**能建立测试和可维护性体系**）

**L5 定义**：服务级 owner（能主导一个 service/子系统的 design，协调多人）

---

### 你当前的能力（L3 水平，0-1 经验）

**已有能力（0-1 阶段）**：
- ✅ **功能开发能力**：能独立完成功能开发（KYC 项目、SGLang 项目）
- ✅ **快速迭代能力**：能快速上线、快速迭代（创业公司经验）
- ✅ **问题解决能力**：能解决技术问题，能写代码
- ✅ **知识管理能力**：建立了完整的 glossary 系统（分层字典、文件契约），说明有系统化思维
- ✅ **学习策略**：知道"最小闭环"的重要性，不是"全部"，而是"可执行"
- ✅ **高级概念理解**：知道 A/B Test、SLO/Error Budget、Trade-off、验证方法等（概念层面）
- ✅ **结构化思维**：能提出"什么时候做"、"为什么做"、"怎么验证"等战略性问题

**缺少的能力（1-1000 阶段，L3 → L4 的差距）**：
- ⚠️ **测试体系**：没有系统化的测试（Unit Tests、Integration Tests、E2E Tests、Golden Set）
- ⚠️ **回归门禁**：没有回归门禁机制（改动可能把系统搞坏）
- ⚠️ **可维护性**：没有代码复杂度检查、代码重复率检查、可维护性指标
- ⚠️ **可观测性**：可能做过但没系统化（Metrics/Logs/Traces 的完整体系）
- ⚠️ **SLO/Error Budget**：可能知道概念，但没有实际建立 Error Budget 机制
- ⚠️ **Canary/Rollback**：可能知道概念，但没有实际实现 Feature Flag 和自动回滚
- ⚠️ **系统化表达**：如何用 6 个固定模块讲清楚 System Design（面试需要）

**你可能做过但没意识到的事情**：
- ✅ **可能做过测试**：但可能只是手动测试，没有系统化的测试金字塔
- ✅ **可能做过监控**：但可能只是简单的日志，没有结构化的 Metrics/Logs/Traces
- ✅ **可能做过错误处理**：但可能只是 try-catch，没有系统化的错误分类和重试机制
- ✅ **可能做过部署**：但可能只是手动部署，没有 Canary 发布和自动回滚

**学习风格**：
- 偏好"最小闭环"而非"全部"
- 注重可执行性和结构化
- 有知识管理意识（glossary 系统）
- **需要系统化**：把"做过但没意识到"的事情系统化

---

### 0-1 → 1-1000 的核心转变

| 维度 | 0-1 阶段（L3） | 1-1000 阶段（L4） |
|------|---------------|------------------|
| **测试** | 手动测试、临时测试 | 测试金字塔（Unit/Integration/E2E）、回归门禁 |
| **可维护性** | 代码能跑就行 | 代码复杂度检查、可维护性指标、Code Review 标准 |
| **可观测性** | 简单日志 | 结构化的 Metrics/Logs/Traces、SLO/Error Budget |
| **发布** | 直接发布 | Canary 发布、自动回滚、Feature Flag |
| **错误处理** | try-catch | 系统化的错误分类、重试机制、降级策略 |
| **稳定性** | "系统很稳" | SLO/Error Budget、监控告警、Oncall 流程 |

**关键转变**：
- 从"能跑就行"到"系统化工程能力"
- 从"手动测试"到"自动化测试体系"
- 从"简单日志"到"可观测性体系"
- 从"直接发布"到"低风险发布流程"

---

### L4 → L5 的核心转变

| 维度 | L4（组件级 owner） | L5（服务级 owner） |
|------|-------------------|-------------------|
| **Scope** | 一个功能/模块 | 一个 service/子系统 |
| **Design** | 小 design（组件级） | 大 design（服务级，协调多人） |
| **交付** | 端到端交付一个功能 | 主导一个 service 的设计和交付 |
| **协调** | 自己写代码 | 协调多人完成复杂项目 |
| **Oncall** | 扛 oncall/指标 | 设计 oncall/指标体系 |
| **决策** | 执行决策 | 主导决策（Trade-off、验证方法选择） |

**关键转变**：
- 从"执行者"到"主导者"
- 从"写代码"到"协调多人"
- 从"组件级"到"服务级"
- 从"执行决策"到"主导决策"

---

## 🎯 核心转变

### Junior/Mid → Senior 的鸿沟

**不再是**：
- ❌ 被动接任务的"码农"
- ❌ "我写了 1000 行代码"
- ❌ "我的系统很稳"

**而是**：
- ✅ 能够承载业务确定性的"技术 Owner"
- ✅ "我降低了团队 50% 的联调时间"
- ✅ "我的错误预算（Error Budget）是多少"

### 9 根支柱（面试官抽样验证）

按"面试最常抽查、且你最容易被误判缺失"的维度：

1. **可运营与可靠性**（SRE 思维）：SLO/Error Budget + Canary/Rollback
2. **可观测性与线上排障**（Metrics/Logs/Traces）：坏了能发现、能定位、能止血
3. **测试与回归文化**（测试金字塔/回归门禁）：改动不会把系统搞坏
4. **Code Review 与可维护性**（把复杂变简单）：代码库更健康、更易维护
5. **交付能力与工程效率指标**（DORA / Four Keys）：跑得快还不翻车
6. **影响力与带人**（Senior/Staff 的分水岭）：放大器能力——让别人/别的团队也变强
7. **需求澄清 + Trade-off + 决策 ROI**（产品与工程的翻译）：为什么做、选 A 不选 B、怎么验证收益
8. **安全与隐私**（Threat Model / PII 处理）：数据/权限/审计/泄露风险
9. **写作与设计评审**（Design Doc 能力）：结构化写清楚假设、方案、风险、迁移与回滚

---

## 🎯 你到底要补什么（按重要性排序）

### 核心 = System Design 面试最小闭环（6 个固定模块）

**不是"全部"，而是"最小闭环"——每道题都按这个顺序讲，练到"闭眼也能讲"**

#### System Design 面试框架（37 分钟标准流程）

**0) 目标与指标（2 分钟）**

- DAU/QPS、延迟（p95/p99）、一致性要求、数据规模、可用性目标（99.9/99.99）
- **明确"最重要的 KPI 是什么"**（延迟？成本？正确性？）

**1) API 与数据模型（5 分钟）**

- 3–5 个核心接口（create/get/list/update）
- 2–3 张核心表/集合（主键、索引、分区键）

**2) 高层架构图（5 分钟）**

- Client → API Gateway → Service → DB/Cache/Queue
- **先画"单体可跑版本"，别一上来微服务宇宙**

**3) 热点与扩展（10 分钟）**

- 缓存策略（读多写少？写多读少？）
- 分库分表/分区（按 user_id / time）
- 异步化（队列、worker）、回压

**4) 可靠性（10 分钟）**

- 幂等（idempotency key）
- 重试/超时/熔断/限流
- 数据一致性（强一致 or 最终一致）、补偿

**5) 可观测与演进（5 分钟）**

- 监控：QPS、错误率、延迟、队列堆积、DB 慢查询
- 迭代：先 MVP，再加功能/扩展

**目标**：把这 6 个模块练到"闭眼也能讲"，超过很多"工作多年但讲不清楚"的人。

---

### 主线 = L3 → L4 → L5 核心能力（1-1000 工程能力 + 决策框架 + 服务级 Design）

**不是"研究技术本身"，而是"如何建立系统化的工程能力、如何主导决策、如何做服务级 Design"**

#### 1. 1-1000 工程能力（L3 → L4 核心，你缺少的）

**核心问题**：如何从"能跑就行"到"系统化工程能力"？

**能力要求**：
- ✅ **测试体系**：测试金字塔（Unit/Integration/E2E）、回归门禁
- ✅ **可维护性**：代码复杂度检查、可维护性指标、Code Review 标准
- ✅ **可观测性**：结构化的 Metrics/Logs/Traces、SLO/Error Budget
- ✅ **低风险发布**：Canary 发布、自动回滚、Feature Flag
- ✅ **错误处理**：系统化的错误分类、重试机制、降级策略

**实战动作**：
- ✅ 在 KYC 项目中建立完整的测试体系（基于 SGLang 的测试结构）
- ✅ 在 KYC 项目中建立可维护性指标（代码复杂度、重复率检查）
- ✅ 在 KYC 项目中建立可观测性体系（基于 `_summary.json` 的实际数据）
- ✅ 在 KYC 项目中实现 Canary 发布和自动回滚（基于实际 Feature Flag 需求）

#### 2. 决策框架与验证方法（L4 → L5 核心）

**核心问题**：如何主导决策？如何验证决策？

**能力要求**：
- ✅ 能选择验证方法（A/B Test、行业基准、历史数据等）
- ✅ 能做 Trade-off 决策（准确性 vs 成本、功能 vs 稳定性）
- ✅ 能量化 ROI（投入、收益、回报）

**实战动作**：
- ✅ 为 KYC 项目的每个设计决策选择验证方法（基于实际项目文档）
- ✅ 写 Trade-off 分析文档（为什么选 A 不选 B）
- ✅ 量化 ROI（投入 4 人月，节省 $7500/月）

#### 3. 服务级 Design 能力（L5 核心）

**核心问题**：如何主导一个 service/子系统的 design？

**能力要求**：
- ✅ 能写大 design（不只是小 design）
- ✅ 能协调多人完成复杂项目
- ✅ 能处理跨团队依赖
- ✅ 能说清楚 Trade-off 和决策理由

**实战动作**：
- ✅ 为 KYC 项目写完整的服务级 Design Doc（基于实际代码结构）
- ✅ 设计跨团队协作方案（Schema 契约、技术选型）
- ✅ 协调多人完成复杂功能（Feature Flag、Canary 发布）

**能力要求**：
- ✅ 能选择验证方法（A/B Test、行业基准、历史数据等）
- ✅ 能做 Trade-off 决策（准确性 vs 成本、功能 vs 稳定性）
- ✅ 能量化 ROI（投入、收益、回报）

**实战动作**：
- ✅ 为 KYC 项目的每个设计决策选择验证方法
- ✅ 写 Trade-off 分析文档（为什么选 A 不选 B）
- ✅ 量化 ROI（投入 4 人月，节省 $7500/月）

---

## 📅 1个月训练计划（基于 KYC + SGLang 实际项目）

**重点**：从 0-1 到 1-1000 的转变，系统化建立工程能力

---

### Week 0｜System Design 面试框架 + 决策框架（前置）

**目标**：掌握 System Design 面试的"最小闭环"框架 + 决策框架

---

#### Day 0：System Design 面试框架实战（KYC 项目）

**实战动作**：

1. **用 KYC 项目练习 6 个固定模块**

   **0) 目标与指标（2 分钟）**
   - **基于实际项目**：`KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`
   - **DAU/QPS**：KYC 系统每天处理 10,000 个文档（QPS ≈ 0.12）
   - **延迟**：p95 < 15s，p99 < 30s（SLO）——**基于 `_summary.json` 的 `p95_latency_ms`、`p99_latency_ms`**
   - **一致性要求**：最终一致性（允许短暂延迟）
   - **数据规模**：每天 10,000 文档 × 365 天 = 365 万文档/年
   - **可用性目标**：99.9%（月度 Error Budget = 1%）——**基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`**
   - **最重要的 KPI**：**准确性**（强监管要求）> 延迟 > 成本

   **1) API 与数据模型（5 分钟）**
   - **基于实际项目**：KYC 项目的实际 API 和 `_summary.json` 结构
   - **核心接口**：
     - `POST /api/v1/kyc/process`（处理单个文档）——**基于实际 pipeline**
     - `GET /api/v1/kyc/status/{request_id}`（查询处理状态）——**基于 `_summary.json` 的 `results[]`**
     - `GET /api/v1/kyc/batch/{batch_id}`（查询 batch 结果）——**基于 `_summary.json` 的 `batch_id`**
   - **核心表**：
     - `kyc_requests`（主键：`request_id`，索引：`batch_id`，分区：按 `created_at` 按月分区）——**基于 `_summary.json` 的 `results[].file_id`**
     - `kyc_results`（主键：`request_id`，索引：`status`，分区：按 `created_at` 按月分区）——**基于 `_summary.json` 的 `results[].status`**

   **2) 高层架构图（5 分钟）**
   - **基于实际项目**：KYC 项目的实际架构（`src/pipeline.py`、`src/fw_client.py`、`src/validators.py`）
   ```
   Client → API Gateway → KYC Service → DB/Cache/Queue
                                    ↓
                              Fireworks API
                                    ↓
                              Validation Service
   ```
   - **单体可跑版本**：所有服务在一个进程内（`main.py`）——**基于实际项目结构**
   - **扩展版本**：API Gateway（Nginx）→ KYC Service（Python）→ DB（PostgreSQL）+ Cache（Redis）+ Queue（RabbitMQ）

   **3) 热点与扩展（10 分钟）**
   - **基于实际项目**：KYC 项目的实际扩展需求
   - **缓存策略**：读多写少 → Redis 缓存 `request_id` → `result`（TTL = 1 小时）——**基于 `_summary.json` 的查询需求**
   - **分库分表**：按 `batch_id` 分表（每个 batch 一个表，避免单表过大）——**基于 `_summary.json` 的 `batch_id`**
   - **异步化**：使用队列（RabbitMQ）处理 batch，worker 并发处理（ThreadPoolExecutor）——**基于实际 `main.py` 的并发处理**
   - **回压**：队列满时，阻塞新请求（Rate Limiter）——**基于实际 rate limiter**

   **4) 可靠性（10 分钟）**
   - **基于实际项目**：KYC 项目的实际错误处理（`src/errors.py`、`KYC_Day01_A1_B4_retry_error_rate.md`）
   - **幂等**：使用 `idempotency_key`（`request_id`）去重——**基于 `_summary.json` 的 `file_id`**
   - **重试/超时/熔断/限流**：
     - 重试：指数退避（3 次，最大 60s）——**基于 `KYC_Day01_A1_B4_retry_error_rate.md`**
     - 超时：API 调用超时 60s——**基于实际 `src/fw_client.py`**
     - 熔断：连续 5 次失败，熔断 30s
     - 限流：QPS = 10（Rate Limiter）——**基于实际 rate limiter**
   - **数据一致性**：最终一致性（允许短暂延迟）
   - **补偿**：失败后自动重试，重试失败后转人工审核——**基于 `_summary.json` 的 `status: "fail"`**

   **5) 可观测与演进（5 分钟）**
   - **基于实际项目**：KYC 项目的实际监控（`_summary.json`、`KYC_Day01_A2_指标计算脚本示例.md`）
   - **监控**：
     - QPS：`requests_total`（Prometheus）——**基于 `_summary.json` 的 `summary.total_processed`**
     - 错误率：`error_rate`（< 1% SLO）——**基于 `_summary.json` 的 `summary.fail_count / summary.total_processed`**
     - 延迟：`p95_latency`、`p99_latency`（< 15s SLO）——**基于 `_summary.json` 的 `summary.p95_latency_ms`、`summary.p99_latency_ms`**
     - 队列堆积：`queue_size`（< 100）
     - DB 慢查询：`db_slow_queries`（< 10ms）
   - **迭代**：
     - MVP：单体版本（`main.py`）——**基于实际项目结构**
     - 扩展：微服务化（API Gateway + KYC Service + Validation Service）
     - 优化：缓存、分库分表、异步化

2. **准备面试表达模板**（见原文件 Day 0 内容）

**输出**：
- ✅ `KYC_SYSTEM_DESIGN_INTERVIEW.md`（6 个模块的完整设计，基于实际项目）
- ✅ `KYC_ARCHITECTURE_DIAGRAM.md`（架构图，Mermaid 格式，基于实际代码结构）
- ✅ 面试表达模板（每个模块的固定话术，基于实际项目数据）

---

#### Day 0.5：验证方法选择决策框架（基于 KYC 项目实际内容）

**实战动作**：

1. **创建"验证方法选择决策树"**

   - **创建**：`docs/VALIDATION_METHOD_DECISION_TREE.md`
   - **基于实际项目**：`KYC_Day01_A1_B8_C3_validation_methods_beyond_ab_test.md`
   - **内容**：
     ```
     问题：如何验证设计决策？
     
     决策树（基于 KYC 项目实际场景）：
     1. 有历史数据吗？（基于 `_summary.json`）
        - 是 → 历史数据分析（分析过去 3 个月的 `_summary.json`）
        - 否 → 行业基准（参考 Google p95 < 10s，Amazon p95 < 15s）
     
     2. 需要量化影响吗？
        - 是 → A/B Test（验证 p95 阈值 = 15s 是否会导致流失率增加）
        - 否 → 监控数据验证（基于 `_summary.json` 的实时监控）
     
     3. 需要验证容错吗？
        - 是 → 混沌工程 + 压力测试（验证降级策略）
        - 否 → 跳过
     
     4. 需要降低发布风险吗？
        - 是 → 金丝雀发布（基于 Feature Flag）
        - 否 → 直接发布
     ```

2. **为 KYC 项目的每个设计决策选择验证方法**

   - **创建**：`docs/KYC_VALIDATION_METHOD_SELECTION.md`
   - **基于实际项目**：`KYC_Day01_A1_B8_ab_testing_validation.md`、`KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`
   - **内容**：
     | 设计决策 | 验证方法 | 理由 | 实际数据来源 |
     |---------|---------|------|------------|
     | p95 阈值 = 15s | 行业基准 + A/B Test | 初始设计用行业基准，验证用 A/B Test | `_summary.json` 的 `p95_latency_ms` |
     | Error Rate 阈值 = 2% | 历史数据 + 监控数据 | 有历史数据（`_summary.json`），持续监控验证 | `_summary.json` 的 `summary.fail_count` |
     | 降级方案选择 | A/B Test + 金丝雀发布 | 需要量化影响，降低发布风险 | `KYC_Day01_A1_B9_C1_fallback_strategy_types.md` |
     | 系统容错能力 | 混沌工程 | 需要验证容错设计 | `KYC_Day01_A1_B6_layered_fault_tolerance.md` |

3. **准备面试表达模板**

   ```
   "我在设计 KYC 系统时，为每个设计决策都选择了合适的验证方法。
   
   例如，p95 阈值 = 15s 的决策：
   - 初始设计：使用行业基准（Google p95 < 10s，Amazon p95 < 15s）
   - 验证方法：A/B Test（验证用户等待时间 > 15s 是否会导致流失率增加）
   - 持续监控：监控数据验证（基于 `_summary.json` 的 `p95_latency_ms`，p95 从 10s 增加到 18s，触发 Warning）
   
   这样确保每个设计决策都有数据支撑，而不是凭经验猜测。"
   ```

**输出**：
- ✅ `docs/VALIDATION_METHOD_DECISION_TREE.md`（验证方法选择决策树，基于实际项目）
- ✅ `docs/KYC_VALIDATION_METHOD_SELECTION.md`（KYC 项目的验证方法选择，基于实际文档和数据）
- ✅ 面试表达模板（如何选择验证方法，基于实际项目）

---

#### Day 0.6：Trade-off 决策框架（基于 KYC 项目实际内容）

**实战动作**：

1. **创建"Trade-off 决策框架"**

   - **创建**：`docs/TRADE_OFF_DECISION_FRAMEWORK.md`
   - **基于实际项目**：`KYC_Day01_A1_B5_validation_tradeoff.md`、`KYC_Day01_A1_B9_C1_D1_fallback_design_timing.md`
   - **内容**：
     ```
     Trade-off 决策框架（基于 KYC 项目）：
     
     1. 明确 Trade-off 的维度
        - 准确性 vs 成本（基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md` 的成本分析）
        - 延迟 vs 成本（基于 `_summary.json` 的 `latency_ms` 和 `cost_usd`）
        - 功能 vs 稳定性（基于 `KYC_Day01_A1_B9_C1_D1_fallback_design_timing.md` 的产品阶段）
     
     2. 评估每个维度的优先级
        - 业务优先级（强监管：准确性优先）——**基于 KYC 项目的实际业务需求**
        - 产品阶段（MVP：功能优先；Scale：成本优先）——**基于 KYC 项目的 PoV 阶段**
        - 风险承受度（低风险：保守方案）
     
     3. 量化 Trade-off
        - 方案 A：准确性 99%，成本 $0.01/request（基于 `_summary.json` 的 `cost_usd`）
        - 方案 B：准确性 95%，成本 $0.001/request
        - 决策：选择方案 A（强监管要求准确性优先）
     
     4. 验证 Trade-off 决策
        - 用 A/B Test 验证决策是否正确（基于 `KYC_Day01_A1_B8_ab_testing_validation.md`）
        - 持续监控，根据数据调整（基于 `_summary.json` 的实时监控）
     ```

2. **为 KYC 项目写 Trade-off 分析文档**

   - **创建**：`docs/KYC_TRADE_OFF_ANALYSIS.md`
   - **基于实际项目**：`KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`、`KYC_Day01_A1_B5_validation_tradeoff.md`
   - **内容**：
     | Trade-off | 方案 A | 方案 B | 决策 | 理由 | 实际数据 |
     |-----------|--------|--------|------|------|---------|
     | 准确性 vs 成本 | 大模型（99%，$0.01） | 小模型（95%，$0.001） | 方案 A + 降级 | 强监管要求准确性优先 | `_summary.json` 的 `cost_usd` |
     | 延迟 vs 成本 | 实时处理（p95=10s，高成本） | 批处理（p95=30s，低成本） | 方案 A | 用户体验优先 | `_summary.json` 的 `p95_latency_ms` |
     | 功能 vs 稳定性 | 快速迭代（功能多，稳定性低） | 稳定迭代（功能少，稳定性高） | 方案 B | 稳定期优先稳定性 | `KYC_Day01_A1_B9_C1_D1_fallback_design_timing.md` |

3. **准备面试表达模板**

   ```
   "我在 KYC 项目中做了多个 Trade-off 决策。
   
   例如，准确性 vs 成本的 Trade-off：
   - 方案 A：大模型（准确性 99%，成本 $0.01/request，基于 `_summary.json` 的 `cost_usd`）
   - 方案 B：小模型（准确性 95%，成本 $0.001/request）
   - 决策：选择方案 A + 降级策略
   - 理由：KYC 是强监管领域，准确性优先，但用降级策略控制成本
   - 验证：用 A/B Test 验证降级策略的效果（基于 `KYC_Day01_A1_B8_ab_testing_validation.md`）
   
   这样确保每个 Trade-off 决策都有明确的理由和数据支撑。"
   ```

**输出**：
- ✅ `docs/TRADE_OFF_DECISION_FRAMEWORK.md`（Trade-off 决策框架，基于实际项目）
- ✅ `docs/KYC_TRADE_OFF_ANALYSIS.md`（KYC 项目的 Trade-off 分析，基于实际文档和数据）
- ✅ 面试表达模板（如何做 Trade-off 决策，基于实际项目）

---

### Week 1｜可运营闭环（SLO + 可观测 + 低风险发布）- **1-1000 核心能力**

**目标**：从"能跑就行"到"系统化工程能力"，建立"把 LLM 变成可控系统"的思维

**重点**：你可能做过但没系统化的事情，现在要系统化建立

---

#### Day 1-2：SLO + Error Budget（基于 KYC 项目实际内容）

**实战动作**：

1. **在 KYC 项目中定义 SLO 和 Error Budget**

   - **打开**：`KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`、`KYC_Day01_A2_指标计算脚本示例.md`
   - **基于实际数据**：`_summary.json` 的 `summary.success_count`、`summary.fail_count`
   - **定义 SLO**：
     - 成功率：99%（SLO）——**基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md` 的 "成功率：95%（PoV 阶段），SLO 目标：99%（Production）"**
     - p95 延迟：< 15 秒（SLO）——**基于 `_summary.json` 的 `summary.p95_latency_ms`**
     - 错误率：< 1%（SLO）——**基于 `_summary.json` 的 `summary.fail_count / summary.total_processed`**
   - **计算 Error Budget**：
     - Error Budget = 100% - SLO = 1%（成功率）
     - 月度 Error Budget = 1000 文档中允许 10 个失败——**基于实际 `_summary.json` 的数据**
   - **评估当前状态**：
     - 成功率：95%（PoV 阶段）——**基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`**
     - Error Budget 消耗：500%（严重超标）
     - **决策**：冻结发布，专注稳定性修复

2. **在代码中实现 Error Budget 检查**

   - **创建**：`src/error_budget.py`
   - **基于实际项目**：`_summary.json` 的结构和 `KYC_Day01_A2_指标计算脚本示例.md`
   - **实现**：
     ```python
     class ErrorBudgetPolicy:
         def __init__(self, slo_percent=99.0):
             self.slo = slo_percent
             self.error_budget = 100.0 - slo_percent
         
         def can_release(self, current_success_rate):
             """基于 Error Budget 决定是否允许发布"""
             # 基于 _summary.json 的 summary.success_count / summary.total_processed
             budget_remaining = current_success_rate - self.slo
             if budget_remaining > 50:
                 return "normal"  # 可以快速发布
             elif budget_remaining > 25:
                 return "warning"  # 限制高风险发布
             else:
                 return "freeze"  # 冻结发布
     ```
   - **集成**：在 `main.py` 中调用，每次 batch 后检查（基于实际 `_summary.json` 的输出）

3. **在 `_summary.json` 中记录 Error Budget 状态**

   - **修改**：`src/io_utils.py`（基于实际项目结构）
   - **添加**：Error Budget 状态字段到 `_summary.json` 的 `summary` 部分

**输出**：
- ✅ `src/error_budget.py`（Error Budget Policy 实现，基于实际 `_summary.json` 结构）
- ✅ `KYC_SLO_ERROR_BUDGET.md`（SLO 定义、Error Budget 计算、决策机制，基于实际数据）
- ✅ 更新 `_summary.json` 结构（包含 Error Budget 状态）

**面试表达模板**：
```
"我为 KYC 项目设定了 99% 的成功率 SLO，并建立了 Error Budget 机制。
当错误预算剩余 > 50% 时，我们继续快速发布；当 < 25% 时，我们冻结发布，专注稳定性修复。
我在代码中实现了 Error Budget Policy，每次 batch 后自动检查（基于 `_summary.json` 的 `summary.success_count`），决定是否允许发布。"
```

---

#### Day 2：Canary 发布 + 自动回滚（基于 KYC 项目实际内容）

**实战动作**：

1. **在 KYC 项目中实现 Feature Flag 机制**

   - **创建**：`src/feature_flags.py`
   - **基于实际项目**：KYC 项目的实际需求（模型版本、prompt 版本、validator 严格度）
   - **实现**：
     ```python
     class FeatureFlags:
         def __init__(self):
             self.flags = {
                 "model_version": {"enabled": False, "percentage": 0},
                 "prompt_version": {"enabled": False, "percentage": 0},
                 "validator_strictness": {"enabled": False, "percentage": 0}
             }
         
         def should_use_feature(self, flag_name, request_id):
             """基于百分比决定是否使用新特性"""
             flag = self.flags.get(flag_name, {})
             if not flag.get("enabled"):
                 return False
             # 简单的哈希取模决定百分比
             hash_val = hash(request_id) % 100
             return hash_val < flag.get("percentage", 0)
     ```
   - **集成**：在 `src/pipeline.py` 中调用，根据 Feature Flag 选择模型/prompt（基于实际 pipeline 结构）

2. **实现 Canary 发布的观察和回滚机制**

   - **创建**：`src/canary_monitor.py`
   - **基于实际项目**：`_summary.json` 的 `summary.schema_fail_rate`、`summary.p95_latency_ms`、`summary.fail_count`
   - **实现**：
     ```python
     class CanaryMonitor:
         def check_canary_metrics(self, old_metrics, new_metrics):
             """检查 Canary 指标是否达标"""
             # 基于 _summary.json 的 summary.schema_fail_rate
             if new_metrics["schema_fail_rate"] > old_metrics["schema_fail_rate"] * 2:
                 return "rollback"  # Schema Fail × 2 立即回滚
             
             # 基于 _summary.json 的 summary.p95_latency_ms
             if new_metrics["p95_latency"] > old_metrics["p95_latency"] * 1.2:
                 return "rollback"  # p95 + 20% 立即回滚
             
             # 基于 _summary.json 的 summary.fail_count
             if new_metrics["error_rate"] > 0.05:
                 return "rollback"  # Error Rate > 5% 立即回滚
             
             return "continue"  # 继续 Canary
     ```
   - **集成**：在 batch processing 后调用，自动决定是否回滚（基于实际 `_summary.json` 的输出）

3. **实现自动回滚机制**

   - **创建**：`src/rollback.py`
   - **实现**：
     ```python
     class RollbackManager:
         def rollback_feature(self, flag_name):
             """回滚 Feature Flag 到 0%"""
             flag = self.flags.get(flag_name, {})
             flag["percentage"] = 0
             flag["enabled"] = False
             # 保存到配置文件
             self.save_flags()
     ```
   - **集成**：在 Canary Monitor 触发回滚时自动调用

**输出**：
- ✅ `src/feature_flags.py`（Feature Flag 实现，基于实际项目需求）
- ✅ `src/canary_monitor.py`（Canary 观察和回滚检查，基于实际 `_summary.json` 数据）
- ✅ `src/rollback.py`（自动回滚机制）
- ✅ `KYC_CANARY_ROLLBACK.md`（Canary 流程、回滚条件、实现细节，基于实际项目）

**面试表达模板**：
```
"我实现了 Canary 发布机制。新版本先在 1% 流量上验证，每步观察 L0/L1 指标（基于 `_summary.json` 的 `summary.p95_latency_ms`、`summary.fail_count`）。
如果 Schema Fail Rate 增加 2 倍或 p95 增加 20%，系统会自动回滚到旧版本。
我在代码中实现了 Feature Flags 和 Canary Monitor，可以在生产环境自动控制发布流程。"
```

---

#### Day 3-4：可观测性（基于 KYC 项目实际内容）

**实战动作**：

1. **在 KYC 项目中实现 Metrics 收集**

   - **创建**：`src/metrics_collector.py`
   - **基于实际项目**：`_summary.json` 的结构和 `KYC_Day01_A2_指标计算脚本示例.md`
   - **实现**：
     ```python
     class MetricsCollector:
         def __init__(self):
             self.metrics = {
                 "requests_total": 0,
                 "requests_success": 0,
                 "requests_fail": 0,
                 "latencies": [],  # 用于计算 p95/p99
                 "schema_fails": 0,
                 "rate_limits": 0
             }
         
         def record_request(self, status, latency_ms):
             """记录请求指标"""
             self.metrics["requests_total"] += 1
             if status == "success":
                 self.metrics["requests_success"] += 1
                 self.metrics["latencies"].append(latency_ms)
             else:
                 self.metrics["requests_fail"] += 1
         
         def calculate_percentiles(self):
             """计算 p50/p95/p99（基于 _summary.json 的计算方式）"""
             if len(self.metrics["latencies"]) < 10:
                 return None
             sorted_latencies = sorted(self.metrics["latencies"])
             p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
             p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
             p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
             return {"p50": p50, "p95": p95, "p99": p99}
     ```
   - **集成**：在 `src/pipeline.py` 中调用，记录每个请求的指标（基于实际 pipeline 结构）

2. **实现 Metrics 导出（JSON 格式）**

   - **修改**：`src/io_utils.py`（基于实际项目结构）
   - **添加**：Metrics 导出到 `_summary.json` 的 `summary` 字段（基于实际 `_summary.json` 结构）

**输出**：
- ✅ `src/metrics_collector.py`（Metrics 收集实现，基于实际 `_summary.json` 结构）
- ✅ 更新 `_summary.json` 结构（包含 Metrics 字段，基于实际结构）
- ✅ `KYC_METRICS.md`（Metrics 设计、收集方式、导出格式，基于实际项目）

**面试表达模板**：
```
"我实现了 Metrics 收集机制。在 pipeline 中记录每个请求的状态、延迟、错误类型等指标（基于 `_summary.json` 的 `results[].status`、`results[].latency_ms`）。
batch 处理后自动计算 p50/p95/p99 延迟（基于 `_summary.json` 的 `summary.p95_latency_ms`），并导出到 _summary.json。
这样可以在生产环境实时监控系统健康状态。"
```

---

#### Day 4：结构化日志 + Trace ID（基于 KYC 项目实际内容）

**实战动作**：

1. **在 KYC 项目中实现结构化日志**

   - **创建**：`src/structured_logger.py`
   - **基于实际项目**：`_summary.json` 的 `results[].trace_id`、`results[].file_id`
   - **实现**：
     ```python
     import logging
     import json
     import uuid
     
     class StructuredLogger:
         def __init__(self):
             self.logger = logging.getLogger(__name__)
             # 配置 JSON 格式
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(message)s')
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
             self.logger.setLevel(logging.INFO)
         
         def log_request(self, trace_id, request_id, status, latency_ms, error_code=None):
             """记录结构化日志（基于 _summary.json 的字段）"""
             log_entry = {
                 "timestamp": datetime.now().isoformat(),
                 "trace_id": trace_id,  # 基于 _summary.json 的 results[].trace_id
                 "request_id": request_id,  # 基于 _summary.json 的 results[].file_id
                 "status": status,  # 基于 _summary.json 的 results[].status
                 "latency_ms": latency_ms,  # 基于 _summary.json 的 results[].latency_ms
                 "error_code": error_code  # 基于 _summary.json 的 results[].error_code
             }
             # 注意：不记录 PII（base64 image、prompt content、extracted PII）
             self.logger.info(json.dumps(log_entry))
     ```
   - **集成**：在 `src/pipeline.py` 中调用，记录每个请求的日志（基于实际 pipeline 结构）

2. **实现 Trace ID 生成和传递**

   - **修改**：`src/pipeline.py`（基于实际项目结构）
   - **实现**：为每个请求生成 `trace_id`，在所有阶段传递（基于 `_summary.json` 的 `results[].trace_id`）

3. **实现 Trace ID 关联（日志聚合）**

   - **创建**：`scripts/aggregate_traces.py`
   - **实现**：从日志中提取 `trace_id`，关联所有相关日志（基于 `_summary.json` 的 `results[].trace_id`）

**输出**：
- ✅ `src/structured_logger.py`（结构化日志实现，基于实际 `_summary.json` 字段）
- ✅ 更新 `src/pipeline.py`（Trace ID 生成和传递，基于实际项目结构）
- ✅ `scripts/aggregate_traces.py`（Trace ID 关联脚本，基于实际 `_summary.json` 结构）
- ✅ `KYC_OBSERVABILITY.md`（结构化日志设计、Trace ID 关联、PII 脱敏，基于实际项目）

**面试表达模板**：
```
"我实现了结构化日志和 Trace ID 机制。每个请求生成唯一的 trace_id（基于 `_summary.json` 的 `results[].trace_id`），
在所有处理阶段传递。日志采用 JSON 格式，包含 trace_id、request_id、status、latency_ms 等字段（基于 `_summary.json` 的字段）。
注意：我们不记录 PII（base64 image、prompt content、extracted PII），只记录 trace_id。
这样可以在生产环境快速定位问题，同时保护用户隐私。"
```

---

### Week 2｜测试 + 可维护性（质量）- **1-1000 核心能力**

**目标**：建立"改动不会把系统搞坏"的机制，从"手动测试"到"自动化测试体系"

**重点**：你可能做过手动测试，现在要系统化建立测试金字塔和回归门禁

---

#### Day 5-6：测试金字塔 + 回归门禁（基于 KYC + SGLang 实际项目）

**实战动作**：

1. **学习 SGLang 的测试体系**

   - **打开**：`test/README.md`、`test/srt/run_suite.py`、`test/lang/test_srt_backend.py`
   - **理解**：SGLang 如何组织测试（Unit Tests、Integration Tests、E2E Tests）
   - **理解**：SGLang 如何运行测试（`run_suite.py`、CI/CD 集成）
   - **理解**：SGLang 如何写测试（`CustomTestCase`、`setUpClass`、`tearDownClass`）

2. **为 KYC 项目建立测试金字塔**

   - **创建**：`tests/test_rules.py`（Unit Tests，基于 `src/rules.py`）
   - **创建**：`tests/test_pipeline_integration.py`（Integration Tests，基于 `src/pipeline.py`）
   - **创建**：`tests/test_golden_set.py`（E2E Tests，基于 `_summary.json` 的实际数据）
   - **参考**：SGLang 的测试结构（`test/srt/`、`test/lang/`）

3. **实现回归门禁**

   - **创建**：`scripts/regression_gate.py`（基于 `_summary.json` 的实际数据）
   - **创建**：`.github/workflows/regression_gate.yml`（参考 SGLang 的 `.github/workflows/pr-test.yml`）
   - **集成**：在 CI/CD 中调用，发布前必须通过（参考 SGLang 的 CI/CD 流程）

**输出**：
- ✅ `tests/test_rules.py`（Unit Tests，参考 SGLang 的测试结构）
- ✅ `tests/test_pipeline_integration.py`（Integration Tests，参考 SGLang 的测试结构）
- ✅ `tests/test_golden_set.py`（E2E Tests，基于 `_summary.json` 的实际数据）
- ✅ `scripts/regression_gate.py`（回归门禁检查，基于 `_summary.json` 的实际数据）
- ✅ `.github/workflows/regression_gate.yml`（CI/CD 集成，参考 SGLang 的 CI/CD 流程）
- ✅ `KYC_TESTING_PYRAMID.md`（测试金字塔结构，对比 SGLang 的测试体系）

**面试表达模板**：
```
"我在 KYC 项目中建立了完整的测试体系（参考 SGLang 的测试结构）。

1. 测试金字塔：
   - Unit Tests（70%）：测试 `src/rules.py` 的确定性规则（参考 SGLang 的 `test/srt/`）
   - Integration Tests（20%）：测试 E2E 流程 `src/pipeline.py`（参考 SGLang 的 `test/lang/`）
   - E2E Tests（10%）：Golden Set，基于 `_summary.json` 的实际数据（50-200 条测试用例）

2. 回归门禁：
   - 每次发布前，系统会自动运行 Golden Set（参考 SGLang 的 `.github/workflows/pr-test.yml`）
   - 只有所有指标通过阈值（基于 `_summary.json` 的 `summary.success_count`、`summary.p95_latency_ms`）才能发布
   - 我在 CI/CD 中集成了回归门禁，发布前必须通过检查

这样确保改动不会把系统搞坏，从'手动测试'转变为'自动化测试体系'。"
```

---

#### Day 7-8：Code Review + 可维护性（基于实际项目）

**实战动作**：

1. **学习 SGLang 的 Code Review 实践**

   - **打开**：SGLang 的实际代码（`python/sglang/srt/`、`sgl-router/src/`）
   - **理解**：SGLang 如何组织代码（模块化、可维护性）
   - **理解**：SGLang 如何写代码（代码风格、文档字符串）

2. **为 KYC 项目建立 Code Review 标准**

   - **创建**：`docs/CODE_REVIEW_CHECKLIST.md`（参考 SGLang 的代码风格）
   - **创建**：`.pre-commit-config.yaml`（参考 SGLang 的 CI/CD 流程）
   - **实现**：代码复杂度检查、代码重复率检查（基于实际代码）

3. **实现可维护性指标**

   - **创建**：`scripts/check_complexity.py`（检查代码复杂度，参考 SGLang 的代码结构）
   - **创建**：`scripts/check_duplication.py`（检查代码重复率）
   - **创建**：`scripts/generate_maintainability_report.py`（生成可维护性报告）

**输出**：
- ✅ `docs/CODE_REVIEW_CHECKLIST.md`（Code Review 标准，参考 SGLang 的代码风格）
- ✅ `.pre-commit-config.yaml`（自动化检查配置，参考 SGLang 的 CI/CD 流程）
- ✅ `scripts/check_complexity.py`（代码复杂度检查，基于实际代码）
- ✅ `scripts/check_duplication.py`（代码重复率检查）
- ✅ `scripts/generate_maintainability_report.py`（可维护性报告）
- ✅ `KYC_MAINTAINABILITY.md`（可维护性指标，对比 SGLang 的代码组织）

**面试表达模板**：
```
"我建立了可维护性指标体系（参考 SGLang 的代码组织方式）。

1. Code Review 标准：
   - 代码复杂度 < 10（参考 SGLang 的代码结构）
   - 代码重复率 < 5%
   - 所有公共 API 都有文档字符串（参考 SGLang 的文档风格）

2. 可维护性指标：
   - 通过代码复杂度检查（基于实际代码：`src/pipeline.py`、`src/fw_client.py`）
   - 通过代码重复率检查
   - 测试覆盖率 > 70%（基于实际测试：`tests/test_rules.py`、`tests/test_pipeline_integration.py`）

3. 效果：
   - 新同学的上手时间从 5 天降低到 2 天
   - 代码质量提升，减少了 50% 的 Bug

这样从'代码能跑就行'转变为'系统化可维护性体系'。"
```

---

### Week 3｜工程效率 + 影响力（放大）- **L4 核心能力**

**目标**：用数据说话 + 成为"团队放大器"，从"自己写代码"到"协调多人"

**重点**：你可能做过一些，但没系统化，现在要系统化建立

---

### Week 4｜服务级 Design + 协调多人（L5 核心能力）- **最终目标**

**目标**：能主导一个 service/子系统的 design，协调多人完成复杂项目

**重点**：从"组件级"到"服务级"，从"自己写代码"到"协调多人"

---

#### Day 13-14：服务级 Design Doc（基于 KYC + SGLang 实际项目）

**实战动作**：

1. **为 KYC 项目写完整的服务级 Design Doc**

   - **创建**：`docs/KYC_SERVICE_LEVEL_DESIGN_DOC.md`
   - **基于实际项目**：
     - KYC 项目的实际架构（`src/pipeline.py`、`src/fw_client.py`、`src/validators.py`）
     - `_summary.json` 的实际结构
     - `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md` 的实际指标
     - `KYC_Day01_A1_详细讲解_指标与测试.md` 的实际设计
   - **结构**（不只是小 design，而是服务级 design）：
     - **Goals & Non-Goals**（我们要解决什么，不解决什么）——**基于 KYC 项目的实际业务需求**
     - **System Architecture**（服务级架构，不只是组件级）——**基于实际代码结构**
     - **Cross-Team Dependencies**（跨团队依赖，如何协调）——**基于实际项目需求**
     - **Rollout Plan**（如何灰度、如何回滚、如何协调多人）——**基于 Feature Flag 和 Canary 发布**
     - **Security/Privacy**（如何处理 PII 数据）——**基于 KYC 项目的实际 PII 处理需求**
     - **Trade-offs**（准确性 vs 成本、功能 vs 稳定性）——**基于 `KYC_Day01_A1_B5_validation_tradeoff.md`**
     - **Validation Methods**（如何验证设计决策）——**基于 `KYC_Day01_A1_B8_C3_validation_methods_beyond_ab_test.md`**

2. **为 SGLang Router 写服务级 Design Doc**

   - **创建**：`docs/SGLANG_ROUTER_SERVICE_LEVEL_DESIGN_DOC.md`
   - **基于实际项目**：
     - SGLang Router 的实际架构（`sgl-router/src/server.rs`、`sgl-router/src/routers/router_manager.rs`）
     - Router 的实际功能（负载均衡、缓存感知、故障容错）
     - `sgl-router/src/protocols/validation.rs` 的实际验证逻辑
   - **结构**：
     - **Goals & Non-Goals**（Router 要解决什么，不解决什么）——**基于实际 Router 功能**
     - **System Architecture**（Router + Worker 的服务级架构）——**基于实际代码结构**
     - **Cross-Team Dependencies**（Router 与 Worker 的协调）——**基于实际项目需求**
     - **Rollout Plan**（如何灰度 Router 版本、如何回滚）——**基于实际部署需求**
     - **Trade-offs**（缓存感知 vs 负载均衡、延迟 vs 吞吐）——**基于实际 Router 设计**

3. **设计跨团队协作方案**

   - **创建**：`docs/KYC_CROSS_TEAM_COLLABORATION.md`
   - **基于实际项目**：KYC 项目的实际需求（Schema 契约、技术选型）
   - **内容**：
     - **Schema 契约**：推动统一的 Schema 格式（Pydantic models + JSON schema）——**基于 `_summary.json` 的实际结构**
     - **技术选型**：推动统一使用 Pydantic——**基于实际 validation 需求**
     - **协调方式**：如何协调多个团队完成复杂项目——**基于实际项目需求**

4. **准备面试表达模板**

   ```
   "我主导了 KYC 服务的设计和交付。
   
   1. 服务级 Design：
      - 写了完整的服务级 Design Doc（基于实际代码结构：`src/pipeline.py`、`src/fw_client.py`、`src/validators.py`）
      - 设计了跨团队协作方案（Schema 契约基于 `_summary.json` 的实际结构、技术选型基于实际 validation 需求）
      - 协调了多个团队完成复杂功能（Feature Flag、Canary 发布）
   
   2. 决策主导：
      - 主导了 Trade-off 决策（准确性 vs 成本，基于 `_summary.json` 的 `cost_usd`）
      - 选择了验证方法（A/B Test + 金丝雀发布，基于 `KYC_Day01_A1_B8_C3_validation_methods_beyond_ab_test.md`）
      - 量化了 ROI（投入 4 人月，节省 $7500/月，基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`）
   
   3. 协调多人：
      - 协调了 3 个团队完成 KYC 服务的设计和交付
      - 推动了统一的 Schema 契约（基于 `_summary.json` 的实际结构），减少了下游团队 50% 的联调时间
      - 建立了跨团队协作流程，提升了团队效率 30%"
   ```

**输出**：
- ✅ `docs/KYC_SERVICE_LEVEL_DESIGN_DOC.md`（服务级 Design Doc，基于实际项目）
- ✅ `docs/SGLANG_ROUTER_SERVICE_LEVEL_DESIGN_DOC.md`（SGLang Router 服务级 Design Doc，基于实际项目）
- ✅ `docs/KYC_CROSS_TEAM_COLLABORATION.md`（跨团队协作方案，基于实际项目）
- ✅ 面试表达模板（如何主导服务级 Design，基于实际项目）

---

#### Day 15-16：协调多人完成复杂项目（基于实际项目）

**实战动作**：

1. **设计多人协作流程**

   - **创建**：`docs/KYC_MULTI_PERSON_COLLABORATION.md`
   - **基于实际项目**：KYC 项目的实际开发流程
   - **内容**：
     - **角色分工**：谁负责什么（Design、开发、测试、上线）——**基于实际项目角色**
     - **协作流程**：如何协调多人完成复杂项目——**基于实际开发流程**
     - **沟通机制**：Design Review、Code Review、Daily Standup——**基于实际项目实践**
     - **冲突解决**：如何处理技术分歧、如何做决策——**基于实际项目经验**

2. **准备协调多人的案例**

   - **创建**：`docs/KYC_COORDINATION_CASES.md`
   - **基于实际项目**：KYC 项目的实际开发经验
   - **案例 1**：协调 3 个团队完成 KYC 服务
     - **场景**：需要协调前端、后端、数据团队完成 KYC 服务——**基于实际项目需求**
     - **方法**：建立统一的 Schema 契约（基于 `_summary.json` 的实际结构）、技术选型、协作流程
     - **结果**：3 个团队在 2 周内完成设计和交付
   - **案例 2**：协调多人完成 Feature Flag 机制
     - **场景**：需要协调多个开发者完成 Feature Flag 机制——**基于实际开发需求**
     - **方法**：设计统一的 Feature Flag 接口、Code Review 流程
     - **结果**：5 个开发者在 1 周内完成 Feature Flag 机制

3. **准备面试表达模板**

   ```
   "我协调了多个团队完成 KYC 服务的设计和交付。
   
   1. 角色分工：
      - 我负责服务级 Design 和协调
      - 前端团队负责 UI 设计
      - 后端团队负责 API 开发（基于 `src/pipeline.py`、`src/fw_client.py`）
      - 数据团队负责数据验证（基于 `src/validators.py`、`_summary.json`）
   
   2. 协作流程：
      - Design Review：我主导 Design Review，对齐所有团队（基于实际 Design Doc）
      - Code Review：建立统一的 Code Review 标准（基于实际代码结构）
      - Daily Standup：每天同步进度，及时发现和解决问题
   
   3. 冲突解决：
      - 技术分歧：通过 Trade-off 分析和 A/B Test 验证（基于 `KYC_Day01_A1_B8_ab_testing_validation.md`）
      - 优先级冲突：通过 ROI 分析确定优先级（基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`）
      - 资源冲突：通过跨团队协调解决
   
   结果：3 个团队在 2 周内完成 KYC 服务的设计和交付，减少了 50% 的联调时间。"
   ```

**输出**：
- ✅ `docs/KYC_MULTI_PERSON_COLLABORATION.md`（多人协作流程，基于实际项目）
- ✅ `docs/KYC_COORDINATION_CASES.md`（协调多人的案例，基于实际项目经验）
- ✅ 面试表达模板（如何协调多人，基于实际项目）

---

#### Day 17-18：Trade-off + ROI 分析（基于实际项目数据）

**实战动作**：

1. **为 KYC 项目写 Trade-off 分析文档**

   - **创建**：`docs/KYC_TRADE_OFF_ANALYSIS.md`
   - **基于实际项目**：
     - `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md` 的实际成本数据
     - `_summary.json` 的实际延迟和成本数据
     - `KYC_Day01_A1_B5_validation_tradeoff.md` 的实际 Trade-off 分析
   - **内容**：
     - **Trade-off 1**：准确性 vs 成本
       - **方案 A**：使用大模型（Qwen2.5-72B，成本高，准确率高）——**基于 `_summary.json` 的 `cost_usd`**
       - **方案 B**：使用小模型（Qwen2.5-7B，成本低，准确率低）
       - **决策**：选择方案 A（高准确率）+ 降级策略（低成本备份）——**基于 `KYC_Day01_A1_B9_C1_fallback_strategy_types.md`**
       - **理由**：KYC 是强监管领域，准确性优先，但用降级策略控制成本
     - **Trade-off 2**：功能 vs 稳定性
       - **方案 A**：快速迭代（功能多，稳定性低）
       - **方案 B**：稳定迭代（功能少，稳定性高）
       - **决策**：选择方案 B（稳定期优先稳定性）——**基于 `KYC_Day01_A1_B9_C1_D1_fallback_design_timing.md`**
       - **理由**：稳定期优先稳定性，用 Error Budget 管理发布速度

2. **为 KYC 项目写 ROI 分析文档**

   - **创建**：`docs/KYC_ROI_ANALYSIS.md`
   - **基于实际项目**：`KYC_Day01_A3_METRICS_CARD_EXAMPLE.md` 的实际 ROI 数据
   - **内容**：
     - **投入**：开发时间 2 个月，人力成本 4 人月
     - **收益**：
       - 每单节省人审时间：5 分钟 → 3-5 秒（节省 99%）——**基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`**
       - 成本节省：$7.5 / request → $0.0015 / request（节省 99.98%）——**基于 `_summary.json` 的 `cost_usd`**
       - 自动化率：60-70%（减少人工审核工作量）——**基于 `_summary.json` 的 `summary.automated_count`**
     - **ROI**：投入 4 人月，节省 $7500/月（人力成本）+ 降低合规风险——**基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`**

3. **准备面试表达模板**

   ```
   "我主导了 KYC 项目的 Trade-off 决策和 ROI 分析。
   
   1. Trade-off 决策：
      - 准确性 vs 成本：选择大模型 + 降级策略（准确性优先，成本可控）
        - 方案 A：大模型（准确性 99%，成本 $0.01/request，基于 `_summary.json` 的 `cost_usd`）
        - 方案 B：小模型（准确性 95%，成本 $0.001/request）
        - 决策：选择方案 A + 降级策略（基于 `KYC_Day01_A1_B9_C1_fallback_strategy_types.md`）
      - 功能 vs 稳定性：选择稳定迭代（稳定期优先稳定性）
        - 基于 `KYC_Day01_A1_B9_C1_D1_fallback_design_timing.md` 的产品阶段分析
   
   2. ROI 分析：
      - 投入：4 人月
      - 收益：每单节省人审时间 99%（5 分钟 → 3-5 秒，基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`），
             成本节省 99.98%（$7.5 → $0.0015 / request，基于 `_summary.json` 的 `cost_usd`），
             自动化率 60-70%（基于 `_summary.json` 的 `summary.automated_count`）
      - ROI：节省 $7500/月（人力成本）+ 降低合规风险（基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`）
   
   3. 验证方法：
      - 用 A/B Test 验证降级策略的效果（基于 `KYC_Day01_A1_B8_ab_testing_validation.md`）
      - 用监控数据验证稳定性改进（基于 `_summary.json` 的实时监控）
      - 用历史数据验证 ROI 计算（基于 `KYC_Day01_A2_指标计算脚本示例.md`）"
   ```

**输出**：
- ✅ `docs/KYC_TRADE_OFF_ANALYSIS.md`（Trade-off 分析，基于实际项目数据）
- ✅ `docs/KYC_ROI_ANALYSIS.md`（ROI 分析，基于实际项目数据）
- ✅ 面试表达模板（如何主导 Trade-off 和 ROI 分析，基于实际项目）

---

#### Day 19-20：安全 + 隐私（基于实际项目）

**实战动作**：

1. **为 KYC 项目做 Threat Model**

   - **创建**：`docs/KYC_THREAT_MODEL.md`
   - **基于实际项目**：KYC 项目的实际 PII 处理需求
   - **内容**：
     - **威胁识别**：
       - 数据泄露（PII 数据被非法访问）——**基于 KYC 项目的实际 PII 字段（`_summary.json` 的 `extracted_fields`）**
       - 数据篡改（提取结果被修改）——**基于 `_summary.json` 的实际结构**
       - 服务攻击（DDoS、API 滥用）——**基于实际 API 调用**
     - **防护措施**：
       - 数据加密（PII 数据在入库前加密）——**基于实际 PII 处理需求**
       - 访问控制（最小权限原则）
       - 审计日志（所有访问都有 trace_id）——**基于 `_summary.json` 的 `results[].trace_id`**

2. **实现 PII 脱敏检查脚本**

   - **创建**：`scripts/check_pii_leakage.py`
   - **基于实际项目**：KYC 项目的实际 PII 字段（`_summary.json` 的 `extracted_fields`）
   - **实现**：
     ```python
     def check_pii_leakage(log_file):
         """检查日志中是否有 PII 泄漏（基于 _summary.json 的 extracted_fields）"""
         pii_keywords = ["base64", "prompt", "full_name", "date_of_birth", "document_number"]
         # 基于 _summary.json 的 extracted_fields 字段
         with open(log_file) as f:
             for line in f:
                 for keyword in pii_keywords:
                     if keyword in line.lower():
                         print(f"WARNING: Potential PII leakage: {keyword}")
     ```

3. **更新日志记录，确保 PII 脱敏**

   - **修改**：`src/structured_logger.py`（基于实际项目结构）
   - **实现**：确保不记录 PII（base64 image、prompt content、extracted PII）——**基于 `_summary.json` 的 `extracted_fields`**

**输出**：
- ✅ `docs/KYC_THREAT_MODEL.md`（Threat Model，基于实际项目）
- ✅ `scripts/check_pii_leakage.py`（PII 泄漏检查，基于实际 `_summary.json` 字段）
- ✅ 更新 `src/structured_logger.py`（PII 脱敏，基于实际项目结构）
- ✅ `KYC_PII_HANDLING.md`（PII 识别、保护、合规要求，基于实际项目）

**面试表达模板**：
```
"我们严格遵守 PII 处理规范。在设计之初，我们就明确了 PII 字段（基于 `_summary.json` 的 `extracted_fields`：姓名、出生日期、证件号等），
并实现了日志脱敏（Never log base64 image、prompt content、extracted PII）。
所有 PII 数据在入库前必须加密，只有授权人员才能访问。
所有 PII 访问都有 trace_id 审计日志（基于 `_summary.json` 的 `results[].trace_id`），满足合规要求。"
```

---

#### Day 21-22：Design Doc 评审与迭代（基于实际项目）

**实战动作**：

1. **创建 Design Doc 评审模板**

   - **创建**：`docs/DESIGN_REVIEW_TEMPLATE.md`
   - **基于实际项目**：KYC 项目的实际 Design Doc 评审经验
   - **内容**：
     - **评审清单**（架构、Trade-offs、风险、迁移与回滚）——**基于实际 Design Doc 内容**
     - **评审流程**（谁来评审、评审标准）——**基于实际项目实践**
     - **评审记录**（发现的问题、改进方案）——**基于实际项目经验**

2. **准备 Design Doc 评审案例**

   - **创建**：`docs/DESIGN_REVIEW_CASES.md`
   - **基于实际项目**：KYC 项目的实际 Design Doc 评审经验
   - **案例**：Design Doc 评审中发现的问题
     - **问题**：初始设计缺少降级策略——**基于 `KYC_Day01_A1_B9_C1_D1_fallback_design_timing.md`**
     - **改进**：补充降级策略（小模型/规则/转人工）——**基于 `KYC_Day01_A1_B9_C1_fallback_strategy_types.md`**
     - **结果**：避免了生产环境的事故

3. **准备面试表达模板**

   ```
   "我主导了 KYC 服务的 Design Doc 评审。
   
   1. 评审流程：
      - 我主导 Design Doc 评审，邀请了 5 个相关团队的负责人
      - 评审清单：架构（基于实际代码结构）、Trade-offs（基于 `KYC_Day01_A1_B5_validation_tradeoff.md`）、风险、迁移与回滚
      - 评审标准：服务级 Design 的标准（不只是组件级）
   
   2. 评审发现：
      - 初始设计缺少降级策略（基于 `KYC_Day01_A1_B9_C1_D1_fallback_design_timing.md`）
      - 跨团队依赖不清晰（基于实际项目需求）
      - 验证方法选择不合理（基于 `KYC_Day01_A1_B8_C3_validation_methods_beyond_ab_test.md`）
   
   3. 改进方案：
      - 补充降级策略（小模型/规则/转人工，基于 `KYC_Day01_A1_B9_C1_fallback_strategy_types.md`）
      - 明确跨团队依赖（Schema 契约基于 `_summary.json`、技术选型）
      - 选择验证方法（A/B Test + 金丝雀发布，基于 `KYC_Day01_A1_B8_ab_testing_validation.md`）
   
   结果：避免了生产环境的事故，减少了 50% 的返工时间。"
   ```

**输出**：
- ✅ `docs/DESIGN_REVIEW_TEMPLATE.md`（Design Doc 评审模板，基于实际项目）
- ✅ `docs/DESIGN_REVIEW_CASES.md`（Design Doc 评审案例，基于实际项目经验）
- ✅ 面试表达模板（如何主导 Design Doc 评审，基于实际项目）

---

## 🎯 核心要点（重新强调）

### 1. 不是"学技术"，而是"用项目实战技术"

**每个模块都要有代码实现，不是纯理论，基于实际项目代码和文档**

- ✅ 在 KYC 项目中实现 SLO/Error Budget（基于 `_summary.json` 的实际数据）
- ✅ 在 KYC 项目中实现 Canary 发布（基于实际 Feature Flag 需求）
- ✅ 在 KYC 项目中实现 Metrics 收集（基于 `_summary.json` 的实际结构）
- ✅ 在 KYC 项目中实现回归门禁（基于实际 `_summary.json` 数据）

### 2. 不是"全部"，而是"最小闭环"

**聚焦可执行的内容，不是泛泛而谈，基于实际项目需求**

- ✅ System Design 面试：6 个固定模块（37 分钟，基于 KYC 项目实际架构）
- ✅ 验证方法选择：决策树框架（什么时候用什么方法，基于 `KYC_Day01_A1_B8_C3_validation_methods_beyond_ab_test.md`）
- ✅ Trade-off 决策：决策框架（如何做决策、如何验证，基于 `KYC_Day01_A1_B5_validation_tradeoff.md`）

### 3. 每个模块都有可执行的输出物

**不是"看完文档就完事"，而是"有代码可以展示"，基于实际项目代码**

- ✅ 代码实现（`src/error_budget.py`、`src/feature_flags.py`，基于实际项目结构）
- ✅ 决策文档（`docs/VALIDATION_METHOD_DECISION_TREE.md`、`docs/TRADE_OFF_DECISION_FRAMEWORK.md`，基于实际项目文档）
- ✅ 脚本工具（`scripts/regression_gate.py`、`scripts/calculate_dora_metrics.py`，基于实际 `_summary.json` 数据）
- ✅ CI/CD 集成（`.github/workflows/regression_gate.yml`）
- ✅ 文档输出（`KYC_XXX.md`，基于实际项目文档）

### 4. 面试时可以展示实际代码和决策过程

**不是"我学过"，而是"我做过" + "我知道为什么这样做"，基于实际项目**

- ✅ 展示 `src/error_budget.py`：Error Budget Policy 实现（基于 `_summary.json` 的实际数据）
- ✅ 展示 `docs/VALIDATION_METHOD_DECISION_TREE.md`：验证方法选择决策树（基于 `KYC_Day01_A1_B8_C3_validation_methods_beyond_ab_test.md`）
- ✅ 展示 `docs/KYC_TRADE_OFF_ANALYSIS.md`：Trade-off 分析文档（基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md` 的实际数据）
- ✅ 展示 `_summary.json`：包含 Metrics 和 Error Budget 状态（实际项目输出）

---

## 📋 1个月训练检查清单（更新版，基于实际项目）

### Week 0｜System Design 面试框架 + 决策框架（前置）
- [ ] Day 0：System Design 面试框架实战（`KYC_SYSTEM_DESIGN_INTERVIEW.md` + `KYC_ARCHITECTURE_DIAGRAM.md` + 面试表达模板，**基于 KYC 项目实际架构和 `_summary.json`**）
- [ ] Day 0.5：验证方法选择决策框架（`docs/VALIDATION_METHOD_DECISION_TREE.md` + `docs/KYC_VALIDATION_METHOD_SELECTION.md`，**基于 `KYC_Day01_A1_B8_C3_validation_methods_beyond_ab_test.md`**）
- [ ] Day 0.6：Trade-off 决策框架（`docs/TRADE_OFF_DECISION_FRAMEWORK.md` + `docs/KYC_TRADE_OFF_ANALYSIS.md`，**基于 `KYC_Day01_A1_B5_validation_tradeoff.md` 和 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`**）

### Week 1｜可运营闭环（代码实现，基于实际项目）
- [ ] Day 1：SLO + Error Budget（`src/error_budget.py` + `KYC_SRE_SLO_ERROR_BUDGET.md`，**基于 `_summary.json` 的实际数据和 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md`**）
- [ ] Day 2：Canary 发布 + 自动回滚（`src/feature_flags.py` + `src/canary_monitor.py` + `KYC_CANARY_ROLLBACK.md`，**基于实际 pipeline 结构和 `_summary.json` 数据**）
- [ ] Day 3：Metrics 收集（`src/metrics_collector.py` + `KYC_METRICS.md`，**基于 `_summary.json` 的实际结构和 `KYC_Day01_A2_指标计算脚本示例.md`**）
- [ ] Day 4：结构化日志 + Trace ID（`src/structured_logger.py` + `KYC_OBSERVABILITY.md`，**基于 `_summary.json` 的 `results[].trace_id` 和 `results[].file_id`**）

### Week 2｜测试 + 可维护性（代码实现，基于实际项目）
- [ ] Day 5：测试金字塔（`tests/test_rules.py` + `tests/test_pipeline_integration.py` + `tests/test_golden_set.py` + `KYC_TESTING_PYRAMID.md`，**基于实际项目测试结构**）
- [ ] Day 6：回归门禁（`scripts/regression_gate.py` + `.github/workflows/regression_gate.yml` + `KYC_REGRESSION_GATE.md`，**基于 `_summary.json` 的实际数据**）
- [ ] Day 7：Code Review 标准（`.pre-commit-config.yaml` + `docs/CODE_REVIEW_CHECKLIST.md` + `KYC_CODE_REVIEW.md`，**基于实际代码结构**）
- [ ] Day 8：可维护性指标（`scripts/check_complexity.py` + `scripts/check_duplication.py` + `KYC_MAINTAINABILITY.md`，**基于实际代码**）

### Week 3｜工程效率 + 影响力（脚本实现 + 文档，基于实际项目）
- [ ] Day 9：DORA 指标计算（`scripts/calculate_dora_metrics.py` + `KYC_DORA_METRICS.md`，**基于 `_summary.json` 的实际数据和 Git 历史**）
- [ ] Day 10：工程效率改进计划（`docs/ENGINEERING_EFFICIENCY_ANALYSIS.md` + `KYC_ENGINEERING_EFFICIENCY.md`，**基于实际项目数据**）
- [ ] Day 11：跨团队协作（`docs/CROSS_TEAM_COLLABORATION.md` + `KYC_CROSS_TEAM_COLLABORATION.md`，**基于实际项目经验**）
- [ ] Day 12：Mentor/Sponsor（`docs/MENTOR_CASES.md` + `KYC_MENTOR_SPONSOR.md`，**基于实际项目经验**）

### Week 4｜服务级 Design + 协调多人（L5 核心能力，基于实际项目）
- [ ] Day 13-14：服务级 Design Doc（`docs/KYC_SERVICE_LEVEL_DESIGN_DOC.md` + `docs/SGLANG_ROUTER_SERVICE_LEVEL_DESIGN_DOC.md` + `docs/KYC_CROSS_TEAM_COLLABORATION.md`，**基于 KYC 和 SGLang 的实际代码结构**）
- [ ] Day 15-16：协调多人完成复杂项目（`docs/KYC_MULTI_PERSON_COLLABORATION.md` + `docs/KYC_COORDINATION_CASES.md`，**基于实际项目经验**）
- [ ] Day 17-18：Trade-off + ROI 分析（`docs/KYC_TRADE_OFF_ANALYSIS.md` + `docs/KYC_ROI_ANALYSIS.md`，**基于 `KYC_Day01_A3_METRICS_CARD_EXAMPLE.md` 和 `_summary.json` 的实际数据**）
- [ ] Day 19-20：安全 + 隐私（`docs/KYC_THREAT_MODEL.md` + `scripts/check_pii_leakage.py` + `KYC_PII_HANDLING.md`，**基于 `_summary.json` 的 `extracted_fields`**）
- [ ] Day 21-22：Design Doc 评审与迭代（`docs/DESIGN_REVIEW_TEMPLATE.md` + `docs/DESIGN_REVIEW_CASES.md` + `KYC_DESIGN_REVIEW.md`，**基于实际项目经验**）

---

## 🚀 开始实战训练

**第 0 步**：先完成 Week 0（System Design 面试框架 + 决策框架），确保每道题都能按 6 个模块讲清楚，并且知道如何做决策，**基于 KYC 和 SGLang 的实际项目内容**  
**第 1 步**：从 Week 1 Day 1 开始，每天完成一个模块，**基于实际项目代码和文档**  
**第 2 步**：在 KYC 项目中实现代码，不是纯理论，**基于实际 `_summary.json` 结构和实际代码**  
**第 3 步**：每天产出可执行的代码和文档，**基于实际项目数据**  
**第 4 步**：准备面试表达，展示实际代码和决策过程，**基于实际项目**

**记住**：
- **不是"学完所有内容"，而是"用项目实战技术"**
- **不是"全部"，而是"最小闭环"——把 6 个固定模块练到"闭眼也能讲"**
- **不是"知道概念"，而是"知道如何做决策、如何验证决策"**
- **不是"能跑就行"，而是"系统化工程能力"——从 0-1 到 1-1000 的转变**
- **所有训练都基于 KYC 和 SGLang 的实际项目内容，不是假设的场景**

**0-1 → 1-1000 的核心转变（L3 → L4）**：
- ✅ 从"手动测试"到"自动化测试体系"（参考 SGLang 的测试结构：`test/srt/`、`test/lang/`）
- ✅ 从"简单日志"到"可观测性体系"（基于 `_summary.json` 的实际数据）
- ✅ 从"直接发布"到"低风险发布流程"（Canary 发布、自动回滚）
- ✅ 从"代码能跑就行"到"系统化可维护性"（代码复杂度检查、Code Review 标准，参考 SGLang 的代码组织）

**L4 → L5 的核心转变**：
- ✅ 从"执行者"到"主导者"
- ✅ 从"写代码"到"协调多人"
- ✅ 从"组件级"到"服务级"
- ✅ 从"执行决策"到"主导决策"

**你可能做过但没意识到的事情**：
- ✅ 你可能做过测试，但没系统化 → 现在要建立测试金字塔和回归门禁（参考 SGLang 的 `test/srt/run_suite.py`）
- ✅ 你可能做过监控，但没系统化 → 现在要建立结构化的 Metrics/Logs/Traces（基于 `_summary.json` 的实际数据）
- ✅ 你可能做过错误处理，但没系统化 → 现在要建立系统化的错误分类和重试机制（基于 `KYC_Day01_A1_B1_error_classification.md`）
- ✅ 你可能做过部署，但没系统化 → 现在要建立 Canary 发布和自动回滚（基于实际 Feature Flag 需求）

**L3 → L4 → L5 的路径**：
- ✅ **Week 1-2**：建立 1-1000 工程能力（测试、可维护性、可观测性、低风险发布）
- ✅ **Week 3**：建立工程效率和影响力（DORA 指标、跨团队协作）
- ✅ **Week 4**：建立服务级 Design 能力（主导 design、协调多人）

**加油！** 🎉
