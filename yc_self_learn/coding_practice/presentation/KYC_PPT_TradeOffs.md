# KYC Project - Key Design Trade-offs
# KYC 项目 - 关键设计权衡

## Slide: Design Trade-offs

---

## 核心Trade-offs概览

### 1. Schema-First Design vs 直接使用LLM输出

| 方案 | 优势 | 劣势 | 选择 | 理由 |
|------|------|------|------|------|
| **Schema-First** ✅ | • 可维护性（Schema变更可追踪）<br>• 可扩展性（新增字段只需更新Schema）<br>• 可测试性（Schema验证可自动化）<br>• 类型安全（Pydantic类型检查） | • 需要预先定义Schema<br>• 可能拒绝有效的非标准输出 | **选择** | **KYC项目需要结构化输出**，Schema-First确保输出符合预期格式，便于后续验证和处理 |
| **直接使用LLM输出** | • 灵活性高（LLM可以自由发挥）<br>• 无需预先定义结构 | • 输出不稳定（每次可能不同）<br>• 难以验证和测试<br>• 难以维护和扩展 | 不选择 | KYC场景需要**可审计、可测试**的输出 |

### 2. 确定性规则引擎 vs 纯LLM决策

| 方案 | 优势 | 劣势 | 选择 | 理由 |
|------|------|------|------|------|
| **确定性规则引擎** ✅ | • 可审计性（所有决策有明确规则依据）<br>• 可测试性（规则可单元测试）<br>• 可解释性（决策原因可追溯）<br>• 可控性（易于修改规则） | • 需要维护规则代码<br>• 可能无法覆盖所有边界情况 | **选择** | **KYC场景需要可审计性**，规则引擎确保决策可追溯、可解释，符合合规要求 |
| **纯LLM决策** | • 灵活性高（LLM可以处理复杂情况）<br>• 无需维护规则 | • 不可审计（黑盒决策）<br>• 不可预测（相同输入可能不同输出）<br>• 难以调试 | 不选择 | 金融合规场景需要**可审计、可解释**的决策过程 |

### 3. Per-File Isolation vs 批量处理

| 方案 | 优势 | 劣势 | 选择 | 理由 |
|------|------|------|------|------|
| **Per-File Isolation** ✅ | • 故障隔离（单个文件失败不影响其他文件）<br>• 高可用性（部分失败不影响整体）<br>• 可并行处理<br>• 独立trace（每个文件独立追踪） | • 需要额外的错误处理逻辑<br>• 可能增加内存使用 | **选择** | **生产环境需要故障隔离**，Per-File Isolation确保单个文件失败不会影响整个batch |
| **批量原子处理** | • 事务性保证（全部成功或全部失败）<br>• 逻辑简单 | • 单个失败影响整个batch<br>• 不可并行<br>• 难以定位问题 | 不选择 | KYC场景处理大量文件，需要**高可用性和故障隔离** |

### 4. Fireworks API vs 本地sglang serve

| 方案 | 优势 | 劣势 | 选择 | 理由 |
|------|------|------|------|------|
| **Fireworks API** (托管) | • 无需管理基础设施<br>• 自动扩缩容<br>• 高可用性<br>• 易于集成 | • 成本较高（按token计费）<br>• 依赖外部服务<br>• 可能有延迟<br>• 数据隐私考虑 | **PoV阶段选择** | PoV阶段快速验证，无需管理基础设施 |
| **本地sglang serve** | • 成本可控（按需付费）<br>• 数据隐私（数据不出本地）<br>• 延迟可控<br>• 无外部依赖 | • 需要管理基础设施<br>• 需要GPU资源<br>• 需要运维工作 | **生产可考虑** | 生产环境可以考虑本地部署，降低成本和提高数据隐私 |

### 5. 严格验证 vs 宽松验证

| 方案 | 优势 | 劣势 | 选择 | 理由 |
|------|------|------|------|------|
| **分层验证 + 可配置阈值** ✅ | • 平衡Error Rate和自动化率<br>• Preprocessing严格（格式检查）<br>• Post-inference可配置（Schema验证）<br>• Post-processing宽松（转人工，不直接失败） | • 需要设计分层策略<br>• 需要配置管理 | **选择** | **平衡设计**：Preprocessing严格避免无效数据，Post-processing宽松（转人工）不推高Error Rate，同时保证输出质量 |
| **严格验证** | • 输出质量高 | • Error Rate高（正常请求也被拒绝）<br>• 用户体验差 | 不选择 | Error Rate过高影响系统可用性 |
| **宽松验证** | • Error Rate低 | • 输出不稳定（错误数据通过）<br>• 业务风险高 | 不选择 | 输出质量不符合KYC要求 |

### 6. 自动重试 vs 快速失败

| 方案 | 优势 | 劣势 | 选择 | 理由 |
|------|------|------|------|------|
| **指数退避重试 + 降级** ✅ | • 提高成功率（临时失败可恢复）<br>• 自动重试可恢复错误<br>• 降级机制保证可用性 | • 可能增加延迟<br>• 需要重试逻辑 | **选择** | **生产环境需要容错性**，自动重试处理临时失败（如网络抖动），降级机制保证系统可用性 |
| **快速失败** | • 延迟低<br>• 逻辑简单 | • 成功率低（临时失败也失败）<br>• 用户体验差 | 不选择 | 生产环境临时失败频繁，需要重试机制 |

### 7. Privacy-Aware Logging vs 完整日志

| 方案 | 优势 | 劣势 | 选择 | 理由 |
|------|------|------|------|------|
| **Privacy-Aware Logging** ✅ | • 合规性（符合PII保护要求）<br>• 安全性（不泄露敏感信息）<br>• 使用trace_id关联 | • 调试时可能需要更多上下文 | **选择** | **KYC处理PII数据**，必须符合隐私保护要求，使用trace_id关联日志，不记录PII明文 |
| **完整日志** | • 调试方便（信息完整） | • 合规风险（PII泄露）<br>• 安全风险 | 不选择 | 金融场景必须符合合规要求 |

---

## 关键Trade-off总结

### 设计原则（Design Principles）

1. **可审计性 > 灵活性**
   - 选择Schema-First和确定性规则引擎，牺牲灵活性换取可审计性

2. **故障隔离 > 原子性**
   - 选择Per-File Isolation，牺牲原子性换取故障隔离和高可用性

3. **平衡 > 极端**
   - 选择分层验证+可配置阈值，在Error Rate和自动化率之间找平衡

4. **容错性 > 简单性**
   - 选择自动重试+降级，增加复杂度但提高系统可靠性

5. **合规性 > 便利性**
   - 选择Privacy-Aware Logging，牺牲调试便利性换取合规性

### 核心决策矩阵

| Trade-off | 选择 | 核心理由 | 影响 |
|-----------|------|----------|------|
| **Schema-First** | ✅ 选择 | 可维护、可测试、可扩展 | 确保输出结构化 |
| **确定性规则引擎** | ✅ 选择 | 可审计、可解释 | 符合合规要求 |
| **Per-File Isolation** | ✅ 选择 | 故障隔离、高可用 | 提高系统可靠性 |
| **分层验证** | ✅ 选择 | 平衡Error Rate和自动化率 | 优化系统性能 |
| **自动重试+降级** | ✅ 选择 | 容错性、可用性 | 提高成功率 |
| **Privacy-Aware Logging** | ✅ 选择 | 合规性、安全性 | 符合隐私保护 |

---

## PPT Slide设计建议

### Slide布局

**标题**: Design Trade-offs & Decisions

**内容结构**:
1. **左侧**: 方案A（优势+劣势）
2. **中间**: VS
3. **右侧**: 方案B（优势+劣势）
4. **底部**: 选择 + 理由

### 视觉设计
- 使用对比色区分选择和不选择的方案
- 用✅标记选择的方案
- 用箭头或连线展示决策流程
- 关键理由用醒目的颜色突出

### 讲解要点

**每个Trade-off讲解1-2分钟**:
1. 说明两个方案（各10秒）
2. 列出优势和劣势（各20秒）
3. 解释选择理由（30-40秒）
4. 展示实际效果（20-30秒）

**重点强调**:
- **为什么做这个选择**（核心理由）
- **实际效果如何**（数据和结果）
- **如果换一个选择会怎样**（对比分析）

---

## 面试话术建议

### 开场
"在设计KYC系统时，我们面临几个关键的trade-off决策..."

### 核心Trade-off讲解

#### Schema-First vs 直接LLM输出
"第一个trade-off是Schema-First设计。虽然需要预先定义Schema，但这确保了输出的可维护性和可测试性。在KYC场景中，我们需要结构化输出以便后续验证和处理。"

#### 确定性规则引擎 vs 纯LLM决策
"第二个trade-off是确定性规则引擎。虽然需要维护规则代码，但这提供了可审计性和可解释性，这对金融合规场景至关重要。"

#### Per-File Isolation vs 批量处理
"第三个trade-off是Per-File Isolation。虽然需要额外的错误处理逻辑，但这确保了故障隔离，单个文件失败不会影响整个batch。"

#### 分层验证 vs 严格/宽松验证
"第四个trade-off是验证策略。我们采用了分层验证+可配置阈值，在Preprocessing严格检查格式，在Post-processing宽松处理（转人工而不是直接失败），这样既保证了输出质量，又不推高Error Rate。"

### 结尾
"通过这些trade-off，我们实现了一个**可审计、可测试、高可用**的系统，满足了KYC场景的合规和可靠性要求。"

---

## 简化版：核心Trade-offs (PPT Bullet Points)

### Slide内容（直接复制到PPT）

```
Key Design Trade-offs

• Schema-First Design vs 直接使用LLM输出
  → 选择Schema-First：可维护、可测试、可扩展

• 确定性规则引擎 vs 纯LLM决策
  → 选择规则引擎：可审计、可解释、符合合规

• Per-File Isolation vs 批量处理
  → 选择Per-File Isolation：故障隔离、高可用

• 分层验证 vs 严格/宽松验证
  → 选择分层验证：平衡Error Rate和自动化率

• 自动重试+降级 vs 快速失败
  → 选择自动重试：容错性、提高成功率
```

---

## English Version: Core Trade-offs (PPT Bullet Points)

### Slide Content (Copy to PPT)

```
Key Design Trade-offs

• Schema-First Design vs Direct LLM Output
  → Choose Schema-First: Maintainable, Testable, Extensible

• Deterministic Rules Engine vs Pure LLM Decision
  → Choose Rules Engine: Auditable, Explainable, Compliant

• Per-File Isolation vs Batch Processing
  → Choose Per-File Isolation: Fault Isolation, High Availability

• Layered Validation vs Strict/Loose Validation
  → Choose Layered Validation: Balance Error Rate and Automation Rate

• Auto Retry + Fallback vs Fast Fail
  → Choose Auto Retry: Fault Tolerance, Higher Success Rate
```

---

## PPT演讲话术（详细版）

### 开场（10秒）
"在设计KYC系统时，我们面临几个关键的trade-off决策。每个决策都体现了我们对**可审计性、可测试性、高可用性**的追求。"

---

### Trade-off 1: Schema-First Design vs 直接使用LLM输出（30秒）

**Bullet Point讲解**：
"第一个trade-off是Schema-First设计。"

**详细讲解**：
"LLM的输出是不稳定的，每次可能略有不同。我们有两个选择：
- 方案A：直接使用LLM的输出，简单但不可控
- 方案B：用Pydantic定义Schema，强制LLM输出符合预定义结构

**我们选择Schema-First**，原因是：
1. **可维护性**：Schema变更可以追踪，所有输出都符合预期格式
2. **可测试性**：我们可以自动化测试Schema验证，不需要人工检查
3. **可扩展性**：新增字段只需更新Schema，不需要修改大量代码

这个决策虽然需要预先定义Schema，但确保了整个系统的稳定性和可维护性。"

---

### Trade-off 2: 确定性规则引擎 vs 纯LLM决策（30秒）

**Bullet Point讲解**：
"第二个trade-off是确定性规则引擎。"

**详细讲解**：
"LLM可以做出决策，但它是黑盒的，不可审计。我们有两个选择：
- 方案A：完全依赖LLM决策，灵活但不可审计
- 方案B：使用确定性规则引擎进行二次验证，可审计但需要维护规则

**我们选择确定性规则引擎**，原因是：
1. **可审计性**：每个决策都有明确的规则依据，可以追溯
2. **可解释性**：当出现问题时，我们可以明确知道是哪个规则触发了决策
3. **合规要求**：金融场景需要可审计的决策过程，纯LLM无法满足

虽然需要维护规则代码，但这符合KYC场景的合规要求，也让我们对系统有更好的控制。"

---

### Trade-off 3: Per-File Isolation vs 批量处理（25秒）

**Bullet Point讲解**：
"第三个trade-off是Per-File Isolation。"

**详细讲解**：
"在处理批量文件时，我们有两个选择：
- 方案A：批量原子处理，全部成功或全部失败
- 方案B：Per-File Isolation，每个文件独立处理，单个失败不影响其他

**我们选择Per-File Isolation**，原因是：
1. **故障隔离**：一个文件失败不会影响整个batch，提高系统可用性
2. **高可用性**：部分失败不影响整体，系统更稳定
3. **可并行处理**：每个文件独立，可以并行处理提高效率

虽然需要额外的错误处理逻辑，但这确保了生产环境的高可用性。"

---

### Trade-off 4: 分层验证 vs 严格/宽松验证（30秒）

**Bullet Point讲解**：
"第四个trade-off是分层验证策略。"

**详细讲解**：
"验证策略直接影响Error Rate和自动化率。我们有两个极端选择：
- 方案A：严格验证，Error Rate高但输出质量好
- 方案B：宽松验证，Error Rate低但输出不稳定

**我们选择分层验证+可配置阈值**，策略是：
1. **Preprocessing严格**：只检查格式、大小等硬性要求，避免无效数据进入
2. **Post-inference可配置**：Schema验证的严格程度可通过Feature Flag调整
3. **Post-processing宽松**：业务规则不直接失败，而是转人工审核，这样既保证了输出质量，又不推高Error Rate

这个设计让我们在Error Rate和自动化率之间找到了平衡点。"

---

### Trade-off 5: 自动重试+降级 vs 快速失败（25秒）

**Bullet Point讲解**：
"第五个trade-off是自动重试和降级机制。"

**详细讲解**：
"当外部服务失败时，我们有两个选择：
- 方案A：快速失败，简单但成功率低
- 方案B：自动重试+降级，复杂但容错性好

**我们选择自动重试+降级**，原因是：
1. **容错性**：临时失败（如网络抖动）可以通过重试恢复
2. **可用性**：当主服务不可用时，降级方案保证系统继续运行
3. **用户体验**：自动处理临时故障，用户感知更好

虽然增加了系统复杂度，但显著提高了系统的可靠性和用户体验。"

---

### 总结（15秒）

**话术**：
"通过这五个trade-off，我们实现了一个**可审计、可测试、高可用**的系统：
- Schema-First确保了输出的稳定性
- 规则引擎保证了可审计性
- Per-File Isolation提供了故障隔离
- 分层验证平衡了Error Rate和自动化率
- 自动重试和降级提高了容错性

这些决策共同满足了KYC场景的**合规和可靠性要求**。"

---

## 精简版话术（5分钟总时长）

### 快速版（每个Trade-off 20秒）

**开场**（5秒）：
"在设计KYC系统时，我们做了几个关键trade-off："

**Trade-off 1**（20秒）：
"Schema-First vs 直接LLM输出。我们选择Schema-First，因为可维护、可测试、可扩展。虽然需要预定义Schema，但确保了输出稳定性。"

**Trade-off 2**（20秒）：
"确定性规则引擎 vs 纯LLM决策。我们选择规则引擎，因为可审计、可解释。金融场景需要可审计的决策过程。"

**Trade-off 3**（20秒）：
"Per-File Isolation vs 批量处理。我们选择Per-File Isolation，因为故障隔离、高可用。单个文件失败不影响整个batch。"

**Trade-off 4**（20秒）：
"分层验证 vs 严格/宽松验证。我们选择分层验证+可配置阈值，平衡Error Rate和自动化率。Preprocessing严格，Post-processing宽松转人工。"

**Trade-off 5**（20秒）：
"自动重试+降级 vs 快速失败。我们选择自动重试+降级，因为容错性、可用性。临时失败可恢复，主服务不可用时降级。"

**结尾**（15秒）：
"通过这些trade-off，我们实现了可审计、可测试、高可用的系统，满足了KYC场景的合规和可靠性要求。"

**总时长：约2分钟（快速版）或 5分钟（详细版）**

---

## English Version: Presentation Script

### Opening (10 seconds)
"When designing the KYC system, we faced several key trade-off decisions. Each decision reflects our pursuit of **auditability, testability, and high availability**."

---

### Trade-off 1: Schema-First Design vs Direct LLM Output (30 seconds)

**Bullet Point Script**:
"The first trade-off is Schema-First design."

**Detailed Script**:
"LLM output is unpredictable - each response may vary slightly. We had two options:
- Option A: Use LLM output directly - simple but uncontrollable
- Option B: Use Pydantic Schema to enforce structured output - requires definition but controlled

**We chose Schema-First** because:
1. **Maintainability**: Schema changes are trackable, all outputs conform to expected format
2. **Testability**: We can automate Schema validation testing, no manual checks needed
3. **Extensibility**: Adding new fields only requires updating Schema, not massive code changes

While this requires upfront Schema definition, it ensures system stability and maintainability."

---

### Trade-off 2: Deterministic Rules Engine vs Pure LLM Decision (30 seconds)

**Bullet Point Script**:
"The second trade-off is deterministic rules engine."

**Detailed Script**:
"LLM can make decisions, but it's a black box - not auditable. We had two options:
- Option A: Rely entirely on LLM decisions - flexible but not auditable
- Option B: Use deterministic rules engine for secondary validation - auditable but requires rule maintenance

**We chose deterministic rules engine** because:
1. **Auditability**: Every decision has explicit rule basis, fully traceable
2. **Explainability**: When issues arise, we know exactly which rule triggered the decision
3. **Compliance**: Financial scenarios require auditable decision processes, pure LLM cannot satisfy

While it requires maintaining rule code, this meets KYC compliance requirements and gives us better system control."

---

### Trade-off 3: Per-File Isolation vs Batch Processing (25 seconds)

**Bullet Point Script**:
"The third trade-off is Per-File Isolation."

**Detailed Script**:
"When processing batch files, we had two options:
- Option A: Atomic batch processing - all succeed or all fail
- Option B: Per-File Isolation - each file processed independently, single failure doesn't affect others

**We chose Per-File Isolation** because:
1. **Fault Isolation**: One file failure doesn't affect the entire batch, improving system availability
2. **High Availability**: Partial failures don't affect the whole, system is more stable
3. **Parallel Processing**: Independent files can be processed in parallel for efficiency

While it requires additional error handling logic, this ensures high availability in production."

---

### Trade-off 4: Layered Validation vs Strict/Loose Validation (30 seconds)

**Bullet Point Script**:
"The fourth trade-off is layered validation strategy."

**Detailed Script**:
"Validation strategy directly affects Error Rate and automation rate. We had two extreme options:
- Option A: Strict validation - high Error Rate but good output quality
- Option B: Loose validation - low Error Rate but unstable output

**We chose Layered Validation + Configurable Thresholds**:
1. **Preprocessing Strict**: Only check hard requirements like format and size to prevent invalid data entry
2. **Post-inference Configurable**: Schema validation strictness adjustable via Feature Flags
3. **Post-processing Loose**: Business rules don't fail directly, but escalate to human review - ensures output quality without increasing Error Rate

This design finds the balance point between Error Rate and automation rate."

---

### Trade-off 5: Auto Retry + Fallback vs Fast Fail (25 seconds)

**Bullet Point Script**:
"The fifth trade-off is auto retry and fallback mechanism."

**Detailed Script**:
"When external services fail, we had two options:
- Option A: Fast fail - simple but low success rate
- Option B: Auto retry + fallback - complex but fault-tolerant

**We chose auto retry + fallback** because:
1. **Fault Tolerance**: Transient failures (like network jitter) can be recovered through retry
2. **Availability**: When primary service is unavailable, fallback ensures system continues running
3. **User Experience**: Automatic handling of transient faults improves user perception

While it increases system complexity, it significantly improves system reliability and user experience."

---

### Summary (15 seconds)

**Script**:
"Through these five trade-offs, we built an **auditable, testable, and highly available** system:
- Schema-First ensures output stability
- Rules engine guarantees auditability
- Per-File Isolation provides fault isolation
- Layered validation balances Error Rate and automation rate
- Auto retry and fallback improve fault tolerance

These decisions collectively meet KYC scenario's **compliance and reliability requirements**."

---

## Concise Version (5 minutes total)

### Quick Version (20 seconds per Trade-off)

**Opening** (5 seconds):
"When designing the KYC system, we made several key trade-offs:"

**Trade-off 1** (20 seconds):
"Schema-First vs Direct LLM Output. We chose Schema-First for maintainability, testability, and extensibility. While it requires upfront Schema definition, it ensures output stability."

**Trade-off 2** (20 seconds):
"Deterministic Rules Engine vs Pure LLM Decision. We chose rules engine for auditability and explainability. Financial scenarios require auditable decision processes."

**Trade-off 3** (20 seconds):
"Per-File Isolation vs Batch Processing. We chose Per-File Isolation for fault isolation and high availability. Single file failure doesn't affect the entire batch."

**Trade-off 4** (20 seconds):
"Layered Validation vs Strict/Loose Validation. We chose Layered Validation + Configurable Thresholds, balancing Error Rate and automation rate. Preprocessing strict, post-processing loose with human escalation."

**Trade-off 5** (20 seconds):
"Auto Retry + Fallback vs Fast Fail. We chose auto retry + fallback for fault tolerance and availability. Transient failures can be recovered, fallback when primary service unavailable."

**Closing** (15 seconds):
"Through these trade-offs, we built an auditable, testable, and highly available system that meets KYC scenario's compliance and reliability requirements."

**Total Duration: ~2 minutes (quick version) or ~5 minutes (detailed version)**
