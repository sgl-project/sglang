# KYC 项目 System Design 面试训练指南

**作者**：Yanda Cheng  
**项目链接**：https://github.com/Nickcp39/kyc_pov/tree/main  
**开始日期**：2025-01  

---

## 🎯 为什么用 KYC 项目来训练 System Design？

### 你的 KYC 项目已经具备了 Senior 级别的设计思维

1. **Schema-First 设计** → 可维护性、可扩展性
2. **确定性规则引擎** → 可审计性、可测试性
3. **Per-File Isolation** → 故障隔离、高可用性
4. **标准化错误处理** → 可观测性、快速定位
5. **Rate Limiting + Retry** → 保护策略、容错性
6. **Privacy-Aware Logging** → 合规性、安全性

### 这正好对应 System Design 面试的核心考察点

| KYC 项目设计 | System Design 面试要点 | 对应天数 |
|-------------|----------------------|---------|
| Schema-First + Validators | 可回归、可测试（Golden Set + 门禁） | Day 3 |
| Deterministic Rules | 可审计性（Metrics + Trace） | Day 1, Day 2 |
| Per-File Isolation | 故障隔离（保护策略） | Day 5 |
| Error Taxonomy | 可观测性（Metrics/Logs/Traces） | Day 2 |
| Rate Limiter + Retry | 保护策略（限流/熔断/重试） | Day 5 |
| Privacy-Aware Logging | 合规性（Runbook + Postmortem） | Day 6 |

---

## 📚 学习流程（7天强化计划）

### 总体流程

```
Day 1: 指标体系 (L0/L1/L2)
    ↓
Day 2: 可观测性 (Metrics/Logs/Traces)
    ↓
Day 3: 回归门禁 (Golden Set + Eval)
    ↓
Day 4: 发布策略 (Feature Flag + Canary + Rollback)
    ↓
Day 5: 保护策略 (限流/熔断/重试/降级/幂等)
    ↓
Day 6: 事故响应 (Runbook + Postmortem)
    ↓
Day 7: 面试固化 (30秒/2分钟/5分钟话术)
```

---

## 🚀 第 1 步：理解你的 KYC 项目（前置准备，1-2小时）

### 必读材料

1. **KYC 项目 README**
   - 项目地址：https://github.com/Nickcp39/kyc_pov/tree/main
   - 重点：理解整体架构和数据流

2. **DESIGN.md**
   - 理解你的设计决策（Schema-First、确定性规则、错误分类等）
   - 找出你在设计时的 trade-off 考虑

3. **关键代码文件**
   - `src/schemas.py` → 理解 Schema-First 设计
   - `src/rules.py` → 理解确定性规则引擎
   - `src/pipeline.py` → 理解 E2E 流程和 per-file isolation
   - `src/rate_limiter.py` → 理解保护策略
   - `src/errors.py` → 理解错误分类

### 绘制你的系统架构图（Mermaid 或手绘）

```mermaid
graph TD
    A[Batch Input] --> B[main.py]
    B --> C[pipeline.py]
    C --> D[preprocessor.py]
    D --> E[rate_limiter.py]
    E --> F[fw_client.py]
    F --> G[Fireworks API]
    G --> H[validators.py]
    H --> I[rules.py]
    I --> J[io_utils.py]
    J --> K[output_results/]
    J --> L[_summary.json]
```

### 找出你的设计亮点（准备面试时的关键点）

✅ **已具备的设计亮点**：
- [x] Schema-First 设计（`schemas.py`）
- [x] 确定性规则引擎（`rules.py`）
- [x] Per-File Isolation（`pipeline.py`）
- [x] 标准化错误处理（`errors.py`）
- [x] Rate Limiting + Retry（`rate_limiter.py`）
- [x] Privacy-Aware Logging（trace_id only）

⚠️ **需要补充的设计亮点**（这正是 7 天训练要补的）：
- [ ] 三层指标体系（L0/L1/L2）
- [ ] 可观测性方案（Metrics/Logs/Traces Dashboard）
- [ ] 回归门禁（Golden Set + 通过阈值）
- [ ] 发布策略（Feature Flag + Canary + Rollback）
- [ ] 完整的保护策略矩阵（限流/熔断/重试/降级/幂等）
- [ ] Runbook + Postmortem 模板

---

## 📅 第 2 步：7 天训练流程（按天执行）

### Day 1｜指标体系：把"成功"定义成可打分的三层指标（L0/L1/L2）

**学习目标**：
- 用 KYC 项目填充三层指标（L0 稳定性、L1 业务收益、L2 长期健康）
- 理解如何用指标度量系统价值

**学习步骤**：

1. **阅读模板文档**
   - 打开：`../Day01_METRICS_CARD.md`
   - 理解三层指标的含义

2. **用 KYC 项目填充指标**
   - 参考：`KYC_DAY01_METRICS_CARD_EXAMPLE.md`（已提供示例）
   - 基于你的实际项目数据填充：
     - L0：成功率、延迟（p95/p99）、错误率
     - L1：每单节省人审时间、成本节省、自动化率
     - L2：变更失败率、Auditability 覆盖率、PII 泄漏事件

3. **理解 Error Budget Policy**
   - 如何用错误预算平衡"发布速度 vs 稳定性"
   - 设定 KYC 项目的 SLO 和错误预算

**输出**：
- ✅ 完成 `KYC_DAY01_METRICS_CARD.md`（你自己的版本）
- ✅ 能够说出："我们系统用三层指标度量成功：L0稳定性99%、L1每单节省5分钟、L2变更失败率<5%"

**时间**：2-3 小时

---

### Day 2｜可观测性：把"我有监控"升级成"我能定位根因"

**学习目标**：
- 设计 KYC 项目的可观测性方案（Metrics/Logs/Traces）
- 理解如何用三类信号快速定位问题

**学习步骤**：

1. **阅读模板文档**
   - 打开：`../Day02_OBSERVABILITY.md`
   - 理解三类信号（Metrics/Logs/Traces）

2. **设计 KYC 项目的可观测性方案**
   - **Metrics**：
     - RPS（batch processing rate）
     - p95/p99 Latency（单文档处理时间）
     - Error Rate（基于 `errors.py` 的错误分类）
     - Schema Validation Fail Rate
     - Rate Limit Trigger Rate
   
   - **Logs**：
     - 结构化日志（trace_id、fw_request_id、model、tokens、latency、error_code）
     - **不记录**：base64 image、prompt content、extracted PII（Privacy-Aware）
   
   - **Traces**：
     - 一个文档的完整处理链路：
       - Span 1: Preprocess（image loading/normalize）
       - Span 2: Rate Limit Acquire
       - Span 3: Fireworks API Call
       - Span 4: Schema Validation
       - Span 5: Deterministic Rules
       - Span 6: Save Result

3. **设计 Dashboard 草图**
   - On-Call Dashboard（实时监控）
   - Business Health Dashboard（业务指标）
   - Tracing Dashboard（链路追踪）

**输出**：
- ✅ 完成 `KYC_DAY02_OBSERVABILITY.md`
- ✅ 能够说出："我们用 Metrics/Logs/Traces 三类信号，通过 trace_id 关联，快速定位根因"

**时间**：2-3 小时

---

### Day 3｜回归与门禁：把 AI 系统变成"可回测工程"

**学习目标**：
- 建立 KYC 项目的回归测试集（Golden Set）
- 设计发布门禁（通过阈值才能发布）

**学习步骤**：

1. **阅读模板文档**
   - 打开：`../Day03_REGRESSION.md`
   - 打开：`../Day03_EVAL_REPORT_TEMPLATE.md`

2. **设计 KYC 项目的 Golden Set**
   - **场景分类**：
     - 正常场景（20%）：清晰、标准格式的 ID
     - 边界场景（30%）：模糊、遮挡、低质量
     - 异常场景（30%）：版式变化、多页、复杂布局
     - 长尾场景（20%）：罕见格式、特殊字符
   
   - **Golden Set 规模**：50-200 条测试用例

3. **设计发布门禁**
   - **门禁指标**：
     - Schema Pass Rate > 95%
     - 字段级准确率 > 90%（critical fields: full_name, date_of_birth, document_number, expiry_date, issuing_country）
     - Fallback Rate < 5%
     - 成本上限：$0.002 / request（tokens < threshold）

4. **与现有测试结合**
   - 你的 `tests/test_rules.py` → 确定性规则回归
   - 你的 `tests/test_validators.py` → Schema 验证回归
   - 补充 E2E 集成测试（真实 API 调用）

**输出**：
- ✅ 完成 `KYC_DAY03_REGRESSION.md`
- ✅ 完成 `KYC_DAY03_EVAL_REPORT_TEMPLATE.md`
- ✅ 能够说出："每次发布前跑 Golden Set，通过门禁才能发布，确保改动不会把系统搞坏"

**时间**：3-4 小时

---

### Day 4｜发布策略：Feature Flag + Canary，把"上线"变成可控实验

**学习目标**：
- 设计 KYC 项目的灰度发布策略
- 定义回滚条件和流程

**学习步骤**：

1. **阅读模板文档**
   - 打开：`../Day04_ROLLOUT_AND_ROLLBACK.md`

2. **设计 KYC 项目的发布策略**（PoV → Production 规划）
   - **Feature Flags**：
     - `model_version`：切换模型（Qwen2.5-VL-32B vs 其他）
     - `prompt_version`：切换 prompt
     - `validator_strictness`：调整验证严格程度（high/medium/low）
   
   - **Canary 发布**：
     - 1% → 5% → 25% → 100%
     - 每步观察：p95、Error Rate、Schema Fail Rate、Cost
   
   - **回滚条件**：
     - Schema Fail Rate × 2 立即回滚
     - p95 + 20% 立即回滚
     - Error Rate > 5% 立即回滚

3. **结合你的设计**
   - 你的 Schema-First 设计 → 版本化发布（`schema_version = "v1"`）
   - 你的确定性规则 → 可以按规则版本发布

**输出**：
- ✅ 完成 `KYC_DAY04_ROLLOUT_AND_ROLLBACK.md`
- ✅ 能够说出："我们用 Feature Flag + Canary 发布，1%→5%→25%→100%，每步观察指标，异常立即回滚"

**时间**：2-3 小时

---

### Day 5｜保护策略：Engineering for Failure（免疫系统）

**学习目标**：
- 完善 KYC 项目的保护策略矩阵
- 理解限流/熔断/重试/降级/幂等的触发→动作→验证

**学习步骤**：

1. **阅读模板文档**
   - 打开：`../Day05_PROTECTION_MATRIX.md`

2. **完善你的保护策略**（基于现有设计）
   - **限流**（已有：`rate_limiter.py`）：
     - 触发：RPS > RPM_LIMIT 或并发 > threshold
     - 动作：返回 429，等待 token
     - 验证：p95 < 15s，429 rate < 5%
   
   - **重试**（已有：`backoff_retry`）：
     - 触发：可恢复错误（API_TIMEOUT, API_CONNECTION_ERROR, API_SERVER_ERROR）
     - 动作：指数退避重试（MAX_RETRIES = 3）
     - 验证：成功率提升
   
   - **熔断**（需要补充）：
     - 触发：Fireworks API 失败率 > 5%
     - 动作：快速失败，返回默认响应
     - 验证：延迟降低
   
   - **降级**（需要补充）：
     - 触发：主模型不可用 或 延迟 > threshold
     - 动作：OCR-only fallback 或 转人工审核
     - 验证：降级后成功率 > 80%
   
   - **幂等**（需要补充）：
     - 触发：重复 request_id（通过 `trace_id` 去重）
     - 动作：返回缓存结果
     - 验证：重复处理率 < 0.1%

**输出**：
- ✅ 完成 `KYC_DAY05_PROTECTION_MATRIX.md`
- ✅ 能够说出："我们设计了限流/熔断/重试/降级/幂等五层保护策略，确保失败可控、可恢复"

**时间**：3-4 小时

---

### Day 6｜事故响应：Runbook + Postmortem，把"出事"变成组织学习

**学习目标**：
- 编写 KYC 项目的 Runbook（运维手册）
- 设计 Postmortem 模板

**学习步骤**：

1. **阅读模板文档**
   - 打开：`../Day06_RUNBOOK.md`
   - 打开：`../Day06_POSTMORTEM.md`

2. **编写 KYC 项目的 Runbook**
   - **告警触发** → 查看 `_summary.json` 或 Dashboard
   - **判断严重性**：
     - Critical：Error Rate > 5% → 立即回滚
     - Warning：Schema Fail Rate > 2% → 触发降级
     - Info：Latency 升高 → 定位根因（Trace → Log）
   
   - **定位根因**：
     - 使用 `trace_id` 关联 Logs 和 Traces
     - 查看哪个 Span 慢（Preprocess / API Call / Validation / Rules）

3. **设计 Postmortem 模板**
   - 基于你的错误分类（`errors.py`）
   - 记录时间线、根因、行动项
   - 重点关注：Schema 变更、规则变更、模型切换

**输出**：
- ✅ 完成 `KYC_DAY06_RUNBOOK.md`
- ✅ 完成 `KYC_DAY06_POSTMORTEM.md`
- ✅ 能够说出："我们有完整的 Runbook，告警触发→查看 Dashboard→定位根因→快速止血，还有 Postmortem 模板把事故变成组织学习"

**时间**：3-4 小时

---

### Day 7｜面试固化：把系统讲成"低风险进化"的评审节奏

**学习目标**：
- 把前 6 天的内容串成 30 秒 / 2 分钟 / 5 分钟三套话术
- 练习在压力下的表达

**学习步骤**：

1. **阅读模板文档**
   - 打开：`../Day07_INTERVIEW_SCRIPT.md`

2. **用 KYC 项目填充面试脚本**
   - **30 秒版本**（Elevator Pitch）：
     - 系统介绍：KYC 文档智能提取系统
     - 核心指标：L0稳定性99%、L1每单节省5分钟、L2变更失败率<5%
     - 低风险进化：Feature Flag + Canary + 回归门禁
   
   - **2 分钟版本**（Overview）：
     - 系统介绍（30秒）
     - 指标体系（30秒）
     - 低风险进化（60秒）
   
   - **5 分钟版本**（Complete Story）：
     - Goal + 三层指标（1分钟）
     - 关键 trade-off（1分钟）
     - Failure modes + 兜底（1.5分钟）
     - Flag + canary + rollback + 回归门禁（1.5分钟）
     - 长期演进（1分钟）

3. **关键要点**（必须能说出）：
   - ✅ "我们用三层指标度量成功：L0稳定性、L1业务收益、L2长期健康"
   - ✅ "我们用 Schema-First + 确定性规则引擎，确保可审计性和可测试性"
   - ✅ "我们设计了限流/熔断/重试/降级/幂等五层保护策略"
   - ✅ "我们用 Feature Flag + Canary 发布，1%→5%→25%→100%，异常立即回滚"
   - ✅ "我们通过回归门禁（Golden Set + 通过阈值）确保改动不会把系统搞坏"

4. **练习**
   - **30 秒版本**：每天练习 3 次（持续 3 天）
   - **2 分钟版本**：每天练习 2 次（持续 3 天）
   - **5 分钟版本**：每天练习 1 次（持续 3 天）
   - **模拟面试**：找朋友/同事练习，让他们提问、打断

**输出**：
- ✅ 完成 `KYC_DAY07_INTERVIEW_SCRIPT.md`
- ✅ 能够流畅说出 5 分钟版本，覆盖所有关键点
- ✅ 能够应对常见问题（"能详细说说 XXX 吗？"、"如果 XXX 怎么办？"）

**时间**：2-3 小时（+ 持续练习）

---

## 🎯 学习优先级（如果时间有限）

### 如果只有 3 天

**Day 1**：指标体系（Day 1）→ 这是基础，必须掌握  
**Day 2**：可观测性（Day 2）→ 快速定位根因的能力  
**Day 3**：面试固化（Day 7）→ 把内容串成话术

### 如果只有 5 天

**Day 1**：指标体系（Day 1）  
**Day 2**：可观测性（Day 2）  
**Day 3**：保护策略（Day 5）→ 这是 Senior 的核心能力  
**Day 4**：发布策略（Day 4）→ 低风险进化的关键  
**Day 5**：面试固化（Day 7）

### 如果有 7 天（完整版）

按照原定计划，每天完成一个模块。

---

## 📝 学习检查清单

### 第 1 步：理解项目（1-2小时）
- [ ] 阅读 KYC 项目 README 和 DESIGN.md
- [ ] 理解关键代码文件（schemas.py, rules.py, pipeline.py）
- [ ] 绘制系统架构图
- [ ] 列出设计亮点（已具备 vs 需要补充）

### 第 2 步：7 天训练（每天 2-4 小时）

#### Day 1｜指标体系
- [ ] 阅读 Day01 模板
- [ ] 用 KYC 项目填充三层指标
- [ ] 理解 Error Budget Policy
- [ ] **输出**：`KYC_DAY01_METRICS_CARD.md`

#### Day 2｜可观测性
- [ ] 阅读 Day02 模板
- [ ] 设计 Metrics/Logs/Traces 方案
- [ ] 设计 Dashboard 草图
- [ ] **输出**：`KYC_DAY02_OBSERVABILITY.md`

#### Day 3｜回归门禁
- [ ] 阅读 Day03 模板
- [ ] 设计 Golden Set（50-200 条）
- [ ] 设计发布门禁（通过阈值）
- [ ] **输出**：`KYC_DAY03_REGRESSION.md` + `KYC_DAY03_EVAL_REPORT_TEMPLATE.md`

#### Day 4｜发布策略
- [ ] 阅读 Day04 模板
- [ ] 设计 Feature Flag + Canary 发布策略
- [ ] 定义回滚条件
- [ ] **输出**：`KYC_DAY04_ROLLOUT_AND_ROLLBACK.md`

#### Day 5｜保护策略
- [ ] 阅读 Day05 模板
- [ ] 完善限流/熔断/重试/降级/幂等矩阵
- [ ] 设计验证方案
- [ ] **输出**：`KYC_DAY05_PROTECTION_MATRIX.md`

#### Day 6｜事故响应
- [ ] 阅读 Day06 模板
- [ ] 编写 Runbook
- [ ] 设计 Postmortem 模板
- [ ] **输出**：`KYC_DAY06_RUNBOOK.md` + `KYC_DAY06_POSTMORTEM.md`

#### Day 7｜面试固化
- [ ] 阅读 Day07 模板
- [ ] 编写 30秒/2分钟/5分钟话术
- [ ] 练习表达（每天 3/2/1 次）
- [ ] **输出**：`KYC_DAY07_INTERVIEW_SCRIPT.md`

### 第 3 步：模拟面试（持续练习）
- [ ] 找朋友/同事模拟面试
- [ ] 录音/视频自己练习
- [ ] 持续改进表达

---

## 💡 关键学习要点

### 1. 你不是在"学大厂流程"，而是在"把你的深技术变成可托付的工业能力"

- 你的 KYC 项目已经有很强的技术深度
- 现在需要把这些深度用"工业能力"的语言表达出来
- 面试官要的不是"你懂技术"，而是"你能当 owner"

### 2. 核心逻辑：SRE 不是追求"永不失败"，而是"失败可控、可恢复"

- **SLO 不要求 100% 满足**，要允许 error budget
- **Canary/Feature Flags**：在真实流量下试错，但把伤害限制在很小范围
- **失败不可怕**，关键是可控、可恢复、可学习

### 3. 面试时要展示的：可度量、可回归、可灰度、可回滚、可复盘

- **可度量**：三层指标（L0/L1/L2）
- **可回归**：Golden Set + 门禁
- **可灰度**：Feature Flag + Canary
- **可回滚**：明确的回滚条件和流程
- **可复盘**：Runbook + Postmortem

---

## 📚 参考资源

### KYC 项目
- GitHub：https://github.com/Nickcp39/kyc_pov/tree/main
- 设计文档：`DESIGN.md`

### System Design 模板
- `../Day01_METRICS_CARD.md` - Day 1 模板
- `../Day02_OBSERVABILITY.md` - Day 2 模板
- `../Day03_REGRESSION.md` - Day 3 模板
- `../Day04_ROLLOUT_AND_ROLLBACK.md` - Day 4 模板
- `../Day05_PROTECTION_MATRIX.md` - Day 5 模板
- `../Day06_RUNBOOK.md` - Day 6 模板
- `../Day07_INTERVIEW_SCRIPT.md` - Day 7 模板

### 示例文档（参考）
- `KYC_DAY01_METRICS_CARD_EXAMPLE.md` - Day 1 示例（已填充）

### Google SRE 参考
- Google SRE Book: https://sre.google/workbook/
- SLO, SLI, SLAs: https://sre.google/workbook/slo/
- Error Budget Policy: https://sre.google/workbook/error-budget-policy/

---

## 🎯 成功标准

### 训练完成后，你应该能够：

1. **在 5 分钟内**讲清楚你的 KYC 系统：
   - Goal + 三层指标（L0/L1/L2）
   - 关键 trade-off
   - Failure modes + 兜底
   - Flag + canary + rollback + 回归门禁
   - 长期演进

2. **应对常见问题**：
   - "能详细说说 XXX 吗？" → 从脚本中抽取相关部分
   - "如果 XXX 怎么办？" → 引用保护策略
   - "为什么不这样设计？" → 引用 trade-off 分析

3. **展示 Senior 级别的能力**：
   - 不是"能写代码"，而是"能当 owner"
   - 不是"能解决问题"，而是"能设计可托付的系统"
   - 不是"能交付"，而是"能低风险进化"

---

## 🚀 开始学习

**第 1 步**：理解你的 KYC 项目（1-2 小时）  
**第 2 步**：从 Day 1 开始，每天完成一个模块  
**第 3 步**：持续练习表达，准备面试

**记住**：这 7 天不是在"学大厂流程"，而是在把你的深技术变成"可托付的工业能力"。

**加油！** 🎉
