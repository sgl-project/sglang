# 实战训练指南：System Design Primer + KYC 项目

**目标**：7天内通过实战训练提升 System Design 面试能力  
**策略**：不全部看，聚焦面试核心内容，用 KYC 项目实战

---

## 📚 System Design Primer 是什么？

### 项目概述

**GitHub**：https://github.com/donnemartin/system-design-primer  
**作者**：Donne Martin  
**Star 数**：245k+（非常知名）

### 核心内容

1. **系统设计基础**（基础概念）
   - 可扩展性（Scalability）、延迟（Latency）、可用性（Availability）
   - 性能指标、CAP 定理、一致性模型

2. **系统设计模式**（设计模式）
   - 负载均衡（Load Balancing）、缓存（Caching）、数据库分片（Database Sharding）
   - 容错（Fault Tolerance）、限流（Rate Limiting）、熔断（Circuit Breaker）

3. **实际系统案例**（实际案例）
   - Twitter、Uber、Instagram、YouTube 等的设计案例

4. **面试准备**（面试指南）
   - 面试流程、常见问题、评分标准

### 为什么不能全部看？

**问题**：
- ❌ 内容太多（几百页）
- ❌ 时间不够（面试前时间有限）
- ❌ 容易分散注意力（不是所有内容都针对面试）

**策略**：
- ✅ 聚焦面试核心内容
- ✅ 用 KYC 项目实战训练
- ✅ 从实战出发，快速迭代

---

## 🎯 实战训练策略：7天速成计划

### 核心理念

**不是"学完所有内容"，而是"掌握面试需要的核心能力"**

**面试官关心的**：
1. ✅ 能否设计一个可扩展的系统（L0/L1/L2 指标）
2. ✅ 能否处理高并发/高可用（保护策略）
3. ✅ 能否快速定位和解决问题（可观测性）
4. ✅ 能否低风险地演进系统（发布策略）

**System Design Primer 的价值**：
- 提供**设计模式和概念**（理论）
- 提供**实际系统案例**（参考）

**KYC 项目的价值**：
- 提供**真实项目经验**（实践）
- 提供**可展示的设计亮点**（证据）

---

## 📅 7天实战训练计划

### Day 1｜指标体系（L0/L1/L2）

**System Design Primer 需要看的内容**：
- [ ] **性能指标**（Performance）：延迟、吞吐、可用性
- [ ] **CAP 定理**（CAP Theorem）：一致性、可用性、分区容错
- [ ] **可用性**（Availability）：99.9% vs 99.99% 的区别

**KYC 项目实战**：
- [ ] 用 KYC 项目填充 L0/L1/L2 指标
- [ ] 理解为什么用 p95/p99 而不是平均值
- [ ] 理解 Error Budget 的作用

**输出**：
- ✅ `KYC_DAY01_METRICS_CARD.md`（已完成）
- ✅ 能够说出："我们系统用三层指标度量成功：L0稳定性99%、L1每单节省5分钟、L2变更失败率<5%"

**时间**：2-3 小时

---

### Day 2｜可观测性（Metrics/Logs/Traces）

**System Design Primer 需要看的内容**：
- [ ] **监控**（Monitoring）：系统监控、应用监控
- [ ] **日志**（Logging）：结构化日志、日志聚合
- [ ] **追踪**（Tracing）：分布式追踪、链路追踪

**KYC 项目实战**：
- [ ] 设计 KYC 项目的可观测性方案（Metrics/Logs/Traces）
- [ ] 设计 Dashboard 草图
- [ ] 理解如何用 trace_id 关联日志

**输出**：
- ✅ `KYC_DAY02_OBSERVABILITY.md`
- ✅ 能够说出："我们用 Metrics/Logs/Traces 三类信号，通过 trace_id 关联，快速定位根因"

**时间**：2-3 小时

---

### Day 3｜回归门禁（Golden Set + Eval）

**System Design Primer 需要看的内容**：
- [ ] **测试**（Testing）：单元测试、集成测试、性能测试
- [ ] **质量保证**（Quality Assurance）：测试策略、发布门禁

**KYC 项目实战**：
- [ ] 设计 KYC 项目的 Golden Set（50-200 条测试用例）
- [ ] 设计发布门禁（通过阈值才能发布）
- [ ] 结合现有的 `tests/test_rules.py`, `tests/test_validators.py`

**输出**：
- ✅ `KYC_DAY03_REGRESSION.md`
- ✅ 能够说出："每次发布前跑 Golden Set，通过门禁才能发布，确保改动不会把系统搞坏"

**时间**：3-4 小时

---

### Day 4｜发布策略（Feature Flag + Canary + Rollback）

**System Design Primer 需要看的内容**：
- [ ] **发布策略**（Deployment Strategies）：蓝绿部署、滚动更新、Canary 发布
- [ ] **版本控制**（Versioning）：API 版本化、向后兼容

**KYC 项目实战**：
- [ ] 设计 KYC 项目的 Feature Flag + Canary 发布策略
- [ ] 定义回滚条件（Schema Fail × 2 或 p95 +20% 立即回滚）
- [ ] 结合现有的 Schema-First 设计（`schema_version = "v1"`）

**输出**：
- ✅ `KYC_DAY04_ROLLOUT_AND_ROLLBACK.md`
- ✅ 能够说出："我们用 Feature Flag + Canary 发布，1%→5%→25%→100%，每步观察指标，异常立即回滚"

**时间**：2-3 小时

---

### Day 5｜保护策略（限流/熔断/重试/降级/幂等）

**System Design Primer 需要看的内容**：
- [ ] **容错**（Fault Tolerance）：重试、熔断、降级
- [ ] **限流**（Rate Limiting）：令牌桶、漏桶、滑动窗口
- [ ] **幂等性**（Idempotency）：幂等键、去重

**KYC 项目实战**：
- [ ] 完善 KYC 项目的保护策略矩阵
- [ ] 基于现有的 `rate_limiter.py` 和 `backoff_retry`
- [ ] 补充熔断、降级、幂等的实现

**输出**：
- ✅ `KYC_DAY05_PROTECTION_MATRIX.md`
- ✅ 能够说出："我们设计了限流/熔断/重试/降级/幂等五层保护策略，确保失败可控、可恢复"

**时间**：3-4 小时

---

### Day 6｜事故响应（Runbook + Postmortem）

**System Design Primer 需要看的内容**：
- [ ] **事故响应**（Incident Response）：On-Call、事故处理流程
- [ ] **复盘**（Postmortem）：事故复盘、根因分析

**KYC 项目实战**：
- [ ] 编写 KYC 项目的 Runbook（告警触发→查看 Dashboard→定位根因→快速止血）
- [ ] 设计 Postmortem 模板（时间线、根因、行动项）
- [ ] 结合现有的错误分类（`errors.py`）和 trace_id

**输出**：
- ✅ `KYC_DAY06_RUNBOOK.md` + `KYC_DAY06_POSTMORTEM.md`
- ✅ 能够说出："我们有完整的 Runbook，告警触发→查看 Dashboard→定位根因→快速止血"

**时间**：3-4 小时

---

### Day 7｜面试固化（30秒/2分钟/5分钟话术）

**System Design Primer 需要看的内容**：
- [ ] **面试流程**（Interview Process）：面试步骤、评分标准
- [ ] **常见问题**（Common Questions）：如何回答设计问题

**KYC 项目实战**：
- [ ] 把前 6 天的内容串成 30 秒 / 2 分钟 / 5 分钟话术
- [ ] 练习表达（每天 3/2/1 次）
- [ ] 模拟面试（找朋友/同事练习）

**输出**：
- ✅ `KYC_DAY07_INTERVIEW_SCRIPT.md`
- ✅ 能够流畅说出 5 分钟版本，覆盖所有关键点
- ✅ 能够应对常见问题（"能详细说说 XXX 吗？"、"如果 XXX 怎么办？"）

**时间**：2-3 小时（+ 持续练习）

---

## 🔥 实战训练方法

### 方法 1：快速定位 + 实战应用

**步骤**：
1. **快速浏览** System Design Primer 的对应章节（15-30分钟）
2. **找出核心概念**（设计模式、设计原则）
3. **用 KYC 项目实战**（填充文档、设计方案）
4. **理解背后的原理**（为什么这样设计）

**示例（Day 5：保护策略）**：
1. **快速浏览**：System Design Primer 的"容错"和"限流"章节（20分钟）
2. **找出核心概念**：重试、熔断、降级、限流、幂等
3. **用 KYC 项目实战**：
   - 查看现有的 `rate_limiter.py`（限流）
   - 查看现有的 `backoff_retry`（重试）
   - 补充熔断、降级、幂等的设计
4. **理解原理**：为什么需要这五层保护？如何验证效果？

---

## 🎯 System Design Primer 核心章节速查

### 必看章节（按优先级）

#### 优先级 1（必须看，Day 1-3）

1. **性能指标**（Performance）
   - 延迟（Latency）、吞吐（Throughput）、可用性（Availability）
   - 为什么用 p95/p99 而不是平均值

2. **监控**（Monitoring）
   - 系统监控、应用监控
   - Metrics/Logs/Traces 三类信号

3. **测试**（Testing）
   - 单元测试、集成测试、性能测试
   - 发布门禁（Release Gate）

#### 优先级 2（建议看，Day 4-5）

4. **发布策略**（Deployment Strategies）
   - 蓝绿部署、滚动更新、Canary 发布
   - 回滚策略

5. **容错**（Fault Tolerance）
   - 重试、熔断、降级
   - 限流（Rate Limiting）

#### 优先级 3（有时间看，Day 6-7）

6. **事故响应**（Incident Response）
   - On-Call、事故处理流程
   - 复盘（Postmortem）

7. **面试准备**（Interview Preparation）
   - 面试流程、常见问题
   - 评分标准

### 可选章节（有时间再看）

- **实际系统案例**（Designing a System）：Twitter、Uber、Instagram
  - 用于参考设计思路，但不是必须
  - 重点是自己能设计 KYC 项目

- **数据库设计**（Database）：SQL vs NoSQL、数据库分片
  - KYC 项目暂不涉及，可跳过

- **缓存**（Caching）：缓存策略、缓存更新
  - KYC 项目暂不涉及，可跳过

---

## 💡 实战训练技巧

### 技巧 1：用 KYC 项目作为载体

**不是"学 System Design Primer"，而是"用 System Design Primer 来完善 KYC 项目"**

**方法**：
1. **看 System Design Primer** 的对应章节（理解概念）
2. **用 KYC 项目填充**（实战应用）
3. **理解背后的原理**（为什么这样设计）

**示例**：
- System Design Primer 说："用 p95/p99 而不是平均值"
- KYC 项目实战："我们的 p95=8.5s, p99=12s，为什么这样设计？"
- 理解原理："平均值被极端值拉高，p95/p99 反映真实用户体验"

### 技巧 2：从问题出发，找答案

**不是"看完所有内容"，而是"遇到问题，找答案"**

**方法**：
1. **遇到问题**（例如：如何设计限流？）
2. **快速定位** System Design Primer 的对应章节
3. **理解概念**（令牌桶、漏桶、滑动窗口）
4. **用 KYC 项目实战**（设计限流策略）

**示例**：
- **问题**：如何设计限流？
- **System Design Primer**：查看"Rate Limiting"章节（10分钟）
- **理解概念**：令牌桶、漏桶、滑动窗口
- **KYC 项目实战**：查看现有的 `rate_limiter.py`，理解实现

### 技巧 3：聚焦面试需要的能力

**不是"学完所有内容"，而是"掌握面试需要的核心能力"**

**面试官关心的**：
1. ✅ 能否设计一个可扩展的系统（L0/L1/L2 指标）
2. ✅ 能否处理高并发/高可用（保护策略）
3. ✅ 能否快速定位和解决问题（可观测性）
4. ✅ 能否低风险地演进系统（发布策略）

**System Design Primer 的价值**：
- 提供**设计模式和概念**（理论）
- 提供**实际系统案例**（参考）

**KYC 项目的价值**：
- 提供**真实项目经验**（实践）
- 提供**可展示的设计亮点**（证据）

---

## 📋 实战训练检查清单

### Day 1：指标体系
- [ ] 快速浏览 System Design Primer 的"性能指标"章节（15分钟）
- [ ] 用 KYC 项目填充 L0/L1/L2 指标
- [ ] 理解为什么用 p95/p99 而不是平均值
- [ ] 理解 Error Budget 的作用
- [ ] **输出**：`KYC_DAY01_METRICS_CARD.md`

### Day 2：可观测性
- [ ] 快速浏览 System Design Primer 的"监控"章节（20分钟）
- [ ] 设计 KYC 项目的可观测性方案（Metrics/Logs/Traces）
- [ ] 设计 Dashboard 草图
- [ ] **输出**：`KYC_DAY02_OBSERVABILITY.md`

### Day 3：回归门禁
- [ ] 快速浏览 System Design Primer 的"测试"章节（20分钟）
- [ ] 设计 KYC 项目的 Golden Set（50-200 条）
- [ ] 设计发布门禁（通过阈值）
- [ ] **输出**：`KYC_DAY03_REGRESSION.md`

### Day 4：发布策略
- [ ] 快速浏览 System Design Primer 的"发布策略"章节（20分钟）
- [ ] 设计 KYC 项目的 Feature Flag + Canary 发布策略
- [ ] 定义回滚条件
- [ ] **输出**：`KYC_DAY04_ROLLOUT_AND_ROLLBACK.md`

### Day 5：保护策略
- [ ] 快速浏览 System Design Primer 的"容错"和"限流"章节（30分钟）
- [ ] 完善 KYC 项目的保护策略矩阵
- [ ] 基于现有的 `rate_limiter.py` 和 `backoff_retry`
- [ ] **输出**：`KYC_DAY05_PROTECTION_MATRIX.md`

### Day 6：事故响应
- [ ] 快速浏览 System Design Primer 的"事故响应"章节（20分钟）
- [ ] 编写 KYC 项目的 Runbook
- [ ] 设计 Postmortem 模板
- [ ] **输出**：`KYC_DAY06_RUNBOOK.md` + `KYC_DAY06_POSTMORTEM.md`

### Day 7：面试固化
- [ ] 快速浏览 System Design Primer 的"面试准备"章节（20分钟）
- [ ] 把前 6 天的内容串成 30 秒 / 2 分钟 / 5 分钟话术
- [ ] 练习表达（每天 3/2/1 次）
- [ ] **输出**：`KYC_DAY07_INTERVIEW_SCRIPT.md`

---

## 🎯 System Design Primer 快速导航

### 核心章节链接

1. **性能指标**（Performance）
   - 路径：`README.md` → `Scalability` → `Performance`
   - 内容：延迟、吞吐、可用性、为什么用 p95/p99

2. **监控**（Monitoring）
   - 路径：`README.md` → `Scalability` → `Monitoring`
   - 内容：系统监控、应用监控、Metrics/Logs/Traces

3. **测试**（Testing）
   - 路径：`README.md` → `Scalability` → `Testing`
   - 内容：单元测试、集成测试、性能测试、发布门禁

4. **发布策略**（Deployment Strategies）
   - 路径：`README.md` → `Scalability` → `Deployment`
   - 内容：蓝绿部署、滚动更新、Canary 发布、回滚策略

5. **容错**（Fault Tolerance）
   - 路径：`README.md` → `Scalability` → `Fault Tolerance`
   - 内容：重试、熔断、降级、限流

6. **事故响应**（Incident Response）
   - 路径：`README.md` → `Scalability` → `Incident Response`
   - 内容：On-Call、事故处理流程、复盘

7. **面试准备**（Interview Preparation）
   - 路径：`README.md` → `Appendix` → `Interview Process`
   - 内容：面试流程、常见问题、评分标准

### 实际系统案例（参考）

**Twitter**：
- 路径：`README.md` → `Designing a System` → `Design Twitter`
- 用途：参考设计思路（高并发、实时性）

**Uber**：
- 路径：`README.md` → `Designing a System` → `Design Uber`
- 用途：参考设计思路（地理位置、实时匹配）

**Instagram**：
- 路径：`README.md` → `Designing a System` → `Design Instagram`
- 用途：参考设计思路（图片存储、feed 生成）

**注意**：这些案例用于参考设计思路，不是必须看。重点是自己能设计 KYC 项目。

---

## 🚀 实战训练流程

### 每天的训练流程

```
1. 快速浏览 System Design Primer 的对应章节（15-30分钟）
   ↓
2. 找出核心概念（设计模式、设计原则）
   ↓
3. 用 KYC 项目实战（填充文档、设计方案）
   ↓
4. 理解背后的原理（为什么这样设计）
   ↓
5. 输出文档（KYC_DAYXX_XXX.md）
```

### 时间分配

**每天 2-4 小时**：
- **System Design Primer**：15-30分钟（快速浏览）
- **KYC 项目实战**：1.5-3小时（填充文档、设计方案）
- **理解原理**：30分钟（为什么这样设计）

**总时间**：7天 × 2-4小时 = 14-28小时

---

## 💡 实战训练示例

### 示例：Day 5 保护策略实战

**Step 1：快速浏览 System Design Primer（20分钟）**
- 打开：`README.md` → `Scalability` → `Fault Tolerance`
- 快速浏览：重试、熔断、降级、限流的概念

**Step 2：找出核心概念（10分钟）**
- 重试：指数退避、最大重试次数
- 熔断：快速失败、状态机（关闭/打开/半开）
- 降级：备用方案、功能降级
- 限流：令牌桶、漏桶、滑动窗口

**Step 3：用 KYC 项目实战（2小时）**
- 查看现有的 `rate_limiter.py`（限流）
- 查看现有的 `backoff_retry`（重试）
- 补充熔断、降级、幂等的设计
- 填充 `KYC_DAY05_PROTECTION_MATRIX.md`

**Step 4：理解原理（30分钟）**
- 为什么需要这五层保护？
- 如何验证效果？（压测、监控）
- 与 KYC 项目的实际场景结合

**输出**：
- ✅ `KYC_DAY05_PROTECTION_MATRIX.md`
- ✅ 能够说出："我们设计了限流/熔断/重试/降级/幂等五层保护策略，确保失败可控、可恢复"

---

## 📚 参考资源

### System Design Primer

- **GitHub**：https://github.com/donnemartin/system-design-primer
- **核心章节**：见上面的"快速导航"部分
- **使用方式**：快速浏览，聚焦面试核心内容

### KYC 项目

- **GitHub**：https://github.com/Nickcp39/kyc_pov/tree/main
- **设计文档**：`DESIGN.md`
- **关键代码**：`src/schemas.py`, `src/rules.py`, `src/pipeline.py`, `src/rate_limiter.py`

### 其他资源

- **Google SRE Book**：https://sre.google/workbook/
- **Martin Fowler 的博客**：https://martinfowler.com/
  - 设计模式、重构、架构设计

---

## 🎯 总结

### 实战训练的核心

1. **不是"学完所有内容"，而是"掌握面试需要的核心能力"**
2. **不是"看完 System Design Primer"，而是"用 System Design Primer 来完善 KYC 项目"**
3. **不是"理论为主"，而是"实战为主，理论为辅"**

### System Design Primer 的价值

- ✅ 提供**设计模式和概念**（理论）
- ✅ 提供**实际系统案例**（参考）
- ✅ **快速定位**面试需要的核心内容

### KYC 项目的价值

- ✅ 提供**真实项目经验**（实践）
- ✅ 提供**可展示的设计亮点**（证据）
- ✅ **实战训练**的最佳载体

### 7天训练计划

- ✅ **Day 1-3**：指标体系、可观测性、回归门禁（基础）
- ✅ **Day 4-5**：发布策略、保护策略（核心）
- ✅ **Day 6-7**：事故响应、面试固化（应用）

---

## 🚀 开始实战训练

**第 1 步**：快速浏览 System Design Primer 的对应章节（15-30分钟）  
**第 2 步**：用 KYC 项目实战（填充文档、设计方案）  
**第 3 步**：理解背后的原理（为什么这样设计）  
**第 4 步**：输出文档（KYC_DAYXX_XXX.md）

**记住**：不是"学完所有内容"，而是"掌握面试需要的核心能力"。

**加油！** 🎉
