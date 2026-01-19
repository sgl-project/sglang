# KYC 项目 vs 7days_speedup 计划：学习进度对比

---
doc_type: reference
layer: L0
scope_in:  对比 KYC 项目已学内容和 7days_speedup 计划，找出未覆盖的内容
scope_out:  具体学习内容（见各 Day 文件）
inputs:   (读者) 需求：了解学习进度，知道还有哪些内容需要学习
outputs:  学习进度对比表 + 未覆盖内容清单 + 下一步学习建议
entrypoints: [ 学习进度总览 ]
children: []
related: [ 7days_speedup 计划, KYC 项目学习内容 ]
---

## 📊 学习进度总览

### 7days_speedup 计划（9 个文件）

| Day | 主题 | 状态 | KYC 项目对应内容 |
|-----|------|------|-----------------|
| **Day 1** | 指标体系（L0/L1/L2） | ✅ **已完成** | A1 详细讲解、A3 业务收益与ROI详解 |
| **Day 2** | 可观测性（Observability） | ❌ **未覆盖** | 无对应文件 |
| **Day 3** | 回归测试（Regression） | ❌ **未覆盖** | 无对应文件 |
| **Day 3** | 评估报告模板（Eval Report） | ❌ **未覆盖** | 无对应文件 |
| **Day 4** | 发布与回滚（Rollout & Rollback） | ⚠️ **部分覆盖** | A2_B3 开发流程（有灰度发布，但不够详细） |
| **Day 5** | 保护矩阵（Protection Matrix） | ⚠️ **部分覆盖** | A1_B6 分层容错、A1_B9 降级策略（但不够系统化） |
| **Day 6** | 事后分析（Postmortem） | ❌ **未覆盖** | 无对应文件 |
| **Day 6** | 运行手册（Runbook） | ⚠️ **部分覆盖** | A4 告警响应机制（有 runbook 概念，但不够详细） |
| **Day 7** | 面试脚本（Interview Script） | ❌ **未覆盖** | 无对应文件 |

---

## ✅ 已完成内容（Day 1）

### Day 1：指标体系（L0/L1/L2）

**7days_speedup 要求**：
- ✅ 三层指标体系（L0/L1/L2）
- ✅ Error Budget Policy
- ✅ 指标监控 Dashboard 设计
- ✅ 指标收集与上报

**KYC 项目已覆盖**：
- ✅ **A1 详细讲解指标与测试.md**：完整的 L0/L1/L2 讲解
- ✅ **A3_METRICS_CARD_EXAMPLE.md**：指标卡片示例
- ✅ **A3 业务收益与ROI详解.md**：L1 业务收益详细讲解
- ✅ **A2 指标计算脚本示例.md**：指标计算方法

**覆盖度**：✅ **100%**（完全覆盖）

---

## ❌ 未覆盖内容（需要学习）

### Day 2：可观测性（Observability）

**7days_speedup 要求**：
- ❌ **三类信号框架**：Metrics、Logs、Traces
- ❌ **Dashboard 设计**：实时监控、业务收益、工程健康
- ❌ **根因定位流程**：Trace → Log → Metrics
- ❌ **告警设计**：告警规则、告警路由、告警降噪

**KYC 项目现状**：
- ⚠️ **A4 告警响应机制详解.md**：有告警相关内容，但缺少可观测性的系统化设计
- ❌ 缺少 Metrics/Logs/Traces 三类信号的完整设计
- ❌ 缺少 Dashboard 设计
- ❌ 缺少根因定位流程

**需要学习**：`Day02_OBSERVABILITY.md`

---

### Day 3：回归测试（Regression）

**7days_speedup 要求**：
- ❌ **Golden Set（黄金测试集）**：构建策略、分类、维护
- ❌ **回归测试流程**：Before/After 对比、指标门禁
- ❌ **Release Gate**：通过阈值、阻断机制

**KYC 项目现状**：
- ⚠️ **A2_B2_C2_D1 测试的设计原理与分层策略.md**：有测试分层，但缺少回归测试的专门讲解
- ❌ 缺少 Golden Set 的构建和维护
- ❌ 缺少回归测试的完整流程
- ❌ 缺少 Release Gate 的详细设计

**需要学习**：`Day03_REGRESSION.md`

---

### Day 3：评估报告模板（Eval Report）

**7days_speedup 要求**：
- ❌ **评估报告模板**：Schema Pass Rate、字段级准确率、延迟对比
- ❌ **Before/After 对比**：版本对比、指标对比
- ❌ **门禁决策**：通过/不通过的判断标准

**KYC 项目现状**：
- ❌ 完全未覆盖

**需要学习**：`Day03_EVAL_REPORT_TEMPLATE.md`

---

### Day 4：发布与回滚（Rollout & Rollback）

**7days_speedup 要求**：
- ⚠️ **Feature Flags**：功能开关、按维度放量
- ⚠️ **Canary 发布**：灰度策略、流量分配、监控指标
- ⚠️ **回滚策略**：回滚触发条件、回滚流程、回滚验证

**KYC 项目现状**：
- ⚠️ **A2_B3 从开发到用户使用的完整流程.md**：有灰度发布概念，但不够详细
- ⚠️ **A2_B3_C1_KYC功能加入苹果手表的完整流程示例.md**：有 Canary 发布示例，但缺少 Feature Flags 和详细策略
- ❌ 缺少 Feature Flags 的详细设计
- ❌ 缺少回滚策略的详细设计

**需要学习**：`Day04_ROLLOUT_AND_ROLLBACK.md`

---

### Day 5：保护矩阵（Protection Matrix）

**7days_speedup 要求**：
- ⚠️ **限流（Rate Limiting）**：触发条件、动作、验证
- ⚠️ **熔断（Circuit Breaker）**：触发条件、动作、验证
- ⚠️ **降级（Fallback）**：触发条件、动作、验证
- ⚠️ **超时（Timeout）**：触发条件、动作、验证
- ⚠️ **重试（Retry）**：触发条件、动作、验证

**KYC 项目现状**：
- ⚠️ **A1_B4_retry_error_rate.md**：有重试相关内容
- ⚠️ **A1_B6_layered_fault_tolerance.md**：有分层容错设计
- ⚠️ **A1_B9_fallback_rate.md** 和 **A1_B9_C1_fallback_strategy_types.md**：有降级策略
- ❌ 缺少系统化的保护矩阵（策略→触发→动作→验证）
- ❌ 缺少限流、熔断、超时的详细设计

**需要学习**：`Day05_PROTECTION_MATRIX.md`

---

### Day 6：事后分析（Postmortem）

**7days_speedup 要求**：
- ❌ **Postmortem 模板**：时间线、影响评估、根因分析、行动项
- ❌ **事故响应流程**：MTTD、MTTR、恢复验证
- ❌ **组织学习**：防止复发、经验积累

**KYC 项目现状**：
- ❌ 完全未覆盖

**需要学习**：`Day06_POSTMORTEM.md`

---

### Day 6：运行手册（Runbook）

**7days_speedup 要求**：
- ⚠️ **告警触发 → 快速止血流程**：查看 Dashboard、判断严重性、立即回滚/触发降级/定位根因
- ⚠️ **常见场景 Runbook**：高错误率、高延迟、队列堆积、API 超时
- ⚠️ **桌面演练**：模拟事故场景

**KYC 项目现状**：
- ⚠️ **A4 告警响应机制详解.md**：有告警相关内容，但缺少详细的 Runbook
- ❌ 缺少可执行的 Runbook 模板
- ❌ 缺少常见场景的处理流程

**需要学习**：`Day06_RUNBOOK.md`

---

### Day 7：面试脚本（Interview Script）

**7days_speedup 要求**：
- ❌ **30 秒版本**（Elevator Pitch）
- ❌ **2 分钟版本**（核心设计）
- ❌ **5 分钟版本**（完整系统设计）

**KYC 项目现状**：
- ❌ 完全未覆盖

**需要学习**：`Day07_INTERVIEW_SCRIPT.md`

---

## 📋 未覆盖内容清单

### 完全未覆盖（5 个）

1. ❌ **Day 2：可观测性（Observability）**
   - Metrics/Logs/Traces 三类信号
   - Dashboard 设计
   - 根因定位流程

2. ❌ **Day 3：回归测试（Regression）**
   - Golden Set 构建和维护
   - 回归测试流程
   - Release Gate 设计

3. ❌ **Day 3：评估报告模板（Eval Report）**
   - 评估报告模板
   - Before/After 对比
   - 门禁决策

4. ❌ **Day 6：事后分析（Postmortem）**
   - Postmortem 模板
   - 事故响应流程
   - 组织学习

5. ❌ **Day 7：面试脚本（Interview Script）**
   - 30 秒/2 分钟/5 分钟版本
   - 面试话术

---

### 部分覆盖（3 个）

1. ⚠️ **Day 4：发布与回滚（Rollout & Rollback）**
   - ✅ 有灰度发布概念
   - ❌ 缺少 Feature Flags 详细设计
   - ❌ 缺少回滚策略详细设计

2. ⚠️ **Day 5：保护矩阵（Protection Matrix）**
   - ✅ 有重试、降级、容错相关内容
   - ❌ 缺少系统化的保护矩阵
   - ❌ 缺少限流、熔断、超时的详细设计

3. ⚠️ **Day 6：运行手册（Runbook）**
   - ✅ 有告警响应机制
   - ❌ 缺少可执行的 Runbook 模板
   - ❌ 缺少常见场景的处理流程

---

## 🎯 下一步学习建议

### 优先级 1：核心缺失（必须学习）

1. **Day 2：可观测性（Observability）**
   - **重要性**：⭐⭐⭐⭐⭐
   - **原因**：可观测性是系统设计的基础，面试必考
   - **建议**：立即学习，结合 KYC 项目设计可观测性方案

2. **Day 4：发布与回滚（Rollout & Rollback）**
   - **重要性**：⭐⭐⭐⭐⭐
   - **原因**：灰度发布和回滚是生产环境的核心能力
   - **建议**：补充 Feature Flags 和回滚策略的详细设计

3. **Day 5：保护矩阵（Protection Matrix）**
   - **重要性**：⭐⭐⭐⭐
   - **原因**：系统保护机制是 Senior 工程师的核心能力
   - **建议**：系统化整理限流、熔断、超时等保护机制

---

### 优先级 2：重要补充（推荐学习）

4. **Day 3：回归测试（Regression）**
   - **重要性**：⭐⭐⭐⭐
   - **原因**：回归测试是保证系统稳定性的关键
   - **建议**：学习 Golden Set 构建和 Release Gate 设计

5. **Day 6：运行手册（Runbook）**
   - **重要性**：⭐⭐⭐⭐
   - **原因**：Runbook 是 On-Call 的核心工具
   - **建议**：补充详细的 Runbook 模板和常见场景

---

### 优先级 3：完善补充（可选学习）

6. **Day 3：评估报告模板（Eval Report）**
   - **重要性**：⭐⭐⭐
   - **原因**：评估报告是 AI 系统特有的需求
   - **建议**：如果需要做模型评估，可以学习

7. **Day 6：事后分析（Postmortem）**
   - **重要性**：⭐⭐⭐
   - **原因**：Postmortem 是组织学习的重要工具
   - **建议**：如果有生产事故，可以学习

8. **Day 7：面试脚本（Interview Script）**
   - **重要性**：⭐⭐⭐⭐
   - **原因**：面试表达是最终目标
   - **建议**：在完成前 6 天内容后，最后学习

---

## 📊 学习进度统计

### 总体进度

- ✅ **已完成**：1/9（11%）
- ⚠️ **部分完成**：3/9（33%）
- ❌ **未完成**：5/9（56%）

### 详细统计

| 类别 | 数量 | 占比 |
|------|------|------|
| ✅ 完全覆盖 | 1 | 11% |
| ⚠️ 部分覆盖 | 3 | 33% |
| ❌ 未覆盖 | 5 | 56% |

---

## 🎯 推荐学习路径

### 路径 1：快速补齐核心（3-4 天）

```
Day 1: Day 2 可观测性（Observability）
Day 2: Day 4 发布与回滚（Rollout & Rollback）
Day 3: Day 5 保护矩阵（Protection Matrix）
Day 4: Day 6 运行手册（Runbook）
```

### 路径 2：完整学习（7 天）

```
Day 1: Day 2 可观测性（Observability）
Day 2: Day 3 回归测试（Regression）
Day 3: Day 3 评估报告模板（Eval Report）
Day 4: Day 4 发布与回滚（Rollout & Rollback）
Day 5: Day 5 保护矩阵（Protection Matrix）
Day 6: Day 6 事后分析（Postmortem）+ Runbook
Day 7: Day 7 面试脚本（Interview Script）
```

---

## 💡 建议

### 立即行动

1. **优先学习 Day 2（可观测性）**
   - 这是系统设计的基础
   - 面试必考内容
   - 可以结合 KYC 项目设计可观测性方案

2. **补充 Day 4（发布与回滚）**
   - 完善灰度发布和回滚策略
   - 补充 Feature Flags 设计
   - 结合 KYC 项目实际场景

3. **系统化 Day 5（保护矩阵）**
   - 整理现有的保护机制
   - 补充缺失的保护策略
   - 形成系统化的保护矩阵

### 学习方式

- ✅ **结合 KYC 项目**：每学一个 Day，就为 KYC 项目设计对应的方案
- ✅ **实际应用**：不只是阅读，要实际设计和实现
- ✅ **面试准备**：最后用 Day 7 的面试脚本整合所有内容

---

## Links

| 类型 | 对象 |
|------|------|
| **Related** | 7days_speedup 计划、KYC 项目学习内容 |
