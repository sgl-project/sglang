# A1_B9：什么是回退率（Fallback Rate）？

---
doc_type: glossary
layer: L1
scope_in:  回退率的定义、公式、与降级策略的关系、在 KYC 项目中的应用
scope_out:  降级策略的具体实现（见 KYC_Day01_A1_B6_layered_fault_tolerance.md）；告警响应机制（见 KYC_Day01_A4_告警响应机制详解.md）
inputs:   总请求数、触发降级/回退的请求数
outputs:  回退率（百分比）；系统可用性保证；降级策略触发条件
entrypoints: [ Definition, 公式与目标 ]
children: [ KYC_Day01_A1_B9_C1_fallback_strategy_types.md（降级方案类型） ]
related: [ KYC_Day01_A1_B6_layered_fault_tolerance.md（分层容错，含降级方案）, KYC_Day01_A4_告警响应机制详解.md（自动降级策略）, KYC_Day01_A1_详细讲解_指标与测试.md（L0 稳定性指标） ]
owner: you
last_updated: 2025-01-01
---

## Definition（定义）

**回退率（Fallback Rate）**：触发降级/回退的请求数 / 总请求数。

- **含义**：系统在遇到问题时，自动切换到降级方案（fallback）的请求占比。
- **边界**：不包括直接失败的请求（直接失败计入 Error Rate，不计入 Fallback Rate）。

---

## 与降级策略（Fallback Strategy）的关系

### 什么是降级策略？

**降级策略（Fallback Strategy）**：当主服务失败或性能下降时，系统自动切换到备用方案，保证系统可用性。

**KYC 项目的降级策略示例**：

```
主服务失败 → 自动切换到降级方案
  - OCR-only（不调用 LLM，只做 OCR 提取）
  - 人工审核队列（不直接失败，转人工处理）
  - 缓存结果（不重新计算，使用缓存）
```

### 回退率 vs 错误率

| 指标 | 定义 | 含义 |
|------|------|------|
| **错误率（Error Rate）** | 直接失败的请求数 / 总请求数 | 系统无法处理，直接返回错误 |
| **回退率（Fallback Rate）** | 触发降级的请求数 / 总请求数 | 系统切换到降级方案，**不直接失败** |

**关键区别**：
- **错误率**：系统直接失败，用户收到错误响应
- **回退率**：系统切换到降级方案，用户收到降级响应（可能准确率低，但系统可用）

---

## 公式与目标（来自 A1 详细讲解）

### 公式

```
回退率 = 触发降级/回退的请求数 / 总请求数
```

### KYC 项目示例

- **当前值**：`0%`（PoV 阶段无降级策略）
- **未来规划**：低质量图片 → OCR-only fallback
- **目标**：`< 5%`（生产环境，参考学习指南）

### 为什么回退率重要？

1. **系统可用性**：回退率 > 0 意味着系统有降级保护，**系统不 down**
2. **用户体验**：即使主服务失败，用户仍能获得降级响应（可能准确率低，但比直接失败好）
3. **系统稳定性**：降级策略是"分层容错"（Defense in Depth）的重要组成部分

---

## 降级策略的触发条件

### KYC 项目的降级触发条件（来自 A4 告警响应机制）

1. **API 超时**：Fireworks API 调用超时（> 30s）→ OCR-only fallback
2. **成功率下降**：Success Rate < 阈值 → 自动降级
3. **延迟过高**：p95 > 阈值 → 自动降级
4. **错误率上升**：Error Rate > 阈值 → 自动降级

### 降级策略示例（来自 A4）

```python
# 自动降级逻辑（简化示例）
if success_rate < 0.95 or p95 > 15.0:
    fallback_strategy = determine_fallback_strategy()
    if fallback_strategy == "ocr_only":
        # OCR-only fallback（不调用 LLM）
        logger.warning("Switching to OCR-only fallback")
        result = ocr_only_extraction(image)
    elif fallback_strategy == "manual_review":
        # 转人工审核
        result = queue_for_manual_review(image)
```

---

## 回退率与系统可用性的关系

### 分层容错设计（来自 B6）

**核心思路**：通过**降级方案（Fallback）**保证系统可用性。

```
主服务失败 → 自动切换到降级方案
  ↓
系统可用性：100%（降级方案保证系统可用）
请求成功率：可能 < 99%（降级方案准确率低），但**系统不 down**
```

**结果**：
- **系统可用性**：100%（降级方案保证系统可用）
- **请求成功率**：可能 < 99%（降级方案准确率低），但**系统不 down**

---

## 回退率的监控与告警

### 监控指标

- **回退率趋势**：回退率是否持续上升？
- **降级策略分布**：哪些降级策略被触发最多？
- **降级后的成功率**：降级方案的准确率如何？

### 告警阈值

- **Warning**：回退率 > 5%（需要关注）
- **Critical**：回退率 > 10%（需要立即处理）

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1 详细讲解（KYC_Day01_A1_详细讲解_指标与测试.md） |
| **Related** | [分层容错设计](./KYC_Day01_A1_B6_layered_fault_tolerance.md)（降级方案）、[告警响应机制](../KYC_Day01_A4_告警响应机制详解.md)（自动降级策略）、[L0 稳定性指标](./KYC_Day01_A1_详细讲解_指标与测试.md) |
