# Day 2_A1_B3：采样策略详解（Sampling Strategy）

---
doc_type: glossary
layer: L3
scope_in:  Trace采样策略（Head-based、Tail-based、Adaptive Sampling）、Log采样策略、Metrics降采样（Downsampling）
scope_out: 具体采样实现代码（见 howto）；采样算法的数学原理（见 L4）；采样平台的配置（见 reference）
inputs:   (读者) 疑问：如果每天有百万级请求，所有Trace都记录，存储成本会很高，怎么优化？如何平衡可观测性和成本？
outputs:  采样策略详解 + 不同采样策略对比 + 实际应用场景 + 成本优化效果
entrypoints: [ 核心问题：Trace采样策略 ]
children: []
related: [ Sampling Strategy, Trace Sampling, Log Sampling, Downsampling, 成本优化, 可观测性, KYC_Day02_A1_可观测性详解.md, KYC_Day02_A1_B4_可观测性成本优化详解.md ]
---

## Definition（定义）

**核心问题**：**如果每天有百万级请求，所有Trace都记录，存储成本会很高，怎么优化？**

**核心答案**：
- ✅ **不是记录所有Trace**：采用采样策略，只记录部分Trace
- ✅ **Tail-based Sampling**：错误请求100%采样，成功请求1%采样
- ✅ **Log采样**：ERROR级别100%记录，INFO级别10%采样
- ✅ **Metrics降采样**：实时指标高精度，历史指标降采样存储

**类比**：
- **采样策略** = **医院体检**（不是所有人都体检，但病人必须体检）
- **Tail-based Sampling** = **错误必须记录，成功可以抽样**
- **降采样** = **近期数据详细，历史数据概览**

---

## 🎯 核心问题

### 问题场景

**场景1：Trace采样**
- "如果每天有百万级请求，所有Trace都记录，存储成本会很高，怎么优化？"
- "如何平衡可观测性和成本？"

**场景2：Log采样**
- "如果所有日志都记录，存储成本会很高，怎么优化？"
- "哪些日志必须记录，哪些可以采样？"

**场景3：Metrics降采样**
- "长期存储Metrics数据，如何优化存储成本？"
- "如何设计Metrics的存储策略？"

---

## 📊 Trace采样策略详解

### 1. Head-based Sampling（请求开始时决定）

**定义**：**在请求开始时决定是否采样**。

**工作原理**：
- ✅ **请求开始时**：根据采样率决定是否采样（如：1%采样）
- ✅ **采样决定**：一旦决定采样，整条Trace都记录
- ✅ **不采样决定**：一旦决定不采样，整条Trace都不记录

**具体数值例子**：

假设我们有 **100 个用户的 statement 请求**，检查 `received == TRUE`：

**场景设置**：
- 总请求数：100 个
- 采样率：1%（100个请求中采样1个）
- 成功请求（`received == TRUE`）：95 个
- 失败请求（`received == FALSE` 或错误）：5 个

**Head-based Sampling 的工作流程**：

| 请求ID | 请求开始时 | 随机数 | 是否采样？ | `received == TRUE`? | 最终结果 |
|--------|-----------|--------|-----------|-------------------|---------|
| Request #1 | `random() = 0.003` | 0.003 | ✅ **采样** | TRUE | ✅ 记录（完整Trace） |
| Request #2 | `random() = 0.052` | 0.052 | ❌ 不采样 | TRUE | ❌ 不记录 |
| Request #3 | `random() = 0.087` | 0.087 | ❌ 不采样 | FALSE | ❌ **不记录（错过错误！）** |
| Request #4 | `random() = 0.234` | 0.234 | ❌ 不采样 | TRUE | ❌ 不记录 |
| ... | ... | ... | ... | ... | ... |
| Request #95 | `random() = 0.991` | 0.991 | ❌ 不采样 | TRUE | ❌ 不记录 |
| Request #96 | `random() = 0.012` | 0.012 | ❌ 不采样 | FALSE | ❌ **不记录（错过错误！）** |
| Request #97 | `random() = 0.456` | 0.456 | ❌ 不采样 | TRUE | ❌ 不记录 |
| Request #98 | `random() = 0.789` | 0.789 | ❌ 不采样 | FALSE | ❌ **不记录（错过错误！）** |
| Request #99 | `random() = 0.234` | 0.234 | ❌ 不采样 | TRUE | ❌ 不记录 |
| Request #100 | `random() = 0.567` | 0.567 | ❌ 不采样 | FALSE | ❌ **不记录（错过错误！）** |

**结果统计**：
- ✅ **采样的请求**：1 个（Request #1，`received == TRUE`）
- ❌ **错过的错误请求**：4-5 个（Request #3, #96, #98, #100 等，`received == FALSE`）
- 📊 **采样率**：1%（1/100 = 1%）
- ⚠️ **错误捕获率**：0%（0/5 = 0%，所有错误都被错过了！）

**关键问题**：
- ❌ **所有错误请求都被错过了**：因为采样决定在请求开始时就做了，无法知道请求是否成功
- ❌ **无法捕获错误**：错误请求可能没被采样到，导致无法追踪问题

**代码示例**：
```python
# Head-based Sampling：请求开始时决定
import random

def should_sample_trace(request_id: str) -> bool:
    """在请求开始时决定是否采样"""
    # 1% 采样率：100个请求中采样1个
    return random.random() < 0.01

# 使用
if should_sample_trace(request_id):
    # 记录完整Trace
    trace = start_trace(request_id)
else:
    # 不记录Trace
    pass
```

**优势**：
- ✅ **简单**：实现简单，性能开销小
- ✅ **可预测**：采样率固定，存储成本可控

**劣势**：
- ❌ **可能错过错误**：如果错误请求没被采样到，就无法追踪
- ❌ **不灵活**：无法根据请求结果调整采样

**适用场景**：
- ✅ **高吞吐量系统**：需要固定采样率控制成本
- ✅ **正常情况监控**：主要用于监控系统性能，不是错误追踪

---

### 2. Tail-based Sampling（请求结束时根据结果决定）

**定义**：**在请求结束时根据结果决定是否采样**。

**工作原理**：
- ✅ **请求开始时**：所有请求都经过采样器，但不一定记录
- ✅ **请求结束时**：根据结果决定是否采样
- ✅ **错误请求100%采样**：所有错误请求都记录
- ✅ **成功请求1%采样**：只有1%的成功请求记录

**具体数值例子**（对比 Head-based，使用相同场景）：

假设我们有 **100 个用户的 statement 请求**，检查 `received == TRUE`：

**场景设置**（与 Head-based 相同）：
- 总请求数：100 个
- 成功请求（`received == TRUE`）：95 个
- 失败请求（`received == FALSE` 或错误）：5 个

**Tail-based Sampling 的工作流程**：

| 请求ID | `received == TRUE`? | 请求结束时的决定 | 随机数（仅成功请求） | 是否采样？ | 最终结果 |
|--------|-------------------|----------------|-------------------|-----------|---------|
| Request #1 | ✅ TRUE | 成功 → 1%采样 | `random() = 0.003` | ✅ **采样** | ✅ 记录（完整Trace） |
| Request #2 | ✅ TRUE | 成功 → 1%采样 | `random() = 0.052` | ❌ 不采样 | ❌ 不记录 |
| Request #3 | ❌ FALSE | **错误 → 100%采样** | - | ✅ **采样** | ✅ **记录（捕获错误！）** |
| Request #4 | ✅ TRUE | 成功 → 1%采样 | `random() = 0.234` | ❌ 不采样 | ❌ 不记录 |
| ... | ... | ... | ... | ... | ... |
| Request #95 | ✅ TRUE | 成功 → 1%采样 | `random() = 0.991` | ❌ 不采样 | ❌ 不记录 |
| Request #96 | ❌ FALSE | **错误 → 100%采样** | - | ✅ **采样** | ✅ **记录（捕获错误！）** |
| Request #97 | ✅ TRUE | 成功 → 1%采样 | `random() = 0.456` | ❌ 不采样 | ❌ 不记录 |
| Request #98 | ❌ FALSE | **错误 → 100%采样** | - | ✅ **采样** | ✅ **记录（捕获错误！）** |
| Request #99 | ✅ TRUE | 成功 → 1%采样 | `random() = 0.234` | ❌ 不采样 | ❌ 不记录 |
| Request #100 | ❌ FALSE | **错误 → 100%采样** | - | ✅ **采样** | ✅ **记录（捕获错误！）** |

**结果统计**（对比 Head-based）：
- ✅ **采样的请求**：约 6-7 个
  - 成功请求采样：约 1 个（95 × 1% ≈ 1 个）
  - **错误请求采样：5 个（5 × 100% = 5 个）**
- ✅ **捕获的错误请求**：5 个（5/5 = 100%）
- 📊 **总采样率**：约 6-7%（6-7/100 = 6-7%）
- ✅ **错误捕获率**：**100%**（5/5 = 100%，所有错误都被捕获！）

**关键优势**：
- ✅ **所有错误请求都被捕获**：因为采样决定在请求结束时才做，可以看到请求结果
- ✅ **100%错误捕获率**：所有 `received == FALSE` 的请求都被记录，不会错过任何错误

**对比 Head-based Sampling**：

| 指标 | Head-based | Tail-based |
|------|-----------|-----------|
| **总采样率** | 1% (1/100) | 6-7% (6-7/100) |
| **成功请求采样** | 1 个 | 约 1 个 |
| **错误请求采样** | 0 个 ❌ | 5 个 ✅ |
| **错误捕获率** | 0% ❌ | **100%** ✅ |
| **是否错过错误** | ✅ **是**（所有错误都错过） | ❌ **否**（所有错误都捕获） |

**代码示例**：
```python
# Tail-based Sampling：请求结束时根据结果决定
import random

def should_sample_trace(request_id: str, status: str, error_code: str = None) -> bool:
    """在请求结束时根据结果决定是否采样"""
    # 错误请求：100% 采样
    if status == "fail" or error_code:
        return True
    
    # 成功请求：1% 采样
    return random.random() < 0.01

# 使用
trace = start_trace(request_id)  # 所有请求都经过采样器
try:
    result = process_request(request_id)
    status = "success"
except Exception as e:
    status = "fail"
    error_code = str(e)

# 请求结束时决定是否记录
if should_sample_trace(request_id, status, error_code):
    # 记录完整Trace
    trace.end(status=status, error_code=error_code)
else:
    # 不记录Trace（丢弃）
    trace.discard()
```

**优势**：
- ✅ **捕获所有错误**：错误请求100%采样，不会错过错误
- ✅ **成本可控**：成功请求只采样1%，存储成本低
- ✅ **灵活**：可以根据请求结果调整采样

**劣势**：
- ⚠️ **实现复杂**：需要缓存请求信息，在请求结束时决定
- ⚠️ **性能开销**：所有请求都经过采样器，有一定性能开销

**适用场景**：
- ✅ **错误追踪**：需要捕获所有错误，不能错过任何错误请求
- ✅ **KYC项目**：错误请求必须记录，成功请求可以采样

**面试话术**：
- ✅ "我们使用**Tail-based Sampling**：错误请求100%采样，成功请求1%采样，这样既能捕获所有错误，又能控制存储成本。"

---

### 3. Adaptive Sampling（根据系统负载动态调整采样率）

**定义**：**根据系统负载动态调整采样率**。

**工作原理**：
- ✅ **低负载时**：提高采样率（如：10%采样）
- ✅ **高负载时**：降低采样率（如：0.1%采样）
- ✅ **错误请求**：始终100%采样（不受负载影响）

**例子**：
```python
# Adaptive Sampling：根据系统负载动态调整采样率
import random
import time

class AdaptiveSampler:
    def __init__(self):
        self.base_sample_rate = 0.01  # 基础采样率：1%
        self.current_load = 0.0  # 当前负载：0.0-1.0
    
    def update_load(self, load: float):
        """更新系统负载"""
        self.current_load = load
    
    def should_sample_trace(self, request_id: str, status: str, error_code: str = None) -> bool:
        """根据负载动态调整采样率"""
        # 错误请求：100% 采样（不受负载影响）
        if status == "fail" or error_code:
            return True
        
        # 成功请求：根据负载动态调整采样率
        if self.current_load < 0.3:  # 低负载：10% 采样
            sample_rate = 0.1
        elif self.current_load < 0.7:  # 中等负载：1% 采样
            sample_rate = 0.01
        else:  # 高负载：0.1% 采样
            sample_rate = 0.001
        
        return random.random() < sample_rate

# 使用
sampler = AdaptiveSampler()

# 更新系统负载（从Metrics获取）
current_rps = 1000
max_rps = 10000
load = current_rps / max_rps
sampler.update_load(load)

# 决定是否采样
if sampler.should_sample_trace(request_id, status, error_code):
    trace.end(status=status, error_code=error_code)
else:
    trace.discard()
```

**优势**：
- ✅ **自适应**：根据系统负载自动调整，既能保证可观测性，又能控制成本
- ✅ **灵活**：可以在低负载时提高采样率，高负载时降低采样率

**劣势**：
- ⚠️ **实现复杂**：需要监控系统负载，动态调整采样率
- ⚠️ **可能不稳定**：采样率频繁变化，可能影响分析

**适用场景**：
- ✅ **负载波动大的系统**：负载变化明显，需要动态调整
- ✅ **成本敏感的系统**：需要根据负载控制成本

---

## 📊 Log采样策略详解

### 1. 错误日志100%记录

**定义**：**ERROR级别日志全部记录**。

**原理**：
- ✅ **ERROR日志**：所有错误日志都必须记录，不能采样
- ✅ **原因**：错误日志是定位问题的关键，不能丢失

**例子**：
```python
# 错误日志：100% 记录
import logging

logger = logging.getLogger(__name__)

try:
    result = process_request(request_id)
except Exception as e:
    # ERROR 级别日志：100% 记录（不采样）
    logger.error(
        f"Request {request_id} failed: {str(e)}",
        extra={"request_id": request_id, "error_code": "PROCESSING_ERROR"}
    )
```

**关键点**：
- ✅ **不采样**：ERROR级别日志不采样，全部记录
- ✅ **必须记录**：错误日志是定位问题的关键，不能丢失

---

### 2. 正常日志采样（INFO级别10%采样）

**定义**：**INFO级别日志可以采样**。

**原理**：
- ✅ **INFO日志**：正常信息日志可以采样（如：10%采样）
- ✅ **原因**：INFO日志主要用于了解系统运行情况，不是定位问题的关键

**例子**：
```python
# 正常日志：10% 采样
import logging
import random

logger = logging.getLogger(__name__)

def log_info_sampled(message: str, **kwargs):
    """INFO级别日志：10% 采样"""
    if random.random() < 0.1:  # 10% 采样
        logger.info(message, extra=kwargs)
    # 不采样时不记录

# 使用
log_info_sampled(
    f"Request {request_id} processed successfully",
    request_id=request_id,
    latency_ms=100
)
```

**关键点**：
- ✅ **采样**：INFO级别日志可以采样（如：10%采样）
- ✅ **成本控制**：通过采样控制存储成本

---

### 3. Debug日志按需记录

**定义**：**DEBUG级别日志只在开发环境记录**。

**原理**：
- ✅ **开发环境**：DEBUG日志全部记录，用于开发调试
- ✅ **生产环境**：DEBUG日志不记录，避免存储成本

**例子**：
```python
# Debug日志：按需记录
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.getenv("ENV") == "development" else logging.INFO)

# 开发环境：DEBUG日志记录
# 生产环境：DEBUG日志不记录（级别过滤）
logger.debug(f"Processing request {request_id}: {request_data}")
```

**关键点**：
- ✅ **环境区分**：开发环境记录，生产环境不记录
- ✅ **成本控制**：生产环境不记录DEBUG日志，避免存储成本

---

## 📊 Metrics降采样（Downsampling）详解

### 1. 实时指标（高精度）

**定义**：**保持高精度，用于实时监控和告警**。

**原理**：
- ✅ **精度**：1分钟粒度，保留详细数据
- ✅ **用途**：实时监控、告警、Dashboard展示
- ✅ **保留时间**：通常保留1-7天

**例子**：
```python
# 实时指标：1分钟粒度，保留7天
metrics_config = {
    "realtime": {
        "interval": "1m",  # 1分钟粒度
        "retention": "7d",  # 保留7天
        "use_cases": ["real-time_monitoring", "alerting", "dashboard"]
    }
}
```

**关键点**：
- ✅ **高精度**：1分钟粒度，保留详细数据
- ✅ **短期保留**：通常保留1-7天，用于实时监控

---

### 2. 聚合指标（降采样存储）

**定义**：**降采样存储，用于历史分析**。

**原理**：
- ✅ **精度**：1小时粒度、1天粒度，降采样存储
- ✅ **用途**：历史分析、趋势分析、报表
- ✅ **保留时间**：通常保留90天-1年

**例子**：
```python
# 聚合指标：降采样存储
metrics_config = {
    "hourly": {
        "interval": "1h",  # 1小时粒度
        "retention": "90d",  # 保留90天
        "use_cases": ["historical_analysis", "trend_analysis"]
    },
    "daily": {
        "interval": "1d",  # 1天粒度
        "retention": "365d",  # 保留1年
        "use_cases": ["long_term_trend", "reporting"]
    }
}
```

**关键点**：
- ✅ **降采样**：1小时粒度、1天粒度，降低存储成本
- ✅ **长期保留**：通常保留90天-1年，用于历史分析

---

### 3. 指标生命周期管理

**完整策略**：
```
实时指标（1分钟粒度）
├─ 保留7天
├─ 用途：实时监控、告警、Dashboard
└─ 7天后 → 降采样为1小时粒度

聚合指标（1小时粒度）
├─ 保留90天
├─ 用途：历史分析、趋势分析
└─ 90天后 → 降采样为1天粒度

长期指标（1天粒度）
├─ 保留1年
└─ 用途：长期趋势、报表
```

**面试话术**：
- ✅ "我们采用分层存储：实时指标（1分钟粒度）保留7天，用于实时监控；聚合指标（1小时/1天粒度）保留90天，用于历史分析。这样既能满足实时监控需求，又能控制存储成本。"

---

## ⚖️ 采样策略对比

### Trace采样策略对比

| 策略 | 采样时机 | 错误采样 | 成功采样 | 优势 | 劣势 | 适用场景 |
|------|---------|---------|---------|------|------|---------|
| **Head-based** | 请求开始时 | 1% | 1% | 简单、可预测 | 可能错过错误 | 高吞吐量系统 |
| **Tail-based** | 请求结束时 | 100% | 1% | 捕获所有错误 | 实现复杂 | **错误追踪（推荐）** |
| **Adaptive** | 动态调整 | 100% | 0.1%-10% | 自适应、灵活 | 实现复杂 | 负载波动大的系统 |

**推荐策略**：
- ✅ **Tail-based Sampling**：错误请求100%采样，成功请求1%采样
- ✅ **原因**：既能捕获所有错误，又能控制存储成本

---

### Log采样策略对比

| 日志级别 | 采样策略 | 原因 | 存储成本 |
|---------|---------|------|---------|
| **ERROR** | 100%记录 | 定位问题的关键 | 高（但必须） |
| **WARN** | 100%记录 | 潜在问题，需要关注 | 中 |
| **INFO** | 10%采样 | 了解系统运行情况 | 低 |
| **DEBUG** | 开发环境记录 | 开发调试用 | 开发环境 |

**推荐策略**：
- ✅ **ERROR/WARN**：100%记录（定位问题的关键）
- ✅ **INFO**：10%采样（成本控制）
- ✅ **DEBUG**：开发环境记录，生产环境不记录

---

### Metrics降采样策略

| 指标类型 | 精度 | 保留时间 | 用途 | 存储成本 |
|---------|------|---------|------|---------|
| **实时指标** | 1分钟 | 7天 | 实时监控、告警 | 高 |
| **聚合指标** | 1小时 | 90天 | 历史分析 | 中 |
| **长期指标** | 1天 | 1年 | 长期趋势 | 低 |

**推荐策略**：
- ✅ **分层存储**：实时指标高精度短期保留，聚合指标降采样长期保留

---

## 💡 实际应用场景（KYC项目）

### 场景1：Trace采样

**需求**：
- ✅ **错误请求**：必须100%记录，不能错过任何错误
- ✅ **成功请求**：可以采样，控制存储成本

**方案**：
```python
# KYC项目：Tail-based Sampling
def should_sample_trace(request_id: str, status: str, error_code: str = None) -> bool:
    """KYC项目Trace采样策略"""
    # 错误请求：100% 采样
    if status == "fail" or error_code:
        return True
    
    # 成功请求：1% 采样
    return random.random() < 0.01
```

**效果**：
- ✅ **错误捕获率**：100%（所有错误都被记录）
- ✅ **存储成本**：降低90%（成功请求只采样1%）

---

### 场景2：Log采样

**需求**：
- ✅ **ERROR日志**：必须100%记录
- ✅ **INFO日志**：可以采样，控制存储成本

**方案**：
```python
# KYC项目：Log采样策略
def log_with_sampling(level: str, message: str, **kwargs):
    """KYC项目Log采样策略"""
    if level == "ERROR":
        # ERROR：100% 记录
        logger.error(message, extra=kwargs)
    elif level == "WARN":
        # WARN：100% 记录
        logger.warning(message, extra=kwargs)
    elif level == "INFO":
        # INFO：10% 采样
        if random.random() < 0.1:
            logger.info(message, extra=kwargs)
    elif level == "DEBUG":
        # DEBUG：只在开发环境记录
        if os.getenv("ENV") == "development":
            logger.debug(message, extra=kwargs)
```

**效果**：
- ✅ **错误日志**：100%记录（定位问题的关键）
- ✅ **存储成本**：INFO日志降低90%（10%采样）

---

### 场景3：Metrics降采样

**需求**：
- ✅ **实时监控**：需要高精度指标（1分钟粒度）
- ✅ **历史分析**：可以降采样存储（1小时/1天粒度）

**方案**：
```python
# KYC项目：Metrics降采样策略
metrics_config = {
    "realtime": {
        "interval": "1m",  # 1分钟粒度
        "retention": "7d",  # 保留7天
        "metrics": ["rps", "error_rate", "p95_latency"]
    },
    "hourly": {
        "interval": "1h",  # 1小时粒度
        "retention": "90d",  # 保留90天
        "metrics": ["rps", "error_rate", "p95_latency"]
    },
    "daily": {
        "interval": "1d",  # 1天粒度
        "retention": "365d",  # 保留1年
        "metrics": ["rps", "error_rate", "p95_latency"]
    }
}
```

**效果**：
- ✅ **实时监控**：1分钟粒度，满足实时监控需求
- ✅ **存储成本**：降低90%（历史数据降采样存储）

---

## 🤔 常见疑问：既然数据库可以保存大量数据，为什么还需要采样？

### 问题：能不能让数据库（MongoDB/MySQL）自己处理，不做采样？

**答案**：**不能！必须在应用层设计采样策略**。

**原因**：

#### 1. **数据库不会自动做采样决策**

**数据库的角色**：
- ✅ **存储数据**：数据库负责存储你写入的数据
- ✅ **查询数据**：数据库负责查询你请求的数据
- ❌ **不会自动采样**：数据库不知道哪些数据重要，哪些不重要

**类比**：
- **数据库** = **仓库**（存储货物）
- **采样策略** = **仓库管理员**（决定哪些货物要存储）
- 仓库不会自动决定存什么，需要管理员决策

**例子**：
```python
# ❌ 错误理解：让数据库自动采样
# 数据库不知道什么时候该采样，什么时候不该采样
database.save_trace(trace)  # 数据库只会保存，不会自动采样

# ✅ 正确理解：在应用层决定是否采样
if should_sample_trace(trace):  # 应用层决定
    database.save_trace(trace)   # 数据库只负责存储
```

---

#### 2. **存储成本依然存在**

**即使数据库可以保存大量数据，存储成本依然很高**：

**云存储成本**（以 AWS S3 为例）：
- **标准存储**：$0.023/GB/月
- **假设**：每天 100 万请求，每个 Trace 10KB
- **每天存储**：100万 × 10KB = 10GB
- **每月存储**：10GB × 30 = 300GB
- **每月成本**：300GB × $0.023 = **$6.9**

**使用 Tail-based Sampling 后**：
- **每天存储**：约 200MB（错误请求 + 1% 成功请求）
- **每月存储**：200MB × 30 = 6GB
- **每月成本**：6GB × $0.023 = **$0.14**
- **成本降低**：98%（从 $6.9 降到 $0.14）

**结论**：即使数据库可以保存，**成本依然存在**，采样可以大幅降低成本。

---

#### 3. **Trace/Log/Metrics 的特殊性**

**Trace/Log/Metrics 和业务数据的区别**：

| 特征 | 业务数据（订单、用户） | Trace/Log/Metrics |
|------|---------------------|-------------------|
| **数据量** | 相对固定 | **巨大且快速增长** |
| **数据价值** | 每一条都有价值 | **大部分价值低**（成功请求的 Trace） |
| **查询模式** | 按业务查询 | **按问题查询**（错误请求最重要） |
| **保留时间** | 长期保留 | **短期保留**（通常 7-30 天） |
| **增长速度** | 可预测 | **不可预测**（请求量变化） |

**例子**：
```python
# 业务数据（订单）：每条都有价值
order = {
    "order_id": "12345",
    "user_id": "user_001",
    "amount": 100.0,
    "status": "paid"
}
database.save(order)  # 必须保存，每条订单都有价值

# Trace 数据：大部分价值低
trace = {
    "request_id": "req_12345",
    "status": "success",  # 成功请求，价值低
    "latency_ms": 100
}
# 不是所有 Trace 都需要保存！
if should_sample_trace(trace):  # 需要采样决策
    database.save_trace(trace)
```

---

#### 4. **查询性能影响**

**即使数据库可以保存大量数据，查询性能也会受影响**：

**问题场景**：
- **100 万条 Trace**（不采样）：查询错误 Trace 需要扫描大量数据
- **2 万条 Trace**（采样后）：查询错误 Trace 只需要扫描少量数据

**例子**：
```sql
-- 不采样：需要扫描 100 万条记录
SELECT * FROM traces 
WHERE status = 'error' 
AND timestamp > '2025-01-19'
-- 扫描时间：可能需要几分钟

-- 采样后：只需要扫描 5 万条记录（错误 + 1% 成功）
SELECT * FROM traces 
WHERE status = 'error' 
AND timestamp > '2025-01-19'
-- 扫描时间：几秒钟
```

**结论**：即使数据库可以保存，**查询性能会受影响**，采样可以提高查询效率。

---

#### 5. **网络传输成本**

**即使数据库可以保存大量数据，网络传输也有成本**：

**问题场景**：
- **不采样**：每天传输 10GB 数据到数据库
- **采样后**：每天传输 200MB 数据到数据库

**成本对比**：
- **网络传输成本**（以 AWS 为例）：$0.09/GB（出站流量）
- **不采样**：10GB × $0.09 = $0.9/天 = **$27/月**
- **采样后**：200MB × $0.09 = $0.018/天 = **$0.54/月**
- **成本降低**：98%（从 $27 降到 $0.54）

**结论**：即使数据库可以保存，**网络传输成本依然存在**，采样可以降低传输成本。

---

### 总结：为什么需要在应用层设计采样策略？

| 原因 | 说明 | 例子 |
|------|------|------|
| **数据库不会自动采样** | 数据库只负责存储，不会自动决定存什么 | 需要在应用层决定是否采样 |
| **存储成本** | 即使数据库可以保存，存储成本依然很高 | $6.9/月 vs $0.14/月（采样后） |
| **数据特殊性** | Trace/Log/Metrics 数据量大、大部分价值低 | 100 万条 vs 2 万条（采样后） |
| **查询性能** | 大量数据会影响查询性能 | 扫描 100 万条 vs 5 万条 |
| **网络传输** | 传输大量数据有成本 | $27/月 vs $0.54/月（采样后） |

**结论**：
- ✅ **必须在应用层设计采样策略**
- ✅ **数据库只负责存储，不负责采样决策**
- ✅ **采样可以大幅降低成本，提高性能**

---

## 💡 成本优化效果

### 优化前（全部记录）

**假设**：
- ✅ **每天请求数**：1,000,000
- ✅ **每个Trace大小**：10KB
- ✅ **存储成本**：$0.023/GB/月

**存储成本**：
```
每天Trace大小 = 1,000,000 × 10KB = 10GB
每月Trace大小 = 10GB × 30 = 300GB
每月存储成本 = 300GB × $0.023/GB = $6.9
```

---

### 优化后（Tail-based Sampling）

**假设**：
- ✅ **每天请求数**：1,000,000
- ✅ **错误率**：1%（10,000个错误请求）
- ✅ **成功请求采样率**：1%（9,900个成功请求被采样）
- ✅ **每个Trace大小**：10KB

**存储成本**：
```
每天采样Trace数 = 10,000（错误）+ 9,900（成功采样）= 19,900
每天Trace大小 = 19,900 × 10KB = 199MB
每月Trace大小 = 199MB × 30 = 5.97GB
每月存储成本 = 5.97GB × $0.023/GB = $0.14

成本降低 = ($6.9 - $0.14) / $6.9 = 98%
```

**效果**：
- ✅ **错误捕获率**：100%（所有错误都被记录）
- ✅ **存储成本**：降低98%（从$6.9降到$0.14）

---

## 💡 总结

### 核心答案

**如果每天有百万级请求，所有Trace都记录，存储成本会很高，怎么优化？**

**方案**：
1. ✅ **Trace采样**：Tail-based Sampling（错误请求100%采样，成功请求1%采样）
2. ✅ **Log采样**：ERROR级别100%记录，INFO级别10%采样
3. ✅ **Metrics降采样**：实时指标高精度短期保留，聚合指标降采样长期保留

**效果**：
- ✅ **错误捕获率**：100%（所有错误都被记录）
- ✅ **存储成本**：降低90%-98%

### 关键要点

1. **Tail-based Sampling**：错误请求100%采样，成功请求1%采样，既能捕获所有错误，又能控制存储成本
2. **Log分级采样**：ERROR级别100%记录，INFO级别10%采样，DEBUG只在开发环境记录
3. **Metrics分层存储**：实时指标高精度短期保留，聚合指标降采样长期保留

### 面试话术

- ✅ "我们使用**Tail-based Sampling**：错误请求100%采样，成功请求1%采样，这样既能捕获所有错误，又能控制存储成本。"
- ✅ "日志采用分级策略：ERROR级别100%记录，INFO级别10%采样，DEBUG只在开发环境记录。"
- ✅ "Metrics采用分层存储：实时指标（1分钟粒度）保留7天，用于实时监控；聚合指标（1小时/1天粒度）保留90天，用于历史分析。"

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1 可观测性详解（[KYC_Day02_A1_可观测性详解.md](./KYC_Day02_A1_可观测性详解.md)） |
| **Related** | Sampling Strategy、Trace Sampling、Log Sampling、Downsampling、成本优化、可观测性、[KYC_Day02_A1_B4_可观测性成本优化详解.md](./KYC_Day02_A1_B4_可观测性成本优化详解.md) |
