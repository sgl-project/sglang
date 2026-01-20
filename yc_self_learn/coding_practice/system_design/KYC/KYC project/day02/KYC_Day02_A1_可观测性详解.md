# Day 2｜可观测性详解：把"我有监控"升级成"我能定位根因"

---
doc_type: tutorial
layer: L2
scope_in:  可观测性的三类信号（Metrics/Logs/Traces）、Dashboard 设计、根因定位流程、可观测性闭环
scope_out: 具体实现代码（见 howto）；告警响应机制（见 A4）；性能优化（见 L4）
inputs:   (读者) 需求：理解可观测性设计，知道如何从告警到定位根因
outputs:  可观测性完整设计 + Dashboard 设计 + 根因定位流程 + KYC 项目实际案例
entrypoints: [ 三类信号框架, 可观测性闭环 ]
children: [ 
  KYC_Day02_A1_B1_Dashboard实现方式详解.md（Dashboard 实现方式详解），
  KYC_Day02_A1_B2_从Dashboard到根因定位的完整流程详解.md（从 Dashboard 到根因定位的完整流程详解），
  KYC_Day02_A1_B3_采样策略详解_Sampling_Strategy.md（采样策略详解：Trace采样、Log采样、Metrics降采样），
  KYC_Day02_A1_B4_可观测性成本优化详解.md（可观测性成本优化详解：存储优化、传输优化）
]
related: [ 可观测性, Metrics, Logs, Traces, Dashboard, 根因定位, KYC_Day01_A4_告警响应机制详解.md, KYC_Day01_A1_详细讲解_指标与测试.md ]
---

## Definition（定义）

**可观测性（Observability）**：**通过 Metrics、Logs、Traces 三类信号，能够快速发现问题、定位根因、快速止血的能力**。

**核心价值**：
- ✅ **不是"少出错"**：系统总会出错
- ✅ **而是"错了能快速发现/定位/止血"**：这是 Senior 工程师的核心价值

**三类信号**（OpenTelemetry 标准）：
- ✅ **Metrics（指标）**：数值型指标，监控系统性能和健康状态
- ✅ **Logs（日志）**：结构化事件记录，包含请求上下文和错误信息
- ✅ **Traces（链路追踪）**：请求在分布式系统中的完整调用链

---

## 🎯 为什么要练可观测性？

### Senior 的价值定位

**不是"少出错"**：
- ❌ 追求系统永远不出错（不现实）
- ❌ 只关注如何避免错误（被动防御）

**而是"错了能快速发现/定位/止血"**：
- ✅ 快速发现问题（Metrics 告警）
- ✅ 快速定位根因（Traces → Logs）
- ✅ 快速止血（自动回滚/降级）

**面试中的价值**：
- ✅ 能讲出"可观测性闭环"：告警 → 定位 → 止血
- ✅ 能设计完整的 Dashboard：实时监控、业务健康、链路追踪
- ✅ 能说明三类信号的关系：Metrics 发现问题，Traces 定位问题，Logs 分析问题

**三类信号的区别和关系（常见误解澄清）**：

**❌ 常见误解**：三类都包含错误信息，为什么要分成三类？一类不就够了吗？

**✅ 核心答案**：三类**不是互斥的，而是互补的**，各有不同的用途和视角：

1. **Metrics（指标）**：**不包含错误信息细节**，只是数值
   - 只能告诉你：**错误率是 5%**（有多少错误）
   - **不能告诉你**：具体是哪些请求出错了、为什么出错
   - **类比**：体温计告诉你发烧了（38.5°C），但不知道是什么病

2. **Traces（链路追踪）**：**展示调用链和时间线**，能看到哪里慢了
   - 只能告诉你：**请求经过了哪些服务，每个服务花了多少时间**
   - **不能告诉你**：具体错误信息是什么（除非在 Trace 中记录了 error）
   - **类比**：看到你去了哪些地方（医院 A → 医院 B → 医院 C），但不知道在每个地方发生了什么

3. **Logs（日志）**：**包含详细的错误信息和上下文**
   - 只能告诉你：**具体发生了什么错误，错误信息是什么**
   - **不能告诉你**：这个错误在整个调用链中的位置（哪个服务调用的哪个服务）
   - **类比**：看到每个地方的详细记录（在医院 A 做了什么检查），但看不到完整的路线图

**实际场景举例**：

假设你的系统突然报错：

**步骤1：Metrics 发现问题**（告警触发）
```
Metrics 告诉你：
- Error Rate: 5% ↑（从 1% 上升到 5%）
- p95 Latency: 8s ↑（从 2s 上升到 8s）

但是：
- ❌ 你不知道是哪些请求出错了
- ❌ 你不知道错误的具体原因
- ❌ 你不知道是哪个服务出了问题
```

**步骤2：Traces 定位问题**（找到有问题的调用链）
```
点击告警中的 Trace ID，看到调用链：

API Gateway (10ms)
  → User Service (50ms)
  → Payment Service (200ms) ⚠️ 慢了
    → Database (150ms) ⚠️ 慢了
    → External API (50ms)

问题：Payment Service 和 Database 都慢了！

但是：
- ❌ 你不知道为什么慢了（超时？数据库连接池满了？）
- ❌ 你不知道具体的错误信息
```

**步骤3：Logs 分析问题**（找到具体错误原因）
```
根据 Trace ID 找到对应的 Logs：

2025-01-19 10:30:15 [ERROR] Payment Service
  Trace ID: trace-abc123
  Message: "Database connection pool exhausted"
  Error: "Unable to acquire connection from pool (10/10 connections in use)"
  Stack trace: ...

2025-01-19 10:30:15 [ERROR] Database
  Trace ID: trace-abc123
  Message: "Query timeout after 5s"
  SQL: "SELECT * FROM payments WHERE user_id = ? AND status = 'pending'"
  Parameters: [user_id=12345]

问题找到了！数据库连接池满了，导致查询超时！
```

**三类信号的关系图**：

```
Metrics（发现问题）
  ↓ "错误率上升了！"
Traces（定位问题）
  ↓ "Payment Service 慢了！"
Logs（分析问题）
  ↓ "数据库连接池满了！"
解决方案
  ↓ "增加连接池大小或优化查询"
```

**为什么不能只用一类？**

**只用 Metrics**：
- ❌ 只能知道有问题，不知道具体是什么问题
- ❌ 无法定位是哪个服务的问题
- ❌ 无法知道错误的具体原因

**只用 Traces**：
- ❌ 能看到调用链，但看不到详细的错误信息
- ❌ 无法快速发现系统整体健康状况（需要逐个查看 Trace）
- ❌ 无法设置告警阈值（告警需要 Metrics）

**只用 Logs**：
- ❌ 日志太多，无法快速找到问题
- ❌ 无法看到调用链的完整视图
- ❌ 无法快速了解系统整体健康（错误率、延迟等）

**三类信号的互补关系**：

| 视角 | Metrics | Traces | Logs |
|------|---------|--------|------|
| **关注点** | 整体健康 | 调用流程 | 具体事件 |
| **数据格式** | 数值（错误率 5%） | 时间线（A→B→C） | 文本（错误信息） |
| **包含错误信息？** | ❌ 只有数值，没有细节 | ⚠️ 可能有，但不完整 | ✅ 有详细错误信息 |
| **用途** | 告警、Dashboard | 定位瓶颈、调用链 | 分析原因、调试 |
| **数据量** | 小（聚合后的数值） | 中（每个请求一条 Trace） | 大（每个事件一条 Log） |
| **查询速度** | 快（预聚合） | 中（索引查询） | 慢（全文搜索） |

**实际工作流程**：

1. **Metrics 告警** → "错误率超过阈值！"
2. **点击告警** → 自动跳转到 Traces，过滤出有问题的请求
3. **点击 Trace** → 自动关联到对应的 Logs，查看详细错误信息
4. **分析 Logs** → 找到具体原因（数据库连接池满了）
5. **解决问题** → 增加连接池大小

**总结**：

- ✅ **三类不是互斥的，而是互补的**：Metrics 发现问题，Traces 定位问题，Logs 分析问题
- ✅ **不是都包含错误信息**：Metrics 没有，Traces 可能有，Logs 有详细错误信息
- ✅ **一类不够**：各有不同的视角和用途，需要结合使用才能快速定位问题
- ✅ **实际使用中会互通**：现代可观测性平台（如 Datadog、Grafana）会自动关联三类数据

---

## 📊 三类信号框架

### 1. Metrics（指标）

**定义**：**数值型指标，用于监控系统性能和健康状态**。

**核心 Metrics（KYC 项目）**：

| Metric | 类型 | 说明 | 告警阈值 | 数据来源 |
|--------|------|------|----------|---------|
| **RPS** (Requests Per Second) | Gauge | 每秒请求数 | `> 1000 req/s` | API Gateway |
| **p95 Latency** | Histogram | 95分位延迟 | `> 15s` | `_summary.json` 的 `latency_ms` |
| **p99 Latency** | Histogram | 99分位延迟 | `> 30s` | `_summary.json` 的 `latency_ms` |
| **Error Rate** | Counter | 错误率 | `> 1%` | `_summary.json` 的 `status: "fail"` |
| **Success Rate** | Counter | 成功率 | `< 99%` | `_summary.json` 的 `status: "success"` |
| **Timeout Rate** | Counter | 超时率 | `> 0.5%` | `error_code: "API_TIMEOUT"` |
| **Schema Fail Rate** | Counter | Schema 验证失败率 | `> 2%` | `error_code: "SCHEMA_VALIDATION_FAILED"` |
| **Fallback Rate** | Counter | 降级触发率 | `> 5%` | `fallback_triggered: true` |
| **Batch Success Rate** | Counter | Batch 处理成功率 | `< 95%` | 完全成功的 batch 数 / 总 batch 数 |
| **Cost per Request** | Gauge | 每请求成本 | `> $0.002` | `_summary.json` 的 `cost_usd` |

**实现方式**：
- ✅ **Prometheus**：开源指标收集
- ✅ **Datadog**：商业监控平台
- ✅ **CloudWatch Metrics**：AWS 云监控
- ✅ **自定义**：基于 `_summary.json` 计算指标

**KYC 项目示例**：
```python
# 从 _summary.json 提取 Metrics
def extract_metrics_from_summary(summary_path: Path) -> dict:
    """从 _summary.json 提取 Metrics"""
    with open(summary_path) as f:
        summary = json.load(f)
    
    results = summary["results"]
    total = len(results)
    successes = sum(1 for r in results if r.get("status") == "success")
    failures = total - successes
    
    latencies = [r.get("latency_ms", 0) for r in results if r.get("status") == "success"]
    latencies.sort()
    
    return {
        "success_rate": successes / total if total > 0 else 0,
        "error_rate": failures / total if total > 0 else 0,
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
        "p99_latency_ms": latencies[int(len(latencies) * 0.99)] if latencies else 0,
        "total_requests": total,
        "total_cost_usd": sum(r.get("cost_usd", 0) for r in results),
        "avg_cost_per_request": sum(r.get("cost_usd", 0) for r in results) / total if total > 0 else 0
    }
```

---

### 2. Logs（日志）

**定义**：**结构化事件记录，包含请求上下文和错误信息**。

**关键日志字段（KYC 项目）**：

```json
{
  "timestamp": "2024-01-01T10:00:00Z",
  "level": "INFO|WARN|ERROR",
  "request_id": "req-xxxxx",
  "trace_id": "trace-xxxxx",
  "span_id": "span-xxxxx",
  "file_id": "doc_001.jpg",
  "batch_id": "batch_001",
  "user_id": "user-xxxxx",
  "model_version": "qwen2.5-vl-32b",
  "prompt_version": "v2.1.0",
  "validator_strictness": "high|medium|low",
  "validator_fail_reason": "field_missing|type_mismatch|...",
  "error_code": "IMAGE_FORMAT_UNSUPPORTED|SCHEMA_VALIDATION_FAILED|API_TIMEOUT|...",
  "latency_breakdown": {
    "preprocess": 10,
    "ocr_vlm": 50,
    "llm": 200,
    "validate": 5,
    "store": 20,
    "total": 285
  },
  "tokens_used": 1500,
  "cost_usd": 0.001,
  "status": "success|fail",
  "needs_review": true,
  "fraud_markers": ["expiry:expired", "low_confidence_critical:document_number"],
  "message": "optional error message"
}
```

**日志级别策略**：

| 级别 | 使用场景 | 示例 |
|------|---------|------|
| **ERROR** | 系统错误、业务异常 | `error_code: "API_TIMEOUT"`, `status: "fail"` |
| **WARN** | 降级触发、限流触发 | `fallback_triggered: true`, `rate_limit_exceeded: true` |
| **INFO** | 关键操作（请求开始/结束、状态变更） | `status: "success"`, `batch_completed: true` |
| **DEBUG** | 详细调试信息（开发环境） | `latency_breakdown`, `tokens_used` |

**实现方式**：
- ✅ **ELK Stack** (Elasticsearch + Logstash + Kibana)
- ✅ **Splunk**
- ✅ **CloudWatch Logs**
- ✅ **结构化日志文件**：JSON 格式，便于解析

**KYC 项目示例**：
```python
# 结构化日志记录
import logging
import json
from datetime import datetime

def log_request(request_id: str, trace_id: str, file_id: str, 
                status: str, latency_ms: int, error_code: str = None):
    """记录结构化日志"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": "ERROR" if status == "fail" else "INFO",
        "request_id": request_id,
        "trace_id": trace_id,
        "file_id": file_id,
        "status": status,
        "latency_ms": latency_ms,
        "error_code": error_code
    }
    
    # 写入日志文件（JSON 格式）
    logging.info(json.dumps(log_entry))
```

---

### 3. Traces（链路追踪）

**定义**：**请求在分布式系统中的完整调用链**。

**关键 Span（KYC 项目）**：

```
┌─────────────────────────────────────────────────────────────┐
│ Trace: trace-xxxxx (Request ID: req-xxxxx, File: doc_001.jpg)│
│ Total Latency: 285ms                                         │
├─────────────────────────────────────────────────────────────┤
│ Span 1: Preprocess (Image Format Check)                      │
│   ├─ Duration: 10ms                                         │
│   ├─ Status: OK                                             │
│   └─ Metadata: {format: "jpg", size: "1024x768"}            │
│                                                              │
│ Span 2: OCR/VLM (Fireworks API Call)                        │
│   ├─ Duration: 50ms                                         │
│   ├─ Status: OK                                             │
│   └─ Metadata: {model: "qwen2.5-vl-32b", tokens: 500}       │
│                                                              │
│ Span 3: LLM Processing (Structured Output)                  │
│   ├─ Duration: 200ms                                        │
│   ├─ Status: OK                                             │
│   └─ Metadata: {model: "qwen2.5-vl-32b", tokens: 1500, cost: $0.001}│
│                                                              │
│ Span 4: Validation (Schema + Rules)                         │
│   ├─ Duration: 5ms                                          │
│   ├─ Status: OK                                             │
│   └─ Metadata: {schema_version: "v1", fields_validated: 5, needs_review: false}│
│                                                              │
│ Span 5: Storage (Write to _summary.json)                    │
│   ├─ Duration: 20ms                                         │
│   └─ Status: OK                                             │
└─────────────────────────────────────────────────────────────┘
```

**Trace ID 关联**：
- ✅ 同一请求的所有日志共享 `trace_id`
- ✅ 可以通过 `trace_id` 串联所有相关日志
- ✅ 快速定位请求在系统中的完整路径

**实现方式**：
- ✅ **OpenTelemetry + Jaeger**：开源链路追踪
- ✅ **OpenTelemetry + Zipkin**：开源链路追踪
- ✅ **AWS X-Ray**：AWS 云链路追踪
- ✅ **自定义 Trace**：基于 `request_id` 和 `trace_id` 关联

**KYC 项目示例**：
```python
# Trace 记录
def create_trace(request_id: str, file_id: str) -> str:
    """创建 Trace ID"""
    trace_id = f"trace-{request_id}"
    return trace_id

def record_span(trace_id: str, span_name: str, duration_ms: int, 
                status: str, metadata: dict = None):
    """记录 Span"""
    span = {
        "trace_id": trace_id,
        "span_name": span_name,
        "duration_ms": duration_ms,
        "status": status,
        "metadata": metadata or {}
    }
    # 写入 Trace 存储（如 Jaeger）
    return span
```

---

## 🔄 可观测性闭环

### 发现问题 → 定位根因 → 快速止血

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Metrics   │ ────▶│   Alert     │ ────▶│  On-Call    │
│  告警触发    │      │  告警规则    │      │  人员通知    │
│ (Error Rate │      │ (Error Rate │      │ (Slack/     │
│  > 1%)      │      │  > 1%)      │      │  Phone)     │
└─────────────┘      └─────────────┘      └─────────────┘
       │                    │                      │
       │                    ▼                      ▼
       │            ┌─────────────┐      ┌─────────────┐
       │            │  Dashboard  │ ────▶│   查看指标   │
       │            │  快速查看    │      │   判断严重性  │
       │            │ (实时监控)   │      │ (Critical/  │
       │            └─────────────┘      │  Warning)   │
       │                    │             └─────────────┘
       ▼                    ▼                      │
┌─────────────┐      ┌─────────────┐              │
│    Traces   │ ────▶│   Trace ID  │ ◀────────────┘
│  获取Trace  │      │  关联日志   │
│ (通过 request│      │ (通过 trace_id│
│   _id)      │      │   查询)     │
└─────────────┘      └─────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐      ┌─────────────┐
│    Logs     │ ────▶│   结构化    │
│  详细日志   │      │   日志字段   │
│ (通过 trace │      │ (error_code,│
│   _id)      │      │  latency_   │
└─────────────┘      │  breakdown) │
                     └─────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   定位根因       │
                   │ (找出慢 Span/   │
                   │  错误原因)      │
                   └─────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Rollback /    │
                   │   Mitigation    │
                   │ (自动回滚/降级)  │
                   └─────────────────┘
```

### 实际案例：KYC 项目错误定位流程

**场景**：Error Rate 突然从 1% 飙升到 5%

**完整流程（7步）**：
```
1. Metrics 告警触发
   ↓
2. 查看 Dashboard（发现问题）
   ↓
3. 获取 Trace ID（选择失败的请求）
   ↓
4. 查看 Trace（了解调用链）
   ↓
5. 查看 Logs（获取详细错误信息）
   ↓
6. 定位根因（分析问题原因）
   ↓
7. 快速止血（解决问题）
```

**详细说明**：
- ✅ **步骤 1-5**：详见 [从 Dashboard 到根因定位的完整流程详解](./KYC_Day02_A1_B2_从Dashboard到根因定位的完整流程详解.md)
- ✅ **步骤 6-7**：详见 [定位问题后的下一步行动详解](./KYC_Day02_A1_B2_C2_定位问题后的下一步行动详解.md)

**快速示例**：
```
步骤 1：Metrics 告警触发
- Error Rate = 5% > 1%（告警阈值）

步骤 2：查看 Dashboard
- Error Breakdown: SCHEMA_VALIDATION_FAILED 占 60%

步骤 3：获取 Trace ID
- trace_id: trace-abc123
- error_code: SCHEMA_VALIDATION_FAILED

步骤 4：查看 Trace
- Span 4: Validation ❌ FAILED
- Error: SCHEMA_VALIDATION_FAILED
- Missing fields: ["date_of_birth"]

步骤 5：查看 Logs
- llm_output 缺少 date_of_birth 字段
- prompt_version: v2.1.0

步骤 6：定位根因
- LLM 输出缺少 date_of_birth 字段
- Prompt 版本 v2.1.0 可能有问题

步骤 7：快速止血
- 临时方案：降低 validator_strictness
- 长期方案：优化 Prompt 版本
```

---

## 📊 Dashboard 设计

### Dashboard 实现方式

**核心答案**：**Dashboard 通常不是自己从零写的，而是使用现成的监控平台（如 Grafana、Datadog、CloudWatch），然后配置和自定义**。

**实现方式对比**：

| 方式 | 说明 | 适用场景 | 成本 |
|------|------|---------|------|
| **现成平台**（推荐） | 使用 Grafana、Datadog、CloudWatch 等 | 大部分公司（小公司到大公司） | 低-中（按使用量付费） |
| **自己开发** | 自己写前端 + 后端 | 特殊需求、大公司自研 | 高（需要开发团队） |
| **开源工具** | Grafana + Prometheus（开源） | 中小公司、成本敏感 | 低（开源免费） |

**KYC 项目推荐**：
- ✅ **小公司/PoV 阶段**：使用 Grafana + Prometheus（开源免费）
- ✅ **大公司/Production**：使用 Datadog 或 CloudWatch（功能更全）

**关键点**：
- ✅ **不是从零写**：使用现成平台，配置 Dashboard
- ✅ **需要配置**：定义指标、设置告警规则、设计 Dashboard 布局
- ✅ **需要集成**：把 Metrics/Logs/Traces 推送到监控平台

---

### Dashboard 1：实时监控视图（On-Call Dashboard）

**目标**：**On-Call 工程师快速判断系统健康状态**。

**布局**：
```
┌─────────────────────────────────────────────────────────────┐
│ [实时 RPS]      [p95 Latency]    [Error Rate]   [Success Rate]│
│   120 req/s       8.5s            5.0%           95.0%      │
├─────────────────────────────────────────────────────────────┤
│ Latency Trends (Last 1 Hour)                                │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ p95 ────▁▃▅▇▆▄▂▁                                     │    │
│ │ p99 ────▁▃▅▇▆▄▂▁                                     │    │
│ │ Target: 15s ────────────────────────────────────────│    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Error Breakdown (Last 1 Hour)                               │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ SCHEMA_VALIDATION_FAILED: 60%  ████████████        │    │
│ │ API_TIMEOUT: 30%                ██████              │    │
│ │ IMAGE_FORMAT_UNSUPPORTED: 10%   ██                  │    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Recent Alerts (Last 30 min)                                │
│ • [10:05] Error Rate > 1% (Current: 5.0%)                   │
│ • [10:10] p95 Latency > 15s (Current: 8.5s)                 │
│ • [10:15] Batch Success Rate < 95% (Current: 90%)           │
└─────────────────────────────────────────────────────────────┘
```

**关键指标**：
- ✅ **RPS**：实时请求数（判断流量是否正常）
- ✅ **p95/p99 Latency**：延迟趋势（判断性能是否下降）
- ✅ **Error Rate**：错误率（判断系统是否稳定）
- ✅ **Success Rate**：成功率（判断系统可用性）
- ✅ **Error Breakdown**：错误类型分布（快速定位问题类型）
- ✅ **Recent Alerts**：最近告警列表（快速了解当前问题）

---

### Dashboard 2：业务健康视图（Business Health Dashboard）

**目标**：**业务团队查看业务指标和成本分析**。

**布局**：
```
┌─────────────────────────────────────────────────────────────┐
│ [Fallback Rate]  [Schema Fail]  [Automation Rate] [Cost/Req]│
│     2.5%            0.8%            65%           $0.0015    │
├─────────────────────────────────────────────────────────────┤
│ Latency Breakdown by Stage (Average, Last 1 Hour)          │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Preprocess: 10ms     ███                             │    │
│ │ OCR/VLM: 50ms       ███████████                      │    │
│ │ LLM: 200ms          ████████████████████████████     │    │
│ │ Validate: 5ms       ██                                 │    │
│ │ Store: 20ms         █████                              │    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Cost Trends (Last 7 Days)                                  │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ $0.0015 ────▁▃▅▇▆▄▂▁                                 │    │
│ │ Target: $0.002 ─────────────────────────────────────│    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Automation Rate Trend (Last 7 Days)                        │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ 65% ────▁▃▅▇▆▄▂▁                                     │    │
│ │ Target: 80% ────────────────────────────────────────│    │
│ └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**关键指标**：
- ✅ **Fallback Rate**：降级触发率（判断系统是否需要降级）
- ✅ **Schema Fail Rate**：Schema 验证失败率（判断 LLM 输出质量）
- ✅ **Automation Rate**：自动化率（判断系统成熟度）
- ✅ **Cost per Request**：每请求成本（判断成本控制）
- ✅ **Latency Breakdown**：各阶段延迟分解（找出性能瓶颈）
- ✅ **Cost Trends**：成本趋势（判断成本是否可控）

---

### Dashboard 3：链路追踪视图（Tracing Dashboard）

**目标**：**通过 Trace ID 查看完整调用链，定位问题**。

**功能**：
- ✅ **输入 `trace_id` 或 `request_id`**：查看完整调用链
- ✅ **显示每个 Span 的耗时和状态**：找出慢 Span
- ✅ **高亮慢 Span**：超过阈值的 Span 用红色标记
- ✅ **关联日志和错误信息**：点击 Span 查看相关日志

**界面示例**：
```
┌─────────────────────────────────────────────────────────────┐
│ Trace Search                                                │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Trace ID: [trace-abc123        ] [Search]          │    │
│ │ Request ID: [req-abc123        ] [Search]          │    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Trace: trace-abc123 (Request: req-abc123, File: doc_001.jpg)│
│ Total Latency: 285ms | Status: ✅ Success                   │
├─────────────────────────────────────────────────────────────┤
│ Timeline View                                               │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ 0ms    50ms   100ms  150ms  200ms  250ms  285ms     │    │
│ │ │      │       │      │      │      │      │        │    │
│ │ ├─Span1─┤                                      │    │    │
│ │ │       ├─Span2─┤                            │    │    │
│ │ │       │       ├────────Span3─────────┤      │    │    │
│ │ │       │       │                      ├─Span4┤    │    │
│ │ │       │       │                      │      ├Span5│    │
│ │ └───────┴───────┴──────────────────────┴──────┴────┘    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Span Details                                                 │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Span 1: Preprocess                                    │    │
│ │   Duration: 10ms ✅                                   │    │
│ │   Metadata: {format: "jpg", size: "1024x768"}        │    │
│ │                                                       │    │
│ │ Span 2: OCR/VLM                                       │    │
│ │   Duration: 50ms ✅                                   │    │
│ │   Metadata: {model: "qwen2.5-vl-32b", tokens: 500}   │    │
│ │                                                       │    │
│ │ Span 3: LLM Processing                                │    │
│ │   Duration: 200ms ⚠️ (最慢)                            │    │
│ │   Metadata: {model: "qwen2.5-vl-32b", tokens: 1500,  │    │
│ │              cost: $0.001}                            │    │
│ │                                                       │    │
│ │ Span 4: Validation                                    │    │
│ │   Duration: 5ms ✅                                     │    │
│ │   Metadata: {schema_version: "v1", fields: 5}         │    │
│ │                                                       │    │
│ │ Span 5: Storage                                       │    │
│ │   Duration: 20ms ✅                                   │    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Associated Logs                                              │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ [10:00:00] INFO: Request started (req-abc123)        │    │
│ │ [10:00:00] INFO: Preprocess completed (10ms)        │    │
│ │ [10:00:00] INFO: OCR/VLM completed (50ms)           │    │
│ │ [10:00:00] INFO: LLM Processing completed (200ms)   │    │
│ │ [10:00:00] INFO: Validation completed (5ms)          │    │
│ │ [10:00:00] INFO: Request completed (285ms, success)  │    │
│ └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 根因定位流程（Trace → Log → Metrics）

### 流程 1：从 Metrics 告警开始

**场景**：Error Rate 突然从 1% 飙升到 5%

**步骤 1：查看 Metrics Dashboard**
```
实时监控 Dashboard：
- Error Rate: 5% (正常 < 1%) ⚠️
- p95 Latency: 8.5s (正常 < 15s) ✅
- Success Rate: 95% (正常 > 99%) ⚠️
- Error Breakdown:
  - SCHEMA_VALIDATION_FAILED: 60% (主要错误)
  - API_TIMEOUT: 30%
  - Other: 10%
```

**步骤 2：选择失败的请求**
```
从 Error Breakdown 中选择一个失败的请求：
- request_id: req-abc123
- error_code: SCHEMA_VALIDATION_FAILED
- file_id: doc_001.jpg
```

---

### 流程 2：通过 Trace ID 查看调用链

**步骤 3：获取 Trace ID**
```
通过 request_id 查询 Trace ID：
- request_id: req-abc123
- trace_id: trace-abc123
```

**步骤 4：查看 Trace**
```
Trace: trace-abc123
├─ Span 1: Preprocess (10ms) ✅
├─ Span 2: OCR/VLM (50ms) ✅
├─ Span 3: LLM Processing (200ms) ✅
├─ Span 4: Validation (5ms) ❌ FAILED
│   └─ Error: SCHEMA_VALIDATION_FAILED
│       └─ Missing fields: ["date_of_birth"]
└─ Span 5: Storage (20ms) ❌ (未执行)
```

**发现**：
- ✅ Preprocess、OCR/VLM、LLM Processing 都正常
- ❌ Validation 阶段失败
- ❌ 缺少 `date_of_birth` 字段

---

### 流程 3：通过 Trace ID 关联 Logs

**步骤 5：查询相关 Logs**
```
通过 trace_id 查询所有相关日志：
{
  "trace_id": "trace-abc123",
  "request_id": "req-abc123",
  "file_id": "doc_001.jpg",
  "level": "ERROR",
  "error_code": "SCHEMA_VALIDATION_FAILED",
  "validator_fail_reason": "field_missing",
  "missing_fields": ["date_of_birth"],
  "llm_output": {
    "full_name": "John Doe",
    "document_number": "123456789",
    "expiry_date": "2025-12-31"
    // 缺少 date_of_birth
  },
  "model_version": "qwen2.5-vl-32b",
  "prompt_version": "v2.1.0",
  "validator_strictness": "high",
  "latency_breakdown": {
    "preprocess": 10,
    "ocr_vlm": 50,
    "llm": 200,
    "validate": 5,
    "total": 265
  }
}
```

**发现**：
- ✅ LLM 输出缺少 `date_of_birth` 字段
- ✅ Validator 严格模式（high）要求所有字段必填
- ✅ Prompt 版本 v2.1.0，模型版本 qwen2.5-vl-32b

---

### 流程 4：分析根因

**步骤 6：根因分析**
```
可能原因：
1. Prompt 版本 v2.1.0 可能有问题
   - 可能没有明确要求输出 date_of_birth
   - 需要检查 Prompt 内容

2. 模型版本 qwen2.5-vl-32b 可能对某些文档格式识别不准确
   - 某些文档格式可能没有 date_of_birth 字段
   - 需要检查文档格式

3. Validator 严格度设置过高
   - 当前设置：high（所有字段必填）
   - 建议：降低到 medium（核心字段必填）

4. 文档质量问题
   - 某些文档可能模糊、遮挡
   - 导致 LLM 无法识别 date_of_birth
```

**步骤 7：验证假设**
```
查询更多失败的请求：
- 查询条件：error_code = "SCHEMA_VALIDATION_FAILED"
- 时间范围：最近 1 小时
- 结果：60% 的失败都是缺少 date_of_birth

进一步分析：
- 这些文档是否都是同一类型？
- Prompt 版本是否都是 v2.1.0？
- 模型版本是否都是 qwen2.5-vl-32b？

结论：
- 问题集中在 Prompt 版本 v2.1.0
- 建议：回滚到 v2.0.0 或优化 v2.1.0
```

---

### 流程 5：快速止血

**步骤 8：立即行动**
```
临时方案（快速止血）：
1. 降低 validator_strictness 从 "high" 到 "medium"
   - 允许缺少非核心字段（如 date_of_birth）
   - 通过 Feature Flag 立即生效

2. 监控验证：
   - 观察 Error Rate 是否下降
   - 观察 Success Rate 是否上升

长期方案（根本解决）：
1. 优化 Prompt 版本 v2.1.0
   - 明确要求输出 date_of_birth
   - 添加示例和说明

2. 回归测试：
   - 使用 Golden Set 测试新 Prompt
   - 确保 Schema Pass Rate > 95%

3. 灰度发布：
   - 先发布到 1% 流量
   - 观察指标，逐步扩大
```

---

## 🚨 告警规则设计

### 告警等级（KYC 项目）

| 等级 | 触发条件 | 响应时间 | 通知方式 | 响应策略 |
|------|---------|---------|---------|---------|
| **Critical（严重）** | Error Rate > 5%<br>p95 > 30s（持续 5 分钟）<br>服务不可用 | 立即（< 5分钟） | 电话/短信 | 自动回滚/熔断 |
| **Warning（警告）** | Error Rate > 1%<br>p95 > 15s（持续 10 分钟）<br>Batch Success Rate < 95% | 15分钟内 | Slack/邮件 | 自动降级/重试 |
| **Info（信息）** | 指标异常但未超阈值<br>降级触发<br>限流触发 | 1小时内 | 记录日志 | 记录，无需立即处理 |

### 告警规则示例（KYC 项目）

```yaml
alerts:
  # Critical 告警
  - name: high_error_rate_critical
    condition: error_rate > 0.05
    duration: 5m
    severity: critical
    notification: phone + slack
    action: auto_rollback
    
  - name: high_latency_critical
    condition: p95_latency > 30000ms
    duration: 5m
    severity: critical
    notification: phone + slack
    action: auto_circuit_breaker
    
  # Warning 告警
  - name: high_error_rate_warning
    condition: error_rate > 0.01
    duration: 10m
    severity: warning
    notification: slack
    action: auto_fallback
    
  - name: high_latency_warning
    condition: p95_latency > 15000ms
    duration: 10m
    severity: warning
    notification: slack
    action: auto_retry
    
  - name: low_batch_success_rate
    condition: batch_success_rate < 0.95
    duration: 10m
    severity: warning
    notification: slack
    action: investigate
```

---

## 💡 KYC 项目可观测性实现检查清单

### Dashboard 实现方式

**核心答案**：**Dashboard 通常不是自己从零写的，而是使用现成的监控平台**。

**推荐方案**：

#### 方案 1：Grafana + Prometheus（开源，推荐小公司）

**步骤**：
1. **安装 Prometheus**：收集 Metrics
2. **安装 Grafana**：可视化 Dashboard
3. **配置数据源**：Grafana 连接 Prometheus
4. **创建 Dashboard**：在 Grafana 中配置 Dashboard（拖拽组件，不需要写代码）

**优点**：
- ✅ **开源免费**：不需要付费
- ✅ **功能强大**：支持 Metrics、Logs、Traces
- ✅ **易于配置**：通过 UI 配置，不需要写代码

**示例**：
```yaml
# prometheus.yml - Prometheus 配置
scrape_configs:
  - job_name: 'kyc-metrics'
    static_configs:
      - targets: ['localhost:9090']

# Grafana Dashboard 配置（通过 UI 配置，不需要写代码）
# 1. 添加数据源：Prometheus
# 2. 创建 Dashboard
# 3. 添加 Panel（图表）
#    - RPS: rate(requests_total[5m])
#    - p95 Latency: histogram_quantile(0.95, latency_bucket)
#    - Error Rate: rate(errors_total[5m]) / rate(requests_total[5m])
```

---

#### 方案 2：Datadog（商业平台，推荐大公司）

**步骤**：
1. **注册 Datadog 账号**：获取 API Key
2. **安装 Datadog Agent**：收集 Metrics/Logs/Traces
3. **推送数据**：代码中集成 Datadog SDK
4. **创建 Dashboard**：在 Datadog UI 中配置 Dashboard（拖拽组件）

**优点**：
- ✅ **功能全面**：Metrics、Logs、Traces 一体化
- ✅ **易于使用**：UI 友好，配置简单
- ✅ **告警集成**：内置告警功能

**示例**：
```python
# 集成 Datadog SDK
from datadog import initialize, statsd

initialize(api_key='your-api-key', app_key='your-app-key')

# 记录 Metrics
statsd.increment('kyc.requests.total')
statsd.histogram('kyc.latency', 285)
statsd.increment('kyc.errors.total', tags=['error_code:SCHEMA_VALIDATION_FAILED'])
```

---

#### 方案 3：CloudWatch（AWS 云平台）

**步骤**：
1. **使用 CloudWatch Metrics**：AWS 服务自动收集 Metrics
2. **使用 CloudWatch Logs**：收集日志
3. **使用 CloudWatch Dashboard**：在 AWS Console 中配置 Dashboard

**优点**：
- ✅ **AWS 集成**：如果使用 AWS，集成简单
- ✅ **自动收集**：AWS 服务自动收集 Metrics
- ✅ **成本可控**：按使用量付费

---

### Metrics 收集

- [ ] **实现 Metrics 收集**：基于 `_summary.json` 计算指标
- [ ] **配置 Prometheus/Datadog/CloudWatch**：推送指标到监控平台
- [ ] **设置告警规则**：Error Rate、Latency、Success Rate 等
- [ ] **建立指标 Dashboard**：在监控平台中配置 Dashboard（不需要写代码，通过 UI 配置）

### Logs 收集

- [ ] **实现结构化日志**：JSON 格式，包含关键字段
- [ ] **配置 Log 收集**：ELK Stack / Splunk / CloudWatch Logs
- [ ] **建立 Log 查询**：通过 `trace_id`、`request_id` 查询
- [ ] **设置 Log 级别**：ERROR、WARN、INFO、DEBUG

### Traces 收集

- [ ] **实现 Trace 收集**：OpenTelemetry + Jaeger / Zipkin
- [ ] **建立 Trace ID 关联**：同一请求的所有 Span 共享 `trace_id`
- [ ] **实现 Span 记录**：记录每个阶段的耗时和状态
- [ ] **建立 Trace Dashboard**：通过 `trace_id` 查看完整调用链

### Dashboard 设计

- [ ] **实时监控 Dashboard**：On-Call 工程师使用
- [ ] **业务健康 Dashboard**：业务团队使用
- [ ] **链路追踪 Dashboard**：开发团队使用
- [ ] **根因定位流程**：Metrics → Trace → Log → 根因

---

## 🎯 总结

### 可观测性的核心价值

**不是"少出错"**，而是**"错了能快速发现/定位/止血"**。

### 三类信号的作用

1. **Metrics（指标）**：**发现问题**
   - 监控系统性能和健康状态
   - 触发告警

2. **Traces（链路追踪）**：**定位问题**
   - 查看完整调用链
   - 找出慢 Span 和失败 Span

3. **Logs（日志）**：**分析问题**
   - 查看详细错误信息
   - 分析根因

### 可观测性闭环

```
Metrics 告警 → Dashboard 查看 → Trace 定位 → Log 分析 → 根因定位 → 快速止血
```

### KYC 项目实现要点

1. ✅ **Metrics**：基于 `_summary.json` 计算指标
2. ✅ **Logs**：结构化日志，包含 `trace_id`、`request_id`、`error_code` 等
3. ✅ **Traces**：记录 Preprocess → OCR/VLM → LLM → Validation → Storage 的完整调用链
4. ✅ **Dashboard**：实时监控、业务健康、链路追踪三个 Dashboard
5. ✅ **根因定位**：通过 `trace_id` 关联 Metrics、Traces、Logs

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | Day 2 可观测性（7days_speedup/Day02_OBSERVABILITY.md） |
| **Related** | 可观测性、Metrics、Logs、Traces、Dashboard、根因定位、[KYC_Day01_A4_告警响应机制详解.md](./KYC_Day01_A4_告警响应机制详解.md)、[KYC_Day01_A1_详细讲解_指标与测试.md](./KYC_Day01_A1_详细讲解_指标与测试.md) |
