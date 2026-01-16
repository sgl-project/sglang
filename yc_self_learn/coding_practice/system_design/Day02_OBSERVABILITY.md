# Day 2｜可观测性：把"我有监控"升级成"我能定位根因"

## 为什么要练

Senior 的价值不是"少出错"，而是"错了能快速发现/定位/止血"。OpenTelemetry 把 observability 讲成 traces/metrics/logs 三类信号。

## 目的

让你在 SD 面试里自然讲出"可观测闭环"。

## 目标

一页《观测设计 + dashboard 草图》。

---

## 三类信号框架

### 1. Metrics（指标）

**定义**：数值型指标，用于监控系统性能和健康状态。

**核心 Metrics**：

| Metric | 类型 | 说明 | 告警阈值 |
|--------|------|------|----------|
| **RPS** (Requests Per Second) | Gauge | 每秒请求数 | `> ____` |
| **p95 Latency** | Histogram | 95分位延迟 | `> ____ms` |
| **p99 Latency** | Histogram | 99分位延迟 | `> ____ms` |
| **Error Rate** | Counter | 错误率 | `> ____%` |
| **Timeout Rate** | Counter | 超时率 | `> ____%` |
| **Queue Depth** | Gauge | 队列深度 | `> ____` |
| **429 Rate** (Rate Limited) | Counter | 限流触发率 | `> ____%` |
| **Fallback Rate** | Counter | 降级触发率 | `> ____%` |
| **Schema Fail Rate** | Counter | Schema 验证失败率 | `> ____%` |

**实现方式**：
- [ ] Prometheus
- [ ] Datadog
- [ ] CloudWatch Metrics
- [ ] 其他：`____`

---

### 2. Logs（日志）

**定义**：结构化事件记录，包含请求上下文和错误信息。

**关键日志字段**：

```json
{
  "timestamp": "2024-01-01T10:00:00Z",
  "level": "INFO|WARN|ERROR",
  "request_id": "req-xxxxx",
  "trace_id": "trace-xxxxx",
  "span_id": "span-xxxxx",
  "user_id": "user-xxxxx",
  "model_version": "v1.2.3",
  "prompt_version": "v2.1.0",
  "validator_strictness": "high|medium|low",
  "validator_fail_reason": "field_missing|type_mismatch|...",
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
  "error": "optional error message"
}
```

**日志级别策略**：
- **ERROR**：系统错误、业务异常
- **WARN**：降级触发、限流触发
- **INFO**：关键操作（请求开始/结束、状态变更）
- **DEBUG**：详细调试信息（开发环境）

**实现方式**：
- [ ] ELK Stack (Elasticsearch + Logstash + Kibana)
- [ ] Splunk
- [ ] CloudWatch Logs
- [ ] 其他：`____`

---

### 3. Traces（链路追踪）

**定义**：请求在分布式系统中的完整调用链。

**关键 Span**：

```
┌─────────────────────────────────────────────────────────────┐
│ Trace: trace-xxxxx (Request ID: req-xxxxx)                  │
│ Total Latency: 285ms                                         │
├─────────────────────────────────────────────────────────────┤
│ Span 1: Preprocess                                           │
│   ├─ Duration: 10ms                                         │
│   └─ Status: OK                                             │
│                                                              │
│ Span 2: OCR/VLM                                              │
│   ├─ Duration: 50ms                                         │
│   ├─ Status: OK                                             │
│   └─ Metadata: {model: "qwen-vl", image_size: "1024x768"}   │
│                                                              │
│ Span 3: LLM Processing                                       │
│   ├─ Duration: 200ms                                        │
│   ├─ Status: OK                                             │
│   └─ Metadata: {model: "qwen2.5", tokens: 1500, cost: $0.001}│
│                                                              │
│ Span 4: Validation                                           │
│   ├─ Duration: 5ms                                          │
│   ├─ Status: OK                                             │
│   └─ Metadata: {schema_version: "v1", fields_validated: 5}  │
│                                                              │
│ Span 5: Storage                                              │
│   ├─ Duration: 20ms                                         │
│   └─ Status: OK                                             │
└─────────────────────────────────────────────────────────────┘
```

**Trace ID 关联**：
- 同一请求的所有日志共享 `trace_id`
- 可以通过 `trace_id` 串联所有相关日志
- 快速定位请求在系统中的完整路径

**实现方式**：
- [ ] OpenTelemetry + Jaeger
- [ ] OpenTelemetry + Zipkin
- [ ] AWS X-Ray
- [ ] 其他：`____`

---

## 可观测性闭环

### 发现问题 → 定位根因 → 快速止血

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Metrics   │ ────▶│   Alert     │ ────▶│  On-Call    │
│  告警触发    │      │  告警规则    │      │  人员通知    │
└─────────────┘      └─────────────┘      └─────────────┘
       │                    │                      │
       │                    ▼                      ▼
       │            ┌─────────────┐      ┌─────────────┐
       │            │  Dashboard  │ ────▶│   查看指标   │
       │            │  快速查看    │      │   判断严重性  │
       │            └─────────────┘      └─────────────┘
       │                    │                      │
       ▼                    ▼                      ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│    Traces   │ ────▶│   Trace ID  │ ────▶│   定位Span  │
│  获取Trace  │      │  关联日志   │      │   找出瓶颈   │
└─────────────┘      └─────────────┘      └─────────────┘
       │                    │                      │
       ▼                    ▼                      ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│    Logs     │ ────▶│   结构化    │ ────▶│   分析错误   │
│  详细日志   │      │   日志字段   │      │   找出根因   │
└─────────────┘      └─────────────┘      └─────────────┘
       │                    │                      │
       └────────────────────┴──────────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Rollback /    │
                   │   Mitigation    │
                   └─────────────────┘
```

---

## Dashboard 设计草图

### Dashboard 1: 实时监控视图（On-Call Dashboard）

**布局**：
```
┌─────────────────────────────────────────────────────────────┐
│ [实时 RPS]      [p95 Latency]    [Error Rate]   [Queue Depth]│
│   1200 req/s      285ms            0.1%           5          │
├─────────────────────────────────────────────────────────────┤
│ Latency Trends (Last 1 Hour)                                │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ p95 ────▁▃▅▇▆▄▂▁                                     │    │
│ │ p99 ────▁▃▅▇▆▄▂▁                                     │    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Error Breakdown                                             │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Timeout: 45%    Schema Fail: 30%   Other: 25%      │    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Recent Alerts (Last 30 min)                                │
│ • [10:05] p95 Latency > 500ms                              │
│ • [10:15] Error Rate > 1%                                  │
└─────────────────────────────────────────────────────────────┘
```

**关键指标**：
- RPS（实时）
- p95/p99 Latency（趋势图）
- Error Rate（饼图 + 趋势）
- Queue Depth（实时）
- 最近告警列表

---

### Dashboard 2: 业务健康视图（Business Health Dashboard）

**布局**：
```
┌─────────────────────────────────────────────────────────────┐
│ [Fallback Rate]  [Schema Fail]  [429 Rate]  [Cost/Request] │
│     2.5%            0.8%           1.2%        $0.001       │
├─────────────────────────────────────────────────────────────┤
│ Latency Breakdown by Stage (Average)                        │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Preprocess: 10ms  ███                               │    │
│ │ OCR/VLM: 50ms     ███████████                       │    │
│ │ LLM: 200ms        ████████████████████████████      │    │
│ │ Validate: 5ms     ██                                 │    │
│ │ Store: 20ms       █████                              │    │
│ └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│ Cost Trends (Last 7 Days)                                  │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ $0.0012 ────▁▃▅▇▆▄▂▁                                 │    │
│ └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**关键指标**：
- Fallback/Schema Fail/429 比率
- 各阶段延迟分解
- 成本趋势

---

### Dashboard 3: 链路追踪视图（Tracing Dashboard）

**功能**：
- 输入 `trace_id` 或 `request_id`，查看完整调用链
- 显示每个 Span 的耗时和状态
- 高亮慢 Span（超过阈值）
- 关联日志和错误信息

---

## 告警规则设计

### 告警等级

1. **Critical（严重）**：立即触发，电话/短信通知
   - Error Rate > 5%
   - p95 > 1000ms（持续 5 分钟）
   - 服务不可用

2. **Warning（警告）**：Slack/邮件通知
   - Error Rate > 1%
   - p95 > 500ms（持续 10 分钟）
   - Queue Depth > 50

3. **Info（信息）**：记录到日志，不通知
   - 指标异常但未超阈值
   - 降级触发

### 告警规则示例

```yaml
alerts:
  - name: high_error_rate
    condition: error_rate > 0.05
    duration: 5m
    severity: critical
    notification: phone + slack
    
  - name: high_latency
    condition: p95_latency > 500ms
    duration: 10m
    severity: warning
    notification: slack
```

---

## 实现检查清单

- [ ] 实现 Metrics 收集（Prometheus/Datadog）
- [ ] 配置结构化日志（JSON 格式）
- [ ] 实现 Trace 收集（OpenTelemetry）
- [ ] 建立 Trace ID 和 Log 的关联
- [ ] 设计并实现 Dashboard
- [ ] 配置告警规则
- [ ] 测试告警通知流程
- [ ] 编写 Dashboard 使用文档

---

## 参考

- OpenTelemetry: https://opentelemetry.io/
- Google SRE Book: [Monitoring Distributed Systems](https://sre.google/workbook/monitoring-distributed-systems/)
- The Three Pillars of Observability: https://www.oreilly.com/library/view/distributed-systems-observability/9781492033431/
