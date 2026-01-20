# Day 2_A1_B4_C2：可观测性成本优化场景详解

---
doc_type: glossary
layer: L4
scope_in:  可观测性成本优化的具体场景（存储成本高、传输成本高、总体成本控制）
scope_out: 具体优化实现代码（见 howto）；成本计算的详细公式（见 reference）；采样算法的数学原理（见 L4）
inputs:   (读者) 疑问：可观测性系统在哪些场景下成本高？如何针对性地优化？
outputs:  成本优化场景详解 + 场景问题分析 + 优化方案 + 实际成本计算
entrypoints: [ 核心问题：可观测性成本优化场景 ]
children: []
related: [ 成本优化, 存储优化, 传输优化, 采样策略, 可观测性, KYC_Day02_A1_B4_可观测性成本优化详解.md ]
---

## Definition（定义）

**核心问题**：**可观测性系统在哪些场景下成本高？如何针对性地优化？**

**核心答案**：
- ✅ **场景1：存储成本高** - 日志、Trace、Metrics 长期存储导致成本高
- ✅ **场景2：传输成本高** - 日志和Trace传输量大导致网络成本高
- ✅ **场景3：总体成本控制** - 需要平衡可观测性和成本

**类比**：
- **存储成本高** = **仓库租金贵**（需要定期清理旧数据）
- **传输成本高** = **快递费贵**（需要批量打包和压缩）
- **总体成本控制** = **预算管理**（在保证效果的前提下降低成本）

---

## 🎯 核心问题

### 问题场景

**场景1：存储成本高**
- "可观测性系统的存储成本很高，如何优化？"
- "如何设计Metrics/Logs/Traces的保留策略？"

**场景2：传输成本高**
- "日志和Trace传输成本很高，如何优化？"
- "如何减少网络传输的数据量？"

**场景3：总体成本控制**
- "如何平衡可观测性和成本？"
- "如何在不影响可观测性的前提下降低成本？"

---

## 📊 场景1：存储成本高

### 问题分析

**问题**：
- ✅ **日志存储成本**：每天产生大量日志，长期存储成本高
- ✅ **Trace存储成本**：每条Trace包含多个span，存储成本高
- ✅ **Metrics存储成本**：高频Metrics数据，长期存储成本高

**成本示例**：

假设系统每天产生：
- **日志**：100万条（每条1KB）= 1GB/天
- **Trace**：50万条（每条10KB）= 5GB/天
- **Metrics**：1000个指标，每分钟1次（每条0.1KB）= 1.44GB/天

**如果不优化**：
- 存储30天：7.5GB × 30 = 225GB
- 存储成本：225GB × $0.023/GB/月 = **$5.18/月**

**如果优化**：
- 日志保留7天（ERROR保留30天）：0.3GB × 30 + 0.7GB × 7 = 15.1GB
- Trace保留7天 + 采样（1%）：5GB × 1% × 7 = 0.35GB
- Metrics降采样（1小时粒度）：1.44GB → 0.024GB/天 × 90 = 2.16GB
- 存储成本：（15.1 + 0.35 + 2.16）GB × $0.023/GB/月 = **$0.40/月**

**成本降低**：从 $5.18 降到 $0.40，降低 **92%**

---

### 优化方案

#### 1. 日志保留策略

**策略**（业界标准和常见实践）：
- ✅ **ERROR日志**：保留30天（定位问题的关键）
  - **业界标准**：大多数公司保留 7-90 天，30 天是常见折中方案
  - **原因**：错误日志是定位问题的关键，需要保留较长时间以便追溯
- ✅ **WARN日志**：保留30天（潜在问题，需要关注）
  - **业界标准**：大多数公司保留 7-30 天，与 ERROR 保持一致
  - **原因**：WARN 可能预示问题，需要保留以便分析趋势
- ✅ **INFO日志**：保留7天（了解系统运行情况）
  - **业界标准**：大多数公司保留 3-14 天，7 天是常见折中方案
  - **原因**：INFO 主要用于了解系统运行情况，短期保留即可
- ✅ **DEBUG日志**：开发环境保留，生产环境不保留
  - **业界标准**：几乎所有公司生产环境都不保留 DEBUG
  - **原因**：DEBUG 主要用于开发调试，生产环境会产生大量噪音

**业界标准参考**：

| 日志级别 | 业界常见保留时间 | 常见范围 | 本文档推荐 |
|---------|----------------|---------|-----------|
| **ERROR** | 30天 | 7-90天 | 30天 ✅ |
| **WARN** | 30天 | 7-30天 | 30天 ✅ |
| **INFO** | 7天 | 3-14天 | 7天 ✅ |
| **DEBUG** | 不保留（生产环境） | 0天 | 不保留 ✅ |

**不同场景的差异**：

1. **大型公司（如 Google, Amazon, Microsoft）**：
   - ERROR：可能保留 90 天或更长时间（合规要求、审计需要）
   - INFO：可能保留 14-30 天（数据分析和趋势分析）
   
2. **中型公司（如大多数互联网公司）**：
   - ERROR：通常保留 30 天（本文档的推荐）
   - INFO：通常保留 7 天（本文档的推荐）

3. **小型公司/创业公司**：
   - ERROR：可能只保留 7-14 天（成本考虑）
   - INFO：可能只保留 3 天（成本考虑）

4. **合规行业（金融、医疗）**：
   - ERROR：可能保留 1-7 年（合规要求）
   - 所有日志：可能都需要长期保留（审计需要）

**总结**：
- ✅ **本文档的策略是业界常见的中型公司标准做法**
- ✅ **不同公司会根据规模、合规要求、成本预算调整保留时间**
- ✅ **核心原则**：ERROR > WARN > INFO > DEBUG（重要性递减，保留时间递减）

**代码示例**：
```python
# 日志保留策略配置
log_retention_policy = {
    "ERROR": {
        "retention_days": 30,
        "reason": "定位问题的关键，必须保留较长时间"
    },
    "WARN": {
        "retention_days": 30,
        "reason": "潜在问题，需要关注"
    },
    "INFO": {
        "retention_days": 7,
        "reason": "了解系统运行情况，短期保留即可"
    },
    "DEBUG": {
        "retention_days": 0,  # 生产环境不保留
        "reason": "开发调试用，生产环境不记录"
    }
}

# 自动清理旧日志
def cleanup_old_logs():
    """清理超过保留时间的日志"""
    for level, policy in log_retention_policy.items():
        cutoff_date = datetime.now() - timedelta(days=policy["retention_days"])
        delete_logs_before(level, cutoff_date)
```

**成本优化效果**：
- ✅ **ERROR日志**：保留30天（必须保留）
- ✅ **INFO日志**：保留7天（比30天减少77%存储成本）

---

#### 日志存储数据库架构和界面

**日志一般保存在什么数据库？**

**是的**，大多数公司是自己搭建或混合使用私有 + 云端的日志数据库。常见的架构包括：

**1. Elasticsearch + Kibana（业界最常用）**

**存储架构**：
- ✅ **Elasticsearch**：分布式搜索引擎，存储日志数据
  - 倒排索引：快速全文搜索
  - 时序索引：按时间分片存储
  - 分布式存储：支持水平扩展
- ✅ **Kibana**：可视化界面，查询和展示日志

**界面组成**：
```
Kibana Logs UI 典型界面：
┌─────────────────────────────────────────────────────────┐
│ [时间范围选择器] Last 15 minutes ▼ [刷新按钮]            │
├─────────────────────────────────────────────────────────┤
│ [服务筛选] [级别筛选] [搜索框: "error" ] [实时查看]     │
├─────────────────────────────────────────────────────────┤
│ 日志列表：                                               │
│ ┌──────┬──────┬──────┬────────────────────────────┐   │
│ │时间  │级别  │服务  │日志内容                     │   │
│ ├──────┼──────┼──────┼────────────────────────────┤   │
│ │10:30 │ERROR │api   │Request failed: timeout...  │   │
│ │10:29 │INFO  │api   │Request processed: 200ms   │   │
│ │10:28 │WARN  │api   │Slow request: 2000ms       │   │
│ └──────┴──────┴──────┴────────────────────────────┘   │
│                                                          │
│ [点击某条日志] → 右侧展开详细面板：                      │
│ - 完整日志内容                                           │
│ - 元数据（主机、容器、标签）                             │
│ - 相关 Trace 链接                                        │
└─────────────────────────────────────────────────────────┘
```

**2. Grafana Loki（轻量级方案）**

**存储架构**：
- ✅ **Loki**：轻量级日志聚合系统
  - 标签索引：只用标签索引，不全文索引（成本低）
  - 对象存储：日志内容存储在 S3/对象存储
  - 查询效率：通过标签快速过滤，然后扫描匹配的日志
- ✅ **Grafana**：统一的 Dashboard，同时展示 Metrics + Logs + Traces

**界面组成**：
```
Grafana Logs Panel 典型界面：
┌─────────────────────────────────────────────────────────┐
│ [时间范围] Last 1 hour ▼ [日志级别] ERROR ▼ [服务] api ▼│
├─────────────────────────────────────────────────────────┤
│ [搜索框: "timeout" ] [实时查看] [导出]                  │
├─────────────────────────────────────────────────────────┤
│ 日志流（时间轴）：                                       │
│ 10:30:15 [ERROR] api-service: Request timeout          │
│ 10:30:10 [INFO]  api-service: Processing request       │
│ 10:30:05 [WARN]  api-service: High latency detected    │
│                                                          │
│ [点击日志] → 展开详情：                                  │
│ - 标签：service=api, level=ERROR                        │
│ - 完整消息内容                                           │
│ - 相关 Metrics 链接                                      │
└─────────────────────────────────────────────────────────┘
```

**3. 云服务（Datadog, AWS CloudWatch, New Relic）**

**存储架构**：
- ✅ **托管服务**：完全托管的日志存储和查询服务
- ✅ **无需运维**：自动扩展、自动备份、自动索引

**界面组成**：
```
Datadog Logs Explorer 典型界面：
┌─────────────────────────────────────────────────────────┐
│ [时间] [环境] [服务] [级别] [搜索] [保存查询]           │
├─────────────────────────────────────────────────────────┤
│ 日志级别分布图：[ERROR: 5] [WARN: 10] [INFO: 1000]     │
├─────────────────────────────────────────────────────────┤
│ 日志列表（可分组）：                                     │
│ [10:30] [ERROR] api-service                            │
│   → Request failed: timeout after 5s                   │
│   → 标签: env=prod, host=api-01, trace_id=abc123       │
│   → [查看 Trace] [查看 Metrics]                         │
│                                                          │
│ [10:29] [INFO]  api-service                            │
│   → Request processed successfully                     │
│   → 标签: env=prod, host=api-02                        │
└─────────────────────────────────────────────────────────┘
```

**4. 自建日志系统架构示例**

**典型架构**：
```
应用日志
   ↓
Fluentd/Filebeat (日志收集)
   ↓
Kafka (消息队列，缓冲)
   ↓
Logstash (日志处理：解析、过滤、结构化)
   ↓
Elasticsearch (存储 + 索引)
   ↓
Kibana (查询 + 可视化)
```

**数据存储结构**：
```json
// Elasticsearch 中一条日志的结构
{
  "@timestamp": "2025-01-19T10:30:15Z",
  "level": "ERROR",
  "service": "api-service",
  "host": "api-server-01",
  "message": "Request failed: timeout after 5s",
  "request_id": "req-12345",
  "trace_id": "trace-abc123",
  "fields": {
    "status_code": 504,
    "latency_ms": 5000,
    "error_type": "timeout"
  },
  "tags": ["production", "api", "error"]
}
```

**界面功能**：

1. **搜索和过滤**：
   - 时间范围选择器（如：Last 15 minutes, Last 1 hour）
   - 日志级别筛选（ERROR, WARN, INFO）
   - 服务/主机/容器标签筛选
   - 全文搜索（搜索日志内容）

2. **日志列表**：
   - 时间戳、级别、服务、消息内容
   - 支持排序、分组、聚合

3. **日志详情**：
   - 点击日志展开详细面板
   - 显示完整消息、元数据、标签
   - 链接到相关 Trace 和 Metrics

4. **Dashboard**：
   - 日志级别趋势图（ERROR/WARN/INFO 数量随时间变化）
   - 服务错误率曲线
   - 日志量（流量）监控

**业界常见选择**：

| 公司类型 | 常见方案 | 原因 |
|---------|---------|------|
| **大型公司** | 自建 Elasticsearch 集群 | 完全控制、成本可控、定制化 |
| **中型公司** | Elasticsearch + Kibana 或 Grafana Loki | 平衡成本和功能 |
| **小型公司/创业** | 云服务（Datadog, CloudWatch） | 快速启动、无需运维 |
| **合规行业** | 自建 + 长期存储（S3） | 合规要求、审计需要 |

---

#### 2. Trace保留策略

**策略**：
- ✅ **Trace保留时间**：7天（用于问题定位）
- ✅ **自动清理**：超过7天的Trace自动清理
- ✅ **采样策略**：结合Tail-based Sampling，进一步降低存储成本

**代码示例**：
```python
# Trace保留策略配置
trace_retention_policy = {
    "retention_days": 7,
    "reason": "Trace主要用于问题定位，7天足够定位大部分问题",
    "cleanup_interval": "daily",  # 每天清理一次
    "sampling_strategy": "tail-based"  # 结合采样策略
}

# 清理逻辑
def cleanup_old_traces():
    """清理超过7天的Trace"""
    cutoff_date = datetime.now() - timedelta(days=7)
    delete_traces_before(cutoff_date)
```

**成本优化效果**：
- ✅ **Trace保留时间**：7天（比永久保留降低存储成本）
- ✅ **结合采样**：Tail-based Sampling进一步降低90%存储成本

---

#### 3. Metrics降采样（Downsampling）

**策略**：
- ✅ **实时指标**：1分钟粒度，保留7天（实时监控）
- ✅ **聚合指标**：1小时粒度，保留90天（历史分析）
- ✅ **长期指标**：1天粒度，保留1年（长期趋势）

**代码示例**：
```python
# Metrics降采样策略配置
metrics_downsampling_policy = {
    "realtime": {
        "interval": "1m",  # 1分钟粒度
        "retention_days": 7,
        "use_cases": ["real-time_monitoring", "alerting"]
    },
    "hourly": {
        "interval": "1h",  # 1小时粒度
        "retention_days": 90,
        "use_cases": ["historical_analysis", "trend_analysis"]
    },
    "daily": {
        "interval": "1d",  # 1天粒度
        "retention_days": 365,
        "use_cases": ["long_term_trend", "reporting"]
    }
}

# 降采样逻辑
def downsample_metrics():
    """定期降采样Metrics"""
    # 实时指标：保留7天
    keep_realtime_metrics(days=7)
    
    # 聚合为小时粒度：保留90天
    aggregate_to_hourly(days=90)
    
    # 聚合为天粒度：保留1年
    aggregate_to_daily(days=365)
```

**成本优化效果**：
- ✅ **实时指标**：1分钟粒度保留7天（详细数据）
- ✅ **历史指标**：1小时粒度保留90天（降采样后数据量减少60倍）
- ✅ **长期指标**：1天粒度保留1年（降采样后数据量减少1440倍）

---

## 📊 场景2：传输成本高

### 问题分析

**问题**：
- ✅ **网络传输成本**：大量日志和Trace需要传输到集中式存储
- ✅ **带宽成本**：高频数据传输占用大量带宽
- ✅ **API调用成本**：每次传输都有API调用成本

**成本示例**：

假设系统每天产生：
- **日志传输**：100万条 × 1KB = 1GB/天
- **Trace传输**：50万条 × 10KB = 5GB/天
- **总传输量**：6GB/天

**如果不优化**：
- 传输成本：6GB/天 × 30天 × $0.09/GB = **$16.2/月**
- API调用成本：150万次调用 × $0.0001/次 = **$15/月**
- **总成本**：$31.2/月

**如果优化（批量上传 + 压缩）**：
- 压缩率：70%（gzip压缩）
- 传输量：6GB × 30% = 1.8GB/天
- 批量上传：每批100条，调用次数减少99%
- 传输成本：1.8GB/天 × 30天 × $0.09/GB = **$4.86/月**
- API调用成本：1.5万次调用 × $0.0001/次 = **$0.15/月**
- **总成本**：$5.01/月

**成本降低**：从 $31.2 降到 $5.01，降低 **84%**

---

### 优化方案

#### 1. 批量上传

**策略**：
- ✅ **批量收集**：本地缓存日志和Trace，批量上传
- ✅ **批量大小**：每批100-1000条（根据系统负载调整）
- ✅ **批量间隔**：每5-10分钟上传一次（平衡实时性和成本）

**如何发送数据到 Datadog？**

**方法1：使用 Datadog HTTP API（直接发送）**

```python
# 使用 Datadog HTTP API 发送日志
import requests
import json
from datetime import datetime

class DatadogLogSender:
    """发送日志到 Datadog"""
    
    def __init__(self, api_key: str, site: str = "datadoghq.com"):
        self.api_key = api_key
        self.site = site
        self.url = f"https://http-intake.logs.{site}/v1/input/{api_key}"
        self.buffer = []
    
    def send_log(self, level: str, message: str, service: str = "my-service", **tags):
        """发送单条日志"""
        log_entry = {
            "timestamp": int(datetime.now().timestamp() * 1000),  # 毫秒时间戳
            "level": level,
            "message": message,
            "service": service,
            "host": "my-host",
            "ddtags": ",".join([f"{k}:{v}" for k, v in tags.items()])  # 标签
        }
        
        response = requests.post(
            self.url,
            json=log_entry,
            headers={"Content-Type": "application/json"}
        )
        return response.status_code == 200
    
    def batch_send_logs(self, logs: list):
        """批量发送日志"""
        response = requests.post(
            self.url,
            json=logs,
            headers={"Content-Type": "application/json"}
        )
        return response.status_code == 200
    
    def add_to_buffer(self, log_entry):
        """添加到缓冲区"""
        self.buffer.append(log_entry)
        
        # 达到批量大小，立即发送
        if len(self.buffer) >= 100:
            self.flush()
    
    def flush(self):
        """发送缓冲区中的所有日志"""
        if self.buffer:
            self.batch_send_logs(self.buffer)
            self.buffer.clear()

# 使用示例
dd_sender = DatadogLogSender(api_key="YOUR_DD_API_KEY")

# 发送单条日志
dd_sender.send_log(
    level="ERROR",
    message="Request failed: timeout after 5s",
    service="api-service",
    env="production",
    request_id="req-12345"
)

# 批量发送日志
logs = [
    {
        "timestamp": int(datetime.now().timestamp() * 1000),
        "level": "ERROR",
        "message": "Request failed",
        "service": "api-service"
    },
    # ... 更多日志
]
dd_sender.batch_send_logs(logs)
```

**方法2：使用 Datadog Python 库（推荐）**

```python
# 安装: pip install datadog
from datadog import initialize, api
import logging
from datadog_logger import DatadogLogHandler

# 初始化 Datadog
options = {
    "api_key": "YOUR_DD_API_KEY",
    "app_key": "YOUR_DD_APP_KEY"  # 可选，用于高级功能
}
initialize(**options)

# 方法2a：使用 Datadog Logger（最简单）
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        DatadogLogHandler(
            api_key="YOUR_DD_API_KEY",
            tags=["env:production", "service:api-service"]
        )
    ]
)

# 使用标准 logging
logger = logging.getLogger(__name__)
logger.error("Request failed: timeout after 5s")
logger.info("Request processed successfully")

# 方法2b：使用 Datadog API 发送自定义日志
api.Event.create(
    title="Request Failed",
    text="Request failed: timeout after 5s",
    alert_type="error",
    tags=["env:production", "service:api-service", "request_id:req-12345"]
)

# 方法2c：发送 Metrics
api.Metric.send(
    metric="api.request.count",
    points=[(int(datetime.now().timestamp()), 100)],
    tags=["env:production", "service:api-service"]
)

api.Metric.send(
    metric="api.request.latency",
    points=[(int(datetime.now().timestamp()), 150)],  # 150ms
    tags=["env:production", "service:api-service"]
)
```

**方法3：使用 Datadog Agent（生产环境推荐）**

```python
# 配置日志收集：将日志写入文件，Datadog Agent 自动收集
import logging
import os

# 设置环境变量（Datadog Agent 会自动读取）
os.environ["DD_API_KEY"] = "YOUR_DD_API_KEY"
os.environ["DD_SERVICE"] = "api-service"
os.environ["DD_ENV"] = "production"
os.environ["DD_TRACE_ENABLED"] = "true"

# 配置日志输出到文件（Datadog Agent 会监控这些文件）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('/var/log/myapp/app.log'),
        logging.StreamHandler()
    ]
)

# 使用标准 logging（Agent 会自动收集）
logger = logging.getLogger(__name__)
logger.error("Request failed: timeout after 5s")
logger.info("Request processed successfully")
```

**方法4：发送 Traces（使用 OpenTelemetry）**

```python
# 安装: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

# 初始化 OpenTelemetry
resource = Resource.create({
    "service.name": "api-service",
    "service.environment": "production"
})

trace_provider = TracerProvider(resource=resource)

# 配置 Datadog Exporter
otlp_exporter = OTLPSpanExporter(
    endpoint="https://api.datadoghq.com/api/v2/traces",  # Datadog Trace API
    headers={
        "DD-API-KEY": "YOUR_DD_API_KEY"
    }
)

trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(trace_provider)

tracer = trace.get_tracer(__name__)

# 创建 Trace
with tracer.start_as_current_span("process_request") as span:
    span.set_attribute("request_id", "req-12345")
    span.set_attribute("method", "POST")
    span.set_attribute("path", "/api/v1/users")
    
    try:
        # 处理请求
        result = process_request()
        span.set_status(trace.Status(trace.StatusCode.OK))
    except Exception as e:
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
```

**完整示例：结合日志、Metrics、Traces**

```python
import logging
from datadog import initialize, api
from opentelemetry import trace

class DatadogObservability:
    """统一的 Datadog 可观测性客户端"""
    
    def __init__(self, api_key: str, service: str = "my-service"):
        self.api_key = api_key
        self.service = service
        
        # 初始化 Datadog
        initialize(api_key=api_key)
        
        # 初始化 OpenTelemetry
        self._init_tracing(api_key)
        
        # 初始化 Logging
        self._init_logging()
    
    def _init_tracing(self, api_key: str):
        """初始化 Tracing"""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        
        resource = Resource.create({
            "service.name": self.service,
            "service.environment": "production"
        })
        
        trace_provider = TracerProvider(resource=resource)
        otlp_exporter = OTLPSpanExporter(
            endpoint="https://api.datadoghq.com/api/v2/traces",
            headers={"DD-API-KEY": api_key}
        )
        trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(trace_provider)
    
    def _init_logging(self):
        """初始化 Logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s %(message)s'
        )
    
    def log_error(self, message: str, **tags):
        """记录错误日志"""
        logging.error(message, extra={"tags": tags})
        api.Event.create(
            title="Error",
            text=message,
            alert_type="error",
            tags=[f"{k}:{v}" for k, v in tags.items()]
        )
    
    def log_info(self, message: str, **tags):
        """记录信息日志"""
        logging.info(message, extra={"tags": tags})
    
    def record_metric(self, metric_name: str, value: float, **tags):
        """记录指标"""
        api.Metric.send(
            metric=metric_name,
            points=[(int(datetime.now().timestamp()), value)],
            tags=[f"{k}:{v}" for k, v in tags.items()]
        )
    
    def create_trace(self, operation_name: str):
        """创建 Trace"""
        return trace.get_tracer(__name__).start_as_current_span(operation_name)

# 使用示例
dd_obs = DatadogObservability(
    api_key="YOUR_DD_API_KEY",
    service="api-service"
)

# 记录日志
dd_obs.log_error(
    "Request failed: timeout after 5s",
    request_id="req-12345",
    endpoint="/api/v1/users"
)

# 记录指标
dd_obs.record_metric(
    "api.request.count",
    value=100,
    service="api-service",
    env="production"
)

# 创建 Trace
with dd_obs.create_trace("process_request") as span:
    span.set_attribute("request_id", "req-12345")
    # 处理请求...
```

**环境变量配置（推荐）**

```bash
# 设置 Datadog 认证信息
export DD_API_KEY="your_api_key"
export DD_SITE="datadoghq.com"  # 或 datadoghq.eu
export DD_SERVICE="api-service"
export DD_ENV="production"
export DD_VERSION="1.0.0"
export DD_TRACE_ENABLED="true"
export DD_LOGS_ENABLED="true"
export DD_METRICS_ENABLED="true"
```

**代码示例（使用环境变量）**

```python
import os
from datadog import initialize

# 从环境变量读取配置
initialize(
    api_key=os.environ.get("DD_API_KEY"),
    app_key=os.environ.get("DD_APP_KEY")  # 可选
)
```

**代码示例**：
```python
# 批量上传配置
batch_upload_config = {
    "batch_size": 100,  # 每批100条
    "batch_interval": 300,  # 每5分钟上传一次
    "max_wait_time": 60  # 最多等待60秒（如果不满100条也上传）
}

# 本地缓存
log_buffer = []

def collect_log(log_entry):
    """收集日志到本地缓存"""
    log_buffer.append(log_entry)
    
    # 如果达到批量大小，立即上传
    if len(log_buffer) >= batch_upload_config["batch_size"]:
        upload_batch(log_buffer)
        log_buffer.clear()

def upload_batch(logs):
    """批量上传日志"""
    # 批量上传（减少API调用次数）
    api_client.batch_upload(logs)

# 定期上传（即使不满100条也上传）
def periodic_upload():
    """定期上传缓存中的日志"""
    if log_buffer:
        upload_batch(log_buffer)
        log_buffer.clear()

# 每5分钟执行一次
schedule.every(5).minutes.do(periodic_upload)
```

**成本优化效果**：
- ✅ **API调用次数**：从150万次降到1.5万次（减少99%）
- ✅ **API调用成本**：从$15/月降到$0.15/月（减少99%）

---

#### 2. 数据压缩

**策略**：
- ✅ **压缩算法**：使用gzip压缩（压缩率70%-90%）
- ✅ **压缩时机**：上传前压缩，接收后解压
- ✅ **压缩格式**：JSON、文本等可压缩格式

**代码示例**：
```python
import gzip
import json

def compress_logs(logs):
    """压缩日志数据"""
    # 转换为JSON字符串
    json_data = json.dumps(logs)
    
    # gzip压缩
    compressed_data = gzip.compress(json_data.encode('utf-8'))
    
    return compressed_data

def upload_compressed_logs(logs):
    """上传压缩后的日志"""
    # 压缩
    compressed_data = compress_logs(logs)
    
    # 上传（传输数据量减少70%-90%）
    api_client.upload(compressed_data, compressed=True)

# 使用
upload_compressed_logs(log_buffer)
```

**成本优化效果**：
- ✅ **传输数据量**：从6GB/天降到1.8GB/天（减少70%）
- ✅ **传输成本**：从$16.2/月降到$4.86/月（减少70%）

---

#### 3. 本地缓存

**策略**：
- ✅ **本地缓存**：先缓存到本地，再批量上传
- ✅ **缓存大小**：限制缓存大小（如100MB），防止内存溢出
- ✅ **故障恢复**：缓存持久化，防止数据丢失

**代码示例**：
```python
import pickle
import os

CACHE_FILE = "/tmp/log_cache.pkl"
MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB

def load_cache():
    """加载本地缓存"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return []

def save_cache(logs):
    """保存到本地缓存"""
    # 限制缓存大小
    if len(logs) * 1024 > MAX_CACHE_SIZE:
        # 超过大小，先上传一部分
        batch_size = len(logs) // 2
        upload_batch(logs[:batch_size])
        logs = logs[batch_size:]
    
    # 保存到文件（持久化）
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(logs, f)

def collect_log_with_cache(log_entry):
    """收集日志（带本地缓存）"""
    # 加载缓存
    log_buffer = load_cache()
    
    # 添加新日志
    log_buffer.append(log_entry)
    
    # 保存缓存
    save_cache(log_buffer)
    
    # 如果达到批量大小，立即上传
    if len(log_buffer) >= batch_upload_config["batch_size"]:
        upload_batch(log_buffer)
        save_cache([])  # 清空缓存
```

**成本优化效果**：
- ✅ **减少传输次数**：批量上传，减少传输次数
- ✅ **提高可靠性**：本地缓存，防止数据丢失

---

## 📊 场景3：总体成本控制

### 问题分析

**问题**：
- ✅ **成本与可观测性的平衡**：提高可观测性会增加成本
- ✅ **不同场景的需求不同**：开发环境、测试环境、生产环境需求不同
- ✅ **动态调整**：需要根据系统负载动态调整策略

**成本构成**：

| 成本项 | 占比 | 优化空间 |
|--------|------|----------|
| **存储成本** | 30% | 通过保留策略和降采样可降低90% |
| **传输成本** | 50% | 通过批量上传和压缩可降低80% |
| **计算成本** | 15% | 通过采样策略可降低90% |
| **其他成本** | 5% | 相对固定 |

**总体优化目标**：
- ✅ **总成本降低**：从$100/月降到$10/月（降低90%）
- ✅ **保持可观测性**：错误捕获率100%，关键指标监控完整
- ✅ **动态调整**：根据系统负载动态调整策略

---

### 优化方案

#### 1. 分层策略

**策略**：
- ✅ **开发环境**：详细日志、全量Trace、高频Metrics
- ✅ **测试环境**：正常日志、采样Trace、中频Metrics
- ✅ **生产环境**：错误日志、Tail-based Sampling、降采样Metrics

**代码示例**：
```python
# 环境配置
env_config = {
    "development": {
        "log_level": "DEBUG",
        "trace_sampling_rate": 1.0,  # 100%采样
        "metrics_interval": "1m"
    },
    "testing": {
        "log_level": "INFO",
        "trace_sampling_rate": 0.1,  # 10%采样
        "metrics_interval": "5m"
    },
    "production": {
        "log_level": "ERROR",
        "trace_sampling_rate": "tail-based",  # Tail-based Sampling
        "metrics_interval": "1m",  # 实时1分钟，历史降采样
        "metrics_downsampling": True
    }
}

# 根据环境配置
def get_config(env):
    return env_config.get(env, env_config["production"])
```

**成本优化效果**：
- ✅ **开发环境**：详细监控（成本较高，但开发需要）
- ✅ **生产环境**：优化策略（成本较低，但保持可观测性）

---

#### 2. 动态调整

**策略**：
- ✅ **负载高时**：降低采样率、延长批量间隔
- ✅ **负载低时**：提高采样率、缩短批量间隔
- ✅ **异常时**：提高采样率、缩短批量间隔（保证捕获异常）

**代码示例**：
```python
def adaptive_strategy(system_load):
    """根据系统负载动态调整策略"""
    if system_load > 0.8:  # 负载高
        return {
            "trace_sampling_rate": 0.01,  # 降低到1%
            "batch_interval": 600,  # 延长到10分钟
            "log_level": "ERROR"  # 只记录错误
        }
    elif system_load < 0.3:  # 负载低
        return {
            "trace_sampling_rate": 0.1,  # 提高到10%
            "batch_interval": 60,  # 缩短到1分钟
            "log_level": "INFO"  # 记录INFO级别
        }
    else:  # 正常负载
        return {
            "trace_sampling_rate": 0.01,  # 1%采样
            "batch_interval": 300,  # 5分钟
            "log_level": "WARN"  # 记录WARN级别
        }

# 检测异常时提高采样率
def detect_anomaly():
    """检测异常"""
    error_rate = get_error_rate()
    if error_rate > 0.01:  # 错误率超过1%
        return {
            "trace_sampling_rate": 0.1,  # 提高到10%
            "batch_interval": 60,  # 缩短到1分钟
            "log_level": "INFO"  # 记录INFO级别
        }
    return adaptive_strategy(get_system_load())
```

**成本优化效果**：
- ✅ **负载高时**：降低采样率，降低成本
- ✅ **负载低时**：提高采样率，提高可观测性
- ✅ **异常时**：自动提高采样率，保证捕获异常

---

## 💡 总结

### 核心答案

**可观测性系统在哪些场景下成本高？如何针对性地优化？**

**答案**：
1. ✅ **场景1：存储成本高**
   - **问题**：日志、Trace、Metrics长期存储导致成本高
   - **优化**：日志保留策略、Trace保留策略、Metrics降采样
   - **效果**：存储成本降低92%

2. ✅ **场景2：传输成本高**
   - **问题**：日志和Trace传输量大导致网络成本高
   - **优化**：批量上传、数据压缩、本地缓存
   - **效果**：传输成本降低84%

3. ✅ **场景3：总体成本控制**
   - **问题**：需要平衡可观测性和成本
   - **优化**：分层策略、动态调整
   - **效果**：总成本降低90%

### 关键要点

1. **存储优化**：通过保留策略和降采样，存储成本降低90%
2. **传输优化**：通过批量上传和压缩，传输成本降低80%
3. **动态调整**：根据系统负载和异常情况动态调整策略

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1_B4 可观测性成本优化详解（[KYC_Day02_A1_B4_可观测性成本优化详解.md](./KYC_Day02_A1_B4_可观测性成本优化详解.md)） |
| **Related** | 成本优化、存储优化、传输优化、采样策略、可观测性 |
