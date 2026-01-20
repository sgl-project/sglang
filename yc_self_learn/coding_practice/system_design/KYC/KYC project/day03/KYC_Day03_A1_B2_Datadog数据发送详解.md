# Datadog 数据发送详解

---
doc_type: tutorial
layer: L2
scope_in:  Datadog 数据存储方式、Python 发送数据到 Datadog（HTTP API）、Metrics/Logs/Events 发送方法
scope_out: 具体 Datadog Agent 配置（见 reference）；高级查询和分析（见 reference）
inputs:  (读者) 需求：理解 Datadog 的数据存储方式，知道如何用 Python 程序发送数据到 Datadog
outputs:  Datadog 数据存储说明 + Python 发送数据完整示例 + HTTP API vs SQL 对比 + KYC 项目实际案例
entrypoints: [ Datadog 数据存储, Python HTTP API, Metrics 发送, Logs 发送 ]
children: []
related: [ Datadog, 可观测性, HTTP API, KYC_Day03_A1_回归测试与门禁详解.md, KYC_Day02_A1_B4_C2_可观测性成本优化场景详解.md ]
---

## Definition（定义）

**核心问题**：**Datadog 的数据存储是 SQL 吗？如何用 Python 程序发送数据到 Datadog？**

**核心答案**：
- ✅ **Datadog 不是 SQL 数据库**：使用 HTTP REST API，不是传统 SQL
- ✅ **发送方式**：使用 `requests.post()` 发送 JSON 数据
- ✅ **API 端点**：Metrics、Logs、Events 有不同的 API 端点
- ✅ **认证方式**：使用 API Key（不是用户名/密码）

---

## 📊 Datadog 数据存储方式

### 1. 什么时候用 SQL？什么时候用 HTTP API？

**完整对比图**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    数据存储方式选择指南                                        │
└─────────────────────────────────────────────────────────────────────────────┘

【场景 1：应用数据存储 - 使用 SQL】
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  应用层                Python 程序          SQL 数据库                       │
│  ┌──────────┐         ┌──────────────┐    ┌──────────────┐                │
│  │ 用户数据 │────────>│ Python       │───>│ MySQL /      │                │
│  │ 订单数据 │         │ 程序         │    │ PostgreSQL   │                │
│  │ 产品数据 │         │              │    │ MongoDB      │                │
│  └──────────┘         └──────────────┘    └──────────────┘                │
│         │                    │                     │                         │
│         │                    │  ① 连接数据库      │                         │
│         │                    │  conn = mysql.     │                         │
│         │                    │    connect(...)    │                         │
│         │                    │<────────────────────┘                         │
│         │                    │                     │                         │
│         │                    │  ② 执行 SQL        │                         │
│         │                    │  INSERT INTO       │                         │
│         │                    │    users VALUES     │                         │
│         │                    ├────────────────────>│                         │
│         │                    │  SELECT * FROM     │                         │
│         │                    │    users WHERE...   │                         │
│         │                    │<────────────────────┤                         │
│         │                    │                     │                         │
│         │                    │  ③ 返回结果         │                         │
│         │                    │  [(id, name, ...)]  │                         │
│         │                    │<────────────────────┘                         │
│         │                    │                     │                         │
│         └────────────────────┴─────────────────────┘                         │
│                                                                             │
│  特点：                                                                     │
│  ✅ 结构化数据（用户、订单、产品）                                          │
│  ✅ 需要复杂查询（JOIN、GROUP BY）                                         │
│  ✅ 需要事务支持（ACID）                                                    │
│  ✅ 需要持久化存储                                                          │
│                                                                             │
│  示例：                                                                     │
│  - 用户注册信息                                                            │
│  - 订单数据                                                                │
│  - 产品目录                                                                │
│  - Golden Set 元数据（用例信息）                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

【场景 2：监控数据发送 - 使用 HTTP API】
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  应用层                Python 程序          Datadog (HTTP API)              │
│  ┌──────────┐         ┌──────────────┐    ┌──────────────┐                │
│  │ Metrics  │────────>│ Python       │───>│ Datadog      │                │
│  │ Logs     │         │ 程序         │    │ HTTP API     │                │
│  │ Events   │         │              │    │              │                │
│  └──────────┘         └──────────────┘    └──────────────┘                │
│         │                    │                     │                         │
│         │                    │  ① HTTP POST        │                         │
│         │                    │  requests.post(     │                         │
│         │                    │    url, json=data   │                         │
│         │                    │  )                  │                         │
│         │                    ├────────────────────>│                         │
│         │                    │  POST /api/v1/      │                         │
│         │                    │    series            │                         │
│         │                    │  {                  │                         │
│         │                    │    "metric": "...", │                         │
│         │                    │    "value": 95.5    │                         │
│         │                    │  }                  │                         │
│         │                    │<────────────────────┤                         │
│         │                    │  HTTP 202 Accepted  │                         │
│         │                    │<────────────────────┘                         │
│         │                    │                     │                         │
│         └────────────────────┴─────────────────────┘                         │
│                                                                             │
│  特点：                                                                     │
│  ✅ 时序数据（Metrics、Logs、Traces）                                       │
│  ✅ 不需要复杂查询（简单聚合即可）                                          │
│  ✅ 不需要事务支持                                                          │
│  ✅ 主要用于监控和告警                                                      │
│                                                                             │
│  示例：                                                                     │
│  - 系统 Metrics（CPU、内存、QPS）                                          │
│  - 应用 Logs（错误日志、访问日志）                                          │
│  - 测试结果（回归测试准确率）                                              │
│  - 告警 Events（系统异常）                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

【场景 3：混合使用 - SQL + HTTP API】
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  KYC 项目示例：                                                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │ Golden Set 元数据（用例信息）                           │               │
│  │ ┌───────────────────────────────────────────────────┐ │               │
│  │ │ 存储位置：PostgreSQL (SQL)                        │ │               │
│  │ │                                                    │ │               │
│  │ │ CREATE TABLE golden_set_cases (                   │ │               │
│  │ │   case_id VARCHAR(100) PRIMARY KEY,              │ │               │
│  │ │   category VARCHAR(50),                          │ │               │
│  │ │   file_url TEXT,                                 │ │               │
│  │ │   expected_fields JSONB                          │ │               │
│  │ │ );                                                │ │               │
│  │ │                                                    │ │               │
│  │ │ INSERT INTO golden_set_cases VALUES (...);       │ │               │
│  │ │ SELECT * FROM golden_set_cases WHERE ...;        │ │               │
│  │ └───────────────────────────────────────────────────┘ │               │
│  └─────────────────────────────────────────────────────────┘               │
│                          │                                                  │
│                          │ 读取用例信息                                      │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │ 运行回归测试                                            │               │
│  │ - 从 PostgreSQL 读取用例                               │               │
│  │ - 从 S3 下载测试文件                                    │               │
│  │ - 运行测试                                              │               │
│  └─────────────────────────────────────────────────────────┘               │
│                          │                                                  │
│                          │ 生成测试结果                                      │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │ 测试结果 Metrics                                        │               │
│  │ ┌───────────────────────────────────────────────────┐ │               │
│  │ │ 发送方式：Datadog HTTP API                         │ │               │
│  │ │                                                    │ │               │
│  │ │ POST https://api.datadoghq.com/api/v1/series     │ │               │
│  │ │ {                                                  │ │               │
│  │ │   "metric": "kyc.test.accuracy",                 │ │               │
│  │ │   "value": 95.5                                   │ │               │
│  │ │ }                                                  │ │               │
│  │ │                                                    │ │               │
│  │ │ ❌ 不能用 SQL：                                    │ │               │
│  │ │ INSERT INTO datadog.metrics VALUES (...);        │ │               │
│  │ └───────────────────────────────────────────────────┘ │               │
│  └─────────────────────────────────────────────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

### 2. 什么时候用 SQL？什么时候用 HTTP API？

**决策树**：

```
需要存储数据？
    │
    ├─ 是结构化数据（用户、订单、产品）？
    │   ├─ 是 → 使用 SQL（MySQL、PostgreSQL）
    │   │        ✅ 需要复杂查询
    │   │        ✅ 需要事务支持
    │   │        ✅ 需要持久化存储
    │   │
    │   └─ 否 → 继续判断
    │
    ├─ 是时序数据（Metrics、Logs、Traces）？
    │   ├─ 是 → 使用 HTTP API（Datadog、Prometheus）
    │   │        ✅ 主要用于监控
    │   │        ✅ 不需要复杂查询
    │   │        ✅ 不需要事务支持
    │   │
    │   └─ 否 → 继续判断
    │
    └─ 是文件数据（图片、PDF、视频）？
        └─ 是 → 使用对象存储（S3、GCS）
                 ✅ 大文件存储
                 ✅ 不需要查询
                 ✅ 通过 HTTP API 访问
```

### 3. 详细对比表

| 维度 | SQL 数据库 | HTTP API (Datadog) |
|------|-----------|-------------------|
| **使用场景** | 应用数据存储 | 监控数据发送 |
| **数据类型** | 结构化数据（用户、订单） | 时序数据（Metrics、Logs） |
| **连接方式** | `mysql.connect()` | `requests.post()` |
| **数据格式** | SQL 语句 | JSON |
| **查询方式** | `SELECT * FROM table` | Datadog Dashboard |
| **事务支持** | ✅ 支持（ACID） | ❌ 不支持 |
| **复杂查询** | ✅ 支持（JOIN、GROUP BY） | ❌ 不支持（简单聚合） |
| **持久化** | ✅ 永久存储 | ✅ 可配置保留时间 |
| **示例** | 用户表、订单表 | CPU 使用率、错误日志 |

### 4. Datadog 不是 SQL 数据库

**关键理解**：

| 存储方式 | 说明 |
|---------|------|
| **不是 SQL** | ❌ 不是 MySQL、PostgreSQL 等关系型数据库 |
| **是 HTTP API** | ✅ 使用 HTTP REST API 发送数据 |
| **数据格式** | ✅ JSON 格式，通过 HTTP POST 请求发送 |
| **查询方式** | ✅ 通过 Datadog Dashboard 查询，不是 SQL 查询 |

**为什么不用 SQL？**
- Datadog 是**监控和可观测性平台**，不是数据存储平台
- Datadog 内部使用**时序数据库**（Time Series Database），但对外暴露的是 **HTTP API**
- 用户不需要直接访问数据库，只需要通过 API 发送数据

---

### 2. Datadog 数据发送方式对比

| 方式 | SQL 数据库 | Datadog HTTP API |
|------|-----------|------------------|
| **连接方式** | `mysql.connect()` 或 `psycopg2.connect()` | `requests.post()` |
| **数据格式** | SQL 语句 | JSON |
| **接口** | `INSERT INTO table VALUES (...)` | `POST /api/v1/series` |
| **认证** | 用户名/密码 | API Key |
| **查询** | `SELECT * FROM table` | Datadog Dashboard |

---

## 🐍 Python 发送数据到 Datadog

### 方法 1：直接使用 HTTP API（最基础）

#### 发送 Metrics（指标）

```python
import requests
import json
from datetime import datetime

# Datadog API 配置
DD_API_KEY = "YOUR_DD_API_KEY"
DD_SITE = "datadoghq.com"  # 或 datadoghq.eu

def send_metric(metric_name: str, value: float, tags: list = None):
    """发送指标到 Datadog"""
    url = f"https://api.{DD_SITE}/api/v1/series"
    
    payload = {
        "series": [{
            "metric": metric_name,
            "points": [[int(datetime.now().timestamp()), value]],
            "tags": tags or []
        }]
    }
    
    headers = {
        "Content-Type": "application/json",
        "DD-API-KEY": DD_API_KEY
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.status_code == 202

# 使用示例
send_metric(
    metric_name="kyc.regression_test.accuracy",
    value=95.5,
    tags=["env:production", "service:kyc"]
)
```

#### 发送 Logs（日志）

```python
def send_log(level: str, message: str, service: str = "kyc-service", **tags):
    """发送日志到 Datadog"""
    url = f"https://http-intake.logs.{DD_SITE}/v1/input/{DD_API_KEY}"
    
    log_entry = {
        "timestamp": int(datetime.now().timestamp() * 1000),  # 毫秒时间戳
        "level": level,
        "message": message,
        "service": service,
        "ddtags": ",".join([f"{k}:{v}" for k, v in tags.items()])
    }
    
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=log_entry, headers=headers)
    return response.status_code == 200

# 使用示例
send_log(
    level="ERROR",
    message="Regression test failed: accuracy dropped from 96% to 94%",
    service="kyc-service",
    env="production",
    test_type="regression"
)
```

#### 发送 Events（事件）

```python
def send_event(title: str, text: str, alert_type: str = "info", **tags):
    """发送事件到 Datadog"""
    url = f"https://api.{DD_SITE}/api/v1/events"
    
    payload = {
        "title": title,
        "text": text,
        "alert_type": alert_type,  # info, warning, error, success
        "tags": [f"{k}:{v}" for k, v in tags.items()]
    }
    
    headers = {
        "Content-Type": "application/json",
        "DD-API-KEY": DD_API_KEY
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.status_code == 202

# 使用示例
send_event(
    title="Regression Test Failed",
    text="Accuracy dropped from 96% to 94%",
    alert_type="error",
    env="production",
    service="kyc",
    test_type="regression"
)
```

---

### 方法 2：使用 Datadog Python 库（推荐）

#### 安装库

```bash
pip install datadog
```

#### 初始化并发送数据

```python
from datadog import initialize, api
from datetime import datetime

# 初始化（不是 SQL 连接！）
initialize(
    api_key="YOUR_DD_API_KEY",
    app_key="YOUR_DD_APP_KEY"  # 可选，用于高级功能
)

# 发送 Metrics
api.Metric.send(
    metric="kyc.regression_test.accuracy",
    points=[(int(datetime.now().timestamp()), 95.5)],
    tags=["env:production", "service:kyc"]
)

# 发送 Events
api.Event.create(
    title="Regression Test Failed",
    text="Accuracy dropped from 96% to 94%",
    alert_type="error",
    tags=["env:production", "service:kyc", "test_type:regression"]
)
```

---

### 方法 3：批量发送（推荐用于生产环境）

```python
import requests
from datetime import datetime

class DatadogSender:
    """批量发送数据到 Datadog"""
    
    def __init__(self, api_key: str, site: str = "datadoghq.com"):
        self.api_key = api_key
        self.site = site
        self.metrics_buffer = []
        self.logs_buffer = []
    
    def add_metric(self, metric_name: str, value: float, tags: list = None):
        """添加指标到缓冲区"""
        self.metrics_buffer.append({
            "metric": metric_name,
            "points": [[int(datetime.now().timestamp()), value]],
            "tags": tags or []
        })
        
        # 达到批量大小，立即发送
        if len(self.metrics_buffer) >= 100:
            self.flush_metrics()
    
    def flush_metrics(self):
        """批量发送指标"""
        if not self.metrics_buffer:
            return
        
        url = f"https://api.{self.site}/api/v1/series"
        payload = {"series": self.metrics_buffer}
        
        headers = {
            "Content-Type": "application/json",
            "DD-API-KEY": self.api_key
        }
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 202:
            count = len(self.metrics_buffer)
            self.metrics_buffer.clear()
            print(f"✅ Sent {count} metrics to Datadog")
        else:
            print(f"❌ Failed to send metrics: {response.text}")
    
    def add_log(self, level: str, message: str, service: str = "kyc-service", **tags):
        """添加日志到缓冲区"""
        log_entry = {
            "timestamp": int(datetime.now().timestamp() * 1000),
            "level": level,
            "message": message,
            "service": service,
            "ddtags": ",".join([f"{k}:{v}" for k, v in tags.items()])
        }
        self.logs_buffer.append(log_entry)
        
        # 达到批量大小，立即发送
        if len(self.logs_buffer) >= 100:
            self.flush_logs()
    
    def flush_logs(self):
        """批量发送日志"""
        if not self.logs_buffer:
            return
        
        url = f"https://http-intake.logs.{self.site}/v1/input/{self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=self.logs_buffer, headers=headers)
        if response.status_code == 200:
            count = len(self.logs_buffer)
            self.logs_buffer.clear()
            print(f"✅ Sent {count} logs to Datadog")
        else:
            print(f"❌ Failed to send logs: {response.text}")

# 使用示例
sender = DatadogSender(api_key="YOUR_DD_API_KEY")

# 添加多个指标
sender.add_metric("kyc.test.accuracy", 95.5, ["env:production"])
sender.add_metric("kyc.test.passed", 95, ["env:production"])
sender.add_metric("kyc.test.failed", 5, ["env:production"])

# 批量发送
sender.flush_metrics()
```

---

## 📋 KYC 项目实际案例：发送回归测试结果到 Datadog

### 完整示例

```python
import requests
import json
from datetime import datetime
from typing import Dict, List

def send_regression_test_results_to_datadog(results: dict, api_key: str):
    """发送回归测试结果到 Datadog"""
    
    DD_SITE = "datadoghq.com"
    
    # 1. 发送 Metrics（指标）
    metrics_url = f"https://api.{DD_SITE}/api/v1/series"
    
    metrics_payload = {
        "series": [
            {
                "metric": "kyc.regression_test.total",
                "points": [[int(datetime.now().timestamp()), results["total"]]],
                "tags": ["env:production", "service:kyc"]
            },
            {
                "metric": "kyc.regression_test.passed",
                "points": [[int(datetime.now().timestamp()), results["passed"]]],
                "tags": ["env:production", "service:kyc"]
            },
            {
                "metric": "kyc.regression_test.failed",
                "points": [[int(datetime.now().timestamp()), results["failed"]]],
                "tags": ["env:production", "service:kyc"]
            },
            {
                "metric": "kyc.regression_test.accuracy",
                "points": [[int(datetime.now().timestamp()), results["accuracy"]]],
                "tags": ["env:production", "service:kyc"]
            }
        ]
    }
    
    metrics_headers = {
        "Content-Type": "application/json",
        "DD-API-KEY": api_key
    }
    
    metrics_response = requests.post(
        metrics_url,
        json=metrics_payload,
        headers=metrics_headers
    )
    
    # 2. 发送 Logs（失败用例）
    if results["failed"] > 0:
        logs_url = f"https://http-intake.logs.{DD_SITE}/v1/input/{api_key}"
        
        for failed_case in results.get("failed_cases", [])[:10]:  # 只发送前10个
            log_entry = {
                "timestamp": int(datetime.now().timestamp() * 1000),
                "level": "ERROR",
                "message": f"Regression test failed: {failed_case['case_id']}",
                "service": "kyc-service",
                "ddtags": f"env:production,test_type:regression,case_id:{failed_case['case_id']}"
            }
            
            requests.post(
                logs_url,
                json=log_entry,
                headers={"Content-Type": "application/json"}
            )
    
    # 3. 发送 Event（如果测试失败）
    if results["failed"] > 0:
        events_url = f"https://api.{DD_SITE}/api/v1/events"
        
        event_payload = {
            "title": "Regression Test Failed",
            "text": f"Failed {results['failed']}/{results['total']} test cases. Accuracy: {results['accuracy']}%",
            "alert_type": "error",
            "tags": [
                "env:production",
                "service:kyc",
                "test_type:regression"
            ]
        }
        
        requests.post(
            events_url,
            json=event_payload,
            headers={
                "Content-Type": "application/json",
                "DD-API-KEY": api_key
            }
        )
    
    return metrics_response.status_code == 202

# 使用示例
results = {
    "total": 100,
    "passed": 95,
    "failed": 5,
    "accuracy": 95.0,
    "failed_cases": [
        {"case_id": "normal_001", "error": "Field mismatch"},
        {"case_id": "edge_002", "error": "Schema validation failed"}
    ]
}

send_regression_test_results_to_datadog(
    results=results,
    api_key="YOUR_DD_API_KEY"
)
```

---

## 🔑 关键 API 端点总结

### Metrics API

```
POST https://api.datadoghq.com/api/v1/series
Headers:
  Content-Type: application/json
  DD-API-KEY: YOUR_API_KEY

Body:
{
  "series": [{
    "metric": "metric.name",
    "points": [[timestamp, value]],
    "tags": ["tag1:value1", "tag2:value2"]
  }]
}
```

### Logs API

```
POST https://http-intake.logs.datadoghq.com/v1/input/YOUR_API_KEY
Headers:
  Content-Type: application/json

Body:
{
  "timestamp": 1234567890000,
  "level": "ERROR",
  "message": "Log message",
  "service": "service-name",
  "ddtags": "env:production,tag:value"
}
```

### Events API

```
POST https://api.datadoghq.com/api/v1/events
Headers:
  Content-Type: application/json
  DD-API-KEY: YOUR_API_KEY

Body:
{
  "title": "Event title",
  "text": "Event description",
  "alert_type": "error",
  "tags": ["env:production", "service:kyc"]
}
```

---

## 📊 对比：SQL vs HTTP API

### SQL 方式（不适用于 Datadog）

```python
# ❌ 错误方式：Datadog 不支持 SQL
import mysql.connector

conn = mysql.connector.connect(
    host="datadog.com",  # ❌ 不存在
    user="user",
    password="password",
    database="datadog"
)

cursor = conn.cursor()
cursor.execute("INSERT INTO metrics VALUES (...)")  # ❌ 不支持
```

### HTTP API 方式（正确方式）

```python
# ✅ 正确方式：使用 HTTP API
import requests

response = requests.post(
    "https://api.datadoghq.com/api/v1/series",
    json={"series": [...]},
    headers={"DD-API-KEY": "YOUR_API_KEY"}
)
```

---

## 💡 总结

### 核心要点

1. **Datadog 不是 SQL 数据库**
   - ❌ 不是 MySQL、PostgreSQL
   - ✅ 使用 HTTP REST API

2. **发送方式**
   - ✅ 使用 `requests.post()` 发送 JSON 数据
   - ✅ 不需要 SQL 连接（`mysql.connect()`）
   - ✅ 使用 API Key 认证

3. **API 端点**
   - Metrics: `https://api.datadoghq.com/api/v1/series`
   - Logs: `https://http-intake.logs.datadoghq.com/v1/input/{API_KEY}`
   - Events: `https://api.datadoghq.com/api/v1/events`

4. **推荐方法**
   - 开发环境：使用 `datadog` Python 库（简单）
   - 生产环境：使用批量发送（高效）

---

**下一步**：
- 查看 [Golden Set 存储和使用详解](./KYC_Day03_A1_B1_Golden_Set存储和使用详解.md)
- 查看 [Release Gate 设计详解](./KYC_Day03_A1_B2_Release_Gate设计详解.md)
