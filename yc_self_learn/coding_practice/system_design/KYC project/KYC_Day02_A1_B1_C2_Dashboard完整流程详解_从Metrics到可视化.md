# Day 2_A1_B1_C2：Dashboard 完整流程详解（从 Metrics 到可视化）

---
doc_type: glossary
layer: L3
scope_in:  Dashboard 的完整流程、Metrics 如何变成可视化、是否需要调用 API、整个数据流向
scope_out: 具体 Dashboard 配置步骤（见 howto）；Dashboard 的高级功能（见 L4）
inputs:   (读者) 疑问：只需要调用 Dashboard 的 API 就能看到 Metrics 的可视化吗？
outputs:  Dashboard 完整流程 + Metrics 数据流向 + 配置方法 + 实际操作
entrypoints: [ 核心问题 ]
children: []
related: [ Dashboard, Grafana, Datadog, Prometheus, Metrics, 可视化, KYC_Day02_A1_B1_Dashboard实现方式详解.md ]
---

## Definition（定义）

**核心问题**：**只需要调用 Dashboard 的 API 就能看到 Metrics 的可视化吗？**

**核心答案**：
- ❌ **不是直接调用 Dashboard 的 API**
- ✅ **而是两步流程**：
  1. **推送 Metrics 到监控平台**（如 Prometheus/Datadog）
  2. **在 Dashboard UI 中配置查询**（自动读取 Metrics 并可视化）

**类比**：
- **Dashboard** = **报表工具**（Excel/Power BI）
- **Metrics** = **数据源**（数据库/Excel 文件）
- **配置查询** = **写 SQL/公式**（不需要写代码，只需要配置）

---

## 🎯 核心问题

### 是不是只需要调用 Dashboard 的 API？

**答案**：**不是！是两步流程，不是直接调用 API**。

**错误理解**：
```
❌ 调用 Dashboard API → 直接看到可视化
```

**正确流程**：
```
✅ 推送 Metrics → 监控平台存储 → Dashboard 配置查询 → 自动可视化
```

---

## 📊 完整流程（从 Metrics 到可视化）

### 流程概览

```
┌─────────────────────────────────────────────────────────────┐
│  1. 你的程序（KYC Pipeline）                                  │
│     ↓                                                         │
│     记录 Metrics（推送到监控平台）                              │
│     ↓                                                         │
│  2. 监控平台（Prometheus/Datadog）                           │
│     ↓                                                         │
│     存储 Metrics 数据                                         │
│     ↓                                                         │
│  3. Dashboard 平台（Grafana/Datadog UI）                     │
│     ↓                                                         │
│     配置查询（通过 UI，不需要写代码）                            │
│     ↓                                                         │
│     自动读取 Metrics 并可视化                                  │
│     ↓                                                         │
│  4. 浏览器（你看到 Dashboard）                                │
│     看到 Metrics 的可视化图表                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 详细步骤（以 Grafana + Prometheus 为例）

### 步骤 1：推送 Metrics 到 Prometheus

**不是调用 Dashboard API，而是推送 Metrics 数据到监控平台**。

**代码示例**：
```python
# 1. 从 _summary.json 计算 Metrics，推送到 Prometheus
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import json

# 定义 Metrics
requests_total = Counter('kyc_requests_total', 'Total requests')
errors_total = Counter('kyc_errors_total', 'Total errors', ['error_code'])
latency_histogram = Histogram('kyc_latency_seconds', 'Request latency')
success_rate = Gauge('kyc_success_rate', 'Success rate')

# 从 _summary.json 计算并推送 Metrics
def push_metrics_to_prometheus(summary_path: str):
    """从 _summary.json 计算 Metrics，推送到 Prometheus"""
    with open(summary_path) as f:
        summary = json.load(f)
    
    results = summary["results"]
    total = len(results)
    successes = sum(1 for r in results if r.get("status") == "success")
    
    # 更新 Metrics（这些 Metrics 会自动暴露给 Prometheus）
    requests_total.inc(total)
    for result in results:
        if result.get("status") == "success":
            latency_histogram.observe(result.get("latency_ms", 0) / 1000)
        else:
            error_code = result.get("error_code", "unknown")
            errors_total.labels(error_code=error_code).inc(1)
    
    success_rate.set(successes / total if total > 0 else 0)

# 启动 Prometheus HTTP 服务器
# Prometheus 会定期从 http://localhost:8000/metrics 拉取数据
start_http_server(8000)
```

**关键点**：
- ✅ **推送 Metrics**：你的程序记录 Metrics，Prometheus 自动拉取
- ✅ **不需要调用 Dashboard API**：Prometheus 自己会定期拉取数据
- ✅ **数据存储**：Prometheus 存储 Metrics 数据

---

### 步骤 2：Prometheus 存储 Metrics

**Prometheus 会自动从你的程序拉取 Metrics 并存储**。

**Prometheus 配置**（`prometheus.yml`）：
```yaml
global:
  scrape_interval: 15s  # 每 15 秒拉取一次

scrape_configs:
  - job_name: 'kyc-metrics'
    static_configs:
      - targets: ['localhost:8000']  # 从你的程序拉取 Metrics
```

**关键点**：
- ✅ **自动拉取**：Prometheus 会定期从你的程序拉取 Metrics
- ✅ **数据存储**：Prometheus 存储 Metrics 数据（时间序列数据库）
- ✅ **不需要你调用 API**：Prometheus 自己会拉取数据

---

### 步骤 3：配置 Grafana Dashboard（通过 UI，不需要写代码）

**这不是调用 API，而是在 Grafana UI 中配置查询**。

**操作步骤**（通过 UI，不需要写代码）：

1. **访问 Grafana**：http://localhost:3000

2. **添加数据源**：
   - 点击 "Configuration" → "Data Sources"
   - 选择 "Prometheus"
   - 输入 Prometheus URL：http://localhost:9090
   - 点击 "Save & Test"

3. **创建 Dashboard**：
   - 点击 "Create" → "Dashboard"
   - 点击 "Add visualization"
   - 选择数据源：Prometheus
   - **配置查询**（这就是"配置"，不是调用 API）：

   ```
   # RPS（每秒请求数）
   rate(kyc_requests_total[5m])
   
   # Error Rate（错误率）
   rate(kyc_errors_total[5m]) / rate(kyc_requests_total[5m])
   
   # p95 Latency（95 分位延迟）
   histogram_quantile(0.95, kyc_latency_seconds_bucket)
   ```

4. **保存 Dashboard**：点击 "Save"

**关键点**：
- ✅ **通过 UI 配置**：不需要写代码，只需要在 UI 中配置查询
- ✅ **自动读取**：Grafana 会自动从 Prometheus 读取 Metrics 数据
- ✅ **自动可视化**：Grafana 会根据查询自动生成图表

---

### 步骤 4：浏览器中查看 Dashboard

**访问 Grafana Dashboard，看到 Metrics 的可视化图表**。

**操作**：
- 访问 http://localhost:3000
- 打开你创建的 Dashboard
- 看到实时的 Metrics 可视化图表

**关键点**：
- ✅ **实时更新**：Dashboard 会自动从 Prometheus 读取最新的 Metrics 数据
- ✅ **可视化图表**：Grafana 会自动生成图表（折线图、柱状图等）
- ✅ **不需要调用 API**：Grafana 会自动读取数据并可视化

---

## 📊 数据流向详解

### 完整数据流向

```
1. KYC Pipeline（你的程序）
   └─ 记录 Metrics（Counter、Histogram、Gauge）
      └─ 暴露 HTTP 端点：http://localhost:8000/metrics
         └─ 返回 Metrics 数据（文本格式）

2. Prometheus（监控平台）
   └─ 定期拉取：http://localhost:8000/metrics
      └─ 存储 Metrics 数据（时间序列数据库）
         └─ 提供查询 API：http://localhost:9090/api/v1/query

3. Grafana（Dashboard 平台）
   └─ 配置数据源：连接到 Prometheus
      └─ 配置查询（PromQL）：
         - rate(kyc_requests_total[5m])
         - histogram_quantile(0.95, kyc_latency_seconds_bucket)
      └─ 自动查询 Prometheus API：http://localhost:9090/api/v1/query
         └─ 获取 Metrics 数据
            └─ 自动生成图表（折线图、柱状图等）

4. 浏览器（你看到的 Dashboard）
   └─ 访问 Grafana：http://localhost:3000
      └─ 看到 Metrics 的可视化图表
```

---

## 💡 关键理解

### 1. 不是调用 Dashboard API，而是配置查询

**错误理解**：
```python
# ❌ 不是这样调用 Dashboard API
response = requests.post(
    "http://grafana:3000/api/dashboard",
    json={"metrics": ["kyc_requests_total"]}
)
```

**正确理解**：
```python
# ✅ 推送 Metrics 到 Prometheus
requests_total.inc(100)

# ✅ 在 Grafana UI 中配置查询（不需要写代码）
# 在 Grafana UI 中输入：rate(kyc_requests_total[5m])
# Grafana 会自动查询 Prometheus 并可视化
```

---

### 2. 配置查询 = 写 SQL/公式（不需要写代码）

**类比**：
- **Excel**：写公式 `=SUM(A1:A10)`，Excel 自动计算并显示结果
- **Grafana**：配置查询 `rate(kyc_requests_total[5m])`，Grafana 自动查询并可视化

**例子**（Grafana UI 配置）：
```
1. 在 Grafana UI 中点击 "Add visualization"
2. 选择数据源：Prometheus
3. 在 "Query" 框中输入：
   rate(kyc_requests_total[5m])
4. 点击 "Apply"
5. Grafana 自动查询 Prometheus 并生成图表
```

**关键点**：
- ✅ **不需要写代码**：只需要在 UI 中配置查询
- ✅ **自动查询**：Grafana 会自动查询 Prometheus 并获取数据
- ✅ **自动可视化**：Grafana 会自动生成图表

---

### 3. Dashboard 自动更新（不需要你调用 API）

**Dashboard 会自动从监控平台读取最新的 Metrics 数据**。

**流程**：
```
1. 你的程序推送 Metrics → Prometheus 存储
2. Grafana Dashboard 配置查询（PromQL）
3. Grafana 自动查询 Prometheus（每几秒查询一次）
4. Grafana 自动更新图表（显示最新的 Metrics 数据）
```

**关键点**：
- ✅ **自动更新**：Dashboard 会自动更新，不需要你调用 API
- ✅ **实时数据**：显示最新的 Metrics 数据
- ✅ **无需手动操作**：配置好后，Dashboard 会自动运行

---

## ⚖️ 对比：错误理解 vs 正确理解

### 错误理解

```
调用 Dashboard API → 直接看到可视化
```

**问题**：
- ❌ 不存在"Dashboard API"让你直接调用
- ❌ Dashboard 不是这样工作的
- ❌ 这种方式需要你自己写代码处理可视化

---

### 正确理解

```
推送 Metrics → 监控平台存储 → Dashboard 配置查询 → 自动可视化
```

**优势**：
- ✅ **不需要写代码**：只需要配置查询
- ✅ **自动更新**：Dashboard 自动从监控平台读取数据
- ✅ **可视化**：Dashboard 自动生成图表

---

## 📊 实际操作示例

### 完整流程（Grafana + Prometheus）

#### 1. 推送 Metrics（你的程序）

```python
from prometheus_client import Counter, start_http_server

requests_total = Counter('kyc_requests_total', 'Total requests')

# 记录 Metrics
requests_total.inc(100)

# 启动 HTTP 服务器（Prometheus 会从这里拉取数据）
start_http_server(8000)
```

#### 2. Prometheus 拉取并存储（自动）

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'kyc-metrics'
    static_configs:
      - targets: ['localhost:8000']
```

**Prometheus 自动**：
- ✅ 每 15 秒拉取一次 `http://localhost:8000/metrics`
- ✅ 存储 Metrics 数据
- ✅ 提供查询 API：`http://localhost:9090/api/v1/query`

#### 3. Grafana 配置查询（通过 UI，不需要写代码）

**操作步骤**：
1. 访问 Grafana：http://localhost:3000
2. 添加数据源：Prometheus（http://localhost:9090）
3. 创建 Dashboard → Add visualization
4. 在 "Query" 框中输入：
   ```
   rate(kyc_requests_total[5m])
   ```
5. 点击 "Apply"
6. 看到图表（自动生成）

#### 4. 浏览器查看 Dashboard（自动更新）

- 访问 Grafana Dashboard
- 看到实时的 Metrics 图表
- **自动更新**（每几秒更新一次，显示最新的 Metrics 数据）

---

## 💡 总结

### 核心答案

**是不是只需要调用 Dashboard 的 API？**
- ❌ **不是**：不是直接调用 Dashboard API
- ✅ **而是两步流程**：
  1. **推送 Metrics 到监控平台**（Prometheus/Datadog）
  2. **在 Dashboard UI 中配置查询**（自动读取 Metrics 并可视化）

### 关键要点

1. **推送 Metrics**：你的程序记录 Metrics，监控平台自动拉取并存储
2. **配置查询**：在 Dashboard UI 中配置查询（不需要写代码）
3. **自动可视化**：Dashboard 自动从监控平台读取数据并生成图表
4. **自动更新**：Dashboard 自动更新，显示最新的 Metrics 数据

### 类比

- **Dashboard** = **Excel/Power BI**（报表工具）
- **Metrics** = **数据库/Excel 文件**（数据源）
- **配置查询** = **写 SQL/公式**（不需要写代码，只需要配置）

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1_B1 Dashboard 实现方式详解（[KYC_Day02_A1_B1_Dashboard实现方式详解.md](./KYC_Day02_A1_B1_Dashboard实现方式详解.md)） |
| **Related** | Dashboard、Grafana、Datadog、Prometheus、Metrics、可视化 |
