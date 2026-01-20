# Day 2_A1_B1：Dashboard 实现方式详解

---
doc_type: glossary
layer: L3
scope_in:  Dashboard 的实现方式、是否需要自己写、现成平台 vs 自己开发、如何配置 Dashboard
scope_out: 具体 Dashboard 配置步骤（见 howto）；Dashboard 的高级功能（见 L4）
inputs:   (读者) 疑问：Dashboard 是自己写的吗？还是用现成的平台？
outputs:  Dashboard 实现方式对比 + 推荐方案 + 配置方法 + 实际例子
entrypoints: [ 核心答案 ]
children: [ 
  KYC_Day02_A1_B1_C1_前端框架React和Vue详解.md（前端框架 React 和 Vue 详解），
  KYC_Day02_A1_B1_C2_Dashboard完整流程详解_从Metrics到可视化.md（Dashboard 完整流程：从 Metrics 到可视化），
  KYC_Day02_A1_B1_C3_Metrics数据安全性与合规性考虑.md（Metrics 数据安全性与合规性考虑），
  KYC_Day02_A1_B1_C4_Dashboard_vs_定期报告_实时监控_vs_被动查看.md（Dashboard vs 定期报告：实时监控 vs 被动查看），
  KYC_Day02_A1_B1_C5_核心概念详解_Trade_off_Input_Output_优势_目的.md（核心概念详解：Trade-off、Input、Output、优势、目的），
  KYC_Day02_A1_B1_C6_指标聚合和降采样详解_Downsampling.md（指标聚合和降采样详解：Downsampling）
]
related: [ Dashboard, Grafana, Datadog, CloudWatch, Prometheus, 可观测性, KYC_Day02_A1_可观测性详解.md ]
---

## Definition（定义）

**核心答案**：**Dashboard 通常不是自己从零写的，而是使用现成的监控平台（如 Grafana、Datadog、CloudWatch），然后配置和自定义**。

**类比**：
- **自己写 Dashboard** = **自己造轮子**（不推荐，除非有特殊需求）
- **使用现成平台** = **用现成的工具**（推荐，大部分公司都这样做）

---

## 🎯 核心答案

### Dashboard 是自己写的吗？

**答案**：**通常不是自己写的，而是使用现成的监控平台**。

**原因**：
- ✅ **现成平台功能强大**：Grafana、Datadog、CloudWatch 等已经实现了完整的 Dashboard 功能
- ✅ **开发成本高**：自己写 Dashboard 需要前端 + 后端 + 数据存储，成本高
- ✅ **维护成本高**：需要持续维护和更新
- ✅ **时间成本高**：开发 Dashboard 需要大量时间

**什么时候自己写**：
- ⚠️ **特殊需求**：现成平台无法满足的特殊需求
- ⚠️ **大公司自研**：有专门的团队开发监控平台
- ⚠️ **成本考虑**：如果使用量非常大，自研可能更经济

---

## 📊 Dashboard 实现方式对比

### 方式 1：现成平台（推荐，大部分公司）

**平台选择**：

| 平台 | 类型 | 成本 | 适用场景 |
|------|------|------|---------|
| **Grafana + Prometheus** | 开源 | 免费 | 中小公司、成本敏感 |
| **Datadog** | 商业 | 按使用量付费 | 大公司、功能需求高 |
| **CloudWatch** | AWS 云平台 | 按使用量付费 | AWS 用户 |
| **New Relic** | 商业 | 按使用量付费 | 大公司 |
| **Splunk** | 商业 | 按使用量付费 | 大公司、日志分析需求高 |

**实现方式**：
- ✅ **安装平台**：安装 Grafana/Datadog Agent 等
- ✅ **配置数据源**：连接 Metrics/Logs/Traces 数据源
- ✅ **配置 Dashboard**：通过 UI 拖拽组件，配置 Dashboard（**不需要写代码**）
- ✅ **设置告警**：配置告警规则

**优点**：
- ✅ **快速上线**：不需要开发，直接配置
- ✅ **功能强大**：现成平台功能完善
- ✅ **持续更新**：平台会持续更新功能
- ✅ **社区支持**：有社区和文档支持

**缺点**：
- ⚠️ **成本**：商业平台需要付费
- ⚠️ **定制化限制**：可能无法满足所有特殊需求

---

### 方式 2：自己开发（不推荐，除非特殊需求）

**实现方式**：
- ✅ **前端开发**：使用 React/Vue 等框架开发 Dashboard UI
- ✅ **后端开发**：开发 API 提供 Metrics/Logs/Traces 数据
- ✅ **数据存储**：使用数据库存储指标数据
- ✅ **实时更新**：实现 WebSocket 或轮询更新数据

**优点**：
- ✅ **完全定制**：可以完全按照需求定制
- ✅ **成本可控**：如果使用量大，可能更经济

**缺点**：
- ❌ **开发成本高**：需要前端 + 后端团队
- ❌ **维护成本高**：需要持续维护和更新
- ❌ **时间成本高**：开发需要大量时间
- ❌ **功能可能不完善**：可能缺少一些高级功能

---

### 方式 3：开源工具（推荐小公司）

**实现方式**：
- ✅ **Grafana + Prometheus**：开源免费
- ✅ **ELK Stack**：Elasticsearch + Logstash + Kibana（日志分析）
- ✅ **Jaeger**：链路追踪

**优点**：
- ✅ **免费**：开源免费
- ✅ **功能强大**：功能完善
- ✅ **社区支持**：有活跃的社区

**缺点**：
- ⚠️ **需要运维**：需要自己部署和维护
- ⚠️ **学习成本**：需要学习如何使用

---

## 💡 推荐方案（KYC 项目）

### 小公司/PoV 阶段：Grafana + Prometheus

**为什么推荐**：
- ✅ **免费**：开源免费，适合 PoV 阶段
- ✅ **功能强大**：支持 Metrics、Logs、Traces
- ✅ **易于配置**：通过 UI 配置，不需要写代码
- ✅ **社区支持**：有大量文档和示例

**实现步骤**：

#### 步骤 1：安装 Prometheus

```bash
# 下载 Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-2.45.0.linux-amd64.tar.gz
cd prometheus-2.45.0.linux-amd64

# 配置 Prometheus
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kyc-metrics'
    static_configs:
      - targets: ['localhost:9090']
```

#### 步骤 2：推送 Metrics 到 Prometheus

```python
# 从 _summary.json 计算 Metrics，推送到 Prometheus
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# 定义 Metrics
requests_total = Counter('kyc_requests_total', 'Total requests')
errors_total = Counter('kyc_errors_total', 'Total errors', ['error_code'])
latency_histogram = Histogram('kyc_latency_seconds', 'Request latency')
success_rate = Gauge('kyc_success_rate', 'Success rate')

# 从 _summary.json 计算并推送 Metrics
def push_metrics_to_prometheus(summary_path: Path):
    """从 _summary.json 计算 Metrics，推送到 Prometheus"""
    with open(summary_path) as f:
        summary = json.load(f)
    
    results = summary["results"]
    total = len(results)
    successes = sum(1 for r in results if r.get("status") == "success")
    failures = total - successes
    
    # 更新 Metrics
    requests_total.inc(total)
    errors_total.labels(error_code='total').inc(failures)
    
    for result in results:
        if result.get("status") == "success":
            latency_histogram.observe(result.get("latency_ms", 0) / 1000)
        else:
            error_code = result.get("error_code", "unknown")
            errors_total.labels(error_code=error_code).inc(1)
    
    success_rate.set(successes / total if total > 0 else 0)

# 启动 Prometheus HTTP 服务器
start_http_server(8000)  # Prometheus 会从 http://localhost:8000/metrics 拉取数据
```

#### 步骤 3：安装 Grafana

```bash
# 安装 Grafana
wget https://dl.grafana.com/oss/release/grafana-10.0.0.linux-amd64.tar.gz
tar xvfz grafana-10.0.0.linux-amd64.tar.gz
cd grafana-10.0.0

# 启动 Grafana
./bin/grafana-server
```

#### 步骤 4：配置 Grafana Dashboard（通过 UI，不需要写代码）

**操作步骤**：
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
   - 配置查询：
     - **RPS**：`rate(kyc_requests_total[5m])`
     - **Error Rate**：`rate(kyc_errors_total[5m]) / rate(kyc_requests_total[5m])`
     - **p95 Latency**：`histogram_quantile(0.95, kyc_latency_seconds_bucket)`
     - **Success Rate**：`kyc_success_rate`

4. **保存 Dashboard**：点击 "Save"

**关键点**：
- ✅ **不需要写代码**：通过 UI 配置
- ✅ **拖拽组件**：选择图表类型，配置查询
- ✅ **实时更新**：Dashboard 会自动更新数据

---

### 大公司/Production：Datadog

**为什么推荐**：
- ✅ **功能全面**：Metrics、Logs、Traces 一体化
- ✅ **易于使用**：UI 友好，配置简单
- ✅ **告警集成**：内置告警功能
- ✅ **SaaS 服务**：不需要自己运维

**实现步骤**：

#### 步骤 1：注册 Datadog 账号

```bash
# 注册 Datadog 账号，获取 API Key
# https://app.datadoghq.com/
```

#### 步骤 2：安装 Datadog Agent

```bash
# 安装 Datadog Agent
DD_API_KEY=your-api-key DD_SITE="datadoghq.com" bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script_agent7.sh)"
```

#### 步骤 3：集成 Datadog SDK

```python
# 集成 Datadog SDK
from datadog import initialize, statsd

initialize(api_key='your-api-key', app_key='your-app-key')

# 记录 Metrics
def record_metrics(result: dict):
    """记录 Metrics 到 Datadog"""
    if result.get("status") == "success":
        statsd.increment('kyc.requests.success')
        statsd.histogram('kyc.latency', result.get("latency_ms", 0))
    else:
        error_code = result.get("error_code", "unknown")
        statsd.increment('kyc.requests.error', tags=[f'error_code:{error_code}'])
    
    statsd.increment('kyc.requests.total')
```

#### 步骤 4：配置 Datadog Dashboard（通过 UI，不需要写代码）

**操作步骤**：
1. **访问 Datadog**：https://app.datadoghq.com/
2. **创建 Dashboard**：
   - 点击 "Dashboards" → "New Dashboard"
   - 点击 "Add Widgets"
   - 选择图表类型（Time Series、Heatmap 等）
   - 配置查询：
     - **RPS**：`sum:kyc.requests.total{*}.as_rate()`
     - **Error Rate**：`sum:kyc.requests.error{*}.as_rate() / sum:kyc.requests.total{*}.as_rate()`
     - **p95 Latency**：`p95:kyc.latency{*}`
3. **保存 Dashboard**：点击 "Save"

**关键点**：
- ✅ **不需要写代码**：通过 UI 配置
- ✅ **自动收集**：Datadog Agent 自动收集系统指标
- ✅ **告警集成**：可以直接在 Dashboard 中配置告警

---

## 🔍 实际例子：KYC 项目 Dashboard 配置

### 使用 Grafana + Prometheus

**Dashboard 配置示例**（通过 Grafana UI 配置）：

```
Dashboard: KYC 实时监控

Panel 1: RPS (Time Series)
- Query: rate(kyc_requests_total[5m])
- Visualization: Time Series
- Y-axis: Requests per second

Panel 2: Error Rate (Time Series)
- Query: rate(kyc_errors_total[5m]) / rate(kyc_requests_total[5m])
- Visualization: Time Series
- Y-axis: Percentage (0-100%)

Panel 3: p95 Latency (Time Series)
- Query: histogram_quantile(0.95, kyc_latency_seconds_bucket)
- Visualization: Time Series
- Y-axis: Milliseconds

Panel 4: Success Rate (Gauge)
- Query: kyc_success_rate
- Visualization: Gauge
- Thresholds: 0-95 (red), 95-99 (yellow), 99-100 (green)

Panel 5: Error Breakdown (Pie Chart)
- Query: sum by (error_code) (rate(kyc_errors_total[5m]))
- Visualization: Pie Chart
```

**关键点**：
- ✅ **不需要写代码**：在 Grafana UI 中配置
- ✅ **拖拽组件**：选择图表类型，输入查询语句
- ✅ **实时更新**：Dashboard 自动更新数据

---

## 📊 对比总结

### 实现方式对比

| 方式 | 开发成本 | 维护成本 | 功能 | 适用场景 |
|------|---------|---------|------|---------|
| **现成平台**（Grafana/Datadog） | **低**（配置即可） | **低**（平台维护） | **高**（功能完善） | **推荐，大部分公司** |
| **自己开发** | **高**（需要开发） | **高**（需要维护） | **中**（可能不完善） | **特殊需求、大公司自研** |
| **开源工具**（Grafana + Prometheus） | **低**（配置即可） | **中**（需要运维） | **高**（功能完善） | **小公司、成本敏感** |

### 推荐方案

**KYC 项目推荐**：
- ✅ **PoV 阶段**：Grafana + Prometheus（开源免费）
- ✅ **Production**：Datadog 或 CloudWatch（功能更全，SaaS 服务）

---

## 🎯 总结

### 核心答案

**Dashboard 通常不是自己从零写的，而是使用现成的监控平台（如 Grafana、Datadog、CloudWatch），然后配置和自定义**。

### 实现方式

1. **现成平台**（推荐）：
   - ✅ Grafana + Prometheus（开源免费）
   - ✅ Datadog（商业平台）
   - ✅ CloudWatch（AWS 云平台）

2. **自己开发**（不推荐，除非特殊需求）：
   - ❌ 开发成本高
   - ❌ 维护成本高
   - ❌ 时间成本高

### 配置方式

- ✅ **不需要写代码**：通过 UI 配置 Dashboard
- ✅ **拖拽组件**：选择图表类型，配置查询
- ✅ **实时更新**：Dashboard 自动更新数据

### KYC 项目推荐

- ✅ **PoV 阶段**：Grafana + Prometheus（开源免费）
- ✅ **Production**：Datadog 或 CloudWatch（功能更全）

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1 可观测性详解（[KYC_Day02_A1_可观测性详解.md](./KYC_Day02_A1_可观测性详解.md)） |
| **Related** | Dashboard、Grafana、Datadog、CloudWatch、Prometheus、可观测性 |
