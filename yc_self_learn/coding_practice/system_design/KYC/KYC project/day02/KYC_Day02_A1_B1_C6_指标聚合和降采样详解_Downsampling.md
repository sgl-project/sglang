# Day 2_A1_B1_C6：指标聚合和降采样详解（Downsampling）

---
doc_type: glossary
layer: L3
scope_in:  Metrics降采样（Downsampling）策略、指标聚合方法、指标生命周期管理
scope_out: 具体降采样实现代码（见 howto）；降采样算法的数学原理（见 L4）；降采样平台配置（见 reference）
inputs:   (读者) 疑问：长期存储Metrics数据，如何优化存储成本？如何设计Metrics的存储策略？
outputs:  降采样策略详解 + 指标聚合方法 + 指标生命周期管理 + 成本优化效果
entrypoints: [ 核心问题：Metrics降采样 ]
children: []
related: [ Downsampling, Metrics, 指标聚合, 指标生命周期, 成本优化, Dashboard, KYC_Day02_A1_B1_Dashboard实现方式详解.md, KYC_Day02_A1_B3_采样策略详解_Sampling_Strategy.md ]
---

## Definition（定义）

**核心问题**：**长期存储Metrics数据，如何优化存储成本？**

**核心答案**：
- ✅ **实时指标**：保持高精度（1分钟粒度），用于实时监控和告警
- ✅ **聚合指标**：降采样存储（1小时粒度、1天粒度），用于历史分析
- ✅ **指标生命周期**：实时数据保留1天，聚合数据保留90天

**类比**：
- **降采样** = **照片压缩**（原图清晰但占用空间大，缩略图模糊但占用空间小）
- **分层存储** = **文件归档**（近期文件详细，历史文件概览）

---

## 🎯 核心问题

### 问题场景

**场景1：存储成本高**
- "长期存储Metrics数据，存储成本会很高，如何优化？"
- "如何设计Metrics的存储策略？"

**场景2：查询性能**
- "历史Metrics数据查询很慢，如何优化？"
- "如何在保证查询性能的同时降低存储成本？"

**场景3：数据价值**
- "实时数据和历史数据的价值不同，如何区分存储？"
- "如何平衡实时监控和历史分析的需求？"

---

## 📊 降采样策略详解

### 1. 实时指标（高精度短期保留）

**定义**：**保持高精度，用于实时监控和告警**。

**特点**：
- ✅ **精度**：1分钟粒度，保留详细数据
- ✅ **用途**：实时监控、告警、Dashboard展示
- ✅ **保留时间**：通常保留1-7天

**例子**：
```python
# 实时指标配置
realtime_metrics_config = {
    "interval": "1m",  # 1分钟粒度
    "retention_days": 7,  # 保留7天
    "metrics": [
        "rps",  # 每秒请求数
        "error_rate",  # 错误率
        "p95_latency",  # 95分位延迟
        "p99_latency",  # 99分位延迟
        "success_rate"  # 成功率
    ],
    "use_cases": [
        "real-time_monitoring",  # 实时监控
        "alerting",  # 告警
        "dashboard"  # Dashboard展示
    ]
}
```

**数据示例**：
```
时间戳                RPS    Error Rate  p95 Latency
2025-01-19 10:00:00  1000   0.01%       150ms
2025-01-19 10:01:00  1050   0.02%       160ms
2025-01-19 10:02:00  980    0.01%       140ms
...
```

**关键点**：
- ✅ **高精度**：1分钟粒度，保留详细数据
- ✅ **短期保留**：通常保留1-7天，用于实时监控

---

### 2. 聚合指标（降采样长期保留）

**定义**：**降采样存储，用于历史分析**。

**特点**：
- ✅ **精度**：1小时粒度、1天粒度，降采样存储
- ✅ **用途**：历史分析、趋势分析、报表
- ✅ **保留时间**：通常保留90天-1年

**例子**：
```python
# 聚合指标配置
aggregated_metrics_config = {
    "hourly": {
        "interval": "1h",  # 1小时粒度
        "retention_days": 90,  # 保留90天
        "aggregation": "avg",  # 聚合方法：平均值
        "metrics": [
            "rps",  # 每秒请求数（平均值）
            "error_rate",  # 错误率（平均值）
            "p95_latency",  # 95分位延迟（平均值）
            "p99_latency"  # 99分位延迟（平均值）
        ],
        "use_cases": [
            "historical_analysis",  # 历史分析
            "trend_analysis"  # 趋势分析
        ]
    },
    "daily": {
        "interval": "1d",  # 1天粒度
        "retention_days": 365,  # 保留1年
        "aggregation": "avg",  # 聚合方法：平均值
        "metrics": [
            "rps",
            "error_rate",
            "p95_latency",
            "p99_latency"
        ],
        "use_cases": [
            "long_term_trend",  # 长期趋势
            "reporting"  # 报表
        ]
    }
}
```

**数据示例**：
```
小时粒度（1小时聚合）：
时间戳                RPS    Error Rate  p95 Latency
2025-01-19 10:00:00  1010   0.013%      150ms  # 1小时内60个1分钟数据的平均值
2025-01-19 11:00:00  1050   0.015%      155ms
...

天粒度（1天聚合）：
时间戳                RPS    Error Rate  p95 Latency
2025-01-19 00:00:00  1020   0.014%      152ms  # 1天内24个1小时数据的平均值
2025-01-20 00:00:00  1030   0.016%      154ms
...
```

**关键点**：
- ✅ **降采样**：1小时粒度、1天粒度，降低存储成本
- ✅ **长期保留**：通常保留90天-1年，用于历史分析

---

## 📊 指标聚合方法

### 1. 平均值聚合（Average）

**定义**：**计算时间窗口内所有数据点的平均值**。

**适用场景**：
- ✅ **RPS（每秒请求数）**：平均值代表平均负载
- ✅ **Error Rate（错误率）**：平均值代表平均错误率
- ✅ **Success Rate（成功率）**：平均值代表平均成功率

**例子**：
```python
# 平均值聚合
def aggregate_average(data_points):
    """计算平均值"""
    return sum(data_points) / len(data_points)

# 1小时内60个1分钟数据的平均值
minute_data = [1000, 1050, 980, 1020, ...]  # 60个数据点
hourly_avg = aggregate_average(minute_data)  # 计算平均值
```

---

### 2. 最大值聚合（Max）

**定义**：**计算时间窗口内所有数据点的最大值**。

**适用场景**：
- ✅ **Latency（延迟）**：最大值代表最坏情况
- ✅ **Error Rate（错误率）**：峰值错误率

**例子**：
```python
# 最大值聚合
def aggregate_max(data_points):
    """计算最大值"""
    return max(data_points)

# 1小时内60个1分钟数据的最大值
minute_data = [150, 160, 140, 155, ...]  # 60个数据点
hourly_max = aggregate_max(minute_data)  # 计算最大值
```

---

### 3. 最小值聚合（Min）

**定义**：**计算时间窗口内所有数据点的最小值**。

**适用场景**：
- ✅ **Latency（延迟）**：最小值代表最好情况
- ✅ **Success Rate（成功率）**：最低成功率

**例子**：
```python
# 最小值聚合
def aggregate_min(data_points):
    """计算最小值"""
    return min(data_points)

# 1小时内60个1分钟数据的最小值
minute_data = [150, 160, 140, 155, ...]  # 60个数据点
hourly_min = aggregate_min(minute_data)  # 计算最小值
```

---

### 4. 分位数聚合（Percentile）

**定义**：**计算时间窗口内所有数据点的分位数**。

**适用场景**：
- ✅ **p95 Latency（95分位延迟）**：分位数代表大多数请求的延迟
- ✅ **p99 Latency（99分位延迟）**：分位数代表极端请求的延迟

**例子**：
```python
# 分位数聚合
def aggregate_percentile(data_points, percentile):
    """计算分位数"""
    sorted_data = sorted(data_points)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[index]

# 1小时内60个1分钟数据的p95分位数
minute_data = [150, 160, 140, 155, ...]  # 60个数据点
hourly_p95 = aggregate_percentile(minute_data, 95)  # 计算p95分位数
```

---

## 📊 指标生命周期管理

### 完整生命周期流程

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

---

### 自动降采样流程

**步骤1：收集实时指标**
```python
# 收集1分钟粒度数据
def collect_realtime_metrics():
    """收集实时指标（1分钟粒度）"""
    metrics = {
        "timestamp": datetime.now(),
        "rps": 1000,
        "error_rate": 0.01,
        "p95_latency": 150
    }
    # 存储到实时指标表
    store_realtime_metrics(metrics)
```

**步骤2：7天后降采样为1小时粒度**
```python
# 降采样为1小时粒度
def downsample_to_hourly(realtime_metrics):
    """将实时指标降采样为1小时粒度"""
    # 按小时分组
    hourly_groups = group_by_hour(realtime_metrics)
    
    # 对每个小时的数据进行聚合
    hourly_metrics = []
    for hour, data_points in hourly_groups.items():
        aggregated = {
            "timestamp": hour,
            "rps": aggregate_average([d["rps"] for d in data_points]),
            "error_rate": aggregate_average([d["error_rate"] for d in data_points]),
            "p95_latency": aggregate_percentile([d["p95_latency"] for d in data_points], 95)
        }
        hourly_metrics.append(aggregated)
    
    # 存储到聚合指标表
    store_hourly_metrics(hourly_metrics)
    
    # 删除已降采样的实时指标
    delete_realtime_metrics(realtime_metrics)
```

**步骤3：90天后降采样为1天粒度**
```python
# 降采样为1天粒度
def downsample_to_daily(hourly_metrics):
    """将小时粒度指标降采样为1天粒度"""
    # 按天分组
    daily_groups = group_by_day(hourly_metrics)
    
    # 对每天的数据进行聚合
    daily_metrics = []
    for day, data_points in daily_groups.items():
        aggregated = {
            "timestamp": day,
            "rps": aggregate_average([d["rps"] for d in data_points]),
            "error_rate": aggregate_average([d["error_rate"] for d in data_points]),
            "p95_latency": aggregate_percentile([d["p95_latency"] for d in data_points], 95)
        }
        daily_metrics.append(aggregated)
    
    # 存储到长期指标表
    store_daily_metrics(daily_metrics)
    
    # 删除已降采样的小时粒度指标
    delete_hourly_metrics(hourly_metrics)
```

---

## 💡 实际应用场景（KYC项目）

### Metrics降采样配置

**配置**：
```python
# KYC项目Metrics降采样配置
kyc_metrics_downsampling_config = {
    "realtime": {
        "interval": "1m",
        "retention_days": 7,
        "metrics": [
            "rps",
            "error_rate",
            "p95_latency",
            "p99_latency",
            "success_rate",
            "schema_fail_rate",
            "fallback_rate"
        ]
    },
    "hourly": {
        "interval": "1h",
        "retention_days": 90,
        "aggregation": "avg",
        "metrics": [
            "rps",
            "error_rate",
            "p95_latency",
            "p99_latency"
        ]
    },
    "daily": {
        "interval": "1d",
        "retention_days": 365,
        "aggregation": "avg",
        "metrics": [
            "rps",
            "error_rate",
            "p95_latency",
            "p99_latency"
        ]
    }
}
```

**效果**：
- ✅ **实时监控**：1分钟粒度，满足实时监控需求
- ✅ **历史分析**：1小时/1天粒度，满足历史分析需求
- ✅ **存储成本**：降低90%（历史数据降采样存储）

---

## 💡 成本优化效果

### 优化前（全部1分钟粒度，保留1年）

**假设**：
- ✅ **每天数据点**：24小时 × 60分钟 = 1,440个数据点
- ✅ **每个数据点大小**：1KB
- ✅ **存储成本**：$0.023/GB/月

**存储成本**：
```
每天数据大小 = 1,440 × 1KB = 1.44MB
每年数据大小 = 1.44MB × 365 = 525.6MB
每月存储成本 = 525.6MB × $0.023/GB = $0.012
```

---

### 优化后（分层存储）

**假设**：
- ✅ **实时指标**：1分钟粒度，保留7天
- ✅ **聚合指标**：1小时粒度，保留90天
- ✅ **长期指标**：1天粒度，保留1年

**存储成本**：
```
实时指标（1分钟粒度，7天）：
每天数据点 = 1,440个
7天数据点 = 1,440 × 7 = 10,080个
7天数据大小 = 10,080 × 1KB = 10.08MB

聚合指标（1小时粒度，90天）：
每天数据点 = 24个（1小时1个）
90天数据点 = 24 × 90 = 2,160个
90天数据大小 = 2,160 × 1KB = 2.16MB

长期指标（1天粒度，365天）：
数据点 = 365个（1天1个）
数据大小 = 365 × 1KB = 365KB

总存储 = 10.08MB + 2.16MB + 0.365MB = 12.6MB
每月存储成本 = 12.6MB × $0.023/GB = $0.0003

成本降低 = ($0.012 - $0.0003) / $0.012 = 97.5%
```

**效果**：
- ✅ **存储成本**：降低97.5%（从$0.012降到$0.0003）
- ✅ **实时监控**：1分钟粒度，满足实时监控需求
- ✅ **历史分析**：1小时/1天粒度，满足历史分析需求

---

## 💡 面试话术

### 核心话术

**长期存储Metrics数据，如何优化存储成本？**

**方案**：
1. ✅ **分层存储**：
   - "我们采用分层存储：实时指标（1分钟粒度）保留7天，用于实时监控；聚合指标（1小时/1天粒度）保留90天，用于历史分析。这样既能满足实时监控需求，又能控制存储成本。"

2. ✅ **自动降采样**：
   - "我们使用自动降采样流程：7天后将实时指标降采样为1小时粒度，90天后降采样为1天粒度，自动清理旧数据，将存储成本降低97.5%。"

3. ✅ **聚合方法**：
   - "我们使用平均值聚合和分位数聚合：RPS和Error Rate使用平均值，Latency使用分位数（p95/p99），确保聚合数据仍然有意义。"

---

## 💡 总结

### 核心答案

**长期存储Metrics数据，如何优化存储成本？**

**方案**：
1. ✅ **实时指标**：1分钟粒度，保留7天（实时监控）
2. ✅ **聚合指标**：1小时粒度，保留90天（历史分析）
3. ✅ **长期指标**：1天粒度，保留1年（长期趋势）

**效果**：
- ✅ **存储成本**：降低97.5%（从$0.012降到$0.0003）
- ✅ **实时监控**：1分钟粒度，满足实时监控需求
- ✅ **历史分析**：1小时/1天粒度，满足历史分析需求

### 关键要点

1. **分层存储**：实时指标高精度短期保留，聚合指标降采样长期保留
2. **自动降采样**：7天后降采样为1小时粒度，90天后降采样为1天粒度
3. **聚合方法**：平均值、最大值、最小值、分位数，根据指标类型选择

### 面试话术

- ✅ "我们采用分层存储：实时指标（1分钟粒度）保留7天，用于实时监控；聚合指标（1小时/1天粒度）保留90天，用于历史分析。这样既能满足实时监控需求，又能控制存储成本。"

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1_B1 Dashboard实现方式详解（[KYC_Day02_A1_B1_Dashboard实现方式详解.md](./KYC_Day02_A1_B1_Dashboard实现方式详解.md)） |
| **Related** | Downsampling、Metrics、指标聚合、指标生命周期、成本优化、Dashboard、[KYC_Day02_A1_B3_采样策略详解_Sampling_Strategy.md](./KYC_Day02_A1_B3_采样策略详解_Sampling_Strategy.md) |
