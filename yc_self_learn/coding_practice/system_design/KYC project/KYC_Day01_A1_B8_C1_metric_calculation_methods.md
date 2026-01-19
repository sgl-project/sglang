# A1_B8_C1：指标计算方法：流失率计算与实时 vs 批处理

---
doc_type: glossary
layer: L3
scope_in:  流失率的计算方法、分析指标的计算方式（实时 vs 批处理）、指标计算的时机和频率
scope_out:  具体指标计算代码实现（见 howto）；实时计算系统架构（见 ADR）；批处理系统设计（见 ADR）
inputs:   (设计) 指标定义、计算频率、实时性要求；(运行时) 用户行为数据、系统指标数据
outputs:  流失率计算结果、指标计算方式选择、实时 vs 批处理的权衡
entrypoints: [ 核心问题, 流失率计算 ]
children: []
related: [ KYC_Day01_A1_B8_ab_testing_validation.md（A/B Test 验证）, KYC_Day01_A2_指标计算脚本示例.md（指标计算脚本） ]
---

## 核心问题

**问题**：
1. 流失率是怎么算的？
2. 这种分析指标是实时计算的吗？

**答案**：
1. **流失率计算**：流失用户数 / 总用户数，通常基于时间窗口（如 7 天内未返回）
2. **计算方式**：**不是所有指标都实时计算**，而是根据指标类型和业务需求选择实时计算或批处理

---

## 流失率计算方法

### 1. 基本定义

**流失率（Churn Rate）**：在特定时间窗口内，流失的用户数 / 总用户数

**公式**：
```
流失率 = 流失用户数 / 总用户数 × 100%
```

---

### 2. 流失用户的定义

**定义方式**（根据业务场景选择）：

| 定义方式 | 说明 | 适用场景 | 示例 |
|---------|------|---------|------|
| **时间窗口未返回** | 在时间窗口内（如 7 天）未返回的用户 | 活跃度分析 | 7 天内未返回的用户 = 流失用户 |
| **明确取消/删除** | 用户明确取消订阅或删除账户 | 订阅服务 | 用户点击"取消订阅" = 流失用户 |
| **长期不活跃** | 超过阈值时间（如 30 天）未活跃 | 长期留存分析 | 30 天内未活跃的用户 = 流失用户 |

**KYC 示例**（A/B Test 中的定义）：
```
流失用户定义：7 天内未返回的用户

计算逻辑：
  1. 记录每个用户的最后访问时间
  2. 如果当前时间 - 最后访问时间 > 7 天 → 标记为流失用户
  3. 流失率 = 流失用户数 / 总用户数 × 100%

示例：
  - 实验组 A：10,000 用户，500 用户 7 天内未返回 → 流失率 = 5%
  - 实验组 B：10,000 用户，1,200 用户 7 天内未返回 → 流失率 = 12%
```

---

### 3. 流失率计算的数据源

**数据源**：

| 数据源 | 说明 | 示例 |
|-------|------|------|
| **用户行为日志** | 记录用户访问、操作等行为 | `user_activity_logs`：user_id, timestamp, action |
| **用户表** | 存储用户基本信息和状态 | `users`：user_id, created_at, last_active_at, status |
| **会话数据** | 记录用户会话信息 | `sessions`：user_id, session_id, start_time, end_time |

**KYC 示例**：
```sql
-- 计算 7 天内未返回的用户数
SELECT 
  COUNT(DISTINCT user_id) as churned_users,
  COUNT(DISTINCT user_id) * 100.0 / (SELECT COUNT(*) FROM users) as churn_rate
FROM users
WHERE last_active_at < NOW() - INTERVAL '7 days'
  AND status = 'active';
```

---

### 4. 流失率计算的时机

**计算时机**：

| 时机 | 说明 | 适用场景 |
|------|------|---------|
| **实时计算** | 每次用户行为发生时计算 | 实时监控、告警 |
| **批处理（每日）** | 每天计算一次 | 日常报表、趋势分析 |
| **批处理（每周）** | 每周计算一次 | 周报、长期趋势分析 |
| **按需计算** | 需要时计算 | A/B Test 分析、临时分析 |

**KYC 示例**：
```
A/B Test 中的流失率计算：
  - 时机：实验结束后，按需计算
  - 方法：批处理（一次性计算）
  - 数据源：用户行为日志 + 用户表
  - 结果：实验组 A 流失率 5%，实验组 B 流失率 12%
```

---

## 分析指标的计算方式：实时 vs 批处理

### 1. 指标分类

**根据实时性要求分类**：

| 指标类型 | 实时性要求 | 计算方式 | 示例 |
|---------|----------|---------|------|
| **实时指标** | 需要立即知道 | 实时计算 | Success Rate、Error Rate、p95 延迟 |
| **准实时指标** | 几分钟内知道即可 | 准实时计算（如 5 分钟） | 用户活跃度、API 调用量 |
| **批处理指标** | 几小时或几天内知道即可 | 批处理（如每日/每周） | 流失率、转化率、用户留存率 |

---

### 2. 实时计算（Real-Time Calculation）

**定义**：每次事件发生时立即计算指标

**适用场景**：
- 需要立即响应的指标（如告警）
- 需要实时监控的系统指标（如 Success Rate、Error Rate）

**实现方式**：
```
事件流处理（Stream Processing）：
  1. 用户行为事件 → Kafka / Kinesis
  2. 流处理引擎（Flink / Spark Streaming）实时计算
  3. 结果写入时序数据库（InfluxDB / TimescaleDB）
  4. 监控系统（Grafana / Datadog）实时展示
```

**KYC 示例**：
```
实时指标：Success Rate、Error Rate、p95 延迟

计算流程：
  1. 每个请求完成后，立即发送事件到 Kafka
  2. Flink 实时计算 Success Rate、Error Rate
  3. 结果写入 InfluxDB
  4. Grafana 实时展示，如果超过阈值 → 触发告警

延迟：< 1 秒
```

---

### 3. 批处理（Batch Processing）

**定义**：定期（如每日/每周）批量计算指标

**适用场景**：
- 不需要立即知道的指标（如流失率、转化率）
- 需要聚合大量数据的指标（如用户留存率）
- 需要历史对比的指标（如月度趋势）

**实现方式**：
```
批处理流程（Batch Processing）：
  1. 数据源：用户行为日志、数据库
  2. 批处理引擎（Spark / Hadoop）定期计算
  3. 结果写入数据仓库（Snowflake / BigQuery）
  4. 报表系统（Tableau / Looker）展示
```

**KYC 示例**：
```
批处理指标：流失率、转化率、用户留存率

计算流程：
  1. 每天凌晨 2 点，Spark 读取昨天的用户行为日志
  2. 计算流失率、转化率等指标
  3. 结果写入 Snowflake
  4. 报表系统展示，用于 A/B Test 分析

延迟：< 24 小时
```

---

### 4. 实时 vs 批处理的权衡

| 维度 | 实时计算 | 批处理 |
|------|---------|--------|
| **延迟** | < 1 秒 | 几小时到几天 |
| **成本** | 高（需要流处理基础设施） | 低（批处理基础设施） |
| **复杂度** | 高（需要处理乱序、重复等） | 低（顺序处理） |
| **准确性** | 可能不准确（乱序、延迟） | 准确（完整数据） |
| **适用场景** | 告警、实时监控 | 报表、分析、A/B Test |

**KYC 示例**：
```
指标选择：

实时计算：
  - Success Rate（需要立即告警）
  - Error Rate（需要立即告警）
  - p95 延迟（需要实时监控）

批处理：
  - 流失率（A/B Test 分析，不需要实时）
  - 转化率（报表分析，不需要实时）
  - 用户留存率（长期趋势分析，不需要实时）
```

---

### 5. 混合方案（Hybrid Approach）

**定义**：根据指标类型和业务需求，组合使用实时计算和批处理

**KYC 示例**：
```
混合方案：

实时计算（用于告警）：
  - Success Rate < 98% → 实时告警
  - Error Rate > 2% → 实时告警
  - p95 > 15s → 实时告警

批处理（用于分析）：
  - 流失率 → 每日批处理，用于 A/B Test 分析
  - 转化率 → 每日批处理，用于报表分析
  - 用户留存率 → 每周批处理，用于长期趋势分析

准实时（用于监控）：
  - 用户活跃度 → 5 分钟准实时，用于监控
  - API 调用量 → 5 分钟准实时，用于监控
```

---

## 流失率在 A/B Test 中的计算

### 1. A/B Test 中的流失率计算

**计算时机**：**实验结束后，按需计算**（批处理）

**计算流程**：
```
1. 实验开始：记录每个用户分配到哪个实验组（A 或 B）
   ↓
2. 实验进行：记录每个用户的行为（访问时间、操作等）
   ↓
3. 实验结束：计算每个实验组的流失率
   ↓
4. 统计分析：计算统计显著性（p-value）
```

**KYC 示例**：
```
实验设计：
  - 实验组 A（p95=10s）：10,000 用户
  - 实验组 B（p95=20s）：10,000 用户
  - 实验时长：2 周

计算流失率：
  1. 查询用户表，找出每个用户的最后访问时间
  2. 如果当前时间 - 最后访问时间 > 7 天 → 标记为流失用户
  3. 计算每个实验组的流失率：
     - 实验组 A：500 用户流失 → 流失率 = 5%
     - 实验组 B：1,200 用户流失 → 流失率 = 12%
  4. 统计分析：p-value = 0.001 < 0.05 ✅

结论：实验组 B 的流失率显著高于实验组 A（+7%）
```

---

### 2. 流失率计算的 SQL 示例

```sql
-- 计算 A/B Test 中的流失率
WITH user_last_activity AS (
  SELECT 
    user_id,
    experiment_group,  -- 'A' or 'B'
    MAX(timestamp) as last_active_at
  FROM user_activity_logs
  WHERE experiment_id = 'exp_001'
    AND timestamp >= '2025-01-01'
    AND timestamp <= '2025-01-15'
  GROUP BY user_id, experiment_group
),
churned_users AS (
  SELECT 
    experiment_group,
    COUNT(DISTINCT user_id) as churned_count
  FROM user_last_activity
  WHERE last_active_at < NOW() - INTERVAL '7 days'
  GROUP BY experiment_group
),
total_users AS (
  SELECT 
    experiment_group,
    COUNT(DISTINCT user_id) as total_count
  FROM user_last_activity
  GROUP BY experiment_group
)
SELECT 
  c.experiment_group,
  c.churned_count,
  t.total_count,
  c.churned_count * 100.0 / t.total_count as churn_rate
FROM churned_users c
JOIN total_users t ON c.experiment_group = t.experiment_group;
```

**结果**：
```
experiment_group | churned_count | total_count | churn_rate
-----------------|---------------|-------------|-----------
A                | 500           | 10,000      | 5.0%
B                | 1,200         | 10,000      | 12.0%
```

---

## 总结

### 核心答案

1. **流失率计算**：
   - 公式：流失用户数 / 总用户数 × 100%
   - 定义：通常基于时间窗口（如 7 天内未返回）
   - 时机：批处理（实验结束后按需计算）

2. **分析指标的计算方式**：
   - **不是所有指标都实时计算**
   - **实时计算**：用于告警、实时监控（如 Success Rate、Error Rate）
   - **批处理**：用于报表、分析、A/B Test（如流失率、转化率）
   - **混合方案**：根据指标类型和业务需求组合使用

### 设计原则

1. **根据实时性要求选择计算方式**：告警用实时，分析用批处理
2. **根据成本选择计算方式**：实时计算成本高，批处理成本低
3. **根据准确性要求选择计算方式**：实时计算可能不准确，批处理准确

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1_B8 A/B Test 验证设计思路（KYC_Day01_A1_B8_ab_testing_validation.md） |
| **Related** | [A2：指标计算脚本示例](./KYC_Day01_A2_指标计算脚本示例.md） |
