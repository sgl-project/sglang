---
title: HydroSense Cloud Backend – Architecture Overview (High-Level)
date: 2026-03-12
author: Yanda Cheng
---

## 系统目标

多国家/多城市海量传感器与摄像头的 **实时数据接入、端到端预警、异地容灾、安全隔离** 及 **多层 ML 融合预测**。

---

## 核心架构分层

```
Device / Gateway
      ↓  MQTT / HTTPS (TLS)
Cloud Ingress (API Gateway + Auth + Rate Limiting)
      ↓
Message Bus (Kafka)
      ↓                          ↓
Hard Real-time Rules        Near Real-time Aggregation     Async ML Inference
(Edge / Light Service)      (Kafka Streams / Flink)        (Model Serving Layer)
      ↓                          ↓                               ↓
                         Alert Topic  ──→  Alert Service  ──→  SMS / Email / Webhook
                                ↓
                    Storage Layer
          ┌──────────────┬───────────────┬───────────────┐
        TSDB           Object Store    PostgreSQL       Ext. Gov Data
     (秒/分级热数据)   (冷归档/模型)   (元数据/配置)   (气象/水利/海洋)
                                ↓
                    Analytics & ML Pipeline
             Local Model → Regional Model → Fusion Model
                                ↓
                   Post-training Eval → Canary → Auto-rollback
```

---

## 数据频率分层

| 层级 | 数据类型 | 保留策略 |
|------|----------|----------|
| 秒级 | 传感器遥测、设备心跳 | 1 个月 |
| 分钟级 | 滚动聚合、仪表盘指标 | 1 年 |
| 日级+ | KPI、报表、模型训练集 | 长期保存 |
| 视频/图像 | 摄像头快照、事件片段 | 按事件触发保存 |

---

## 关键设计要点

- **幂等接入**：每条消息携带 `device_id + sequence_id + event_time`，流处理层去重 + watermark 处理乱序。
- **多租户隔离**：按 `tenant_id + region` 分区存储，API 行级过滤，审计日志全覆盖。
- **异地容灾**：每 Region 独立堆栈，元数据异步跨 Region 复制，DNS 全局负载均衡故障切换。
- **多层 ML**：Local（站点级）→ Regional（城市/流域）→ Fusion（多模态综合），交叉政府数据做第三方标签验证，显著优于单一数据源竞品。
- **发布安全**：新模型先离线回测，再 canary 灰度，线上指标劣化自动回滚。
