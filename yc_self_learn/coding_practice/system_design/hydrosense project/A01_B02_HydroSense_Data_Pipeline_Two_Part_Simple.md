# A01_B02: HydroSense Data Pipeline (Two-Part Simple Version)

**Author**: Yanda Cheng  
**Project**: HydroSense IoT Platform  
**Purpose**: Interview-friendly two-part explanation (Field-side + Cloud-side)

---

## Overview

**15-second shortest version**:

Field-side performs edge processing + caching, data returns via BLE gateway/offline export/centralized base station; hardware terminals (RTU) communicate via Modbus RTU/TCP. Cloud-side unified ingestion performs auth, deduplication, QC, and layered storage, ultimately providing dashboard, alerting, device & version management, and supports integration with customer platforms or hosted deployment.

---

## Part 1: Field-Side (Hardware + Transport)

**Target**: Low-power, anti-interference, long-term operation in field; supports multiple backhaul modes.

**Duration**: 30-60 seconds

---

### Sensor/MCU

After sampling, perform basic processing on edge:
- **Trigger**: Event detection (threshold/rate/pattern)
- **Disturbance removal**: Anti-noise filtering
- **Anomaly reduction**: Suppress outliers
- **Missing value imputation**: Fill gaps
- **Lightweight CNN prediction**: Optional edge ML

Write data to local cache by `ts + seq`:
- **Ring Buffer** (memory): Recent 1000 records, fast access
- **Flash** (persistent): Append by seq, segmented storage (daily/weekly chunks)
- **Watermark**: Track max_seq for incremental sync

---

### Three Backhaul Modes (Choose by field conditions)

#### Mode A: Real-time / Near-real-time
```
BLE → Gateway/Base Station → Cloud
```
- **Latency**: Seconds
- **Power**: Low-power BLE
- **Use case**: Networked environment, real-time monitoring

#### Mode B: Offline
```
Device/Collector local storage → Wired export every few months
```
- **Latency**: Batch export every few months
- **Reliability**: Wired connection, physical determinism
- **Use case**: Extreme low-power, no network, long maintenance cycles

#### Mode C: Centralized
```
Multiple devices wired → Same base station → Unified management & upload
```
- **Efficiency**: Batch operations, site-level unified management
- **Reliability**: Wired connection, physical determinism
- **Use case**: Centralized deployment, unified power supply, easy ops

---

### Industrial Protocol Support

Hardware terminals (**RTU** - Remote Terminal Unit) communicate via **Modbus RTU + Modbus TCP**:
- RTU devices connect via Modbus RTU (serial) or Modbus TCP (Ethernet)
- Field aggregation endpoints read slave data as registers
- Translate Modbus registers to unified Telemetry Record
- Same data model regardless of protocol

---

### Part 1 Closing Statement

**Field-side ensures data is collectable, storable, exportable, and retransmittable.**

---

## Part 2: Cloud-Side (Ingestion + Storage + Product)

**Target**: Unified ingestion of all modes, quality control, analytics, visualization, and operations.

**Duration**: 30-60 seconds

---

### Ingest

**Gateway/BaseStation/Offline Export → Auth + Schema Check + Dedup (device_id+seq) → Queue**

Regardless of Mode A/B/C, all data enters the same **Ingest** (API/MQTT):
- **Auth**: Device/base station identity verification
- **Schema Check**: Fields, types, ts sanity, crc
- **Dedup**: `(device_id, seq)` as global unique identifier (idempotent)
- **Queue**: Write to message queue/stream (decouple peaks)

---

### Process + Store

**QC/Normalize/Tag Mapping → Raw Archive + Time-Series DB + Metadata DB**

**Processing**:
- **QC**: Missing detection, out-of-order tolerance, anomaly annotation
- **Normalize**: Data normalization and standardization
- **Tag Mapping**: Map Modbus registers to unified Telemetry fields

**Storage**:
1. **Raw Archive** (Object Storage): Traceability and replay validation
2. **Time-Series DB**: Clean/Feature data for dashboard queries and alerting
3. **Metadata DB**: Device/site/tenant, versions (fw/config/model), register mappings, alert rules

---

### Serve + Ops

**Dashboards/Alerts/APIs + Fleet & Versioning (canary/rollback) + Observability**

**Serving**:
- **Dashboards**: Query by site/device time range (supports downsample)
- **Alerts**: Offline, abnormal rainfall, long-term missing, drift → notifications
- **APIs**: Data export, device management, integration endpoints

**Fleet & Versioning**:
- **Fleet Management**: Device status, last seen, data gaps, firmware version distribution
- **Versioning**: OTA campaigns with canary deployment and rollback support

**Observability**:
- Metrics, logs, traces for system monitoring
- Performance dashboards and alerting

**Deployment Options**:
- **Customer Platform Integration**: API-based integration
- **Hosted Deployment**: Managed service (e.g., Alibaba Cloud)

---

### Part 2 Closing Statement

**Cloud-side "unifies" multi-mode data and productizes operations and business value.**

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Part 1: Field-Side                          │
│                    Hardware + Transport                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Sensor/MCU: Sample → Edge Processing → Local Cache             │
│ • Trigger / Disturbance removal / Anomaly reduction /          │
│   Missing imputation / CNN predict                             │
│ • Cache by ts+seq (Ring Buffer + Flash)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Mode A      │ │  Mode B      │ │  Mode C      │
│  BLE →       │ │  Offline →   │ │  Wired →     │
│  Gateway     │ │  Wired Export│ │  Base Station│
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └───────────────┼────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Part 2: Cloud-Side                         │
│                    Ingestion + Storage + Product                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Ingest: Gateway/BaseStation/Offline Export →                  │
│ Auth + Schema Check + Dedup (device_id+seq) → Queue            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Process + Store: QC/Normalize/Tag Mapping →                  │
│ Raw Archive + Time-Series DB + Metadata DB                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Serve + Ops: Dashboards/Alerts/APIs +                         │
│ Fleet & Versioning (canary/rollback) + Observability           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Resume Bullet Points (English, ~30 words each)

### Bullet 1: Field-Side
**Designed field-side edge processing pipeline: STM32 performs sensor sampling, filtering, anomaly reduction, missing value imputation, and lightweight CNN prediction; supports three backhaul modes (BLE real-time, offline batch, wired aggregation) with unified Telemetry Record schema; hardware terminals (RTU) communicate via Modbus RTU/TCP industrial protocol.**

### Bullet 2: Cloud-Side
**Built unified cloud ingestion and processing pipeline: Multi-mode data enters single ingest layer (API/MQTT) with authentication, schema validation, and idempotent deduplication; stream QC performs quality annotation and feature derivation; dual-write to time-series DB (clean/feature), object storage (raw archive), and metadata DB; provides dashboard, alerting, device management, and supports customer platform integration or hosted deployment.**

### Bullet 3: System Integration
**Implemented end-to-end IoT platform supporting field-to-cloud data flow: Field-side ensures data collectability, storability, exportability, and retransmittability; cloud-side unifies multi-mode data and productizes operations and business value; supports real-time monitoring, batch reporting, and model replay validation.**

---

## Interview Scripts

### 30-60 Second Version (Part 1)
"Field-side targets low-power, anti-interference, long-term operation. After sampling, STM32 performs edge processing: trigger detection, disturbance removal, anomaly reduction, missing value imputation, and optional lightweight CNN prediction. Data is cached locally by ts+seq in ring buffer and flash. We support three backhaul modes: Mode A uses BLE to gateway for real-time; Mode B stores offline and exports via wired connection every few months; Mode C uses wired aggregation to base station for centralized management. Hardware terminals (RTU) communicate via Modbus RTU and TCP. Field-side ensures data is collectable, storable, exportable, and retransmittable."

### 30-60 Second Version (Part 2)
"Cloud-side unifies ingestion of all modes. Regardless of A/B/C, data enters the same ingest layer via API or MQTT. We perform authentication, schema validation, and idempotent deduplication using device_id and seq. Then stream QC handles missing detection, out-of-order tolerance, quality annotation, and feature derivation. We dual-write to three storage layers: raw archive to object storage for traceability, clean and feature data to time-series DB for dashboard and alerting, and metadata to relational DB for device and version management. Finally, we provide dashboard visualization, alerting, device management, OTA campaigns, and export APIs. This supports both customer platform integration and hosted deployment. Cloud-side unifies multi-mode data and productizes operations and business value."

### 15-Second Shortest Version
"Field-side performs edge processing plus caching, data returns via BLE gateway, offline export, or centralized base station; hardware terminals (RTU) communicate via Modbus RTU/TCP. Cloud-side unified ingestion performs auth, deduplication, QC, and layered storage, ultimately providing dashboard, alerting, device and version management, and supports integration with customer platforms or hosted deployment."

---

**Document Version**: v1.0  
**Last Updated**: 2025-01
