# HydroSense IoT System Design

## 1. IoT 系统架构概览

HydroSense IoT 系统采用**边缘智能 + 云端管理**的混合架构，支持多种传输模式，适应不同部署场景。

```
Sensor Node (STM32) ←→ Transport Layer (BLE/Wired/Offline) 
←→ Gateway/Base Station ←→ Cloud Platform
```

## 2. 端侧 IoT 设计（STM32）

### 2.1 硬件架构

**核心组件**：
- **MCU**：STM32（低功耗系列，如 STM32L4）
- **传感器**：温度、湿度、雨量、风速等（通过 I2C/SPI/ADC 连接）
- **通信模块**：BLE 模块（如 Nordic nRF52）
- **存储**：Flash（数据持久化）+ RAM（运行时缓存）
- **电源管理**：电池 + 低功耗管理电路

**功耗设计**：
- **Active 模式**：~20mA（采样 + 处理）
- **Sleep 模式**：~10μA（深度睡眠）
- **BLE 传输**：~15mA（发送时）

### 2.2 固件架构

**分层设计**：
```
Application Layer (业务逻辑)
    ↓
Middleware Layer (BLE/存储/OTA)
    ↓
HAL Layer (硬件抽象)
    ↓
Hardware (STM32 + Sensors)
```

**核心模块**：
- **Sensor Driver**：传感器驱动（I2C/SPI/ADC）
- **BLE Stack**：BLE 协议栈（广播/GATT）
- **Storage Manager**：Flash 存储管理（分段/索引）
- **OTA Manager**：固件升级管理（下载/校验/安装）
- **Power Manager**：电源管理（Duty Cycle/Sleep）
- **Edge Pipeline**：数据处理管道（滤波/检测/预测）

### 2.3 低功耗策略

#### 2.3.1 Duty Cycling

**策略**：
- **采样周期**：1 分钟（平稳期）→ 1 秒（事件期）
- **处理周期**：采样后立即处理
- **传输周期**：处理完成后立即传输（事件期）或延迟传输（平稳期）
- **Sleep 周期**：空闲时进入深度睡眠

**功耗优化**：
- 采样时唤醒，采样后立即 sleep
- 事件触发时提升频率，事件结束后降频
- BLE 传输完成后立即 sleep

#### 2.3.2 事件触发策略

**触发条件**：
- **阈值触发**：传感器值超过阈值
- **变化率触发**：短时间内变化率过大
- **模式匹配**：匹配预设异常模式

**频率调整**：
- **平稳期**：1 sample/min（省电）
- **事件期**：1 sample/sec（保证数据完整性）

#### 2.3.3 边缘计算换传输

**策略**：
- 端侧做异常抑制、缺失填补、扰动去除
- 减少无效数据上传（节省传输功耗）
- 只上传关键数据（事件数据 + 定期心跳）

### 2.4 数据缓存策略

#### 2.4.1 Ring Buffer（内存）

**用途**：最近 N 条数据的快速访问

**结构**：
```c
typedef struct {
    uint32_t head;      // 写入位置
    uint32_t tail;      // 读取位置
    uint32_t size;      // 缓冲区大小
    Packet_t buffer[1000];  // 数据包数组
} RingBuffer_t;
```

**特点**：
- 循环覆盖（FIFO）
- 快速读写（O(1)）
- 内存占用小

#### 2.4.2 Flash 存储（持久化）

**用途**：长期数据存储（离线模式）

**分段策略**：
- 每段 1 小时数据
- 每段带索引（起止 seq/ts）
- 循环覆盖（最老段被覆盖）

**存储结构**：
```
Flash Layout:
[Bootloader][App][Config][Data Segment 1][Data Segment 2]...[Index]
```

**索引结构**：
```c
typedef struct {
    uint32_t start_seq;
    uint32_t end_seq;
    uint32_t start_ts;
    uint32_t end_ts;
    uint32_t data_offset;
    uint32_t data_size;
    uint32_t crc;
} SegmentIndex_t;
```

### 2.5 BLE 通信设计

#### 2.5.1 广播模式

**用途**：超低功耗场景，无需配对

**广播包结构**：
```
[Preamble:1B][Access Address:4B][Header:2B]
[Payload:34B][CRC:3B]
```

**Payload 结构**：
```
[device_id:4B][seq:4B][ts:4B][sensor_data:20B][crc:2B]
总大小：34 bytes
```

**特点**：
- 超低功耗（~10μA，广播时 ~15mA）
- 无需配对
- 包小，易丢包（需要应用层容错）

**应用层容错**：
- 序号检测（seq）
- 时间戳校验（ts）
- CRC 校验
- 网关侧去重

#### 2.5.2 GATT 连接模式

**用途**：需要双向通信的场景（配置下发、OTA）

**服务结构**：
```
Service: HydroSense Service (UUID: 0x1800)
├── Characteristic: Telemetry (Notify)
│   └── UUID: 0x2A00
├── Characteristic: Configuration (Read/Write)
│   └── UUID: 0x2A01
├── Characteristic: OTA Control (Write)
│   └── UUID: 0x2A02
└── Characteristic: Status (Read)
    └── UUID: 0x2A03
```

**通信流程**：
1. Gateway 扫描设备
2. 建立 GATT 连接
3. 订阅 Telemetry Notify
4. 设备发送数据（Notify）
5. Gateway 发送 ACK（Write）
6. 设备确认收到 ACK

**可靠性机制**：
- 应用层 ACK（关键事件包必须 ACK）
- 重传机制（最多 N 次）
- 连接断开检测（自动重连）

### 2.6 有线通信设计

**用途**：维护/升级/救砖

**接口**：UART/USB/RS485

**协议**：
```
[Header:2B][Command:1B][Length:2B][Payload:N bytes][CRC:2B]
```

**命令集**：
- `HELLO`：设备身份/版本/容量
- `GET_WATERMARK`：获取同步 watermark
- `FETCH_RANGE`：拉取数据范围
- `ACK`：确认同步
- `OTA_START`：开始 OTA
- `OTA_DATA`：OTA 数据块
- `OTA_END`：OTA 结束

**特点**：
- 高可靠性（物理连接）
- 双向通信（可下发命令）
- 支持批量操作（多设备）

## 3. Gateway/Base Station 设计

### 3.1 Gateway 架构（BLE → Cloud）

**核心功能**：
- BLE 设备管理（扫描/连接/调度）
- 数据去重与补传协调
- 批量上云（聚合/压缩）
- 本地缓存（store-and-forward）
- 配置下发（可选）

**硬件**：
- BLE 模块（扫描/连接）
- 网络模块（4G/WiFi/以太网）
- 存储（本地缓存）
- 电源（电池/市电）

**软件架构**：
```
Application Layer
├── Device Manager (BLE 设备管理)
├── Data Manager (数据去重/补传)
├── Upload Manager (批量上云)
├── Cache Manager (本地缓存)
└── Config Manager (配置下发)
    ↓
BLE Stack / Network Stack
    ↓
Hardware (BLE Module / Network Module)
```

### 3.2 Gateway 设备管理

#### 3.2.1 扫描策略

**广播模式**：
- 持续扫描（低功耗）
- 解析广播包
- 提取 device_id + seq + data
- 去重（(device_id, seq)）

**GATT 连接模式**：
- 按需连接（事件触发）
- 连接调度（避免同时连接太多设备）
- 连接池管理（复用连接）

#### 3.2.2 去重与补传

**去重机制**：
- 维护每个设备的 seq 窗口（如最近 1000 条）
- 检测缺失 seq（gap detection）
- 请求补传（GATT 模式）

**补传流程**：
1. 检测缺失 seq（如 seq 100, 102, 103 → 缺失 101）
2. 发送 `FETCH_RANGE(101, 101)` 请求
3. 设备返回缺失数据
4. 更新 seq 窗口

### 3.3 Gateway 批量上云

**批量策略**：
- **时间窗口**：每 10 秒批量上传一次
- **数量窗口**：每 100 条批量上传一次
- **事件触发**：关键事件立即上传

**批量包结构**：
```json
{
  "gateway_id": "GW001",
  "batch_id": "BATCH123",
  "timestamp": 1704067200,
  "devices": [
    {
      "device_id": "DEV001",
      "packets": [
        {"seq": 100, "data": {...}},
        {"seq": 101, "data": {...}}
      ]
    }
  ]
}
```

**压缩**：批量包压缩（gzip）后上传

### 3.4 Base Station 架构（有线汇聚）

**核心功能**：
- 多设备并发采集（多路串口/USB/RS485）
- 统一设备管理（站点级 asset inventory）
- 批量同步（增量拉取、断点续传）
- 统一上传（站点级数据上云）
- 现场 OTA（批量升级）

**硬件**：
- 多路串口/USB/RS485 接口
- 网络模块（4G/WiFi/以太网）
- 存储（本地数据库）
- 电源（市电）

**软件架构**：
```
Application Layer
├── Device Manager (多设备管理)
├── Sync Manager (批量同步)
├── Upload Manager (统一上传)
├── OTA Manager (现场 OTA)
└── Site Manager (站点管理)
    ↓
Serial/USB Stack / Network Stack
    ↓
Hardware (Serial/USB / Network Module)
```

### 3.5 Base Station 批量同步

**同步流程**：
1. **设备发现**：扫描连接的设备
2. **HELLO 交换**：获取设备信息（版本/容量）
3. **Watermark 同步**：获取每个设备的 watermark
4. **增量拉取**：按设备拉取增量数据
5. **断点续传**：支持中断后继续
6. **批量上传**：统一上传到云端

**并发策略**：
- 多线程/多进程并发采集
- 每个设备独立线程
- 共享上传队列

## 4. 云端 IoT 平台设计

### 4.1 IoT Gateway（云端接入层）

**核心功能**：
- 协议适配（MQTT/HTTP/CoAP）
- 设备认证（证书/密钥）
- 路由（按租户/站点/设备）
- 限流（按租户配额）
- 监控（QPS/延迟/错误率）

**架构**：
```
IoT Gateway
├── Protocol Adapter (MQTT/HTTP/CoAP)
├── AuthN/AuthZ (设备认证/授权)
├── Router (路由/分区)
├── Rate Limiter (限流)
└── Monitor (监控)
    ↓
Message Queue (Kafka/RabbitMQ)
```

### 4.2 设备认证

**认证方式**：
- **证书认证**：X.509 证书（TLS）
- **密钥认证**：预共享密钥（PSK）
- **Token 认证**：JWT Token（HTTP）

**设备身份**：
- 每设备唯一 device_id
- 每设备唯一密钥/证书
- 密钥轮换（定期更新）

**认证流程**：
1. 设备连接 IoT Gateway
2. Gateway 验证设备证书/密钥
3. Gateway 检查设备状态（是否禁用）
4. Gateway 分配 topic/route
5. 建立连接

### 4.3 路由与分区

**路由策略**：
- **按租户路由**：`tenant/{tenant_id}/devices/{device_id}`
- **按站点路由**：`tenant/{tenant_id}/sites/{site_id}/devices/{device_id}`
- **按设备类型路由**：`tenant/{tenant_id}/device_types/{type}/devices/{device_id}`

**分区策略**：
- 消息队列按 tenant_id 分区
- 子分区按 site_id（可选）
- 哈希分区按 device_id（负载均衡）

### 4.4 限流策略

**限流维度**：
- **按租户限流**：每个租户的 QPS 限制
- **按设备限流**：每个设备的 QPS 限制
- **按站点限流**：每个站点的 QPS 限制

**限流算法**：
- Token Bucket（令牌桶）
- Sliding Window（滑动窗口）

**限流处理**：
- 超限请求：返回 429 Too Many Requests
- 限流告警：超限时发送告警

## 5. 设备管理（Device Management）

### 5.1 设备注册

**注册流程**：
1. 管理员创建设备（device_id、密钥/证书）
2. 设备信息写入元数据库
3. 设备密钥/证书下发（通过安全渠道）
4. 设备首次连接时验证身份
5. 设备状态更新为"已激活"

**设备信息**：
```json
{
  "device_id": "DEV001",
  "tenant_id": "TENANT001",
  "site_id": "SITE001",
  "device_type": "rainfall_sensor",
  "firmware_version": "v1.2.3",
  "model_version": "v1.0.0",
  "config_version": "v1.0.0",
  "status": "active",
  "created_at": "2024-01-01T00:00:00Z",
  "last_seen": "2024-01-15T12:00:00Z"
}
```

### 5.2 设备分组

**分组策略**：
- **按站点分组**：同一站点的设备
- **按设备类型分组**：同一类型的设备
- **按地理位置分组**：同一地理区域的设备
- **自定义分组**：用户自定义分组

**分组用途**：
- 批量操作（配置下发、OTA）
- 数据分析（站点级聚合）
- 告警规则（站点级告警）

### 5.3 设备 Shadow（设备影子）

**概念**：设备在云端的虚拟表示，包含设备的期望状态和报告状态。

**Shadow 结构**：
```json
{
  "device_id": "DEV001",
  "reported": {
    "firmware_version": "v1.2.3",
    "config": {
      "sampling_rate": 60,
      "threshold": 30.0
    },
    "status": {
      "battery": 85,
      "rssi": -65,
      "last_seen": "2024-01-15T12:00:00Z"
    }
  },
  "desired": {
    "firmware_version": "v1.2.4",
    "config": {
      "sampling_rate": 30,
      "threshold": 25.0
    }
  },
  "delta": {
    "firmware_version": "v1.2.4",
    "config": {
      "sampling_rate": 30
    }
  }
}
```

**Shadow 同步**：
- 设备上报 → 更新 reported
- 云端下发 → 更新 desired
- 计算 delta → 下发到设备
- 设备确认 → 更新 reported

### 5.4 远程诊断

**诊断功能**：
- **设备状态查询**：在线状态、健康度、最后在线时间
- **数据质量分析**：缺失率、重复率、延迟统计
- **日志查询**：设备端日志、云端日志
- **性能分析**：采样频率、传输成功率、电池消耗

**诊断接口**：
```
GET /api/v1/devices/{device_id}/status
GET /api/v1/devices/{device_id}/quality
GET /api/v1/devices/{device_id}/logs
GET /api/v1/devices/{device_id}/performance
```

## 6. OTA 系统设计

### 6.1 OTA 架构

**核心组件**：
- **Firmware Repository**：固件包存储
- **Campaign Manager**：发布活动管理
- **OTA Agent**：设备端 OTA 代理
- **OTA Gateway**：OTA 传输网关

**OTA 流程**：
```
Firmware Upload → Campaign Create → Device Selection → 
OTA Push → Device Download → Device Verify → Device Install → 
Device Report → Campaign Monitor
```

### 6.2 固件包管理

**固件包结构**：
```
Firmware Package:
[Header][Metadata][Firmware Binary][Signature][Footer]
```

**Metadata**：
```json
{
  "version": "v1.2.4",
  "size": 1024000,
  "hash": "sha256:...",
  "signature": "rsa:...",
  "device_type": "rainfall_sensor",
  "min_version": "v1.0.0",
  "release_notes": "..."
}
```

**存储**：对象存储（S3/OSS）

### 6.3 Campaign 管理

**Campaign 结构**：
```json
{
  "campaign_id": "CAMPAIGN001",
  "firmware_version": "v1.2.4",
  "target_devices": {
    "type": "site",
    "value": ["SITE001", "SITE002"]
  },
  "rollout_strategy": {
    "type": "gradual",
    "phases": [
      {"percentage": 1, "duration": "1h"},
      {"percentage": 10, "duration": "2h"},
      {"percentage": 50, "duration": "4h"},
      {"percentage": 100, "duration": "8h"}
    ]
  },
  "health_checks": {
    "crash_rate_threshold": 0.01,
    "reboot_threshold": 3,
    "data_gap_threshold": 0.1
  },
  "rollback_conditions": {
    "max_failures": 10,
    "critical_metrics": ["crash_rate", "data_gap"]
  },
  "status": "running",
  "created_at": "2024-01-15T00:00:00Z"
}
```

### 6.4 灰度发布策略

**策略类型**：
- **按站点灰度**：1% → 10% → 50% → 100%
- **按批次灰度**：第一批 → 第二批 → 第三批 → 全部
- **按设备类型灰度**：A 类型 → B 类型 → C 类型

**健康门禁**：
- 升级后 crash rate 超过阈值 → 暂停
- 升级后 reboot 次数超过阈值 → 暂停
- 升级后 data gap 超过阈值 → 暂停

**自动回滚**：
- N 次失败 → 自动回滚
- 关键指标触发阈值 → 自动回滚

### 6.5 OTA 传输

**传输方式**：
- **BLE OTA**：通过 BLE GATT 传输（小包，分块传输）
- **有线 OTA**：通过 UART/USB 传输（大包，快速传输）
- **云端 OTA**：通过 MQTT/HTTP 传输（需要网络）

**分块传输**：
- 固件包分块（如每块 512 bytes）
- 逐块传输
- 每块校验（CRC）
- 全部传输完成后整体校验（hash）

**断点续传**：
- 记录已传输块号
- 中断后从断点继续
- 支持重传失败块

## 7. 安全设计

### 7.1 设备身份安全

**密钥管理**：
- 每设备唯一密钥/证书
- 密钥存储在安全区域（如 TEE/SE）
- 密钥轮换（定期更新）

**证书管理**：
- X.509 证书（TLS）
- 证书链验证
- 证书撤销列表（CRL）

### 7.2 传输安全

**加密**：
- TLS 1.3（MQTT/HTTP）
- BLE 加密（配对后加密传输）

**完整性**：
- CRC 校验（应用层）
- 签名校验（可选）

### 7.3 权限控制

**RBAC**：
- 租户隔离（不同租户数据隔离）
- 角色权限（管理员/操作员/查看者）
- API 权限（按功能权限）

**API 认证**：
- API Key（服务端调用）
- OAuth 2.0（用户调用）

### 7.4 审计日志

**审计内容**：
- 设备注册/删除
- 配置下发
- OTA 操作
- 数据导出
- 权限变更

**审计存储**：
- 审计日志存储（不可篡改）
- 日志查询接口
- 日志告警（异常操作）

## 8. 监控与可观测性

### 8.1 设备监控

**监控指标**：
- 在线设备数
- 离线设备数
- 设备健康度
- 设备数据质量

**告警规则**：
- 设备离线超过 N 分钟 → 告警
- 设备健康度低于阈值 → 告警
- 设备数据质量持续恶化 → 告警

### 8.2 系统监控

**监控指标**：
- IoT Gateway QPS
- 消息队列延迟
- 存储延迟
- 查询延迟

**SLO**：
- 数据延迟 < 5 秒（P99）
- 缺失率 < 0.1%
- 告警响应时间 < 30 秒

### 8.3 链路追踪

**追踪点**：
- 端侧采样
- 传输
- 云端接收
- 处理
- 存储
- 查询

**追踪 ID**：贯穿全链路的 trace_id

**用途**：
- 问题排查
- 性能分析
- 数据追溯
