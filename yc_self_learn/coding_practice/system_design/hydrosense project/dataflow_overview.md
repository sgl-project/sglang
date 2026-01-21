# HydroSense Data Flow Overview

## 1. 数据流总览

HydroSense 数据流采用**端到端**设计，从传感器采样到最终用户查询，形成完整的数据生命周期。

```
Sensor Sampling → Edge Processing → Transport → Cloud Ingress 
→ Stream Processing → Storage → Analytics → Serving → User
```

## 2. 端侧数据流

### 2.1 采样阶段

**输入**：传感器硬件信号（ADC 值）

**处理**：
- ADC 采样（如 10Hz）
- 硬件滤波（去噪）

**输出**：原始采样值数组

**数据量**：每个传感器 ~100 bytes/sample

### 2.2 预处理阶段

**输入**：原始采样值

**处理流程**：
```
原始值 → 去抖 → 滤波 → 温漂补偿 → 单位转换
```

**输出**：清洗后的传感器数据

**数据量**：每个传感器 ~50 bytes/sample（压缩后）

### 2.3 智能处理阶段

**输入**：清洗后的传感器数据

**处理流程**：
```
清洗数据 → Trigger → Abnormal Reduce → Missing Fitting 
→ 扰动去除 → CNN Predict → 打包
```

**输出**：完整数据包（含 sensor data + flags + pred + metadata）

**数据包结构**：
```json
{
  "device_id": "DEV001",
  "seq": 12345,
  "ts": 1704067200,
  "sensor": {
    "temperature": 25.3,
    "humidity": 60.5,
    "rainfall": 0.0,
    "wind_speed": 2.1
  },
  "edge_flags": {
    "trigger": false,
    "abnormal_reduced": false,
    "missing_filled": false
  },
  "edge_pred": {
    "rainfall_pred": 0.0,
    "confidence": 0.85
  },
  "fw_version": "v1.2.3",
  "battery": 85,
  "rssi": -65
}
```

**数据量**：~200 bytes/packet

### 2.4 缓存阶段

**输入**：完整数据包

**处理**：
- 写入 Ring Buffer（内存，最近 1000 条）
- 写入 Flash（持久化，按时间窗口分段）

**输出**：缓存索引 + watermark（max seq）

**存储策略**：
- Ring Buffer：循环覆盖（FIFO）
- Flash：分段存储（每段 1 小时，带索引）

## 3. 传输数据流（按模式详细展开）

### 3.1 Mode A：BLE → Gateway → Cloud（实时/准实时）【主要模式，占比 ~70%】

#### 3.1.1 端侧数据产生与缓存

**Edge 产生数据**：
1. **Raw Data**：原始传感器采样值（ADC 值）
2. **Clean Data**：预处理后数据（去抖/滤波/温漂补偿）
3. **Feature Data**：边缘特征（滑窗统计、变化率）
4. **Pred Data**：CNN 预测结果（降雨预测、异常检测）

**数据打包**：
```
Raw + Clean + Feature + Pred → 统一数据包格式 → 添加 seq/ts/crc → 完整数据包
```

**缓存策略**：
- **Ring Buffer（内存）**：
  - 大小：最近 1000 条（~200KB）
  - 用途：快速访问、补传
  - 覆盖策略：FIFO（循环覆盖）
- **Flash（持久化）**：
  - 分段存储：每段 1 小时数据（~720KB）
  - 索引：每段带 header（start_seq, end_seq, start_ts, end_ts, crc）
  - 用途：断网恢复后补传、离线导出

**Watermark 管理**：
- **内存 watermark**：Ring Buffer 的 max_seq（快速查询）
- **Flash watermark**：Flash 中已确认同步的 max_seq（持久化）
- **云端 watermark**：云端已确认接收的 max_seq（通过 ACK 更新）

#### 3.1.2 BLE 传输层（Gateway 侧）

**传输模式选择**：

**模式 A1：BLE 广播（超低功耗）**
```
数据包 → 简化广播包（34 bytes）→ Gateway 扫描接收 → 解析 → 去重
```

**广播包结构**：
```
[device_id:4B][seq:4B][ts:4B][sensor_data:20B][crc:2B]
总大小：34 bytes
```

**特点**：
- 超低功耗（~10μA sleep，~15mA 广播）
- 无需配对
- 包小，易丢包（需要应用层容错）

**模式 A2：BLE GATT 连接（双向通信）**
```
数据包 → GATT Notify → Gateway 接收 → ACK → 解析 → 去重
```

**GATT 服务结构**：
- Telemetry Characteristic（Notify）：上报数据
- Configuration Characteristic（Read/Write）：配置下发
- OTA Control Characteristic（Write）：OTA 控制
- Status Characteristic（Read）：设备状态

**特点**：
- 双向通信（可下发配置/OTA）
- 可靠性更高（ACK 机制）
- 功耗较高（连接维护）

**Gateway 去重与补传**：
1. **去重机制**：
   - 维护每个设备的 seq 窗口（如最近 1000 条）
   - 检测缺失 seq（gap detection）
   - 按 (device_id, seq) 去重

2. **补传流程**（GATT 模式）：
   ```
   检测缺失 seq（如 seq 100, 102, 103 → 缺失 101）
   → 发送 FETCH_RANGE(101, 101) 请求
   → 设备返回缺失数据
   → 更新 seq 窗口
   ```

3. **批量上云策略**：
   - **时间窗口**：每 10 秒批量上传一次
   - **数量窗口**：每 100 条批量上传一次
   - **事件触发**：关键事件（trigger）立即上传

**Gateway 本地缓存**（Store-and-Forward）：
- **用途**：网关断网时也不丢数据
- **存储**：本地数据库/文件系统
- **恢复**：网络恢复后自动补传

#### 3.1.3 Gateway → Cloud 数据流

**批量数据包结构**：
```json
{
  "gateway_id": "GW001",
  "batch_id": "BATCH123",
  "timestamp": 1704067200,
  "transport_mode": "BLE_GATT",
  "devices": [
    {
      "device_id": "DEV001",
      "packets": [
        {
          "seq": 100,
          "ts": 1704067200,
          "data": {
            "sensor": {...},
            "edge_features": {...},
            "edge_pred": {...},
            "flags": {...}
          },
          "crc": "0xABCD"
        },
        {"seq": 101, ...}
      ],
      "missing_seqs": [95, 97]  // 检测到的缺失 seq
    },
    {"device_id": "DEV002", "packets": [...]}
  ],
  "compressed": true,
  "compression_algorithm": "gzip"
}
```

**传输协议**：
- **MQTT**（优先）：QoS 1，至少一次投递
- **HTTP**（备用）：POST 请求，支持重试

**处理流程**：
1. Gateway 批量打包（聚合多设备数据）
2. 压缩（gzip，减少带宽）
3. MQTT/HTTP 上传到云端 IoT Gateway
4. 云端接收确认（ACK）

#### 3.1.4 云端处理流程（Mode A）

**云端 Ingress**：
```
MQTT/HTTP → IoT Gateway → 认证 → 路由 → 消息队列
```

**处理步骤**：
1. **协议解析**：MQTT payload / HTTP body 解析
2. **身份认证**：Gateway 证书验证
3. **路由**：按 tenant_id 路由到对应 topic
4. **Schema 校验**：payload 格式校验
5. **时间戳校验**：ts 合理性检查（不能太旧/太新）
6. **签名校验**：可选的数据完整性校验

**消息队列**：
- 分区策略：按 tenant_id 分区，子分区按 site_id
- 特性：至少一次投递（QoS 1），按租户限流

**流处理（Streaming QC）**：
1. **去重处理**：
   - 提取 (device_id, seq)
   - 查询去重表（Redis，TTL = 24 小时）
   - 已存在 → 丢弃；不存在 → 写入去重表 → 输出

2. **乱序处理**：
   - 允许小窗口乱序（±5 分钟）
   - 按 ts 排序
   - 超窗口数据标记为"延迟数据"

3. **数据质量检查**：
   - 缺失检测（seq 不连续）
   - 越界检测（传感器值超出合理范围）
   - 延迟统计（ts 与接收时间的差值）

4. **实时特征计算**：
   - 滑窗统计（1 分钟窗口：mean, std, min, max）
   - 雨强计算（瞬时雨强、累计雨量）
   - 频域分析（FFT 功率谱、主频）
   - 扰动指标（信噪比、稳定性指标）

5. **告警评估**：
   - 规则匹配（离线告警、传感器漂移、异常雨强）
   - 阈值判断
   - 告警去重（避免重复告警）

**Dual-Write 输出**：
- **Raw Stream**：原始数据 → 时序库（原始表），用于追溯
- **Clean/Feature Stream**：清洗后数据 + 特征 → 时序库（清洗表 + 特征表），用于看板/分析/告警

#### 3.1.5 OTA/配置下发（Mode A）

**OTA 流程**（通过 BLE GATT）：
1. 云端发起 OTA Campaign
2. Gateway 接收 OTA 指令
3. Gateway 通过 GATT 连接下发 OTA 控制命令
4. 设备进入 OTA 模式
5. 分块传输固件包（每块 512 bytes）
6. 设备校验（每块 CRC，整体 hash）
7. 设备安装新固件
8. 设备上报升级结果

**配置下发流程**：
1. 云端更新 Device Shadow（desired config）
2. Gateway 检测到 delta（desired ≠ reported）
3. Gateway 通过 GATT 下发配置
4. 设备更新配置
5. 设备上报确认（reported = desired）

**实时性**：
- OTA/配置下发通过 GATT 连接，实时性高（秒级）
- 支持批量下发（Gateway 管理多设备）

### 3.2 Mode B：纯离线存储 → 数据线导出（批处理）【占比 ~10%】

#### 3.2.1 端侧数据产生与缓存

**Edge 产生数据**（与 Mode A 相同）：
- Raw Data、Clean Data、Feature Data、Pred Data

**缓存策略**（重点在 Flash）：
- **Ring Buffer（内存）**：最近 1000 条（临时缓存）
- **Flash（持久化）**：长期存储，按月份/按块分段

**Flash 存储结构**：
```
Segment 1: [header][data_block_1][data_block_2]...[index]
Segment 2: [header][data_block_1][data_block_2]...[index]
...
```

**Segment Header**：
```json
{
  "start_seq": 1000,
  "end_seq": 2000,
  "start_ts": 1704067200,
  "end_ts": 1704070800,
  "data_size": 200000,
  "crc": "0xABCD",
  "segment_id": "SEG001"
}
```

**数据完整性保证**：
- **分段日志 + 索引**：按月份/按块存，带起止 seq/ts，便于增量导出
- **CRC 校验**：每段带 CRC，确保数据完整性
- **循环覆盖**：Flash 满后覆盖最老段（带索引，可追溯）

#### 3.2.2 离线导出流程

**导出触发**：
- 维护人员到现场
- 设备进入维护模式（通过物理按键/有线连接）
- 有线连接建立（UART/USB）

**增量同步协议**：
1. **HELLO 交换**：
   - Base Station 发送：`HELLO {base_station_id, sync_protocol_version}`
   - 设备返回：`HELLO_ACK {device_id, fw_version, storage_capacity, segment_list}`

2. **GET_WATERMARK**：
   - Base Station 发送：`GET_WATERMARK {device_id}`
   - 设备返回：`WATERMARK {max_seq, max_ts, segment_list}`
   - Base Station 记录：上次同步到的 watermark（max seq / time）

3. **FETCH_RANGE**（增量拉取）：
   - Base Station 发送：`FETCH_RANGE {device_id, seq_start, seq_end, segment_id}`
   - 设备返回：`DATA_BLOCK {seq_start, seq_end, data[], crc}`
   - Base Station 校验：CRC 校验，确保数据完整性

4. **ACK**（确认同步）：
   - Base Station 发送：`ACK {device_id, max_seq, timestamp}`
   - 设备更新：Flash watermark（持久化）

5. **断点续传**：
   - 任何时候中断都能继续
   - 通过 watermark 记录同步进度
   - 支持分段拉取（大文件分块传输）

**设备健康快照**（加分点）：
- 导出时附带设备健康信息：
  - 电池电量、故障码
  - 版本信息（fw_version, model_version, config_version）
  - 近 30 天缺失率
  - 数据质量统计

#### 3.2.3 Base Station/PC → Cloud 数据流

**批量数据包结构**：
```json
{
  "export_id": "EXPORT789",
  "export_timestamp": 1704067200,
  "export_mode": "OFFLINE_WIRED",
  "devices": [
    {
      "device_id": "DEV001",
      "watermark_before": {"max_seq": 1000, "max_ts": 1704000000},
      "watermark_after": {"max_seq": 5000, "max_ts": 1704067200},
      "packets": [
        {"seq": 1001, "ts": 1704000060, "data": {...}},
        ...
      ],
      "health_snapshot": {
        "battery": 85,
        "fault_codes": [],
        "fw_version": "v1.2.3",
        "missing_rate_30d": 0.01
      }
    }
  ]
}
```

**传输协议**：
- **HTTP POST**：批量上传到云端
- **断点续传**：支持大文件分块上传

#### 3.2.4 云端处理流程（Mode B）

**云端 Ingress**：
```
HTTP POST → IoT Gateway → 认证 → 路由 → 消息队列
```

**批处理特性**：
- **批量去重**：按 (device_id, seq) 批量去重
- **批量写入**：批量写入时序库（提高吞吐）
- **幂等保证**：同一条 (device_id, seq) 重传不产生重复写入

**流处理（与 Mode A 相同）**：
1. 去重处理
2. 乱序处理（允许更大窗口，如 ±1 小时）
3. 数据质量检查
4. 特征计算（批处理模式）
5. 告警评估（基于历史数据）

**Dual-Write 输出**（与 Mode A 相同）：
- Raw Stream → 时序库（原始表）
- Clean/Feature Stream → 时序库（清洗表 + 特征表）

#### 3.2.5 OTA/配置下发（Mode B）

**OTA 流程**（维护窗口）：
1. 云端发起 OTA Campaign（目标：离线设备）
2. 维护人员到现场
3. 有线连接建立
4. Base Station 下发 OTA 控制命令
5. 设备进入 OTA 模式
6. 分块传输固件包
7. 设备校验并安装
8. 设备上报升级结果

**配置下发流程**（维护窗口）：
1. 云端更新 Device Shadow
2. 维护人员到现场
3. 有线连接建立
4. Base Station 下发配置
5. 设备更新配置
6. 设备上报确认

**特点**：
- **延迟高**：需要维护人员到现场
- **可靠性高**：有线连接，物理确定性
- **适合场景**：极端低功耗、无网、维护周期长

### 3.3 Mode C：多设备有线汇聚 → Base Station → Cloud（集中式）【次要模式，占比 ~20%】

#### 3.3.1 端侧数据产生与缓存（与 Mode A 相同）

**Edge 产生数据**：
- Raw Data、Clean Data、Feature Data、Pred Data（与 Mode A 相同）

**缓存策略**：
- **Ring Buffer（内存）**：最近 1000 条
- **Flash（持久化）**：分段存储，每段 1 小时

**Watermark 管理**：
- 与 Mode A 相同，但通过有线连接同步

#### 3.3.2 Base Station 并发采集

**连接方式**：
- **多路串口**：RS232/RS485（如 10 路）
- **USB Hub**：多设备 USB 连接
- **以太网**：设备支持以太网时

**并发采集策略**：
- **多线程/多进程**：每个设备独立线程
- **连接池管理**：复用连接，减少开销
- **超时控制**：单设备超时不影响其他设备

**采集流程**：
```
1. Base Station 扫描连接的设备
2. 建立连接（串口/USB/以太网）
3. HELLO 交换（设备身份/版本/容量）
4. GET_WATERMARK（获取每个设备的 watermark）
5. FETCH_RANGE（按设备拉取增量数据）
6. 断点续传（支持中断后继续）
```

**去重处理**：
- 按设备维护 seq 窗口
- 按 (device_id, seq) 去重
- 检测缺失 seq，请求补传

#### 3.3.3 Base Station 站点级处理

**统一设备管理**：
- **Asset Inventory**：站点级设备列表
  - 设备 ID、版本、健康度
  - 最后同步时间、数据量统计
- **设备状态监控**：
  - 在线/离线状态
  - 电池电量、故障码
  - 数据质量指标（缺失率、延迟）

**站点级聚合**：
- **实时聚合**：
  - 站点平均温度/湿度
  - 站点总雨量
  - 站点设备健康度
- **统计聚合**：
  - 站点数据量统计
  - 站点告警统计
  - 站点设备在线率

**批量同步策略**：
- **时间窗口**：每 5 分钟批量同步一次
- **数量窗口**：每设备 1000 条批量同步
- **优先级**：关键事件优先同步

#### 3.3.4 Base Station → Cloud 数据流

**批量数据包结构**：
```json
{
  "base_station_id": "BS001",
  "site_id": "SITE001",
  "batch_id": "BATCH456",
  "timestamp": 1704067200,
  "transport_mode": "WIRED_AGGREGATION",
  "devices": [
    {
      "device_id": "DEV001",
      "watermark": {"max_seq": 1000, "max_ts": 1704067200},
      "packets": [
        {"seq": 1001, "ts": 1704067260, "data": {...}},
        {"seq": 1002, "ts": 1704067320, "data": {...}}
      ],
      "missing_seqs": []
    },
    {"device_id": "DEV002", ...}
  ],
  "site_aggregation": {
    "avg_temperature": 25.3,
    "total_rainfall": 10.5,
    "device_count": 10,
    "online_count": 9,
    "health_score": 0.95
  },
  "compressed": true
}
```

**传输协议**：
- **MQTT**（优先）：QoS 1
- **HTTP**（备用）：POST 请求

**处理流程**：
1. Base Station 批量打包（聚合多设备数据 + 站点级聚合）
2. 压缩（gzip）
3. MQTT/HTTP 上传到云端 IoT Gateway
4. 云端接收确认（ACK）

#### 3.3.5 云端处理流程（Mode C）

**云端 Ingress**（与 Mode A 相同）：
```
MQTT/HTTP → IoT Gateway → 认证 → 路由 → 消息队列
```

**流处理（Streaming QC）**（与 Mode A 相同）：
1. 去重处理
2. 乱序处理
3. 数据质量检查
4. 实时特征计算
5. 告警评估

**Dual-Write 输出**（与 Mode A 相同）：
- Raw Stream → 时序库（原始表）
- Clean/Feature Stream → 时序库（清洗表 + 特征表）

**站点级数据处理**：
- **站点级聚合**：Base Station 已做站点级聚合，云端可直接使用
- **站点级告警**：基于站点级聚合数据触发告警
- **站点级报表**：生成站点级日报/月报

#### 3.3.6 OTA/配置下发（Mode C）

**批量 OTA 流程**：
1. 云端发起 OTA Campaign（目标：站点级设备）
2. Base Station 接收 OTA 指令
3. Base Station 批量下发 OTA 控制命令（通过有线连接）
4. 多设备并发进入 OTA 模式
5. 分块传输固件包（每块 512 bytes，并发传输）
6. 设备校验并安装
7. Base Station 收集升级结果
8. Base Station 统一上报升级结果

**批量配置下发流程**：
1. 云端更新 Device Shadow（站点级设备 desired config）
2. Base Station 检测到 delta
3. Base Station 批量下发配置（通过有线连接）
4. 设备更新配置
5. Base Station 收集确认
6. Base Station 统一上报确认

**优势**：
- **效率高**：批量操作，比逐个设备更高效
- **可靠性高**：有线连接，物理确定性
- **统一管理**：站点级统一管理，便于运维

#### 3.3.7 Mode C 典型场景

**场景 1：维护时插线汇聚**
- 维护人员到现场
- 插线连接多个设备
- Base Station 批量同步数据
- 统一上传到云端
- 完成维护后拔线

**场景 2：永久走线集中采集**
- 站点永久部署 Base Station
- 多个传感器通过有线连接到 Base Station
- Base Station 持续采集并上传
- 支持集中供电、统一管理

## 4. 云端接入数据流

### 4.1 Gateway → Cloud

**数据流**：
```
Gateway 批量数据包 → MQTT/HTTP → IoT Gateway → 认证 → 路由 → 消息队列
```

**批量数据包结构**：
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
    },
    {
      "device_id": "DEV002",
      "packets": [...]
    }
  ]
}
```

**处理流程**：
1. **协议解析**：MQTT payload / HTTP body 解析
2. **身份认证**：Gateway 证书验证
3. **路由**：按 tenant_id 路由到对应 topic
4. **Schema 校验**：payload 格式校验
5. **时间戳校验**：ts 合理性检查
6. **签名校验**：可选的数据完整性校验

**输出**：验证通过的数据包 + 路由信息

### 4.2 消息队列数据流

**数据流**：
```
验证通过的数据包 → 消息队列（按 tenant 分区）→ 流处理消费
```

**分区策略**：
- 主分区：tenant_id
- 子分区：site_id（可选）
- 哈希分区：device_id（负载均衡）

**特性**：
- 至少一次投递（QoS 1）
- 按租户限流
- 死信队列（处理失败的消息）

## 5. 流处理数据流

### 5.1 去重处理流

**输入**：来自消息队列的数据包

**处理**：
```
数据包 → 提取 (device_id, seq) → 查询去重表 → 
[已存在] → 丢弃
[不存在] → 写入去重表 → 输出
```

**去重表**：
- 存储：Redis（TTL = 24 小时）
- 结构：`device_id:seq` → timestamp

**输出**：去重后的数据包

### 5.2 乱序处理流

**输入**：去重后的数据包

**处理**：
```
数据包 → 提取 ts → 窗口排序 → 
[在窗口内] → 按 ts 排序输出
[超窗口] → 标记为"延迟数据" → 输出
```

**窗口策略**：
- 允许乱序窗口：±5 分钟
- 超窗口数据：标记但不丢弃（用于追溯）

**输出**：按时间排序的数据流

### 5.3 数据质量检查流

**输入**：排序后的数据包

**处理流程**：
```
数据包 → 缺失检测 → 重复检测 → 越界检测 → 
单位异常检测 → 延迟统计 → 质量标记
```

**检查项**：
- **缺失检测**：seq 不连续 → 标记缺失区间
- **重复检测**：相同 (device_id, seq) → 去重（已处理）
- **越界检测**：传感器值超出合理范围 → 标记异常
- **单位异常**：单位不匹配 → 标记错误
- **延迟统计**：ts 与接收时间的差值 → 记录延迟

**输出**：质量标记 + 异常记录

**质量标记结构**：
```json
{
  "quality_flags": {
    "missing": false,
    "out_of_range": false,
    "unit_error": false,
    "delayed": false
  },
  "delay_ms": 120,
  "missing_ranges": []
}
```

### 5.4 实时特征计算流

**输入**：质量检查后的数据包

**处理流程**：
```
数据包 → 滑窗统计 → 雨强计算 → 频域分析 → 
扰动指标计算 → 特征向量组装
```

**特征类型**：

**统计特征**（1 分钟窗口）：
- mean, std, min, max
- 变化率（rate of change）

**雨强特征**：
- 瞬时雨强（rainfall_rate）
- 累计雨量（accumulated_rainfall）

**频域特征**（FFT）：
- 主频（dominant_frequency）
- 功率谱（power_spectrum）

**扰动指标**：
- 信噪比（SNR）
- 稳定性指标（stability_index）

**输出**：特征向量

**特征向量结构**：
```json
{
  "device_id": "DEV001",
  "ts": 1704067200,
  "window": "1min",
  "features": {
    "statistical": {
      "mean": 25.3,
      "std": 0.5,
      "min": 24.8,
      "max": 25.8
    },
    "rainfall": {
      "rate": 0.0,
      "accumulated": 0.0
    },
    "frequency": {
      "dominant_freq": 0.1,
      "power": 0.05
    },
    "disturbance": {
      "snr": 30.5,
      "stability": 0.95
    }
  }
}
```

### 5.5 告警评估流

**输入**：特征向量 + 数据包

**处理流程**：
```
特征向量 → 规则匹配 → 阈值判断 → 告警去重 → 
告警升级 → 告警事件生成
```

**告警规则示例**：
- **离线告警**：设备超过 5 分钟未上报
- **传感器漂移**：长期趋势异常（如温度持续上升）
- **异常雨强**：rainfall_rate > 阈值
- **疑似故障**：数据质量持续恶化（缺失率 > 10%）

**告警事件结构**：
```json
{
  "alert_id": "ALT001",
  "device_id": "DEV001",
  "site_id": "SITE001",
  "tenant_id": "TENANT001",
  "alert_type": "abnormal_rainfall",
  "severity": "high",
  "timestamp": 1704067200,
  "details": {
    "rainfall_rate": 50.5,
    "threshold": 30.0
  }
}
```

**输出**：告警事件

### 5.6 双写输出流

**Raw Stream**：
```
数据包 → 原始数据表（时序库）
```

**Clean/Feature Stream**：
```
数据包 + 特征向量 → 清洗数据表 + 特征表（时序库）
```

**存储策略**：
- Raw：完整数据（用于追溯）
- Clean：清洗后数据（用于看板）
- Feature：特征数据（用于分析）

## 6. 存储数据流

### 6.1 时序数据写入流

**输入**：Raw/Clean/Feature 数据流

**处理流程**：
```
数据流 → 批量缓冲（时间窗口/数量窗口）→ 分区路由 → 
压缩 → 写入时序库 → 写入确认
```

**批量策略**：
- 时间窗口：每 10 秒批量写入一次
- 数量窗口：每 1000 条批量写入一次
- 两者取先到者

**分区策略**：
- 按时间分区：按天/按月
- 按租户分区：tenant_id
- 按站点分区：site_id（可选）

**压缩**：列式压缩（如 Parquet）

**输出**：写入确认 + 存储位置

### 6.2 元数据更新流

**输入**：数据包 + 质量标记

**处理流程**：
```
数据包 → 提取元数据 → 异步更新元数据库
```

**更新内容**：
- 设备最后在线时间：`last_seen = ts`
- 设备健康度：基于数据质量计算
- 设备版本信息：`fw_version`
- 站点统计：设备数、数据量、告警数

**更新策略**：
- 异步更新（不阻塞主流程）
- 批量更新（减少数据库压力）
- 定时聚合（如每小时更新一次站点统计）

### 6.3 对象存储写入流

**输入**：固件包、日志、报表

**处理流程**：
```
文件 → 计算 hash → 上传对象存储 → 更新元数据库
```

**存储内容**：
- 固件包：`firmware/{version}/{hash}.bin`
- 端侧日志：`logs/{device_id}/{date}.log`
- 离线报表：`reports/{tenant_id}/{date}.json`
- 模型文件：`models/{model_id}/{version}.onnx`

## 7. 分析数据流

### 7.1 实时分析流

**查询流**：
```
用户查询 → Query API → 查询时序库 → 数据聚合 → 返回结果
```

**查询类型**：
- **实时看板**：最近 N 分钟的数据聚合
- **实时告警**：流式规则匹配
- **实时统计**：在线设备数、数据量

**缓存策略**：
- 最近 1 小时数据缓存（Redis）
- 定时刷新（如每 10 秒）

### 7.2 离线分析流（Batch ETL）

**日报生成流**：
```
当日原始数据 → 按站点/设备聚合 → 生成日报 → 存储对象存储
```

**月报生成流**：
```
当月数据 → 趋势分析 → 异常统计 → 设备健康度评估 → 生成月报
```

**校准数据生成流**：
```
历史数据 + 校准规则 → 传感器校准参数计算 → 生成校准参数
```

**模型训练数据回放流**：
```
历史数据 + 标签 → 特征工程 → 数据增强 → 生成训练数据集
```

## 8. Serving 数据流

### 8.1 Query API 流

**查询流**：
```
用户请求 → API Gateway → 权限检查 → 查询时序库 → 
数据聚合/下采样 → 返回 JSON
```

**查询示例**：
```
GET /api/v1/devices/DEV001/telemetry?start=1704067200&end=1704070800&interval=1min
```

**处理流程**：
1. 参数校验
2. 权限检查（租户隔离）
3. 查询时序库（按时间范围）
4. 数据聚合（按 interval）
5. 下采样（如需要）
6. 返回 JSON

### 8.2 Dashboard 流

**数据流**：
```
前端请求 → WebSocket → 后端查询时序库/缓存 → 
数据格式化 → WebSocket 推送（实时更新）
```

**实时更新**：
- 前端订阅设备/站点
- 后端定时查询（如每 10 秒）
- WebSocket 推送更新

### 8.3 Alert 流

**告警流**：
```
告警事件 → 告警服务 → 匹配订阅者 → 通知渠道（短信/邮件/企业微信）
```

**告警去重**：
- 相同告警在 N 分钟内只发送一次
- 告警升级机制（持续异常 → 升级告警级别）

## 9. 数据流指标与监控

### 9.1 端到端延迟

**测量点**：
- T1：端侧采样时间（ts）
- T2：云端接收时间
- T3：存储完成时间
- T4：查询返回时间

**延迟指标**：
- **传输延迟**：T2 - T1（目标 < 5 秒，P99）
- **处理延迟**：T3 - T2（目标 < 1 秒，P99）
- **查询延迟**：T4 - T3（目标 < 500ms，P99）

### 9.2 数据完整性

**指标**：
- **缺失率**：缺失数据条数 / 总数据条数（目标 < 0.1%）
- **重复率**：重复数据条数 / 总数据条数（目标 < 0.01%）
- **乱序率**：乱序数据条数 / 总数据条数（目标 < 1%）

### 9.3 吞吐量

**指标**：
- **端侧采样频率**：samples/second
- **云端接收 QPS**：messages/second
- **处理 QPS**：processed messages/second
- **存储 QPS**：stored records/second

### 9.4 链路追踪

**追踪 ID**：贯穿全链路的 trace_id

**追踪点**：
- 端侧采样
- 传输
- 云端接收
- 处理
- 存储
- 查询

**用途**：
- 问题排查
- 性能分析
- 数据追溯
