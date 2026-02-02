# HydroSense System Design

## 0. 一句话定位

HydroSense 是一套**多设备、多租户**的环境传感 IoT 平台：端侧（STM32）做触发/检测/异常抑制/缺失填补/扰动去除 + 小型 CNN 预测，云端负责数据接入、设备与固件版本管理（OTA）、时序存储、质量控制、分析可视化、对外集成。

**核心设计亮点**：We supported multiple field deployment modes (BLE-to-gateway, offline store-and-forward via wired harvesting, and multi-sensor wired aggregation to a base station) by standardizing the data contract and sync protocol across transports, ensuring idempotent ingestion, incremental sync, and long-term low-power reliability.

## 1. 需求与约束

### 1.1 功能需求

- **设备接入**：海量设备上报（传感器数据、状态、日志、心跳）
- **端侧智能**：trigger/detection、abnormal reduce、missing fitting、扰动去除、CNN 预测（轻量）
- **云端数据产品**：实时/离线分析、报表/可视化、告警、回溯查询
- **设备管理**：注册/认证、配置下发、远程诊断、分组/站点管理
- **软件/固件版本管理**：灰度发布、回滚、分批 OTA、设备端版本追踪
- **多租户/多客户**：不同公司、不同项目隔离；也支持接入第三方云（托管或客户自管）

### 1.2 非功能需求

- **可靠性**：断网/弱网容忍，端侧缓存；云端至少一次投递 + 去重
- **可扩展**：设备数/上报频率增长；按租户隔离限流
- **安全合规**：设备身份、传输加密、权限、审计
- **可观测性**：端到端链路追踪（设备→云→分析→告警）

## 2. 总体架构

### 2.1 架构层次

```
Edge（STM32） → IoT 接入层 → 消息/流处理 → 存储层（时序+对象+元数据） 
→ 分析层（实时+离线） → 产品层（可视化/告警/API） → 运维层（设备/版本/监控）
```

### 2.2 核心组件

#### Device/Edge
- Sensors + STM32 Firmware
- Edge pipeline（滤波/检测/填补/模型推理）
- Local buffer（flash/ring buffer）
- Uplink（BLE/有线/离线）

#### Ingress
- IoT Gateway / MQTT Broker / HTTP Ingest
- AuthN（设备证书/密钥）
- Topic/Route（按租户/站点/设备）

#### Stream Processing
- Queue（Kafka 类 / 云消息队列）
- Real-time QC（去噪、去异常、重复包去重、延迟统计）
- Rule Engine / Alert Evaluator

#### Storage
- Time-series DB（原始/清洗后/特征）
- Object Storage（固件包、日志、批处理输出）
- Metadata DB（设备、站点、版本、租户、策略）

#### Analytics & Serving
- Batch ETL（日报/月报、校准、模型训练数据）
- Query Service（聚合、downsample）
- Dashboard / API（可视化、对外集成）

#### Device & OTA Management
- Device registry / shadow
- Config push
- OTA campaign（灰度/回滚/分批）

## 3. 端侧设计（STM32）

### 3.1 端侧数据链

```
采样 → 预处理（去抖/滤波/温漂补偿）→ 事件触发（trigger）→ 异常抑制（abnormal reduce）
→ 缺失填补（missing fitting）→ 扰动去除（抗干扰）→ 小模型推理（CNN predict）
→ 打包上报 + 本地缓存
```

### 3.2 端侧关键机制

- **断网缓存**：ring buffer + watermark（按时间戳顺序补传）
- **触发上报**：平稳期低频、事件期高频（省电/省流量）
- **消息幂等**：每条记录带 device_id + seq + timestamp，云端可去重
- **模型/规则双轨**：规则兜底（可解释、鲁棒），CNN 提升预测能力（可在云端回放验证）

## 4. 传输模式设计（Field-First Architecture - 系统级归一化）

### 4.1 统一抽象：一个 Edge，多条回传路径（Transport 可插拔）

**核心设计理念**：HydroSense 采用"系统级归一化"架构，将多种采集/回传模式抽象为**同一套 Edge 数据与版本体系 + 多种 Transport/Backhaul 插件**，而非临时拼凑的多套系统。

**端侧（Sensor Node）统一职责**（所有模式共享）：
1. **采样 + 端侧算法**：trigger/detection、abnormal reduce、missing fitting、扰动去除、轻量 CNN 预测
2. **本地持久化日志**：ring buffer / flash，按时间与 seq 写入
3. **导出统一数据包格式**：不管 BLE 还是有线，payload 结构完全一致

**关键设计原则**：
- **统一数据模型**：所有模式共享同一个数据 schema
- **统一序号/幂等机制**：(device_id, seq) 作为全局唯一标识
- **统一压缩/校验**：相同的数据压缩算法和 CRC 校验
- **统一签名机制**（可选）：相同的数据签名算法

**优势**：
- 切换传输模式不影响云端/分析层
- 云端处理逻辑完全统一（去重/QC/存储）
- 设备固件只需实现 Transport 插件层，核心逻辑复用
- 支持混合部署（同一站点部分设备走 A，部分走 C）

### 4.2 三种传输模式

#### Mode A：BLE → 基站（Gateway）→ 云（实时/准实时）

**适用场景**：有网关、希望接近实时、能做告警

**链路**：`Sensor(BLE) → Gateway(4G/WiFi/Eth) → Cloud`

**Gateway 职责**：
- 连接/扫描调度：管理多设备 BLE（广播 or GATT）
- 去重与补传协调：按 (device_id, seq) 去重；缺口请求补传
- 批量上云：聚合/压缩后上传（节省带宽）
- 本地缓存：网关断网时也不丢（store-and-forward）
- 下发配置/阈值/模型版本（可选）：把云端策略推到端侧

**BLE 通信模式**：
- **模式 A1：BLE 广播**（广播包里带简化 telemetry）
  - 优点：超低功耗、无需配对、网关可同时扫很多设备
  - 缺点：包小、容易丢，需要应用层容错
- **模式 A2：BLE GATT 连接**（notify + write）
  - 优点：可双向（下发配置/取缓存/ACK），可靠性更可控
  - 缺点：连接维护耗电更高、并发受限

**抗干扰/可靠性机制**：
- 序号 + 时间戳 + CRC：每条样本 seq + ts + crc，网关侧能检测丢包/乱序/坏包
- 应用层 ACK + 重传（轻量）：关键事件包（trigger）必须 ACK，否则重传 N 次
- 本地缓存补传：断链时写 ring buffer，恢复后"按 seq"补齐
- 自适应发送策略：平稳期低频（省电），事件期高频 + 更强纠错

#### Mode B：纯离线（存硬件）→ 几个月一次数据线导出

**适用场景**：野外极端低功耗、无网、维护周期长

**链路**：`Sensor(Flash) → Technician(Wired) → 导出到基站/PC → 云（批处理）`

**数据完整性保证**：
- 分段日志 + 索引：按月份/按块存，带起止 seq/ts，便于增量导出
- 增量同步协议：基站/PC 记录"上次同步到的 watermark（max seq / time）"，下次只拉增量
- 云端幂等：同一条 (device_id, seq) 重传不产生重复写入

**加分点**：离线导出也能附带"设备健康快照"（电池、故障码、版本、近 30 天缺失率等），便于运维。

#### Mode C：多设备有线汇聚 → 统一基站管理与上传（集中式）

**适用场景**：一个站点很多传感器，维护时插线汇聚，或永久走线集中供电/集中采集

**链路**：`Many Sensors(Wired) → Hub/Base Station → Cloud`

**Base Station 职责**：
- 多设备并发采集：多路串口/USB/RS485
- **工业协议支持**：硬件终端（RTU - Remote Terminal Unit）通过 Modbus RTU（串口）或 Modbus TCP（以太网）通信；Base Station 读取 Modbus 寄存器并转换为统一 Telemetry Record
- 统一设备管理：站点级 asset inventory（设备列表/版本/健康）
- 批量同步：按设备 watermark 拉取增量；支持断点续传
- 统一上传：把站点级数据统一上云，顺便做站点级聚合（日报、告警）
- 现场 OTA：必要时基站对多设备批量升级（比逐个设备更高效）

### 4.3 模式选择策略框架（按约束自动落地）

**设计理念**：将"模式选择"做成策略框架，而非人为操作，体现生产系统的工程化思维。

**策略维度**：

1. **Connectivity（连接性）**
   - 有网关 → Mode A（BLE → Gateway → Cloud）
   - 无网 → Mode B（离线存储）或 Mode C（有线汇聚）

2. **Power（功耗）**
   - 电池紧张 → 降频 + 多缓存 → Mode B/C 更适合
   - 市电/集中供电 → Mode A/C 均可

3. **Latency（延迟）**
   - 需要告警/实时监控 → Mode A（实时/准实时）
   - 允许延迟（批处理） → Mode B/C

4. **Scale（规模）**
   - 设备很多且集中 → Mode C（有线汇聚，统一管理）
   - 设备分散且少量 → Mode A（BLE）或 Mode B（离线）

5. **Cost（成本）**
   - 流量成本高 → Mode A 的批量上传 + 端侧压缩
   - 维护成本高 → Mode B 的周期导出

**策略执行**：
- 设备启动时根据环境约束自动选择模式
- 运行时可根据条件动态切换（如网关故障时降级到 Mode B）
- 云端可下发模式切换指令（通过配置下发）

### 4.4 低功耗策略（野外长期运行的核心卖点）

- **Duty Cycling**：采样与发包分离，平时 deep sleep
- **事件触发优先**：只有触发/异常时提升上报码率
- **边缘计算换传输**：端侧做 abnormal reduce / 扰动去除 / missing fitting，减少无效上行
- **网关侧聚合上传**：BLE→网关近场低功耗，网关再批量上云（减少蜂窝功耗成本）

### 4.5 有线升级（数据线）的定位

**不是生产数据链路，主要用于**：
- 固件升级 / 工厂校准 / debug / 救砖（BLE 不可靠时的兜底）
- 可靠性兜底：野外 BLE 环境不稳定时，需要"物理确定性"路径
- 救砖：固件升级失败、bootloader 异常时只能走线
- 工厂校准/批量烧录：生产效率
- 安全隔离：有线模式可进入维护态（maintenance mode），和正常运行态严格区分

## 5. 云端核心组件拆解

### 5.1 接入层（IoT Ingress）

- **协议**：MQTT（优先）或 HTTPS
- **认证**：每设备唯一密钥/证书；按租户隔离 topic
- **入口校验**：payload schema 校验、时间戳合理性、签名校验

### 5.2 消息队列 / 路由

- 把"接入"与"处理"解耦：避免峰值把后端打挂
- 每条消息进入队列后，按租户/站点分区，便于限流与隔离

### 5.3 实时处理（Streaming QC）

**处理内容分 3 类**：
- **数据质量**：缺失/重复/乱序/延迟、单位异常、越界
- **实时特征**：滑窗均值/方差、雨强、累计量、频域扰动指标
- **告警**：离线、传感器漂移、异常雨强、疑似故障

**输出两条流**：
- Raw（用于追溯）
- Clean/Feature（用于看板、分析、告警）

### 5.4 存储层（三库分离）

- **时序库**：原始 + 清洗 + 特征（支持按时间范围聚合、downsample）
- **元数据库（RDB）**：设备/租户/站点/版本/策略/告警配置
- **对象存储**：固件包、端侧日志、离线报表、模型文件

### 5.5 服务层（Serving / API）

- **Query API**：按设备/站点/时间范围查；支持聚合（分钟/小时/天）
- **Device API**：注册、分组、状态、最后在线时间、健康度
- **Alert API**：告警订阅、通知渠道（短信/邮件/企业微信等）
- **Integration API**：对接第三方平台（客户自有云/数据仓库）

## 6. OTA & 版本管理

### 6.1 关键对象

- **FirmwareArtifact**：固件包（版本号、hash、大小、签名）
- **ReleaseChannel**：stable / beta / canary
- **Campaign**：一次发布活动（目标设备集合、分批策略、窗口期、回滚条件）
- **DeviceVersionState**：设备当前版本、上次升级时间、升级结果

### 6.2 灰度策略

- **按站点/按批次**：1% → 10% → 50% → 100%
- **健康门禁**：升级后 crash rate / reboot / data gap / battery 指标恶化则暂停
- **自动回滚**：N 次失败或关键指标触发阈值

## 7. 多租户与对外接入

### 7.1 模式 A：客户自管云

**提供**：设备固件、数据格式、边缘算法、参考云端 pipeline

**客户负责**：IoT 平台、存储、可视化

**优点**：客户合规更容易；**缺点**：对体验可控性低

### 7.2 模式 B：托管云

**提供**：全栈（接入→处理→存储→看板→OTA）

**优点**：交付快、持续运维；**缺点**：多租户隔离、安全与成本更敏感

**统一层**：标准化数据合同（schema）+ 设备身份体系 + 版本/配置体系

## 8. 数据模型（Schema-First）

### 8.1 Telemetry（上报数据）

```json
{
  "tenant_id": "string",
  "device_id": "string",
  "ts": "timestamp",
  "seq": "monotonic_integer",
  "sensor": {
    "temperature": "float",
    "humidity": "float",
    "rainfall": "float",
    "wind": "float"
  },
  "edge_flags": {
    "trigger": "boolean",
    "abnormal_reduced": "boolean",
    "missing_filled": "boolean"
  },
  "edge_pred": "CNN_output (optional)",
  "fw_version": "string",
  "rssi": "integer",
  "battery": "integer"
}
```

### 8.2 Device Shadow（设备状态）

- **reported**：当前版本、配置、生存状态
- **desired**：目标版本、目标配置（阈值、上报频率、模型版本）

### 8.3 统一数据包格式（所有 Transport 模式共享）

**核心字段**（固定结构，所有模式一致）：
```json
{
  "tenant_id": "string",
  "site_id": "string",
  "device_id": "string",
  "ts": "timestamp (采样时间)",
  "seq": "monotonic_integer (单调递增序号)",
  "payload": {
    "sensor": {...},
    "edge_features": {...},
    "edge_pred": {...},
    "flags": {...}
  },
  "fw_version": "string",
  "model_version": "string",
  "config_version": "string",
  "crc": "uint16 (完整性校验)",
  "signature": "string (可选，数据签名)"
}
```

**设计要点**：
- **固定字段顺序**：便于解析和校验
- **固定字段类型**：便于序列化/反序列化
- **向后兼容**：新增字段放在 payload 内，不破坏现有解析

### 8.4 统一同步协议（所有 Transport 模式共享）

**协议设计**：不管 BLE/有线/离线，都使用同一套同步协议，确保一致性。

**核心命令**：

1. **HELLO**
   - **用途**：设备身份/版本/容量信息交换
   - **请求**：`HELLO {device_id, fw_version, storage_capacity, data_segments}`
   - **响应**：`HELLO_ACK {gateway_id, sync_protocol_version}`

2. **GET_WATERMARK**
   - **用途**：获取设备当前同步 watermark（已同步到的最大 seq）
   - **请求**：`GET_WATERMARK {device_id}`
   - **响应**：`WATERMARK {max_seq, max_ts, segment_list}`

3. **FETCH_RANGE**
   - **用途**：拉取指定范围的数据
   - **请求**：`FETCH_RANGE {device_id, seq_start, seq_end, segment_id}`
   - **响应**：`DATA_BLOCK {seq_start, seq_end, data[], crc}`

4. **ACK**
   - **用途**：确认已同步到某个 seq
   - **请求**：`ACK {device_id, max_seq, timestamp}`
   - **响应**：`ACK_CONFIRM {status}`

5. **断点续传机制**
   - 任何时候中断都能继续
   - 通过 watermark 记录同步进度
   - 支持分段拉取（大文件分块传输）

**协议优势**：
- **统一性**：所有模式使用相同协议，代码复用
- **可靠性**：ACK 机制保证数据不丢失
- **效率**：增量同步，只拉取新数据
- **容错**：断点续传，网络中断不影响

## 9. 可靠性与一致性

### 9.1 至少一次投递

- MQTT QoS1 / HTTP 重试

### 9.2 云端幂等去重

- (device_id, seq) 或 (device_id, ts, hash) 去重写入

### 9.3 乱序处理

- 允许小窗口乱序，落库后按 ts 聚合

### 9.4 回压保护

- 入口限流 + 队列缓冲；按租户配额

### 9.5 降级策略

- 实时流处理挂了也不丢数据（raw 先落对象存储/队列）

## 10. 安全

### 10.1 设备身份

- 唯一密钥/证书；密钥轮换

### 10.2 传输

- TLS

### 10.3 权限

- 租户隔离（RBAC），API key / OAuth（对外集成）

### 10.4 审计

- 关键操作（OTA、配置下发、导出数据）可追踪

## 11. 典型容量估算

### 11.1 假设

- 10,000 devices
- 平稳期 1/min，上报包 200 bytes
- 事件期 1/sec（占比 1% 时间）

### 11.2 流量估算

- **平稳流量**：10k * 1/min ≈ 167 msg/s
- **事件峰值**：10k * 1/sec * 1% ≈ 100 msg/s（按更保守可上到数千 msg/s 设计）

### 11.3 存储策略

- 时序按 downsample（分钟/小时/天）多层保留
- raw 冷存对象存储

## 12. 可观测性

### 12.1 端到端链路追踪

- 设备→云→分析→告警全链路追踪
- 关键指标：TTD（Time To Detect）、数据延迟、缺失率

### 12.2 SLO 指标

- **数据延迟**：端侧采样到云端入库的延迟
- **缺失率**：按设备/站点统计数据缺失比例
- **告警响应时间**：异常事件到告警触发的延迟
