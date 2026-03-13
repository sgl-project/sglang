---
title: HydroSense Cloud Backend – System Design
date: 2026-03-12
author: Yanda Cheng
---

> 本系统采用 **“实时主链路 + 异步多模态分析 + 分层存储 + 按国家训练与发布模型”** 的总体架构。  
> 核心秒级传感器链路负责 24x7 高频数据接入与告警，摄像头及政府/公共数据作为辅助模态接入，用于误报抑制、事件验证与区域化模型增强。

---

### 1. 背景与规模假设

HydroSense 部署了大量 **摄像头与环境传感器**（雨量计、流量计、水位计等），分布在多个城市和国家，长期 24x7 运行。典型规模假设：

- 单城市：数百到数千台设备  
- 单国家：**1 万–10 万台设备**  
- 采样频率：
  - 关键传感器：**1 秒 1 条**（秒级遥测）
  - 一般传感器：10–60 秒 1 条
  - 摄像头：事件驱动 snapshot / clip，上报频率远低于传感器
- 粗略吞吐量（单国家）：
  - 核心秒级遥测：**10⁴–10⁵ events/s 级别**

在这个量级下，系统必须支持：

- 高并发设备接入、稳定的流式处理与存储写入
- 按城市/国家划分的多 Region 部署与数据主权控制
- 长期历史数据保存与多源数据训练的 ML pipeline

---

### 2. 总体架构概览

整体上分为以下几个层次：

- **Device Layer（设备层）**
  - 传感器 + 摄像头，通过网关或本地边缘节点接入互联网。
  - 使用 MQTT（或 MQTT over WebSocket）/ HTTPS 与云端通信，协议尽量轻量、低功耗。

- **Edge / Gateway Layer（边缘层）**
  - 本地协议适配：RS-485、Modbus、自研串口协议等。
  - 轻量预处理：简单去噪、聚合（例如 1 秒内多次采样取均值）、压缩。
  - 断网缓存：网络中断时在网关本地落盘缓存，恢复后按顺序批量重传。

- **Cloud Ingress（云端接入层）**
  - API Gateway + Ingestion Service：
    - 支持 MQTT over WebSocket / HTTPS REST。
    - JWT / API Key / 设备凭证认证；绑定 `tenant_id`、`device_id`、`gateway_id`。
    - 限流（rate limiting）、IP 黑名单、基础 WAF。

- **Streaming & Processing Layer（流处理层）**
  - 消息总线：Kafka / Pulsar（或云厂商托管等价服务）。
  - 三层处理：
    1. **硬实时规则**：简单阈值/心跳检测，延迟目标秒级以内。
    2. **窗口统计与聚合**：rolling mean/variance、event count、趋势分数等。
    3. **异步复杂模型与多模态关联**：通过独立推理服务完成，不阻塞主链路。

- **Storage Layer（存储层）**
  - 热数据：TSDB（如 TimescaleDB / InfluxDB）保存秒级/分钟级遥测。
  - 冷数据：对象存储（S3/GCS/Azure Blob）保存长周期原始/聚合数据、事件快照。
  - 配置/元数据：关系型 DB（PostgreSQL/MySQL）。

- **Analytics & ML Layer（分析与模型层）**
  - 批处理任务（Airflow / Prefect + Python）。
  - 多源数据清洗、特征工程、训练/重训、评估与发布。

- **Serving & API Layer（对外服务层）**
  - REST / gRPC API、Web 控制台。
  - 多租户仪表盘、告警中心、报表导出。

- **Alerting & Notification（告警与通知）**
  - 订阅告警 Topic，发送短信/邮件/Webhook/IM 通知。
  - 支持降噪（抑制规则、合并告警、冷却时间）。

---

### 3. 数据类型与频率分层

为了在成本、实时性与可分析性之间平衡，数据按频率与用途分为四层：

- **Layer 1：实时秒级遥测**
  - 内容：雨量、流量、水位、设备状态、心跳等关键指标。
  - 频率：1s/1 条（或更高），直接驱动实时告警与主监控链路。
  - 特点：高吞吐、低延迟、仅保留有限时间。

- **Layer 2：分钟级聚合数据**
  - 内容：rolling mean / max / min / variance、事件计数、状态切换次数等。
  - 频率：1 分钟或 5 分钟聚合一次。
  - 特点：适合 TSDB 查询，支撑仪表盘与运营分析，可保留更久。

- **Layer 3：低频业务与环境数据**
  - 内容：日报/月报、站点维护记录、人工巡检记录、外部天气/区域信息等。
  - 频率：小时级、日级、月级。
  - 特点：主要用于离线分析、报表与模型训练，不在实时链路中。

- **Layer 4：视频/图像模态**
  - 内容：camera snapshot、事件短视频、边缘推理结果（检测框、分割 mask、embedding 等）。
  - 特点：大对象、写入成本高；与传感器事件异步关联，通过元数据与时间戳/空间位置进行 join。

---

### 4. 数据完整性与重传处理

考虑到设备与网关存在断网与批量重传场景，需要对 **幂等性、去重与乱序** 做明确设计。

- **消息字段设计**
  - `device_id`：设备唯一标识。
  - `gateway_id`：网关标识。
  - `tenant_id`：租户标识（城市/客户）。
  - `sequence_id`：设备侧单调递增序号（或 `(gateway_id, local_seq)`）。
  - `event_time`：设备生成该条观测的业务时间。
  - `ingest_time`：云端接收到消息的时间。
  - `message_id`：由设备/网关生成的 UUID（可选）。

- **去重策略**
  - 接入层或流处理层以 `(device_id, sequence_id)` 或 `message_id` 作为幂等键。
  - 最近时间窗口内维护去重缓存（如 window cache 或 compacted topic）。
  - 写入存储时二次校验（尤其是关键计费/法定数据）。

- **乱序与迟到数据处理**
  - 以 `event_time` 作为主要排序字段。
  - 在流处理层设定 watermark 与最大乱序时间（如 5–10 分钟）。
  - 对超出窗口的迟到数据标记 `late_data_flag`，并单独下游处理（例如仅存档不参与实时告警）。

- **批量重传与限流**
  - 网关断网恢复后允许批量 replay 历史数据。
  - replay 流与实时流在 ingress 或 topic 层做限流/隔离，避免冲击实时告警链路。
  - 可将 replay 流标记 `replay=true`，下游选择不同优先级与处理逻辑。

---

### 5. 多地区部署与异地灾备

- **Region 设计**
  - 按大洲或国家划分 Region（如 `ap-southeast-1`, `eu-central-1`）。
  - 每个 Region 拥有独立的 Ingress + Streaming + Storage 堆栈，满足数据主权与合规要求。

- **跨 Region 复制**
  - 关键元数据与聚合时序数据异步复制到 DR Region。
  - 对象存储开启跨 Region 复制（CRR），保证长期归档多副本。

- **故障切换**
  - 使用 DNS / Global Load Balancer 在 Region 故障时引导新连接到 DR Region。
  - 一般采用「本地写入为主，跨 Region 只读或延迟写入」策略；必要时允许降级模式（仅告警、不做复杂分析）。

---

### 6. 安全与权限模型

- **设备与网关认证**
  - 设备与网关具有独立身份标识，可使用 per-device certificate 或预共享凭证。
  - 初始激活与授权可结合 CD key/序列号体系；更细粒度的密钥签发与轮换由设备安全体系负责。

- **设备到云**
  - 全链路 TLS 加密（HTTPS / MQTT over TLS）。
  - 仅暴露有限的 Ingestion 端点，最小权限原则。

- **云内服务间通信**
  - Service Mesh（如 Istio/Linkerd）或 mTLS，统一证书与访问策略管理。
  - RBAC 控制各服务可访问的表、Topic 与 API。

- **租户隔离**
  - 所有消息带 `tenant_id`，存储按 `(tenant_id, region)` 分区。
  - API 层基于用户身份和 `tenant_id` 做行级权限控制。
  - 审计日志记录所有跨租户访问尝试与高敏操作。

---

### 7. 存储与保留策略（Retention Policy）

为平衡成本、性能与可追溯性，采用分层保留：

- **秒级原始遥测（Raw second-level telemetry）**
  - 保留时间：**约 1 个月**。
  - 用途：实时排障、短期回放、近期模型调试。

- **分钟级聚合数据（Minute-level aggregates）**
  - 保留时间：**12 个月**。
  - 用途：仪表盘查询、中长期趋势分析、运营复盘、合规报表。

- **日级及更高粒度数据（Daily and higher-level aggregates）**
  - 保留时间：长期。
  - 用途：长期运营分析、监管留存、历史建模与对比实验。

- **临时中间结果（Intermediate artifacts）**
  - 保留时间：约 **两个季度**，到期自动清理。
  - 用途：短期实验、迁移验证与 debug。

实际部署时，各 Region 可根据本地法规与客户合同微调 retention 策略。

---

### 8. 数据模型（高层）

- **Device 表**
  - 字段：`device_id`，`tenant_id`，`type`（sensor/camera/...），`location`（city/country/GPS），`gateway_id`，`status`，`installed_at`，`firmware_version`。

- **Telemetry 表（或 TSDB 视图）**
  - 字段：`time`，`device_id`，`metric_type`（temperature/flow/...），`value`，`quality_flag`，`replay_flag`，`late_data_flag`。

- **Alert 表**
  - 字段：`alert_id`，`tenant_id`，`device_id`，`severity`，`rule_id`，`trigger_time`，`resolved_time`，`status`（open/ack/closed），`source`（rule/model/manual）。

- **Model Version 表**
  - 字段：`model_name`，`version`，`region`，`train_dataset_id`，`metrics_json`，`deploy_status`（canary/active/rolled_back），`created_at`，`deployed_at`。

---

### 9. Post-training Evaluation 流程与 Ground Truth

- **Ground Truth 来源**
  - 人工标注数据（例如现场巡检记录、事件复盘报告）。
  - 历史设备或旧版本系统的稳定输出（作为对比基线）。
  - 多模态交叉验证结果：设备观测 + 摄像头图像 + 政府事件记录。

- **区域漂移与区域化模型**
  - 不同国家/地区在降雨模式、地形、城市排水系统上差异巨大。
  - 模型训练与评估以 **国家/区域** 为基本单元，避免“一个模型打天下”。

- **评估与上线流程**
  1. 从对象存储/TSDB 抽取代表性数据（覆盖多个城市、多种设备类型）。
  2. 用 candidate 模型在历史数据上离线回放，记录预测与真实结果。
  3. 计算误差分布、召回率/精确率、提前预警时间、误报/漏报率等指标。
  4. 与当前线上模型对比，若在关键指标上显著优于 baseline 且无不可接受 trade-off，则标记为可推广。
  5. 通过配置中心/API 进行 **canary 发布**（小部分租户/设备），实时监控线上指标。
  6. 若线上表现劣化超阈值，则 **自动回滚** 到旧版本，并记录触发条件与诊断信息。

---

### 10. 端到端预警链路示例

1. 某城市一批传感器检测到雨量/流量异常，数据通过网关发送至云端 Ingestion API。  
2. 流处理层实时计算窗口特征（近期涨幅、超阈值持续时间等），并调用在线模型判断是否为「潜在洪涝风险」事件。  
3. 告警服务接收“高优先级告警”事件，查找该租户的通知策略（SMS + 邮件 + Webhook）。  
4. 向客户运维团队推送告警，同时在控制台将该城市/设备标红，并附上最近时序曲线与模型解释（关键特征、风险评分等）。  
5. 所有告警记录写入 Alert 表，后续用于事件复盘与模型再训练。  

---

### 11. 多层 ML 系统与外部数据融合

HydroSense 的核心差异化在于：不仅使用自身设备数据，还系统性引入 **多政府部门与公共数据源**，形成多层次的交叉验证与融合建模，使模型效果和稳定性 **显著优于仅依赖单一数据源的竞品**。

- **数据来源层次**
  - 设备数据：摄像头、雨量计、流量计、水位计等高频时序与图像数据。
  - 气象部门：降雨量预报、雷达回波、温湿度、风向风速、台风路径等。
  - 水利/海洋部门：河道/水库实时水位、闸门开度、潮位、海浪高度等。
  - 城建/市政：排水管网设计、地形/地势、道路易涝点分布、历史积水记录等。

- **特征与标签构造**
  - 在 **时间 + 空间（经纬度/行政区域/流域）** 维度对齐多源数据，构建统一的时空网格。
  - 设备数据作为局地高分辨率观测，政府数据提供宏观背景与边界条件。
  - 标签包含：已确认的内涝/积水事件、泵站超载、道路封闭、险情等级等。

- **多层 ML 架构**
  1. **局地模型（Local Models）**  
     针对单站点/小区域短期预测（0–2 小时降雨–水位响应），捕捉本地设备与地形的细节。
  2. **区域模型（Regional Models）**  
     以城市/流域为粒度的中短期风险预测（2–24 小时内涝/洪涝概率），依赖气象、水利等宏观数据。
  3. **融合模型（Fusion / Meta-Model）**  
     以局地与区域模型输出为输入，再结合实时观测与历史事件记录，给出综合风险评分与优先级排序。

- **交叉验证与一致性检查**
  - 使用政府部门 **官方观测与事件记录** 作为“第三方标签”对自有模型交叉验证。
  - 上线前要求在召回率、误报率、提前预警时间等维度 **同时优于单源 baseline**。
  - 对模型输出分歧较大的区域标记为“高不确定性区域”，优先安排线下巡查与人工复核。

- **对业务效果的影响**
  - 在官方观测稀疏区域，通过自有设备构建高分辨率场景，**显著减少漏报**。
  - 设备故障/掉线时，区域模型与历史模式提供 **容错与插补**，避免监控盲区。
  - 面向政府/大企业客户，可给出“设备观测 + 官方数据 + 历史事件”的联合证据链，提高预警可信度。

---

### 12. 开发阶段与时间预估（粗略）

假设团队规模约为 **4–6 名工程师 + 1 名数据/ML 工程师**，并有产品/项目支持，给出一个保守但可落地的时间预估：

- **Phase 0 – 需求澄清与 MVP 范围（2–3 周）**
  - 明确首批接入城市/国家、设备类型、并发规模。
  - 定义必须支持的预警类型与 SLA（例如预警延迟 < 1 分钟）。
  - 梳理合规与数据主权约束（哪些数据不能跨境/必须本地存储）。

- **Phase 1 – 设备接入与基础云端链路（6–8 周）**
  - 搭建 API Gateway + Ingestion Service（MQTT/HTTPS）。
  - 实现设备认证、租户标识、基础限流与监控。
  - 部署单 Region 的消息队列与 TSDB，支持基础仪表盘。
  - 完成少量试点城市/站点接入与稳定性验证。

- **Phase 2 – 告警链路与控制台（6–8 周）**
  - 搭建流处理作业，实现简单规则告警与基础窗口统计。
  - 实现告警服务与通知集成（邮件/SMS/Webhook）。
  - 上线多租户 Web 控制台与 Alert 生命周期管理。

- **Phase 3 – 存储分层与异地备份（4–6 周）**
  - 完成热/冷数据拆分与生命周期策略设置。
  - 实现跨 Region 复制与基础 DR 演练。
  - 完善权限模型与审计日志。

- **Phase 4 – 多层 ML 系统初版（8–12 周）**
  - 收集、清洗并对齐多源政府数据（气象、水利、海洋、城建等）。
  - 建立 Local / Regional / Fusion 模型的初版训练与部署 pipeline。
  - 引入 post-training evaluation 与 canary 发布机制。
  - 在 1–2 个重点城市做 A/B 测试，与竞品/现有方案对比提前量、精度、误报率等。

- **Phase 5 – 优化与规模化推广（持续迭代，3–6 个月）**
  - 持续优化模型与规则引擎，扩展更多城市与国家。
  - 打磨控制台体验、报表与对外 API，支持合作伙伴集成。
  - 针对政府/大企业客户需求提供专有云/本地化部署方案。

整体来看，**从零到一个支持多地区和多源数据融合的生产系统，大致需要 9–12 个月**；若只做「单 Region + 基础告警」的 MVP，约 **3–4 个月** 可交付首版并开始真实数据上的迭代。

---

### 13. 简历与面试角度可强调的要点（备注）

在对外讲述 HydroSense 经验时，可以强调：

- 你在该系统中主要负责：
  - **云端数据与 ML pipeline 的需求分析和系统方案设计**；
  - 多政府部门/多数据源（气象、水利、海洋、市政）数据接入与特征工程设计；
  - 多层 ML 评估与 post-training evaluation 的设计与落地，使模型效果 **显著优于只用单一数据源的竞品**。
- 讲解路径可以是：**总体架构 → 数据分层 → 流处理与告警 → 多源数据与模型 → 实际业务指标与对比结果**，让面试官看到你既有工程思维，又有建模与业务抽象能力。

---
title: HydroSense Cloud Backend – High-Level System Design
date: 2026-03-12
author: Yanda Cheng
---

### 1. 背景与目标

HydroSense 部署了大量 **摄像头与环境传感器**，分布在多个城市和国家。系统需要：

- **实时数据采集与同步**：秒级/分钟级上报，支持高并发设备接入。
- **端到端预警**：从设备侧异常 → 云端检测 → 通知客户（短信/邮件/控制面板）。
- **异地容灾与数据长期保存**：满足监管与业务分析需要。
- **安全传输与权限隔离**：设备到云、云到客户均需加密与租户隔离。

### 2. 总体架构概览

- **Device Layer（设备层）**
  - 传感器 + 摄像头，通过网关或本地边缘节点接入互联网。
  - 使用 MQTT/HTTPS 与云端通信，尽量轻量、低功耗。

- **Edge / Gateway Layer（边缘层）**
  - 负责本地协议适配（RS-485、Modbus、私有协议等）和简单预处理（去噪、聚合、缓存）。
  - 断网时本地缓存，恢复后批量重传，保证数据完整性。

- **Cloud Ingress（云端接入层）**
  - API Gateway + Ingestion Service：
    - 支持 MQTT over WebSocket / HTTPS REST。
    - JWT / API Key 认证，设备分组与租户（tenant）标识。
    - 限流（rate limiting）和基本防护（IP 黑名单、WAF）。

- **Streaming & Processing Layer（流处理层）**
  - 消息总线：Kafka / Pulsar（或云厂商的托管等价服务）。
  - 流处理：Flink / Kafka Streams / Spark Streaming 用于：
    - 实时规则检测（阈值、模式识别）。
    - 在线特征计算（滑动窗口均值、方差、频域特征等）。
    - 触发预警事件（写入告警 Topic）。

- **Storage Layer（存储层）**
  - **热数据（Hot）**：TSDB（如 TimescaleDB / InfluxDB）存储最近 30–90 天时序数据，支撑仪表盘与告警查询。
  - **冷数据（Cold / Archive）**：对象存储（S3/GCS/Azure Blob）按分区（tenant/region/year/month）归档原始数据与压缩后的聚合结果。
  - **配置与元数据**：关系型数据库（PostgreSQL/MySQL）存储：
    - 设备与网关清单、地理位置、租户信息。
    - 告警策略、通知渠道配置。
    - 模型版本与 A/B 测试配置。

- **Analytics & ML Layer（分析与模型层）**
  - 批处理/离线任务（Airflow / Prefect + Python）：
    - 日/周度数据聚合与 KPI 计算。
    - 训练/重训预测模型（漂移检测、健康评估、预警模型等）。
  - **Post-training Evaluation**：
    - 在对象存储中的历史数据上对新模型进行离线回测。
    - 比较新旧模型的误差分布、召回率、误报率等。
    - 将结果写回元数据 DB，并在控制台展示。

- **Serving & API Layer（对外服务层）**
  - REST / gRPC API：
    - 实时读取设备最新状态与历史曲线。
    - 配置告警阈值、通知频道、设备分组。
    - 查询模型版本、评估报告。
  - 多租户控制台（Web UI）：
    - 仪表盘：设备在线率、告警情况、历史趋势。
    - 告警中心：当前告警、处理记录、SLA 指标。
    - 报表导出：CSV/PDF 周报、月报。

- **Alerting & Notification（告警与通知）**
  - 告警服务订阅「告警 Topic」：
    - 支持规则引擎（threshold / anomaly / ML-based）。
    - 与邮件、SMS、Webhook、Slack/Teams 集成。
    - 支持抑制策略（告警合并、冷却时间）。

### 3. 多地区部署与异地灾备

- **Region 设计**
  - 按大洲或国家划分 Region（如 `ap-southeast-1`, `eu-central-1`）。
  - 每个 Region 拥有独立的 Ingress + Streaming + Storage 堆栈，满足数据主权要求。

- **跨 Region 复制**
  - 元数据和关键时序数据通过异步复制到 DR Region。
  - 对象存储配置跨 Region 复制（CRR），保证长期归档的多副本。

- **故障切换**
  - 使用 DNS / Global Load Balancer 在 Region 故障时引导新连接到 DR Region。
  - 允许「读多写一」策略：正常情况下就近写入本 Region，跨 Region 只读或延迟写入。

### 4. 安全与权限模型

- **设备到云**
  - TLS 加密（HTTPS / MQTT over TLS）。
  - 设备证书或预共享密钥（PSK），定期轮换。
  - 设备只允许访问有限的 Ingestion 端点，最小权限。

- **云内服务间通信**
  - Service Mesh（如 Istio/Linkerd）或 mTLS，统一策略控制。
  - RBAC（基于角色的访问控制），不同服务只可访问必须的数据表和 Topic。

- **租户隔离**
  - 每条消息均带 tenant ID，数据存储按 tenant + region 分区。
  - API 层基于用户身份和 tenant 绑定进行行级过滤。
  - 日志与审计记录所有跨 tenant 访问尝试。

### 5. 数据模型（高层）

- **Device 表**
  - `device_id`，`tenant_id`，`type`（sensor/camera/...），`location`（city/country/GPS），`gateway_id`，`status`。
- **Telemetry 表（热数据 TSDB 视图）**
  - `time`，`device_id`，`metric_type`（temperature/flow/...），`value`，`quality_flag`。
- **Alert 表**
  - `alert_id`，`tenant_id`，`device_id`，`severity`，`rule_id`，`trigger_time`，`status`（open/ack/closed）。
- **Model Version 表**
  - `model_name`，`version`，`train_dataset_id`，`metrics_json`，`deploy_status`（canary/active/rolled_back）。

### 6. Post-training Evaluation 流程

1. **抽取数据**：从对象存储/TSDB 拉取一段时间内的代表性数据（多个城市、多种设备类型）。
2. **离线回放**：用 candidate 模型对历史数据做推理，记录预测结果与真实结果。
3. **指标计算**：误差分布、召回/精确率、告警提前量（提前发现异常的时间窗口）、误报/漏报率。
4. **对比与阈值判断**：与当前生产模型对比，如果满足阈值（或在某些 tenant 上更优）则标记可推广。
5. **灰度发布**：通过配置中心/API 将部分租户或设备切换到新模型（canary），监控线上指标。
6. **自动回滚**：若线上指标劣化超过阈值，通过配置 API 自动回滚至旧版本。

### 7. 端到端预警链路示例

1. 某城市的一批传感器检测到雨量/流量异常，数据通过网关发送至云端 Ingestion API。
2. 流处理层实时计算窗口特征，并调用在线模型判断是否为「潜在洪涝风险」事件。
3. 告警服务接收「高优先级告警」事件，查找该租户配置的通知策略（SMS + 邮件 + Webhook）。
4. 向客户运维团队推送告警，同时在控制台将该城市/设备标红，并附带最近的时序曲线与模型解释。
5. 所有告警记录写入 Alert 表，便于后续复盘与模型再训练。

### 8. 后续可以扩展的方向

- 加入 **多云/混合云部署**，部分 Region 使用本地机房作为边缘节点。
- 引入 **权限更细粒度的多层租户**（集团/子公司/站点三级结构）。
- 扩展摄像头视频流到 **边缘推理 + 云端索引** 的多模态架构（视频+传感器联合异常检测）。

