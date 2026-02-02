# A09: Rad-Linter System Design Diagram
# Rad-Linter 系统设计图

**作者**：Yanda Cheng  
**项目**：Rad-Linter  
**创建日期**：2025-01  
**设计风格**：参考 KYC 项目 + 工业级架构设计

---

## 📋 目录

1. [系统设计概览](#系统设计概览)
2. [完整系统架构图](#完整系统架构图)
3. [核心组件详解](#核心组件详解)
4. [数据流与交互](#数据流与交互)
5. [关键设计决策](#关键设计决策)
6. [性能与扩展性](#性能与扩展性)

---

## 系统设计概览

### 核心定位

Rad-Linter 是一个**医学影像报告质量检查系统**，在医生签字前进行跨模态一致性检查，确保报告与影像证据的一致性。

### 设计原则

1. **证据优先**：从"生成式"到"核验式"，先提取结构化证据，再基于证据判断
2. **可审计性**：所有结论可追溯，支持完全重放
3. **本地部署**：On-Prem GPU 资源管理，满足医院数据安全要求
4. **人工闭环**：AI 辅助而非替代医生，提供证据和上下文
5. **工业级稳定性**：Safety、Automation Rate、Latency、Auditability 四个维度严格达标

### 核心 KPI

- **Safety**：高风险错误漏检率 < 1%
- **Automation Rate**：自动放行率 > 85%
- **Latency**：P95 < 8 秒，P99 < 15 秒
- **Auditability**：100% 可追溯，100% 可重放

---

## 完整系统架构图

### 7层架构设计（On-Prem 部署）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Rad-Linter System (On-Prem)                         │
│                    Trigger: New Medical Report Arrives (R0)                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 0: Ingestion / Event（数据接入层）                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Report System (RIS/Dictation)                                           │ │
│ │ • HL7/FHIR / Internal API                                               │ │
│ └───────────────────────┬─────────────────────────────────────────────────┘ │
│                         │                                                    │
│                         ▼                                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Ingestor (Idempotent)                                                    │ │
│ │ • case_id 生成（UUID v4）                                                │ │
│ │ • 原始输入快照（snapshot raw input）                                      │ │
│ │ • 幂等性保证（deduplication）                                            │ │
│ │ • 队列化（q_realtime / q_batch）                                         │ │
│ └───────────────────────┬─────────────────────────────────────────────────┘ │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Precheck / Schema Gate（预检查/契约层）                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Contract Validator                                                       │ │
│ │ • 解析报告文本（parse report text）                                      │ │
│ │ • PII 脱敏（if needed）                                                  │ │
│ │ • CaseInputV1 Schema 验证（Pydantic/JSON Schema）                         │ │
│ │ • 快速失败（fast fail on invalid input）                                  │ │
│ │ • 执行时间：50-100ms                                                     │ │
│ └───────────────────────┬─────────────────────────────────────────────────┘ │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          │ pass
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Evidence Builder（证据构建层）⭐ 核心护城河                        │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Evidence Orchestrator                                                    │ │
│ │ • 获取历史检查（fetch prior exams, optional）                           │ │
│ │ • 获取影像引用（fetch imaging refs, DICOM）                              │ │
│ │ • 检查 FactStore 缓存（check cache）                                     │ │
│ └───────┬───────────────────────────────┬─────────────────────────────────┘ │
│         │                               │                                   │
│         ▼                               ▼                                   │
│ ┌───────────────────────┐   ┌─────────────────────────────────────────────┐ │
│ │ (2a) Text Facts       │   │ (2b) Visual Facts                           │ │
│ │ CPU Path              │   │ GPU/CPU Path                                 │ │
│ │ ┌─────────────────┐   │   │ ┌───────────────────────────────────────┐   │ │
│ │ │ NLP Extractor   │   │   │ │ CV / Measurement Pipeline             │   │ │
│ │ │ • entities      │   │   │ │ • detect/segment/measure              │   │ │
│ │ │ • relations     │   │   │ │ • laterality/location                  │   │ │
│ │ │ • negation      │   │   │ │ • evidence_refs (slices)              │   │ │
│ │ │ • uncertainty   │   │   │ └───────────────────────────────────────┘   │ │
│ │ │ • span offsets  │   │   │                                             │ │
│ │ └─────────────────┘   │   │                                             │ │
│ │ Output:               │   │ Output:                                     │ │
│ │ report_facts_v*        │   │ visual_facts_v*                             │ │
│ └───────────────────────┘   └─────────────────────────────────────────────┘ │
│         │                               │                                   │
│         └───────────────┬───────────────┘                                   │
│                         ▼                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Fact Store (Versioned)                                                    │ │
│ │ • report_facts_v*（版本化文本事实）                                       │ │
│ │ • visual_facts_v*（版本化视觉事实）                                       │ │
│ │ • hashes + provenance（哈希和溯源）                                       │ │
│ │ • Cache Management（缓存管理）                                            │ │
│ └───────────────────────┬─────────────────────────────────────────────────┘ │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Rule Gate（规则闸门层）                                            │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Rule Engine (Deterministic, Fast)                                        │ │
│ │ • Laterality mismatch（左右侧不一致）                                     │ │
│ │ • Measurement/unit conflicts（测量值/单位冲突）                           │ │
│ │ • Required fields missing（必填字段缺失）                                 │ │
│ │ • Template consistency（模板一致性）                                      │ │
│ │ • 执行时间：10-50ms                                                      │ │
│ └───────┬───────────────────────┬─────────────────────────────────────────┘ │
│         │                       │                                             │
│    Hard Fail                    │ Soft Flag / Pass                            │
│    (直接拦截)                   │ (进入下一层)                                │
│         │                       │                                             │
│         ▼                       ▼                                             │
└─────────┼───────────────────────┼─────────────────────────────────────────────┘
          │                       │
          │                       ▼
          │         ┌─────────────────────────────────────────────────────────┐
          │         │ Layer 4: LLM Judge（模型判决层）- GPU                    │
          │         │ ┌─────────────────────────────────────────────────────┐ │
          │         │ │ Request Router (Heterogeneous GPU Management)        │ │
          │         │ │ • 实时采集 Worker 指标                                │ │
          │         │ │   - inflight（当前处理请求数）                        │ │
          │         │ │   - VRAM（剩余显存）                                  │ │
          │         │ │   - TTFT（Time To First Token）                      │ │
          │         │ │   - P95（最近 P95 延迟）                             │ │
          │         │ │   - OOM（OOM 次数）                                  │ │
          │         │ │ • 选择最佳 Worker（Shortest-Estimated-Completion）   │ │
          │         │ │ • 熔断/降级机制                                       │ │
          │         │ └───────────────────────┬─────────────────────────────┘ │
          │         │                         │                               │
          │         │                         ▼                               │
          │         │ ┌─────────────────────────────────────────────────────┐ │
          │         │ │ LLM Worker(s) (SGLang/vLLM Replicas)                │ │
          │         │ │ • Schema-constrained JSON output                    │ │
          │         │ │ • Dynamic batching（动态批处理）                     │ │
          │         │ │ • Bounded retry on parse/low-conf（有限重试）        │ │
          │         │ │ • 执行时间：2-8s（主要延迟来源）                      │ │
          │         │ └───────────────────────┬─────────────────────────────┘ │
          │         └─────────────────────────┼───────────────────────────────┘
          │                                   │
          │                                   ▼
          └───────────────────────────────────┼───────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 5: Policy Gate（策略层）                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Policy / Triage                                                          │ │
│ │ • Severity: high/med/low（风险分级）                                     │ │
│ │ • Action: block/suggest/review（动作决策）                               │ │
│ │ • Fallback if LLM unavailable（降级策略）                                │ │
│ │ • 执行时间：10-50ms                                                      │ │
│ └───────────────────────┬─────────────────────────────────────────────────┘ │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 6: Human-in-the-loop（人工闭环层）                                    │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Review UI / Worklist                                                     │ │
│ │ • 显示问题和证据引用（show issues + evidence_refs）                       │ │
│ │ • 高亮报告文本位置（highlight report spans）                             │ │
│ │ • 收集医生反馈（accept/ignore + reason）                                 │ │
│ │ • 反馈回流（feedback loop）                                               │ │
│ └───────────────────────┬─────────────────────────────────────────────────┘ │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 7: Audit & Observability（审计与可观测层）                           │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Audit Log + Metrics + Replay                                             │ │
│ │ • Model/Prompt/Facts Versions（版本记录）                                 │ │
│ │ • TTFT/P95/Automation Rate（性能指标）                                   │ │
│ │ • Reproduce by case_id（按 case_id 完全重放）                            │ │
│ │ • Complete Trace（完整调用链追踪）                                        │ │
│ │ • Always-on（始终开启）                                                  │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

Output: LintResultV1
• issues[]（问题列表）
• severity（严重程度）
• evidence pointers（证据引用）
• recommended action（推荐动作）
• metadata（元数据：版本、时间戳等）
```

---

## 核心组件详解

### Layer 0: Ingestion（数据接入层）

**职责**：
- 接收来自报告系统的请求（RIS/Dictation）
- 生成唯一的 case_id（UUID v4）
- 保存原始输入快照（用于审计重放）
- 队列化管理（区分实时和批量队列）

**关键特性**：
- **幂等性**：同一报告多次接收只处理一次（基于 report_hash）
- **原始快照**：保存原始输入，支持后续审计
- **队列管理**：q_realtime（实时队列）、q_batch（批量队列）

**技术实现**：
- 消息队列：RabbitMQ / Redis Streams
- 幂等性：基于 report_hash 的分布式锁
- 快照存储：S3 / 本地文件系统

### Layer 1: Precheck / Schema Gate（预检查/契约层）

**职责**：
- 解析报告文本
- PII 脱敏（如需要）
- CaseInputV1 Schema 验证
- 快速失败（格式错误直接拒绝）

**关键特性**：
- **快速执行**：通常在 50-100ms 内完成
- **Schema-first**：严格校验，不符合 Schema 直接拒绝
- **PII 保护**：敏感信息脱敏处理

**技术实现**：
- Schema 验证：Pydantic / JSON Schema
- PII 脱敏：spaCy / Presidio

### Layer 2: Evidence Builder（证据构建层）⭐ **核心护城河**

**职责**：
- 获取历史检查（可选）
- 获取影像引用（DICOM）
- 检查 FactStore 缓存
- 提取文本和视觉证据

#### (2a) Text Facts（文本证据）- CPU 路径

**职责**：
- 实体识别（entities）
- 关系抽取（relations）
- 否定识别（negation）
- 不确定性识别（uncertainty）
- 文本位置定位（span offsets）

**输出**：
- `report_facts_v*`：结构化文本事实
- 每个事实包含：fact_id、span_ref、entity、laterality、location、attributes

**技术栈**：
- NLP 模型：spaCy / BioBERT / ClinicalBERT
- 关系抽取：基于规则 + 模型

#### (2b) Visual Facts（视觉证据）- GPU/CPU 路径

**职责**：
- 检测/分割（detect/segment）
- 测量（measure）
- 左右侧识别（laterality）
- 位置定位（location）
- 证据引用（evidence_refs：切片索引）

**输出**：
- `visual_facts_v*`：结构化视觉事实
- 每个事实包含：fact_id、type、laterality、location、attributes、evidence_refs

**技术栈**：
- CV 模型：TorchXRayVision / 自定义检测模型
- GPU 加速：CUDA / PyTorch

#### Fact Store（证据存储）

**职责**：
- 版本化存储证据（report_facts_v*、visual_facts_v*）
- 哈希和溯源信息（hashes + provenance）
- 缓存管理（减少重复计算）

**关键特性**：
- **版本化**：每个证据都有版本号
- **可追溯性**：记录证据的生成过程
- **缓存机制**：相似报告复用证据，减少计算

**技术实现**：
- 存储：S3 / 本地文件系统
- 版本管理：基于时间戳 + 哈希
- 缓存：Redis / 本地缓存

### Layer 3: Rule Gate（规则闸门层）

**职责**：
- 执行确定性规则检查
- 快速过滤明显错误
- 输出：Hard Fail / Soft Flag / Pass

**规则类型**：
- **Laterality mismatch**：左右侧不一致
- **Measurement/unit conflicts**：测量值/单位冲突
- **Required fields missing**：必填字段缺失
- **Template consistency**：模板一致性检查

**输出分类**：
- **Hard Fail**：直接报错（极高风险）
- **Soft Flag**：进入 LLM Judge 进一步判定
- **Pass**：无问题，自动放行

**关键特性**：
- **低成本**：规则检查通常在 10-50ms 内完成
- **确定性**：相同的输入产生相同的结果
- **可审计**：所有规则都有明确的 ID 和版本

**技术实现**：
- 规则引擎：自定义规则引擎 / Drools
- 规则配置：YAML / JSON

### Layer 4: LLM Judge（模型判决层）- GPU

**职责**：
- 处理 Soft Flag 案例
- 生成高质量标签
- 识别争议案例

#### Request Router（请求路由器）- 异构 GPU 管理

**职责**：
- 实时采集 Worker 指标
- 选择最佳 Worker（基于多个信号）
- 动态路由请求

**关键指标**：
- **inflight**：当前正在处理的请求数
- **VRAM**：剩余显存
- **TTFT**：Time To First Token
- **P95**：最近 P95 延迟
- **OOM**：OOM 次数

**路由策略**：
- 最短完成时间估算（Shortest-Estimated-Completion-Time）
- 熔断/降级机制

**技术实现**：
- 指标采集：Prometheus / 自定义指标
- 路由算法：加权轮询 / 最短队列

#### LLM Worker(s)（LLM 工作节点）- SGLang/vLLM 副本

**职责**：
- Schema-constrained JSON 输出
- Dynamic batching（动态批处理）
- Bounded retry on parse/low-conf（有限重试）

**关键特性**：
- **Schema-constrained**：输出符合预定义 Schema
- **Dynamic batching**：动态批处理提高吞吐
- **Bounded retry**：解析失败或低置信度时重试

**技术实现**：
- 推理框架：SGLang / vLLM
- Schema 约束：JSON Schema / Pydantic
- 批处理：动态批处理策略

### Layer 5: Policy Gate（策略层）

**职责**：
- 根据检查结果决定具体动作
- 风险分级（high/med/low）
- 动作决策（block/suggest/review）
- Fallback 机制（LLM 不可用时的降级策略）

**动作类型**：
- **Block**：阻止签字（高风险）
- **Suggest**：建议编辑（中低风险）
- **Review**：人工复核（不确定情况）

**关键特性**：
- **可配置**：策略可以通过配置文件调整
- **Fallback**：LLM 不可用时的降级策略
- **可追溯**：所有决策都有明确的策略依据

**技术实现**：
- 策略配置：YAML / JSON
- 决策引擎：自定义决策引擎

### Layer 6: Human-in-the-loop（人工闭环层）

**职责**：
- 显示问题和证据引用
- 高亮报告文本位置
- 收集医生反馈（accept/ignore + reason）

**界面功能**：
- **Issue Display**：显示发现的问题
- **Evidence Refs**：显示证据引用（visual_facts、report_facts）
- **Report Spans**：高亮报告文本位置
- **Action Buttons**：采纳/忽略按钮
- **Reason Field**：记录医生反馈原因

**关键特性**：
- **用户友好**：界面清晰，操作简单
- **信息完整**：提供所有必要的证据和上下文
- **反馈回流**：收集的反馈用于系统改进

**技术实现**：
- 前端框架：React / Vue
- 后端 API：RESTful API / GraphQL

### Layer 7: Audit & Observability（审计与可观测层）

**职责**：
- 记录所有关键操作的审计日志
- 收集性能和业务指标
- 支持按 case_id 完全重放

**记录内容**：
- **Model/Prompt/Facts Versions**：所有组件的版本号
- **TTFT/P95/Automation Rate**：性能指标
- **Reproduce by case_id**：支持完全重放

**关键特性**：
- **Always-on**：始终开启，不遗漏任何记录
- **Complete Trace**：完整的调用链追踪
- **Replay Capability**：支持按 case_id 完全重放

**技术实现**：
- 日志系统：ELK Stack / Loki
- 指标系统：Prometheus + Grafana
- 追踪系统：Jaeger / Zipkin

---

## 数据流与交互

### Main Data Flow Paths (Parallel Comparison)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Path 1: Auto-Pass              │ Path 2: Rule Block           │ Path 3: LLM Judge            │
│ (No Issues Found)              │ (Hard Fail)                  │ (Soft Flag)                  │
├────────────────────────────────┼──────────────────────────────┼─────────────────────────────┤
│                                │                               │                               │
│ R0: New Report Arrives         │ R0: New Report Arrives       │ R0: New Report Arrives       │
│         ↓                       │         ↓                     │         ↓                     │
│ Layer 0: Ingestion             │ Layer 0: Ingestion            │ Layer 0: Ingestion           │
│ • Generate case_id              │ • Generate case_id            │ • Generate case_id            │
│ • Snapshot raw input            │ • Snapshot raw input          │ • Snapshot raw input         │
│         ↓                       │         ↓                     │         ↓                     │
│ Layer 1: Schema Gate           │ Layer 1: Schema Gate         │ Layer 1: Schema Gate        │
│ • Validate CaseInputV1          │ • Validate CaseInputV1        │ • Validate CaseInputV1       │
│         ↓                       │         ↓                     │         ↓                     │
│ Layer 2: Evidence Builder      │ Layer 2: Evidence Builder     │ Layer 2: Evidence Builder    │
│ • Extract Text Facts (CPU)      │ • Text: "left effusion"      │ • Text: "mild effusion"      │
│ • Extract Visual Facts (GPU)    │ • Visual: "right detected"   │ • Visual: "moderate (0.75)" │
│         ↓                       │         ↓                     │         ↓                     │
│ Layer 3: Rule Gate             │ Layer 3: Rule Gate            │ Layer 3: Rule Gate           │
│ • Rule check                    │ • Laterality mismatch        │ • Low confidence case        │
│ • Output: Pass                  │ • Output: Hard Fail          │ • Output: Soft Flag           │
│         ↓                       │         ↓                     │         ↓                     │
│         │                       │         │                     │ Layer 4: LLM Judge           │
│         │                       │         │                     │ • Route to best worker       │
│         │                       │         │                     │ • Generate judgment          │
│         │                       │         │                     │         ↓                     │
│ Layer 5: Policy Gate           │ Layer 5: Policy Gate         │ Layer 5: Policy Gate         │
│ • Decision: Auto-approve        │ • Decision: Block sign-off   │ • Decision: Suggest edit    │
│         ↓                       │         ↓                     │         ↓                     │
│         │                       │ Layer 6: Human Review        │ Layer 6: Human Review        │
│         │                       │ • Show issues + evidence      │ • Show suggestions           │
│         │                       │         ↓                     │         ↓                     │
│ Layer 7: Audit & Observability │ Layer 7: Audit & Observability│ Layer 7: Audit & Observability│
│ • Log audit trail               │ • Log audit trail            │ • Log audit trail            │
│ • Update metrics                │ • Update metrics             │ • Update metrics             │
│         ↓                       │         ↓                     │         ↓                     │
│ Output: LintResultV1            │ Output: LintResultV1         │ Output: LintResultV1         │
│ status: pass                    │ status: fail                 │ status: flag                 │
│                                 │ action: block                │ action: suggest              │
└────────────────────────────────┴──────────────────────────────┴─────────────────────────────┘
```

### 关键数据结构

#### CaseInputV1（输入 Schema）

```json
{
  "case_id": "case_001",
  "report_text": "Patient presents with left pleural effusion...",
  "imaging_refs": {
    "dicom_series_id": "series_001",
    "study_date": "2025-01-01",
    "modality": "CXR"
  },
  "prior_exams": [
    {
      "exam_id": "exam_001",
      "date": "2024-12-01"
    }
  ],
  "metadata": {
    "patient_id": "patient_001",
    "exam_date": "2025-01-01",
    "radiologist_id": "radiologist_001"
  }
}
```

#### LintResultV1（输出 Schema）

```json
{
  "case_id": "case_001",
  "status": "pass" | "flag" | "fail",
  "items": [
    {
      "issue_type": "laterality_mismatch" | "omission" | "contradiction" | "...",
      "severity": "high" | "med" | "low",
      "supporting_facts": ["vf_001", "rf_002"],
      "report_spans": [{"start": 120, "end": 135}],
      "recommended_action": "block" | "suggest" | "review",
      "confidence": 0.95,
      "explanation": "Visual fact vf_001 shows left pleural effusion (confidence 0.95), but report fact rf_001 states 'no effusion' (span 120-135). This is a contradiction (negation conflict)."
    }
  ],
  "summary": {
    "total_issues": 2,
    "high_severity": 1,
    "med_severity": 1,
    "low_severity": 0
  },
  "metadata": {
    "model_version": "qwen2.5-vl-32b",
    "prompt_version": "v1.2",
    "schema_version": "v1.0",
    "fact_store_version": "v1.0",
    "rule_version": "v1.0"
  },
  "version": "v1.0",
  "timestamp": "2025-01-01T10:00:00Z"
}
```

#### Visual Facts（视觉事实）

```json
{
  "fact_id": "vf_001",
  "type": "effusion",
  "laterality": "left",
  "location": "pleural_space",
  "attributes": {
    "size": "large",
    "severity": "moderate",
    "confidence": 0.95
  },
  "evidence_refs": {
    "screenshot_index": "img_001",
    "mask_version": "v1.0",
    "measurement_source": "auto_detection"
  },
  "provenance": {
    "model_version": "torchxrayvision_v1.0",
    "extraction_timestamp": "2025-01-01T10:00:00Z"
  }
}
```

#### Report Facts（报告事实）

```json
{
  "fact_id": "rf_001",
  "span_ref": {
    "start": 120,
    "end": 135,
    "text": "left pleural effusion"
  },
  "entity": "effusion",
  "laterality": "left",
  "location": "pleural",
  "attributes": {
    "negation": false,
    "severity": "moderate"
  },
  "provenance": {
    "extractor_version": "nlp_extractor_v1.0",
    "extraction_timestamp": "2025-01-01T10:00:00Z"
  }
}
```

---

## 关键设计决策

### 1. 7层架构设计

**分层原则**：
- **Layer 0-1**：数据接入和验证（薄层，快速失败）
- **Layer 2**：证据构建（核心护城河，从"生成式"到"核验式"）
- **Layer 3**：规则闸门（低成本过滤，确定性规则）
- **Layer 4**：LLM 判决（GPU 推理，异构管理）
- **Layer 5**：策略决策（动作映射，可配置）
- **Layer 6**：人工闭环（反馈收集，AI 辅助）
- **Layer 7**：审计与可观测（始终开启，完整追溯）

**设计优势**：
- **职责单一**：每层职责清晰，边界明确
- **易于维护**：每层可以独立开发、测试、部署
- **可扩展性**：每层可以独立扩展
- **可测试性**：每层可以独立测试

### 2. 证据优先设计（核心护城河）

**设计理念**：
- 把任务从"生成式"变成"核验式"
- 先提取结构化证据（Visual Facts + Report Facts）
- 再基于证据进行判断（Rule Gate + LLM Judge）

**优势**：
- **可审计性**：所有结论都能追溯到原始证据
- **可解释性**：提供明确的证据引用
- **可复现性**：基于结构化证据，结果可复现
- **可测试性**：证据提取可以独立测试

### 3. 双层判决机制

**规则先行 + LLM 后判**：
- **Rule Gate**：低成本快速过滤（Hard Fail / Soft Flag / Pass）
- **LLM Judge**：处理复杂情况（Soft Flag 进一步判定）

**平衡**：
- **成本控制**：规则先行，减少 LLM 调用
- **准确性保证**：LLM 处理复杂情况，提高准确性

### 4. 异构 GPU 管理

**关键特性**：
- **Request Router**：智能路由请求到最佳 Worker
- **实时指标采集**：inflight/VRAM/TTFT/P95/OOM
- **动态调度**：根据实时负载选择最佳 Worker
- **熔断机制**：OOM 或性能下降时自动降权

**优势**：
- **资源利用率**：充分利用异构 GPU 资源
- **性能优化**：选择最优 Worker，减少延迟
- **容错性**：自动降级，提高系统稳定性

### 5. 可审计性优先

**设计原则**：
- **版本化**：所有组件都有版本号（模型、规则、Schema）
- **可追溯**：所有结论都能追溯到原始证据
- **可重放**：按 case_id 完全重放处理过程
- **Always-on**：审计和可观测层始终开启

**实现**：
- **版本管理**：所有组件版本化（model_version、prompt_version、rule_version）
- **完整链路**：从输入到输出的完整调用链都有 trace
- **不可变性**：输入和中间结果不可修改，只能追加

### 6. 人工闭环设计

**设计理念**：
- 不是"AI 替代医生"，而是"AI 辅助医生"
- 提供证据和上下文，让医生做最终决策
- 收集反馈，持续改进系统

**实现**：
- **Review UI**：清晰的界面，显示问题和证据
- **反馈收集**：记录医生的 accept/ignore 和 reason
- **反馈回流**：反馈用于系统改进（规则优化、模型训练）

---

## Key Trade-offs（关键权衡）

Rad-Linter 系统在设计过程中面临三个核心 trade-offs，这些权衡直接影响系统的四个核心 KPI（Safety、Automation Rate、Latency、Auditability）：

### Trade-off 1: Safety vs Automation Rate（安全性与自动化率的权衡）

**The Conflict**：
- **Higher Safety** (lower false negative rate) requires stricter rules and more aggressive blocking
- **Higher Automation Rate** requires fewer interruptions and more auto-approvals
- These two goals are fundamentally in tension

**Rad-Linter's Approach**：
- **Two-tier decision system**: Rule Gate (fast, deterministic) + LLM Judge (accurate, expensive)
- **Rule Gate filters obvious errors** (Hard Fail) → immediate block, maintains safety
- **LLM Judge handles edge cases** (Soft Flag) → nuanced judgment, preserves automation rate
- **Configurable thresholds**: Adjust rule strictness based on error budget

**Metrics**：
- Safety: High-risk error miss rate < 1%
- Automation Rate: Auto-approval rate > 85%
- **Balance point**: ~5% block rate, ~10% human review rate

**Why This Matters**：
- Hospital environment requires **zero tolerance for critical errors** (Safety is non-negotiable)
- But **excessive interruptions** destroy doctor trust and workflow efficiency
- The two-tier system allows **strict safety for critical cases** while **preserving automation for normal cases**

---

### Trade-off 2: Latency vs Accuracy（延迟与准确性的权衡）

**The Conflict**：
- **Lower Latency** requires simpler models, fewer LLM calls, faster processing
- **Higher Accuracy** requires complex models, more computation, LLM Judge for edge cases
- Medical reports need both: fast turnaround AND high accuracy

**Rad-Linter's Approach**：
- **Evidence-first architecture**: Extract structured facts first (parallel CPU/GPU paths)
- **Rule Gate as fast filter**: 10-50ms deterministic rules catch 80%+ of errors
- **LLM Judge only for edge cases**: ~20% of cases need LLM, reducing average latency
- **Heterogeneous GPU routing**: Smart worker selection minimizes LLM latency
- **Async processing**: Evidence Builder can run async, doesn't block main flow

**Metrics**：
- P50 < 3s (median), P95 < 8s (SLO), P99 < 15s (critical)
- LLM Judge adds 2-8s but only for ~20% of cases
- **Average latency**: ~4-6s (acceptable for pre-sign-off workflow)

**Why This Matters**：
- Doctors need **fast feedback** before sign-off (workflow integration)
- But **accuracy cannot be sacrificed** (patient safety)
- The architecture **optimizes for common case** (fast rule path) while **preserving accuracy** (LLM path for edge cases)

---

### Trade-off 3: Evidence Extraction Cost vs Auditability（证据提取成本与可审计性的权衡）

**The Conflict**：
- **Higher Auditability** requires extracting structured evidence (Visual Facts + Report Facts), versioning everything, storing full traces
- **Lower Cost** would skip evidence extraction, use simpler models, minimal logging
- Medical systems require **full auditability** (regulatory compliance) but have **limited compute budget** (on-prem GPU constraints)

**Rad-Linter's Approach**：
- **Evidence-first design**: Extract structured facts upfront (Visual Facts + Report Facts)
  - Cost: 1-3s latency, GPU/CPU resources
  - Benefit: 100% traceability, full replay capability
- **Versioned Fact Store**: All evidence versioned and cached
  - Cost: Storage overhead, cache management
  - Benefit: Reproducible results, audit trail
- **Complete trace logging**: Every decision logged with versions
  - Cost: Log storage, processing overhead
  - Benefit: Full auditability, regulatory compliance

**Metrics**：
- Auditability coverage: 100% (all conclusions traceable)
- Replay capability: 100% (all cases reproducible)
- **Evidence extraction cost**: 1-3s latency, but enables everything else

**Why This Matters**：
- **Regulatory requirement**: Medical systems MUST be auditable (FDA, HIPAA compliance)
- **Legal protection**: Full trace protects against liability claims
- **System improvement**: Evidence-based design enables continuous learning
- The **upfront cost** (evidence extraction) enables **downstream benefits** (auditability, explainability, improvement)

---

### Summary: How Rad-Linter Balances These Trade-offs

```
┌─────────────────────────────────────────────────────────────────┐
│                    Trade-off Balancing Strategy                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Trade-off 1: Safety vs Automation Rate                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Solution: Two-tier decision system                          │ │
│ │ • Rule Gate: Fast, strict for critical errors (Safety)      │ │
│ │ • LLM Judge: Accurate for edge cases (Automation Rate)       │ │
│ │ Result: ~95% automation rate, <1% error miss rate          │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ Trade-off 2: Latency vs Accuracy                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Solution: Evidence-first + smart routing                    │ │
│ │ • Rule Gate: Fast path (10-50ms) for 80%+ cases             │ │
│ │ • LLM Judge: Accurate path (2-8s) for 20% cases           │ │
│ │ • Async processing: Evidence Builder doesn't block           │ │
│ │ Result: P95 < 8s, high accuracy maintained                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ Trade-off 3: Cost vs Auditability                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Solution: Evidence-first architecture                        │ │
│ │ • Extract structured facts upfront (1-3s cost)              │ │
│ │ • Version everything (storage cost)                          │ │
│ │ • Complete trace logging (processing cost)                   │ │
│ │ Result: 100% auditability, regulatory compliance            │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ Key Insight:                                                    │
│ • Evidence extraction is the "foundation" that enables          │
│   all other optimizations (safety, accuracy, auditability)     │
│ • The upfront cost is justified by downstream benefits          │
│ • Two-tier system allows optimizing each path independently     │
└─────────────────────────────────────────────────────────────────┘
```

**Design Philosophy**：
1. **Evidence-first**: Extract structured facts upfront (costs 1-3s) but enables everything else
2. **Two-tier filtering**: Fast rules catch obvious errors, LLM handles edge cases
3. **Smart routing**: Optimize common case (fast path) while preserving accuracy (LLM path)
4. **Version everything**: Accept storage/logging costs for full auditability (regulatory requirement)

---

## 性能与扩展性

### 性能指标

#### 延迟分解

```
端到端延迟分解（P95）：
├─ Layer 0: Ingestion: 100-200ms
├─ Layer 1: Schema Gate: 50-100ms
├─ Layer 2: Evidence Builder: 1-3s
│  ├─ Visual Feature Extraction: 0.5-2s
│  └─ Report Parsing: 0.5-1s
├─ Layer 3: Rule Gate: 10-50ms
├─ Layer 4: LLM Judge: 2-8s（主要延迟来源）
│  ├─ API Call: 2-6s
│  ├─ Retry（如有）: 0-2s
│  └─ Fallback（如有）: 0-1s
├─ Layer 5: Policy Gate: 10-50ms
└─ Layer 7: Audit & Observability: 20-50ms

总延迟: ~4-12s (P95)
```

#### 目标指标

- **P50 < 3 秒**（中位数）
- **P95 < 8 秒**（SLO 目标）
- **P99 < 15 秒**（Critical 不能超过）
- **P99.9 < 30 秒**（极端情况）

### 扩展性设计

#### 水平扩展

1. **Layer 0: Ingestion**
   - 多实例部署，负载均衡
   - 消息队列支持分区

2. **Layer 2: Evidence Builder**
   - Text Facts：CPU 实例水平扩展
   - Visual Facts：GPU 实例水平扩展
   - Fact Store：分布式存储（S3）

3. **Layer 4: LLM Judge**
   - 多 Worker 部署，Request Router 智能路由
   - 异构 GPU 管理，充分利用资源

#### 垂直扩展

1. **GPU 资源**
   - 支持不同规格的 GPU（A100、H100、RTX 4090 等）
   - Request Router 根据 GPU 规格智能路由

2. **缓存策略**
   - Fact Store 缓存，减少重复计算
   - 相似报告复用证据

#### 异步处理

1. **Evidence Builder**
   - 可以异步执行，不阻塞主流程
   - 支持批量处理

2. **Audit & Observability**
   - 异步记录，不阻塞主流程
   - 批量写入，提高性能

### 容错与降级

#### 容错机制

1. **LLM Judge 不可用**
   - Fallback 到 Rule Gate 结果
   - 降级策略：只使用规则检查

2. **Evidence Builder 失败**
   - 使用缓存的事实
   - 降级策略：只使用文本事实

3. **Request Router 故障**
   - 降级到轮询策略
   - 熔断机制：自动降权故障 Worker

#### 降级策略

1. **性能降级**
   - LLM Judge 超时：使用 Rule Gate 结果
   - Evidence Builder 超时：使用缓存的事实

2. **功能降级**
   - LLM Judge 不可用：只使用规则检查
   - Visual Facts 不可用：只使用文本事实

---

## 总结

Rad-Linter 系统采用 7 层架构设计，核心特点：

1. **证据优先**：从"生成式"到"核验式"，先提取结构化证据，再基于证据判断
2. **双层判决**：规则先行 + LLM 后判，平衡成本控制和准确性
3. **异构 GPU 管理**：智能路由到最佳 Worker，充分利用资源
4. **可审计性**：版本化、可追溯、可重放，满足医院严格需求
5. **人工闭环**：AI 辅助而非替代医生，提供证据和上下文

**核心 KPI**：
- **Safety**：高风险错误漏检率 < 1%
- **Automation Rate**：自动放行率 > 85%
- **Latency**：P95 < 8 秒，P99 < 15 秒
- **Auditability**：100% 可追溯，100% 可重放

通过这套架构，Rad-Linter 能够满足医院的严格需求：**Safety、Automation Rate、Latency、Auditability**。
