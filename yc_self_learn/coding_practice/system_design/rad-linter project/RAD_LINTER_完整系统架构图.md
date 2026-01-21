# Rad-Linter 完整系统架构图
# Rad-Linter Complete System Architecture Diagram

**作者**：Yanda Cheng  
**项目**：Rad-Linter  
**架构版本**：v1.0  
**设计原则**：7层架构 + 工业级设计

---

## 📋 目录

1. [架构图](#架构图)
2. [组件说明](#组件说明)
3. [数据流说明](#数据流说明)
4. [关键设计点](#关键设计点)

---

## 架构图

### 完整系统架构图（On-Prem 部署）

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Rad-Linter (On-Prem)                         │
│               Trigger: New Medical Report arrives (R0)               │
└──────────────────────────────────────────────────────────────────────┘

(0) Ingestion / Event
┌───────────────┐     ┌─────────────────────────┐
│ Report System │ --> │ Ingestor (idempotent)   │
│ (RIS/Dictation)│     │ - case_id 생성          │
└───────────────┘     │ - snapshot(raw input)   │
                      │ - enqueue(q_realtime)    │
                      └───────────┬─────────────┘
                                  │
                                  v
(1) Precheck / Schema Gate (fast)
                      ┌──────────────────────────┐
                      │ Contract Validator        │
                      │ - parse report text       │
                      │ - PII scrub (if needed)   │
                      │ - CaseInputV1 validate    │
                      └───────────┬──────────────┘
                                  │ pass
                                  v
(2) Evidence Builder (async + cache)
     ┌───────────────────────────────┐
     │ Evidence Orchestrator          │
     │ - fetch prior exams (optional) │
     │ - fetch imaging refs (DICOM)   │
     │ - check FactStore cache        │
     └───────────┬───────────┬───────┘
                 │           │
     (2a) Text facts         │ (2b) Visual facts
     CPU path                │ GPU/CPU path
┌────────────────────┐       │      ┌───────────────────────────┐
│ NLP Extractor       │       │      │ CV / Measurement Pipeline │
│ - entities/relations│       │      │ - detect/segment/measure  │
│ - negation/uncert.  │       │      │ - laterality/location     │
│ - span offsets       │       │      │ - evidence_refs (slices)  │
└─────────┬──────────┘       │      └───────────┬───────────────┘
          │                   │                  │
          └──────────┬────────┴───────────┬──────┘
                     v                    v
                ┌────────────────────────────────┐
                │ Fact Store (versioned)          │
                │ - report_facts_v*               │
                │ - visual_facts_v*               │
                │ - hashes + provenance           │
                └───────────┬────────────────────┘
                            │
                            v
(3) Rule Gate (cheap, deterministic)
                ┌────────────────────────────────┐
                │ Rule Engine                     │
                │ - laterality mismatch           │
                │ - measurement/unit conflicts    │
                │ - required fields missing       │
                │ - hard fail vs soft flag        │
                └───────┬───────────┬───────────┘
                        │           │
                 hard fail          │ soft/uncertain
                        │           v
                        │   (4) LLM Judge (GPU)
                        │   ┌──────────────────────────────────────┐
                        │   │ Request Router (heterogeneous GPUs)   │
                        │   │ - reads: inflight/VRAM/TTFT/P95/OOM   │
                        │   │ - picks best worker                   │
                        │   └───────────────┬──────────────────────┘
                        │                   │
                        │                   v
                        │   ┌──────────────────────────────────────┐
                        │   │ LLM Worker(s) (SGLang/vLLM replicas)  │
                        │   │ - schema-constrained JSON output      │
                        │   │ - dynamic batching                    │
                        │   │ - bounded retry on parse/low-conf     │
                        │   └───────────────┬──────────────────────┘
                        │                   │
                        v                   v
(5) Policy Gate (decision & action)
                ┌────────────────────────────────┐
                │ Policy / Triage                │
                │ - severity: high/med/low       │
                │ - action: block/suggest/review │
                │ - fallback if LLM unavailable  │
                └───────────┬────────────────────┘
                            │
                            v
(6) Human-in-the-loop (optional but standard)
                ┌────────────────────────────────┐
                │ Review UI / Worklist           │
                │ - show issues + evidence_refs  │
                │ - highlight report spans       │
                │ - accept/ignore + reason       │
                └───────────┬────────────────────┘
                            │ feedback
                            v
(7) Audit & Observability (always-on)
                ┌────────────────────────────────┐
                │ Audit Log + Metrics + Replay    │
                │ - model/prompt/facts versions   │
                │ - TTFT/P95/automation rate      │
                │ - reproduce by case_id          │
                └────────────────────────────────┘

Outputs
- LintResultV1 (issues[], severity, evidence pointers, recommended action)
- Optional: annotated report suggestions (never overwrite original automatically)
```

---

## 组件说明

### (0) Ingestion / Event（数据接入层）

**职责**：
- 接收来自报告系统的请求
- 生成唯一的 case_id
- 保存原始输入快照（用于审计重放）
- 将请求加入实时队列（q_realtime）

**关键特性**：
- **幂等性**：同一报告多次接收只处理一次
- **原始快照**：保存原始输入，支持后续审计
- **队列管理**：区分实时和批量队列

### (1) Precheck / Schema Gate（预检查/契约层）

**职责**：
- 解析报告文本
- PII 脱敏（如需要）
- CaseInputV1 Schema 验证
- 快速失败（格式错误直接拒绝）

**关键特性**：
- **快速执行**：通常在 50-100ms 内完成
- **Schema-first**：严格校验，不符合 Schema 直接拒绝
- **PII 保护**：敏感信息脱敏处理

### (2) Evidence Builder（证据构建层）⭐ **核心护城河**

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

#### Fact Store（证据存储）

**职责**：
- 版本化存储证据（report_facts_v*、visual_facts_v*）
- 哈希和溯源信息（hashes + provenance）
- 缓存管理（减少重复计算）

**关键特性**：
- **版本化**：每个证据都有版本号
- **可追溯性**：记录证据的生成过程
- **缓存机制**：相似报告复用证据，减少计算

### (3) Rule Gate（规则闸门层）

**职责**：
- 执行确定性规则检查
- 快速过滤明显错误
- 输出：Hard Fail / Soft Flag

**规则类型**：
- **Laterality mismatch**：左右侧不一致
- **Measurement/unit conflicts**：测量值/单位冲突
- **Required fields missing**：必填字段缺失
- **Template consistency**：模板一致性检查

**输出分类**：
- **Hard Fail**：直接报错（极高风险）
- **Soft Flag**：进入 LLM Judge 进一步判定

**关键特性**：
- **低成本**：规则检查通常在 10-50ms 内完成
- **确定性**：相同的输入产生相同的结果
- **可审计**：所有规则都有明确的 ID 和版本

### (4) LLM Judge（模型判决层）- GPU

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

#### LLM Worker(s)（LLM 工作节点）- SGLang/vLLM 副本

**职责**：
- Schema-constrained JSON 输出
- Dynamic batching（动态批处理）
- Bounded retry on parse/low-conf（有限重试）

**关键特性**：
- **Schema-constrained**：输出符合预定义 Schema
- **Dynamic batching**：动态批处理提高吞吐
- **Bounded retry**：解析失败或低置信度时重试

### (5) Policy Gate（策略层）

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

### (6) Human-in-the-loop（人工闭环层）

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

### (7) Audit & Observability（审计与可观测层）

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

---

## 数据流说明

### 主要数据流路径

#### 路径 1：正常流程（无问题）

```
R0: New Report Arrives
  ↓
(0) Ingestion: case_id + snapshot
  ↓
(1) Schema Gate: CaseInputV1 validate
  ↓
(2) Evidence Builder: report_facts + visual_facts
  ↓
(3) Rule Gate: Pass (无规则违反)
  ↓
(5) Policy Gate: Auto-approve
  ↓
Output: LintResultV1 (status: pass)
```

#### 路径 2：规则拦截（Hard Fail）

```
R0: New Report Arrives
  ↓
(0) Ingestion: case_id + snapshot
  ↓
(1) Schema Gate: CaseInputV1 validate
  ↓
(2) Evidence Builder: report_facts + visual_facts
  ↓
(3) Rule Gate: Hard Fail (Laterality mismatch)
  ↓
(5) Policy Gate: Block sign-off
  ↓
(6) Human-in-the-loop: Review UI
  ↓
Output: LintResultV1 (status: fail, recommended_action: block)
```

#### 路径 3：LLM Judge 判定（Soft Flag）

```
R0: New Report Arrives
  ↓
(0) Ingestion: case_id + snapshot
  ↓
(1) Schema Gate: CaseInputV1 validate
  ↓
(2) Evidence Builder: report_facts + visual_facts
  ↓
(3) Rule Gate: Soft Flag (不确定情况)
  ↓
(4) LLM Judge:
    ├─ Request Router: 选择最佳 Worker
    └─ LLM Worker: 生成判定结果
  ↓
(5) Policy Gate: Suggest edit 或 Block sign-off
  ↓
(6) Human-in-the-loop: Review UI (可选)
  ↓
Output: LintResultV1 (status: flag, recommended_action: suggest/block)
```

---

## 关键设计点

### 1. 7层架构设计

**分层原则**：
- **Layer 0-1**：数据接入和验证（薄层）
- **Layer 2**：证据构建（核心护城河）
- **Layer 3**：规则闸门（低成本过滤）
- **Layer 4**：LLM 判决（GPU 推理）
- **Layer 5**：策略决策（动作映射）
- **Layer 6**：人工闭环（反馈收集）
- **Layer 7**：审计与可观测（始终开启）

### 2. 异构 GPU 管理

**关键特性**：
- **Request Router**：智能路由请求到最佳 Worker
- **实时指标采集**：inflight/VRAM/TTFT/P95/OOM
- **动态调度**：根据实时负载选择最佳 Worker
- **熔断机制**：OOM 或性能下降时自动降权

### 3. 证据构建层（核心护城河）

**设计理念**：
- 把任务从"生成式"变成"核验式"
- 先提取结构化证据，再基于证据进行判断
- 版本化存储，支持重放和对比

### 4. 双层判决机制

**规则先行 + LLM 后判**：
- **Rule Gate**：低成本快速过滤（Hard Fail / Soft Flag）
- **LLM Judge**：处理复杂情况（Soft Flag 进一步判定）
- **平衡**：成本控制 + 准确性保证

### 5. 可审计性优先

**设计原则**：
- **版本化**：所有组件都有版本号
- **可追溯**：所有结论都能追溯到原始证据
- **可重放**：按 case_id 完全重放处理过程
- **Always-on**：审计和可观测层始终开启

### 6. 人工闭环

**设计理念**：
- 不是"AI 替代医生"，而是"AI 辅助医生"
- 提供证据和上下文，让医生做最终决策
- 收集反馈，持续改进系统

---

## 输出说明

### LintResultV1

**结构**：
```json
{
  "case_id": "case_001",
  "status": "pass" | "flag" | "fail",
  "items": [
    {
      "issue_type": "laterality_mismatch" | "omission" | "contradiction" | ...,
      "severity": "high" | "med" | "low",
      "supporting_facts": ["vf_001", "rf_002"],
      "report_spans": [{"start": 120, "end": 135}],
      "recommended_action": "block" | "suggest" | "review",
      "confidence": 0.95,
      "explanation": "..."
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
    "fact_store_version": "v1.0"
  },
  "version": "v1.0",
  "timestamp": "2025-01-01T10:00:00Z"
}
```

### 可选输出：标注报告建议

**重要原则**：
- **Never overwrite original automatically**：绝不自动覆盖原始报告
- 只提供标注和建议，让医生决定是否采纳
- 保存原始报告和标注版本，支持审计

---

## 与 KYC 项目的对比

| 维度 | KYC 项目 | Rad-Linter |
|------|---------|-----------|
| **触发方式** | Batch Input | Real-time Event (R0) |
| **资源管理** | 托管 API（Fireworks） | 本地 GPU（异构管理） |
| **证据构建** | OCR + 字段提取 | Visual Facts + Report Facts |
| **判决机制** | Schema + Rules | Rules + LLM Judge（GPU） |
| **策略层** | 决策策略 | 策略闸门（Policy Gate） |
| **人工闭环** | 可选 | 标准流程（Review UI） |
| **可观测性** | Metrics/Logs/Traces | Audit + Metrics + Replay |

---

## 总结

Rad-Linter 完整系统架构采用 7 层设计：

1. **Layer 0-1**：数据接入和验证
2. **Layer 2**：证据构建（核心护城河）
3. **Layer 3**：规则闸门（低成本过滤）
4. **Layer 4**：LLM 判决（GPU 推理，异构管理）
5. **Layer 5**：策略决策（动作映射）
6. **Layer 6**：人工闭环（反馈收集）
7. **Layer 7**：审计与可观测（始终开启）

**核心特点**：
- **On-Prem 部署**：本地 GPU 资源管理
- **异构 GPU 调度**：智能路由到最佳 Worker
- **证据优先**：从"生成式"到"核验式"
- **可审计性**：完整的追溯和重放能力
- **人工闭环**：AI 辅助而非替代医生

通过这套架构，Rad-Linter 能够满足医院的严格需求：**Safety、Automation Rate、Latency、Auditability**。