# A01: Rad-Linter Pipeline 架构图
# Rad-Linter Pipeline Architecture Diagram

**作者**：Yanda Cheng  
**项目**：Rad-Linter  
**设计风格**：参考 KYC 项目 PPT 风格

---

## 📋 目录

1. [Pipeline 概览](#pipeline-概览)
2. [完整 Pipeline 架构图](#完整-pipeline-架构图)
3. [核心组件详解](#核心组件详解)
4. [数据流详解](#数据流详解)

---

## Pipeline 概览

Rad-Linter Pipeline 是从医学影像报告到 LoRA 微调模型的端到端训练流程，包含 6 个核心步骤：

```
Step 0: 数据子集处理
    ↓
Step 1: 格式对齐
    ↓
Step 2: 视觉特征提取
    ↓
Step 3: 规则基础标签构造
    ↓
Step 3.5: LLM Judge 标签生成
    ↓
Step 4: LoRA 训练
    ↓
Step 5: 三面板评估
```

---

## 完整 Pipeline 架构图

### 主流程图（KYC 风格）

```
┌─────────────────────────────────────────────────────────────────┐
│                    Indiana CXR Dataset                          │
│                    (原始医学影像数据集)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 0: Subset Data                                             │
│ • 数据子集处理                                                  │
│ • 筛选符合条件的影像报告对                                      │
│ • 数据质量检查                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Align OpenI Format                                      │
│ • 格式标准化                                                    │
│ • 统一数据结构                                                  │
│ • OpenI 格式对齐                                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Extract Visual Features                                 │
│ • TorchXRayVision 模型提取视觉特征                              │
│ • 检测/分割/测量                                                │
│ • 生成 visual_facts.jsonl                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Construct Labels (Rule-Based)                           │
│ • 规则基础标签生成                                              │
│ • 合成验证集（Poison Factory）                                  │
│ • 生成 Rule-Based Labels                                        │
└──────┬───────────────────────────────┬──────────────────────────┘
       │                               │
       ▼                               ▼
┌──────────────────────┐    ┌─────────────────────────────────────┐
│ visual_facts.jsonl   │    │ Rule-Based Labels                   │
│ • 结构化视觉事实     │    │ • 规则生成的标签                    │
│ • 可追溯的事实ID     │    │ • 可审计的规则依据                  │
└──────────────────────┘    └─────────────────────────────────────┘
       │                               │
       └──────────────┬────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3.5: Judge Labels (LLM Judge)                              │
│ • SGLang Judge Server (Docker 化部署)                           │
│ • 生成 Judge 标签                                               │
│ • 识别争议案例（Judge ≠ Rule）                                  │
└──────┬───────────────────────────────┬──────────────────────────┘
       │                               │
       ▼                               ▼
┌──────────────────────┐    ┌─────────────────────────────────────┐
│ Rule Labels          │    │ Judge Labels                        │
│ • 规则标签           │    │ • LLM 生成的高质量标签              │
│ • 100% 可审计        │    │ • 识别边界案例                      │
└──────────────────────┘    └─────────────────────────────────────┘
       │                               │
       └──────────────┬────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Train LoRA                                               │
│ • LoRA 微调训练                                                  │
│ • 参数高效微调                                                    │
│ • 模型版本管理                                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Evaluate (Three-Panel Evaluation)                       │
│ • Rule Adherence: 模型 vs Rule                                  │
│ • Silver Agreement: 模型 vs Judge                               │
│ • Judge-Rule Gap: Judge vs Rule                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation Results                           │
│ • Rule Adherence: 100%                                          │
│ • Silver Agreement: 88.74% accuracy, 80.0% F1                  │
│ • Judge-Rule Gap: 88.74% agreement (43 个争议案例)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心组件详解

### Step 0: Subset Data（数据子集处理）

**职责**：
- 从 Indiana CXR 数据集中筛选符合条件的影像报告对
- 数据质量检查（格式、完整性、可用性）
- 数据分布平衡

**输出**：
- 筛选后的数据集
- 数据质量报告

### Step 1: Align OpenI Format（格式对齐）

**职责**：
- 将不同来源的数据转换为统一的 OpenI 格式
- 标准化数据结构
- 统一字段命名和格式

**输出**：
- 标准化的数据集（OpenI 格式）
- 格式对齐报告

### Step 2: Extract Visual Features（视觉特征提取）⭐ **核心组件**

**职责**：
- 使用 TorchXRayVision 模型提取影像视觉特征
- 检测/分割/测量（病变区域、解剖结构）
- 生成结构化的 visual_facts.jsonl

**关键能力**：
- **检测**：病变区域检测
- **分割**：解剖结构分割
- **测量**：自动测量尺寸、面积、体积
- **Laterality**：左右侧识别
- **位置定位**：解剖区域定位

**输出**：
- `visual_facts.jsonl`：结构化视觉事实
- 每个事实包含：fact_id、type、laterality、location、attributes、confidence、evidence_refs

**技术栈**：
- TorchXRayVision（预训练医学影像分析模型）
- GPU 加速（可选）

### Step 3: Construct Labels (Rule-Based)（规则基础标签构造）

**职责**：
- 基于规则的标签生成
- 合成验证集（Poison Factory）
- 生成可审计的规则标签

**关键能力**：
- **确定性规则**：基于临床知识的确定性规则
- **可审计性**：所有标签都有明确的规则依据
- **可测试性**：规则可以独立测试和验证

**输出**：
- Rule-Based Labels
- 规则依据（rule_id、rule_version）

### Step 3.5: Judge Labels (LLM Judge)（LLM Judge 标签生成）⭐ **核心组件**

**职责**：
- 使用 SGLang Judge Server 生成高质量标签
- 识别争议案例（Judge ≠ Rule）
- 生成可解释的标签

**关键能力**：
- **高质量标签**：基于 LLM 的高质量标签
- **争议识别**：比较 Judge 和 Rule 标签，识别差异
- **可解释性**：所有结论都有明确的证据支撑

**技术栈**：
- SGLang Judge Server（Docker 化部署）
- 版本固定配置（docker_judge_versions.yaml）

**输出**：
- Judge Labels
- 争议案例列表（Judge ≠ Rule）

### Step 4: Train LoRA（LoRA 训练）

**职责**：
- 使用生成的标签训练 LoRA 模型
- 参数高效微调
- 模型版本管理

**关键能力**：
- **参数高效**：只训练少量参数，节省计算资源
- **可复现性**：固定随机种子和超参数
- **版本管理**：记录模型版本和训练配置

**输出**：
- 训练好的 LoRA 模型
- 训练日志和配置

### Step 5: Evaluate (Three-Panel Evaluation)（三面板评估）⭐ **核心亮点**

**职责**：
- 评估模型性能
- 三个维度的评估（Rule Adherence、Silver Agreement、Judge-Rule Gap）

**三个维度**：

1. **Rule Adherence（规则遵循率）**
   - **定义**：模型 vs Rule 标签的一致性
   - **指标**：100%（完美学习规则模式）
   - **意义**：评估模型是否学会了规则模式

2. **Silver Agreement（模型与 Judge 一致性）**
   - **定义**：模型 vs Judge 标签的一致性
   - **指标**：88.74% accuracy, 80.0% F1
   - **意义**：评估模型与高质量标签的一致性

3. **Judge-Rule Gap（Judge 与 Rule 差异）**
   - **定义**：Judge vs Rule 标签的一致性
   - **指标**：88.74% agreement（43 个争议案例）
   - **意义**：揭示规则的局限性，识别需要改进的地方

**输出**：
- 三个维度的评估结果
- 详细的评估报告

---

## 数据流详解

### 数据流路径

```
1. Indiana CXR Dataset
   ↓
   Step 0: Subset Data
   ↓
   筛选后的数据集

2. 筛选后的数据集
   ↓
   Step 1: Align OpenI Format
   ↓
   标准化的数据集（OpenI 格式）

3. 标准化的数据集
   ↓
   Step 2: Extract Visual Features
   ↓
   visual_facts.jsonl（结构化视觉事实）

4. visual_facts.jsonl + 标准化数据集
   ↓
   Step 3: Construct Labels (Rule-Based)
   ↓
   Rule-Based Labels

5. visual_facts.jsonl + Rule-Based Labels
   ↓
   Step 3.5: Judge Labels (LLM Judge)
   ↓
   Judge Labels + 争议案例列表

6. Rule-Based Labels + Judge Labels
   ↓
   Step 4: Train LoRA
   ↓
   训练好的 LoRA 模型

7. 训练好的 LoRA 模型 + Rule-Based Labels + Judge Labels
   ↓
   Step 5: Evaluate (Three-Panel Evaluation)
   ↓
   评估结果（Rule Adherence / Silver Agreement / Judge-Rule Gap）
```

### 关键数据结构

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
  }
}
```

#### Lint Items（检查项）

```json
{
  "issue_type": "contradiction",
  "severity": "high",
  "supporting_facts": ["vf_001", "rf_002"],
  "report_spans": [{"start": 120, "end": 135}],
  "recommended_action": "block",
  "confidence": 0.95,
  "explanation": "Visual fact vf_001 shows left pleural effusion (confidence 0.95), but report fact rf_001 states 'no effusion' (span 120-135). This is a contradiction (negation conflict)."
}
```

---

## 设计亮点

### 1. 双层标签生成机制

**Rule-Based + LLM Judge 双层机制**：
- **Rule-Based**：保证可审计性、可测试性
- **LLM Judge**：提升标签质量、识别边界案例
- **双层机制**：平衡可审计性和标签质量

### 2. 三面板评估框架

**三个维度全面评估**：
- **Rule Adherence**：确保模型学会规则
- **Silver Agreement**：评估模型实际性能
- **Judge-Rule Gap**：揭示规则局限性

### 3. 模块化设计

**Step 独立**：
- 每个 Step 可以单独运行和调试
- 易于维护和扩展
- 支持断点续传

### 4. 可复现性

**版本固定**：
- Docker 版本固定（docker_judge_versions.yaml）
- 固定随机种子和超参数
- 配置版本化

### 5. 可审计性

**完整的追溯链**：
- 所有标签都有明确的规则或模型依据
- 所有结论都可以追溯到原始证据
- 支持完全重放

---

## 与 KYC 项目的对比

| 维度 | KYC 项目 | Rad-Linter Pipeline |
|------|---------|-------------------|
| **核心目标** | 文档信息提取 | 模型训练 Pipeline |
| **数据流** | 输入 → 处理 → 输出 | 数据 → 训练 → 模型 |
| **关键组件** | LLM 推理、规则引擎 | 视觉特征提取、标签生成、模型训练 |
| **评估方式** | 准确率、自动化率 | 三面板评估（Rule/Judge/Pred） |
| **部署方式** | 托管 API（Fireworks） | 本地部署（Docker） |
| **资源需求** | API 调用 | GPU 资源（本地调度） |

---

## 总结

Rad-Linter Pipeline 是一个端到端的训练流程，从原始数据到最终模型：

1. **数据预处理**：Step 0-1 处理原始数据
2. **特征提取**：Step 2 提取视觉特征
3. **标签生成**：Step 3-3.5 生成双层标签
4. **模型训练**：Step 4 训练 LoRA 模型
5. **模型评估**：Step 5 三面板评估

**核心亮点**：
- 双层标签生成（Rule + LLM Judge）
- 三面板评估框架
- 模块化设计
- 可复现性保证
- 可审计性优先

通过这个 Pipeline，Rad-Linter 能够生成高质量的 LoRA 模型，用于医学影像报告质量检查。