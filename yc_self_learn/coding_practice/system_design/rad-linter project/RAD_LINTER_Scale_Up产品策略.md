# Rad-Linter Scale Up 产品策略
# Rad-Linter Scale Up Product Strategy

**作者**：Yanda Cheng  
**项目**：Rad-Linter  
**产品定位**：Gmail-style Quality Gate for Radiology Reports  
**核心洞察**：不是"AI 替代医生"，而是"写作流里的实时 lint + 轻量干预 + 可解释提示"

---

## 📋 目录

1. [产品定位与类比](#产品定位与类比)
2. [终极形态：Clinical Gmail-style Linter](#终极形态clinical-gmail-style-linter)
3. [Scale Up 产品策略](#scale-up-产品策略)
4. [关键 KPI](#关键-kpi)
5. [Scale Up 路线图](#scale-up-路线图)
6. [医疗专属要求](#医疗专属要求)
7. [一句话产品定位](#一句话产品定位)

---

## 产品定位与类比

### 核心洞察

**Rad-Linter 真要 scale up，最像的确实不是"自动写报告"，而是 Gmail 那种"拼写/语法/收件人提醒"的质量门禁（quality gate）**

**类比对比**：

| 维度 | Gmail | Rad-Linter |
|------|-------|-----------|
| **产品形态** | 拼写/语法检查器 | 报告质量检查器 |
| **核心价值** | 减少写作错误，提升沟通质量 | 减少医疗错误，提升报告质量 |
| **交互方式** | Inline highlights + Side panel | Inline highlights + Side panel |
| **干预级别** | Suggestion（大多） | Suggestion（低风险）+ Block（高风险） |
| **用户体验** | 写作流里的实时提示 | 签字流里的实时提示 |
| **产品理念** | 减少低成本错误且不烦人 | 减少高风险错误且不打断工作流 |

### 为什么这个类比很重要？

**决定 Scale Up 路线**：

如果你把它当成 Gmail-style linter，你的 scale up 路线会更清晰：

1. ✅ **轻量接入优先**：不要求一上来拿到完美影像证据
2. ✅ **默认不打断，除非高风险**：没把握就走 review，不要强行下结论
3. ✅ **反馈闭环是规模化发动机**：医生接受/忽略/修改 = 最有价值的 post-train 数据

---

## 终极形态：Clinical Gmail-style Linter

### 产品形态

**嵌入签字/审核流，成为写作体验的一部分**

Rad-Linter 不是让医生跳出当前工作流去使用一个额外系统，而是把"发现问题"做成写作体验的一部分。

### Layer 1-5 对应的用户体验

#### 1. Inline Highlights（内联高亮）

**对应 Layer 2-4**：Evidence Builder + Rule Gate + LLM Judge

**功能**：
- 在报告正文里高亮 span（使用 span_ref）
- 不同类型的 issue 用不同颜色标识
- 鼠标悬停显示详细说明

**UI 示例**：
```
报告正文：
"No [laterality mismatch: 左 vs 右] pleural effusion is seen."
   ↑ 红色波浪线（Warn级别）
   
"Cardiomegaly is [missing measurement: 心脏大小未提及] present."
   ↑ 黄色波浪线（Info级别）
```

**实现**：
- 使用 `span_ref`（start, end）定位报告原文位置
- 不同类型 issue 用不同颜色：红色（Block）、黄色（Warn）、灰色（Info）

#### 2. Side Panel Issues（侧边栏问题卡片）

**对应 Layer 5**：Policy Gate

**功能**：
- 右侧列出所有问题卡片
- 每个卡片包含：issue_type / severity / evidence
- 支持一键操作

**UI 示例**：
```
┌─────────────────────────────────────┐
│ Issues (3)                          │
├─────────────────────────────────────┤
│ 🔴 Block: Laterality Mismatch       │
│    Visual fact: Left effusion       │
│    Report text: "Right effusion"    │
│    [Accept] [Ignore] [Review]       │
├─────────────────────────────────────┤
│ 🟡 Warn: Missing Measurement        │
│    Required field: Heart size       │
│    [Accept] [Ignore]                │
├─────────────────────────────────────┤
│ ⚪ Info: Style Suggestion           │
│    Optional: Use active voice       │
│    [Accept] [Ignore]                │
└─────────────────────────────────────┘
```

**卡片结构**：
- **Severity 标识**：🔴 Block / 🟡 Warn / ⚪ Info
- **Issue Type**：Laterality Mismatch / Missing Measurement / Style Suggestion
- **Evidence**：显示相关证据（visual_facts、report_facts）
- **Action Buttons**：Accept / Ignore / Review

#### 3. One-Click Actions（一键操作）

**对应 Layer 6**：Human-in-the-loop

**功能**：
- **Accept（接受）**：采纳系统建议，自动修正报告
- **Ignore（忽略）**：忽略系统建议，强制填原因（用于反馈回流）
- **Review（复核）**：送人工复核队列（带证据包）

**强制原因填写**：
- Ignore 时强制填写原因（用于后续分析和模型改进）
- 原因选项：False positive / Not relevant / Other reason

#### 4. Quiet Suggestions vs Hard Blocks（静默建议 vs 硬性拦截）

**对比 Gmail**：

| 级别 | Gmail | Rad-Linter |
|------|-------|-----------|
| **Suggestion** | 拼写错误（大多） | 低风险问题（风格/可选优化） |
| **Block** | 发送前提醒（少数） | 高风险错误（laterality/关键测量矛盾/严重遗漏）必须处理 |

**策略**：
- **Info（灰）**：风格/可选优化（不阻断）
- **Warn（黄）**：可能错误（建议修改/确认）
- **Block（红）**：高风险错误（laterality/关键测量矛盾/严重遗漏）必须处理

---

## Scale Up 产品策略

### 核心策略：从"拦截器"变成"分级提醒器"

**Gmail 的核心不是"全对"，而是减少低成本错误并且不烦人。**

**Rad-Linter 同样要做一个平衡**：

- ✅ **减少错误**：特别是高风险错误（laterality、关键测量）
- ✅ **不烦人**：低风险问题只是建议，不阻断工作流
- ✅ **可解释**：每条提示都有明确的证据支撑

### 三档干预级别（最像 Gmail，也最容易落地）

#### Level 1: Info（灰）- 风格/可选优化

**特征**：
- 不阻断工作流
- 只是建议，可以选择忽略
- 不强制处理

**示例**：
- 风格建议（使用主动语态）
- 可选优化（术语标准化）
- 格式建议（段落结构）

**UI**：
- 灰色波浪线
- 侧边栏显示为 Info 级别
- 只有 Accept / Ignore 选项

#### Level 2: Warn（黄）- 可能错误

**特征**：
- 建议修改/确认
- 不强制阻断，但建议处理
- 需要医生确认

**示例**：
- 测量值范围异常（但不矛盾）
- 必填字段缺失
- 模板格式问题

**UI**：
- 黄色波浪线
- 侧边栏显示为 Warn 级别
- Accept / Ignore / Review 选项

#### Level 3: Block（红）- 高风险错误

**特征**：
- **必须处理**才能签字
- 阻止签字流程
- 强制复核或修正

**示例**：
- Laterality 错误（左右侧不一致）
- 关键测量矛盾
- 严重遗漏（关键发现未提及）

**UI**：
- 红色波浪线 + 红色高亮
- 侧边栏显示为 Block 级别
- 只有 Review 选项（送人工复核）或强制修正

### Policy Gate（Layer 5）对应产品语言

**原技术语言**：
- Policy Gate（策略闸门）

**产品语言**：
- **Inline Suggestions + Severity-based Interventions**
  - Inline Suggestions：内联建议（Inline Highlights）
  - Severity-based Interventions：基于严重程度的分级干预

---

## 关键 KPI

### Gmail 那套 KPI，但加上医疗安全要求

#### 1. Coverage（覆盖率）

**定义**：能覆盖多少模板/科室/报告类型

**目标**：
- 主要报告类型覆盖率 > 90%
- 主要科室覆盖率 > 85%
- 模板覆盖率 > 80%

**测量方式**：
- 按 report_type 统计
- 按 department 统计
- 按 template 统计

#### 2. Precision（少打扰）

**定义**：医生点"ignore"的比例（误报率）

**目标**：
- False Positive Rate < 5%（误报率）
- Ignore Rate < 10%（医生忽略率）
- 医生满意度 > 85%

**测量方式**：
- 统计医生点击 "Ignore" 的比例
- 收集医生反馈和满意度评分
- 分析被忽略的 issue 类型

#### 3. Recall（高风险必抓）

**定义**：高风险错误（laterality/关键结论）漏检率

**目标**：
- **高风险错误漏检率 < 1%**（Critical）
- 中等风险错误漏检率 < 5%
- 低风险错误检出率 > 80%

**测量方式**：
- 高风险错误（Laterality、关键测量）漏检率
- 中等风险错误（必填字段、测量值范围）漏检率
- 低风险错误（格式、风格）检出率

#### 4. Time Saved（节省时间）

**定义**：平均每份报告减少多少返工时间

**目标**：
- 平均每份报告节省 > 2 分钟
- 减少返工率 > 50%
- 减少人工审核时间 > 30%

**测量方式**：
- 对比使用 Rad-Linter 前后的返工时间
- 统计减少的人工审核时间
- 分析医生工作流程效率提升

#### 5. Automation Rate（自动化率）

**定义**：自动放行比例（不过度打扰医生）

**目标**：
- 自动放行率 > 85%
- 人工复核率 < 10%
- 拦截率 < 5%

**测量方式**：
- 统计自动放行（无 issue）的比例
- 统计人工复核的比例
- 统计拦截（Block）的比例

---

## Scale Up 路线图

### 阶段 1：轻量接入优先（MVP）

**策略**：不要求一上来拿到完美影像证据

**实现**：
1. **先做报告文本侧的高置信错误**
   - Laterality 检查（文本中的左右侧识别）
   - 测量单位检查（mm vs cm）
   - 否定冲突（"no effusion" vs 检测到 effusion）
   - 模板缺段检查（必填段落缺失）

2. **暂不接入 visual facts**
   - 先基于文本规则做检查
   - 建立医生信任和使用习惯
   - 收集反馈数据

3. **保守策略**
   - 只提示高置信错误
   - 低置信度转为人工复核
   - 不强制阻断工作流

### 阶段 2：逐步接入 Visual Facts

**策略**：有就加分，没有就保守升级 review

**实现**：
1. **逐步接入 visual facts**
   - 先从简单场景开始（X-ray 胸部检查）
   - 逐步扩展到复杂场景（CT、MRI）
   - 建立 visual facts 质量监控

2. **保守升级策略**
   - 如果 visual facts 质量不够，转为人工复核
   - 不强制依赖 visual facts
   - 建立 visual facts 置信度阈值

3. **反馈闭环**
   - 收集医生对 visual facts 相关提示的反馈
   - 持续改进 visual facts 提取质量
   - 优化 visual facts 与 report facts 的对齐算法

### 阶段 3：默认不打断，除非高风险

**策略**：和 Gmail 一样，宁可不提示，也别乱提示

**实现**：
1. **保守策略**
   - 低置信度不提示，转为人工复核
   - 只提示高置信错误
   - 建立置信度阈值机制

2. **分级干预**
   - Info（灰）：不阻断，只是建议
   - Warn（黄）：建议修改，不强制
   - Block（红）：必须处理才能签字

3. **用户可配置**
   - 允许医生自定义哪些类型的 issue 需要提示
   - 支持科室级别的配置
   - 支持个人偏好设置

### 阶段 4：反馈闭环规模化

**策略**：反馈闭环是规模化发动机

**实现**：
1. **收集反馈数据**
   - 医生 Accept / Ignore / Review 的所有操作
   - 医生填写的 ignore 原因
   - 医生修改后的报告内容

2. **反馈回流机制**
   - Accept → 变成训练数据（SFT）
   - Ignore → 变成误报数据（优化规则阈值）
   - Review → 变成专家标注数据（DPO/GRPO）

3. **持续改进**
   - 基于反馈持续优化模型
   - 基于反馈优化规则阈值
   - 基于反馈改进提示质量

---

## 医疗专属要求

### 为什么 Rad-Linter 比 Gmail 多两层"医疗专属"？

**为了安全合规，需要比 Gmail 多做两件事：**

### 1. 证据引用是强制的

#### Gmail 提醒错别字不需要证据

**Gmail**：
- "拼写错误：'recieve' 应该是 'receive'"
- 不需要证据，只是语言规则

#### Rad-Linter 必须提供证据

**Rad-Linter**：
- **每条 issue 绑定 span_ref**（报告原文位置）
- **每条"影像相关" issue 绑定 fact_id**（visual_facts 的证据）
- **不允许凭空补"图像事实"**

**实现**：
```json
{
  "issue_type": "laterality_mismatch",
  "severity": "high",
  "supporting_facts": ["vf_001", "rf_002"],  // 必须绑定 fact_id
  "report_spans": [                          // 必须绑定 span_ref
    {"start": 120, "end": 135, "text": "right pleural effusion"}
  ],
  "explanation": "Visual fact vf_001 shows left pleural effusion (confidence 0.95), but report fact rf_002 states 'right pleural effusion' (span 120-135). This is a laterality mismatch."
}
```

### 2. 审计与可重放

#### Gmail 不需要审计

**Gmail**：
- 拼写检查不需要审计
- 不需要追溯为什么提示

#### Rad-Linter 必须完全可审计

**Rad-Linter**：
- **case_id + model/prompt/facts 版本化**
- **出事能解释"为什么提示、基于什么证据、当时模型版本是什么"**
- **支持按 case_id 完全重放**

**实现**：
```json
{
  "case_id": "case_001",
  "metadata": {
    "model_version": "qwen2.5-vl-32b",
    "prompt_version": "v1.2",
    "schema_version": "v1.0",
    "fact_store_version": "v1.0",
    "rule_version": "v1.0"
  },
  "audit_trail": [
    {
      "timestamp": "2025-01-01T10:00:00Z",
      "layer": "L2_Evidence_Builder",
      "operation": "extract_visual_facts",
      "result": "visual_facts_v1.0"
    },
    {
      "timestamp": "2025-01-01T10:00:01Z",
      "layer": "L3_Rule_Gate",
      "operation": "check_laterality",
      "result": "soft_flag"
    },
    {
      "timestamp": "2025-01-01T10:00:02Z",
      "layer": "L4_LLM_Judge",
      "operation": "judge_laterality_mismatch",
      "result": "high_confidence_mismatch"
    }
  ]
}
```

---

## 一句话产品定位

### 核心定位

**"Rad-Linter is a Gmail-style quality gate for radiology reports: an inline, evidence-grounded, policy-aware linter that flags high-risk inconsistencies and omissions before sign-off, with conservative escalation and full auditability."**

### 关键词拆解

#### 1. Gmail-style quality gate

**含义**：
- 类似 Gmail 的拼写/语法检查器
- 写作流里的实时提示
- 不打断工作流，只是提醒

#### 2. Inline

**含义**：
- 内联高亮（Inline Highlights）
- 在报告正文里直接显示问题
- 不跳出当前工作流

#### 3. Evidence-grounded

**含义**：
- 每条提示都有明确的证据支撑
- 必须绑定 visual_facts 和 report_facts
- 不允许凭空判断

#### 4. Policy-aware

**含义**：
- 基于策略的分级干预
- Info / Warn / Block 三档干预
- 可配置的策略规则

#### 5. Flags high-risk inconsistencies and omissions

**含义**：
- 重点标记高风险错误
- Laterality 错误、关键测量矛盾、严重遗漏
- 不只是风格建议

#### 6. Before sign-off

**含义**：
- 签字前的质量门禁
- 防止错误报告被签署
- 关键时间点拦截

#### 7. Conservative escalation

**含义**：
- 保守的升级策略
- 没把握就走 review，不强行下结论
- 宁可不提示，也别乱提示

#### 8. Full auditability

**含义**：
- 完全可审计
- 所有决策都可追溯
- 支持按 case_id 完全重放

---

## 医生端 UI 交互草图要点

### 界面布局

```
┌─────────────────────────────────────────────────────────────────┐
│  Radiology Report Editor                    [Save] [Sign-off]   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Report Text:                                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ No [laterality mismatch: 左 vs 右] pleural effusion    │    │
│  │   ↑ 红色波浪线 + 红色高亮                               │    │
│  │                                                           │    │
│  │ Cardiomegaly is [missing measurement] present.           │    │
│  │                  ↑ 黄色波浪线                           │    │
│  │                                                           │    │
│  │ The patient shows [style suggestion: use active voice].  │    │
│  │                          ↑ 灰色波浪线                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Issues (3)                         [Clear All]          │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │ 🔴 Block: Laterality Mismatch                           │    │
│  │    Visual fact: Left pleural effusion (confidence 0.95) │    │
│  │    Report text: "Right pleural effusion" (span 120-135) │    │
│  │    [Accept] [Ignore] [Review]                          │    │
│  │                                                          │    │
│  │ 🟡 Warn: Missing Measurement                            │    │
│  │    Required field: Heart size                           │    │
│  │    [Accept] [Ignore]                                   │    │
│  │                                                          │    │
│  │ ⚪ Info: Style Suggestion                               │    │
│  │    Optional: Use active voice                           │    │
│  │    [Accept] [Ignore]                                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 交互要点

#### 1. Inline Highlights

**功能**：
- 报告正文里直接显示问题（使用波浪线）
- 不同严重程度用不同颜色：红色（Block）、黄色（Warn）、灰色（Info）
- 鼠标悬停显示详细说明

**实现**：
- 使用 `span_ref`（start, end）定位报告原文位置
- 点击波浪线跳转到对应的侧边栏问题卡片

#### 2. Side Panel Issues

**功能**：
- 右侧列出所有问题卡片
- 每个卡片包含：severity 标识、issue_type、evidence、action buttons
- 支持一键操作（Accept / Ignore / Review）

**实现**：
- 按严重程度排序：Block > Warn > Info
- 每个卡片可以折叠/展开
- 点击 Accept 自动修正报告（如果是可自动修正的问题）

#### 3. One-Click Actions

**功能**：
- **Accept（接受）**：采纳系统建议，自动修正报告
- **Ignore（忽略）**：忽略系统建议，强制填原因
- **Review（复核）**：送人工复核队列（带证据包）

**强制原因填写**：
- Ignore 时弹出对话框，强制填写原因
- 原因选项：False positive / Not relevant / Other reason
- 原因收集用于后续分析和模型改进

#### 4. 状态反馈

**功能**：
- 操作后显示反馈（已接受 / 已忽略 / 已送复核）
- 卡片状态更新（已处理 / 待处理）
- 实时更新问题数量

---

## 与 Gmail 的对比总结

| 维度 | Gmail | Rad-Linter |
|------|-------|-----------|
| **产品形态** | 拼写/语法检查器 | 报告质量检查器 |
| **交互方式** | Inline highlights + Side panel | Inline highlights + Side panel |
| **干预级别** | Suggestion（大多） | Info / Warn / Block（三档） |
| **证据要求** | 不需要证据 | 必须绑定 evidence（医疗安全） |
| **审计要求** | 不需要审计 | 完全可审计（医疗合规） |
| **产品理念** | 减少低成本错误且不烦人 | 减少高风险错误且不打断工作流 |
| **反馈闭环** | 用户接受/忽略 | 医生接受/忽略（用于 post-train） |

---

## 总结

### 核心要点

1. **产品定位**：Gmail-style quality gate for radiology reports
2. **产品形态**：Inline highlights + Side panel + One-click actions
3. **干预策略**：三档干预（Info / Warn / Block），保守升级
4. **关键 KPI**：Coverage、Precision、Recall、Time Saved、Automation Rate
5. **Scale Up 路线**：轻量接入 → 逐步接入 Visual Facts → 默认不打断 → 反馈闭环
6. **医疗专属**：强制证据引用 + 完全可审计

### Scale Up 成功的关键

1. ✅ **轻量接入优先**：不要求一上来拿到完美影像证据
2. ✅ **默认不打断，除非高风险**：没把握就走 review，不要强行下结论
3. ✅ **反馈闭环是规模化发动机**：医生接受/忽略/修改 = 最有价值的 post-train 数据

### 一句话产品定位

**"Rad-Linter is a Gmail-style quality gate for radiology reports: an inline, evidence-grounded, policy-aware linter that flags high-risk inconsistencies and omissions before sign-off, with conservative escalation and full auditability."**