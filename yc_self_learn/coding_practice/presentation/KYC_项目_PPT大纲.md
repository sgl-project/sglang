# KYC 项目 PPT 大纲

## 📋 总体结构

**三个版本**：
- **1分钟版本**：核心亮点（Elevator Pitch）- 1-2 页 slide
- **5分钟版本**：核心架构和设计（Overview）- 5-8 页 slide
- **30分钟版本**：完整技术细节（Deep Dive）- 15-20 页 slide（每页 1.5-2 分钟讲解）

**说明**：
- 每个 slide 需要讲解、互动、提问，不能假设每分钟一页
- 30分钟版本包含：讲解（20-25分钟）+ 互动提问（5-10分钟）
- 核心内容可以重复出现在不同版本，但详细程度不同

---

## 🎯 1分钟版本（Elevator Pitch）

**总共 1-2 页 slide，核心是快速传达价值**

### Slide 1: 项目介绍 + 核心亮点（60秒讲解）
- **标题**：KYC 文档智能提取系统
- **一句话**：基于多模态 LLM 的 KYC 文档自动提取和验证系统
- **核心价值**：自动化 KYC 审核流程，每单节省 5 分钟人工审核时间
- **核心亮点**（3 个要点，每个 15 秒）：
  1. **三层指标体系**：L0 稳定性 99%、L1 每单节省 5 分钟、L2 变更失败率 < 5%
  2. **低风险进化**：Feature Flag + Canary Release（1%→5%→25%→100%）+ 回归门禁
  3. **技术架构**：Schema-First 设计、确定性规则引擎、Per-File Isolation

**讲解要点**：
- 开场（10秒）：问题陈述
- 解决方案（20秒）：核心价值和成果
- 亮点展示（30秒）：3 个核心要点快速带过

---

## 🎯 5分钟版本（Overview）

**总共 5-8 页 slide，重点展示核心架构和设计亮点**

### Part 1: 项目概述（45秒）- Slide 1
- **问题**：KYC 审核需要大量人工处理，效率低、成本高（15秒）
- **解决方案**：基于多模态 LLM（Qwen2.5-VL-32B）的自动化提取系统（15秒）
- **成果**：每单节省 5 分钟，自动化率 95%（15秒）

### Part 2: 系统架构（45秒）- Slide 2
- **架构图**：Batch Input → Preprocessor → Rate Limiter → Fireworks API → Schema Validator → Deterministic Rules → Output
- **讲解要点**：每个组件的职责（20秒）+ 数据流（25秒）

### Part 3: 核心设计（90秒）- Slides 3-4

#### Slide 3: Schema-First + 确定性规则引擎（45秒）
- **Schema-First**：所有输出都符合预定义的 Schema，可维护、可扩展（20秒）
- **确定性规则引擎**：基于规则的二次验证，可审计、可测试（25秒）

#### Slide 4: Per-File Isolation + 标准化错误处理（45秒）
- **Per-File Isolation**：故障隔离、高可用（20秒）
- **标准化错误处理**：错误分类、重试策略（25秒）

### Part 4: 三层指标体系（60秒）- Slide 5
- **L0 稳定性**：成功率 99%、p95 < 15s、错误率 < 1%（20秒）
- **L1 业务收益**：每单节省 5 分钟、自动化率 95%、成本节省 $0.498/单（20秒）
- **L2 长期健康**：变更失败率 < 5%、Auditability 覆盖率 100%、PII 泄漏事件 0（20秒）

### Part 5: 可观测性（45秒）- Slide 6
- **三类信号**：Metrics（实时监控）、Logs（结构化日志）、Traces（链路追踪）（25秒）
- **可观测性闭环**：告警触发 → 定位问题 → 分析根因 → 快速止血（20秒）

### Part 6: 低风险进化（75秒）- Slide 7
- **Feature Flag + Canary Release**：1% → 5% → 25% → 100%，逐步扩大流量（35秒）
- **回归门禁**：Golden Set + Release Gate，确保改动不会把系统搞坏（40秒）

### Part 7: 总结（30秒）- Slide 8
- **核心价值**：可度量、可回归、可灰度、可回滚、可复盘（30秒）

---

## 🎯 30分钟版本（Deep Dive）

**总共 15-20 页 slide，每页讲解 1.5-2 分钟，包含互动和提问**

**时间分配**：
- 讲解时间：20-25 分钟
- 互动提问：5-10 分钟
- 每页 slide：1.5-2 分钟讲解

### Part 1: 项目概述（3分钟）

#### Slide 1: 项目背景和问题（1.5分钟）
- **业务背景**：KYC 审核流程介绍（30秒）
- **痛点**：人工审核效率低、成本高、易出错（40秒）
- **目标**：自动化 KYC 审核，提升效率和准确性（20秒）

#### Slide 2: 解决方案概述（1.5分钟）
- **技术栈**：多模态 LLM（Qwen2.5-VL-32B）+ Fireworks API（40秒）
- **核心能力**：文档 OCR、字段提取、验证、决策（40秒）
- **成果**：每单节省 5 分钟，自动化率 95%（30秒）

#### Slide 3: 系统架构图（详细版）（讲解 2 分钟）
```
┌─────────────┐
│ Batch Input │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Preprocessor│ (图片预处理、格式转换)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Rate Limiter │ (RPS 限制、并发控制)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Fireworks API│ (多模态 LLM 推理)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Schema Valid.│ (Pydantic 验证)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Rules     │ (确定性规则引擎)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Output    │ (结构化结果 + 决策)
└─────────────┘
```

### Part 2: 核心设计详解（6-7分钟）- Slides 4-6

**每页 slide 讲解 2 分钟，重点讲设计决策和 trade-off**

#### Slide 4: Schema-First 设计
- **定义**：所有输出都符合预定义的 Schema
- **实现**：Pydantic 模型
  ```python
  class KYCExtractionResult(BaseModel):
      full_name: str
      date_of_birth: str
      document_number: str
      expiry_date: str
      issuing_country: str
      ...
  ```
- **优势**：
  - 可维护：Schema 变更可追踪
  - 可扩展：新增字段只需更新 Schema
  - 可测试：Schema 验证可自动化

#### Slide 5: 确定性规则引擎
- **定义**：基于规则的二次验证和决策
- **规则类型**：
  - 格式验证：日期格式、ID 格式
  - 一致性检查：前后页信息一致性
  - 逻辑验证：出生日期 < 过期日期
- **优势**：
  - 可审计：所有决策有明确规则
  - 可测试：规则可单元测试
  - 可解释：决策原因可追溯

#### Slide 6: Per-File Isolation
- **定义**：每个文件独立处理，互不影响
- **实现**：
  - 独立的 trace_id
  - 独立的错误处理
  - 独立的结果存储
- **优势**：
  - 故障隔离：一个文件失败不影响其他文件
  - 高可用：部分失败不影响整体
  - 可并行：可以并行处理多个文件

#### Slide 7: 标准化错误处理
- **错误分类**：
  - API_TIMEOUT：API 超时
  - API_CONNECTION_ERROR：连接错误
  - SCHEMA_VALIDATION_ERROR：Schema 验证失败
  - RULE_VIOLATION：规则违反
- **错误处理策略**：
  - 可恢复错误：重试（指数退避）
  - 不可恢复错误：Fallback 或转人工审核

#### Slide 8: Rate Limiting + Retry
- **Rate Limiter**：
  - RPS 限制：防止 API 过载
  - 并发控制：限制同时处理的请求数
- **Retry 策略**：
  - 指数退避：1s → 2s → 4s
  - 最大重试次数：3 次
  - 可恢复错误才重试

#### Slide 9: Privacy-Aware Logging
- **原则**：不记录 PII（个人身份信息）
- **记录内容**：
  - trace_id
  - fw_request_id
  - model、tokens、latency
  - error_code
- **不记录**：
  - base64 image
  - prompt content
  - extracted PII

### Part 3: 三层指标体系详解（6-7分钟）- Slides 7-10

**每页 slide 讲解 1.5-2 分钟，重点讲指标设计和 SLO**

#### Slide 10: L0 稳定性指标（详细）
- **成功率（Success Rate）**
  - 定义：正常响应的请求数 / 总请求数
  - 目标：99%（Production）
  - 当前值：95%（PoV 阶段）
  - 告警阈值：< 98%
  
- **延迟指标（Latency）**
  - p50：3-5 秒（中位数）
  - p95：8-10 秒（SLO 目标：< 15 秒）
  - p99：15-20 秒（SLO 目标：< 30 秒）
  - 组成分析：
    - Preprocess：100-200ms
    - Rate Limiter Acquire：0-1000ms
    - Fireworks API Call：2000-8000ms
    - Schema Validation：50-100ms
    - Deterministic Rules：10-50ms
    - Save Result：20-50ms

- **错误率（Error Rate）**
  - 目标：< 1%
  - 告警阈值：> 5%

#### Slide 11: L1 业务收益指标（详细）
- **每单节省时间**：5 分钟
  - 人工审核：10 分钟/单
  - 自动化处理：5 分钟/单
  - 节省：5 分钟/单
  
- **自动化率**：95%
  - 自动通过：90%
  - 自动拒绝：5%
  - 转人工审核：5%
  
- **成本节省**：每单 $0.002
  - API 调用成本：$0.002/单
  - 人工审核成本：$0.50/单
  - 节省：$0.498/单

#### Slide 12: L2 长期健康指标（详细）
- **变更失败率**：< 5%
  - 定义：发布后需要回滚的比例
  - 目标：< 5%
  - 当前值：3%
  
- **Auditability 覆盖率**：100%
  - 所有决策都有 trace_id
  - 所有决策都有规则依据
  - 所有决策都可追溯
  
- **PII 泄漏事件**：0
  - 日志不记录 PII
  - 结果存储加密
  - 访问控制严格

#### Slide 13: Error Budget Policy
- **定义**：允许的错误预算，平衡发布速度和稳定性
- **SLO**：p95 < 15 秒，成功率 > 99%
- **Error Budget**：每月允许 1% 的错误率
- **策略**：错误预算用完后，暂停新功能发布，专注修复

### Part 4: 可观测性详解（5-6分钟）- Slides 11-13

**每页 slide 讲解 1.5-2 分钟，重点讲可观测性闭环和实践**

#### Slide 14: 三类信号框架（详细）
- **Metrics（指标）**
  - RPS（batch processing rate）
  - p95/p99 Latency
  - Error Rate
  - Schema Validation Fail Rate
  - Rate Limit Trigger Rate
  
- **Logs（日志）**
  - 结构化日志格式：
    ```json
    {
      "trace_id": "xxx",
      "fw_request_id": "yyy",
      "model": "qwen2.5-vl-32b",
      "tokens": 1000,
      "latency_ms": 5000,
      "error_code": null
    }
    ```
  - 不记录 PII
  
- **Traces（链路追踪）**
  - Span 1: Preprocess（image loading/normalize）
  - Span 2: Rate Limit Acquire
  - Span 3: Fireworks API Call
  - Span 4: Schema Validation
  - Span 5: Deterministic Rules
  - Span 6: Save Result

#### Slide 15: 可观测性闭环（详细）
```
1. 告警触发（Metrics）
   - Error Rate > 5%
   - p95 Latency > 15s
   
2. 定位问题（Traces）
   - 点击告警中的 trace_id
   - 查看调用链，找到慢的 Span
   
3. 分析根因（Logs）
   - 使用 trace_id 查询相关日志
   - 查看错误信息和上下文
   
4. 快速止血（自动回滚）
   - 触发回滚条件
   - 自动回滚到稳定版本
```

#### Slide 16: Dashboard 设计（详细）
- **On-Call Dashboard**
  - 实时监控：成功率、延迟、错误率
  - 告警列表：当前告警和状态
  - 趋势图：过去 24 小时指标趋势
  
- **Business Health Dashboard**
  - 自动化率：实时自动化率
  - 成本节省：累计节省成本
  - 处理量：每日处理量
  
- **Tracing Dashboard**
  - 调用链可视化
  - 耗时分析：每个 Span 的耗时
  - 错误追踪：错误请求的调用链

### Part 5: 回归测试详解（4-5分钟）- Slides 14-15

**每页 slide 讲解 2-2.5 分钟，重点讲 Golden Set 和 Release Gate**

#### Slide 17: Golden Set 构建策略
- **目标规模**：50-200 条测试用例
- **场景分类**：
  - 正常场景（20%）：清晰、标准格式的 ID
  - 边界场景（30%）：模糊、遮挡、低质量
  - 异常场景（30%）：版式变化、多页、复杂布局
  - 长尾场景（20%）：罕见格式、特殊字符
- **构建方式**：手动和自动结合
  - 手动挑选：关键场景、边界情况
  - 自动生成：大规模数据、初步筛选

#### Slide 18: Release Gate 设计
- **门禁指标**：
  - Schema Pass Rate > 95%
  - 字段级准确率 > 90%（critical fields）
  - Fallback Rate < 5%
  - 成本上限：$0.002 / request
- **阻断机制**：
  - 不达标禁止发布
  - 自动触发回滚
  - 通知相关团队

#### Slide 19: 回归测试流程
```
1. Before 测试（当前版本）
   - 运行 Golden Set
   - 记录指标：Schema Pass Rate、字段级准确率
   
2. 代码变更
   - 修改 Prompt
   - 更新模型版本
   - 调整规则
   
3. After 测试（新版本）
   - 运行 Golden Set
   - 记录指标：Schema Pass Rate、字段级准确率
   
4. 对比分析
   - Before vs After
   - 识别退化（Regression）
   - 分析原因
   
5. 发布决策
   - 通过门禁 → 发布
   - 不通过门禁 → 修复或回滚
```

### Part 6: 发布策略详解（4-5分钟）- Slides 16-17

**每页 slide 讲解 2-2.5 分钟，重点讲 Canary Release 和 Rollback 机制**

#### Slide 20: Feature Flag 设计
- **模型版本切换**
  - 选项：Qwen2.5-VL-32B、Qwen2.5-VL-7B、GPT-4-Vision
  - Canary 百分比：5%
  
- **Prompt 版本切换**
  - 选项：v1、v2
  - Canary 百分比：10%
  
- **验证器严格程度**
  - 选项：high、medium、low
  - Canary 百分比：0%（暂时不启用）

#### Slide 21: Canary Release 流程
```
阶段 1：1% 流量（观察 1 小时）
    ↓
    ├─ 指标正常 → 进入阶段 2
    └─ 指标异常 → 立即回滚

阶段 2：5% 流量（观察 2 小时）
    ↓
    ├─ 指标正常 → 进入阶段 3
    └─ 指标异常 → 立即回滚

阶段 3：25% 流量（观察 4 小时）
    ↓
    ├─ 指标正常 → 进入阶段 4
    └─ 指标异常 → 立即回滚

阶段 4：100% 流量（全量发布）
    ↓
    ├─ 指标正常 → 发布成功
    └─ 指标异常 → 立即回滚
```

#### Slide 22: 观察指标和阈值
| 指标 | 阈值 | 动作 |
|------|------|------|
| **Schema Pass Rate** | < 95% | 立即回滚 |
| **p95 Latency** | > 15s（+20%） | 立即回滚 |
| **Error Rate** | > 5% | 立即回滚 |
| **Cost per Request** | > $0.002 | 观察，不立即回滚 |

#### Slide 23: Rollback 机制
- **回滚条件**：
  - Schema Fail Rate × 2
  - p95 Latency + 20%
  - Error Rate > 5%
- **回滚流程**：
  1. 检测到异常指标
  2. 触发回滚条件检查
  3. 确认回滚（自动或手动）
  4. 执行回滚操作（切换 Feature Flag、更新配置）
  5. 验证回滚效果
  6. 记录回滚事件

### Part 7: 保护策略详解（2-3分钟）- Slide 18

**单页 slide 讲解 2-3 分钟，快速带过限流、重试、降级、幂等**

#### Slide 24: 限流（Rate Limiting）
- **触发**：RPS > RPM_LIMIT 或并发 > threshold
- **动作**：返回 429，等待 token
- **验证**：p95 < 15s，429 rate < 5%

#### Slide 25: 重试（Retry）
- **触发**：可恢复错误（API_TIMEOUT, API_CONNECTION_ERROR, API_SERVER_ERROR）
- **动作**：指数退避重试（MAX_RETRIES = 3）
- **验证**：成功率提升

#### Slide 26: 降级（Degradation）
- **触发**：主模型不可用 或 延迟 > threshold
- **动作**：OCR-only fallback 或 转人工审核
- **验证**：降级后成功率 > 80%

#### Slide 27: 幂等（Idempotency）
- **触发**：重复 request_id（通过 trace_id 去重）
- **动作**：返回缓存结果
- **验证**：重复处理率 < 0.1%

### Part 8: 总结和未来规划（2-3分钟）- Slides 19-20

#### Slide 19: 核心成果（1.5分钟）
- **技术成果**：Schema-First、确定性规则引擎、三层指标体系、可观测性闭环、低风险进化（50秒）
- **业务成果**：每单节省 5 分钟、自动化率 95%、成本节省 $0.498/单（40秒）

#### Slide 20: 未来规划（1-1.5分钟）
- **短期**：扩展到更多文档类型、优化模型性能（30秒）
- **中期**：支持多语言、引入更先进模型（30秒）
- **长期**：端到端 KYC Orchestration 模块（30秒）

### 互动提问环节（5-10分钟）
- 预留时间回答面试官问题
- 准备常见问题的答案（见下文）

---

## 📝 演讲要点

### 1分钟版本要点
- **核心**：突出三层指标体系和低风险进化
- **Slide 数量**：1-2 页
- **时间分配**：项目介绍（10秒）+ 核心价值（20秒）+ 三大亮点（30秒）
- **关键话术**：
  - "我们用三层指标度量成功：L0稳定性99%、L1每单节省5分钟、L2变更失败率<5%"
  - "我们用 Feature Flag + Canary Release（1%→5%→25%→100%）确保低风险进化"

### 5分钟版本要点
- **核心**：展示完整的系统设计和工程能力
- **Slide 数量**：5-8 页
- **时间分配**：项目概述（45秒）+ 系统架构（45秒）+ 核心设计（90秒）+ 三层指标（60秒）+ 可观测性（45秒）+ 低风险进化（75秒）+ 总结（30秒）
- **关键话术**：
  - "Schema-First 设计确保可维护性和可扩展性"
  - "确定性规则引擎确保可审计性和可测试性"
  - "可观测性闭环：告警 → 定位 → 止血"

### 30分钟版本要点
- **核心**：深入技术细节，展示 Senior 级别的系统设计能力
- **Slide 数量**：15-20 页
- **时间分配**：
  - 讲解：20-25 分钟（每页 1.5-2 分钟）
  - 互动提问：5-10 分钟
- **关键话术**：
  - 详细解释每个设计决策的 trade-off
  - 展示完整的工程实践（测试、发布、监控）
  - 强调可度量、可回归、可灰度、可回滚、可复盘

### 讲解技巧
1. **不要念 slide**：slide 只提供要点，你负责展开讲解
2. **留白时间**：每页 slide 讲解后停顿 2-3 秒，给听众思考时间
3. **互动**：30分钟版本中适当提问"这里大家有疑问吗？"
4. **重点重复**：关键数字和概念可以重复强调
5. **故事化**：用实际案例来说明设计决策（比如"某次事故..."）

---

## 🎯 准备建议

1. **练习顺序**：
   - 先练习 1 分钟版本（每天 3 次，持续 3 天）
   - 再练习 5 分钟版本（每天 2 次，持续 3 天）
   - 最后练习 30 分钟版本（每天 1 次，持续 3 天）

2. **关键数字要记住**：
   - L0：成功率 99%、p95 < 15s、错误率 < 1%
   - L1：每单节省 5 分钟、自动化率 95%、成本节省 $0.498/单
   - L2：变更失败率 < 5%、Auditability 覆盖率 100%、PII 泄漏事件 0

3. **准备常见问题**：
   - "能详细说说 XXX 吗？" → 从 30 分钟版本中抽取相关部分
   - "如果 XXX 怎么办？" → 引用保护策略
   - "为什么不这样设计？" → 引用 trade-off 分析

---

## 📚 相关文档

- [学习指南_KYC项目SystemDesign训练.md](./学习指南_KYC项目SystemDesign训练.md)
- [KYC_Day01_A1_详细讲解_指标与测试.md](./day01/KYC_Day01_A1_详细讲解_指标与测试.md)
- [KYC_Day02_A1_可观测性详解.md](./day02/KYC_Day02_A1_可观测性详解.md)
- [KYC_Day03_A1_回归测试与门禁详解.md](./day03/KYC_Day03_A1_回归测试与门禁详解.md)
- [KYC_Day04_A1_发布策略与回滚详解.md](./day04/KYC_Day04_A1_发布策略与回滚详解.md)
