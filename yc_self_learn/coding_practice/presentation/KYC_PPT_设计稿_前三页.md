# KYC Project PPT 设计稿 - 前三页
# 直接照着这个设计稿制作PPT

---

## SLIDE 1: 项目背景和问题

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 背景色: 深蓝色 #1e3a8a | 文字: 白色 #FFFFFF | 字体: 32pt 加粗          │ │
│  │ KYC Project Background & Core Problems                                  │ │
│  │ KYC 项目背景与核心问题                                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌──────────────────────────────────────────────────────┐  ┌──────────────┐ │
│  │ 左侧内容区 (60% 宽度)                                │  │ 右侧配图区   │ │
│  │ 背景: 浅灰色渐变 #f5f5f5 → #e8e8e8                   │  │ (40% 宽度)   │ │
│  │ 文字: 黑色 #000000 | 字体: 18pt                      │  │              │ │
│  │                                                      │  │ [可放置图表] │ │
│  │  📋 Business Background / 业务背景                   │  │ - KYC流程图  │ │
│  │  (30秒讲解)                                          │  │ - 痛点数据图 │ │
│  │                                                      │  │              │ │
│  │  • KYC (Know Your Customer) review is a core        │  │              │ │
│  │    compliance process in financial services         │  │              │ │
│  │  • Reviewers need to extract key information from   │  │              │ │
│  │    large volumes of ID cards, passports, and other  │  │              │ │
│  │    documents                                         │  │              │ │
│  │  • Traditional workflow relies on manual review,    │  │              │ │
│  │    which is time-consuming and error-prone          │  │              │ │
│  │                                                      │  │              │ │
│  │  💡 Pain Points Analysis / 痛点分析                 │  │              │ │
│  │  (40秒讲解)                                          │  │              │ │
│  │                                                      │  │              │ │
│  │  1. Low Efficiency / 效率低下                       │  │              │ │
│  │     • Manual review takes an average of 10 minutes  │  │              │ │
│  │       per case                                       │  │              │ │
│  │     • Repetitive work consumes significant          │  │              │ │
│  │       reviewer time                                  │  │              │ │
│  │                                                      │  │              │ │
│  │  2. High Cost / 成本高昂                            │  │              │ │
│  │     • Manual review cost: $0.50 per case            │  │              │ │
│  │     • Human resource costs grow linearly with       │  │              │ │
│  │       business growth                                │  │              │ │
│  │                                                      │  │              │ │
│  │  3. Inconsistent Quality / 质量不稳定               │  │              │ │
│  │     • Human fatigue leads to increased error rates  │  │              │ │
│  │     • Difficult to guarantee 100% accuracy          │  │              │ │
│  │                                                      │  │              │ │
│  │  🎯 Project Goals / 项目目标                        │  │              │ │
│  │  (20秒讲解)                                          │  │              │ │
│  │                                                      │  │              │ │
│  │  • Automate KYC review process to reduce manual     │  │              │ │
│  │    intervention                                      │  │              │ │
│  │  • Improve processing efficiency and shorten        │  │              │ │
│  │    review time                                       │  │              │ │
│  │  • Ensure accuracy and consistency, reduce error    │  │              │ │
│  │    rates                                             │  │              │ │
│  │                                                      │  │              │ │
│  └──────────────────────────────────────────────────────┘  └──────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 文字内容（直接复制）：

**标题栏（深蓝色背景，白色文字）：**
```
KYC Project Background & Core Problems
KYC 项目背景与核心问题
```

**左侧内容区：**

```
📋 Business Background / 业务背景
(30秒讲解)

• KYC (Know Your Customer) review is a core compliance process in financial services
• Reviewers need to extract key information from large volumes of ID cards, passports, and other documents
• Traditional workflow relies on manual review, which is time-consuming and error-prone

💡 Pain Points Analysis / 痛点分析
(40秒讲解)

1. Low Efficiency / 效率低下
   • Manual review takes an average of 10 minutes per case
   • Repetitive work consumes significant reviewer time

2. High Cost / 成本高昂
   • Manual review cost: $0.50 per case
   • Human resource costs grow linearly with business growth

3. Inconsistent Quality / 质量不稳定
   • Human fatigue leads to increased error rates
   • Difficult to guarantee 100% accuracy

🎯 Project Goals / 项目目标
(20秒讲解)

• Automate KYC review process to reduce manual intervention
• Improve processing efficiency and shorten review time
• Ensure accuracy and consistency, reduce error rates
```

---

## SLIDE 2: 解决方案概述

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 背景: 白色 #FFFFFF                                                      │ │
│  │ 顶部渐变色带: 蓝色 #3b82f6 → 紫色 #8b5cf6                              │ │
│  │ ───────────────────────────────────────────────────────────────────────│ │
│  │ 标题栏: 深蓝色 #1e3a8a | 文字: 白色 #FFFFFF | 字体: 32pt 加粗          │ │
│  │ Automated Solution Based on Multimodal LLM                             │ │
│  │ 基于多模态 LLM 的自动化解决方案                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 上部区域: 技术栈 (40秒讲解)                                            │ │
│  │ 字体: 24pt 加粗                                                        │ │
│  │                                                                        │ │
│  │ 🤖 Core Technology / 核心技术                                          │ │
│  │                                                                        │ │
│  │     Qwen2.5-VL-32B (Multimodal Model)                                 │ │
│  │                    ↓                                                  │ │
│  │     Inference API (Fireworks API / sglang serve)                     │ │
│  │     (Hosted or Local Inference Service)                              │ │
│  │                    ↓                                                  │ │
│  │     FastAPI Service (Orchestration)                                  │ │
│  │                    ↓                                                  │ │
│  │     Pydantic Schema (Validation & Contracts)                         │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 中部区域: 核心能力 (40秒讲解)                                          │ │
│  │ 字体: 24pt 加粗                                                        │ │
│  │                                                                        │ │
│  │ 🔍 Four Core Capabilities / 四大核心能力                              │ │
│  │                                                                        │ │
│  │  ┌──────────────┬─────────────────────────┬──────────────────────────┐│ │
│  │  │ Capability   │ Description             │ Value                    ││ │
│  │  │ / 能力       │ / 说明                  │ / 价值                   ││ │
│  │  ├──────────────┼─────────────────────────┼──────────────────────────┤│ │
│  │  │ 📄 Document  │ Identify and extract    │ Support multiple         ││ │
│  │  │    OCR       │ text from images        │ document formats         ││ │
│  │  │ 文档 OCR     │ 识别和提取图像中的文字   │ 支持多种证件格式         ││ │
│  │  ├──────────────┼─────────────────────────┼──────────────────────────┤│ │
│  │  │ 🔎 Field     │ Structured extraction   │ Standardized output      ││ │
│  │  │    Extraction│ of key info (name, DOB, │ format                   ││ │
│  │  │ 字段提取     │ ID number)              │ 标准化输出格式           ││ │
│  │  ├──────────────┼─────────────────────────┼──────────────────────────┤│ │
│  │  │ ✅ Auto      │ Deterministic rule-based│ Ensure data accuracy     ││ │
│  │  │    Validation│ secondary validation    │ 确保数据准确性           ││ │
│  │  │ 自动验证     │ 基于确定性规则的二次验证 │                          ││ │
│  │  ├──────────────┼─────────────────────────┼──────────────────────────┤│ │
│  │  │ 🎯 Smart     │ Auto approve/reject/    │ Reduce manual            ││ │
│  │  │    Decision  │ escalate to human       │ intervention             ││ │
│  │  │ 智能决策     │ 自动通过/拒绝/转人工审核 │ 减少人工干预             ││ │
│  │  └──────────────┴─────────────────────────┴──────────────────────────┘│ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 下部区域: 核心成果 (30秒讲解)                                          │ │
│  │ 字体: 24pt 加粗                                                        │ │
│  │                                                                        │ │
│  │ 📊 Key Metrics / 关键指标                                              │ │
│  │                                                                        │ │
│  │  ✅ Time Saved per Case: 5 minutes                                    │ │
│  │     (Manual 10 min → Automated 5 min)                                 │ │
│  │     每单节省时间: 5 分钟 (人工 10 分钟 → 自动化 5 分钟)                │ │
│  │                                                                        │ │
│  │  ✅ Automation Rate: 95%                                               │ │
│  │     (90% auto-approve + 5% auto-reject)                               │ │
│  │     自动化率: 95% (90% 自动通过 + 5% 自动拒绝)                         │ │
│  │                                                                        │ │
│  │  ✅ Cost Savings: $0.498 per case                                     │ │
│  │     (API cost $0.002 vs Manual cost $0.50)                            │ │
│  │     成本节省: $0.498/单 (API成本 $0.002 vs 人工成本 $0.50)            │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 文字内容（直接复制）：

**标题栏（深蓝色背景，白色文字）：**
```
Automated Solution Based on Multimodal LLM
基于多模态 LLM 的自动化解决方案
```

**上部区域 - 技术栈：**
```
🤖 Core Technology / 核心技术

    Qwen2.5-VL-32B (Multimodal Model)
                   ↓
    Inference API (Fireworks API / sglang serve)
    (Hosted or Local Inference Service)
                   ↓
    FastAPI Service (Orchestration)
                   ↓
    Pydantic Schema (Validation & Contracts)
```

**中部区域 - 核心能力（表格）：**
```
🔍 Four Core Capabilities / 四大核心能力

┌──────────────┬─────────────────────────┬──────────────────────────┐
│ Capability   │ Description             │ Value                    │
│ / 能力       │ / 说明                  │ / 价值                   │
├──────────────┼─────────────────────────┼──────────────────────────┤
│ 📄 Document  │ Identify and extract    │ Support multiple         │
│    OCR       │ text from images        │ document formats         │
│ 文档 OCR     │ 识别和提取图像中的文字   │ 支持多种证件格式         │
├──────────────┼─────────────────────────┼──────────────────────────┤
│ 🔎 Field     │ Structured extraction   │ Standardized output      │
│    Extraction│ of key info (name, DOB, │ format                   │
│ 字段提取     │ ID number)              │ 标准化输出格式           │
├──────────────┼─────────────────────────┼──────────────────────────┤
│ ✅ Auto      │ Deterministic rule-based│ Ensure data accuracy     │
│    Validation│ secondary validation    │ 确保数据准确性           │
│ 自动验证     │ 基于确定性规则的二次验证 │                          │
├──────────────┼─────────────────────────┼──────────────────────────┤
│ 🎯 Smart     │ Auto approve/reject/    │ Reduce manual            │
│    Decision  │ escalate to human       │ intervention             │
│ 智能决策     │ 自动通过/拒绝/转人工审核 │ 减少人工干预             │
└──────────────┴─────────────────────────┴──────────────────────────┘
```

**下部区域 - 核心成果：**
```
📊 Key Metrics / 关键指标

 ✅ Time Saved per Case: 5 minutes
    (Manual 10 min → Automated 5 min)
    每单节省时间: 5 分钟 (人工 10 分钟 → 自动化 5 分钟)

 ✅ Automation Rate: 95%
    (90% auto-approve + 5% auto-reject)
    自动化率: 95% (90% 自动通过 + 5% 自动拒绝)

 ✅ Cost Savings: $0.498 per case
    (API cost $0.002 vs Manual cost $0.50)
    成本节省: $0.498/单 (API成本 $0.002 vs 人工成本 $0.50)
```

---

## SLIDE 3: 系统架构图（详细版）

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 背景: 纯白色 #FFFFFF                                                    │ │
│  │ 标题栏: 深蓝色 #1e3a8a | 文字: 白色 #FFFFFF | 字体: 32pt 加粗          │ │
│  │ System Architecture Design                                               │ │
│  │ 系统架构设计                                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│                                                                              │
│                              ┌──────────────────────────────┐               │
│                              │   Batch Input                │               │
│                              │   (Batch Document Input)     │               │
│                              │   (批量文档输入)              │               │
│                              │  圆角矩形 | 浅蓝背景 #e0f2fe │               │
│                              │  深蓝边框 | 20pt 加粗        │               │
│                              └──────────────┬───────────────┘               │
│                                             │                                │
│                                             ▼ (蓝色箭头 #3b82f6)            │
│                              ┌──────────────────────────────┐               │
│                              │   Preprocessor               │               │
│                              │   (Image Preprocessing &     │               │
│                              │    Conversion)               │               │
│                              │   (图片预处理、格式转换)      │               │
│                              │   • Image loading &          │               │
│                              │     normalization            │               │
│                              │   • Format unification       │               │
│                              │     (JPEG/PNG)               │               │
│                              │   • Size optimization        │               │
│                              └──────────────┬───────────────┘               │
│                                             │                                │
│                                             ▼                                │
│                              ┌──────────────────────────────┐               │
│                              │   Rate Limiter               │               │
│                              │   (RPS Limiting &            │               │
│                              │    Concurrency Ctrl)         │               │
│                              │   (RPS 限制、并发控制)        │               │
│                              │   • RPS limit: prevent       │               │
│                              │     API overload             │               │
│                              │   • Concurrency control      │               │
│                              │   • Token Bucket algorithm   │               │
│                              └──────────────┬───────────────┘               │
│                                             │                                │
│                                             ▼                                │
│                              ┌──────────────────────────────┐               │
│                              │   Multimodal LLM             │               │
│                              │   Inference Service          │               │
│                              │   (Fireworks API / sglang)   │               │
│                              │   (多模态 LLM 推理服务)        │               │
│                              │   • Model: Qwen2.5-VL-32B    │               │
│                              │   • Service: Fireworks API   │               │
│                              │     or sglang serve          │               │
│                              │   • Call latency: 2-8 sec    │               │
│                              │   • Token usage: ~1000/req   │               │
│                              │   橙色背景 #f97316 (突出显示) │               │
│                              └──────────────┬───────────────┘               │
│                                             │                                │
│                                             ▼                                │
│                              ┌──────────────────────────────┐               │
│                              │   Schema Validator           │               │
│                              │   (Pydantic Validation)      │               │
│                              │   (Pydantic 验证)            │               │
│                              │   • Field type checking      │               │
│                              │   • Required field           │               │
│                              │     validation               │               │
│                              │   • Format validation        │               │
│                              └──────────────┬───────────────┘               │
│                                             │                                │
│                                             ▼                                │
│                              ┌──────────────────────────────┐               │
│                              │   Deterministic Rules        │               │
│                              │   (Deterministic Rule        │               │
│                              │    Engine)                   │               │
│                              │   (确定性规则引擎)            │               │
│                              │   • Format validation        │               │
│                              │     (date, ID)               │               │
│                              │   • Consistency check        │               │
│                              │     (front/back)             │               │
│                              │   • Logic validation         │               │
│                              │     (DOB < expiry)           │               │
│                              └──────────────┬───────────────┘               │
│                                             │                                │
│                                             ▼                                │
│                              ┌──────────────────────────────┐               │
│                              │   Output                     │               │
│                              │   (Structured Results +      │               │
│                              │    Decision)                 │               │
│                              │   (结构化结果 + 决策)        │               │
│                              │   • Extracted structured     │               │
│                              │     data                     │               │
│                              │   • Decision (approve/       │               │
│                              │     reject/manual)           │               │
│                              │   • Trace ID (for tracking)  │               │
│                              └──────────────────────────────┘               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 底部说明区域                                                            │ │
│  │                                                                        │ │
│  │ 🔄 Data Flow Description / 数据流说明                                 │ │
│  │ • Input: Batch uploaded document images (JPG, PNG formats supported)  │ │
│  │ • Processing: Preprocessing → Rate limiting → LLM inference →         │ │
│  │   Validation → Rule engine                                            │ │
│  │ • Output: Structured KYC data + Automated decision result             │ │
│  │                                                                        │ │
│  │ ⏱️ Component Latency / 各组件耗时                                     │ │
│  │ • Preprocessor: 100-200ms                                             │ │
│  │ • Rate Limiter Acquire: 0-1000ms (depends on current load)           │ │
│  │ • LLM Inference API Call: 2000-8000ms (main latency source) ⚠️       │ │
│  │   (Fireworks API or sglang serve)                                     │ │
│  │ • Schema Validation: 50-100ms                                         │ │
│  │ • Deterministic Rules: 10-50ms                                        │ │
│  │ • Save Result: 20-50ms                                                │ │
│  │ • Total p95: 8-10 seconds                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 文字内容（直接复制）：

**标题栏（深蓝色背景，白色文字）：**
```
System Architecture Design
系统架构设计
```

**架构流程图（垂直排列，从上到下）：**

```
┌──────────────────────────────┐
│   Batch Input                │
│   (Batch Document Input)     │
│   (批量文档输入)              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Preprocessor               │
│   (Image Preprocessing &     │
│    Conversion)               │
│   (图片预处理、格式转换)      │
│   • Image loading &          │
│     normalization            │
│   • Format unification       │
│     (JPEG/PNG)               │
│   • Size optimization        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Rate Limiter               │
│   (RPS Limiting &            │
│    Concurrency Ctrl)         │
│   (RPS 限制、并发控制)        │
│   • RPS limit: prevent       │
│     API overload             │
│   • Concurrency control      │
│   • Token Bucket algorithm   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Multimodal LLM             │
│   Inference Service          │
│   (Fireworks API / sglang)   │
│   (多模态 LLM 推理服务)        │
│   • Model: Qwen2.5-VL-32B    │
│   • Service: Fireworks API   │
│     or sglang serve          │
│   • Call latency: 2-8 sec    │
│   • Token usage: ~1000/req   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Schema Validator           │
│   (Pydantic Validation)      │
│   (Pydantic 验证)            │
│   • Field type checking      │
│   • Required field           │
│     validation               │
│   • Format validation        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Deterministic Rules        │
│   (Deterministic Rule        │
│    Engine)                   │
│   (确定性规则引擎)            │
│   • Format validation        │
│     (date, ID)               │
│   • Consistency check        │
│     (front/back)             │
│   • Logic validation         │
│     (DOB < expiry)           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Output                     │
│   (Structured Results +      │
│    Decision)                 │
│   (结构化结果 + 决策)        │
│   • Extracted structured     │
│     data                     │
│   • Decision (approve/       │
│     reject/manual)           │
│   • Trace ID (for tracking)  │
└──────────────────────────────┘
```

**底部说明区域：**

```
🔄 Data Flow Description / 数据流说明
• Input: Batch uploaded document images (JPG, PNG formats supported)
  输入: 批量上传的证件图片（支持 JPG、PNG 格式）
• Processing: Preprocessing → Rate limiting → LLM inference → Validation → Rule engine
  处理: 经过预处理、限流、LLM推理、验证、规则引擎
• Output: Structured KYC data + Automated decision result
  输出: 结构化的 KYC 数据 + 自动化决策结果

⏱️ Component Latency / 各组件耗时
• Preprocessor: 100-200ms
• Rate Limiter Acquire: 0-1000ms (depends on current load / 取决于当前负载)
• LLM Inference API Call: 2000-8000ms (main latency source / 主要延迟来源) ⚠️
  (Fireworks API or sglang serve / Fireworks API 或 sglang serve)
• Schema Validation: 50-100ms
• Deterministic Rules: 10-50ms
• Save Result: 20-50ms
• Total p95: 8-10 seconds / 总计 p95: 8-10 秒
```

---

## 颜色代码速查表

### 主要颜色
- **深蓝色 (标题栏)**: #1e3a8a
- **蓝色 (渐变/强调)**: #3b82f6
- **浅蓝色 (组件背景)**: #e0f2fe
- **紫色 (渐变)**: #8b5cf6
- **橙色 (强调组件)**: #f97316
- **白色 (文字/背景)**: #FFFFFF
- **浅灰色 (背景渐变)**: #f5f5f5 → #e8e8e8
- **黑色 (正文)**: #000000

### 字体大小
- **标题**: 32pt, 加粗
- **小标题**: 24pt, 加粗
- **正文**: 18pt
- **说明文字**: 14pt
- **组件名称**: 20pt, 加粗

### 布局比例
- **Slide 1**: 左侧文字区 60%, 右侧配图区 40%
- **Slide 2**: 上部技术栈, 中部核心能力(表格), 下部核心成果
- **Slide 3**: 顶部标题, 中间架构流程图(垂直居中), 底部说明区

---

## 制作提示

1. **Slide 1**: 右侧配图可以用传统流程 vs 自动化流程的对比图，或者三个痛点的数据可视化图表

2. **Slide 2**: 
   - 顶部可以用渐变色带装饰
   - 表格可以添加浅灰色隔行背景
   - 底部三个指标可以用图标 + 数字的卡片式设计

3. **Slide 3**: 
   - 每个组件用圆角矩形卡片
   - Multimodal LLM Inference Service 组件用橙色背景突出显示
   - 箭头用蓝色，可以添加箭头动画
   - 底部可以用时间线的方式展示各组件耗时

4. **动画建议**:
   - Slide 1: 内容从左到右淡入，配图从右到左淡入
   - Slide 2: 三个区域依次出现
   - Slide 3: 架构组件从上到下依次出现，箭头跟随出现

5. **注意事项**:
   - 保持整体配色一致
   - 留白要充足，不要塞得太满
   - 英文和中文要清晰可读
   - 关键数字要用大字体突出显示
