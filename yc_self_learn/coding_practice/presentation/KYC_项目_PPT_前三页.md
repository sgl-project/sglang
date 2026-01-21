# KYC Project Presentation - First Three Slides
# KYC 项目演示文稿 - 前三页

---

## Slide 1: Project Background & Core Problems
## Slide 1: 项目背景和问题

### Design Notes / 设计说明
- **Background Color / 背景色**: Light gray gradient (#f5f5f5 to #e8e8e8)
- **Header Area / 标题区域**: Dark blue (#1e3a8a), white text
- **Layout / 布局**: Text on left (60%), illustration on right (40%) - KYC workflow diagram or pain points visualization
- **Typography / 字体**: Title 32pt bold, body text 18pt

### Slide Content / 页面内容

**Title / 标题**: 
- **KYC Project Background & Core Problems**
- **KYC 项目背景与核心问题**

**Content Area (Left 60%) / 正文区域（左侧，占60%宽度）**:

#### 📋 Business Background / 业务背景（30 seconds / 30秒讲解）

**English / 英文**:
- KYC (Know Your Customer) review is a core compliance process in financial services
- Reviewers need to extract key information from large volumes of ID cards, passports, and other documents
- Traditional workflow relies on manual review, which is time-consuming and error-prone

**中文对照**:
- KYC（Know Your Customer）审核是金融服务的核心合规流程
- 审核人员需要从大量身份证件、护照等文档中提取关键信息
- 传统流程依赖人工审核，耗时且容易出错

---

#### 💡 Pain Points Analysis / 痛点分析（40 seconds / 40秒讲解）

**1. Low Efficiency / 效率低下**

**English / 英文**:
- Manual review takes an average of 10 minutes per case
- Repetitive work consumes significant reviewer time

**中文对照**:
- 人工审核每单平均耗时 10 分钟
- 大量重复性工作占用审核人员时间

**2. High Cost / 成本高昂**

**English / 英文**:
- Manual review cost: $0.50 per case
- Human resource costs grow linearly with business growth

**中文对照**:
- 人工审核成本：$0.50/单
- 随着业务增长，人力成本呈线性增长

**3. Inconsistent Quality / 质量不稳定**

**English / 英文**:
- Human fatigue leads to increased error rates
- Difficult to guarantee 100% accuracy

**中文对照**:
- 人工疲劳导致错误率上升
- 难以保证100%准确性

---

#### 🎯 Project Goals / 项目目标（20 seconds / 20秒讲解）

**English / 英文**:
- **Automate KYC review process** to reduce manual intervention
- **Improve processing efficiency** and shorten review time
- **Ensure accuracy and consistency**, reduce error rates

**中文对照**:
- **自动化 KYC 审核流程**，减少人工干预
- **提升处理效率**，缩短审核时间
- **保证准确性和一致性**，降低错误率

**Illustration Area (Right 40%) / 配图区域（右侧，占40%宽度）**:
- Optional: KYC workflow comparison diagram (Traditional vs Automated)
- Or: Pain points data visualization chart

---

---

## Slide 2: Solution Overview
## Slide 2: 解决方案概述

### Design Notes / 设计说明
- **Background Color / 背景色**: White, gradient header (blue #3b82f6 to purple #8b5cf6)
- **Header Area / 标题区域**: Dark blue (#1e3a8a), white text
- **Layout / 布局**: Three-section layout (top, middle, bottom)
- **Typography / 字体**: Title 32pt, subtitles 24pt, body 18pt
- **Color Scheme / 配色方案**: Blue and purple as primary colors to highlight technical advancement

### Slide Content / 页面内容

**Title / 标题**:
- **Automated Solution Based on Multimodal LLM**
- **基于多模态 LLM 的自动化解决方案**

---

**Top Section: Tech Stack / 上部区域: 技术栈**（40 seconds / 40秒讲解）

#### 🤖 Core Technology / 核心技术

**English / 英文**:
```
Multimodal LLM: Qwen2.5-VL-32B
    ↓
Fireworks API (Inference Service)
    ↓
Python + Pydantic + FastAPI
```

**中文对照**:
```
多模态 LLM: Qwen2.5-VL-32B
    ↓
Fireworks API (推理服务)
    ↓
Python + Pydantic + FastAPI
```

---

**Middle Section: Core Capabilities / 中部区域: 核心能力**（40 seconds / 40秒讲解）

#### 🔍 Four Core Capabilities / 四大核心能力

| Capability / 能力 | Description / 说明 | Value / 价值 |
|-------------------|-------------------|--------------|
| 📄 **Document OCR / 文档 OCR** | Identify and extract text from images / 识别和提取图像中的文字信息 | Support multiple document formats / 支持多种证件格式 |
| 🔎 **Field Extraction / 字段提取** | Structured extraction of key info (name, DOB, ID number) / 结构化提取姓名、生日、证件号等关键信息 | Standardized output format / 标准化输出格式 |
| ✅ **Auto Validation / 自动验证** | Deterministic rule-based secondary validation / 基于确定性规则的二次验证 | Ensure data accuracy / 确保数据准确性 |
| 🎯 **Smart Decision / 智能决策** | Auto approve/reject/escalate to human / 自动通过/拒绝/转人工审核 | Reduce manual intervention / 减少人工干预 |

---

**Bottom Section: Key Results / 下部区域: 核心成果**（30 seconds / 30秒讲解）

#### 📊 Key Metrics / 关键指标

**English / 英文**:
- ✅ **Time Saved per Case**: 5 minutes (Manual 10 min → Automated 5 min)
- ✅ **Automation Rate**: 95% (90% auto-approve + 5% auto-reject)
- ✅ **Cost Savings**: $0.498 per case (API cost $0.002 vs Manual cost $0.50)

**中文对照**:
- ✅ **每单节省时间**: 5 分钟（人工 10 分钟 → 自动化 5 分钟）
- ✅ **自动化率**: 95%（90% 自动通过 + 5% 自动拒绝）
- ✅ **成本节省**: $0.498/单（API成本 $0.002 vs 人工成本 $0.50）

**Visual Elements / 视觉元素**:
- Top right: Tech stack icons (LLM model, API, code symbols)
- Bottom: Results data visualization (bar chart or pie chart)

---

---

## Slide 3: System Architecture Diagram (Detailed)
## Slide 3: 系统架构图（详细版）

### Design Notes / 设计说明
- **Background Color / 背景色**: Pure white (#ffffff)
- **Header Area / 标题区域**: Dark blue (#1e3a8a), white text
- **Layout / 布局**: Vertical flowchart, card-style design for each component
- **Component Style / 组件样式**: Rounded rectangles, light blue background (#e0f2fe), dark blue border
- **Arrows / 箭头**: Blue arrows (#3b82f6) indicating data flow direction
- **Typography / 字体**: Component names 20pt bold, description text 14pt

### Slide Content / 页面内容

**Title / 标题**:
- **System Architecture Design**
- **系统架构设计**

**Architecture Flowchart (Vertical, Centered) / 架构流程图（垂直布局，居中显示）**:

```
┌─────────────────────────────────────┐
│         Batch Input                 │
│    (Batch Document Input)           │
│    (批量文档输入)                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│        Preprocessor                 │
│   (Image Preprocessing & Conversion)│
│   (图片预处理、格式转换)             │
│   • Image loading & normalization   │
│   • Format unification (JPEG/PNG)   │
│   • Size optimization                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       Rate Limiter                  │
│   (RPS Limiting & Concurrency Ctrl) │
│   (RPS 限制、并发控制)               │
│   • RPS limit: prevent API overload │
│   • Concurrency control              │
│   • Token Bucket algorithm           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Fireworks API                  │
│   (Multimodal LLM Inference)        │
│   (多模态 LLM 推理)                  │
│   • Model: Qwen2.5-VL-32B           │
│   • Call latency: 2-8 seconds       │
│   • Token usage: ~1000 tokens/req   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     Schema Validator                 │
│   (Pydantic Validation)             │
│   (Pydantic 验证)                    │
│   • Field type checking             │
│   • Required field validation        │
│   • Format validation                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Deterministic Rules              │
│   (Deterministic Rule Engine)       │
│   (确定性规则引擎)                   │
│   • Format validation (date, ID)    │
│   • Consistency check (front/back)   │
│   • Logic validation (DOB < expiry)  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         Output                      │
│   (Structured Results + Decision)   │
│   (结构化结果 + 决策)                │
│   • Extracted structured data        │
│   • Decision (approve/reject/manual) │
│   • Trace ID (for tracking)          │
└─────────────────────────────────────┘
```

**Bottom Explanation Area / 底部说明区域**:

#### 🔄 Data Flow Description / 数据流说明

**English / 英文**:
- **Input**: Batch uploaded document images (JPG, PNG formats supported)
- **Processing**: Preprocessing → Rate limiting → LLM inference → Validation → Rule engine
- **Output**: Structured KYC data + Automated decision result

**中文对照**:
- **输入**: 批量上传的证件图片（支持 JPG、PNG 格式）
- **处理**: 经过预处理、限流、LLM推理、验证、规则引擎
- **输出**: 结构化的 KYC 数据 + 自动化决策结果

---

#### ⏱️ Component Latency / 各组件耗时

**English / 英文**:
- **Preprocessor**: 100-200ms
- **Rate Limiter Acquire**: 0-1000ms (depends on current load)
- **Fireworks API Call**: 2000-8000ms (main latency source)
- **Schema Validation**: 50-100ms
- **Deterministic Rules**: 10-50ms
- **Save Result**: 20-50ms
- **Total p95**: 8-10 seconds

**中文对照**:
- **Preprocessor**: 100-200ms
- **Rate Limiter Acquire**: 0-1000ms（取决于当前负载）
- **Fireworks API Call**: 2000-8000ms（主要延迟来源）
- **Schema Validation**: 50-100ms
- **Deterministic Rules**: 10-50ms
- **Save Result**: 20-50ms
- **总计 p95**: 8-10 秒

**Visual Enhancements / 视觉增强**:
- Each component card can use different shades of blue
- Key components (Fireworks API) can be highlighted with more prominent color (e.g., orange #f97316)
- Arrows can have animation effects (in PPT)
- Add timeline visualization at the bottom

---

---

## Design Recommendations Summary
## 设计建议总结

### Overall Design Style / 整体设计风格

1. **Color Scheme / 配色方案**: 
   - Primary: Dark blue (#1e3a8a) / 主色：深蓝色（#1e3a8a）
   - Secondary: Blue (#3b82f6), Light blue (#e0f2fe) / 辅助色：蓝色（#3b82f6）、浅蓝色（#e0f2fe）
   - Accent: Purple (#8b5cf6), Orange (#f97316) / 强调色：紫色（#8b5cf6）、橙色（#f97316）

2. **Typography Guidelines / 字体规范**:
   - Title: 32pt, bold / 标题：32pt，加粗
   - Subtitles: 24pt, bold / 小标题：24pt，加粗
   - Body text: 18pt / 正文：18pt
   - Description text: 14pt / 说明文字：14pt

3. **Visual Elements / 视觉元素**:
   - Use icons to enhance visual impact (emoji or professional icons) / 使用图标增强视觉效果（emoji 或专业图标）
   - Use shadows and rounded corners appropriately for modern design / 适当使用阴影和圆角，使设计更现代
   - Maintain consistent spacing and layout / 保持一致的间距和布局

4. **Data Visualization / 数据可视化**:
   - Use charts to display key metrics / 使用图表展示关键指标
   - Flowcharts should be clear and easy to understand / 流程图要清晰易懂
   - Highlight numbers (use large fonts or special colors) / 数字要突出显示（使用大字体或特殊颜色）

### Tool Recommendations / 制作工具建议

**English / 英文**:
- **PowerPoint/Keynote**: Professional presentation tools with animation support
- **Google Slides**: Online collaboration
- **Markdown + Marp**: Code-based PPT creation
- **Figma**: Professional design tool, can export as images for PPT

**中文对照**:
- **PowerPoint/Keynote**: 专业演示工具，支持动画
- **Google Slides**: 在线协作
- **Markdown + Marp**: 代码化制作PPT
- **Figma**: 专业设计工具，可导出为图片插入PPT

---

## Presentation Tips for English Interview / 英文面试演讲技巧

### Speaking Points / 讲解要点

**Slide 1 (1.5 minutes / 1.5分钟)**:
- Start with business context and explain why KYC automation is important
- Clearly articulate the three pain points with concrete examples
- State project goals clearly

**Slide 2 (1.5 minutes / 1.5分钟)**:
- Highlight the tech stack choice (why Qwen2.5-VL-32B, why Fireworks API)
- Explain each of the four core capabilities with real-world impact
- Emphasize the business results (time saved, automation rate, cost savings)

**Slide 3 (2 minutes / 2分钟)**:
- Walk through the architecture flow, explaining each component's purpose
- Discuss design trade-offs (why this architecture, why these components)
- Explain latency breakdown and optimization opportunities

### Key Phrases for English Presentation / 英文演讲关键短语

**Opening / 开场**:
- "Let me start by introducing the business context..."
- "Today I'll present a KYC automation system that..."

**Transitions / 过渡**:
- "Moving on to the solution..."
- "Now let's look at the system architecture..."
- "The key design principle here is..."

**Emphasizing Points / 强调要点**:
- "What's important here is..."
- "The key takeaway is..."
- "This is critical because..."

**Closing / 结尾**:
- "In summary..."
- "The main points to remember are..."
- "Are there any questions about..."
