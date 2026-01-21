# KYC Pipeline - 5 Layers Structure
# KYC Pipeline - 5层结构（基于实际System Design）

根据KYC项目的实际架构和Traces结构整理

---

## Layer 1: Input & Preprocessing Layer
## Layer 1: 输入与预处理层

### 组件（Components）
- **Batch Input** (批量文档输入)
  - Document images (JPG/PNG)
  - Batch upload
  - File format validation

- **Preprocessor** (图片预处理)
  - Image loading & normalization
  - Format conversion (统一格式)
  - Size optimization (尺寸优化)
  - Image format check

### 功能说明（Function）
- 接收批量上传的文档图片
- 对图片进行预处理和格式统一
- 确保图片符合后续处理要求

### 对应Span
- **Span 1: Preprocess (Image Format Check)**
  - Duration: 10-200ms
  - 检查图片格式、加载、规范化

---

## Layer 2: Rate Limiting & Orchestration Layer
## Layer 2: 限流与编排层

### 组件（Components）
- **Rate Limiter** (RPS限制与并发控制)
  - RPS limit (防止API过载)
  - Concurrency control (并发控制)
  - Token Bucket algorithm (令牌桶算法)
  - Request queuing (请求排队)

### 功能说明（Function）
- 控制请求速率，防止API过载
- 管理并发请求数量
- 保护下游服务稳定性

### 对应Span
- **Span 2: Rate Limit Acquire**
  - Duration: 0-1000ms (depends on load)
  - 获取令牌，等待处理机会

---

## Layer 3: Inference Layer
## Layer 3: 推理层（核心层）

### 组件（Components）
- **OCR/VLM** (OCR/视觉语言模型)
  - Fireworks API Call or sglang serve
  - Model: Qwen2.5-VL-32B
  - Image-to-text extraction
  - Duration: 50ms+

- **LLM Processing** (结构化输出)
  - Fireworks API Call or sglang serve
  - Model: Qwen2.5-VL-32B
  - Structured data extraction
  - Duration: 200-8000ms (主要延迟来源)

### 功能说明（Function）
- OCR/VLM提取图像中的文字信息
- LLM处理生成结构化输出
- 这是整个pipeline的核心层
- 主要延迟来源（2-8秒）

### 对应Span
- **Span 2: OCR/VLM (Fireworks API Call)**
  - Duration: 50ms+
  - 图像到文字的提取

- **Span 3: LLM Processing (Structured Output)**
  - Duration: 200-8000ms
  - 结构化数据提取
  - Tokens: ~1000/request
  - Cost: ~$0.002/request

---

## Layer 4: Validation Layer
## Layer 4: 验证层

### 组件（Components）
- **Schema Validator** (Pydantic验证)
  - Field type checking (字段类型检查)
  - Required fields validation (必填字段验证)
  - Format validation (格式校验)
  - Schema version: v1

- **Deterministic Rules** (确定性规则引擎)
  - Format validation (date, ID format)
  - Consistency check (前后页信息一致性)
  - Logic validation (DOB < expiry date)
  - Rule-based verification (基于规则的验证)

### 功能说明（Function）
- Schema验证确保数据结构正确
- 确定性规则引擎进行二次验证和逻辑检查
- 确保数据质量和业务规则合规

### 对应Span
- **Span 4: Validation (Schema + Rules)**
  - Duration: 5-150ms
  - Schema版本验证
  - 字段级验证
  - 业务规则验证

---

## Layer 5: Output & Storage Layer
## Layer 5: 输出与存储层

### 组件（Components）
- **Output** (结构化结果 + 决策)
  - Extracted structured data (提取的结构化数据)
  - Decision result (approve/reject/manual)
  - Trace ID (for tracking)
  - Metadata (version, timestamp, etc.)

- **Storage** (存储)
  - Write to `_summary.json`
  - Save to `output_results/`
  - Store trace information

### 功能说明（Function）
- 输出结构化的KYC数据
- 生成自动化决策结果
- 存储结果用于后续审计和追踪
- 包含追踪ID用于链路追踪

### 对应Span
- **Span 5: Storage (Write to _summary.json)**
  - Duration: 20-50ms
  - 写入结果文件
  - 保存追踪信息

---

## 数据流向（基于实际架构）

```
Layer 1: Input & Preprocessing
    • Batch Input (批量文档输入)
    • Preprocessor (图片预处理)
    ↓
Layer 2: Rate Limiting & Orchestration
    • Rate Limiter (RPS限制与并发控制)
    ↓
Layer 3: Inference ← 核心层
    • OCR/VLM (Fireworks API or sglang)
    • LLM Processing (结构化输出)
    ↓
Layer 4: Validation
    • Schema Validator (Pydantic验证)
    • Deterministic Rules (确定性规则引擎)
    ↓
Layer 5: Output & Storage
    • Output (结构化结果 + 决策)
    • Storage (写入_summary.json)
```

### 对应实际代码模块

根据学习指南中的架构图：

```
Batch Input → main.py → pipeline.py
    ↓
preprocessor.py (Layer 1)
    ↓
rate_limiter.py (Layer 2)
    ↓
fw_client.py → Fireworks API (Layer 3)
    ↓
validators.py (Layer 4: Schema Validator)
    ↓
rules.py (Layer 4: Deterministic Rules)
    ↓
io_utils.py → output_results/ + _summary.json (Layer 5)
```

---

## 各Layer延迟统计（基于实际Traces）

- **Layer 1 (Input & Preprocessing)**: 10-200ms
  - Preprocess: 10ms (Image Format Check)
  
- **Layer 2 (Rate Limiting)**: 0-1000ms
  - Rate Limit Acquire: 0-1000ms (depends on load)
  
- **Layer 3 (Inference)**: 250-8000ms ⚠️ (main latency source)
  - OCR/VLM: 50ms+
  - LLM Processing: 200-8000ms (structured output)
  
- **Layer 4 (Validation)**: 5-150ms
  - Schema Validation: 50-100ms
  - Deterministic Rules: 10-50ms
  
- **Layer 5 (Output & Storage)**: 20-50ms
  - Storage: 20ms (Write to _summary.json)
  
- **Total p95**: 8-10 seconds

---

## 讲解顺序建议（基于实际架构）

### 逐层讲解（每层1.5-2分钟）

1. **Layer 1: Input & Preprocessing** (1.5分钟)
   - 介绍输入格式（Batch Input）
   - 讲解预处理步骤（Preprocessor）
   - 说明图片格式检查和规范化

2. **Layer 2: Rate Limiting & Orchestration** (1.5分钟)
   - 讲解限流和并发控制的重要性
   - 说明Token Bucket算法
   - 强调保护下游服务的作用

3. **Layer 3: Inference** (2分钟) ⭐ **重点讲解**
   - OCR/VLM：图像到文字提取
   - LLM Processing：结构化输出
   - 说明这是整个pipeline的核心层
   - 强调主要延迟来源（2-8秒）

4. **Layer 4: Validation** (2分钟)
   - Schema Validator：数据结构验证
   - Deterministic Rules：确定性规则引擎
   - 讲解验证的重要性（可审计、可测试）

5. **Layer 5: Output & Storage** (1分钟)
   - 输出结构化结果和决策
   - 存储结果用于审计和追踪
   - Trace ID的作用

### 讲解技巧

- **强调核心层**：Layer 3是核心，延迟占比最大
- **说明trade-off**：验证层确保质量，但会增加延迟
- **展示可观测性**：每个Layer都有对应的Span，便于追踪
- **强调设计亮点**：Schema-First、确定性规则引擎、Per-File Isolation
