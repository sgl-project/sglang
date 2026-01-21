# KYC System Architecture - 5 Layers Structure
# KYC 系统架构 - 5层结构（架构视角）

基于系统架构视角的分层结构，每个Layer负责不同的职责和抽象层次

---

## Layer 1: API/Interface Layer
## Layer 1: API/接口层

### 职责（Responsibility）
- **对外接口**：提供HTTP API接口，接收客户端请求
- **请求验证**：验证请求格式、参数、认证
- **协议转换**：将HTTP请求转换为内部数据结构
- **响应封装**：将内部处理结果封装为HTTP响应

### 组件（Components）
- **FastAPI Service** (FastAPI服务)
  - HTTP endpoints (POST /v1/kyc/extract)
  - Request validation (Pydantic models)
  - Authentication & Authorization
  - Rate limiting (第一层限流)
  - Error handling (统一错误处理)
  - Response formatting (统一响应格式)

### 关键特性（Key Features）
- **RESTful API设计**：符合REST规范的接口设计
- **异步处理**：支持异步请求处理，提高并发能力
- **请求验证**：使用Pydantic进行请求参数验证
- **统一错误处理**：标准化的错误响应格式

### 设计原则（Design Principles）
- **单一职责**：只负责HTTP接口层的职责
- **薄层设计**：尽量薄，不做业务逻辑处理
- **协议无关**：内部不依赖具体的HTTP实现

### 与其他Layer的交互
- **输入**：接收HTTP请求（multipart/form-data）
- **输出**：调用 Application Layer 进行处理
- **依赖**：Application/Orchestration Layer

---

## Layer 2: Application/Orchestration Layer
## Layer 2: 应用/编排层

### 职责（Responsibility）
- **业务流程编排**：协调各个组件的执行顺序
- **业务逻辑处理**：实现核心业务逻辑
- **状态管理**：管理请求处理状态
- **错误恢复**：处理错误并决定恢复策略
- **批处理管理**：管理批量文件的处理流程

### 组件（Components）
- **Pipeline Orchestrator** (Pipeline编排器)
  - `pipeline.py`：核心编排逻辑
  - Batch processing (批量处理)
  - Per-file isolation (文件级隔离)
  - Error handling & recovery (错误处理和恢复)
  - Retry logic (重试逻辑)

- **Preprocessor** (预处理组件)
  - `preprocessor.py`：图片预处理
  - Image loading & normalization
  - Format conversion
  - Size optimization

- **Rate Limiter** (限流组件)
  - `rate_limiter.py`：请求限流
  - RPS limiting
  - Concurrency control
  - Token Bucket algorithm

### 关键特性（Key Features）
- **工作流编排**：定义清晰的执行流程
- **故障隔离**：Per-File Isolation，单个文件失败不影响其他文件
- **限流保护**：防止下游服务过载
- **异步处理**：支持异步批量处理

### 设计原则（Design Principles）
- **编排而非实现**：负责协调，不负责具体实现
- **可观测性**：记录每个步骤的trace信息
- **容错设计**：每步都有错误处理和降级方案

### 与其他Layer的交互
- **输入**：从 API Layer 接收请求
- **输出**：调用 External Service Layer 和 Validation Layer
- **依赖**：External Service Layer, Validation Layer, Storage Layer

---

## Layer 3: External Service Layer
## Layer 3: 外部服务层

### 职责（Responsibility）
- **外部服务抽象**：封装外部服务调用
- **服务适配**：适配不同的外部服务提供商
- **调用管理**：管理API调用、重试、超时
- **服务降级**：当服务不可用时提供降级方案

### 组件（Components）
- **LLM Inference Service** (LLM推理服务)
  - `fw_client.py`：Fireworks API客户端
  - Model selection (模型选择)
  - API call management (API调用管理)
  - Retry logic (重试逻辑)
  - Timeout handling (超时处理)

- **Service Abstraction** (服务抽象)
  - Support for Fireworks API
  - Support for sglang serve (local)
  - Provider switching (提供商切换)
  - Fallback mechanism (降级机制)

### 关键特性（Key Features）
- **服务解耦**：应用层不直接依赖具体的外部服务
- **多提供商支持**：支持Fireworks API和本地sglang serve
- **容错设计**：自动重试、超时处理、降级方案
- **成本控制**：监控API调用成本和token使用

### 设计原则（Design Principles）
- **接口抽象**：定义统一的服务接口
- **可替换性**：可以替换不同的服务提供商
- **透明性**：对上层隐藏服务实现的细节

### 与其他Layer的交互
- **输入**：从 Application Layer 接收处理请求
- **输出**：返回LLM推理结果
- **依赖**：Fireworks API 或 sglang serve (外部服务)

---

## Layer 4: Validation/Processing Layer
## Layer 4: 验证/处理层

### 职责（Responsibility）
- **数据验证**：验证LLM返回的数据结构
- **规则执行**：执行业务规则和逻辑验证
- **数据转换**：将原始数据转换为标准格式
- **决策生成**：基于验证结果生成决策

### 组件（Components）
- **Schema Validator** (Schema验证器)
  - `validators.py`：Pydantic模型验证
  - Field type checking (字段类型检查)
  - Required fields validation (必填字段验证)
  - Format validation (格式校验)
  - Schema versioning (Schema版本管理)

- **Deterministic Rules Engine** (确定性规则引擎)
  - `rules.py`：业务规则执行
  - Format validation (日期、ID格式验证)
  - Consistency check (前后页信息一致性检查)
  - Logic validation (业务逻辑验证，如DOB < expiry date)
  - Rule versioning (规则版本管理)

### 关键特性（Key Features）
- **Schema-First设计**：所有输出都符合预定义的Schema
- **确定性规则**：基于规则的验证，可审计、可测试
- **可扩展性**：易于添加新的验证规则
- **版本管理**：支持Schema和规则的版本管理

### 设计原则（Design Principles）
- **可审计性**：所有验证决策都有明确的规则依据
- **可测试性**：规则可以独立测试
- **可维护性**：规则集中管理，易于修改

### 与其他Layer的交互
- **输入**：从 External Service Layer 接收LLM结果
- **输出**：验证后的结构化数据和决策结果
- **依赖**：Storage Layer（保存验证结果）

---

## Layer 5: Storage/Persistence Layer
## Layer 5: 存储/持久化层

### 职责（Responsibility）
- **数据持久化**：保存处理结果和中间数据
- **文件存储**：存储文档图片和处理结果
- **元数据管理**：管理trace_id、版本等元数据
- **数据查询**：提供数据查询和检索功能

### 组件（Components）
- **File Storage** (文件存储)
  - `io_utils.py`：文件读写工具
  - Output directory structure (输出目录结构)
  - Result file writing (结果文件写入)
  - Summary file generation (汇总文件生成)

- **Data Storage** (数据存储)
  - `_summary.json`：处理结果汇总
  - Trace information (追踪信息)
  - Metadata storage (元数据存储)
  - Audit logs (审计日志)

### 关键特性（Key Features）
- **可追溯性**：保存完整的处理轨迹
- **审计支持**：支持数据审计和合规要求
- **隐私保护**：不存储PII明文数据
- **结构化存储**：使用JSON等结构化格式

### 设计原则（Design Principles）
- **持久化保证**：确保关键数据不丢失
- **可审计性**：支持数据审计和追溯
- **隐私合规**：符合PII保护要求

### 与其他Layer的交互
- **输入**：从 Validation Layer 接收验证结果
- **输出**：保存到文件系统
- **依赖**：文件系统或数据库

---

## Layer间交互流程（Inter-Layer Flow）

```
Client (HTTP Request)
    ↓
┌─────────────────────────────────────────┐
│ Layer 1: API/Interface Layer            │
│ - FastAPI Service                       │
│ - Request validation                    │
│ - Protocol conversion                   │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Layer 2: Application/Orchestration      │
│ - Pipeline Orchestrator                 │
│ - Preprocessor                          │
│ - Rate Limiter                          │
└─────┬─────────────────┬─────────────────┘
      │                 │
      ↓                 ↓
┌─────────────┐  ┌──────────────────────┐
│ Layer 3:    │  │ Layer 4:             │
│ External    │→ │ Validation/          │
│ Service     │  │ Processing           │
│ - LLM API   │  │ - Schema Validator   │
│             │  │ - Rules Engine       │
└─────────────┘  └──────────┬───────────┘
                            ↓
                 ┌─────────────────────────┐
                 │ Layer 5:                │
                 │ Storage/Persistence     │
                 │ - File Storage          │
                 │ - Data Storage          │
                 └─────────────────────────┘
                            ↓
                      (Output Files)
```

---

## 各Layer的核心职责总结

| Layer | 核心职责 | 关键组件 | 设计原则 |
|-------|---------|---------|---------|
| **Layer 1: API/Interface** | HTTP接口、协议转换 | FastAPI Service | 薄层、单一职责 |
| **Layer 2: Application/Orchestration** | 业务流程编排、状态管理 | Pipeline, Preprocessor, Rate Limiter | 编排、容错 |
| **Layer 3: External Service** | 外部服务调用、服务抽象 | LLM Inference Service | 抽象、可替换 |
| **Layer 4: Validation/Processing** | 数据验证、规则执行 | Schema Validator, Rules Engine | 可审计、可测试 |
| **Layer 5: Storage/Persistence** | 数据持久化、文件存储 | File Storage, Data Storage | 可追溯、合规 |

---

## 设计亮点（Key Design Highlights）

### 1. **分层解耦**（Layered Decoupling）
- 每个Layer职责单一，边界清晰
- Layer间通过定义良好的接口交互
- 易于单独测试和维护

### 2. **可扩展性**（Extensibility）
- **Layer 1**: 易于添加新的API端点
- **Layer 2**: 易于添加新的编排步骤
- **Layer 3**: 易于切换不同的LLM服务提供商
- **Layer 4**: 易于添加新的验证规则
- **Layer 5**: 易于切换不同的存储后端

### 3. **容错设计**（Fault Tolerance）
- 每层都有独立的错误处理机制
- Layer 2的Per-File Isolation确保故障隔离
- Layer 3的服务降级机制确保系统可用性

### 4. **可观测性**（Observability）
- 每个Layer都记录trace信息
- 通过trace_id关联跨Layer的调用
- 支持Metrics/Logs/Traces三层监控

### 5. **可测试性**（Testability）
- 每层都可以独立测试
- 通过Mock和Stub隔离依赖
- 支持单元测试、集成测试、端到端测试

---

## 讲解建议（Presentation Tips）

### 逐层讲解（每层2-3分钟）

1. **Layer 1: API/Interface Layer** (2分钟)
   - 强调薄层设计，只负责HTTP接口
   - 展示RESTful API设计和Pydantic验证
   - 说明统一错误处理的重要性

2. **Layer 2: Application/Orchestration Layer** (3分钟) ⭐
   - **重点讲解**：这是核心编排层
   - 展示Pipeline编排逻辑
   - 强调Per-File Isolation的设计
   - 说明限流和容错机制

3. **Layer 3: External Service Layer** (2分钟)
   - 强调服务抽象和可替换性
   - 展示多提供商支持（Fireworks API / sglang）
   - 说明容错设计和降级机制

4. **Layer 4: Validation/Processing Layer** (2分钟)
   - 强调Schema-First设计
   - 展示确定性规则引擎
   - 说明可审计性和可测试性

5. **Layer 5: Storage/Persistence Layer** (1分钟)
   - 说明数据持久化和可追溯性
   - 强调隐私保护和合规要求

### 整体架构讲解（1-2分钟）
- 展示Layer间交互流程
- 强调分层解耦的优势
- 说明整体设计的可扩展性和容错性

---

## 与数据流Layer的区别

| 视角 | 数据流Layer | 系统架构Layer |
|------|------------|--------------|
| **关注点** | 数据如何流动 | 系统如何组织 |
| **Layer划分** | 按处理步骤 | 按职责和抽象层次 |
| **Layer 1** | Input & Preprocessing | API/Interface |
| **Layer 2** | Rate Limiting | Application/Orchestration |
| **Layer 3** | Inference | External Service |
| **Layer 4** | Validation | Validation/Processing |
| **Layer 5** | Output & Storage | Storage/Persistence |
| **适用场景** | 讲解处理流程 | 讲解系统设计 |

---

## 总结

这是基于**系统架构视角**的分层结构，每个Layer代表不同的职责和抽象层次：

- **Layer 1**：对外接口，薄层设计
- **Layer 2**：核心编排，业务流程
- **Layer 3**：外部服务，抽象封装
- **Layer 4**：数据验证，规则执行
- **Layer 5**：数据持久化，可追溯性

通过清晰的分层，实现了**高内聚、低耦合**的系统设计，便于扩展、测试和维护。
