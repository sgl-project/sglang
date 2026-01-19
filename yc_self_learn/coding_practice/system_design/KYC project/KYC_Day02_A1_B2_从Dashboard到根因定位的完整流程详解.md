# Day 2_A1_B2：从 Dashboard 到根因定位的完整流程详解

---
doc_type: glossary
layer: L3
scope_in:  从 Dashboard 选择失败的请求、使用 request_id/trace_id 定位根因、查看 Logs 和 Traces 的详细流程
scope_out: 具体 Logs/Traces 查询实现（见 howto）；深入的错误分析（见 L4）
inputs:   (读者) 疑问：如何从 Dashboard 选择一个失败的请求，然后定位根因？
outputs:  从 Dashboard 到根因定位的完整流程 + request_id/trace_id 的作用 + Logs/Traces 查询 + 实际例子
entrypoints: [ 核心问题 ]
children: [ 
  KYC_Day02_A1_B2_C1_Traces和Span详解_链路追踪是什么.md（Traces 和 Span 详解：链路追踪是什么），
  KYC_Day02_A1_B2_C2_定位问题后的下一步行动详解.md（定位问题后的下一步行动详解），
  KYC_Day02_A1_B2_C4_LLM_Processing降速原因详解.md（LLM Processing 降速原因详解）
]
related: [ Dashboard, Metrics, Logs, Traces, request_id, trace_id, 根因定位, KYC_Day02_A1_可观测性详解.md ]
---

## Definition（定义）

**核心问题**：**如何从 Dashboard 选择一个失败的请求，然后定位根因？**

**核心答案**：
- ✅ **步骤 1**：从 Dashboard 选择失败的请求，获取 `request_id`、`trace_id`、`file_id`、`error_code`
- ✅ **步骤 2**：使用 `trace_id` 查看 **Traces（链路追踪）**，了解请求在系统中的完整调用链
- ✅ **步骤 3**：使用 `request_id`/`trace_id` 查看 **Logs（日志）**，查看详细的错误信息和上下文
- ✅ **步骤 4**：根据 Traces 和 Logs 的信息，定位根因并解决问题

**关键标识符**：
- ✅ **request_id**：请求的唯一标识符（用于追踪单个请求）
- ✅ **trace_id**：链路追踪的唯一标识符（用于追踪请求在分布式系统中的完整调用链）
- ✅ **file_id**：文件的唯一标识符（用于追踪单个文件）
- ✅ **error_code**：错误代码（用于快速识别错误类型）

---

## 🎯 核心问题

### 从 Dashboard 到根因定位的完整流程

**场景**：你在 Dashboard 上看到 Error Rate 突然升高，需要定位根因。

**完整流程**：
```
1. Dashboard 发现问题
   ↓
2. 选择一个失败的请求，获取关键标识符
   ↓
3. 使用 trace_id 查看 Traces（链路追踪）
   ↓
4. 使用 request_id/trace_id 查看 Logs（日志）
   ↓
5. 根据 Traces 和 Logs 定位根因
   ↓
6. 解决问题
```

---

## 📊 详细步骤

### 步骤 1：从 Dashboard 选择失败的请求

**场景**：你在 Dashboard 上看到 Error Rate 突然升高。

**操作步骤**：

1. **打开 Dashboard**（Grafana/Datadog）
   - 访问 Dashboard：http://localhost:3000
   - 查看 Error Rate 图表
   - 发现 Error Rate 突然从 1% 升高到 5%

2. **点击失败的请求**（在 Dashboard 上）
   - 点击 Error Rate 图表中的某个时间点
   - 或点击某个失败请求的详情
   - 查看失败的请求列表

3. **获取关键标识符**：
   ```
   - request_id: req-abc123（请求的唯一标识符）
   - trace_id: trace-abc123（链路追踪的唯一标识符）
   - file_id: doc_001.jpg（文件的唯一标识符）
   - error_code: SCHEMA_VALIDATION_FAILED（错误代码）
   ```

**关键点**：
- ✅ **request_id**：用于追踪单个请求的所有信息
- ✅ **trace_id**：用于追踪请求在分布式系统中的完整调用链
- ✅ **file_id**：用于追踪单个文件的所有信息
- ✅ **error_code**：用于快速识别错误类型

---

### 步骤 2：使用 trace_id 查看 Traces（链路追踪）

**目的**：了解请求在系统中的完整调用链，找出哪个环节出了问题。

**操作步骤**：

1. **打开 Tracing Dashboard**（Jaeger/Zipkin/Datadog APM）
   - 访问 Jaeger：http://localhost:16686
   - 或访问 Datadog APM：https://app.datadoghq.com/apm

2. **输入 trace_id**：
   ```
   trace_id: trace-abc123
   ```

3. **查看 Trace 详情**（完整的调用链）：
   ```
   Trace: trace-abc123
   ├─ Span 1: API Gateway（入口）
   │  ├─ latency: 5ms
   │  └─ status: OK
   │
   ├─ Span 2: KYC Service（主服务）
   │  ├─ latency: 100ms
   │  └─ status: OK
   │
   ├─ Span 3: Preprocessing（预处理）
   │  ├─ latency: 50ms
   │  └─ status: OK
   │
   ├─ Span 4: LLM Inference（LLM 推理）
   │  ├─ latency: 5000ms
   │  └─ status: OK
   │
   ├─ Span 5: Schema Validation（Schema 验证）← 问题在这里！
   │  ├─ latency: 10ms
   │  ├─ status: ERROR
   │  └─ error_code: SCHEMA_VALIDATION_FAILED
   │
   └─ Span 6: Post-processing（后处理）
      └─ status: SKIPPED（因为验证失败）
   ```

4. **定位问题**：
   - 看到 Span 5（Schema Validation）失败了
   - 错误代码：`SCHEMA_VALIDATION_FAILED`
   - 这是问题的根因

**关键点**：
- ✅ **Traces 显示完整的调用链**：从入口到出口的每个步骤
- ✅ **每个 Span 都有 latency 和 status**：可以快速看出哪个环节慢或失败了
- ✅ **trace_id 是全局唯一的**：可以追踪请求在分布式系统中的所有组件

---

### 步骤 3：使用 request_id/trace_id 查看 Logs（日志）

**目的**：查看详细的错误信息和上下文，了解为什么会失败。

**操作步骤**：

1. **打开 Logs Dashboard**（ELK/Kibana/Datadog Logs）
   - 访问 Kibana：http://localhost:5601
   - 或访问 Datadog Logs：https://app.datadoghq.com/logs

2. **输入 request_id 或 trace_id**：
   ```
   request_id: req-abc123
   或
   trace_id: trace-abc123
   ```

3. **查看 Logs 详情**（详细的错误信息和上下文）：
   ```json
   {
     "timestamp": "2024-01-15T10:30:00Z",
     "level": "ERROR",
     "request_id": "req-abc123",
     "trace_id": "trace-abc123",
     "file_id": "doc_001.jpg",
     "span_name": "Schema Validation",
     "error_code": "SCHEMA_VALIDATION_FAILED",
     "error_message": "Schema validation failed: missing required field 'user_name'",
     "llm_output": {
       "status": "success",
       "result": {
         "user_id": "u123",
         "birth_date": "1990-01-01"
         // 缺少 "user_name" 字段
       }
     },
     "schema_expected": {
       "user_id": "string",
       "user_name": "string",  // 必需字段
       "birth_date": "string"
     },
     "schema_actual": {
       "user_id": "string",
       "birth_date": "string"
       // 缺少 "user_name" 字段
     }
   }
   ```

4. **定位根因**：
   - 看到错误信息：`"missing required field 'user_name'"`
   - LLM 输出缺少 `user_name` 字段
   - Schema 验证失败，因为 `user_name` 是必需字段

**关键点**：
- ✅ **Logs 包含详细的错误信息**：错误消息、上下文、输入输出等
- ✅ **request_id/trace_id 关联所有相关的 Logs**：可以看到请求在整个系统中的所有日志
- ✅ **结构化日志（JSON）**：方便查询和分析

---

### 步骤 4：根据 Traces 和 Logs 定位根因

**分析过程**：

1. **从 Traces 看出**：
   - Schema Validation 环节失败了
   - 错误代码：`SCHEMA_VALIDATION_FAILED`

2. **从 Logs 看出**：
   - LLM 输出缺少 `user_name` 字段
   - Schema 验证失败，因为 `user_name` 是必需字段

3. **根因分析**：
   ```
   根因：LLM 输出格式不符合 Schema 要求
   
   具体原因：
   - LLM 推理成功（status: "success"）
   - 但输出缺少必需字段 "user_name"
   - Schema 验证失败（因为 user_name 是必需字段）
   - 请求失败（error_code: SCHEMA_VALIDATION_FAILED）
   ```

4. **可能的解决方案**：
   - **方案 1**：修改 LLM Prompt，明确要求输出 `user_name` 字段
   - **方案 2**：调整 Schema，将 `user_name` 改为可选字段（如果业务允许）
   - **方案 3**：在 Schema 验证失败时，触发 Fallback 或人工审核

---

## 💡 实际例子（KYC 项目）

### 完整流程示例

**场景**：你在 Dashboard 上看到 Error Rate 突然从 1% 升高到 5%，需要定位根因。

**步骤 1：Dashboard 发现问题**

```
打开 Dashboard（Grafana）：
- Error Rate: 5%（突然升高，平时是 1%）
- Error Type: SCHEMA_VALIDATION_FAILED（占 80%）
- Time Range: 过去 1 小时

点击某个失败的请求，获取：
- request_id: req-abc123
- trace_id: trace-abc123
- file_id: doc_001.jpg
- error_code: SCHEMA_VALIDATION_FAILED
```

---

**步骤 2：查看 Traces（链路追踪）**

```
打开 Jaeger，输入 trace_id: trace-abc123

查看 Trace 详情：
┌─────────────────────────────────────────────────────────┐
│ Trace: trace-abc123                                      │
│ Total Duration: 5165ms                                   │
├─────────────────────────────────────────────────────────┤
│ Span 1: API Gateway                      [5ms]  ✅ OK   │
│ Span 2: KYC Service                      [100ms] ✅ OK  │
│ Span 3: Preprocessing                    [50ms]  ✅ OK  │
│ Span 4: LLM Inference                    [5000ms] ✅ OK │
│ Span 5: Schema Validation                [10ms]  ❌ ERROR │
│   └─ error_code: SCHEMA_VALIDATION_FAILED               │
│ Span 6: Post-processing                  [SKIPPED]      │
└─────────────────────────────────────────────────────────┘

结论：Schema Validation 环节失败了
```

---

**步骤 3：查看 Logs（日志）**

```
打开 Kibana，输入 request_id: req-abc123

查看 Logs 详情：
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "ERROR",
  "request_id": "req-abc123",
  "trace_id": "trace-abc123",
  "file_id": "doc_001.jpg",
  "span_name": "Schema Validation",
  "error_code": "SCHEMA_VALIDATION_FAILED",
  "error_message": "Schema validation failed: missing required field 'user_name'",
  "llm_output": {
    "status": "success",
    "result": {
      "user_id": "u123",
      "birth_date": "1990-01-01"
      // 缺少 "user_name" 字段
    }
  },
  "schema_expected": {
    "user_id": "string",
    "user_name": "string",  // 必需字段
    "birth_date": "string"
  },
  "schema_actual": {
    "user_id": "string",
    "birth_date": "string"
    // 缺少 "user_name" 字段
  },
  "prompt_hash": "abc123...",
  "model_version": "qwen2.5-vl-32b-v1.0"
}

结论：LLM 输出缺少 "user_name" 字段
```

---

**步骤 4：定位根因并解决问题**

```
根因分析：
1. LLM 推理成功（status: "success"）
2. 但输出格式不符合 Schema 要求（缺少 "user_name" 字段）
3. Schema 验证失败（因为 user_name 是必需字段）
4. 请求失败（error_code: SCHEMA_VALIDATION_FAILED）

可能的解决方案：
方案 1：修改 LLM Prompt
- 在 Prompt 中明确要求输出 "user_name" 字段
- 示例：在 Prompt 中添加 "必须包含 user_name 字段"

方案 2：调整 Schema
- 将 "user_name" 改为可选字段（如果业务允许）
- 但可能影响业务逻辑，需要业务团队确认

方案 3：触发 Fallback
- 在 Schema 验证失败时，触发 Fallback 或人工审核
- 保证系统可用性，但会增加成本

推荐方案：方案 1（修改 LLM Prompt）
- 最简单、最直接
- 不影响业务逻辑
- 提高自动化率
```

---

## 📊 关键标识符的作用

### request_id（请求的唯一标识符）

**作用**：
- ✅ **追踪单个请求**：可以追踪单个请求的所有信息（Logs、Metrics、Traces）
- ✅ **关联相关数据**：可以将同一个请求的所有相关数据关联起来
- ✅ **问题排查**：可以快速找到某个失败请求的所有相关信息

**使用场景**：
```
场景：你想查看某个失败的请求的所有相关信息

操作：
1. 从 Dashboard 获取 request_id: req-abc123
2. 在 Logs Dashboard 中搜索 request_id: req-abc123
3. 在 Traces Dashboard 中搜索 request_id: req-abc123（如果有关联）
4. 在 Metrics Dashboard 中查看该请求的 Metrics（如果有）

结果：能看到该请求的所有相关信息
```

---

### trace_id（链路追踪的唯一标识符）

**作用**：
- ✅ **追踪完整调用链**：可以追踪请求在分布式系统中的完整调用链
- ✅ **关联多个服务**：可以将同一个请求在多个服务中的信息关联起来
- ✅ **性能分析**：可以分析请求在每个服务中的延迟和状态

**使用场景**：
```
场景：你想查看某个请求在分布式系统中的完整调用链

操作：
1. 从 Dashboard 获取 trace_id: trace-abc123
2. 在 Traces Dashboard（Jaeger/Zipkin）中输入 trace_id: trace-abc123
3. 查看完整的调用链（所有 Span）

结果：能看到该请求在系统中的完整调用链，包括：
- 每个服务的延迟（latency）
- 每个服务的状态（status）
- 哪个服务失败了（error）
```

---

### file_id（文件的唯一标识符）

**作用**：
- ✅ **追踪单个文件**：可以追踪单个文件的所有信息（处理结果、错误信息等）
- ✅ **问题排查**：可以快速找到某个文件的处理结果
- ✅ **数据溯源**：可以追溯某个文件的完整处理流程

**使用场景**：
```
场景：你想查看某个文件的处理结果

操作：
1. 从 Dashboard 获取 file_id: doc_001.jpg
2. 在 Logs Dashboard 中搜索 file_id: doc_001.jpg
3. 在 Metrics Dashboard 中查看该文件的 Metrics（如果有）

结果：能看到该文件的所有相关信息
```

---

### error_code（错误代码）

**作用**：
- ✅ **快速识别错误类型**：可以快速识别错误的类型（SCHEMA_VALIDATION_FAILED、API_TIMEOUT 等）
- ✅ **错误分类统计**：可以统计不同类型的错误数量和分布
- ✅ **问题定位**：可以快速定位到特定的错误类型

**使用场景**：
```
场景：你想查看某种类型的所有错误

操作：
1. 从 Dashboard 看到 error_code: SCHEMA_VALIDATION_FAILED
2. 在 Logs Dashboard 中搜索 error_code: SCHEMA_VALIDATION_FAILED
3. 查看所有 SCHEMA_VALIDATION_FAILED 错误的详细信息

结果：能看到所有 SCHEMA_VALIDATION_FAILED 错误的详细信息
```

---

## 💡 最佳实践

### 1. 关键标识符的生成和传递

**request_id**：
- ✅ **在 API Gateway 生成**：每个请求在进入系统时生成唯一的 `request_id`
- ✅ **传递到所有服务**：将 `request_id` 传递到所有服务（通过 HTTP Header、Context 等）
- ✅ **记录在所有 Logs 中**：所有 Logs 都应包含 `request_id`

**trace_id**：
- ✅ **在 API Gateway 生成**：每个请求在进入系统时生成唯一的 `trace_id`
- ✅ **传递到所有服务**：将 `trace_id` 传递到所有服务（通过 HTTP Header、Context 等）
- ✅ **用于 Traces**：`trace_id` 用于关联 Traces 中的所有 Span

**file_id**：
- ✅ **在文件上传时生成**：每个文件在上传时生成唯一的 `file_id`
- ✅ **记录在所有 Logs 中**：所有与文件相关的 Logs 都应包含 `file_id`

**error_code**：
- ✅ **标准化错误代码**：使用标准化的错误代码（如 `SCHEMA_VALIDATION_FAILED`、`API_TIMEOUT` 等）
- ✅ **记录在所有 Logs 中**：所有错误 Logs 都应包含 `error_code`

---

### 2. 结构化日志（JSON）

**结构化日志的好处**：
- ✅ **方便查询**：可以使用 `request_id`、`trace_id`、`error_code` 等字段快速查询
- ✅ **自动化分析**：可以自动化分析日志（统计错误数量、分析错误类型等）
- ✅ **关联数据**：可以将不同系统的日志关联起来

**结构化日志的格式**：
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "ERROR",
  "request_id": "req-abc123",
  "trace_id": "trace-abc123",
  "file_id": "doc_001.jpg",
  "span_name": "Schema Validation",
  "error_code": "SCHEMA_VALIDATION_FAILED",
  "error_message": "Schema validation failed: missing required field 'user_name'",
  "context": {
    // 额外的上下文信息
  }
}
```

---

### 3. Dashboard → Traces → Logs 的导航流程

**推荐流程**：
```
1. Dashboard（发现问题）
   - 看到 Error Rate 升高
   - 点击失败的请求，获取关键标识符（request_id、trace_id、error_code）

2. Traces（了解调用链）
   - 使用 trace_id 查看完整的调用链
   - 找出哪个环节失败了

3. Logs（查看详细错误信息）
   - 使用 request_id/trace_id 查看详细的错误信息和上下文
   - 了解为什么会失败

4. 根因分析（定位问题）
   - 根据 Traces 和 Logs 的信息，定位根因
   - 制定解决方案
```

---

## 📊 总结

### 核心答案

**如何从 Dashboard 选择一个失败的请求，然后定位根因？**

**完整流程**：
1. **Dashboard 发现问题**：看到 Error Rate 升高，点击失败的请求
2. **获取关键标识符**：`request_id`、`trace_id`、`file_id`、`error_code`
3. **查看 Traces**：使用 `trace_id` 查看完整的调用链，找出哪个环节失败
4. **查看 Logs**：使用 `request_id`/`trace_id` 查看详细的错误信息和上下文
5. **定位根因**：根据 Traces 和 Logs 的信息，定位根因并解决问题

### 关键标识符

| 标识符 | 作用 | 使用场景 |
|--------|------|---------|
| **request_id** | 追踪单个请求 | 查看某个请求的所有相关信息 |
| **trace_id** | 追踪完整调用链 | 查看请求在分布式系统中的完整调用链 |
| **file_id** | 追踪单个文件 | 查看某个文件的所有相关信息 |
| **error_code** | 快速识别错误类型 | 查看某种类型的所有错误 |

### 最佳实践

1. **关键标识符的生成和传递**：在 API Gateway 生成，传递到所有服务
2. **结构化日志（JSON）**：方便查询和自动化分析
3. **Dashboard → Traces → Logs 的导航流程**：从发现问题到定位根因的完整流程

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1 可观测性详解（[KYC_Day02_A1_可观测性详解.md](./KYC_Day02_A1_可观测性详解.md)） |
| **Related** | Dashboard、Metrics、Logs、Traces、request_id、trace_id、根因定位 |
