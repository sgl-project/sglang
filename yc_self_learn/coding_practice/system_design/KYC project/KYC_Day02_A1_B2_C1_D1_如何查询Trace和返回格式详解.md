# Day 2_A1_B2_C1_D1：如何查询 Trace 和返回格式详解

---
doc_type: glossary
layer: L3
scope_in:  如何查询 Trace、使用 trace_id 查询、返回的数据格式、不同平台的查询方式
scope_out: 具体 Traces 实现（见 howto）；深入的分布式追踪架构（见 L4）
inputs:   (读者) 疑问：如何查询 Trace？调用 trace 会返回什么？
outputs:  如何查询 Trace + 返回的数据格式 + 不同平台的查询方式 + 实际例子
entrypoints: [ 核心问题 ]
children: [ KYC_Day02_A1_B2_C1_D1_E1_JSON结构和树状结构的关系详解.md（JSON 结构和树状结构的关系详解） ]
related: [ Traces, Span, trace_id, 查询 Trace, Jaeger, Zipkin, Datadog APM, KYC_Day02_A1_B2_C1_Traces和Span详解_链路追踪是什么.md ]
---

## Definition（定义）

**核心问题**：**如何查询 Trace？调用 trace 会返回什么？**

**核心答案**：
- ✅ **不是"call trace"**：而是**使用 trace_id 查询 Trace**
- ✅ **返回的结构**：**是的，会返回类似的结构**（Trace 包含多个 Span，每个 Span 有 latency、status 等信息）
- ✅ **不同平台**：不同平台（Jaeger、Zipkin、Datadog）的查询方式和返回格式略有不同

**关键理解**：
- ✅ **查询方式**：使用 trace_id 在 Tracing Dashboard 中查询
- ✅ **返回格式**：返回 Trace 的结构化数据（包含多个 Span）
- ✅ **可视化**：平台会将这些数据显示为树状结构或时序图

---

## 🎯 核心问题

### 如何查询 Trace？

**不是"call trace"**，而是**使用 trace_id 查询 Trace**。

**流程**：
```
1. 获取 trace_id（从 Dashboard 或 Logs）
   ↓
2. 打开 Tracing Dashboard（Jaeger/Zipkin/Datadog APM）
   ↓
3. 输入 trace_id 查询
   ↓
4. 返回 Trace 的结构化数据
   ↓
5. 平台显示为树状结构或时序图
```

---

## 📊 如何查询 Trace（不同平台）

### 平台 1：Jaeger

**查询方式**：

1. **打开 Jaeger UI**：
   ```
   访问：http://localhost:16686
   ```

2. **输入 trace_id**：
   ```
   在搜索框中输入：trace-abc123
   或
   选择 Service 和时间范围，然后输入 trace_id
   ```

3. **点击"Search"**：

4. **查看 Trace 详情**：
   ```
   平台会显示 Trace 的树状结构或时序图
   ```

**返回的数据格式（JSON）**：
```json
{
  "traceID": "trace-abc123",
  "spans": [
    {
      "spanID": "span-1",
      "operationName": "API Gateway",
      "startTime": 1234567890000000,
      "duration": 5000000,
      "tags": {
        "http.status_code": "200",
        "span.kind": "server"
      },
      "logs": []
    },
    {
      "spanID": "span-2",
      "operationName": "KYC Service",
      "startTime": 1234567895000000,
      "duration": 100000000,
      "tags": {
        "http.status_code": "200",
        "span.kind": "server"
      },
      "logs": []
    },
    {
      "spanID": "span-5",
      "operationName": "Schema Validation",
      "startTime": 1234572890000000,
      "duration": 10000000,
      "tags": {
        "error": "true",
        "error.code": "SCHEMA_VALIDATION_FAILED"
      },
      "logs": [
        {
          "timestamp": 1234572900000000,
          "fields": [
            {
              "key": "error",
              "value": "Schema validation failed: missing required field 'user_name'"
            }
          ]
        }
      ]
    }
  ]
}
```

**可视化显示（树状结构）**：
```
Trace: trace-abc123
├─ Span 1: API Gateway
│  ├─ latency: 5ms
│  └─ status: OK
│
├─ Span 2: KYC Service
│  ├─ latency: 100ms
│  └─ status: OK
│
...
```

---

### 平台 2：Zipkin

**查询方式**：

1. **打开 Zipkin UI**：
   ```
   访问：http://localhost:9411
   ```

2. **输入 trace_id**：
   ```
   在搜索框中输入：trace-abc123
   ```

3. **点击"Run Query"**：

4. **查看 Trace 详情**：
   ```
   平台会显示 Trace 的时序图
   ```

**返回的数据格式（JSON）**：
```json
{
  "traceId": "trace-abc123",
  "id": "span-1",
  "name": "API Gateway",
  "timestamp": 1234567890000000,
  "duration": 5000000,
  "tags": {
    "http.status_code": "200"
  },
  "annotations": []
}
```

**可视化显示（时序图）**：
```
时间轴 →

[0ms]  [5ms]  [105ms]  [155ms]      [5155ms]
 │       │       │        │             │
 ├─ Span 1 (5ms)           │             │
 └─ API Gateway            │             │
                           │             │
       ├─ Span 2 (100ms)   │             │
       └─ KYC Service      │             │
                           │             │
              ├─ Span 3 (50ms)           │
              └─ Preprocessing           │
                                         │
                    ├─ Span 4 (5000ms)   │
                    └─ LLM Inference     │
                                         │
                              ├─ Span 5 (10ms) ❌
                              └─ Schema Validation
```

---

### 平台 3：Datadog APM

**查询方式**：

1. **打开 Datadog APM**：
   ```
   访问：https://app.datadoghq.com/apm
   ```

2. **输入 trace_id**：
   ```
   在搜索框中输入：trace-abc123
   或
   从 Traces 列表中选择
   ```

3. **点击"Search"**：

4. **查看 Trace 详情**：
   ```
   平台会显示 Trace 的树状结构和时序图
   ```

**返回的数据格式（JSON）**：
```json
{
  "trace_id": "trace-abc123",
  "spans": [
    {
      "span_id": "span-1",
      "name": "API Gateway",
      "start": 1234567890000000,
      "duration": 5000000,
      "meta": {
        "http.status_code": "200",
        "span.kind": "server"
      }
    },
    {
      "span_id": "span-5",
      "name": "Schema Validation",
      "start": 1234572890000000,
      "duration": 10000000,
      "error": 1,
      "meta": {
        "error.type": "SCHEMA_VALIDATION_FAILED",
        "error.message": "Schema validation failed: missing required field 'user_name'"
      }
    }
  ]
}
```

**可视化显示（树状结构 + 时序图）**：
```
Trace: trace-abc123
├─ Span 1: API Gateway
│  ├─ latency: 5ms
│  └─ status: OK
│
...
```

---

## 💡 实际例子

### 例子 1：使用 Jaeger 查询 Trace

**场景**：你在 Dashboard 上看到一个失败的请求，trace_id 是 `trace-abc123`。

**操作步骤**：

1. **打开 Jaeger UI**：
   ```
   访问：http://localhost:16686
   ```

2. **输入 trace_id**：
   ```
   在搜索框中输入：trace-abc123
   ```

3. **点击"Search"**：

4. **查看 Trace 详情**：
   ```
   Jaeger 会返回 Trace 的数据，并显示为树状结构：
   
   Trace: trace-abc123
   ├─ Span 1: API Gateway
   │  ├─ latency: 5ms
   │  └─ status: OK
   │
   ├─ Span 2: KYC Service
   │  ├─ latency: 100ms
   │  └─ status: OK
   │
   ├─ Span 3: Preprocessing
   │  ├─ latency: 50ms
   │  └─ status: OK
   │
   ├─ Span 4: LLM Inference
   │  ├─ latency: 5000ms
   │  └─ status: OK
   │
   ├─ Span 5: Schema Validation
   │  ├─ latency: 10ms
   │  ├─ status: ERROR
   │  └─ error_code: SCHEMA_VALIDATION_FAILED
   │
   └─ Span 6: Post-processing
      └─ status: SKIPPED
   ```

**关键点**：
- ✅ **输入 trace_id**：在 Jaeger UI 中输入 trace_id
- ✅ **返回结构化数据**：Jaeger 返回 Trace 的 JSON 数据
- ✅ **可视化显示**：Jaeger 将数据显示为树状结构

---

### 例子 2：使用 API 查询 Trace

**场景**：你想通过 API 查询 Trace（而不是通过 UI）。

**使用 Jaeger API**：

```python
import requests

# Jaeger API 查询 Trace
trace_id = "trace-abc123"
jaeger_url = "http://localhost:16686/api/traces/{trace_id}".format(trace_id=trace_id)

response = requests.get(jaeger_url)
trace_data = response.json()

# 返回的数据格式
print(trace_data)
# {
#   "data": [
#     {
#       "traceID": "trace-abc123",
#       "spans": [
#         {
#           "spanID": "span-1",
#           "operationName": "API Gateway",
#           "duration": 5000000,
#           ...
#         },
#         ...
#       ]
#     }
#   ]
# }
```

**使用 Zipkin API**：

```python
import requests

# Zipkin API 查询 Trace
trace_id = "trace-abc123"
zipkin_url = "http://localhost:9411/api/v2/trace/{trace_id}".format(trace_id=trace_id)

response = requests.get(zipkin_url)
trace_data = response.json()

# 返回的数据格式
print(trace_data)
# [
#   {
#     "traceId": "trace-abc123",
#     "id": "span-1",
#     "name": "API Gateway",
#     "duration": 5000000,
#     ...
#   },
#   ...
# ]
```

**关键点**：
- ✅ **API 查询**：可以通过 API 查询 Trace（不需要 UI）
- ✅ **返回 JSON**：API 返回的是 JSON 格式的数据
- ✅ **需要解析**：需要解析 JSON 数据，提取 Span 信息

---

## 📊 返回的数据结构

### Trace 的数据结构

**Trace 包含**：
- ✅ **trace_id**：Trace 的唯一标识符
- ✅ **spans**：Span 列表（包含所有步骤）
- ✅ **duration**：总耗时（所有 Span 的总和）

**每个 Span 包含**：
- ✅ **span_id**：Span 的唯一标识符
- ✅ **operation_name**：Span 的名称（如 "API Gateway"、"LLM Inference"）
- ✅ **start_time**：开始时间
- ✅ **duration/latency**：延迟（单位：微秒或毫秒）
- ✅ **status**：状态（OK、ERROR）
- ✅ **tags/meta**：标签/元数据（包含 error_code 等）
- ✅ **logs**：日志（包含错误信息等）

---

### 返回格式的对比

| 平台 | 返回格式 | 数据结构 |
|------|---------|---------|
| **Jaeger** | JSON | `{ "traceID": "...", "spans": [...] }` |
| **Zipkin** | JSON | `[ { "traceId": "...", ... }, ... ]` |
| **Datadog** | JSON | `{ "trace_id": "...", "spans": [...] }` |

**关键点**：
- ✅ **都是 JSON**：所有平台都返回 JSON 格式的数据
- ✅ **结构相似**：都包含 trace_id 和 spans
- ✅ **字段名略有不同**：不同平台的字段名略有不同（如 `traceID` vs `trace_id`）

---

## 💡 总结

### 核心答案

**如何查询 Trace？**

- ✅ **不是"call trace"**：而是**使用 trace_id 查询 Trace**
- ✅ **查询方式**：在 Tracing Dashboard（Jaeger/Zipkin/Datadog）中输入 trace_id
- ✅ **返回格式**：返回 Trace 的结构化数据（JSON 格式）

**调用 trace 会返回什么？**

- ✅ **返回结构化数据**：Trace 包含多个 Span，每个 Span 有 latency、status 等信息
- ✅ **可视化显示**：平台会将这些数据显示为树状结构或时序图
- ✅ **格式略有不同**：不同平台的返回格式略有不同，但结构相似

### 关键要点

1. **查询方式**：使用 trace_id 在 Tracing Dashboard 中查询
2. **返回格式**：返回 JSON 格式的 Trace 数据（包含多个 Span）
3. **可视化**：平台会自动将数据显示为树状结构或时序图

### 实际流程

```
1. 获取 trace_id（从 Dashboard 或 Logs）
   ↓
2. 打开 Tracing Dashboard（Jaeger/Zipkin/Datadog）
   ↓
3. 输入 trace_id 查询
   ↓
4. 平台返回 Trace 的 JSON 数据
   ↓
5. 平台自动显示为树状结构或时序图
   ↓
6. 你可以看到完整的调用链（所有 Span）
```

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1_B2_C1 Traces 和 Span 详解（[KYC_Day02_A1_B2_C1_Traces和Span详解_链路追踪是什么.md](./KYC_Day02_A1_B2_C1_Traces和Span详解_链路追踪是什么.md)） |
| **Related** | Traces、Span、trace_id、查询 Trace、Jaeger、Zipkin、Datadog APM |
