# Day 2_A1_B2_C1：Traces 和 Span 详解（链路追踪是什么）

---
doc_type: glossary
layer: L3
scope_in:  Traces（链路追踪）是什么、Span 是什么、如何理解 Trace 的调用链、latency 和 status 的含义
scope_out: 具体 Traces 实现（见 howto）；深入的分布式追踪架构（见 L4）
inputs:   (读者) 疑问：Trace 和 Span 是什么？如何理解 Trace 的调用链？
outputs:  Traces 和 Span 的解释 + Trace 调用链的理解 + 实际例子 + 如何阅读 Trace 结果
entrypoints: [ 核心问题 ]
children: [ KYC_Day02_A1_B2_C1_D1_如何查询Trace和返回格式详解.md（如何查询 Trace 和返回格式详解） ]
related: [ Traces, Span, 链路追踪, 分布式追踪, trace_id, 调用链, KYC_Day02_A1_B2_从Dashboard到根因定位的完整流程详解.md ]
---

## Definition（定义）

**核心问题**：**Trace 和 Span 是什么？如何理解 Trace 的调用链？**

**核心答案**：
- ✅ **Trace（链路追踪）**：**一个请求在分布式系统中的完整调用链**，记录从入口到出口的所有步骤
- ✅ **Span（跨度）**：**Trace 中的一个步骤**，记录每个服务的调用情况（延迟、状态等）
- ✅ **trace_id**：**Trace 的唯一标识符**，用于关联同一个请求在多个服务中的所有信息

**类比**：
- **Trace** = **一次旅行的完整路线**（从起点到终点的所有路径）
- **Span** = **旅途中经过的每个城市**（每个城市的路程、时间、是否顺利）
- **trace_id** = **旅行编号**（通过这个编号，可以查看整个旅行的所有记录）

---

## 🎯 核心问题

### Trace（链路追踪）是什么？

**简单理解**：**Trace 记录了一个请求从进入系统到离开系统的完整过程**。

**类比**：
```
场景：你在网上购物，从下单到收货的完整过程

Trace（链路追踪）：
1. 你在网站下单（入口）
2. 订单系统处理订单
3. 支付系统处理支付
4. 库存系统检查库存
5. 仓库系统准备发货
6. 物流系统配送
7. 你收到货物（出口）

这就是一个 Trace（完整的调用链）
```

**KYC 项目的例子**：
```
场景：一个 KYC 请求，从 API 调用到返回结果的完整过程

Trace（链路追踪）：
1. API Gateway（入口：接收请求）
2. KYC Service（主服务：处理请求）
3. Preprocessing（预处理：处理图片）
4. LLM Inference（LLM 推理：提取信息）
5. Schema Validation（Schema 验证：验证输出格式）
6. Post-processing（后处理：保存结果）
7. 返回结果（出口：返回给用户）

这就是一个 Trace（完整的调用链）
```

---

### Span（跨度）是什么？

**简单理解**：**Span 是 Trace 中的一个步骤**，记录每个服务的调用情况。

**类比**：
```
场景：你在旅途中经过的每个城市

Span（跨度）：
1. 城市 A（起点）
   - 路程：100 公里
   - 时间：1 小时
   - 状态：顺利到达

2. 城市 B（中转站）
   - 路程：200 公里
   - 时间：2 小时
   - 状态：顺利到达

3. 城市 C（终点）
   - 路程：150 公里
   - 时间：1.5 小时
   - 状态：顺利到达

每个城市就是一个 Span（一个步骤）
```

**KYC 项目的例子**：
```
Trace 中的每个 Span（步骤）：

Span 1: API Gateway（入口）
- latency: 5ms（延迟：5 毫秒）
- status: OK（状态：成功）

Span 2: KYC Service（主服务）
- latency: 100ms（延迟：100 毫秒）
- status: OK（状态：成功）

Span 3: Preprocessing（预处理）
- latency: 50ms（延迟：50 毫秒）
- status: OK（状态：成功）

Span 4: LLM Inference（LLM 推理）
- latency: 5000ms（延迟：5000 毫秒 = 5 秒）
- status: OK（状态：成功）

Span 5: Schema Validation（Schema 验证）← 问题在这里！
- latency: 10ms（延迟：10 毫秒）
- status: ERROR（状态：失败）
- error_code: SCHEMA_VALIDATION_FAILED（错误代码）

每个 Span 记录了一个服务的调用情况
```

---

## 📊 如何理解 Trace 的调用链

### Trace 的结构（树状结构）

**Trace 的格式**：
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

---

### 如何阅读 Trace 结果

**步骤 1：理解树状结构**

```
Trace: trace-abc123
├─ Span 1: ... （第一层：第一个步骤）
├─ Span 2: ... （第一层：第二个步骤）
│  ├─ latency: ... （第二层：Span 2 的属性）
│  └─ status: ... （第二层：Span 2 的属性）
```

**树状结构的含义**：
- ✅ **├─**：表示一个分支（一个 Span）
- ✅ **│**：表示继续向下（下一层的属性）
- ✅ **└─**：表示最后一个分支（最后一个 Span）

**例子**：
```
Trace: trace-abc123
├─ Span 1: API Gateway
│  ├─ latency: 5ms      ← Span 1 的属性
│  └─ status: OK        ← Span 1 的属性
│
└─ Span 2: KYC Service
   ├─ latency: 100ms    ← Span 2 的属性
   └─ status: OK        ← Span 2 的属性
```

---

**步骤 2：理解每个 Span 的信息**

**每个 Span 包含的信息**：
- ✅ **Span 名称**：这个 Span 是什么（如 "API Gateway"、"LLM Inference"）
- ✅ **latency（延迟）**：这个 Span 花了多少时间（单位：毫秒 ms）
- ✅ **status（状态）**：这个 Span 是否成功（OK = 成功，ERROR = 失败）
- ✅ **error_code（错误代码）**：如果失败，错误代码是什么

**例子**：
```
Span 5: Schema Validation（Schema 验证）
├─ latency: 10ms           ← 这个 Span 花了 10 毫秒
├─ status: ERROR           ← 这个 Span 失败了
└─ error_code: SCHEMA_VALIDATION_FAILED  ← 错误代码：Schema 验证失败
```

**关键点**：
- ✅ **latency**：可以看出哪个 Span 慢（延迟高）
- ✅ **status**：可以看出哪个 Span 失败（ERROR）
- ✅ **error_code**：可以快速识别错误类型

---

**步骤 3：找出问题**

**如何找出问题**：

1. **找出失败的 Span**：
   ```
   Span 5: Schema Validation
   ├─ status: ERROR  ← 状态是 ERROR，说明这个 Span 失败了
   ```

2. **找出慢的 Span**：
   ```
   Span 4: LLM Inference
   ├─ latency: 5000ms  ← 延迟是 5000 毫秒（5 秒），说明这个 Span 很慢
   ```

3. **找出错误的 Span**：
   ```
   Span 5: Schema Validation
   ├─ status: ERROR
   └─ error_code: SCHEMA_VALIDATION_FAILED  ← 错误代码，可以快速识别错误类型
   ```

**例子（完整的 Trace）**：
```
Trace: trace-abc123
├─ Span 1: API Gateway
│  ├─ latency: 5ms      ← 快（5 毫秒）
│  └─ status: OK        ← 成功
│
├─ Span 2: KYC Service
│  ├─ latency: 100ms    ← 快（100 毫秒）
│  └─ status: OK        ← 成功
│
├─ Span 3: Preprocessing
│  ├─ latency: 50ms     ← 快（50 毫秒）
│  └─ status: OK        ← 成功
│
├─ Span 4: LLM Inference
│  ├─ latency: 5000ms   ← ⚠️ 慢（5000 毫秒 = 5 秒）
│  └─ status: OK        ← 成功
│
├─ Span 5: Schema Validation
│  ├─ latency: 10ms     ← 快（10 毫秒）
│  ├─ status: ERROR     ← ❌ 失败（这是问题）
│  └─ error_code: SCHEMA_VALIDATION_FAILED  ← 错误代码
│
└─ Span 6: Post-processing
   └─ status: SKIPPED   ← 跳过（因为验证失败）

结论：
1. Span 4 很慢（5 秒），可能需要优化
2. Span 5 失败了（ERROR），这是主要问题
3. Span 6 被跳过了（因为 Span 5 失败）
```

---

## 💡 实际例子（KYC 项目）

### 例子 1：成功的 Trace

**场景**：一个 KYC 请求成功处理

**Trace 结果**：
```
Trace: trace-success-123
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
│  ├─ latency: 2000ms（2 秒）
│  └─ status: OK
│
├─ Span 5: Schema Validation
│  ├─ latency: 10ms
│  └─ status: OK
│
└─ Span 6: Post-processing
   ├─ latency: 20ms
   └─ status: OK

总耗时：5 + 100 + 50 + 2000 + 10 + 20 = 2185ms（约 2.2 秒）
所有 Span 都成功（status: OK）
```

---

### 例子 2：失败的 Trace

**场景**：一个 KYC 请求失败（Schema 验证失败）

**Trace 结果**：
```
Trace: trace-fail-456
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
│  ├─ latency: 5000ms（5 秒）
│  └─ status: OK  ← LLM 推理成功
│
├─ Span 5: Schema Validation  ← 问题在这里！
│  ├─ latency: 10ms
│  ├─ status: ERROR  ← 失败
│  └─ error_code: SCHEMA_VALIDATION_FAILED
│
└─ Span 6: Post-processing
   └─ status: SKIPPED  ← 跳过（因为 Span 5 失败）

结论：
1. Span 1-4 都成功（API Gateway、KYC Service、Preprocessing、LLM Inference）
2. Span 5 失败（Schema Validation），错误代码：SCHEMA_VALIDATION_FAILED
3. Span 6 被跳过（因为 Span 5 失败，不需要后处理）
4. 根因：LLM 输出格式不符合 Schema 要求
```

**根因分析**：
- ✅ **Span 4（LLM Inference）成功**：LLM 推理成功，输出了结果
- ✅ **Span 5（Schema Validation）失败**：Schema 验证失败，说明 LLM 输出的格式不符合要求
- ✅ **根因**：LLM 输出缺少必需字段（如 `user_name`），导致 Schema 验证失败

---

## 📊 Trace 的时序图理解

### 时序图（Timeline View）

**Trace 也可以用时序图表示**：

```
时间轴 →

[0ms]  [5ms]  [105ms]  [155ms]      [5155ms]  [5165ms]
 │       │       │        │             │         │
 │       │       │        │             │         │
 ├─ Span 1 (5ms)           │             │         │
 └─ API Gateway            │             │         │
                           │             │         │
       ├─ Span 2 (100ms)   │             │         │
       └─ KYC Service      │             │         │
                           │             │         │
              ├─ Span 3 (50ms)           │         │
              └─ Preprocessing           │         │
                                         │         │
                    ├─ Span 4 (5000ms)   │         │
                    └─ LLM Inference     │         │
                                         │         │
                              ├─ Span 5 (10ms) ❌  │
                              └─ Schema Validation │
                                                   │
                                        └─ Span 6 (SKIPPED)
                                           Post-processing

总耗时：5165ms（约 5.2 秒）
```

**关键点**：
- ✅ **Span 1-4 顺序执行**：每个 Span 等待前一个 Span 完成后才开始
- ✅ **Span 5 失败**：Schema Validation 失败，导致 Span 6 被跳过
- ✅ **Span 4 最慢**：LLM Inference 花了 5000ms（5 秒），是最大的瓶颈

---

## 💡 总结

### 核心答案

**Trace 和 Span 是什么？**

- ✅ **Trace（链路追踪）**：一个请求在分布式系统中的完整调用链
- ✅ **Span（跨度）**：Trace 中的一个步骤，记录每个服务的调用情况

**如何理解 Trace 的调用链？**

- ✅ **树状结构**：Trace 是树状的，每个 Span 是一个分支
- ✅ **每个 Span 包含**：名称、延迟（latency）、状态（status）、错误代码（error_code）
- ✅ **找出问题**：找出失败的 Span（status: ERROR）或慢的 Span（latency 高）

### 关键要点

1. **Trace = 完整调用链**：记录从入口到出口的所有步骤
2. **Span = 一个步骤**：记录每个服务的调用情况
3. **trace_id = 唯一标识符**：用于关联同一个请求在多个服务中的所有信息

### 如何阅读 Trace 结果

1. **看结构**：理解树状结构（├─、│、└─）
2. **看每个 Span**：查看 latency、status、error_code
3. **找出问题**：找出失败的 Span 或慢的 Span

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1_B2 从 Dashboard 到根因定位的完整流程详解（[KYC_Day02_A1_B2_从Dashboard到根因定位的完整流程详解.md](./KYC_Day02_A1_B2_从Dashboard到根因定位的完整流程详解.md)） |
| **Related** | Traces、Span、链路追踪、分布式追踪、trace_id、调用链 |
