# SGLang 完整请求流程详解（纠正版）

## 📋 目录

1. [核心纠正](#1-核心纠正)
2. [完整流程概览](#2-完整流程概览)
3. [场景 1: 使用 Router（多 Worker）](#3-场景-1-使用-router多-worker)
4. [场景 2: 不使用 Router（单 Worker）](#4-场景-2-不使用-router单-worker)
5. [关键步骤详解](#5-关键步骤详解)
6. [常见误解纠正](#6-常见误解纠正)

---

## 1. 核心纠正

### 1.1 用户理解中的错误

**错误理解**:
```
1. 发送 request
2. 转成 token
3. router 先做 prefix match
4. 然后决定如何分配 GPU
5. 最后在 API call
6. 然后返回
```

**问题**:
- ❌ **Tokenize 时机错误**: Tokenize 不是在 Router 之前，而是在 Worker 内部
- ❌ **Router 的 Prefix Match**: Router 做的是**文本级别的近似匹配**，不是 token 级别的
- ❌ **GPU 分配位置错误**: GPU 分配不在 Router，而是在 Worker 内部的 Scheduler
- ❌ **API Call 理解错误**: 客户端只有一次 API 调用，Router 只是转发

---

### 1.2 正确理解

**正确流程**:
```
1. 客户端发送 HTTP 请求到 Router（或直接到 Worker）
2. Router 提取文本，做近似前缀匹配（文本级别），选择 Worker
3. Router 转发 HTTP 请求到选中的 Worker
4. Worker 接收请求 → TokenizerManager 开始 Tokenize
5. Worker 内部的 Scheduler 做真实的前缀缓存匹配（token 级别）
6. Scheduler 构建 Batch，分配 GPU 资源
7. GPU 执行推理，生成 token IDs
8. Detokenizer 将 token IDs 转回文本
9. Worker 返回 HTTP 响应
10. Router 返回响应给客户端
```

---

## 2. 完整流程概览

### 2.1 两种场景

**场景 1: 使用 Router（多 Worker 部署）**
```
客户端 → Router → Worker → GPU → Worker → Router → 客户端
```

**场景 2: 不使用 Router（单 Worker 部署）**
```
客户端 → Worker → GPU → Worker → 客户端
```

---

### 2.2 关键组件

| 组件 | 位置 | 职责 |
|------|------|------|
| **Router** | 应用层（Layer 7） | 在多个 Worker 之间路由请求 |
| **Worker** | 应用层（Layer 7） | 处理推理请求 |
| **TokenizerManager** | Worker 内部 | Tokenize 文本 |
| **Scheduler** | Worker 内部 | 调度和 GPU 分配 |
| **GPU** | Worker 内部 | 执行模型推理 |

---

## 3. 场景 1: 使用 Router（多 Worker）

### 3.1 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: 客户端发送 HTTP 请求                                 │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP POST /v1/chat/completions
                     │ Body: {"messages": [...], "model": "..."}
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: Router 接收请求                                      │
│  - 解析 HTTP 请求                                            │
│  - 提取请求文本（extract_text_for_routing）                 │
└────────────────────┬────────────────────────────────────────┘
                     │ 文本: "tell me what is sglang"
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: Router 做近似前缀匹配（文本级别）                    │
│  - 查询 Radix Tree（存储的是文本，不是 token）              │
│  - 计算匹配率                                                │
│  - 选择最合适的 Worker                                       │
└────────────────────┬────────────────────────────────────────┘
                     │ 选择 Worker 1（匹配率 80%）
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 4: Router 转发 HTTP 请求到 Worker                      │
│  - HTTP POST Worker 1:8000/v1/chat/completions             │
│  - 转发完整的 HTTP 请求（包括 JSON body）                  │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP 请求
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 5: Worker 接收请求                                      │
│  - FastAPI 接收 HTTP 请求                                   │
│  - 解析 JSON body                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 6: TokenizerManager 开始 Tokenize                      │
│  - 提取文本: "tell me what is sglang"                       │
│  - 调用 tokenizer.encode()                                  │
│  - 生成 token IDs: [101, 202, 303, ...]                    │
└────────────────────┬────────────────────────────────────────┘
                     │ token IDs
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 7: Scheduler 做真实前缀缓存匹配（token 级别）          │
│  - 调用 RadixCache.match_prefix()                          │
│  - 查找最长的匹配前缀（基于 token IDs）                     │
│  - 返回 prefix_indices: [0, 1, 2, 3, 4, 5]                │
└────────────────────┬────────────────────────────────────────┘
                     │ prefix_indices
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 8: Scheduler 构建 Batch，分配 GPU                      │
│  - 构建 ScheduleBatch                                        │
│  - 分配 GPU 内存（KV Cache）                                │
│  - 准备推理数据                                              │
└────────────────────┬────────────────────────────────────────┘
                     │ Batch 数据
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 9: GPU 执行推理                                         │
│  - 使用缓存的 KV Cache（prefix_indices）                    │
│  - 只计算新部分的注意力                                      │
│  - 生成新的 token IDs                                        │
└────────────────────┬────────────────────────────────────────┘
                     │ 新 token IDs
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 10: Detokenizer 将 token IDs 转回文本                 │
│  - 调用 tokenizer.decode()                                  │
│  - 生成文本: "SGLang 是一个高性能的..."                     │
└────────────────────┬────────────────────────────────────────┘
                     │ 文本
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 11: Worker 返回 HTTP 响应                              │
│  - 构建 JSON 响应                                           │
│  - 返回给 Router                                             │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP Response
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 12: Router 返回响应给客户端                            │
│  - 转发 HTTP 响应                                           │
│  - 客户端收到最终结果                                        │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.2 详细步骤说明

#### 步骤 1: 客户端发送 HTTP 请求

**代码示例**:
```python
import requests

response = requests.post(
    url="http://router:30000/v1/chat/completions",
    json={
        "model": "qwen/qwen2.5-0.5b-instruct",
        "messages": [
            {"role": "user", "content": "tell me what is sglang"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)
```

**请求内容**:
- **URL**: `http://router:30000/v1/chat/completions`
- **Method**: POST
- **Body**: JSON 格式，包含 messages、model 等

---

#### 步骤 2: Router 接收请求

**代码位置**: `sgl-router/src/routers/http/router.rs:156`

```rust
pub async fn route_typed_request<T>(&self, ...) -> Response {
    // 1. 提取请求文本（用于路由决策）
    let text = typed_req.extract_text_for_routing();
    // text = "tell me what is sglang" (原始文本，未 tokenize)
    
    // 2. 选择 Worker
    let worker = match self.select_worker_for_model(model_id, Some(&text)) {
        Some(w) => w,
        None => return error_response(),
    };
    
    // 3. 转发 HTTP 请求到 Worker
    let response = self.client
        .post(worker.url())  // http://worker1:8000/v1/chat/completions
        .json(&typed_req)    // 转发完整的 JSON body
        .send()
        .await?;
    
    return response;
}
```

**关键点**:
- ✅ **提取文本**: Router 提取原始文本（未 tokenize）
- ✅ **文本级别匹配**: Router 使用文本做前缀匹配
- ✅ **转发请求**: Router 转发完整的 HTTP 请求到 Worker

---

#### 步骤 3: Router 做近似前缀匹配（文本级别）

**代码位置**: `sgl-router/src/policies/cache_aware.rs:302`

```rust
// Router 的近似前缀匹配（文本级别）
let text = request_text.unwrap_or("");  // "tell me what is sglang"

// 查询 Radix Tree（存储的是文本，不是 token）
let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

if let Some(tree) = tree {
    // 文本级别的前缀匹配
    let (matched_text, matched_worker) = tree.prefix_match(text);
    // matched_text = "tell me what is" (文本匹配)
    
    // 计算匹配率（基于字符数）
    let match_rate = matched_text.chars().count() as f32 / text.chars().count() as f32;
    // match_rate = 0.8 (80%)
    
    // 选择 Worker
    let selected_url = if match_rate > self.config.cache_threshold {
        matched_worker.to_string()  // 选择匹配的 Worker
    } else {
        tree.get_smallest_tenant()  // 选择缓存空间最大的 Worker
    };
}
```

**关键点**:
- ✅ **文本级别**: Router 使用原始文本做匹配，不进行 tokenize
- ✅ **近似树**: Router 维护的是近似 Radix Tree（存储文本，不是 token）
- ✅ **快速决策**: 文本匹配很快（< 1ms），不需要 tokenize

---

#### 步骤 4: Router 转发 HTTP 请求到 Worker

**代码位置**: `sgl-router/src/routers/http/router.rs:199`

```rust
// Router 转发 HTTP 请求到 Worker
let response = self.client
    .post(worker.url())  // http://worker1:8000/v1/chat/completions
    .json(&typed_req)    // 转发完整的 JSON body
    .send()
    .await?;
```

**关键点**:
- ✅ **HTTP 转发**: Router 转发完整的 HTTP 请求
- ✅ **不修改内容**: Router 不修改请求内容，只是转发
- ✅ **透明代理**: Router 作为透明代理，客户端不知道有 Router

---

#### 步骤 5-6: Worker 接收请求并 Tokenize

**代码位置**: `python/sglang/srt/managers/tokenizer_manager.py:146`

```python
# Worker 接收 HTTP 请求
@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(request: ChatCompletionRequest):
    # 调用 TokenizerManager
    return await tokenizer_manager.generate_request(...)

# TokenizerManager 开始 Tokenize
class TokenizerManager:
    async def generate_request(self, obj: GenerateReqInput):
        # 1. 提取文本
        text = obj.text  # "tell me what is sglang"
        
        # 2. Tokenize（第一次 tokenize）
        token_ids = self.tokenizer.encode(text)
        # token_ids = [101, 202, 303, 404, 505, 606]
        
        # 3. 发送到 Scheduler
        await self.send_to_scheduler(...)
```

**关键点**:
- ✅ **第一次 Tokenize**: Worker 内部的 TokenizerManager 才开始 tokenize
- ✅ **Token IDs**: 生成 token IDs，用于后续的前缀缓存匹配

---

#### 步骤 7: Scheduler 做真实前缀缓存匹配（token 级别）

**代码位置**: `python/sglang/srt/managers/schedule_policy.py:181`

```python
# Scheduler 做真实的前缀缓存匹配（token 级别）
r.prefix_indices, r.last_node, r.host_hit_length = (
    self.tree_cache.match_prefix(
        rid=r.rid,
        key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
        # prefix_ids = [101, 202, 303, 404, 505, 606] (token IDs)
    )
)
# 返回: prefix_indices = [0, 1, 2, 3, 4, 5] (KV Cache 索引)
```

**关键点**:
- ✅ **Token 级别**: Scheduler 使用 token IDs 做匹配
- ✅ **真实缓存**: 匹配的是真实的 KV Cache（GPU 内存中的）
- ✅ **返回索引**: 返回 KV Cache 在 GPU 内存中的索引

---

#### 步骤 8: Scheduler 构建 Batch，分配 GPU

**代码位置**: `python/sglang/srt/managers/scheduler.py:256`

```python
# Scheduler 构建 Batch，分配 GPU
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    # 1. 构建 Batch
    new_batch = self.get_new_batch_prefill()
    
    # 2. 分配 GPU 内存
    # - 使用 prefix_indices 获取缓存的 KV Cache
    # - 分配新部分的 KV Cache 内存
    
    # 3. 准备推理数据
    # - 构建 ForwardBatch
    # - 准备 Q/K/V 张量
    
    return new_batch
```

**关键点**:
- ✅ **GPU 分配**: Scheduler 负责分配 GPU 内存
- ✅ **使用缓存**: 使用 prefix_indices 获取缓存的 KV Cache
- ✅ **构建 Batch**: 构建推理 Batch，准备发送到 GPU

---

#### 步骤 9-12: GPU 推理 → Detokenize → 返回响应

**流程**:
```
GPU 推理 → 生成 token IDs → Detokenize → 构建 HTTP 响应 → 返回
```

**关键点**:
- ✅ **GPU 推理**: 使用缓存的 KV Cache，只计算新部分
- ✅ **Detokenize**: 将 token IDs 转回文本
- ✅ **返回响应**: Worker 返回 HTTP 响应给 Router，Router 转发给客户端

---

## 4. 场景 2: 不使用 Router（单 Worker）

### 4.1 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: 客户端发送 HTTP 请求                                 │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP POST /v1/chat/completions
                     │ 直接发送到 Worker（不经过 Router）
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: Worker 接收请求                                      │
│  - FastAPI 接收 HTTP 请求                                   │
│  - 解析 JSON body                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 3: TokenizerManager 开始 Tokenize                      │
│  - 提取文本                                                  │
│  - 调用 tokenizer.encode()                                  │
│  - 生成 token IDs                                            │
└────────────────────┬────────────────────────────────────────┘
                     │ token IDs
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 4: Scheduler 做真实前缀缓存匹配（token 级别）          │
│  - 调用 RadixCache.match_prefix()                          │
│  - 查找最长的匹配前缀                                        │
│  - 返回 prefix_indices                                      │
└────────────────────┬────────────────────────────────────────┘
                     │ prefix_indices
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 5: Scheduler 构建 Batch，分配 GPU                      │
│  - 构建 ScheduleBatch                                        │
│  - 分配 GPU 内存                                            │
│  - 准备推理数据                                              │
└────────────────────┬────────────────────────────────────────┘
                     │ Batch 数据
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 6: GPU 执行推理                                         │
│  - 使用缓存的 KV Cache                                       │
│  - 生成新的 token IDs                                        │
└────────────────────┬────────────────────────────────────────┘
                     │ 新 token IDs
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 7: Detokenizer 将 token IDs 转回文本                 │
│  - 调用 tokenizer.decode()                                  │
│  - 生成文本                                                  │
└────────────────────┬────────────────────────────────────────┘
                     │ 文本
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ 步骤 8: Worker 返回 HTTP 响应给客户端                       │
│  - 构建 JSON 响应                                           │
│  - 直接返回给客户端（不经过 Router）                        │
└─────────────────────────────────────────────────────────────┘
```

---

### 4.2 与场景 1 的区别

| 步骤 | 场景 1（使用 Router） | 场景 2（不使用 Router） |
|------|---------------------|----------------------|
| **请求发送** | 客户端 → Router | 客户端 → Worker（直接） |
| **前缀匹配** | Router 做近似匹配（文本级别） | 无（单 Worker，不需要路由） |
| **Worker 选择** | Router 选择 Worker | 无（只有一个 Worker） |
| **Tokenize** | Worker 内部 | Worker 内部（相同） |
| **GPU 分配** | Worker 内部的 Scheduler | Worker 内部的 Scheduler（相同） |
| **响应返回** | Worker → Router → 客户端 | Worker → 客户端（直接） |

---

## 5. 关键步骤详解

### 5.1 Router 的前缀匹配 vs Scheduler 的前缀匹配

**Router 的前缀匹配（文本级别）**:
```rust
// Router 使用原始文本做匹配
let text = "tell me what is sglang";
let (matched_text, matched_worker) = tree.prefix_match(text);
// matched_text = "tell me what is" (文本匹配)
```

**特点**:
- ✅ **文本级别**: 使用原始文本（字符串）
- ✅ **近似匹配**: 匹配的是近似 Radix Tree（存储文本历史）
- ✅ **快速决策**: 不需要 tokenize，速度快（< 1ms）
- ✅ **目的**: 选择最合适的 Worker

---

**Scheduler 的前缀匹配（token 级别）**:
```python
# Scheduler 使用 token IDs 做匹配
token_ids = [101, 202, 303, 404, 505, 606]
prefix_indices = tree_cache.match_prefix(
    key=RadixKey(token_ids=token_ids)
)
# prefix_indices = [0, 1, 2, 3, 4, 5] (KV Cache 索引)
```

**特点**:
- ✅ **Token 级别**: 使用 token IDs（整数数组）
- ✅ **真实匹配**: 匹配的是真实的 KV Cache（GPU 内存中的）
- ✅ **返回索引**: 返回 KV Cache 在 GPU 内存中的索引
- ✅ **目的**: 获取缓存的 KV Cache，减少 GPU 计算

---

### 5.2 Tokenize 的时机

**错误理解**: Tokenize 在 Router 之前

**正确理解**: Tokenize 在 Worker 内部

**流程**:
```
1. 客户端发送 HTTP 请求（包含文本）
   ↓
2. Router 提取文本（不 tokenize）
   ↓
3. Router 做文本级别的前缀匹配
   ↓
4. Router 转发 HTTP 请求到 Worker
   ↓
5. Worker 接收请求
   ↓
6. TokenizerManager 开始 Tokenize ← 第一次 tokenize
   ↓
7. Scheduler 使用 token IDs 做前缀缓存匹配
```

**关键点**:
- ✅ **Router 不 Tokenize**: Router 只提取文本，不进行 tokenize
- ✅ **Worker 才 Tokenize**: Worker 内部的 TokenizerManager 才开始 tokenize
- ✅ **两次匹配**: Router 做文本匹配（选 Worker），Scheduler 做 token 匹配（用缓存）

---

### 5.3 GPU 分配的时机

**错误理解**: GPU 分配在 Router

**正确理解**: GPU 分配在 Worker 内部的 Scheduler

**流程**:
```
1. Router 选择 Worker（不涉及 GPU）
   ↓
2. Router 转发请求到 Worker
   ↓
3. Worker 接收请求
   ↓
4. TokenizerManager Tokenize
   ↓
5. Scheduler 做前缀缓存匹配
   ↓
6. Scheduler 构建 Batch ← 开始分配 GPU
   ↓
7. Scheduler 分配 GPU 内存（KV Cache）
   ↓
8. GPU 执行推理
```

**关键点**:
- ✅ **Router 不分配 GPU**: Router 只负责路由，不涉及 GPU
- ✅ **Scheduler 分配 GPU**: Worker 内部的 Scheduler 负责分配 GPU 内存
- ✅ **GPU 在 Worker**: GPU 是 Worker 的资源，不是 Router 的资源

---

## 6. 常见误解纠正

### 6.1 误解 1: Tokenize 在 Router 之前

**错误理解**:
```
客户端 → Tokenize → Router → Worker
```

**正确理解**:
```
客户端 → Router（提取文本） → Worker → Tokenize
```

**原因**:
- Router 只需要文本做近似匹配，不需要 tokenize
- Tokenize 有开销，Router 避免不必要的 tokenize
- Worker 内部才需要 token IDs 做真实的前缀缓存匹配

---

### 6.2 误解 2: Router 做 token 级别的前缀匹配

**错误理解**:
```
Router 使用 token IDs 做前缀匹配
```

**正确理解**:
```
Router 使用原始文本做前缀匹配（文本级别）
Scheduler 使用 token IDs 做前缀匹配（token 级别）
```

**原因**:
- Router 的 Radix Tree 存储的是文本（字符），不是 token
- Router 的匹配是近似的，用于选择 Worker
- Scheduler 的匹配是精确的，用于获取真实的 KV Cache

---

### 6.3 误解 3: GPU 分配在 Router

**错误理解**:
```
Router 决定如何分配 GPU
```

**正确理解**:
```
Router 决定选择哪个 Worker
Worker 内部的 Scheduler 决定如何分配 GPU
```

**原因**:
- Router 是应用层负载均衡器，不涉及 GPU
- GPU 是 Worker 的资源，由 Worker 内部的 Scheduler 管理
- Router 只负责路由，Worker 负责推理

---

### 6.4 误解 4: Router 会再次 API Call

**错误理解**:
```
客户端 → Router → API Call → Worker
```

**正确理解**:
```
客户端 → Router（转发 HTTP 请求） → Worker
```

**原因**:
- Router 只是转发 HTTP 请求，不是重新发起 API Call
- Router 作为透明代理，客户端不知道有 Router
- 只有一次 API Call（客户端发起）

---

## 7. 总结

### 7.1 完整流程总结

**使用 Router（多 Worker）**:
```
1. 客户端发送 HTTP 请求到 Router
2. Router 提取文本，做文本级别的近似前缀匹配
3. Router 选择最合适的 Worker
4. Router 转发 HTTP 请求到 Worker
5. Worker 接收请求，TokenizerManager 开始 Tokenize
6. Scheduler 做 token 级别的真实前缀缓存匹配
7. Scheduler 构建 Batch，分配 GPU 内存
8. GPU 执行推理，生成 token IDs
9. Detokenizer 将 token IDs 转回文本
10. Worker 返回 HTTP 响应给 Router
11. Router 返回响应给客户端
```

**不使用 Router（单 Worker）**:
```
1. 客户端直接发送 HTTP 请求到 Worker
2. Worker 接收请求，TokenizerManager 开始 Tokenize
3. Scheduler 做 token 级别的真实前缀缓存匹配
4. Scheduler 构建 Batch，分配 GPU 内存
5. GPU 执行推理，生成 token IDs
6. Detokenizer 将 token IDs 转回文本
7. Worker 返回 HTTP 响应给客户端
```

---

### 7.2 关键点总结

| 步骤 | 位置 | 内容 | 时机 |
|------|------|------|------|
| **文本提取** | Router | 提取原始文本 | Router 接收请求时 |
| **近似前缀匹配** | Router | 文本级别的匹配 | Router 选择 Worker 时 |
| **HTTP 转发** | Router | 转发完整 HTTP 请求 | Router 选择 Worker 后 |
| **Tokenize** | Worker | 文本 → token IDs | Worker 接收请求后 |
| **真实前缀匹配** | Worker | token 级别的匹配 | Scheduler 处理请求时 |
| **GPU 分配** | Worker | 分配 GPU 内存 | Scheduler 构建 Batch 时 |
| **GPU 推理** | Worker | 执行模型推理 | Batch 准备好后 |
| **Detokenize** | Worker | token IDs → 文本 | GPU 推理完成后 |

---

### 7.3 纠正总结

**纠正的错误理解**:
1. ❌ Tokenize 在 Router 之前 → ✅ Tokenize 在 Worker 内部
2. ❌ Router 做 token 级别的匹配 → ✅ Router 做文本级别的匹配
3. ❌ GPU 分配在 Router → ✅ GPU 分配在 Worker 内部的 Scheduler
4. ❌ Router 会再次 API Call → ✅ Router 只是转发 HTTP 请求

**正确的理解**:
1. ✅ Router 负责路由（选择 Worker），不涉及 Tokenize 和 GPU
2. ✅ Worker 负责推理（Tokenize、GPU 分配、推理）
3. ✅ Router 做文本级别的近似匹配，Scheduler 做 token 级别的真实匹配
4. ✅ 只有一次 API Call（客户端发起），Router 只是转发

---

**结论**: 
- **Router** 是应用层负载均衡器，负责在多个 Worker 之间路由请求
- **Worker** 是推理服务，负责 Tokenize、GPU 分配、推理等
- **Tokenize** 在 Worker 内部，不在 Router
- **GPU 分配** 在 Worker 内部的 Scheduler，不在 Router
- **两次前缀匹配**: Router 做文本匹配（选 Worker），Scheduler 做 token 匹配（用缓存）

文档已保存到: `yc_self_learn/md/22_SGLang完整请求流程详解_纠正版.md`

