# Day 2_A1_B4_C2_D1：LangChain 性能优化详解

---
doc_type: glossary
layer: L3
scope_in:  LangChain 性能优化、瓶颈分析、优化策略（缓存、异步、批处理、RAG 优化）
scope_out: 具体框架使用教程（见 howto）；LangChain 架构设计（见 L4）；LangChain 深度优化（见 L4）
inputs:   (读者) 疑问：如何优化 LangChain 的性能？有哪些性能瓶颈？如何提升响应速度和降低成本？
outputs:  性能优化策略 + 瓶颈分析 + 优化技巧 + 代码示例 + 性能对比
entrypoints: [ 核心问题：LangChain 性能优化 ]
children: []
related: [ LangChain, 性能优化, 缓存, 异步, 批处理, RAG 优化, Day02_A1_B4_C2_LLM应用框架对比_LangChain_AutoGen_MetaDeck.md ]
---

## Definition（定义）

**核心问题**：**如何优化 LangChain 的性能？有哪些性能瓶颈？**

**核心答案**：
- ✅ **性能瓶颈**：LLM 调用延迟、链式调用开销、检索开销、大上下文窗口
- ✅ **优化策略**：缓存、异步、批处理、智能检索、流式输出、模型路由
- ✅ **Trade-off**：延迟 vs 吞吐量、成本 vs 质量、缓存 vs 新鲜度

**类比**：
- **LangChain 性能优化** = **优化生产线**（减少等待时间、提高效率、降低成本）

---

## 🎯 核心问题

### 问题场景

**场景1：响应慢**
- "LangChain 应用响应太慢，如何优化？"
- "用户等待时间太长，如何减少延迟？"

**场景2：成本高**
- "LangChain 应用成本太高，如何降低成本？"
- "Token 使用量太大，如何减少 Token 消耗？"

**场景3：吞吐量低**
- "LangChain 应用吞吐量太低，如何提升？"
- "无法处理大量并发请求，如何优化？"

---

## 📊 性能瓶颈分析

### 1. LLM 调用延迟（主要瓶颈）

**问题**：
- ✅ **LLM 调用是 IO-bound**：每次调用都有网络延迟
- ✅ **等待时间**：需要等待 LLM 响应才能继续
- ✅ **累积延迟**：多个链式调用会累积延迟

**例子**：
```python
# 性能问题：串行调用
chain1_result = chain1.run(input1)    # 等待 500ms
chain2_result = chain2.run(chain1_result)  # 等待 500ms
chain3_result = chain3.run(chain2_result)  # 等待 500ms
# 总延迟：1500ms
```

**影响**：
- ⚠️ **延迟高**：每个调用都要等待
- ⚠️ **吞吐量低**：无法并行处理多个请求
- ⚠️ **用户体验差**：响应时间长

---

### 2. 链式调用开销

**问题**：
- ✅ **多个链式调用**：多个 LLM 调用串联，延迟累积
- ✅ **中间结果处理**：每次调用都需要处理中间结果
- ✅ **Prompt 模板处理**：每次调用都需要处理 Prompt 模板

**例子**：
```python
# 链式调用：延迟累积
from langchain.chains import SequentialChain

pipeline = SequentialChain(
    chains=[chain1, chain2, chain3],  # 3 个链式调用
    input_variables=["input"],
    output_variables=["output"]
)

# 总延迟 = chain1延迟 + chain2延迟 + chain3延迟
# 500ms + 500ms + 500ms = 1500ms
```

**影响**：
- ⚠️ **延迟累积**：多个调用延迟相加
- ⚠️ **复杂度高**：需要管理多个链式调用

---

### 3. 检索开销（RAG 场景）

**问题**：
- ✅ **向量数据库查询**：每次检索都需要查询向量数据库
- ✅ **文档检索**：需要检索大量文档
- ✅ **Embedding 计算**：如果每次都重新计算 embedding

**例子**：
```python
# RAG 系统：检索开销
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 每次查询都需要检索
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# 检索 5 个文档：可能耗时 100-500ms

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 总延迟 = 检索延迟 + LLM 调用延迟
# 200ms + 500ms = 700ms
```

**影响**：
- ⚠️ **检索延迟**：向量数据库查询需要时间
- ⚠️ **Token 消耗**：检索到的文档会增加 Token 消耗

---

### 4. 大上下文窗口

**问题**：
- ✅ **Token 消耗大**：大上下文窗口消耗更多 Token
- ✅ **处理时间长**：处理大上下文需要更多时间
- ✅ **成本高**：更多 Token 意味着更高成本

**例子**：
```python
# 问题：上下文窗口太大
prompt = f"""
系统提示词：{system_prompt}  # 100 tokens
用户问题：{user_question}  # 50 tokens
检索到的文档：
{document1}  # 500 tokens
{document2}  # 500 tokens
{document3}  # 500 tokens
{document4}  # 500 tokens
{document5}  # 500 tokens

总 Token 数：2650 tokens
LLM 处理时间：~1-2秒
成本：$0.01（假设 $0.002/1K tokens）
"""

# 优化后：减少文档数量
prompt = f"""
系统提示词：{system_prompt}  # 100 tokens
用户问题：{user_question}  # 50 tokens
检索到的文档：
{top_document1}  # 300 tokens（只取最相关的）
{top_document2}  # 300 tokens

总 Token 数：750 tokens（减少 72%）
LLM 处理时间：~0.5-1秒（减少 50%）
成本：$0.003（减少 70%）
"""
```

**影响**：
- ⚠️ **延迟高**：处理大上下文需要更多时间
- ⚠️ **成本高**：更多 Token 意味着更高成本
- ⚠️ **可能超限**：可能超过模型的上下文窗口限制

---

## 💡 性能优化策略

### 1. 缓存优化（最重要）

**策略**：**缓存 LLM 响应、Embedding、检索结果**。

**优势**：
- ✅ **大幅降低延迟**：缓存命中时延迟接近 0ms
- ✅ **大幅降低成本**：不需要重复调用 LLM
- ✅ **提升用户体验**：响应速度快

**实现方法**：

#### a) LLM 响应缓存

```python
# 方法1：使用 LangChain 内置缓存
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

# 内存缓存（开发环境）
set_llm_cache(InMemoryCache())

# 或者使用 Redis 缓存（生产环境）
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0)
set_llm_cache(RedisCache(redis_client=redis_client))
```

**效果**：
- ✅ **延迟降低**：从 500ms 降到 < 1ms（缓存命中时）
- ✅ **成本降低**：从 $0.01 降到 $0（缓存命中时）
- ✅ **命中率**：通常 30-70%（取决于场景）

---

#### b) Embedding 缓存

```python
# 方法2：缓存 Embedding
from langchain.embeddings import OpenAIEmbeddings
from functools import lru_cache

# 使用 LRU 缓存（Python 内置）
@lru_cache(maxsize=1000)
def get_embedding(text: str) -> list:
    """缓存 Embedding 结果"""
    embeddings = OpenAIEmbeddings(base_url="http://sglang-server/v1")
    return embeddings.embed_query(text)

# 使用
doc_embedding = get_embedding("KYC 文档内容")  # 第一次：500ms
doc_embedding = get_embedding("KYC 文档内容")  # 第二次：< 1ms（缓存命中）
```

**效果**：
- ✅ **延迟降低**：从 500ms 降到 < 1ms（缓存命中时）
- ✅ **成本降低**：从 $0.001 降到 $0（缓存命中时）
- ✅ **命中率**：通常 50-90%（相同文档频繁查询）

---

#### c) 检索结果缓存

```python
# 方法3：缓存检索结果
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_retrieve(query: str) -> list:
    """缓存检索结果"""
    # 使用 query 的 hash 作为 key
    query_hash = hashlib.md5(query.encode()).hexdigest()
    
    # 检索文档（只在缓存未命中时执行）
    results = vectorstore.similarity_search(query, k=5)
    return results

# 使用
docs = cached_retrieve("什么是 KYC？")  # 第一次：200ms
docs = cached_retrieve("什么是 KYC？")  # 第二次：< 1ms（缓存命中）
```

**效果**：
- ✅ **延迟降低**：从 200ms 降到 < 1ms（缓存命中时）
- ✅ **成本降低**：减少向量数据库查询次数

---

### 2. 异步和批处理优化

**策略**：**使用异步调用和批处理，并行处理多个请求**。

**优势**：
- ✅ **提升吞吐量**：可以并行处理多个请求
- ✅ **减少等待时间**：不需要等待单个请求完成

**实现方法**：

#### a) 异步调用

```python
# 方法1：异步调用
import asyncio
from langchain.llms import OpenAI

llm = OpenAI(base_url="http://sglang-server/v1")

# 串行调用（慢）
results = []
for query in queries:  # 100 个查询
    result = llm.predict(query)  # 每个 500ms
    results.append(result)
# 总时间：100 × 500ms = 50 秒

# 异步调用（快）
async def process_queries(queries):
    tasks = [llm.apredict(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(process_queries(queries))  # 100 个查询
# 总时间：约 1-2 秒（并行处理，取决于并发限制）
```

**效果**：
- ✅ **延迟降低**：从 50 秒降到 1-2 秒（100 个查询）
- ✅ **吞吐量提升**：从 2 QPS 提升到 50-100 QPS

---

#### b) 批处理

```python
# 方法2：批处理
from langchain.llms import OpenAI

llm = OpenAI(base_url="http://sglang-server/v1")

# 单个调用（慢）
results = []
for query in queries:  # 100 个查询
    result = llm.predict(query)  # 每个 500ms
    results.append(result)
# 总时间：100 × 500ms = 50 秒

# 批处理（快）
results = llm.batch(queries)  # 100 个查询一次性处理
# 总时间：约 2-3 秒（批处理更高效）
```

**效果**：
- ✅ **延迟降低**：从 50 秒降到 2-3 秒（100 个查询）
- ✅ **吞吐量提升**：从 2 QPS 提升到 30-50 QPS
- ✅ **成本降低**：批处理可能有折扣（取决于提供商）

---

### 3. 智能检索优化（RAG 场景）

**策略**：**优化检索策略，减少检索文档数量，提高检索质量**。

**优势**：
- ✅ **降低延迟**：减少检索文档数量
- ✅ **降低成本**：减少 Token 消耗
- ✅ **提高质量**：只检索最相关的文档

**实现方法**：

#### a) 减少检索文档数量

```python
# 优化前：检索 5 个文档
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# 总 Token：2500 tokens（5 × 500 tokens）
# 延迟：200ms（检索）+ 1000ms（LLM）= 1200ms

# 优化后：只检索 2 个最相关的文档
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# 总 Token：1000 tokens（2 × 500 tokens，减少 60%）
# 延迟：100ms（检索）+ 500ms（LLM）= 600ms（减少 50%）
```

**效果**：
- ✅ **延迟降低**：从 1200ms 降到 600ms（减少 50%）
- ✅ **成本降低**：从 $0.01 降到 $0.004（减少 60%）

---

#### b) 优化 Chunk Size

```python
# 优化前：Chunk Size 太大
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个 chunk 1000 字符
    chunk_overlap=200
)
# 问题：Chunk 太大，包含无关信息

# 优化后：Chunk Size 适中
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # 每个 chunk 500 字符
    chunk_overlap=100
)
# 优势：Chunk 更精确，只包含相关信息
```

**效果**：
- ✅ **检索质量提升**：更精确的文档匹配
- ✅ **Token 消耗降低**：更小的 Chunk 意味着更少的 Token

---

#### c) 使用更高效的向量数据库

```python
# 优化前：使用慢速向量数据库
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents, embeddings)
# 检索延迟：200ms

# 优化后：使用快速向量数据库（支持异步）
from langchain.vectorstores import Qdrant
import qdrant_client

client = qdrant_client.QdrantClient(url="http://localhost:6333")
vectorstore = Qdrant(client=client, collection_name="kyc_docs")
# 检索延迟：50ms（异步支持，减少 75%）
```

**效果**：
- ✅ **延迟降低**：从 200ms 降到 50ms（减少 75%）
- ✅ **吞吐量提升**：支持异步查询，可以并行处理

---

### 4. 流式输出优化

**策略**：**使用流式输出，减少用户感知延迟**。

**优势**：
- ✅ **用户感知延迟降低**：用户不需要等待完整响应
- ✅ **用户体验提升**：可以实时看到生成的内容

**实现方法**：

```python
# 优化前：等待完整响应
result = chain.run(query)  # 等待 2 秒才返回
print(result)  # 用户等待 2 秒才看到结果

# 优化后：流式输出
for chunk in chain.stream(query):  # 立即开始输出
    print(chunk, end="", flush=True)  # 用户立即看到内容
# 用户感知延迟：< 100ms（看到第一个 token）
```

**效果**：
- ✅ **用户感知延迟降低**：从 2000ms 降到 < 100ms（看到第一个 token）
- ✅ **用户体验提升**：可以实时看到生成的内容

---

### 5. 模型路由优化

**策略**：**根据任务复杂度，选择不同规模的模型**。

**优势**：
- ✅ **降低成本**：简单任务使用小模型（便宜）
- ✅ **降低延迟**：小模型响应更快
- ✅ **提高质量**：复杂任务使用大模型（质量好）

**实现方法**：

```python
# 方法：根据任务复杂度选择模型
from langchain.llms import OpenAI

def get_model_for_task(task: str, complexity: str):
    """根据任务复杂度选择模型"""
    if complexity == "simple":
        # 简单任务：使用小模型（便宜、快速）
        return OpenAI(
            model="gpt-3.5-turbo",
            base_url="http://sglang-server/v1"
        )
    elif complexity == "complex":
        # 复杂任务：使用大模型（质量好）
        return OpenAI(
            model="gpt-4",
            base_url="http://sglang-server/v1"
        )

# 使用
simple_task = "回答简单问题：什么是 KYC？"
complex_task = "分析复杂的 KYC 合规问题：..."

simple_model = get_model_for_task(simple_task, "simple")  # 快速、便宜
complex_model = get_model_for_task(complex_task, "complex")  # 质量好
```

**效果**：
- ✅ **成本降低**：简单任务成本降低 90%（$0.01 → $0.001）
- ✅ **延迟降低**：简单任务延迟降低 50%（500ms → 250ms）
- ✅ **质量保证**：复杂任务仍然使用大模型

---

### 6. Prompt 优化

**策略**：**优化 Prompt，减少 Token 消耗，提高效率**。

**优势**：
- ✅ **降低 Token 消耗**：更短的 Prompt 意味着更少的 Token
- ✅ **降低成本**：更少的 Token 意味着更低的成本
- ✅ **降低延迟**：更短的 Prompt 意味着更快的处理

**实现方法**：

```python
# 优化前：Prompt 太长、包含无关信息
prompt = f"""
你是一个专业的 KYC 合规专家。
请仔细分析以下用户信息：
用户姓名：{user_name}
用户年龄：{user_age}
用户地址：{user_address}
用户职业：{user_occupation}
用户收入：{user_income}
用户银行账户：{user_bank_account}
用户信用卡：{user_credit_card}
用户投资历史：{user_investment_history}
用户风险偏好：{user_risk_preference}
...

请根据以上信息，判断该用户是否符合 KYC 要求。
请详细说明你的判断依据。
请列出所有可能的风险点。
请提供详细的合规建议。
...
"""
# Token 数：500 tokens

# 优化后：Prompt 简洁、只包含必要信息
prompt = f"""
分析 KYC 请求：
姓名：{user_name}
年龄：{user_age}
职业：{user_occupation}
收入：{user_income}

判断是否符合 KYC 要求，并说明风险点。
"""
# Token 数：100 tokens（减少 80%）
```

**效果**：
- ✅ **Token 消耗降低**：从 500 tokens 降到 100 tokens（减少 80%）
- ✅ **成本降低**：从 $0.01 降到 $0.002（减少 80%）
- ✅ **延迟降低**：从 1000ms 降到 400ms（减少 60%）

---

### 7. 记忆管理优化

**策略**：**优化对话历史管理，避免发送过长的历史**。

**优势**：
- ✅ **降低 Token 消耗**：不发送过长的历史
- ✅ **降低成本**：更少的 Token 意味着更低的成本

**实现方法**：

```python
# 优化前：发送完整对话历史
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
# 问题：历史越来越长，Token 消耗越来越大

# 优化后：只保留最近的对话或总结
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,  # 最多 1000 tokens
    return_messages=True
)
# 优势：超过限制时，自动总结旧对话
```

**效果**：
- ✅ **Token 消耗降低**：从 5000 tokens 降到 1000 tokens（减少 80%）
- ✅ **成本降低**：从 $0.05 降到 $0.01（减少 80%）
- ✅ **上下文保持**：通过总结保留重要信息

---

## ⚖️ 优化策略 Trade-off

### 1. 缓存 vs 新鲜度

**缓存**：
- ✅ **优势**：大幅降低延迟和成本
- ⚠️ **劣势**：可能返回过时的数据

**Trade-off**：
```
如果你需要实时数据 → 减少缓存（放弃性能优势）
如果你可以接受略微过时 → 使用缓存（获得性能和成本优势）
```

---

### 2. 批处理 vs 延迟

**批处理**：
- ✅ **优势**：提升吞吐量、降低成本
- ⚠️ **劣势**：增加单个请求的延迟（需要等待批处理完成）

**Trade-off**：
```
如果你重视吞吐量 → 使用批处理（放弃低延迟）
如果你重视延迟 → 使用异步（放弃批处理优势）
```

---

### 3. 模型大小 vs 成本

**大模型**：
- ✅ **优势**：质量好
- ⚠️ **劣势**：成本高、延迟高

**小模型**：
- ✅ **优势**：成本低、延迟低
- ⚠️ **劣势**：质量可能不如大模型

**Trade-off**：
```
如果你重视质量 → 使用大模型（放弃成本和性能优势）
如果你重视成本和性能 → 使用小模型（放弃质量优势）
或者：根据任务复杂度选择模型（智能路由）
```

---

## 💡 实际应用场景（KYC项目）

### 场景1：RAG 系统优化

**优化前**：
- ❌ **检索延迟**：200ms（检索 5 个文档）
- ❌ **LLM 延迟**：1000ms（处理大上下文）
- ❌ **总延迟**：1200ms
- ❌ **成本**：$0.01 per request

**优化后**（应用多个策略）：
- ✅ **检索延迟**：50ms（使用 Qdrant 异步，只检索 2 个文档）
- ✅ **LLM 延迟**：400ms（优化 Prompt，减少 Token）
- ✅ **缓存命中**：30% 命中率（平均延迟降低 30%）
- ✅ **总延迟**：约 300ms（减少 75%）
- ✅ **成本**：约 $0.003 per request（减少 70%）

**优化策略**：
1. ✅ **使用 Qdrant**（异步向量数据库）
2. ✅ **减少检索文档数量**（从 5 个降到 2 个）
3. ✅ **缓存检索结果**（30% 命中率）
4. ✅ **优化 Prompt**（减少 Token 消耗）

---

### 场景2：链式调用优化

**优化前**：
- ❌ **串行调用**：chain1 (500ms) → chain2 (500ms) → chain3 (500ms)
- ❌ **总延迟**：1500ms

**优化后**（使用异步）：
- ✅ **并行调用**：chain1、chain2、chain3 并行执行（如果可能）
- ✅ **总延迟**：约 500ms（减少 67%）

**代码示例**：
```python
# 优化前：串行调用
result1 = chain1.run(input)      # 500ms
result2 = chain2.run(result1)    # 500ms
result3 = chain3.run(result2)    # 500ms
# 总延迟：1500ms

# 优化后：并行调用（如果可能）
import asyncio

async def parallel_chains(input):
    # 并行执行（如果依赖允许）
    tasks = [
        chain1.arun(input),
        chain2.arun(input),  # 如果不需要 result1
        chain3.arun(input)   # 如果不需要 result2
    ]
    results = await asyncio.gather(*tasks)
    return results

# 或者：合并链式调用（减少调用次数）
# 将多个链式调用合并为一个调用
combined_prompt = f"""
步骤1：{step1_prompt}
步骤2：{step2_prompt}
步骤3：{step3_prompt}
"""
result = llm.predict(combined_prompt)  # 只需要一次调用
```

---

### 场景3：批处理优化

**优化前**：
- ❌ **单个调用**：100 个查询，每个 500ms
- ❌ **总时间**：50 秒
- ❌ **吞吐量**：2 QPS

**优化后**（使用批处理）：
- ✅ **批处理**：100 个查询一次性处理
- ✅ **总时间**：约 2-3 秒
- ✅ **吞吐量**：30-50 QPS（提升 15-25 倍）

**代码示例**：
```python
# 优化前：单个调用
results = []
for query in queries:  # 100 个查询
    result = llm.predict(query)  # 每个 500ms
    results.append(result)
# 总时间：50 秒

# 优化后：批处理
results = llm.batch(queries, config={"max_concurrency": 10})  # 100 个查询
# 总时间：约 2-3 秒（批处理更高效）
```

---

## 📊 性能优化效果对比

### 优化前后对比

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| **平均延迟** | 1200ms | 300ms | ⬇️ 75% |
| **P95 延迟** | 2000ms | 600ms | ⬇️ 70% |
| **吞吐量（QPS）** | 2 | 30-50 | ⬆️ 15-25x |
| **成本 per request** | $0.01 | $0.003 | ⬇️ 70% |
| **缓存命中率** | 0% | 30% | ⬆️ 30% |

**优化策略组合**：
1. ✅ **缓存**：30% 命中率，减少 30% 延迟和成本
2. ✅ **异步/批处理**：提升 15-25x 吞吐量
3. ✅ **智能检索**：减少 50% 检索延迟
4. ✅ **Prompt 优化**：减少 60% Token 消耗
5. ✅ **流式输出**：用户感知延迟降低到 < 100ms

---

## 💡 总结

### 核心答案

**如何优化 LangChain 的性能？**

**答案**：
1. ✅ **缓存优化**：缓存 LLM 响应、Embedding、检索结果（最重要）
2. ✅ **异步/批处理**：使用异步调用和批处理，并行处理多个请求
3. ✅ **智能检索**：优化检索策略，减少检索文档数量
4. ✅ **流式输出**：使用流式输出，减少用户感知延迟
5. ✅ **模型路由**：根据任务复杂度选择不同规模的模型
6. ✅ **Prompt 优化**：优化 Prompt，减少 Token 消耗
7. ✅ **记忆管理**：优化对话历史管理，避免发送过长的历史

**主要性能瓶颈是什么？**

**答案**：
1. ✅ **LLM 调用延迟**：每次调用都有网络延迟（500ms+）
2. ✅ **链式调用开销**：多个调用延迟累积
3. ✅ **检索开销**：向量数据库查询需要时间（200ms+）
4. ✅ **大上下文窗口**：大上下文消耗更多 Token 和时间

**优化效果**：

| 策略 | 延迟改善 | 成本改善 | 吞吐量改善 |
|------|---------|---------|-----------|
| **缓存** | ⬇️ 30-70% | ⬇️ 30-70% | - |
| **异步/批处理** | - | ⬇️ 10-20% | ⬆️ 15-25x |
| **智能检索** | ⬇️ 50% | ⬇️ 60% | - |
| **Prompt 优化** | ⬇️ 60% | ⬇️ 80% | - |

### 关键要点

1. **缓存是最重要的优化**：可以大幅降低延迟和成本
2. **异步/批处理提升吞吐量**：适合处理大量请求
3. **智能检索优化 RAG**：减少检索文档数量，提高质量
4. **Prompt 优化减少 Token**：降低成本，提升速度

### 面试话术

- ✅ "我们使用多层缓存策略：LLM 响应缓存（Redis）、Embedding 缓存（内存）、检索结果缓存（LRU），缓存命中率达到 30-70%，延迟降低 30-70%，成本降低 30-70%。"
- ✅ "我们使用异步调用和批处理：对于独立的任务，使用 `asyncio.gather` 并行处理；对于相似的任务，使用 `batch` 批处理，吞吐量提升 15-25 倍。"
- ✅ "我们优化 RAG 检索：减少检索文档数量（从 5 个降到 2 个），使用高效的向量数据库（Qdrant 异步），检索延迟降低 75%，Token 消耗降低 60%。"

---

## Links

| 类型 | 对象 |
|------|------|
| **Parent** | A1_B4_C2 LLM 应用框架对比（[Day02_A1_B4_C2_LLM应用框架对比_LangChain_AutoGen_MetaDeck.md](./Day02_A1_B4_C2_LLM应用框架对比_LangChain_AutoGen_MetaDeck.md)） |
| **Related** | LangChain、性能优化、缓存、异步、批处理、RAG 优化 |
