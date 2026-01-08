# A01: LLM 推理系统设计

## 📋 目录

1. [什么是系统设计？](#什么是系统设计)
2. [LLM 推理系统概述](#llm-推理系统概述)
3. [核心组件详解](#核心组件详解)
4. [系统设计流程](#系统设计流程)
5. [实际案例：SGLang 架构](#实际案例sglang-架构)

---

## 什么是系统设计？

### 系统设计的定义

**系统设计（System Design）** 是指设计和构建**大规模、可扩展、可靠**的软件系统的过程。它不仅仅是写代码，而是：

1. **架构设计**：定义系统的整体结构
2. **组件设计**：设计各个模块和它们之间的交互
3. **性能优化**：确保系统满足性能要求
4. **可扩展性**：系统能够处理增长的需求
5. **可靠性**：系统能够处理故障和错误

### 系统设计 vs 算法设计

| 维度 | 算法设计 | 系统设计 |
|------|---------|---------|
| **关注点** | 单个问题的解决方案 | 整个系统的架构 |
| **规模** | 单机、单线程 | 分布式、多机器 |
| **时间范围** | 毫秒到秒级 | 长期运行（天、月、年） |
| **复杂度** | 时间复杂度、空间复杂度 | 可扩展性、可靠性、一致性 |
| **例子** | 排序算法、图算法 | 搜索引擎、社交网络、LLM 推理服务 |

### 系统设计的核心问题

在设计一个系统时，我们需要回答以下问题：

1. **功能需求**：系统需要做什么？
2. **性能需求**：系统需要多快？能处理多少请求？
3. **可扩展性**：如何扩展到 10x、100x 规模？
4. **可靠性**：如何处理故障？如何保证可用性？
5. **一致性**：数据如何保持一致？
6. **权衡（Trade-offs）**：延迟 vs 吞吐量？一致性 vs 可用性？

---

## LLM 推理系统概述

### 什么是 LLM 推理系统？

**LLM 推理系统** 是一个用于**部署和运行大语言模型**的系统，它需要：

1. **接收用户请求**：处理文本输入
2. **运行模型推理**：在 GPU 上执行模型计算
3. **生成响应**：返回生成的文本
4. **管理资源**：高效利用 GPU、内存等资源

### LLM 推理系统的挑战

#### 1. **高延迟要求**
- **TTFT (Time To First Token)**：用户希望尽快看到第一个 token
- **E2E (End-to-End) 延迟**：整个响应时间要短
- **挑战**：模型很大，计算需要时间

#### 2. **高吞吐量要求**
- **QPS (Queries Per Second)**：需要处理大量并发请求
- **吞吐量 (Throughput)**：每秒生成的 tokens 数
- **挑战**：GPU 资源有限，如何最大化利用率？

#### 3. **内存管理**
- **模型权重**：70B 模型需要 140GB+ 内存
- **KV Cache**：每个请求需要缓存历史 tokens 的 K、V
- **挑战**：GPU 内存有限，如何高效管理？

#### 4. **动态负载**
- **请求长度不同**：输入从几个 token 到几千个 token
- **输出长度不同**：输出从几个 token 到几百个 token
- **挑战**：如何动态调度和批处理？

---

## 核心组件详解

### 1. 请求调度器（Scheduler）

**作用**：决定哪些请求应该被处理，以及何时处理。

#### 核心功能

```
1. 请求队列管理
   - 等待队列（Waiting Queue）：新到达的请求
   - 运行队列（Running Queue）：正在处理的请求
   - 完成队列（Finished Queue）：已完成的请求

2. 批处理决策
   - 何时创建新的 batch？
   - 哪些请求应该加入 batch？
   - batch 大小应该是多少？

3. 资源管理
   - 检查 GPU 内存是否足够
   - 检查 KV Cache 内存池是否满
   - 决定是否接受新请求
```

#### 调度策略

**a) FCFS (First Come First Served)**
- 先来先服务
- 简单，但可能不是最优

**b) LPM (Longest Prefix Match)**
- 优先处理有最长前缀匹配的请求
- 提高缓存命中率（RadixAttention）

**c) Priority Scheduling**
- 根据优先级调度
- 高优先级请求可以抢占低优先级请求

**d) 动态批处理（Dynamic Batching）**
- 根据资源情况动态调整 batch 大小
- 平衡延迟和吞吐量

#### 代码示例（SGLang）

```python
# python/sglang/srt/managers/scheduler.py

class Scheduler:
    def __init__(self):
        self.waiting_queue = []  # 等待队列
        self.running_batch = None  # 当前运行的 batch
        
    def get_new_batch_prefill(self):
        """创建新的 prefill batch"""
        # 1. 检查是否有等待的请求
        if len(self.waiting_queue) == 0:
            return None
        
        # 2. 检查资源是否足够
        if not self.has_enough_memory():
            return None
        
        # 3. 选择请求加入 batch
        batch = self.select_requests_for_batch()
        return batch
    
    def select_requests_for_batch(self):
        """选择请求加入 batch"""
        # 根据调度策略选择请求
        # - FCFS: 按到达时间
        # - LPM: 按前缀匹配长度
        # - Priority: 按优先级
        ...
```

---

### 2. 批处理管理器（Batch Manager）

**作用**：管理批处理的生命周期，包括创建、更新、完成。

#### 批处理的两种类型

**a) Prefill Batch（预填充批处理）**
- 处理新请求的输入部分
- 计算所有输入 tokens 的注意力
- 生成 KV Cache

**b) Decode Batch（解码批处理）**
- 处理正在生成的请求
- 每次只处理一个 token
- 使用已有的 KV Cache

#### 连续批处理（Continuous Batching）

**传统静态批处理的问题**：
```
请求 1: [输入] → [生成 token 1] → [生成 token 2] → [生成 token 3] → 完成
请求 2: [输入] → [生成 token 1] → [生成 token 2] → [生成 token 3] → 完成
请求 3: [输入] → [生成 token 1] → [生成 token 2] → [生成 token 3] → 完成

问题：
- 所有请求必须同时完成
- 如果请求 1 先完成，也要等待其他请求
- GPU 利用率低
```

**连续批处理的优势**：
```
时间步 1: [请求 1, 请求 2, 请求 3] → 生成 token 1
时间步 2: [请求 1, 请求 2, 请求 3] → 生成 token 2
时间步 3: [请求 1 完成，移除] [请求 2, 请求 3, 新请求 4] → 生成 token 3
时间步 4: [请求 2, 请求 3, 请求 4] → 生成 token 4

优势：
- 请求完成后立即移除
- 新请求可以立即加入
- GPU 利用率高
```

#### 代码示例（SGLang）

```python
# python/sglang/srt/managers/schedule_batch.py

class ScheduleBatch:
    def __init__(self):
        self.reqs = []  # 批处理中的请求列表
        
    def filter_batch(self):
        """过滤已完成的请求"""
        # 移除已完成的请求
        self.reqs = [req for req in self.reqs if not req.finished()]
        
    def add_new_requests(self, new_reqs):
        """添加新请求到批处理"""
        # 在 decode 阶段，可以动态添加新请求
        self.reqs.extend(new_reqs)
```

---

### 3. KV Cache 内存管理器

**作用**：管理 KV Cache 的内存分配和释放。

#### KV Cache 是什么？

在 Transformer 的注意力机制中：
- **K (Key)**：用于计算注意力权重
- **V (Value)**：用于生成输出

**为什么需要缓存？**
- 在自回归生成中，每个新 token 都需要访问所有历史 tokens
- 如果不缓存，需要重新计算所有历史 tokens 的 K、V
- 缓存后，只需要计算新 token 的 K、V

#### KV Cache 内存大小

```
每个 token 的 KV Cache 大小：
  = 2 × num_kv_heads × head_dim × dtype_size

例如（LLaMA-2 70B，使用 GQA）：
  = 2 × 8 × 128 × 2 bytes
  = 4 KB per token

对于 2048 个 tokens：
  = 4 KB × 2048 = 8 MB per request

对于 1000 个并发请求：
  = 8 MB × 1000 = 8 GB
```

#### 内存池管理

**a) 固定大小内存池**
```
总内存池大小 = 固定值（如 72 GB）
每个请求按需分配内存
当内存不足时，拒绝新请求或撤回旧请求
```

**b) 分页管理（Paged Memory）**
```
将内存分成固定大小的页（如 16 tokens/page）
每个请求按页分配
减少内存碎片
支持动态扩展
```

**c) 前缀缓存（Prefix Caching）**
```
使用 Radix Tree 存储共享前缀
多个请求可以共享相同的前缀
减少内存占用
```

#### 代码示例（SGLang）

```python
# python/sglang/srt/mem_cache/allocator.py

class TokenToKVPoolAllocator:
    def __init__(self, size, page_size):
        self.size = size  # 总内存池大小
        self.page_size = page_size  # 每页大小
        self.free_pages = []  # 空闲页列表
        
    def alloc(self, need_size):
        """分配内存"""
        num_pages = need_size // self.page_size
        if num_pages > len(self.free_pages):
            return None  # 内存不足
        
        # 分配页面
        pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        return pages
    
    def free(self, pages):
        """释放内存"""
        self.free_pages.extend(pages)
```

---

### 4. 模型服务（Model Server）

**作用**：在 GPU 上执行模型推理。

#### 推理流程

**a) Prefill 阶段**
```
输入: [token_1, token_2, ..., token_n]
过程:
  1. Token Embedding
  2. Position Embedding
  3. Transformer Layers (n 次前向传播)
  4. 生成 KV Cache
  5. 输出最后一个 token 的 hidden state
输出: hidden state (用于生成第一个 token)
```

**b) Decode 阶段**
```
输入: 上一个 token 的 hidden state
过程:
  1. Token Embedding (只有新 token)
  2. Transformer Layers (1 次前向传播)
  3. 使用已有的 KV Cache
  4. 更新 KV Cache (添加新 token 的 K、V)
  5. 输出下一个 token 的 hidden state
输出: hidden state (用于生成下一个 token)
```

#### 性能优化

**a) FlashAttention**
- 优化注意力计算的内存访问
- 减少 HBM (High Bandwidth Memory) 访问
- 提高计算效率

**b) 量化（Quantization）**
- FP16/BF16：半精度浮点数
- INT8/INT4：整数量化
- 减少内存占用和计算量

**c) 模型并行（Model Parallelism）**
- 将模型分片到多个 GPU
- 减少单个 GPU 的内存压力
- 提高可扩展性

---

### 5. 负载均衡器（Load Balancer）

**作用**：将请求分发到多个模型服务实例。

#### 负载均衡策略

**a) Round Robin（轮询）**
- 依次分发到每个实例
- 简单，但可能不均匀

**b) Least Connections（最少连接）**
- 分发到连接数最少的实例
- 更均匀的负载分布

**c) Cache-Aware Routing（缓存感知路由）**
- 根据请求的前缀匹配路由
- 提高缓存命中率
- SGLang 的 RadixAttention 使用此策略

#### 代码示例（SGLang Router）

```rust
// sgl-router/src/policies/cache_aware.rs

fn route_request(request: &Request, workers: &[Worker]) -> WorkerId {
    // 1. 找到前缀匹配率最高的 worker
    let best_match = workers.iter()
        .map(|w| (w.id, calculate_prefix_match(request, w)))
        .max_by_key(|(_, match_rate)| *match_rate);
    
    // 2. 如果匹配率足够高，路由到该 worker
    if best_match.match_rate > CACHE_THRESHOLD {
        return best_match.worker_id;
    }
    
    // 3. 否则，路由到负载最轻的 worker
    return find_least_loaded_worker(workers);
}
```

---

## 系统设计流程

### 步骤 1: 需求澄清（5-10 分钟）

**关键问题**：
1. **功能需求**：
   - 支持哪些模型？
   - 支持哪些功能（流式输出、结构化输出等）？
   - 是否需要多租户？

2. **性能需求**：
   - QPS 要求是多少？
   - TTFT 目标是多少？
   - E2E 延迟目标是多少？

3. **规模需求**：
   - 预期并发用户数？
   - 平均请求长度？
   - 峰值负载？

4. **约束条件**：
   - GPU 资源限制？
   - 内存限制？
   - 成本限制？

### 步骤 2: 高层设计（10-15 分钟）

**画出系统架构图**：

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Load Balancer   │  ← 负载均衡
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Scheduler     │  ← 请求调度
│  - Waiting Q    │
│  - Running Q    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Batch Manager  │  ← 批处理管理
│  - Prefill      │
│  - Decode       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  KV Cache Pool  │  ← 内存管理
│  - Allocator    │
│  - Prefix Cache │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Model Server   │  ← 模型推理
│  - GPU Cluster  │
└─────────────────┘
```

### 步骤 3: 详细设计（15-20 分钟）

**深入关键组件**：

#### a) 调度器设计
- 使用什么调度策略？
- 如何管理请求队列？
- 如何决定 batch 大小？

#### b) 内存管理设计
- KV Cache 内存池大小？
- 使用什么分配策略？
- 如何处理内存不足？

#### c) 批处理设计
- 如何实现连续批处理？
- 如何动态添加/移除请求？
- 如何平衡延迟和吞吐量？

#### d) 性能优化
- 使用哪些优化技术？
- 如何提高 GPU 利用率？
- 如何减少延迟？

### 步骤 4: 扩展和优化（10-15 分钟）

**讨论可扩展性**：
- 如何扩展到 10x 规模？
- 如何扩展到 100x 规模？
- 瓶颈在哪里？

**讨论容错和恢复**：
- 如何处理 GPU 故障？
- 如何处理网络故障？
- 如何保证服务可用性？

**讨论监控和告警**：
- 需要监控哪些指标？
- 如何设置告警？
- 如何调试问题？

---

## 实际案例：SGLang 架构

### SGLang 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    SGLang Router                        │
│  - Cache-Aware Routing                                  │
│  - Load Balancing                                       │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ SGLang       │      │ SGLang       │
│ Worker 1     │      │ Worker 2     │
│              │      │              │
│ ┌──────────┐ │      │ ┌──────────┐ │
│ │Scheduler │ │      │ │Scheduler │ │
│ └────┬─────┘ │      │ └────┬─────┘ │
│      │       │      │      │       │
│ ┌────▼─────┐ │      │ ┌────▼─────┐ │
│ │Batch Mgr │ │      │ │Batch Mgr │ │
│ └────┬─────┘ │      │ └────┬─────┘ │
│      │       │      │      │       │
│ ┌────▼─────┐ │      │ ┌────▼─────┐ │
│ │KV Cache  │ │      │ │KV Cache  │ │
│ │Pool      │ │      │ │Pool      │ │
│ └────┬─────┘ │      │ └────┬─────┘ │
│      │       │      │      │       │
│ ┌────▼─────┐ │      │ ┌────▼─────┐ │
│ │Model     │ │      │ │Model     │ │
│ │Runner    │ │      │ │Runner    │ │
│ └──────────┘ │      │ └──────────┘ │
└──────────────┘      └──────────────┘
```

### SGLang 的核心特性

#### 1. RadixAttention（前缀缓存）
- 使用 Radix Tree 存储共享前缀
- 多个请求共享相同前缀的 KV Cache
- 显著减少内存占用和计算量

#### 2. 零开销调度器
- 高效的批处理调度
- 最小化 CPU 开销
- 智能的请求优先级管理

#### 3. 连续批处理
- 动态添加/移除请求
- 最大化 GPU 利用率
- 平衡延迟和吞吐量

#### 4. Prefill-Decode 分离
- 独立扩展 Prefill 和 Decode 阶段
- 针对性地优化资源分配
- 提高整体性能

#### 5. 多种并行策略
- 张量并行（Tensor Parallelism）
- 流水线并行（Pipeline Parallelism）
- 数据并行（Data Parallelism）
- 专家并行（Expert Parallelism）

---

## 📝 总结

### 系统设计的核心要点

1. **理解需求**：明确功能、性能、规模需求
2. **设计架构**：从高层到细节，逐步深入
3. **考虑权衡**：延迟 vs 吞吐量，内存 vs 计算
4. **关注可扩展性**：如何扩展到更大规模
5. **保证可靠性**：如何处理故障和错误

### LLM 推理系统的关键组件

1. **调度器**：决定何时处理哪些请求
2. **批处理管理器**：管理批处理的生命周期
3. **KV Cache 管理器**：管理内存分配和释放
4. **模型服务**：在 GPU 上执行推理
5. **负载均衡器**：分发请求到多个实例

### 下一步学习

- 深入学习每个组件的实现细节
- 阅读 SGLang/vLLM 源码
- 练习系统设计题目
- 理解性能优化技术

**通过系统学习和实践，你将能够设计出高性能、可扩展的 LLM 推理系统！** 🚀

