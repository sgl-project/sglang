# B01: LLM 推理系统设计面试问答

## 📋 目录

1. [基础概念问题](#基础概念问题)
2. [系统架构问题](#系统架构问题)
3. [请求处理流程问题](#请求处理流程问题)
4. [性能优化问题](#性能优化问题)
5. [深入设计问题](#深入设计问题)

---

## 基础概念问题

### Q1: 什么是系统设计？系统设计和算法设计有什么区别？

**面试官意图**：考察候选人对系统设计基础概念的理解

**参考答案**：

**系统设计**是设计和构建**大规模、可扩展、可靠**的软件系统的过程。它关注的是整个系统的架构，而不仅仅是单个问题的解决方案。

**系统设计 vs 算法设计**：

| 维度 | 算法设计 | 系统设计 |
|------|---------|---------|
| **关注点** | 单个问题的解决方案 | 整个系统的架构 |
| **规模** | 单机、单线程 | 分布式、多机器 |
| **时间范围** | 毫秒到秒级 | 长期运行（天、月、年） |
| **复杂度** | 时间复杂度、空间复杂度 | 可扩展性、可靠性、一致性 |
| **例子** | 排序算法、图算法 | 搜索引擎、LLM 推理服务 |

**系统设计的核心问题**：
1. 功能需求：系统需要做什么？
2. 性能需求：系统需要多快？能处理多少请求？
3. 可扩展性：如何扩展到 10x、100x 规模？
4. 可靠性：如何处理故障？如何保证可用性？
5. 权衡（Trade-offs）：延迟 vs 吞吐量？一致性 vs 可用性？

---

### Q2: LLM 推理系统面临哪些主要挑战？

**面试官意图**：考察候选人对 LLM 推理系统特殊性的理解

**参考答案**：

LLM 推理系统面临 4 个主要挑战：

#### 1. **高延迟要求**
- **TTFT (Time To First Token)**：用户希望尽快看到第一个 token
- **E2E (End-to-End) 延迟**：整个响应时间要短
- **挑战**：模型很大（70B+ 参数），计算需要时间

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

**实际影响**：
- 如果只关注延迟 → 吞吐量低，GPU 利用率低
- 如果只关注吞吐量 → 延迟高，用户体验差
- 需要在延迟和吞吐量之间找到平衡

---

### Q3: 请解释一下 LLM 推理系统的核心组件有哪些？

**面试官意图**：考察候选人对系统架构的整体理解

**参考答案**：

LLM 推理系统有 5 个核心组件：

#### 1. **请求调度器（Scheduler）**
- **作用**：决定哪些请求应该被处理，以及何时处理
- **核心功能**：
  - 请求队列管理（Waiting Queue、Running Queue）
  - 批处理决策（何时创建 batch、哪些请求加入）
  - 资源管理（检查 GPU 内存、KV Cache 内存池）

#### 2. **批处理管理器（Batch Manager）**
- **作用**：管理批处理的生命周期
- **核心功能**：
  - Prefill Batch：处理新请求的输入
  - Decode Batch：处理正在生成的请求
  - 连续批处理（Continuous Batching）：动态添加/移除请求

#### 3. **KV Cache 内存管理器**
- **作用**：管理 KV Cache 的内存分配和释放
- **核心功能**：
  - 内存池管理（固定大小、分页管理）
  - 前缀缓存（Prefix Caching）：共享相同前缀的 KV Cache
  - 内存分配和释放

#### 4. **模型服务（Model Server）**
- **作用**：在 GPU 上执行模型推理
- **核心功能**：
  - Prefill 阶段：处理输入，生成 KV Cache
  - Decode 阶段：使用 KV Cache 生成输出
  - 性能优化（FlashAttention、量化、模型并行）

#### 5. **负载均衡器（Load Balancer / Router）**
- **作用**：将请求分发到多个模型服务实例
- **核心功能**：
  - 负载均衡（Round Robin、Least Connections）
  - Cache-Aware Routing：根据前缀匹配路由
  - 健康检查和故障转移

**组件交互**：
```
Client → Router → Worker → Scheduler → Batch Manager → KV Cache Manager → Model Server
```

---

## 系统架构问题

### Q4: 请设计一个支持 1000 QPS 的 LLM 推理服务，你会如何设计？

**面试官意图**：考察候选人的系统设计能力，这是最经典的 LLM 推理系统设计题目

**参考答案**：

我会按照以下步骤设计：

#### 步骤 1: 需求澄清（5-10 分钟）

**关键问题**：
1. **功能需求**：
   - 支持哪些模型？（假设 LLaMA-2 70B）
   - 支持流式输出吗？（是）
   - 是否需要多租户？（否）

2. **性能需求**：
   - QPS：1000 requests/second
   - TTFT：< 100ms
   - E2E 延迟：< 2s（平均）

3. **规模需求**：
   - 平均输入长度：100 tokens
   - 平均输出长度：50 tokens
   - 峰值负载：2000 QPS

4. **约束条件**：
   - GPU：H100 (80GB)
   - 成本：尽可能优化

#### 步骤 2: 高层设计（10-15 分钟）

**系统架构**：

```
┌─────────────────────────────────────────────────────────┐
│                    Client Layer                         │
│  - 1000 QPS 请求                                        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Load Balancer (Router)                      │
│  - Cache-Aware Routing                                  │
│  - Load Balancing                                      │
│  - Health Check                                         │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Worker 1     │      │ Worker 2     │
│ (4x H100)    │      │ (4x H100)    │
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

#### 步骤 3: 详细设计（15-20 分钟）

**a) 性能估算**

```
假设 LLaMA-2 70B (使用 GQA):
  - 模型权重: 140GB (FP16)
  - KV Cache per token: 2 × 8 × 128 × 2 = 4KB
  - 平均序列长度: 150 tokens (100 input + 50 output)
  - KV Cache per request: 4KB × 150 = 600KB

GPU 资源需求:
  - 模型权重: 140GB (需要 2x H100 做模型并行)
  - KV Cache 池: 72GB (90% of 80GB)
  - 可缓存 tokens: 72GB / 4KB = 18M tokens
  - 可并发请求数: 18M / 150 = 120,000 requests

吞吐量估算:
  - 单个 H100 解码吞吐量: ~200 tokens/s
  - 4x H100 解码吞吐量: ~800 tokens/s
  - 1000 QPS × 50 tokens/request = 50,000 tokens/s
  - 需要 GPU 数: 50,000 / 800 = 63 GPUs
  - 考虑模型并行 (2x) 和冗余: 需要 ~150 GPUs

实际部署:
  - 2 个 Worker，每个 4x H100
  - 每个 Worker 处理 ~500 QPS
  - 使用模型并行 (2x) 和流水线并行 (2x)
```

**b) 核心设计**

**1. 连续批处理（Continuous Batching）**
```
优势:
  - 动态添加/移除请求
  - 最大化 GPU 利用率
  - 平衡延迟和吞吐量

实现:
  - 使用 PrefillAdder 管理批处理
  - 动态调整 batch 大小
  - 请求完成后立即移除
```

**2. KV Cache 内存池管理**
```
设计:
  - 固定大小内存池: 72GB per GPU
  - 分页管理: 16 tokens/page
  - 前缀缓存: RadixAttention

分配策略:
  - 按需分配
  - LRU 替换（如果内存不足）
  - 前缀共享
```

**3. 调度策略**
```
策略: LPM (Longest Prefix Match)
  - 优先处理有最长前缀匹配的请求
  - 提高缓存命中率
  - 减少重复计算

Fallback: FCFS (如果前缀匹配率低)
```

**4. 负载均衡**
```
Router 策略: Cache-Aware Routing
  - 维护每个 Worker 的近似 Radix Tree
  - 前缀匹配率高 → 路由到匹配的 Worker
  - 前缀匹配率低 → 路由到负载最轻的 Worker
```

#### 步骤 4: 扩展和优化（10-15 分钟）

**a) 扩展到 10x 规模（10,000 QPS）**
```
方案:
  1. 水平扩展: 增加 Worker 数量（20 个 Worker）
  2. 优化批处理: 增加 batch size
  3. 优化缓存: 提高前缀缓存命中率
  4. 模型量化: 使用 INT8 量化，减少内存占用

瓶颈:
  - Router 可能成为瓶颈 → 使用多个 Router 实例
  - 网络带宽 → 使用更高速的网络
```

**b) 容错和恢复**
```
方案:
  1. 健康检查: Router 定期检查 Worker 健康状态
  2. 故障转移: 自动路由到健康的 Worker
  3. 检查点: 定期保存 KV Cache 状态（可选）
  4. 监控告警: 监控 QPS、延迟、错误率

恢复策略:
  - Worker 故障 → 自动从负载均衡中移除
  - Router 故障 → 使用多个 Router 实例
```

**c) 监控指标**
```
关键指标:
  - QPS: 每秒请求数
  - TTFT: Time To First Token
  - E2E 延迟: End-to-End 延迟
  - 吞吐量: tokens/second
  - GPU 利用率
  - KV Cache 内存使用率
  - 错误率

告警阈值:
  - QPS < 800 → 告警
  - TTFT > 200ms → 告警
  - GPU 利用率 < 80% → 告警
  - 错误率 > 1% → 告警
```

---

## 请求处理流程问题

### Q5: 请详细解释一下请求从到达到最后返回结果的完整流程？

**面试官意图**：考察候选人对请求处理流程的深入理解

**参考答案**：

完整流程分为 6 个阶段：

#### 阶段 1: 请求到达和 Router 分配

```
Client 发送请求
    ↓
Router 接收请求
    ↓
Router 选择 Worker:
  - 检查 Worker 健康状态
  - 计算前缀匹配率（Cache-Aware）
  - 检查负载（Load Balancing）
  - 选择最优 Worker
    ↓
Router 转发请求到选中的 Worker
```

**代码示例**：
```rust
// sgl-router/src/policies/cache_aware.rs

fn select_worker(&self, workers: &[Worker], request_text: &str) -> Option<usize> {
    // 1. 检查负载是否不平衡
    if is_imbalanced(workers) {
        // 负载均衡：选择负载最轻的 Worker
        return find_least_loaded_worker(workers);
    }
    
    // 2. 缓存感知路由
    let (matched_text, matched_worker) = tree.prefix_match(request_text);
    let match_rate = matched_text.len() as f32 / request_text.len() as f32;
    
    if match_rate > cache_threshold {
        // 匹配率高 → 路由到匹配的 Worker
        return find_worker_index(workers, matched_worker);
    } else {
        // 匹配率低 → 路由到缓存空间最大的 Worker
        return find_worker_with_smallest_tree(workers);
    }
}
```

#### 阶段 2: Worker 接收请求并创建 Req 对象

```
Worker API Server 接收请求
    ↓
解析请求:
  - 提取 input_text
  - 提取 sampling_params (max_tokens, temperature, etc.)
  - Tokenization: text → token_ids
    ↓
创建 Req 对象:
  - rid: 请求唯一 ID
  - origin_input_ids: token IDs
  - sampling_params: 采样参数
  - priority: 优先级（如果有）
    ↓
初始化前缀缓存:
  - 检查 Radix Tree 是否有匹配的前缀
  - 记录 prefix_indices
```

**代码示例**：
```python
# python/sglang/srt/managers/schedule_batch.py

class Req:
    def __init__(self, rid, input_text, sampling_params, ...):
        self.rid = rid
        self.origin_input_ids = tokenize(input_text)
        self.sampling_params = sampling_params
        self.output_ids = []
        self.prefix_indices = []
        
    def init_next_round_input(self, tree_cache):
        # 检查前缀缓存
        match_result = tree_cache.match_prefix(self.origin_input_ids)
        self.prefix_indices = match_result.matched_indices
```

#### 阶段 3: 添加到 Waiting Queue

```
Req 对象创建完成
    ↓
添加到 Waiting Queue:
  - scheduler.waiting_queue.append(req)
  - 记录进入队列时间: req.time_stats.wait_queue_entry_time
    ↓
等待调度器处理
```

**代码示例**：
```python
# python/sglang/srt/managers/scheduler.py

def process_input_requests(self, recv_reqs: List[Req]):
    for req in recv_reqs:
        req.init_next_round_input(self.tree_cache)
        self._add_request_to_queue(req, is_retracted=False)

def _add_request_to_queue(self, req: Req, is_retracted: bool):
    req.time_stats.wait_queue_entry_time = time.perf_counter()
    self.waiting_queue.append(req)
```

#### 阶段 4: 调度器批处理决策

```
调度器循环检查:
  - len(waiting_queue) > 0?
  - 资源是否足够?
  - 是否应该创建新 batch?
    ↓
批处理决策:
  - 根据调度策略排序 waiting_queue
  - 选择可以加入 batch 的请求
  - 估算内存需求
  - 决定 batch 大小
    ↓
创建 ScheduleBatch:
  - 从 waiting_queue 移除选中的请求
  - 创建 ScheduleBatch 对象
  - 分配 KV Cache 内存
```

**代码示例**：
```python
# python/sglang/srt/managers/scheduler.py

def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
    if len(self.waiting_queue) == 0:
        return None
    
    adder = PrefillAdder(...)
    
    # 遍历 waiting_queue，选择请求
    for req in self.waiting_queue:
        if not self.has_enough_resources(req):
            break
        
        req.init_next_round_input(self.tree_cache)
        res = adder.add_one_req(req, ...)
        
        if res == AddReqStatus.SUCCESS:
            continue
        else:
            break
    
    # 从 waiting_queue 移除已选中的请求
    selected_reqs = adder.can_run_list
    for req in selected_reqs:
        self.waiting_queue.remove(req)
    
    # 创建 ScheduleBatch
    if len(selected_reqs) > 0:
        new_batch = ScheduleBatch.init_new(selected_reqs, ...)
        return new_batch
```

#### 阶段 5: Prefill 和 Decode 阶段

```
Prefill 阶段:
  - 处理所有输入 tokens
  - 计算注意力（对所有输入 tokens）
  - 生成 KV Cache
  - 输出最后一个 token 的 hidden state
    ↓
Decode 阶段（循环）:
  - 使用 hidden state 生成下一个 token
  - 更新 KV Cache（添加新 token 的 K、V）
  - 检查是否完成（EOS token 或 max_tokens）
  - 如果未完成，继续下一个 decode step
```

**代码示例**：
```python
# Prefill 阶段
def prefill(batch):
    input_ids = batch.input_ids  # [batch_size, seq_len]
    
    # 前向传播
    hidden_states = model.forward(input_ids)
    
    # 生成 KV Cache
    k_cache, v_cache = generate_kv_cache(hidden_states)
    
    # 保存 KV Cache
    batch.kv_cache = (k_cache, v_cache)
    
    return hidden_states[:, -1, :]  # 最后一个 token 的 hidden state

# Decode 阶段
def decode(batch, hidden_state):
    # 生成下一个 token
    next_token_logits = model.lm_head(hidden_state)
    next_token_id = sample(next_token_logits, batch.sampling_params)
    
    # 更新 KV Cache
    new_k, new_v = model.forward_one_token(next_token_id, hidden_state)
    batch.kv_cache.append((new_k, new_v))
    
    # 检查是否完成
    if next_token_id == EOS_TOKEN or len(batch.output_ids) >= max_tokens:
        batch.mark_finished()
    
    return next_token_id
```

#### 阶段 6: 返回结果

```
请求完成:
  - 从 running_batch 移除
  - 释放 KV Cache 内存
  - 更新前缀缓存（Radix Tree）
    ↓
返回结果给 Client:
  - 流式输出: 每个 token 生成后立即返回
  - 非流式输出: 所有 tokens 生成后返回
    ↓
清理:
  - 释放 Req Pool 索引
  - 更新统计信息
```

**代码示例**：
```python
# python/sglang/srt/managers/scheduler.py

def process_batch_result_decode(self, batch, result):
    for req, next_token_id in zip(batch.reqs, result.next_token_ids):
        req.output_ids.append(next_token_id)
        
        if req.finished():
            # 请求完成
            req.time_stats.completion_time = time.perf_counter()
            
            # 更新前缀缓存
            self.tree_cache.cache_finished_req(req)
            
            # 释放资源
            self.token_to_kv_pool_allocator.free(req.kv_cache_indices)
            
            # 返回结果
            self.send_response_to_client(req)
```

---

### Q6: Router 如何做请求分配？为什么需要 Router？

**面试官意图**：考察候选人对 Router 作用的理解

**参考答案**：

#### Router 的分配策略

**1. Cache-Aware Routing（缓存感知路由）**

```
目标: 提高缓存命中率，减少重复计算

流程:
  1. Router 维护每个 Worker 的近似 Radix Tree
  2. 新请求到达时，Router 计算与每个 Worker 的前缀匹配率
  3. 如果匹配率 > cache_threshold (如 0.8):
     → 路由到匹配率最高的 Worker（利用缓存）
  4. 如果匹配率 ≤ cache_threshold:
     → 路由到树大小最小的 Worker（最多可用缓存空间）
```

**代码示例**：
```rust
// sgl-router/src/policies/cache_aware.rs

let (matched_text, matched_worker) = tree.prefix_match(request_text);
let match_rate = matched_text.chars().count() as f32 / request_text.chars().count() as f32;

if match_rate > self.config.cache_threshold {
    // 匹配率高 → 路由到匹配的 Worker
    RouterMetrics::record_cache_hit();
    return find_worker_index(workers, matched_worker);
} else {
    // 匹配率低 → 路由到缓存空间最大的 Worker
    RouterMetrics::record_cache_miss();
    return find_worker_index(workers, tree.get_smallest_tenant());
}
```

**2. Load Balancing（负载均衡）**

```
目标: 平衡各个 Worker 的负载

流程:
  1. Router 跟踪每个 Worker 的负载（pending requests）
  2. 检查系统是否不平衡:
     - (max_load - min_load) > abs_threshold (如 10)
     - max_load > min_load * rel_threshold (如 1.5)
  3. 如果不平衡:
     → 路由到负载最轻的 Worker
  4. 如果平衡:
     → 使用 Cache-Aware Routing
```

**代码示例**：
```rust
let loads: Vec<usize> = workers.iter().map(|w| w.load()).collect();
let max_load = *loads.iter().max().unwrap_or(&0);
let min_load = *loads.iter().min().unwrap_or(&0);

let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
    && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

if is_imbalanced {
    // 使用负载均衡
    let min_load_idx = healthy_indices
        .iter()
        .min_by_key(|&&idx| workers[idx].load())
        .copied()?;
    return Some(min_load_idx);
}
```

#### 为什么需要 Router？

**场景 1: 多个 Worker 部署**

```
没有 Router:
  Client → 随机选择一个 Worker
  问题:
    - 负载可能不均衡（Worker 1: 20 请求，Worker 2: 4 请求）
    - 无法利用缓存（相同前缀的请求可能被分配到不同 Worker）
    - 故障时无法自动转移

有 Router:
  Client → Router → 选择最优 Worker
  优势:
    - 负载均衡（每个 Worker 处理 ~8 请求）
    - 缓存感知路由（相同前缀路由到同一 Worker）
    - 自动故障转移（不健康的 Worker 自动移除）
```

**场景 2: 缓存优化**

```
没有 Router:
  每个 Worker 独立处理请求
  问题:
    - 相同前缀的请求可能被分配到不同 Worker
    - 无法共享缓存
    - 重复计算（浪费 GPU 资源）

有 Router:
  Router 维护全局缓存状态
  优势:
    - 相同前缀的请求路由到同一 Worker
    - 共享缓存，减少计算
    - 提高吞吐量（5x 加速）
```

**场景 3: 可扩展性**

```
Router 的优势:
  1. 集中式决策: Router 拥有全局视角，可以做出最优决策
  2. 避免 Worker 间通信: Worker 不需要知道其他 Worker 的状态
  3. 解耦设计: Router 和 Worker 职责分离，各司其职
  4. 性能优化: Router 可以快速做出路由决策，避免 Worker 的额外开销
```

---

## 性能优化问题

### Q7: 如何平衡延迟和吞吐量？连续批处理是如何工作的？

**面试官意图**：考察候选人对性能优化的理解，这是 LLM 推理系统的核心问题

**参考答案**：

#### 延迟 vs 吞吐量的权衡

**延迟（Latency）**：
- **TTFT**: Time To First Token（用户看到第一个 token 的时间）
- **E2E**: End-to-End 延迟（整个响应时间）
- **目标**: 尽可能低（< 100ms TTFT，< 2s E2E）

**吞吐量（Throughput）**：
- **QPS**: Queries Per Second（每秒请求数）
- **Tokens/s**: 每秒生成的 tokens 数
- **目标**: 尽可能高（最大化 GPU 利用率）

**权衡关系**：
```
更大的 batch size:
  ✅ 更高的吞吐量（GPU 利用率高）
  ❌ 更高的延迟（需要等待更多请求）

更小的 batch size:
  ✅ 更低的延迟（请求可以更快处理）
  ❌ 更低的吞吐量（GPU 利用率低）
```

**解决方案：连续批处理（Continuous Batching）**

#### 连续批处理的工作原理

**传统静态批处理的问题**：

```
时间步 1: [请求 1, 请求 2, 请求 3] → 生成 token 1
时间步 2: [请求 1, 请求 2, 请求 3] → 生成 token 2
时间步 3: [请求 1 完成，但必须等待] [请求 2, 请求 3] → 生成 token 3
时间步 4: [请求 2, 请求 3] → 生成 token 4

问题:
  - 请求 1 在第 2 步就完成了，但要等到第 3 步才能移除
  - GPU 利用率低（处理已完成的请求）
  - 延迟高（等待其他请求完成）
```

**连续批处理的优势**：

```
时间步 1: [请求 1, 请求 2, 请求 3] → 生成 token 1
时间步 2: [请求 1, 请求 2, 请求 3] → 生成 token 2
时间步 3: [请求 1 完成，立即移除] [请求 2, 请求 3, 新请求 4] → 生成 token 3
时间步 4: [请求 2, 请求 3, 请求 4] → 生成 token 4

优势:
  - 请求完成后立即移除
  - 新请求可以立即加入
  - GPU 利用率高（只处理活跃请求）
  - 延迟低（请求不需要等待其他请求）
```

**实现细节**：

```python
# python/sglang/srt/managers/schedule_batch.py

class ScheduleBatch:
    def filter_batch(self):
        """过滤已完成的请求"""
        # 移除已完成的请求
        self.reqs = [req for req in self.reqs if not req.finished()]
        
        # 更新 batch 大小
        self.batch_size = len(self.reqs)
    
    def add_new_requests(self, new_reqs):
        """在 decode 阶段动态添加新请求"""
        # 检查资源是否足够
        if self.has_enough_memory(new_reqs):
            # 添加新请求
            self.reqs.extend(new_reqs)
            
            # 分配 KV Cache 内存
            for req in new_reqs:
                req.kv_cache_indices = self.alloc_kv_cache(req)
```

**动态调整 batch 大小**：

```python
def calculate_optimal_batch_size(self) -> int:
    """根据等待队列长度动态调整 batch 大小"""
    
    waiting_count = len(self.waiting_queue)
    
    if waiting_count > 100:
        # 队列很长 → 增加 batch size（提高吞吐量）
        return self.max_batch_size
    elif waiting_count < 10:
        # 队列很短 → 减小 batch size（降低延迟）
        return min(8, self.max_batch_size)
    else:
        # 中等负载 → 平衡
        return self.max_batch_size // 2
```

**效果**：
- ✅ **延迟优化**：请求完成后立即返回，不需要等待
- ✅ **吞吐量优化**：GPU 始终处理活跃请求，利用率高
- ✅ **动态平衡**：根据负载自动调整 batch 大小

---

### Q8: KV Cache 是如何管理的？如何减少内存占用？

**面试官意图**：考察候选人对内存管理的理解

**参考答案**：

#### KV Cache 的作用

**为什么需要 KV Cache？**

在 Transformer 的自回归生成中：
- 每个新 token 都需要访问所有历史 tokens
- 如果不缓存，需要重新计算所有历史 tokens 的 K、V
- 缓存后，只需要计算新 token 的 K、V

**KV Cache 大小计算**：

```
每个 token 的 KV Cache 大小:
  = 2 × num_kv_heads × head_dim × dtype_size

例如（LLaMA-2 70B，使用 GQA）:
  = 2 × 8 × 128 × 2 bytes
  = 4 KB per token

对于 2048 个 tokens:
  = 4 KB × 2048 = 8 MB per request

对于 1000 个并发请求:
  = 8 MB × 1000 = 8 GB
```

#### KV Cache 管理策略

**1. 内存池管理**

```python
# python/sglang/srt/mem_cache/allocator.py

class TokenToKVPoolAllocator:
    def __init__(self, size, page_size):
        self.size = size  # 总内存池大小（如 72GB）
        self.page_size = page_size  # 每页大小（如 16 tokens）
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

**2. 前缀缓存（Prefix Caching）**

```
原理:
  - 使用 Radix Tree 存储共享前缀
  - 多个请求可以共享相同的前缀
  - 减少内存占用和计算量

示例:
  请求 1: "What is the capital of France?"
  请求 2: "What is the capital of Germany?"
  
  共享前缀: "What is the capital of "
  
  传统方式:
    - 请求 1: 缓存 8 tokens
    - 请求 2: 缓存 8 tokens
    - 总计: 16 tokens
  
  前缀缓存:
    - 共享前缀: 缓存 6 tokens（共享）
    - 请求 1: 缓存 2 tokens（"France?"）
    - 请求 2: 缓存 2 tokens（"Germany?"）
    - 总计: 10 tokens（减少 37.5%）
```

**3. GQA (Grouped-Query Attention)**

```
传统 MHA:
  - num_heads = 64
  - 每个 token: 2 × 64 × head_dim × 2 = 256 × head_dim bytes

GQA (LLaMA-2 70B):
  - num_heads = 64 (Q)
  - num_kv_heads = 8 (K, V)
  - 每个 token: 2 × 8 × head_dim × 2 = 32 × head_dim bytes
  - 减少了 8 倍！

效果:
  - 在相同内存下，可以缓存 8 倍更多的 tokens
  - 或者可以用更少的内存达到相同的容量
```

#### 内存不足时的处理

**策略 1: 撤回请求（Retract）**

```python
# python/sglang/srt/managers/scheduler.py

def update_running_batch(self, batch):
    # 检查内存是否足够
    if not batch.check_decode_mem():
        # 内存不足 → 撤回一些请求
        retracted_reqs, new_token_ratio = batch.retract_decode()
        
        # 将撤回的请求重新加入 waiting_queue
        for req in retracted_reqs:
            self._add_request_to_queue(req, is_retracted=True)
        
        logger.info(
            "KV cache pool is full. Retract requests. "
            f"#retracted_reqs: {len(retracted_reqs)}"
        )
```

**策略 2: LRU 替换**

```
如果内存池满:
  1. 找到最近最少使用的请求
  2. 释放其 KV Cache
  3. 分配给新请求
  4. 如果请求还在运行，需要重新计算（性能损失）
```

**策略 3: 拒绝新请求**

```
如果内存池满且无法撤回:
  1. 拒绝新请求
  2. 返回错误（SERVICE_UNAVAILABLE）
  3. 等待现有请求完成
```

---

## 深入设计问题

### Q9: 如果让你设计一个 KV Cache 管理系统，支持 10,000 并发请求，你会如何设计？

**面试官意图**：考察候选人的深入设计能力，这是一个经典的系统设计题目

**参考答案**：

我会按照以下步骤设计：

#### 步骤 1: 需求澄清

**关键问题**：
1. **并发请求数**: 10,000
2. **平均序列长度**: 假设 200 tokens (100 input + 100 output)
3. **模型**: LLaMA-2 70B (GQA: 8 KV heads)
4. **GPU 内存**: H100 (80GB)
5. **目标**: 最大化内存利用率，最小化内存碎片

#### 步骤 2: 内存需求估算

```
每个 token 的 KV Cache:
  = 2 × 8 × 128 × 2 = 4 KB

每个请求的 KV Cache:
  = 4 KB × 200 = 800 KB

10,000 个请求的总内存:
  = 800 KB × 10,000 = 8 GB

加上模型权重 (140GB) 和其他开销:
  - 模型权重: 140GB (需要模型并行)
  - KV Cache: 8GB
  - 其他: 2GB
  - 总计: 150GB (需要 2x H100)
```

#### 步骤 3: 系统设计

**a) 内存池设计**

```
方案: 分页管理（Paged Memory）

设计:
  - 将内存分成固定大小的页（如 16 tokens/page）
  - 每个请求按页分配
  - 减少内存碎片
  - 支持动态扩展

优势:
  ✅ 减少内存碎片
  ✅ 支持动态扩展
  ✅ 便于内存复用
```

**代码设计**：
```python
class PagedKVCacheAllocator:
    def __init__(self, total_size, page_size=16):
        self.total_size = total_size  # 总内存大小
        self.page_size = page_size    # 每页大小（tokens）
        self.num_pages = total_size // (page_size * 4 * 1024)  # 页数
        
        # 页表：记录每页的使用情况
        self.page_table = [None] * self.num_pages  # None = 空闲
        
        # 请求到页的映射
        self.req_to_pages = {}  # {req_id: [page_ids]}
        
    def alloc(self, req_id, num_tokens):
        """为请求分配内存"""
        num_pages = (num_tokens + self.page_size - 1) // self.page_size
        
        # 找到连续的空闲页
        free_pages = self.find_free_pages(num_pages)
        if len(free_pages) < num_pages:
            return None  # 内存不足
        
        # 分配页
        for page_id in free_pages[:num_pages]:
            self.page_table[page_id] = req_id
        
        self.req_to_pages[req_id] = free_pages[:num_pages]
        return free_pages[:num_pages]
    
    def free(self, req_id):
        """释放请求的内存"""
        if req_id not in self.req_to_pages:
            return
        
        # 释放页
        for page_id in self.req_to_pages[req_id]:
            self.page_table[page_id] = None
        
        del self.req_to_pages[req_id]
```

**b) 前缀缓存设计**

```
方案: Radix Tree + 共享机制

设计:
  - 使用 Radix Tree 存储共享前缀
  - 多个请求共享相同前缀的 KV Cache
  - 减少内存占用

实现:
  1. 插入请求时，检查是否有匹配的前缀
  2. 如果有匹配，共享前缀的 KV Cache
  3. 只存储差异部分
```

**代码设计**：
```python
class RadixTreeCache:
    def __init__(self):
        self.root = RadixNode()
        self.cache = {}  # {prefix: kv_cache}
    
    def insert(self, req_id, tokens, kv_cache):
        """插入请求的 KV Cache"""
        # 查找最长匹配前缀
        matched_prefix, matched_node = self.find_longest_match(tokens)
        
        if matched_prefix:
            # 有匹配的前缀 → 共享
            shared_kv = self.cache[matched_prefix]
            new_kv = kv_cache[len(matched_prefix):]  # 只存储差异部分
            
            # 更新树
            self.root.insert(tokens, req_id)
            
            return shared_kv, new_kv
        else:
            # 没有匹配 → 完整存储
            self.cache[tokens] = kv_cache
            self.root.insert(tokens, req_id)
            return kv_cache, None
    
    def find_longest_match(self, tokens):
        """查找最长匹配前缀"""
        node = self.root
        matched_prefix = []
        
        for token in tokens:
            if token in node.children:
                node = node.children[token]
                matched_prefix.append(token)
            else:
                break
        
        return matched_prefix, node
```

**c) 缓存替换策略**

```
方案: LRU (Least Recently Used) + 优先级

设计:
  - 维护每个请求的最后访问时间
  - 内存不足时，优先替换最近最少使用的请求
  - 考虑请求优先级（高优先级请求不替换）

实现:
  1. 使用 LRU 队列跟踪请求访问顺序
  2. 内存不足时，从队列尾部选择请求替换
  3. 检查请求优先级，高优先级跳过
```

**代码设计**：
```python
class LRUCacheManager:
    def __init__(self, max_size):
        self.max_size = max_size
        self.current_size = 0
        self.lru_queue = deque()  # LRU 队列
        self.req_access_time = {}  # {req_id: last_access_time}
    
    def access(self, req_id):
        """访问请求，更新 LRU"""
        self.req_access_time[req_id] = time.time()
        
        # 移动到队列头部
        if req_id in self.lru_queue:
            self.lru_queue.remove(req_id)
        self.lru_queue.appendleft(req_id)
    
    def evict_if_needed(self, new_req_size, priority_threshold=10):
        """如果需要，驱逐请求"""
        while self.current_size + new_req_size > self.max_size:
            if not self.lru_queue:
                return None  # 无法驱逐
            
            # 从队列尾部选择（最近最少使用）
            evict_req_id = self.lru_queue.pop()
            
            # 检查优先级
            req_priority = self.get_priority(evict_req_id)
            if req_priority > priority_threshold:
                # 高优先级，跳过
                self.lru_queue.appendleft(evict_req_id)
                continue
            
            # 驱逐请求
            evicted_size = self.get_req_size(evict_req_id)
            self.current_size -= evicted_size
            return evict_req_id
        
        return None
```

#### 步骤 4: 优化策略

**a) 内存压缩**

```
方案: 量化 KV Cache

设计:
  - 使用 INT8 量化 KV Cache
  - 减少 50% 内存占用
  - 性能损失 < 1%

实现:
  - 量化 K、V 矩阵
  - 存储量化参数（scale, zero_point）
  - 推理时反量化
```

**b) 分层缓存**

```
方案: 热数据在 GPU，冷数据在 CPU

设计:
  - 活跃请求的 KV Cache 在 GPU
  - 不活跃请求的 KV Cache 在 CPU
  - 需要时再加载到 GPU

实现:
  - 使用 LRU 判断活跃度
  - 定期将冷数据 offload 到 CPU
  - 请求恢复时从 CPU 加载
```

**c) 预分配和复用**

```
方案: 预分配固定大小的内存块

设计:
  - 预分配多个固定大小的内存块
  - 请求使用完后，内存块可以复用
  - 减少分配/释放开销

实现:
  - 维护空闲内存块池
  - 请求使用完后，返回池中
  - 新请求从池中获取
```

---

### Q10: 如果系统出现性能瓶颈，你会如何排查和优化？

**面试官意图**：考察候选人的问题排查和优化能力

**参考答案**：

我会按照以下步骤排查和优化：

#### 步骤 1: 监控关键指标

**关键指标**：

```
1. 延迟指标:
   - TTFT: Time To First Token
   - E2E 延迟: End-to-End 延迟
   - P50/P95/P99 延迟

2. 吞吐量指标:
   - QPS: Queries Per Second
   - Tokens/s: 每秒生成的 tokens 数
   - GPU 利用率

3. 资源指标:
   - GPU 内存使用率
   - KV Cache 内存使用率
   - CPU 使用率
   - 网络带宽

4. 错误指标:
   - 错误率
   - 超时率
   - 内存不足错误
```

#### 步骤 2: 识别瓶颈

**常见瓶颈**：

**a) GPU 利用率低**

```
症状:
  - GPU 利用率 < 80%
  - 吞吐量低
  - 等待队列很长

可能原因:
  1. Batch size 太小
  2. 调度器开销大
  3. 内存不足，无法组成大 batch

排查方法:
  - 检查 batch size 分布
  - 检查调度器 CPU 使用率
  - 检查 KV Cache 内存使用率

优化方案:
  - 增加 batch size
  - 优化调度器（减少检查开销）
  - 增加 KV Cache 内存池大小
```

**b) 延迟高**

```
症状:
  - TTFT > 200ms
  - E2E 延迟 > 2s
  - 用户投诉响应慢

可能原因:
  1. 等待队列太长（请求等待时间久）
  2. Batch size 太大（需要等待更多请求）
  3. Prefill 阶段慢（输入太长）

排查方法:
  - 检查 waiting_queue 长度
  - 检查平均等待时间
  - 检查 prefill 时间

优化方案:
  - 减少 batch size（降低延迟）
  - 增加 Worker 数量（减少等待）
  - 优化 Prefill（使用 FlashAttention）
```

**c) 内存不足**

```
症状:
  - 频繁出现 "KV cache pool is full"
  - 请求被撤回（retract）
  - OOM 错误

可能原因:
  1. KV Cache 内存池太小
  2. 请求序列太长
  3. 并发请求数太多

排查方法:
  - 检查 KV Cache 内存使用率
  - 检查平均序列长度
  - 检查并发请求数

优化方案:
  - 增加 KV Cache 内存池大小
  - 使用前缀缓存（减少内存占用）
  - 使用 GQA（减少 KV Cache 大小）
  - 限制最大序列长度
```

#### 步骤 3: 优化策略

**a) 调度器优化**

```
问题: 调度器 CPU 开销大

优化:
  1. 使用 batch_is_full 标志，跳过不必要的检查
  2. 减少 waiting_queue 遍历次数
  3. 使用更高效的调度策略（LPM）

代码示例:
```python
# 优化前
def get_new_batch_prefill(self):
    if len(self.waiting_queue) == 0:
        return None
    # 每次都检查所有请求...

# 优化后
def get_new_batch_prefill(self):
    if self.running_batch.batch_is_full:
        return None  # 快速返回
    if len(self.waiting_queue) == 0:
        return None
    # 只在需要时检查...
```

**b) 批处理优化**

```
问题: Batch size 不合适

优化:
  1. 动态调整 batch size
  2. 根据等待队列长度调整
  3. 平衡延迟和吞吐量

代码示例:
```python
def calculate_optimal_batch_size(self):
    waiting_count = len(self.waiting_queue)
    
    if waiting_count > 100:
        # 队列长 → 增加 batch size（提高吞吐量）
        return self.max_batch_size
    elif waiting_count < 10:
        # 队列短 → 减小 batch size（降低延迟）
        return min(8, self.max_batch_size)
    else:
        return self.max_batch_size // 2
```

**c) 内存优化**

```
问题: 内存不足

优化:
  1. 使用前缀缓存（RadixAttention）
  2. 使用 GQA（减少 KV Cache 大小）
  3. 使用量化（INT8 量化 KV Cache）

效果:
  - 前缀缓存: 减少 30-50% 内存占用
  - GQA: 减少 8x KV Cache 大小
  - 量化: 减少 50% 内存占用
```

#### 步骤 4: 性能测试

**测试方法**：

```
1. 压力测试:
   - 逐步增加 QPS
   - 观察性能指标变化
   - 找到性能拐点

2. 延迟测试:
   - 测量不同 batch size 下的延迟
   - 找到延迟和吞吐量的平衡点

3. 内存测试:
   - 测试不同并发数下的内存使用
   - 找到内存瓶颈

4. 对比测试:
   - 优化前后对比
   - 验证优化效果
```

---

## 📝 总结

### 面试准备建议

1. **理解基础概念**：
   - 系统设计 vs 算法设计
   - LLM 推理系统的挑战
   - 核心组件的作用

2. **掌握设计流程**：
   - 需求澄清
   - 高层设计
   - 详细设计
   - 扩展和优化

3. **熟悉关键技术**：
   - 连续批处理
   - KV Cache 管理
   - 前缀缓存
   - 负载均衡

4. **准备实际案例**：
   - SGLang 架构
   - vLLM 架构
   - 性能优化经验

5. **练习表达能力**：
   - 清晰表达设计思路
   - 讨论权衡和优化
   - 承认不确定性

**通过系统学习和实践，你将能够自信地回答 LLM 推理系统设计的面试问题！** 🚀

