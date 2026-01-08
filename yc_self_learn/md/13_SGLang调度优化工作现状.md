# SGLang 调度优化工作现状

## 📋 目录

1. [v0.4 发布中的调度优化](#1-v04-发布中的调度优化)
2. [当前正在进行的调度优化](#2-当前正在进行的调度优化)
3. [调度优化技术详解](#3-调度优化技术详解)
4. [调度参数调优](#4-调度参数调优)
5. [未来发展方向](#5-未来发展方向)

---

## 1. v0.4 发布中的调度优化

**发布时间**: 2024年12月

**主要优化**:

### 1.1 Zero-Overhead Batch Scheduler（零开销批处理调度器）

**目标**: 最小化调度开销，提高吞吐量

**实现方式**:
- **`batch_is_full` 优化**: 当批处理已满时，跳过 prefill 检查，减少 CPU 开销
- **高效的批处理调度**: 优化了请求选择和批处理构建逻辑
- **智能的请求优先级管理**: 支持优先级调度和抢占

**代码位置**: `python/sglang/srt/managers/schedule_batch.py:885`

```python
# Tell whether the current running batch is full so that we can skip
# the check of whether to prefill new requests.
# This is an optimization to reduce the overhead of the prefill check.
batch_is_full: bool = False
```

**效果**: 减少了调度器的 CPU 开销，提高了整体吞吐量

---

### 1.2 Cache-Aware Load Balancer（缓存感知负载均衡器）

**目标**: 在多个 worker 之间智能路由请求，最大化缓存命中率

**实现位置**: `sgl-router/src/policies/cache_aware.rs`

**核心策略**:

#### a) **Cache-Aware Routing（缓存感知路由）**
- 为每个 worker 维护一个**近似 Radix Tree**
- 基于请求历史，无需直接查询缓存状态
- 存储原始文本字符（而非 token IDs），避免 tokenization 开销

**路由逻辑**:
```rust
// 1. 找到前缀匹配率最高的 worker
let match_rate = find_highest_prefix_match(request, workers);

if match_rate > cache_threshold {
    // 2. 如果匹配率足够高，路由到该 worker（利用缓存）
    route_to_worker_with_highest_match();
} else {
    // 3. 否则，路由到树大小最小的 worker（最多可用缓存空间）
    route_to_worker_with_smallest_tree();
}
```

#### b) **Load Balancing（负载均衡）**
- 跟踪每个 worker 的待处理请求数
- 当系统不平衡时，路由到最不繁忙的 worker

**不平衡检测**:
```rust
let is_imbalanced = (max_load - min_load) > abs_threshold
    && max_load > min_load * rel_threshold;

if is_imbalanced {
    // 使用最短队列策略
    route_to_worker_with_min_load();
} else {
    // 使用缓存感知路由
    route_based_on_cache();
}
```

**配置参数**:
- `cache_threshold`: 最小前缀匹配率（0.0-1.0）
- `balance_abs_threshold`: 绝对负载差异阈值
- `balance_rel_threshold`: 相对负载比率阈值
- `eviction_interval_secs`: LRU 驱逐间隔
- `max_tree_size`: 每个树的最大节点数

**效果**: 
- ✅ 提高缓存命中率
- ✅ 平衡负载分布
- ✅ 减少内存占用（LRU 驱逐）

---

## 2. 当前正在进行的调度优化

### 2.1 Schedule Conservativeness（调度保守性）

**参数**: `--schedule-conservativeness`

**默认值**: `1.0`

**作用**: 控制调度器接受新请求的保守程度

**代码位置**: `python/sglang/srt/managers/scheduler.py:568`

```python
self.init_new_token_ratio = min(
    global_config.default_init_new_token_ratio
    * server_args.schedule_conservativeness,
    1.0,
)
```

**工作原理**:
- **值越大**: 调度器越保守，预留更多内存空间
- **值越小**: 调度器越激进，接受更多请求

**使用场景**:

**场景 1: Token Usage 过低**
```
现象: token usage < 0.9 且 #queue-req > 0
原因: 调度器太保守，不敢接受新请求
解决: 降低 --schedule-conservativeness 到 0.3
```

**场景 2: 频繁出现 KV Cache 满**
```
现象: 频繁看到 "KV cache pool is full. Retract requests."
原因: 调度器太激进，接受了太多请求
解决: 提高 --schedule-conservativeness 到 1.3
```

**KV Cache 管理与 GQA 的关系：**

SGLang 的 KV Cache 管理是在**运行时层面**，与 GQA 的**模型架构层面**优化是互补的：

**1. 模型架构层面（GQA）**：
```
GQA 通过减少 KV 头数来减少每个 token 的 KV Cache 大小：

每个 token 的 KV Cache 大小 = 2 × num_kv_heads × head_dim × dtype_size

传统 MHA (LLaMA-1):
  - num_kv_heads = 64
  - 每个 token: 2 × 64 × head_dim × 2 bytes = 256 × head_dim bytes

GQA (LLaMA-2 70B):
  - num_kv_heads = 8 (num_groups)
  - 每个 token: 2 × 8 × head_dim × 2 bytes = 32 × head_dim bytes
  - 减少了 8 倍！
```

**2. 运行时层面（SGLang 调度）**：
```
SGLang 管理一个 KV Cache 内存池（KV cache pool）：

内存池大小 = 总 GPU 内存 × mem_fraction_static
  - 例如: 80GB GPU × 0.9 = 72GB 用于 KV Cache

每个请求的内存需求：
  - 需要为所有历史 token 分配 KV Cache 空间
  - 内存需求 = 每个 token 的 KV Cache 大小 × 序列长度
  - 序列长度 = input_tokens + output_tokens

调度器的工作：
  1. 估算每个请求的内存需求
  2. 检查内存池是否有足够空间
  3. 如果空间不足，撤回（retract）一些请求
```

**3. 两者的关系**：
```
GQA 的影响：
  ✅ 减少了每个 token 的 KV Cache 大小（8 倍减少）
  ✅ 在相同内存池下，可以缓存更多 token
  ✅ 可以同时处理更多请求或更长序列

SGLang 调度的作用：
  ✅ 即使有 GQA，如果请求太多或序列太长，内存池仍可能满
  ✅ 调度器需要智能地管理内存池，决定接受多少请求
  ✅ 通过 --schedule-conservativeness 控制内存预留策略

示例计算（LLaMA-2 70B）：
  假设:
    - GPU 内存: 80GB
    - KV Cache 池: 72GB (90%)
    - head_dim: 128
    - 每个 token KV Cache: 2 × 8 × 128 × 2 = 4KB
  
  可缓存的 token 数:
    72GB / 4KB = 18,000,000 tokens
  
  如果平均每个请求 2000 tokens:
    可同时处理: 18,000,000 / 2000 = 9,000 个请求
  
  但如果调度器太激进，接受了 10,000 个请求:
    → KV cache pool is full!
    → 需要撤回一些请求
```

**4. 与 Head 的关系**：
```
KV Cache 内存池的大小直接依赖于 head 数量：

代码实现 (python/sglang/srt/mem_cache/memory_pool.py:402):
  class MHATokenToKVPool:
      def __init__(self, head_num, head_dim, layer_num, ...):
          # [size, head_num, head_dim] for each layer
          self.k_buffer = [
              torch.zeros((size, head_num, head_dim), ...)
              for _ in range(layer_num)
          ]
          self.v_buffer = [
              torch.zeros((size, head_num, head_dim), ...)
              for _ in range(layer_num)
          ]

内存大小计算:
  总内存 = 2 × size × head_num × head_dim × layer_num × dtype_size
  
  其中:
    - head_num: 在 GQA 中，这是 num_kv_heads（如 8）
    - size: 内存池可容纳的 token 数
    - layer_num: Transformer 层数（如 80）

GQA 的收益:
  - 如果 head_num 从 64 减少到 8
  - 内存池可以容纳 8 倍更多的 token
  - 或者可以用更少的内存达到相同的容量
```

**总结**：
- ✅ **GQA（架构层）**：减少每个 token 的 KV Cache 大小（通过减少 KV 头数）
- ✅ **SGLang 调度（运行时层）**：管理 KV Cache 内存池，决定接受多少请求
- ✅ **两者互补**：GQA 让内存更高效，调度器让内存使用更智能
- ✅ **都与 head 相关**：KV Cache 大小 = `2 × num_kv_heads × head_dim × seq_len`

**调优建议**:
- 如果 `token usage < 0.9` 且 `#queue-req > 0` → 降低到 `0.3`
- 如果频繁出现 `KV cache pool is full` → 提高到 `1.3`
- 偶尔出现（~1次/分钟）是正常的

---

### 2.2 Schedule Policy（调度策略）

**参数**: `--schedule-policy`

**可选值**:
- `fcfs`: First Come First Served（先来先服务，默认）
- `lpm`: Longest Prefix Match（最长前缀匹配）
- `dfs`: Depth-First Search（深度优先搜索）
- `priority`: 优先级调度

**代码位置**: `python/sglang/srt/managers/schedule_policy.py:66`

```python
class CacheAwarePolicy(str, Enum):
    FCFS = "fcfs"  # first come first served
    LPM = "lpm"    # longest prefix match
    DFS = "dfs"    # depth-first search
    PRIORITY = "priority"
```

#### a) **LPM (Longest Prefix Match)**

**目标**: 重新排序请求，鼓励更多缓存命中

**工作原理**:
```python
def _sort_by_longest_prefix(waiting_queue, tree_cache):
    """按最长前缀匹配排序"""
    for req in waiting_queue:
        prefix_len = len(req.prefix_indices)  # 匹配的前缀长度
        # 按 prefix_len 降序排序
    return sorted(waiting_queue, key=lambda r: -len(r.prefix_indices))
```

**优势**:
- ✅ 提高 RadixAttention 缓存命中率
- ✅ 减少内存占用
- ✅ 提高吞吐量

**劣势**:
- ❌ 引入更多调度开销
- ❌ 可能增加某些请求的等待时间

**使用场景**: 当工作负载有很多共享前缀时（如多轮对话、批量处理相似请求）

**文档建议**: `docs/advanced_features/hyperparameter_tuning.md:77`
```markdown
If the workload has many shared prefixes, try `--schedule-policy lpm`.
Here, `lpm` stands for longest prefix match. It reorders requests to encourage
more cache hits but introduces more scheduling overhead.
```

---

### 2.3 In-Batch Prefix Caching（批内前缀缓存）

**目标**: 在同一个 batch 内利用前缀缓存

**代码位置**: `python/sglang/srt/managers/schedule_policy.py:187`

```python
# NOTE(sang): This logic is for in-batch prefix caching;
# If there are more than 1 request that have small matching prefix from
# existing cache, but all those requests share the same prefix, we prefer
# to schedule only one of them so that we can increase the cache hit rate.
if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
    in_batch_matching_prefixes, _, _, _ = (
        self.waiting_queue_radix_tree.match_prefix(...)
    )
    if len(in_batch_matching_prefixes) >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD:
        temporary_deprioritized.add(r.rid)
```

**工作原理**:
1. 检查请求的前缀匹配长度
2. 如果匹配长度很小（≤ `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD`）
3. 检查是否有其他请求共享相同前缀
4. 如果有多个请求共享前缀，只调度其中一个
5. 其他请求在后续 batch 中调度，可以利用批内缓存

**效果**: 
- ✅ 提高批内缓存命中率
- ✅ 减少内存占用
- ✅ 提高吞吐量

---

### 2.4 Priority Scheduling（优先级调度）

**功能**: 支持请求优先级和抢占

**代码位置**: `python/sglang/srt/managers/schedule_policy.py:641`

```python
def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
    """
    Preempt running requests to serve the new request if the priority
    threshold is met and token count sum is verified.
    """
    # 1. 按优先级排序运行中的请求
    sorted_running_reqs = sorted(
        self.running_batch.reqs,
        key=lambda x: (x.priority, -x.time_stats.wait_queue_entry_time),
    )
    
    # 2. 找到可以抢占的请求
    preemptible_reqs = []
    for running_req in sorted_running_reqs:
        if can_preempt(running_req, req):
            preemptible_reqs.append(running_req)
    
    # 3. 抢占并释放内存
    if preemptible_reqs:
        preempt_and_release_memory(preemptible_reqs)
        return True
    
    return False
```

**配置参数**:
- `--enable-priority-scheduling`: 启用优先级调度
- `--schedule-low-priority-values-first`: 低优先级值优先
- `--priority-scheduling-preemption-threshold`: 抢占阈值

**效果**:
- ✅ 高优先级请求可以抢占低优先级请求
- ✅ 提高重要请求的响应速度
- ✅ 支持 QoS（服务质量）保证

---

## 3. 调度优化技术详解

### 3.1 batch_is_full 优化

**问题**: 每次循环都检查是否可以添加新的 prefill 请求，开销较大

**解决方案**: 使用 `batch_is_full` 标志，跳过不必要的检查

**代码位置**: `python/sglang/srt/managers/scheduler.py:1820`

```python
def get_new_batch_prefill(self):
    # 检查前置条件
    if (
        self.running_batch.batch_is_full or len(self.waiting_queue) == 0
    ) and self.chunked_req is None:
        return None  # 快速返回，跳过后续检查
```

**优化效果**:
- ✅ 减少 CPU 开销
- ✅ 提高调度器吞吐量
- ✅ 降低延迟

---

### 3.2 内存预算优化

**问题**: 如何准确估算内存需求，避免 OOM 或内存浪费

**解决方案**: 使用 `new_token_ratio` 和 `schedule_conservativeness`

**代码位置**: `python/sglang/srt/managers/scheduler.py:568`

```python
self.init_new_token_ratio = min(
    global_config.default_init_new_token_ratio
    * server_args.schedule_conservativeness,
    1.0,
)
```

**工作原理**:
- **new_token_ratio**: 估算每个请求的平均输出 token 数
- **schedule_conservativeness**: 调整保守程度
- **内存预算**: `total_tokens = input_tokens + max_new_tokens * new_token_ratio`

**优化效果**:
- ✅ 更准确的内存估算
- ✅ 减少请求被撤回（retract）的情况
- ✅ 提高内存利用率

---

### 3.3 Chunked Prefill 优化

**问题**: 长序列请求占用大量内存，可能导致 OOM

**解决方案**: 分块处理长序列请求

**代码位置**: `python/sglang/srt/managers/schedule_policy.py:609`

```python
if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
    # 非 Chunked Prefill：一次性处理
    self.can_run_list.append(req)
else:
    # Chunked Prefill：分块处理
    trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
    req.extend_input_len = trunc_len
    self.new_chunked_req = req  # 标记为 chunked
```

**优化效果**:
- ✅ 支持更长的序列
- ✅ 减少内存占用
- ✅ 避免 OOM 错误

**调优参数**: `--chunked-prefill-size`
- 默认值: 根据模型和 GPU 自动设置
- 如果 OOM，可以降低到 `4096` 或 `2048`

---

## 4. 调度参数调优

### 4.1 关键参数总结

| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|---------|
| `--schedule-conservativeness` | `1.0` | 调度保守性 | token usage < 0.9 → 0.3<br>频繁 OOM → 1.3 |
| `--schedule-policy` | `fcfs` | 调度策略 | 有共享前缀 → `lpm` |
| `--chunked-prefill-size` | 自动 | Chunked Prefill 大小 | OOM → 4096/2048 |
| `--max-running-requests` | 自动 | 最大运行请求数 | OOM → 降低 |
| `--mem-fraction-static` | 自动 | 静态内存比例 | 高并发 → 提高<br>OOM → 降低 |

---

### 4.2 调优流程

**步骤 1: 监控关键指标**

```bash
# 查看日志中的关键指标
Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, 
cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317
```

**关键指标**:
- `#queue-req`: 等待队列中的请求数（健康范围: 100-2000）
- `token usage`: KV Cache 内存利用率（目标: > 0.9）
- `gen throughput`: 生成吞吐量（token/s）

**步骤 2: 诊断问题**

**问题 1: Token Usage 过低**
```
现象: token usage < 0.9 且 #queue-req > 0
原因: 调度器太保守
解决: --schedule-conservativeness 0.3
```

**问题 2: 频繁 OOM**
```
现象: 频繁看到 "KV cache pool is full. Retract requests."
原因: 调度器太激进或内存不足
解决: 
  - --schedule-conservativeness 1.3
  - --chunked-prefill-size 4096
  - --max-running-requests 降低
  - --mem-fraction-static 降低
```

**问题 3: 吞吐量低**
```
现象: gen throughput 较低
原因: 批处理大小太小或调度开销大
解决:
  - 增加 #queue-req（提高请求提交速度）
  - --schedule-policy lpm（如果有共享前缀）
  - --mem-fraction-static 提高（增加并发）
```

**步骤 3: 迭代优化**

1. 调整参数
2. 运行一段时间
3. 观察指标变化
4. 重复步骤 1-3

---

## 5. 未来发展方向

### 5.1 已知的优化方向

根据代码和文档，SGLang 团队可能正在或计划进行以下优化：

#### a) **更智能的调度策略**
- 基于请求特征的动态调度策略选择
- 机器学习辅助的调度决策
- 自适应参数调整

#### b) **更好的缓存管理**
- 更智能的缓存驱逐策略
- 跨 worker 的缓存共享
- 分层缓存（Hierarchical Cache）优化

#### c) **更高效的批处理**
- 动态批处理大小调整
- 更智能的请求分组
- 批处理形状优化

#### d) **更好的资源管理**
- 更准确的内存预测
- 动态资源分配
- 多租户资源隔离

---

### 5.2 社区贡献方向

如果你想为 SGLang 的调度优化做贡献，可以考虑以下方向：

1. **性能分析**: 分析调度器的瓶颈，找出优化点
2. **新调度策略**: 实现新的调度策略（如基于 QoS 的调度）
3. **参数自动调优**: 实现自动参数调优工具
4. **监控和可视化**: 改进调度器的监控和可视化工具
5. **文档改进**: 改进调度优化的文档和示例

---

## 6. 总结

### 6.1 当前调度优化状态

✅ **已完成**:
- Zero-Overhead Batch Scheduler（v0.4）
- Cache-Aware Load Balancer（v0.4）
- Schedule Conservativeness 参数
- 多种调度策略（FCFS, LPM, DFS, Priority）
- In-Batch Prefix Caching
- Priority Scheduling 和抢占

🔄 **持续优化**:
- 调度参数调优
- 性能监控和改进
- 文档和示例完善

📋 **未来方向**:
- 更智能的调度策略
- 更好的缓存管理
- 更高效的批处理
- 更好的资源管理

---

### 6.2 关键代码位置

| 优化 | 代码位置 |
|------|---------|
| Zero-Overhead Scheduler | `schedule_batch.py:885` |
| Cache-Aware Load Balancer | `sgl-router/src/policies/cache_aware.rs` |
| Schedule Conservativeness | `scheduler.py:568` |
| Schedule Policy | `schedule_policy.py:66` |
| In-Batch Prefix Caching | `schedule_policy.py:187` |
| Priority Scheduling | `schedule_policy.py:641` |

---

### 6.3 参考资料

- [v0.4 Release Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [Hyperparameter Tuning Guide](docs/advanced_features/hyperparameter_tuning.md)
- [Router Documentation](docs/advanced_features/router.md)
- [GitHub Roadmap](https://github.com/sgl-project/sglang/issues/7736)

---

**结论**: SGLang **确实在做调度优化**，并且已经取得了显著成果。v0.4 版本引入了 Zero-Overhead Batch Scheduler 和 Cache-Aware Load Balancer，同时持续优化调度参数和策略。未来还会继续改进调度器的性能和功能。🎯

