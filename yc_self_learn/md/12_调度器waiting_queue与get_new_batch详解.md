# 调度器 waiting_queue 与 get_new_batch 详解

## 📋 目录

1. [waiting_queue 的具体内容](#1-waiting_queue-的具体内容)
2. [get_next_batch_to_run() 主流程](#2-get_next_batch_to_run-主流程)
3. [get_new_batch_prefill() 详细实现](#3-get_new_batch_prefill-详细实现)
4. [PrefillAdder 核心逻辑](#4-prefilladder-核心逻辑)
5. [调度决策示例](#5-调度决策示例)

---

## 1. waiting_queue 的具体内容

### 1.1 数据结构

**代码位置**: `python/sglang/srt/managers/scheduler.py:509`

```python
class Scheduler:
    def __init__(self, ...):
        # 等待队列：存储等待被调度的请求
        self.waiting_queue: List[Req] = []
```

**类型**: `List[Req]` - Python 列表，存储 `Req` 对象

---

### 1.2 Req 对象的结构

**代码位置**: `python/sglang/srt/managers/schedule_batch.py:432`

```python
class Req:
    """The input and output status of a request."""
    
    def __init__(
        self,
        rid: str,                          # 请求 ID
        origin_input_text: str,            # 原始输入文本
        origin_input_ids: List[int],       # Token IDs
        sampling_params: SamplingParams,   # 采样参数
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        lora_id: Optional[str] = None,     # LoRA 适配器 ID
        priority: Optional[int] = None,    # 优先级
        ...
    ):
        # 输入和输出信息
        self.rid = rid                     # 请求唯一标识
        self.origin_input_text = origin_input_text
        self.origin_input_ids = origin_input_ids  # [101, 202, 303, ...]
        self.output_ids = []               # 生成的 token IDs
        self.fill_ids = []                 # fill_ids = origin_input_ids + output_ids
        
        # 采样参数
        self.sampling_params = sampling_params
        # sampling_params 包含:
        #   - temperature: float
        #   - max_new_tokens: int
        #   - top_p: float
        #   - stop: List[str]
        #   - ...
        
        # 内存池信息
        self.req_pool_idx: Optional[int] = None  # 在内存池中的索引
        
        # 完成状态
        self.finished_reason = None
        self.finished_output = None
        self.to_abort = False
        
        # 优先级
        self.priority = priority
        
        # LoRA
        self.lora_id = lora_id
        
        # 前缀缓存相关
        self.prefix_indices = []           # 缓存的 token 索引
        self.last_node = None              # Radix Tree 节点
        self.last_host_node = None
        self.host_hit_length = 0          # 前缀匹配长度
        
        # 时间统计
        self.time_stats = TimeStats()
        self.time_stats.wait_queue_entry_time = 0.0  # 进入队列的时间
        
        # 其他字段...
```

**waiting_queue 中的 Req 对象包含**:
- ✅ **请求标识**: `rid`（唯一 ID）
- ✅ **输入数据**: `origin_input_ids`（token IDs）
- ✅ **采样参数**: `sampling_params`（temperature, max_tokens 等）
- ✅ **优先级**: `priority`（用于优先级调度）
- ✅ **LoRA ID**: `lora_id`（如果使用 LoRA）
- ✅ **前缀缓存信息**: `prefix_indices`, `last_node`（RadixAttention 相关）
- ✅ **时间统计**: `time_stats`（记录等待时间等）

**示例**:
```python
waiting_queue = [
    Req(
        rid="req-001",
        origin_input_ids=[101, 202, 303],
        sampling_params=SamplingParams(temperature=0.7, max_new_tokens=100),
        priority=1,
        lora_id=None,
        ...
    ),
    Req(
        rid="req-002",
        origin_input_ids=[404, 505],
        sampling_params=SamplingParams(temperature=0.9, max_new_tokens=50),
        priority=2,
        lora_id="lora-123",
        ...
    ),
    ...
]
```

---

## 2. get_next_batch_to_run() 主流程

**代码位置**: `python/sglang/srt/managers/scheduler.py:1739`

这是调度器的**核心入口函数**，决定下一个要运行的 batch。

### 2.1 函数签名和返回值

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    """
    获取下一个要运行的 batch
    
    返回:
        ScheduleBatch: 要运行的 batch（Prefill 或 Decode）
        None: 没有可运行的 batch
    """
```

**返回值**: `Optional[ScheduleBatch]`
- `ScheduleBatch`: 包含要运行的请求列表和相关信息
- `None`: 没有可运行的请求（GPU 空闲或队列为空）

---

### 2.2 完整实现流程

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    """获取下一个要运行的 batch"""
    
    # ========== 步骤 1: 处理 Chunked Request ==========
    chunked_req_to_exclude = set()
    if self.chunked_req:
        # 如果有未完成的 chunked request，先处理它
        chunked_req_to_exclude.add(self.chunked_req)
        self.tree_cache.cache_unfinished_req(self.chunked_req, chunked=True)
        # 释放内存
        self.req_to_token_pool.free(self.chunked_req.req_pool_idx)
    
    # ========== 步骤 2: 合并 Prefill Batch 到 Running Batch ==========
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        # 过滤已完成的请求
        self.last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
        
        # 如果不是 prefill-only，合并到 running_batch
        if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
            if self.running_batch.is_empty():
                self.running_batch = self.last_batch
            else:
                # 合并两个 batch
                self.running_batch.merge_batch(self.last_batch)
    
    # ========== 步骤 3: 获取新的 Prefill Batch ==========
    new_batch = self.get_new_batch_prefill()  # ← 核心函数！
    
    # ========== 步骤 4: 处理 DP Attention ==========
    need_dp_attn_preparation = require_mlp_sync(self.server_args)
    if need_dp_attn_preparation and not self.spec_algorithm.is_none():
        new_batch = self.prepare_mlp_sync_batch(new_batch)
        need_dp_attn_preparation = new_batch is None
    
    # ========== 步骤 5: 决定运行哪个 Batch ==========
    if new_batch is not None:
        # 优先运行 Prefill（新请求）
        ret = new_batch
    else:
        # 没有新的 Prefill，运行 Decode（继续生成）
        if not self.running_batch.is_empty():
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None
        else:
            ret = None  # 没有可运行的请求
    
    # ========== 步骤 6: 处理 DP Attention ==========
    if need_dp_attn_preparation:
        ret = self.prepare_mlp_sync_batch(ret)
    
    return ret
```

**关键决策逻辑**:
```
有新的 Prefill Batch?
    ├─ 是 → 运行 Prefill Batch（优先处理新请求）
    └─ 否 → 运行 Decode Batch（继续生成已开始的请求）
```

---

## 3. get_new_batch_prefill() 详细实现

**代码位置**: `python/sglang/srt/managers/scheduler.py:1810`

这是**最核心的调度函数**，从 `waiting_queue` 中选择请求组成新的 Prefill Batch。

### 3.1 函数签名

```python
def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
    """
    从 waiting_queue 中选择请求，组成新的 Prefill Batch
    
    返回:
        ScheduleBatch: 新的 Prefill Batch
        None: 无法创建新的 Batch（队列为空、内存不足等）
    """
```

---

### 3.2 完整实现（逐行解释）

```python
def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
    # ========== 步骤 1: 处理 Grammar Queue ==========
    if self.grammar_queue:
        # 将 grammar 准备好的请求移到 waiting_queue
        self.move_ready_grammar_requests()
    
    # ========== 步骤 2: 检查是否可以运行 Prefill ==========
    if self.try_preemption:
        # 如果启用抢占，重置 batch_is_full 标志
        self.running_batch.batch_is_full = False
    
    # 检查前置条件
    if (
        self.running_batch.batch_is_full or len(self.waiting_queue) == 0
    ) and self.chunked_req is None:
        return None  # 批处理已满或队列为空，无法创建新 batch
    
    # ========== 步骤 3: 检查可分配的请求数量 ==========
    running_bs = len(self.running_batch.reqs)
    
    if (
        self.get_num_allocatable_reqs(running_bs) <= 0
        and not self.chunked_req
        and not self.try_preemption
    ):
        # 没有可分配的请求槽位
        self.running_batch.batch_is_full = True
        return None
    
    # ========== 步骤 4: 检查 Hierarchical Cache ==========
    if self.enable_hierarchical_cache:
        self.tree_cache.check_hicache_events()
    
    # ========== 步骤 5: 计算优先级 ==========
    self.policy.calc_priority(self.waiting_queue)
    # 根据优先级对 waiting_queue 排序（如果需要）
    
    # ========== 步骤 6: 创建 PrefillAdder ==========
    adder = PrefillAdder(
        page_size=self.page_size,
        tree_cache=self.tree_cache,
        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        running_batch=self.running_batch,
        new_token_ratio=self.new_token_ratio,
        rem_input_tokens=self.max_prefill_tokens,
        rem_chunk_tokens=self.chunked_prefill_size,
        mixed_with_decode_tokens=running_bs if self.is_mixed_chunk else 0,
        priority_scheduling_preemption_threshold=self.priority_scheduling_preemption_threshold,
    )
    # PrefillAdder 负责：
    # - 检查内存是否足够
    # - 检查是否可以添加请求
    # - 管理 can_run_list（可以运行的请求列表）
    
    # ========== 步骤 7: 处理 Chunked Request ==========
    if self.chunked_req is not None:
        # 如果有未完成的 chunked request，先添加它
        self.chunked_req.init_next_round_input()
        self.chunked_req = adder.add_chunked_req(self.chunked_req)
    
    # ========== 步骤 8: 遍历 waiting_queue，尝试添加请求 ==========
    if self.enable_lora:
        lora_set = set([req.lora_id for req in self.running_batch.reqs])
    
    for req in self.waiting_queue:
        # ---------- 8.1 检查 LoRA 兼容性 ----------
        if self.enable_lora and not self.tp_worker.can_run_lora_batch(
            lora_set
            | set([req.lora_id for req in adder.can_run_list])
            | set([req.lora_id])
        ):
            # LoRA 不兼容，停止添加
            self.running_batch.batch_is_full = True
            break
        
        # ---------- 8.2 检查批处理大小限制 ----------
        running_bs = len(self.running_batch.reqs) - len(adder.preempt_list)
        if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
            self.running_batch.batch_is_full = True
        
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Prefill 模式下，还需要检查内存池大小
            if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                self.running_batch.batch_is_full = True
        
        # ---------- 8.3 如果批处理已满，尝试抢占 ----------
        if self.running_batch.batch_is_full:
            if not self.try_preemption:
                break  # 不允许抢占，停止添加
            if not adder.preempt_to_schedule(req, self.server_args):
                break  # 抢占失败，停止添加
        
        # ---------- 8.4 检查 HiCache Prefetch ----------
        if self.enable_hicache_storage:
            prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
            if not prefetch_done:
                # Prefetch 未完成，跳过这个请求
                continue
        
        # ---------- 8.5 初始化请求的下一次输入 ----------
        req.init_next_round_input(self.tree_cache)
        # 这会：
        # - 匹配前缀缓存（RadixAttention）
        # - 设置 prefix_indices（缓存的 token 索引）
        # - 设置 extend_input_len（需要处理的新 token 数量）
        
        # ---------- 8.6 尝试添加请求到 Batch ----------
        res = adder.add_one_req(
            req,
            has_chunked_req=(self.chunked_req is not None),
            truncation_align_size=self.truncation_align_size,
        )
        # res 可能是：
        # - AddReqResult.CONTINUE: 成功添加，继续添加下一个
        # - AddReqResult.NO_TOKEN: 内存不足，停止添加
        # - AddReqResult.OTHER: 其他原因，停止添加
        
        # ---------- 8.7 处理添加结果 ----------
        if res != AddReqResult.CONTINUE:
            if res == AddReqResult.NO_TOKEN:
                # 内存不足
                if self.enable_hierarchical_cache:
                    self.running_batch.batch_is_full = len(
                        adder.can_run_list
                    ) > 0 or (not self.running_batch.is_empty())
                else:
                    self.running_batch.batch_is_full = True
            break  # 停止添加请求
    
    # ========== 步骤 9: 获取可以运行的请求列表 ==========
    can_run_list: List[Req] = adder.can_run_list
    
    if len(can_run_list) == 0:
        return None  # 没有可以运行的请求
    
    # ========== 步骤 10: 记录等待时间 ==========
    if self.enable_metrics:
        for req in can_run_list:
            req.add_latency(RequestStage.PREFILL_WAITING)
            # 计算: time.perf_counter() - req.time_stats.wait_queue_entry_time
    
    # ========== 步骤 11: 更新 waiting_queue ==========
    self.waiting_queue = [
        x for x in self.waiting_queue if x not in set(can_run_list)
    ]
    # 从 waiting_queue 中移除已选中的请求
    
    # ========== 步骤 12: 处理被抢占的请求 ==========
    if adder.preempt_list:
        # 被抢占的请求重新加入 waiting_queue
        for req in adder.preempt_list:
            self._add_request_to_queue(req)
    
    # ========== 步骤 13: 处理新的 Chunked Request ==========
    if adder.new_chunked_req is not None:
        assert self.chunked_req is None
        self.chunked_req = adder.new_chunked_req
    
    if self.chunked_req:
        self.chunked_req.is_chunked += 1
    
    # ========== 步骤 14: 打印统计信息 ==========
    if self.current_scheduler_metrics_enabled():
        self.log_prefill_stats(adder, can_run_list, running_bs, 0)
    
    # ========== 步骤 15: 记录 Forward Entry Time ==========
    for req in can_run_list:
        if req.time_stats.forward_entry_time == 0:
            req.time_stats.forward_entry_time = time.perf_counter()
            if self.enable_metrics:
                self.metrics_collector.observe_queue_time(
                    req.time_stats.get_queueing_time(),
                )
    
    # ========== 步骤 16: 创建新的 ScheduleBatch ==========
    new_batch = ScheduleBatch.init_new(
        can_run_list,                      # 选中的请求列表
        self.req_to_token_pool,
        self.token_to_kv_pool_allocator,
        self.tree_cache,
        self.model_config,
        self.enable_overlap,
    )
    
    return new_batch
```

---

## 4. PrefillAdder 核心逻辑

**代码位置**: `python/sglang/srt/managers/schedule_policy.py:315`

`PrefillAdder` 负责**决定哪些请求可以添加到 batch**，管理内存预算。

### 4.1 PrefillAdder 的初始化

```python
class PrefillAdder:
    def __init__(
        self,
        page_size: int,                              # KV Cache 页大小
        tree_cache: BasePrefixCache,                 # 前缀缓存（RadixAttention）
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,  # KV Cache 分配器
        running_batch: ScheduleBatch,               # 当前正在运行的 batch
        new_token_ratio: float,                     # 新 token 比例（用于估算内存）
        rem_input_tokens: int,                       # 剩余的输入 token 预算
        rem_chunk_tokens: Optional[int],            # 剩余的 chunk token 预算
        mixed_with_decode_tokens: int = 0,          # 与 decode 混合的 token 数
        priority_scheduling_preemption_threshold: int = 0,
    ):
        self.page_size = page_size
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        
        # 内存预算
        self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        
        # 结果列表
        self.can_run_list = []          # 可以运行的请求列表
        self.preempt_list = []          # 被抢占的请求列表
        self.new_chunked_req = None     # 新的 chunked request
        
        # 统计信息
        self.log_hit_tokens = 0         # 命中缓存的 token 数
        self.log_input_tokens = 0       # 输入的 token 数
```

**关键属性**:
- `can_run_list`: 可以运行的请求列表（会被添加到 batch）
- `preempt_list`: 被抢占的请求列表（需要重新调度）
- `rem_total_tokens`: 剩余的总 token 预算（内存限制）
- `rem_input_tokens`: 剩余的输入 token 预算（Prefill 限制）

---

### 4.2 内存预算计算

```python
@property
def rem_total_tokens(self):
    """计算剩余的总 token 预算"""
    if self.is_hybrid:
        # Hybrid 模式（Full + SWA）
        available_and_evictable = min(
            self.token_to_kv_pool_allocator.full_available_size()
            + self.tree_cache.full_evictable_size(),
            self.token_to_kv_pool_allocator.swa_available_size()
            + self.tree_cache.swa_evictable_size(),
        )
    else:
        # 普通模式
        available_and_evictable = (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
        )
    
    return available_and_evictable - self.rem_total_token_offset
```

**计算逻辑**:
```
剩余 token 预算 = (可用内存 + 可驱逐内存) - 已分配的内存
```

---

### 4.3 add_one_req() 核心逻辑

**代码位置**: `python/sglang/srt/managers/schedule_policy.py:553`

```python
def add_one_req(
    self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
) -> AddReqResult:
    """
    尝试添加一个请求到 batch
    
    返回:
        AddReqResult.CONTINUE: 成功添加，继续添加下一个
        AddReqResult.NO_TOKEN: 内存不足，停止添加
        AddReqResult.OTHER: 其他原因，停止添加
    """
    
    # ========== 步骤 1: 计算请求需要的总 token 数 ==========
    total_tokens = req.extend_input_len + min(
        req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS
    )
    # total_tokens = 输入 token 数 + 最大输出 token 数
    
    # ========== 步骤 2: 调整输入 token 数（考虑前缀缓存） ==========
    real_input_tokens = req.extend_input_len - req.host_hit_length
    real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
    # real_input_tokens: 实际需要处理的 token 数（排除缓存命中）
    prefix_len = len(req.prefix_indices)  # 缓存的 token 数
    
    # ========== 步骤 3: 检查总内存预算 ==========
    if total_tokens >= self.rem_total_tokens:
        return AddReqResult.NO_TOKEN  # 内存不足
    
    # ========== 步骤 4: 检查输入 token 预算 ==========
    if real_input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
        return AddReqResult.OTHER  # 输入 token 预算不足
    
    # ========== 步骤 5: 锁定 Radix Tree 节点 ==========
    with self._lock_node(req.last_node):
        # 锁定节点，防止并发修改
        
        # ========== 步骤 6: 再次检查内存（锁定后可能变化） ==========
        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN
        
        # ========== 步骤 7: 处理前缀缓存命中 ==========
        if req.host_hit_length > 0:
            # 如果有前缀缓存命中，加载缓存的 KV
            new_indices, req.last_node = self.tree_cache.init_load_back(
                req.last_host_node, req.host_hit_length
            )
            req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            prefix_len = len(req.prefix_indices)
            req.last_matched_prefix_len = prefix_len
        
        # ========== 步骤 8: 计算页对齐的输入 token 数 ==========
        input_tokens = self.ceil_paged_tokens(req.extend_input_len)
        
        # ========== 步骤 9: 再次检查输入 token 预算 ==========
        if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER
        
        # ========== 步骤 10: 决定是否使用 Chunked Prefill ==========
        if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
            # ========== 情况 A: 非 Chunked Prefill ==========
            # 请求可以一次性处理完
            self.can_run_list.append(req)  # 添加到可以运行的列表
            
            # 锁定 Radix Tree 节点（防止被驱逐）
            if self.is_hybrid:
                swa_uuid_for_lock = self.tree_cache.inc_lock_ref(req.last_node)
                req.swa_uuid_for_lock = swa_uuid_for_lock
            else:
                self.tree_cache.inc_lock_ref(req.last_node)
            
            # 更新内存预算
            self._update_prefill_budget(
                prefix_len,                    # 缓存的 token 数（不占用新内存）
                input_tokens,                  # 需要处理的输入 token 数
                min(
                    req.sampling_params.max_new_tokens,
                    CLIP_MAX_NEW_TOKENS,
                ),  # 最大输出 token 数
            )
        else:
            # ========== 情况 B: Chunked Prefill ==========
            # 请求太大，需要分块处理
            
            # 计算可以处理的 chunk 大小
            trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
            if trunc_len <= 0:
                return AddReqResult.OTHER
            
            # 处理对齐要求
            if truncation_align_size is not None:
                if trunc_len < truncation_align_size:
                    return AddReqResult.OTHER
                else:
                    trunc_len = truncation_align_size * (
                        trunc_len // truncation_align_size
                    )
            
            # 截断请求
            req.extend_input_len = trunc_len
            req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]
            
            # 添加到可以运行的列表
            self.can_run_list.append(req)
            self.new_chunked_req = req  # 标记为 chunked request
            
            # 锁定节点
            if self.is_hybrid:
                swa_uuid_for_lock = self.tree_cache.inc_lock_ref(req.last_node)
                req.swa_uuid_for_lock = swa_uuid_for_lock
            else:
                self.tree_cache.inc_lock_ref(req.last_node)
            
            # 更新预算（chunked prefill 不预留输出 token）
            self._update_prefill_budget(prefix_len, trunc_len, 0)
    
    # ========== 步骤 11: 返回预算状态 ==========
    return self.budget_state()
```

**关键决策点**:

1. **内存检查**:
   ```python
   if total_tokens >= self.rem_total_tokens:
       return AddReqResult.NO_TOKEN  # 内存不足
   ```

2. **前缀缓存处理**:
   ```python
   if req.host_hit_length > 0:
       # 加载缓存的 KV，减少需要处理的 token 数
       prefix_len = len(req.prefix_indices)  # 缓存的 token 数
       req.extend_input_len = ...  # 减少需要处理的 token 数
   ```

3. **Chunked Prefill 决策**:
   ```python
   if input_tokens <= self.rem_chunk_tokens:
       # 一次性处理
   else:
       # 分块处理
       trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
   ```

---

### 4.4 _update_prefill_budget() 内存预算更新

```python
def _update_prefill_budget(
    self, prefix_len: int, extend_input_len: int, max_new_tokens: int
):
    """更新内存预算"""
    
    # 页对齐
    extend_input_len = self.ceil_paged_tokens(extend_input_len)
    
    # 更新总 token 偏移（已分配的内存）
    self.rem_total_token_offset += extend_input_len + max_new_tokens
    # extend_input_len: 输入 token 占用的内存
    # max_new_tokens: 输出 token 预留的内存
    
    # 更新当前 token 偏移（当前 batch 占用的内存）
    self.cur_rem_token_offset += extend_input_len
    
    # 更新输入 token 预算
    self.rem_input_tokens -= extend_input_len
    
    # 更新 chunk token 预算
    if self.rem_chunk_tokens is not None:
        self.rem_chunk_tokens -= extend_input_len
    
    # 统计信息
    self.log_hit_tokens += prefix_len      # 缓存的 token（节省的内存）
    self.log_input_tokens += extend_input_len  # 输入的 token
```

**内存预算计算**:
```
已分配内存 = rem_total_token_offset
  = sum(extend_input_len + max_new_tokens for each req)

剩余内存 = 总内存 - 已分配内存
  = available_size + evictable_size - rem_total_token_offset
```

---

### 4.5 budget_state() 预算状态检查

```python
def budget_state(self):
    """检查预算状态"""
    
    if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
        return AddReqResult.NO_TOKEN  # 总内存不足
    
    if self.rem_input_tokens <= 0 or (
        self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0
    ):
        return AddReqResult.OTHER  # 输入 token 预算不足
    
    return AddReqResult.CONTINUE  # 可以继续添加
```

---

## 5. 调度决策示例

### 示例 1: 简单场景（无前缀缓存）

**初始状态**:
```python
waiting_queue = [
    Req(rid="req-1", origin_input_ids=[1,2,3], max_new_tokens=50),
    Req(rid="req-2", origin_input_ids=[4,5,6,7], max_new_tokens=100),
    Req(rid="req-3", origin_input_ids=[8,9], max_new_tokens=30),
]

running_batch = ScheduleBatch(reqs=[])  # 空的
rem_total_tokens = 1000
rem_input_tokens = 500
```

**调度过程**:

```python
# 遍历 waiting_queue
for req in waiting_queue:
    # req-1: extend_input_len=3, max_new_tokens=50
    # total_tokens = 3 + 50 = 53
    # 检查: 53 < 1000 ✅, 3 < 500 ✅
    # → 添加到 can_run_list
    
    # req-2: extend_input_len=4, max_new_tokens=100
    # total_tokens = 4 + 100 = 104
    # 检查: 104 < (1000-53) ✅, 4 < (500-3) ✅
    # → 添加到 can_run_list
    
    # req-3: extend_input_len=2, max_new_tokens=30
    # total_tokens = 2 + 30 = 32
    # 检查: 32 < (1000-53-104) ✅, 2 < (500-3-4) ✅
    # → 添加到 can_run_list

can_run_list = [req-1, req-2, req-3]
waiting_queue = []  # 清空
```

**结果**: 所有请求都被选中，组成一个 batch。

---

### 示例 2: 内存不足场景

**初始状态**:
```python
waiting_queue = [
    Req(rid="req-1", origin_input_ids=[1]*100, max_new_tokens=200),  # 大请求
    Req(rid="req-2", origin_input_ids=[2]*50, max_new_tokens=100),
    Req(rid="req-3", origin_input_ids=[3]*10, max_new_tokens=50),
]

rem_total_tokens = 500
rem_input_tokens = 200
```

**调度过程**:

```python
for req in waiting_queue:
    # req-1: extend_input_len=100, max_new_tokens=200
    # total_tokens = 100 + 200 = 300
    # 检查: 300 < 500 ✅, 100 < 200 ✅
    # → 添加到 can_run_list
    # 更新预算: rem_total_tokens = 500 - 300 = 200
    #           rem_input_tokens = 200 - 100 = 100
    
    # req-2: extend_input_len=50, max_new_tokens=100
    # total_tokens = 50 + 100 = 150
    # 检查: 150 < 200 ✅, 50 < 100 ✅
    # → 添加到 can_run_list
    # 更新预算: rem_total_tokens = 200 - 150 = 50
    #           rem_input_tokens = 100 - 50 = 50
    
    # req-3: extend_input_len=10, max_new_tokens=50
    # total_tokens = 10 + 50 = 60
    # 检查: 60 < 50 ❌
    # → 返回 AddReqResult.NO_TOKEN
    # → 停止添加

can_run_list = [req-1, req-2]
waiting_queue = [req-3]  # req-3 继续等待
```

**结果**: 只有前两个请求被选中，第三个请求继续在 `waiting_queue` 中等待。

---

### 示例 3: 前缀缓存场景（RadixAttention）

**初始状态**:
```python
waiting_queue = [
    Req(rid="req-1", origin_input_ids=[1,2,3,4,5], ...),
    Req(rid="req-2", origin_input_ids=[1,2,3,6,7], ...),  # 前缀 [1,2,3] 相同
]

# Radix Tree 中已缓存 [1,2,3]
tree_cache.match_prefix([1,2,3]) → prefix_indices=[0,1,2]
```

**调度过程**:

```python
for req in waiting_queue:
    # req-1: origin_input_ids=[1,2,3,4,5]
    # 匹配前缀: [1,2,3] → prefix_indices=[0,1,2], prefix_len=3
    # extend_input_len = 5 - 3 = 2  # 只需要处理 [4,5]
    # total_tokens = 2 + 50 = 52  # 大幅减少！
    # → 添加到 can_run_list
    
    # req-2: origin_input_ids=[1,2,3,6,7]
    # 匹配前缀: [1,2,3] → prefix_indices=[0,1,2], prefix_len=3
    # extend_input_len = 5 - 3 = 2  # 只需要处理 [6,7]
    # total_tokens = 2 + 50 = 52
    # → 添加到 can_run_list

can_run_list = [req-1, req-2]
# 两个请求共享前缀缓存，内存占用大幅减少！
```

**结果**: 两个请求都利用前缀缓存，内存占用从 `(5+50)*2=110` 减少到 `(2+50)*2=104`。

---

### 示例 4: Chunked Prefill 场景

**初始状态**:
```python
waiting_queue = [
    Req(rid="req-1", origin_input_ids=[1]*1000, max_new_tokens=200),  # 超长请求
]

rem_total_tokens = 2000
rem_input_tokens = 500
rem_chunk_tokens = 100  # Chunk 大小限制
```

**调度过程**:

```python
for req in waiting_queue:
    # req-1: extend_input_len=1000
    # total_tokens = 1000 + 200 = 1200
    # 检查: 1200 < 2000 ✅
    # 但是: 1000 > 100 (rem_chunk_tokens) ❌
    # → 使用 Chunked Prefill
    
    # 计算 chunk 大小
    trunc_len = 100 // page_size * page_size  # 假设 page_size=16
    trunc_len = 96  # 可以处理的 chunk 大小
    
    # 截断请求
    req.extend_input_len = 96
    req.fill_ids = req.fill_ids[:prefix_len + 96]
    
    # 添加到 can_run_list
    self.can_run_list.append(req)
    self.new_chunked_req = req  # 标记为 chunked
    
    # 更新预算（不预留输出 token）
    self._update_prefill_budget(prefix_len, 96, 0)

can_run_list = [req-1]  # 但只处理前 96 个 token
self.chunked_req = req-1  # 剩余的 token 下次处理
```

**结果**: 请求被分块处理，第一次只处理前 96 个 token，剩余的 token 在后续循环中处理。

---

## 6. ScheduleBatch 的结构

**代码位置**: `python/sglang/srt/managers/schedule_batch.py:868`

```python
@dataclasses.dataclass
class ScheduleBatch:
    """存储 batch 的所有信息"""
    
    # 请求列表
    reqs: List[Req]
    
    # 内存池和缓存
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    
    # Batch 配置
    model_config: ModelConfig
    forward_mode: ForwardMode  # PREFILL, DECODE, MIXED
    
    # 批处理张量（用于 GPU 推理）
    input_ids: torch.Tensor          # [batch_size], int64
    req_pool_indices: torch.Tensor   # [batch_size], int64
    seq_lens: torch.Tensor           # [batch_size], int64
    out_cache_loc: torch.Tensor      # [batch_size], int64
    output_ids: torch.Tensor         # [batch_size], int64
    
    # 前缀信息
    prefix_lens: List[int]           # 每个请求的前缀长度
    extend_lens: List[int]            # 每个请求的扩展长度
    
    # 采样信息
    sampling_info: SamplingBatchInfo
    
    # 其他字段...
```

**ScheduleBatch 包含**:
- ✅ **请求列表**: `reqs`（选中的请求）
- ✅ **GPU 张量**: `input_ids`, `seq_lens` 等（用于模型推理）
- ✅ **内存索引**: `req_pool_indices`, `out_cache_loc`（KV Cache 位置）
- ✅ **采样信息**: `sampling_info`（temperature, top_p 等）

---

## 7. 完整的调度流程图

```
┌─────────────────────────────────────────────────────────┐
│ event_loop_normal()                                      │
│   while True:                                            │
└───────────────┬─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ recv_requests()                                         │
│   - 非阻塞接收新请求 (zmq.NOBLOCK)                       │
│   - 返回: List[Req]                                      │
└───────────────┬─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ process_input_requests(recv_reqs)                        │
│   - 处理每个请求                                          │
│   - 添加到 waiting_queue                                 │
└───────────────┬─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ get_next_batch_to_run()                                  │
│   ├─ 合并 last_batch → running_batch                    │
│   ├─ get_new_batch_prefill() ← 核心！                   │
│   │   ├─ 检查 waiting_queue 是否为空                     │
│   │   ├─ 计算优先级                                       │
│   │   ├─ 创建 PrefillAdder                               │
│   │   ├─ 遍历 waiting_queue:                             │
│   │   │   ├─ 检查 LoRA 兼容性                            │
│   │   │   ├─ 检查批处理大小                               │
│   │   │   ├─ 检查内存预算                                 │
│   │   │   ├─ req.init_next_round_input()                 │
│   │   │   │   └─ 匹配前缀缓存 (RadixAttention)          │
│   │   │   └─ adder.add_one_req()                         │
│   │   │       ├─ 检查内存是否足够                          │
│   │   │       ├─ 处理前缀缓存                              │
│   │   │       ├─ 决定是否 Chunked Prefill                 │
│   │   │       └─ 更新内存预算                             │
│   │   ├─ 获取 can_run_list                               │
│   │   ├─ 更新 waiting_queue                              │
│   │   └─ 创建 ScheduleBatch                              │
│   └─ 返回 batch 或 None                                  │
└───────────────┬─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ if batch:                                                │
│   run_batch(batch)  # GPU 推理                           │
│   process_batch_result(batch, result)                    │
└─────────────────────────────────────────────────────────┘
```

---

## 8. 关键数据结构总结

### 8.1 waiting_queue

```python
waiting_queue: List[Req] = []
```

**内容**: `Req` 对象列表
- 每个 `Req` 包含请求的所有信息
- 按优先级排序（如果需要）
- 等待被调度到 GPU

---

### 8.2 can_run_list

```python
adder.can_run_list: List[Req] = []
```

**内容**: 可以运行的请求列表
- 从 `waiting_queue` 中选出的请求
- 满足所有条件（内存、LoRA、优先级等）
- 会被添加到新的 `ScheduleBatch`

---

### 8.3 ScheduleBatch

```python
ScheduleBatch(
    reqs: List[Req],                    # 请求列表
    input_ids: torch.Tensor,            # GPU 张量
    seq_lens: torch.Tensor,             # 序列长度
    sampling_info: SamplingBatchInfo,  # 采样信息
    ...
)
```

**内容**: 准备发送到 GPU 的 batch
- 包含所有需要的信息
- 张量已准备好
- 可以直接调用 `run_batch()`

---

## 9. 关键代码位置总结

| 组件 | 代码位置 | 功能 |
|------|---------|------|
| `waiting_queue` | `scheduler.py:509` | 等待队列定义 |
| `get_next_batch_to_run()` | `scheduler.py:1739` | 主调度函数 |
| `get_new_batch_prefill()` | `scheduler.py:1810` | Prefill Batch 创建 |
| `PrefillAdder` | `schedule_policy.py:315` | 请求添加逻辑 |
| `add_one_req()` | `schedule_policy.py:553` | 单个请求添加 |
| `Req` 类 | `schedule_batch.py:432` | 请求对象定义 |
| `ScheduleBatch` 类 | `schedule_batch.py:868` | Batch 对象定义 |

---

## 10. 为什么 Batching 会伤害 TFFT（Time to First Token）？

### 10.1 问题背景

**TFFT (Time to First Token)** 是从用户发送请求到收到第一个生成 token 的时间间隔。这是衡量用户体验的关键指标。

### 10.2 Batching 导致 TFFT 增加的原因

#### 原因 1: 等待组成 Batch 的延迟

**核心问题**: 调度器会**等待收集多个请求**组成 batch，而不是立即处理单个请求。

**流程分析**:

```python
# 在 get_new_batch_prefill() 中
for req in self.waiting_queue:  # ← 遍历整个队列
    # 尝试添加请求到 batch
    res = adder.add_one_req(req, ...)
    if res == AddReqResult.CONTINUE:
        # 继续添加下一个请求，而不是立即运行
        continue
    else:
        break  # 停止添加，但已经等待了多个请求

# 只有收集完所有请求后，才创建 batch 并运行
can_run_list = adder.can_run_list
new_batch = ScheduleBatch.init_new(can_run_list, ...)
```

**时间线示例**:

```
时间轴:
t0: 请求 A 到达 waiting_queue
t1: 调度器调用 get_new_batch_prefill()
    - 发现请求 A 可以运行
    - 但继续遍历队列，寻找更多请求组成 batch
t2: 请求 B 到达 waiting_queue
t3: 调度器继续添加请求 B 到 batch
t4: 调度器决定 batch 已满或停止添加
t5: 创建 ScheduleBatch 并运行
t6: GPU 开始处理（Prefill）
t7: 第一个 token 生成 ← TFFT 结束

TFFT = t7 - t0 = (等待时间) + (Prefill 时间)
```

**如果没有 batching**:
- 请求 A 到达后立即运行
- TFFT = (Prefill 时间)  ← 更短！

#### 原因 2: 批处理大小限制导致的排队

**代码位置**: `scheduler.py:320`

```python
# 检查批处理大小限制
if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
    self.running_batch.batch_is_full = True
    # 即使还有内存，也可能因为 batch 大小限制而停止添加
```

**影响**:
- 如果当前 batch 已满，新请求必须等待当前 batch 完成
- 即使 GPU 有足够资源，也要等待

#### 原因 3: 内存预算共享导致的延迟

**代码位置**: `schedule_policy.py:542-547`

```python
# 检查总内存预算
if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN  # 内存不足

# 检查输入 token 预算
if real_input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
    return AddReqResult.OTHER  # 输入 token 预算不足
```

**影响**:
- 多个请求共享内存预算
- 如果前面的请求占用了大量内存，后续请求必须等待
- 即使单个请求可以运行，也要等待其他请求释放内存

#### 原因 4: 优先级调度导致的等待

**代码位置**: `scheduler.py:277`

```python
# 计算优先级
self.policy.calc_priority(self.waiting_queue)
# 根据优先级对 waiting_queue 排序
```

**影响**:
- 低优先级请求可能被高优先级请求插队
- 即使低优先级请求先到达，也要等待高优先级请求处理完

### 10.3 具体示例

**场景**: 两个请求几乎同时到达

```
时间线:
t=0ms:  请求 A 到达 waiting_queue
t=1ms:  请求 B 到达 waiting_queue
t=2ms:  调度器调用 get_new_batch_prefill()
        - 添加请求 A 到 can_run_list
        - 继续遍历，添加请求 B 到 can_run_list
        - 创建 batch = [A, B]
t=3ms:  GPU 开始处理 batch
t=50ms: Prefill 完成（假设需要 50ms）
t=51ms: 第一个 token 生成

TFFT_A = 51ms
TFFT_B = 50ms
```

**如果没有 batching**:
```
时间线:
t=0ms:  请求 A 到达 waiting_queue
t=1ms:  调度器立即运行请求 A
t=2ms:  GPU 开始处理请求 A
t=50ms: Prefill 完成
t=51ms: 第一个 token 生成

TFFT_A = 51ms（相同）

t=1ms:  请求 B 到达 waiting_queue
t=2ms:  调度器立即运行请求 B（如果 GPU 空闲）
t=3ms:  GPU 开始处理请求 B
t=52ms: Prefill 完成
t=53ms: 第一个 token 生成

TFFT_B = 52ms（如果并行处理，可能更短）
```

**关键差异**:
- **有 batching**: 请求 B 必须等待请求 A 一起处理，即使它们可以并行
- **无 batching**: 请求可以立即处理，减少等待时间

### 10.4 权衡：Batching vs TFFT

**Batching 的好处**:
- ✅ **提高 GPU 利用率**: 批量处理更高效
- ✅ **提高吞吐量**: 一次处理多个请求
- ✅ **减少内存碎片**: 统一管理内存

**Batching 的代价**:
- ❌ **增加 TFFT**: 等待组成 batch 的时间
- ❌ **增加延迟**: 请求必须等待其他请求
- ❌ **不公平**: 后到达的请求可能先处理（如果优先级更高）

### 10.5 如何平衡 Batching 和 TFFT？

#### 10.5.1 实际系统中的平衡策略

**1. 动态批处理超时（Dynamic Batching Timeout）**

**代码位置**: `async_dynamic_batch_tokenizer.py:86-101`

```python
# 收集请求，但设置超时
start_time = asyncio.get_running_loop().time()
while len(prompts) < self.max_batch_size:
    elapsed = asyncio.get_running_loop().time() - start_time
    if elapsed >= self.batch_wait_timeout_s:  # 超时阈值
        break  # 不再等待，立即处理
    
    # 等待新请求，但不超过剩余超时时间
    remaining_time = self.batch_wait_timeout_s - elapsed
    try:
        prompt = await asyncio.wait_for(
            self._queue.get(), remaining_time
        )
        prompts.append(prompt)
    except asyncio.TimeoutError:
        break  # 超时，立即处理已有请求
```

**策略**:
- ✅ **有请求立即处理**: 如果队列为空，立即处理单个请求（不等待）
- ✅ **超时机制**: 设置 `batch_wait_timeout_s`（如 2ms），超过时间立即处理
- ✅ **平衡点**: 在等待更多请求和立即处理之间找到平衡

**2. 优先级队列 + 抢占机制**

**代码位置**: `schedule_policy.py:641-697`

```python
def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
    """高优先级请求可以抢占低优先级请求"""
    priority_diff = req.priority - running_req.priority
    if priority_diff > self.priority_scheduling_preemption_threshold:
        # 抢占低优先级请求
        preemptible_reqs.append(running_req)
        return True
    return False
```

**策略**:
- ✅ **优先级排序**: 高优先级请求优先处理
- ✅ **抢占机制**: 高优先级请求可以抢占正在运行的低优先级请求
- ✅ **阈值控制**: `priority_scheduling_preemption_threshold` 防止频繁抢占

**3. 等待时间感知调度**

**代码位置**: `schedule_policy.py:274-285`

```python
def _sort_by_priority_and_fcfs(waiting_queue, ...):
    """按优先级和到达时间排序"""
    waiting_queue.sort(
        key=lambda x: (
            -x.priority,  # 优先级（高优先级优先）
            x.time_stats.wait_queue_entry_time  # 到达时间（先到先服务）
        )
    )
```

**策略**:
- ✅ **FCFS (First Come First Served)**: 同优先级请求按到达时间排序
- ✅ **防止饥饿**: 避免低优先级请求无限等待

#### 10.5.2 算法角度：类似 LeetCode 问题

这个问题本质上是**多目标优化问题**，类似于以下 LeetCode 问题：

**1. LeetCode 621: Task Scheduler（任务调度器）**

**相似点**:
- 需要在执行时间和吞吐量之间平衡
- 使用优先级队列管理任务
- 需要考虑等待时间和资源利用率

**核心思路**:
```python
# 伪代码
def schedule_tasks(tasks, n):
    # 使用优先级队列
    heap = PriorityQueue()
    for task in tasks:
        heap.push((priority, arrival_time, task))
    
    result = []
    while heap:
        # 贪心策略：选择优先级最高且等待时间最长的任务
        task = heap.pop()
        result.append(task)
        
        # 更新等待时间
        update_waiting_time(task)
```

**2. LeetCode 253: Meeting Rooms II（会议室问题 II）**

**相似点**:
- 需要分配有限资源（GPU/内存）给多个请求
- 需要在延迟和吞吐量之间平衡
- 使用贪心算法 + 优先级队列

**核心思路**:
```python
# 伪代码
def allocate_resources(requests, max_batch_size, timeout):
    # 按到达时间排序
    requests.sort(key=lambda x: x.arrival_time)
    
    batch = []
    start_time = time.now()
    
    for req in requests:
        # 检查超时
        if time.now() - start_time >= timeout:
            process_batch(batch)  # 立即处理
            batch = [req]
            start_time = time.now()
        elif len(batch) < max_batch_size:
            batch.append(req)
        else:
            process_batch(batch)  # 批次已满，立即处理
            batch = [req]
            start_time = time.now()
    
    if batch:
        process_batch(batch)
```

**3. 滑动窗口 + 超时机制**

**核心思路**:
```python
def dynamic_batching_with_timeout(requests, max_batch_size, timeout_ms):
    """
    动态批处理算法：
    - 目标：最大化吞吐量，最小化 TFFT
    - 策略：滑动窗口 + 超时机制
    """
    batch = []
    window_start = time.now()
    
    while True:
        # 等待新请求，但设置超时
        try:
            req = wait_for_request(timeout=timeout_ms)
            batch.append(req)
            
            # 检查批次是否已满
            if len(batch) >= max_batch_size:
                process_batch(batch)
                batch = []
                window_start = time.now()
        except TimeoutError:
            # 超时：立即处理已有请求
            if batch:
                process_batch(batch)
                batch = []
            window_start = time.now()
```

#### 10.5.3 算法设计模式

**1. 贪心算法（Greedy Algorithm）**

**策略**: 每次选择"最优"的请求加入 batch
- **最优标准**: 优先级最高、等待时间最长、内存占用最小等
- **优点**: 简单高效，实时性好
- **缺点**: 可能不是全局最优

```python
def greedy_batching(waiting_queue, max_batch_size):
    """贪心策略：优先选择高优先级请求"""
    batch = []
    # 按优先级排序
    waiting_queue.sort(key=lambda x: -x.priority)
    
    for req in waiting_queue:
        if can_add_to_batch(req, batch) and len(batch) < max_batch_size:
            batch.append(req)
        else:
            break  # 无法添加更多请求
    
    return batch
```

**2. 动态规划（Dynamic Programming）**

**策略**: 考虑所有可能的 batch 组合，选择最优
- **状态**: `dp[i][j]` = 处理前 i 个请求，batch 大小为 j 时的最小平均 TFFT
- **优点**: 全局最优
- **缺点**: 时间复杂度高，不适合实时系统

```python
def dp_batching(requests, max_batch_size):
    """
    DP 状态: dp[i][j] = 处理前 i 个请求，最后一个 batch 大小为 j 的最小平均 TFFT
    转移: dp[i][j] = min(dp[i-k][k] + calculate_tfft(batch) for k in [1, min(j, max_batch_size)])
    """
    n = len(requests)
    dp = [[float('inf')] * (max_batch_size + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, min(i, max_batch_size) + 1):
            # 尝试将最后 j 个请求组成一个 batch
            batch = requests[i-j:i]
            tfft = calculate_average_tfft(batch)
            dp[i][j] = min(dp[i-j][k] + tfft for k in range(max_batch_size + 1))
    
    return min(dp[n])
```

**3. 多目标优化（Multi-Objective Optimization）**

**策略**: 同时优化多个目标（吞吐量、TFFT、公平性）
- **方法**: 加权求和、Pareto 最优、约束优化
- **目标函数**: `minimize: α * avg_tfft + β * (1/throughput) + γ * unfairness`

```python
def multi_objective_batching(requests, weights):
    """
    多目标优化：
    - 目标1: 最小化平均 TFFT
    - 目标2: 最大化吞吐量
    - 目标3: 最小化不公平性（等待时间方差）
    """
    alpha, beta, gamma = weights
    
    def objective(batch):
        avg_tfft = calculate_avg_tfft(batch)
        throughput = len(batch) / calculate_time(batch)
        unfairness = calculate_waiting_time_variance(batch)
        
        return alpha * avg_tfft + beta * (1/throughput) + gamma * unfairness
    
    # 使用启发式算法（如遗传算法、模拟退火）找到近似最优解
    best_batch = heuristic_search(requests, objective)
    return best_batch
```

#### 10.5.4 SGLang 的实际实现

SGLang 采用了**混合策略**，结合了多种算法思想：

1. **贪心 + 超时**: 优先选择高优先级请求，但设置超时阈值
2. **优先级队列**: 使用堆（heap）管理请求优先级
3. **抢占机制**: 高优先级请求可以抢占低优先级请求
4. **FCFS 公平性**: 同优先级请求按到达时间排序

**代码示例**:
```python
# SGLang 的调度逻辑（简化版）
def get_new_batch_prefill(self):
    # 1. 按优先级排序（贪心）
    self.policy.calc_priority(self.waiting_queue)
    
    # 2. 创建批处理器
    adder = PrefillAdder(...)
    
    # 3. 遍历队列，添加请求（贪心 + 约束）
    for req in self.waiting_queue:
        # 检查内存、LoRA 兼容性等约束
        if can_add(req, adder):
            adder.add_one_req(req)
        else:
            # 尝试抢占（多目标优化）
            if self.try_preemption and adder.preempt_to_schedule(req):
                continue
            else:
                break  # 无法添加，停止
    
    # 4. 返回 batch
    return ScheduleBatch.init_new(adder.can_run_list, ...)
```

#### 10.5.5 参数调优建议

**关键参数**:
1. **`batch_wait_timeout_s`**: 批处理等待超时（如 2ms）
   - **太小**: 无法充分利用 batching，吞吐量低
   - **太大**: TFFT 增加，用户体验差
   - **建议**: 根据请求到达频率动态调整

2. **`priority_scheduling_preemption_threshold`**: 抢占阈值
   - **太小**: 频繁抢占，影响吞吐量
   - **太大**: 无法及时响应高优先级请求
   - **建议**: 根据业务需求设置（如 5-10）

3. **`max_batch_size`**: 最大批次大小
   - **太小**: 无法充分利用 GPU
   - **太大**: 内存不足，TFFT 增加
   - **建议**: 根据 GPU 内存和模型大小设置

**调优策略**:
- **低延迟场景**: 减小 `batch_wait_timeout_s`，优先处理单个请求
- **高吞吐场景**: 增大 `batch_wait_timeout_s`，等待更多请求组成 batch
- **混合场景**: 使用优先级调度，高优先级请求低延迟，低优先级请求高吞吐

#### 10.5.6 算法思路总结

**核心问题**: 如何在**吞吐量（Throughput）**和**延迟（TFFT）**之间找到平衡？

**LeetCode 类似问题**:
1. **LeetCode 621: Task Scheduler** - 任务调度，优先级队列
2. **LeetCode 253: Meeting Rooms II** - 资源分配，贪心算法
3. **LeetCode 358: Rearrange String k Distance Apart** - 间隔调度，堆/优先级队列

**算法设计模式**:
1. **贪心算法**: 每次选择最优请求（优先级最高、等待时间最长）
2. **滑动窗口**: 设置超时窗口，窗口内收集请求组成 batch
3. **优先级队列**: 使用堆管理请求优先级
4. **多目标优化**: 同时优化吞吐量、TFFT、公平性

**实际系统策略**:
- ✅ **超时机制**: 设置 `batch_wait_timeout_s`，超过时间立即处理
- ✅ **优先级调度**: 高优先级请求优先处理，可以抢占低优先级请求
- ✅ **FCFS 公平性**: 同优先级请求按到达时间排序
- ✅ **动态调整**: 根据系统负载动态调整参数

**关键洞察**:
> **没有完美的解决方案，只有权衡（Trade-off）**。系统需要在吞吐量和延迟之间找到平衡点，这个平衡点取决于业务需求：
> - 如果**用户体验优先**（如实时对话），优先考虑 TFFT，减小 batch 大小和等待时间
> - 如果**成本优先**（如离线批处理），优先考虑吞吐量，增大 batch 大小和等待时间
> - 如果**混合场景**，使用优先级调度，不同优先级采用不同策略

### 10.6 SGLang 的实际选择和设计

既然没有完美解决方案，SGLang 在当前版本中做了以下**具体选择和设计**：

#### 10.6.1 默认配置：偏向吞吐量

**SGLang 的默认配置**（`server_args.py`）:

```python
# 调度策略
schedule_policy: str = "fcfs"  # 先到先服务，保证公平性
enable_priority_scheduling: bool = False  # 默认关闭优先级调度
schedule_conservativeness: float = 1.0  # 保守调度，避免内存溢出

# 批处理参数
max_prefill_tokens: int = 16384  # 较大的 Prefill 批次大小
priority_scheduling_preemption_threshold: int = 10  # 抢占阈值（如果启用）

# 内存管理
chunked_prefill_size: Optional[int] = None  # 默认不限制，允许大请求
```

**设计理念**:
- ✅ **优先考虑吞吐量**: 默认配置偏向高吞吐量场景（离线批处理、大规模部署）
- ✅ **公平性优先**: 使用 FCFS（先到先服务），保证请求公平处理
- ✅ **保守调度**: `schedule_conservativeness = 1.0` 避免内存溢出，但可能降低吞吐量

#### 10.6.2 核心设计：没有显式超时机制

**关键发现**: SGLang 的调度器**没有使用显式的批处理超时机制**（与 tokenizer 不同）

**代码对比**:

```python
# Tokenizer 有超时机制（async_dynamic_batch_tokenizer.py）
batch_wait_timeout_s: float = 0.002  # 2ms 超时
if elapsed >= self.batch_wait_timeout_s:
    break  # 超时立即处理

# 调度器没有超时机制（scheduler.py）
def get_new_batch_prefill(self):
    # 直接遍历队列，没有超时检查
    for req in self.waiting_queue:
        res = adder.add_one_req(req, ...)
        if res != AddReqResult.CONTINUE:
            break  # 只有内存不足时才停止
```

**设计原因**:
1. **调度器层面更复杂**: 需要考虑内存、LoRA 兼容性、前缀缓存等多个约束
2. **事件驱动**: 调度器在每次 `get_next_batch_to_run()` 调用时都会尝试创建新 batch
3. **通过其他机制缓解**: 使用 RadixAttention、Chunked Prefill 等技术间接减少 TFFT

#### 10.6.3 实际策略：贪心算法 + 约束优化

**SGLang 采用的策略**:

```python
def get_new_batch_prefill(self):
    # 1. 按策略排序（FCFS/LPM/LOF）
    self.policy.calc_priority(self.waiting_queue)
    
    # 2. 贪心遍历：尽可能添加更多请求
    for req in self.waiting_queue:
        # 检查多个约束
        if can_add(req, adder):  # 内存、LoRA、前缀缓存等
            adder.add_one_req(req)
        else:
            # 尝试抢占（如果启用）
            if self.try_preemption and adder.preempt_to_schedule(req):
                continue
            else:
                break  # 无法添加，停止
    
    # 3. 立即返回 batch（不等待）
    return ScheduleBatch.init_new(adder.can_run_list, ...)
```

**特点**:
- ✅ **贪心算法**: 每次尽可能添加更多请求到 batch
- ✅ **立即处理**: 不等待超时，满足约束就立即创建 batch
- ✅ **多约束优化**: 同时考虑内存、LoRA、前缀缓存等多个因素

#### 10.6.4 缓解 TFFT 的技术手段

虽然调度器没有超时机制，但 SGLang 通过**其他技术手段**缓解 TFFT：

**1. RadixAttention（前缀缓存）**
- **效果**: 减少 Prefill 时间，间接减少 TFFT
- **原理**: 缓存相同前缀，只计算新部分
- **代码**: `req.init_next_round_input(self.tree_cache)` 匹配前缀

**2. Chunked Prefill（分块处理）**
- **效果**: 大请求不阻塞小请求
- **原理**: 将大请求分块处理，小请求可以立即运行
- **代码**: `chunked_prefill_size` 参数控制

**3. 调度策略优化**
- **FCFS**: 保证公平性，避免饥饿
- **LPM (Longest Prefix Match)**: 优先处理有前缀缓存的请求
- **LOF (Longest Output First)**: 优先处理长输出请求（提高吞吐量）

**4. 优先级调度（可选）**
- **默认关闭**: `enable_priority_scheduling = False`
- **如果启用**: 高优先级请求可以抢占低优先级请求
- **阈值控制**: `priority_scheduling_preemption_threshold = 10`

#### 10.6.5 参数调优建议（来自官方文档）

**根据 `docs/advanced_features/hyperparameter_tuning.md`**:

**1. 高吞吐量场景（默认）**:
```bash
# 默认配置已经偏向吞吐量
--schedule-policy fcfs
--schedule-conservativeness 1.0
--max-prefill-tokens 16384
```

**2. 低延迟场景**:
```bash
# 减小批次大小，优先处理单个请求
--schedule-conservativeness 0.3  # 更激进，更快处理
--max-prefill-tokens 8192  # 减小 Prefill 批次
--enable-priority-scheduling  # 启用优先级调度
```

**3. 混合场景**:
```bash
# 使用优先级调度，不同优先级采用不同策略
--enable-priority-scheduling
--priority-scheduling-preemption-threshold 5
--schedule-policy lpm  # 利用前缀缓存
```

#### 10.6.6 SGLang 的设计哲学

**SGLang 的选择**:

1. **默认偏向吞吐量**: 
   - 适合大规模部署场景（每天生成数万亿 tokens）
   - 优先考虑成本效益（GPU 利用率）

2. **通过技术手段缓解延迟**:
   - RadixAttention 减少 Prefill 时间
   - Chunked Prefill 避免大请求阻塞
   - 调度策略优化（LPM/LOF）

3. **提供灵活性**:
   - 用户可以通过参数调整平衡点
   - 支持优先级调度（可选）
   - 支持多种调度策略（FCFS/LPM/LOF）

4. **事件驱动而非时间驱动**:
   - 不在调度器层面使用超时机制
   - 每次事件循环都尝试创建新 batch
   - 通过其他机制（如 RadixAttention）间接优化延迟

**总结**:
> SGLang 选择了**"默认偏向吞吐量，通过技术手段缓解延迟"**的策略。这是一个**实用主义**的选择：
> - 对于**大规模生产环境**（SGLang 的主要使用场景），吞吐量比单个请求的 TFFT 更重要
> - 通过 **RadixAttention** 等技术，可以在不牺牲吞吐量的情况下显著减少 TFFT
> - 用户可以通过参数调整来适应不同的业务需求

---

## 11. 总结

### waiting_queue 的内容
- **类型**: `List[Req]`
- **元素**: `Req` 对象，包含请求的所有信息
- **用途**: 存储等待被调度的请求

### get_new_batch_prefill() 的工作流程
1. **检查前置条件**: 队列是否为空、批处理是否已满
2. **计算优先级**: 对 `waiting_queue` 排序
3. **创建 PrefillAdder**: 管理内存预算和请求添加
4. **遍历 waiting_queue**: 尝试添加每个请求
5. **检查条件**: LoRA 兼容性、内存预算、批处理大小
6. **匹配前缀缓存**: 利用 RadixAttention 减少内存占用
7. **更新 waiting_queue**: 移除已选中的请求
8. **创建 ScheduleBatch**: 准备发送到 GPU

### 关键决策点
- **内存检查**: `total_tokens < rem_total_tokens`
- **前缀缓存**: `prefix_len` 减少需要处理的 token 数
- **Chunked Prefill**: `input_tokens > rem_chunk_tokens` 时启用
- **优先级调度**: 高优先级请求可以抢占低优先级请求

这就是调度器的完整实现！🎯

