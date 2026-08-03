# 第 8 章 内存与 KV Cache：RadixAttention 与层级化缓存

## 8.1 显存去哪儿了

推理时显存主要被三样东西占据：

1. **模型权重**；
2. **KV Cache**：每个 token 每层都要保存 key/value，随请求总长度线性增长，通常占显存大头；
3. **中间激活**（激活值、CUDA graph 缓冲等）。

`server_args.py` 的 `mem_fraction_static`（默认约 0.9）决定了把多少显存留给 KV Cache，启动日志会打印"KV Cache is allocated ..."。本章聚焦 KV Cache 的组织方式。

## 8.2 两级池子

SGLang 把 KV 管理拆成两个池（都在 `mem_cache/memory_pool.py`）：

| 池 | 类 | 作用 |
| --- | --- | --- |
| `req_to_token_pool` | `ReqToTokenPool`（第 256 行） | 记录"每个请求的每个 token 对应哪个 KV 位置"，形状 `(max_reqs, max_len)` |
| `token_to_kv_pool` | `KVCache` 子类（MHA/MLA/...） | 实际的 K/V 张量，按 token 位置组织 |

`ReqToTokenPool.req_to_token[req_idx, seq_pos] = kv_loc` 这行映射就是 paged KV 的核心：**逻辑 token 位置 → 物理 KV 位置**。

## 8.3 不同模型架构的 KV 池

`memory_pool.py` 里 `KVCache` 的抽象基类（第 1581 行）声明了 `get_key_buffer/get_value_buffer/set_kv_buffer` 等接口，具体实现按注意力类型分：

| 实现 | 适用 |
| --- | --- |
| `MHATokenToKVPool`（第 1702 行） | 标准 MHA/GQA 模型 |
| `MLATokenToKVPool`（第 3866 行） | DeepSeek 等 MLA 模型（只存压缩后的 latent KV） |
| `DSATokenToKVPool`（第 4276 行） | DeepSeek Sparse Attention |
| `MHATokenToKVPoolFP4` / `MXFP8` | 量化 KV cache |
| `HybridLinearKVPool` | 线性注意力/混合架构 |

MLA 是理解 DeepSeek 系列的关键：它不存每层的完整 K/V，而是存低秩压缩向量，`set_kv_buffer` 时现场展开。`layers/radix_attention.py` 里对应 `RadixAttention` 层会按模型类型分发到不同内核。

## 8.4 分配器：分页的思想

`mem_cache/allocator/` 提供分配策略：

- `paged.py`：页对齐分配（`BaseTokenToKVPoolAllocator`），按 `page_size` 分配连续页；
- `token.py`：token 粒度分配（可跨页，适合 HiCache 等场景）；
- `base.py`：接口与原子计数器。

调度器组批时问 allocator"还能分配多少页"，运行中每次 extend/decode 都要 `alloc`/`free`。这也是 `enable_memory_saver`（TorchMemorySaverAdapter）等显存复用技巧的挂载点。

## 8.5 RadixAttention：让公共前缀共享

`mem_cache/radix_cache.py` 的 `RadixCache`（第 279 行）把 KV 页组织成 **radix 树（前缀树）**：

- 树的每个节点存一段连续 token 的 KV 页引用；
- 两个请求有公共前缀时，前缀段的 KV 页被共享，各自只分配差异部分；
- 请求结束后，其 KV 段 `insert` 回树中（`cache_finished_req`，第 434 行），供后续请求复用。

```python
def match_prefix(self, params) -> MatchResult:
    """Find the longest cached prefix of key in the radix tree."""
    value, last_node = self._match_prefix_helper(self.root_node, key)
    ...

def insert(self, params) -> InsertResult:
    ...
    prefix_len, last_node = self._insert_helper(self.root_node, key, value, ...)
```

多轮对话、few-shot、共享 system prompt、Agent 场景里，这种复用能把 prefill 时间砍掉大半——这就是 README 里 "RadixAttention 5x 加速" 的来源。

## 8.6 树上的精细操作

`RadixKey`（第 59 行）是"可哈希、可分页"的 token 序列；`TreeNode`（第 220 行）带优先级、访问时间等元数据。树上操作包括：

- **match_prefix**：匹配时若命中的是节点内部某一段，会**分裂节点**（split），让边界精确，便于后续共享；
- **insert**：把新段插入，必要时合并相邻节点；
- **cache_finished_req**：请求结束时把 token→KV 映射写回树，同时**释放请求独占的页**（`free_segment`）；
- **eviction**：容量超限时按策略（`evict_policy.py`）淘汰叶子节点，如 LRU/LFU；
- **extra_key 命名空间**：`RadixKey` 支持 `extra_key`，不同 LoRA/会话可强制不共享前缀，避免 KV 污染。

## 8.7 缓存命中怎么回报给用户

调度器的 `load_snapshot`、`cache_report` 机制会把命中 token 数统计进 metrics；HTTP 响应里 `cached_tokens` 字段（OpenAI usage 中）就来自这里。`/flush_cache` 端点则整体清空这棵树。

## 8.8 更复杂的缓存形态（进阶预览）

`mem_cache/` 里还躺着不少高级缓存：

- `unified_cache/`、`hybrid_cache/`：统一/混合缓存（HiCache 生态，支持更大容量、异构存储）；
- `swa_radix_cache.py`、`pure_swa_radix_cache.py`：滑动窗口注意力模型的缓存；
- `mamba_radix_cache.py`、`hi_mamba_radix_cache.py`：Mamba/状态空间模型的缓存；
- `chunk_cache.py`、`multimodal_cache.py`：按块缓存、多模态特征缓存；
- `deepseek_v4_memory_pool.py` / `compress_state`：DeepSeek-V4 的压缩状态缓存；
- `cpp_radix_tree/`：C++ 实现的 radix 树（`tree_v2.cpp`），供大规模场景用。

这些是理解 SGLang 演进脉络的富矿：从"一棵树"到"一套缓存生态"。

## 8.9 本章小结

- KV Cache 由 `req_to_token_pool`（逻辑映射）与 `token_to_kv_pool`（物理存储）两级池子管理。
- 按架构分池：MHA/MLA/DSA/量化/线性注意力各有实现。
- RadixCache 用前缀树让公共前缀共享 KV 页，是 SGLang 最具标志性的设计。
- 缓存不是"一次性写入"，而是一个有匹配、插入、分裂、淘汰的活结构。
- 下一章看 GPU 执行侧：这些 KV 怎么被注意力层消费，以及 CUDA graph 如何加速。
