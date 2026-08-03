# 第 8 章 KV Cache 与 RadixAttention 代码走读

> 代码来自 `python/sglang/srt/mem_cache/`：`memory_pool.py`（池子）与 `radix_cache.py`（前缀树）。

## 8.1 两级池子：逻辑位置 → 物理位置

KV 管理分两层（都在 `memory_pool.py`）：

```text
req_to_token_pool   # 逻辑层：req_to_token[req][seq_pos] = kv_loc
token_to_kv_pool    # 物理层：实际的 K/V 张量，按 kv_loc 索引
```

`ReqToTokenPool`（第 256 行）的真相就是一张二维表：

```python
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(self, size, max_context_len, device, enable_memory_saver):
        ...
        self._alloc_size = size + 1   # +1 是给 CUDA graph padding 的 dummy 行
        self.req_to_token = torch.zeros(
            (self._alloc_size, max_context_len), dtype=torch.int32, device=device
        )
```

`req_to_token[req_idx, pos]` 存的是"这个请求的第 pos 个 token 的 KV 在物理池的哪个位置"。这一层映射是 paged attention 的基础：**逻辑序列连续，物理存储可以乱序**。

## 8.2 物理池：按模型架构分化

`KVCache` 抽象基类（第 1581 行）定义接口：

```python
class KVCache(abc.ABC):
    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor: ...
    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor: ...
    @abc.abstractmethod
    def set_kv_buffer(self, layer: RadixAttention, loc, cache_k, cache_v): ...
```

具体实现按注意力架构分化：

| 实现 | 位置 | 存什么 |
| --- | --- | --- |
| `MHATokenToKVPool` | 第 1702 行 | 每层完整 K/V（标准 MHA/GQA） |
| `MLATokenToKVPool` | 第 3866 行 | 只存 MLA 压缩后的 latent KV，用时现场展开 |
| `DSATokenToKVPool` | 第 4276 行 | DeepSeek Sparse Attention 专用 |
| `MHATokenToKVPoolFP4` / `MXFP8` | — | 量化 KV |

MLA 池是最值得读的实现：DeepSeek 系列不存每层的完整 K/V，而是存低秩投影后的压缩向量，`set_kv_buffer` 时用共享矩阵展开成实际注意力要用的 K/V。**省显存，但算力换显存**——这也是为什么 MLA 模型需要专门的 attention kernel。

## 8.3 分配器：谁来给"页"

`mem_cache/allocator/` 提供两种粒度：

- `paged.py`：页对齐分配，`BaseTokenToKVPoolAllocator` 维护 free 页列表与原子计数；
- `token.py`：token 粒度（HiCache 等场景）。

调度器组批时问 `allocator.get_available_size()`，运行中每次 extend/decode 都 `alloc`/`free`。`PrefillAdder` 的显存预算就来自这里——第 7 章的"①显存检查"最终就是问它。

## 8.4 RadixKey：能匹配、能分页、能哈希的 token 序列

`radix_cache.py` 第 59 行：

```python
class RadixKey:
    def __init__(self, token_ids, extra_key=None, is_bigram=False):
        ...
    def page_aligned(self, page_size: int) -> RadixKey:   # 截断到页边界
    def match(self, other, page_size=1) -> int:           # 返回公共前缀长度
    def hash_page(self, start, end, prior_hash) -> str:   # 分页哈希
```

三个设计点：

1. `extra_key`：额外命名空间。LoRA、会话、采样盐不同时，即使 token 相同也**不共享缓存**——隔离正确性，防止缓存污染。
2. `page_aligned`：匹配只在页边界对齐的位置发生，保证共享的 KV 是整页的，内存池才能安全引用。
3. `hash_page`：分页哈希用于快速比较（大模型下逐 token 比较太慢）。

## 8.5 匹配：match_prefix 与节点分裂

`match_prefix`（第 352 行）找"最长的缓存前缀"，核心是 `_match_prefix_helper`（第 648 行）：

```python
def _match_prefix_helper(self, node, key):
    child_key = key.child_key(self.page_size)
    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):
            # 命中的地方在一个节点的“中间”→ 必须分裂
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)      # 整段命中，继续往下
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)
    return value, node
```

分裂 `_split_node`（第 674 行）是理解前缀树的关键：

```python
def _split_node(self, key, child, split_len):
    # new_node 继承 child 的优先级（它代表被共享的前缀）
    new_node = TreeNode(priority=child.priority)
    new_node.children = {key[split_len:].child_key(self.page_size): child}
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len].clone()
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:].clone()
    new_node.parent.children[key.child_key(self.page_size)] = new_node
    return new_node
```

场景：缓存里存了 100 个 token 的节点，新请求只命中前 30 个。如果不分裂，共享粒度是"整个 100"，后续 31~100 的缓存全部浪费。分裂后：30 个 token 的公共段成为新节点，两个分支各自持有剩余部分，**公共段从此可以被任意请求共享**。

## 8.6 插入与请求结束落缓存

`insert`（第 412 行）把一段 token → KV 映射写进树，`_insert_helper` 沿树找公共前缀，只写入差异部分。

请求结束时走 `cache_finished_req`（第 434 行）：

```python
def cache_finished_req(self, req, is_insert=True, *, kv_len_to_handle):
    token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
    kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :len(token_ids)]
    radix_key = RadixKey(token_ids, req.extra_key, ...).page_aligned(self.page_size)
    ...
    result = self.insert(InsertParams(key=radix_key, value=kv_indices, priority=...))
    # 从 result.prefix_len 之后的部分释放请求独占的 KV 页
    self.token_to_kv_pool_allocator.free_segment(...)
```

两件事：把 KV 引用写进树（共享），同时**释放请求独占的那部分页**（不浪费）。树里的引用会阻止这些页被释放，直到节点被淘汰。

## 8.7 淘汰：缓存满了怎么办

`evict_policy.py` 提供策略（LRU/LFU 等），`RadixCache` 用 `get_eviction_strategy` 选择。淘汰的是**叶子节点**（没有子节点的节点）：因为只有叶子不影响其他共享者。

> **不变量：只有叶子节点可以被淘汰。** 淘汰中间节点会破坏还在使用它的请求。

## 8.8 自己动手的实验

1. `--enable-cache-report` 启动，连续发两次**完全相同的** prompt，看第二次响应的 `cached_tokens` 是否等于 prompt 长度。
2. 发三条 prompt：`A+B`、`A+C`、`A+B+D`（A/B/C/D 是明显的分隔段落），观察第二次 `A+B` 的命中数。理解"任意公共前缀共享"。
3. `--disable-radix-cache` 开关，对比同一批 20 个共享前缀请求的总耗时。
4. 用 `/flush_cache` 清缓存，再发同样请求，观察命中数归零。

## 8.9 本章小结

- 两级池子（req_to_token / token_to_kv）实现"逻辑连续、物理乱序"。
- 物理池按架构分化：MHA / MLA / DSA / 量化各有实现。
- RadixKey 是匹配单元，extra_key 是隔离命名空间，page_aligned 保证整页共享。
- 匹配时节点分裂让共享粒度精确到 token。
- 请求结束落缓存并释放独占页；淘汰只动叶子。

> 下一章看 GPU 侧：这些 KV 怎么被注意力层消费，CUDA graph 又怎么把 decode 提速。
