# RadixAttention 详解

## 🎯 什么是 RadixAttention？

**RadixAttention** 是 SGLang 的核心技术创新之一，用于实现**前缀缓存（Prefix Caching）**。它使用 **Radix Tree（基数树）** 数据结构来高效地存储和检索 LLM 请求的 KV Cache。

### 核心价值

- **性能提升**: 提供高达 **5x 的推理加速**
- **内存效率**: 智能缓存，减少重复计算
- **自动启用**: 默认开启，无需额外配置

## 📚 基础知识

### 什么是前缀缓存？

在 LLM 推理中，很多请求具有相同的**前缀（Prefix）**：

```
请求1: "你好，请介绍一下人工智能"
请求2: "你好，请介绍一下机器学习"
请求3: "你好，请介绍一下深度学习"

共同前缀: "你好，请介绍一下"
```

**传统方式**:
```
每个请求都重新计算前缀 → 浪费计算资源
```

**前缀缓存方式**:
```
计算一次前缀 → 缓存结果 → 后续请求直接使用 → 大幅加速
```

### 什么是 Radix Tree（基数树）？

Radix Tree 是一种压缩的前缀树（Trie），用于高效存储和查找具有共同前缀的字符串序列。

#### 特点：
- **压缩存储**: 相同前缀只存储一次
- **快速查找**: O(k) 时间复杂度（k 是 key 长度）
- **内存高效**: 共享公共前缀，节省内存

#### 示例结构：

```
根节点 (root)
├─ "你好"
│  ├─ "，请介绍一下"
│  │  ├─ "人工智能" [KV Cache 1]
│  │  ├─ "机器学习" [KV Cache 2]
│  │  └─ "深度学习" [KV Cache 3]
│  └─ "，今天天气" [KV Cache 4]
└─ "再见" [KV Cache 5]
```

## 🔧 RadixAttention 工作原理

### 1. 核心组件

在 SGLang 中，RadixAttention 主要包含：

#### a) RadixCache (`python/sglang/srt/mem_cache/radix_cache.py`)

```python
class RadixCache(BasePrefixCache):
    """使用 Radix Tree 实现的前缀缓存"""
    
    def match_prefix(self, key: RadixKey) -> MatchResult:
        """
        查找最长的缓存前缀
        
        返回:
        - device_indices: KV Cache 索引
        - last_node: 匹配到的最后一个节点
        - host_hit_length: 主机端命中长度
        """
```

#### b) RadixAttention Layer (`python/sglang/srt/layers/radix_attention.py`)

```python
class RadixAttention(nn.Module):
    """注意力层实现，支持前缀缓存"""
    
    def forward(self, q, k, v, forward_batch, ...):
        """前向传播，自动使用前缀缓存"""
```

### 2. 工作流程

#### 步骤 1: 请求到达

```
用户请求: "你好，请介绍一下人工智能"
```

#### 步骤 2: 前缀匹配

```python
# 在调度器中匹配前缀
r.prefix_indices, r.last_node, r.host_hit_length = (
    self.tree_cache.match_prefix(
        rid=r.rid, 
        key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
    )
)
```

**匹配过程**:
1. 从根节点开始遍历 Radix Tree
2. 查找最长的匹配前缀
3. 返回匹配到的 KV Cache 索引

#### 步骤 3: KV Cache 重用

```
如果找到匹配前缀:
    - 使用缓存的 KV Cache
    - 只计算新部分（"人工智能"）
    
如果没有匹配:
    - 计算完整序列
    - 将结果存入 Radix Tree
```

#### 步骤 4: 更新缓存

```
计算完成后，将新的 KV Cache 存储到 Radix Tree 中
供后续请求使用
```

### 3. 数据结构详解

#### RadixKey

```python
@dataclasses.dataclass
class RadixKey:
    """Radix Tree 的键"""
    token_ids: List[int]  # Token ID 序列
    extra_key: Optional[str] = None  # 额外的命名空间键
```

**extra_key 的作用**:
- 隔离不同 LoRA 适配器的缓存
- 区分不同的采样参数
- 分离不同的检索增强上下文

#### TreeNode

```python
class TreeNode:
    """Radix Tree 的节点"""
    key: RadixKey  # 节点对应的键
    value: List[torch.Tensor]  # KV Cache 值
    children: Dict  # 子节点
    lock_ref: int  # 引用计数（用于锁定）
```

#### MatchResult

```python
@dataclasses.dataclass
class MatchResult:
    """前缀匹配结果"""
    device_indices: torch.Tensor  # KV Cache 索引
    last_device_node: TreeNode  # 最后一个设备节点
    last_host_node: TreeNode  # 最后一个主机节点
    host_hit_length: int  # 主机端命中长度
```

## 🚀 性能优化

### 1. 批内前缀缓存（In-Batch Prefix Caching）

当多个请求共享相同前缀时：

```python
# 在等待队列中查找批内匹配
if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
    in_batch_matching_prefixes = (
        self.waiting_queue_radix_tree.match_prefix(...)
    )
    # 如果多个请求共享前缀，优先调度一个，提高缓存命中率
```

**优化策略**:
- 优先调度有共同前缀的请求
- 提高缓存命中率
- 减少重复计算

### 2. 分页支持（Paged Mode）

```python
if self.page_size == 1:
    # 单页模式：精确匹配
    self.key_match_fn = _key_match_page_size1
else:
    # 分页模式：按页对齐
    self.key_match_fn = partial(_key_match_paged, page_size=page_size)
```

**优势**:
- 支持更长的序列
- 更好的内存管理
- 减少内存碎片

### 3. 缓存淘汰策略

支持两种淘汰策略：

#### a) LRU (Least Recently Used)

```python
if eviction_policy.lower() == "lru":
    self.eviction_strategy: EvictionStrategy = LRUStrategy()
```

- **原理**: 淘汰最久未使用的节点
- **适用**: 大多数场景

#### b) LFU (Least Frequently Used)

```python
elif eviction_policy.lower() == "lfu":
    self.eviction_strategy: EvictionStrategy = LFUStrategy()
```

- **原理**: 淘汰使用频率最低的节点
- **适用**: 有明确热点数据的场景

## 💡 应用场景

### 1. 多轮对话

```
第1轮: "你好，请介绍一下人工智能"
第2轮: "你好，请介绍一下人工智能。它能做什么？"

第2轮可以直接使用第1轮的前缀缓存
```

### 2. 批量处理相似请求

```
请求1: "介绍一下机器学习"
请求2: "介绍一下深度学习"
请求3: "介绍一下强化学习"

共同前缀: "介绍一下"
```

### 3. 提示词模板

```
模板: "你是AI助手。请回答以下问题：{问题}"

所有使用该模板的请求都共享前缀
```

### 4. 函数调用

```
系统提示: "你是一个函数调用助手。..."
用户问题: "..."
助手回复: "..."

系统提示部分可以缓存，所有请求共享
```

## 📊 性能数据

### SGLang 官方数据

根据 SGLang 博客（2024/01）：

- **推理加速**: 高达 **5x** 加速
- **内存节省**: 显著减少内存使用
- **适用场景**: 多轮对话、批量处理、提示词模板

### 实际效果

```
场景: 100 个请求，都有相同的前缀（100 tokens）

传统方式:
- 每个请求计算 100 tokens 前缀
- 总计算量: 100 × 100 = 10,000 tokens

RadixAttention:
- 计算一次 100 tokens 前缀
- 其他 99 个请求重用
- 总计算量: 100 tokens
- 加速比: 10,000 / 100 = 100x（仅前缀部分）
```

## 🔍 代码实现细节

### 1. 前缀匹配算法

```python
def match_prefix(self, key: RadixKey) -> MatchResult:
    """查找最长的匹配前缀"""
    
    # 1. 从根节点开始
    current_node = self.root_node
    
    # 2. 遍历 token_ids
    for token_id in key.token_ids:
        # 3. 查找子节点
        if token_id in current_node.children:
            current_node = current_node.children[token_id]
        else:
            # 4. 没有找到，返回当前匹配结果
            break
    
    # 5. 返回匹配的 KV Cache 索引
    return collect_indices(current_node)
```

### 2. 节点插入

```python
def insert(self, key: RadixKey, value: torch.Tensor):
    """插入新的 KV Cache"""
    
    # 1. 匹配现有前缀
    match_result = self.match_prefix(key)
    
    # 2. 从匹配点继续插入
    current_node = match_result.last_node
    
    # 3. 创建新节点
    for token_id in remaining_tokens:
        new_node = TreeNode(key=token_id)
        current_node.children[token_id] = new_node
        current_node = new_node
    
    # 4. 存储 KV Cache
    current_node.value = value
```

### 3. 缓存淘汰

```python
def evict(self):
    """淘汰最少使用的节点"""
    
    # 1. 使用策略选择要淘汰的节点
    node_to_evict = self.eviction_strategy.select()
    
    # 2. 释放 KV Cache 内存
    self.token_to_kv_pool_allocator.free(node_to_evict.value)
    
    # 3. 从树中删除节点
    self._remove_node(node_to_evict)
```

## ⚙️ 配置和使用

### 1. 默认启用

RadixAttention **默认自动启用**，无需额外配置。

### 2. 禁用 RadixAttention

如果需要禁用（不推荐）：

```python
# 在启动参数中添加
--disable-radix-cache
```

### 3. 配置缓存策略

```python
# 使用 LRU 策略（默认）
eviction_policy: "lru"

# 使用 LFU 策略
eviction_policy: "lfu"
```

### 4. 监控缓存效果

SGLang 提供了缓存统计信息：

```python
# 查看缓存命中率
cache_hit_rate = cache_stats.hits / cache_stats.total

# 查看缓存大小
cache_size = cache_stats.size
```

## 🎓 总结

### RadixAttention 的核心优势

1. ✅ **性能**: 提供 5x 推理加速
2. ✅ **智能**: 自动识别和缓存公共前缀
3. ✅ **高效**: 使用 Radix Tree 优化存储和查找
4. ✅ **灵活**: 支持批内缓存、分页、多种淘汰策略
5. ✅ **易用**: 默认启用，无需配置

### 关键理解

> **RadixAttention = Radix Tree + KV Cache + 前缀匹配**

- **Radix Tree**: 高效存储结构
- **KV Cache**: 缓存的计算结果
- **前缀匹配**: 识别可重用的部分

### 适用场景

- ✅ 多轮对话
- ✅ 批量处理相似请求
- ✅ 使用提示词模板
- ✅ 函数调用场景
- ✅ 长上下文处理

---

## 📚 相关资源

- [SGLang 博客 - RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [SGLang 开发重点与技术创新](./13_02_SGLang_开发重点与技术创新.md)
- [RadixCache 源码](../python/sglang/srt/mem_cache/radix_cache.py)
- [RadixAttention 源码](../python/sglang/srt/layers/radix_attention.py)

