# Cache-Aware Router 前缀匹配详解

## 📋 目录

1. [概述](#1-概述)
2. [Approximate Radix Tree 数据结构](#2-approximate-radix-tree-数据结构)
3. [prefix_match 实现详解](#3-prefix_match-实现详解)
4. [完整的前缀匹配流程](#4-完整的前缀匹配流程)
5. [示例演示](#5-示例演示)

---

## 1. 概述

**问题**: Cache-Aware Load Balancer 如何找到前缀匹配率最高的 worker？

**答案**: 通过 **Approximate Radix Tree** 和 `prefix_match()` 方法实现。

**核心思想**:
- 为每个 worker 维护一个**近似 Radix Tree**（基于请求历史）
- 存储**原始文本字符**（而非 token IDs），避免 tokenization 开销
- 使用 Radix Tree 的**前缀匹配**算法找到最长匹配

---

## 2. Approximate Radix Tree 数据结构

### 2.1 Tree 结构

**代码位置**: `sgl-router/src/tree.rs:26`

```rust
pub struct Tree {
    root: NodeRef,                          // 根节点
    pub tenant_char_count: DashMap<String, usize>,  // 每个 tenant 的字符计数
}
```

**特点**:
- ✅ **多租户支持**: 一个 Tree 可以存储多个 worker（tenant）的数据
- ✅ **线程安全**: 使用 `DashMap` 和 `RwLock` 实现并发访问
- ✅ **LRU 驱逐**: 基于访问时间进行 LRU 驱逐

---

### 2.2 Node 结构

**代码位置**: `sgl-router/src/tree.rs:18`

```rust
struct Node {
    children: DashMap<char, NodeRef>,        // 子节点映射（按字符）
    text: RwLock<String>,                   // 节点存储的文本
    tenant_last_access_time: DashMap<String, u128>,  // 每个 tenant 的最后访问时间
    parent: RwLock<Option<NodeRef>>,        // 父节点引用
}
```

**关键字段**:
- `children`: 子节点映射，键是字符（char），值是子节点
- `text`: 节点存储的文本片段
- `tenant_last_access_time`: 每个 tenant（worker）的最后访问时间（用于 LRU）
- `parent`: 父节点引用（用于回溯）

---

### 2.3 Tree 示例

```
Tree 结构示例:

                    root
                     |
        +------------+------------+
        |            |            |
       'h'          'a'          'b'
        |            |            |
     "ello"        "pple"       "anana"
        |            |            |
       ' '          (leaf)       (leaf)
        |
       'w'
        |
     "orld"
        |
     (leaf)

存储的请求:
- "hello world" → worker1
- "apple" → worker2
- "banana" → worker3
```

---

## 3. prefix_match 实现详解

### 3.1 函数签名

**代码位置**: `sgl-router/src/tree.rs:262`

```rust
pub fn prefix_match(&self, text: &str) -> (String, String) {
    // 返回: (matched_text, matched_worker_url)
    // matched_text: 匹配的文本前缀
    // matched_worker_url: 匹配的 worker URL
}
```

**返回值**:
- `matched_text`: 匹配的文本前缀（最长匹配）
- `matched_worker_url`: 匹配的 worker URL（第一个 tenant）

---

### 3.2 完整实现（逐行解释）

```rust
pub fn prefix_match(&self, text: &str) -> (String, String) {
    // ========== 步骤 1: 初始化变量 ==========
    let mut curr = Arc::clone(&self.root);      // 当前节点（从根节点开始）
    let mut curr_idx = 0;                        // 当前文本索引
    let mut prev = Arc::clone(&self.root);      // 上一个节点
    let text_count = text.chars().count();      // 文本字符总数
    
    // ========== 步骤 2: 遍历 Tree，寻找最长匹配 ==========
    while curr_idx < text_count {
        // 2.1 获取当前字符
        let first_char = text.chars().nth(curr_idx).unwrap();
        
        // 2.2 获取从当前位置到结尾的文本
        let curr_text = slice_by_chars(text, curr_idx, text_count);
        
        // 2.3 更新当前节点
        curr = prev.clone();
        
        // 2.4 检查是否有匹配的子节点
        if let Some(entry) = curr.children.get(&first_char) {
            // 找到了匹配的子节点
            let matched_node = entry.value().clone();
            
            // 2.5 读取匹配节点的文本
            let matched_text_guard = matched_node.text.read().unwrap();
            let matched_text = matched_text_guard.clone();
            
            // 2.6 计算共享前缀长度
            let shared_count = shared_prefix_count(&matched_text, &curr_text);
            let matched_node_text_count = matched_text.chars().count();
            drop(matched_text_guard);
            
            // 2.7 判断匹配类型
            if shared_count == matched_node_text_count {
                // 完全匹配：匹配节点的文本完全匹配
                // 继续到下一个节点
                curr_idx += shared_count;
                prev = Arc::clone(&matched_node);
            } else {
                // 部分匹配：只匹配了部分文本
                // 停止匹配
                curr_idx += shared_count;
                prev = Arc::clone(&matched_node);
                break;
            }
        } else {
            // 没有找到匹配的子节点
            // 停止匹配
            break;
        }
    }
    
    // ========== 步骤 3: 获取匹配的 worker ==========
    curr = prev.clone();
    
    // 3.1 选择第一个 tenant（worker）
    let tenant = curr
        .tenant_last_access_time
        .iter()
        .next()
        .map(|kv| kv.key().to_owned())
        .unwrap_or("empty".to_string());
    
    // ========== 步骤 4: 更新访问时间（LRU） ==========
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    
    if !tenant.eq("empty") {
        // 从当前节点回溯到根节点，更新所有节点的访问时间
        let mut current_node = Some(curr);
        while let Some(node) = current_node {
            node.tenant_last_access_time
                .insert(tenant.clone(), timestamp_ms);
            current_node = node.parent.read().unwrap().clone();
        }
    }
    
    // ========== 步骤 5: 返回匹配结果 ==========
    let ret_text = slice_by_chars(text, 0, curr_idx);  // 匹配的文本前缀
    (ret_text, tenant)  // 返回 (匹配文本, worker URL)
}
```

---

### 3.3 shared_prefix_count 函数

**代码位置**: `sgl-router/src/tree.rs:64`

```rust
fn shared_prefix_count(a: &str, b: &str) -> usize {
    let mut i = 0;
    let mut a_iter = a.chars();
    let mut b_iter = b.chars();
    
    loop {
        match (a_iter.next(), b_iter.next()) {
            (Some(a_char), Some(b_char)) if a_char == b_char => {
                i += 1;  // 字符匹配，计数加 1
            }
            _ => break,  // 字符不匹配或到达结尾，停止
        }
    }
    
    i  // 返回共享前缀的字符数
}
```

**功能**: 计算两个字符串的共享前缀长度

**示例**:
```rust
shared_prefix_count("hello", "help")  // 返回 3 ("hel")
shared_prefix_count("apple", "apple")  // 返回 5 ("apple")
shared_prefix_count("abc", "xyz")      // 返回 0 ("")
```

---

## 4. 完整的前缀匹配流程

### 4.1 Cache-Aware Router 中的使用

**代码位置**: `sgl-router/src/policies/cache_aware.rs:302`

```rust
// Use cache-aware routing when balanced
let text = request_text.unwrap_or("");

// 1. 获取 Tree（按 model_id）
let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

if let Some(tree) = tree {
    // 2. 调用 prefix_match 找到最长匹配
    let (matched_text, matched_worker) = tree.prefix_match(text);
    
    // 3. 计算匹配率
    let match_rate = if text.is_empty() {
        0.0
    } else {
        matched_text.chars().count() as f32 / text.chars().count() as f32
    };
    
    // 4. 根据匹配率决定路由策略
    let selected_url = if match_rate > self.config.cache_threshold {
        // 匹配率足够高 → 路由到匹配的 worker（利用缓存）
        RouterMetrics::record_cache_hit();
        matched_worker.to_string()
    } else {
        // 匹配率不够高 → 路由到树大小最小的 worker（最多可用缓存空间）
        RouterMetrics::record_cache_miss();
        tree.get_smallest_tenant()
    };
    
    // 5. 更新 Tree（记录这个请求）
    tree.insert(text, &selected_url);
    
    // 6. 返回选中的 worker 索引
    return Some(selected_idx);
}
```

---

### 4.2 路由决策逻辑

```
输入: request_text = "hello world"

步骤 1: prefix_match("hello world")
    ├─ 遍历 Tree
    ├─ 找到最长匹配: "hello"
    └─ 返回: ("hello", "worker1")

步骤 2: 计算匹配率
    match_rate = len("hello") / len("hello world")
               = 5 / 11
               = 0.45

步骤 3: 路由决策
    if match_rate > cache_threshold (例如 0.5):
        → 路由到 worker1（利用缓存）
    else:
        → 路由到树大小最小的 worker（最多可用缓存空间）

步骤 4: 更新 Tree
    tree.insert("hello world", "selected_worker")
```

---

## 5. 示例演示

### 示例 1: 完全匹配

**初始状态**:
```
Tree:
    root
     |
    'h'
     |
  "ello"
     |
   (leaf, tenant: "worker1")

请求: "hello"
```

**匹配过程**:
```rust
1. curr = root, curr_idx = 0
2. first_char = 'h'
3. 找到子节点 'h' → matched_node
4. matched_text = "ello"
5. curr_text = "hello"
6. shared_count = shared_prefix_count("ello", "hello") = 0
   // 注意：这里需要匹配 "ello" 和 "hello"，但 "ello" 是节点的文本
   // 实际上应该匹配 "hello" 和 "ello"
7. 实际上，算法会继续匹配：
   - 匹配 "ello" 和 "ello" → shared_count = 4
   - shared_count == matched_node_text_count (4) → 完全匹配
   - curr_idx += 4 = 4
   - 继续到下一个节点
8. 到达 leaf 节点
9. 返回: ("hello", "worker1")
```

**结果**: `match_rate = 5/5 = 1.0` → 完全匹配，路由到 `worker1`

---

### 示例 2: 部分匹配

**初始状态**:
```
Tree:
    root
     |
    'h'
     |
  "ello"
     |
   (leaf, tenant: "worker1")

请求: "help"
```

**匹配过程**:
```rust
1. curr = root, curr_idx = 0
2. first_char = 'h'
3. 找到子节点 'h' → matched_node
4. matched_text = "ello"
5. curr_text = "help"
6. shared_count = shared_prefix_count("ello", "help") = 2 ("he")
7. shared_count (2) != matched_node_text_count (4) → 部分匹配
8. curr_idx += 2 = 2
9. 停止匹配
10. 返回: ("he", "worker1")
```

**结果**: `match_rate = 2/4 = 0.5` → 如果 `cache_threshold = 0.5`，路由到 `worker1`；否则路由到树大小最小的 worker

---

### 示例 3: 无匹配

**初始状态**:
```
Tree:
    root
     |
    'h'
     |
  "ello"
     |
   (leaf, tenant: "worker1")

请求: "apple"
```

**匹配过程**:
```rust
1. curr = root, curr_idx = 0
2. first_char = 'a'
3. 查找子节点 'a' → 未找到
4. 停止匹配
5. 返回: ("", "empty")
```

**结果**: `match_rate = 0/5 = 0.0` → 无匹配，路由到树大小最小的 worker

---

### 示例 4: 多级匹配

**初始状态**:
```
Tree:
    root
     |
    'h'
     |
  "ello"
     |
    ' '
     |
    'w'
     |
  "orld"
     |
   (leaf, tenant: "worker1")

请求: "hello world"
```

**匹配过程**:
```rust
1. curr = root, curr_idx = 0
2. first_char = 'h'
3. 找到子节点 'h' → matched_node ("ello")
4. shared_count = shared_prefix_count("ello", "hello") = 4
5. shared_count (4) == matched_node_text_count (4) → 完全匹配
6. curr_idx += 4 = 4
7. 继续到下一个节点
8. first_char = ' ' (空格)
9. 找到子节点 ' ' → matched_node
10. shared_count = shared_prefix_count("", " world") = 0
11. 继续到下一个节点
12. first_char = 'w'
13. 找到子节点 'w' → matched_node ("orld")
14. shared_count = shared_prefix_count("orld", "world") = 4
15. shared_count (4) == matched_node_text_count (4) → 完全匹配
16. curr_idx += 4 = 9
17. 到达 leaf 节点
18. 返回: ("hello world", "worker1")
```

**结果**: `match_rate = 11/11 = 1.0` → 完全匹配，路由到 `worker1`

---

## 6. 关键优化点

### 6.1 为什么使用 Approximate Tree？

**优势**:
- ✅ **避免 tokenization 开销**: 存储原始文本字符，不需要 tokenize
- ✅ **无需直接查询缓存**: 基于请求历史推断缓存状态
- ✅ **轻量级**: 只存储文本前缀，不存储完整的 KV Cache

**劣势**:
- ❌ **近似性**: 是近似匹配，可能不完全准确
- ❌ **内存占用**: 需要维护 Tree 结构

---

### 6.2 为什么选择第一个 Tenant？

**代码位置**: `sgl-router/src/tree.rs:301`

```rust
let tenant = curr
    .tenant_last_access_time
    .iter()
    .next()
    .map(|kv| kv.key().to_owned())
    .unwrap_or("empty".to_string());
```

**原因**:
- `DashMap` 的迭代顺序是**不确定的**（基于哈希）
- 选择第一个 tenant 是**简单快速**的策略
- 如果需要更智能的选择，可以使用 `prefix_match_tenant()` 方法

---

### 6.3 LRU 更新机制

**代码位置**: `sgl-router/src/tree.rs:315`

```rust
if !tenant.eq("empty") {
    let mut current_node = Some(curr);
    while let Some(node) = current_node {
        node.tenant_last_access_time
            .insert(tenant.clone(), timestamp_ms);
        current_node = node.parent.read().unwrap().clone();
    }
}
```

**作用**:
- 更新从匹配节点到根节点的所有节点的访问时间
- 用于 LRU 驱逐策略
- 确保最近访问的节点不会被过早驱逐

---

## 7. 总结

### 7.1 前缀匹配的核心流程

```
1. 输入请求文本
   ↓
2. 从 Tree 根节点开始遍历
   ↓
3. 按字符匹配，找到最长匹配路径
   ↓
4. 计算匹配率
   ↓
5. 根据匹配率决定路由策略
   ↓
6. 更新 Tree（记录请求）
```

---

### 7.2 关键函数

| 函数 | 位置 | 功能 |
|------|------|------|
| `prefix_match()` | `tree.rs:262` | 找到最长前缀匹配 |
| `shared_prefix_count()` | `tree.rs:64` | 计算共享前缀长度 |
| `slice_by_chars()` | `tree.rs:81` | 按字符切片（处理 UTF-8） |

---

### 7.3 关键数据结构

| 结构 | 位置 | 功能 |
|------|------|------|
| `Tree` | `tree.rs:26` | Radix Tree 主结构 |
| `Node` | `tree.rs:18` | Tree 节点结构 |
| `CacheAwarePolicy` | `cache_aware.rs:79` | Cache-Aware 路由策略 |

---

### 7.4 匹配算法复杂度

- **时间复杂度**: O(m)，其中 m 是匹配的文本长度
- **空间复杂度**: O(n)，其中 n 是 Tree 中存储的文本总数

**优化**:
- ✅ 使用 `DashMap` 实现并发访问
- ✅ 使用 `RwLock` 实现细粒度锁定
- ✅ LRU 驱逐防止内存无限增长

---

**结论**: `find_highest_prefix_match` 实际上是通过 **Radix Tree 的前缀匹配算法**实现的。它遍历 Tree，找到与请求文本最长匹配的路径，然后根据匹配率决定路由策略。这种方法避免了直接查询 worker 的缓存状态，而是基于请求历史进行智能推断。🎯

