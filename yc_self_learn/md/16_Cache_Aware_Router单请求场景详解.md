# Cache-Aware Router 单请求场景详解

## 📋 目录

1. [问题：单请求时是否使用 Cache-Aware Router？](#1-问题单请求时是否使用-cache-aware-router)
2. [单请求场景的完整流程](#2-单请求场景的完整流程)
3. [不同阶段的处理逻辑](#3-不同阶段的处理逻辑)
4. [实际示例](#4-实际示例)

---

## 1. 问题：单请求时是否使用 Cache-Aware Router？

### 1.1 核心答案

**答案**: **是的，每次请求都会使用 Cache-Aware Router**，不管是一个请求还是多个请求。

**关键点**:
- ✅ Cache-Aware Router 是**每次请求都会调用的路由策略**
- ✅ 不是只在多个请求并发时使用
- ✅ 单请求时也会调用，只是匹配率可能是 0

---

### 1.2 调用位置

**代码位置**: `sgl-router/src/routers/http/router.rs:171`

```rust
let worker = match self.select_worker_for_model(model_id, Some(&text)) {
    Some(w) => w,
    None => {
        // 没有可用 worker
        return error_response();
    }
};
```

**每次请求都会调用**:
- 不管是第一个请求还是第 N 个请求
- 不管是单个请求还是多个并发请求
- 都会调用 `select_worker_for_model()` → `policy.select_worker()`

---

## 2. 单请求场景的完整流程

### 2.1 第一次请求（Tree 不存在）

**场景**: 发送第一个请求 "tell me what is sglang"

**代码位置**: `sgl-router/src/policies/cache_aware.rs:298`

```rust
// Use cache-aware routing when balanced
let text = request_text.unwrap_or("");  // "tell me what is sglang"

// Get the tree reference
let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

if let Some(tree) = tree {
    // Tree 存在，使用 cache-aware 路由
    ...
} else {
    // Tree 不存在（第一次请求）
    debug!(
        "Warning: No tree found for model '{}', using random worker selection",
        model_id
    );
    // 返回一个随机的健康 worker
    let mut rng = rand::rng();
    let random_idx = rng.random_range(0..healthy_indices.len());
    Some(healthy_indices[random_idx])
}
```

**流程**:
```
1. 请求到达: "tell me what is sglang"
   ↓
2. 调用 select_worker()
   ↓
3. 查找 Tree: self.trees.get(model_id)
   ↓
4. Tree 不存在 → 返回 None
   ↓
5. 进入 else 分支
   ↓
6. 随机选择一个健康的 worker
   ↓
7. 返回 worker 索引
```

**结果**: 
- ✅ 路由到一个随机的 worker
- ❌ 没有使用 cache-aware 逻辑（因为 Tree 不存在）
- ❌ 但是 Tree 会在 `init_workers()` 时初始化

---

### 2.2 Tree 存在但为空（第一次使用该模型）

**场景**: Tree 已初始化，但还没有任何请求历史

**代码位置**: `sgl-router/src/policies/cache_aware.rs:300`

```rust
if let Some(tree) = tree {
    // Tree 存在
    let (matched_text, matched_worker) = tree.prefix_match(text);
    // matched_text = "" (空字符串，因为没有匹配)
    // matched_worker = "empty" (因为没有 tenant)
    
    let match_rate = if text.is_empty() {
        0.0
    } else {
        matched_text.chars().count() as f32 / text.chars().count() as f32
        // = 0 / text_length = 0.0
    };
    
    let selected_url = if match_rate > self.config.cache_threshold {
        // match_rate = 0.0 < cache_threshold → false
        RouterMetrics::record_cache_hit();
        matched_worker.to_string()
    } else {
        // 进入这个分支
        RouterMetrics::record_cache_miss();
        tree.get_smallest_tenant()  // ← 选择树大小最小的 worker
    };
    
    // 更新 Tree（记录这个请求）
    tree.insert(text, &selected_url);  // ← 插入请求到 Tree
}
```

**流程**:
```
1. 请求到达: "tell me what is sglang"
   ↓
2. 调用 select_worker()
   ↓
3. 查找 Tree: self.trees.get(model_id) → 找到 Tree
   ↓
4. 调用 tree.prefix_match("tell me what is sglang")
   ↓
5. Tree 为空 → 没有匹配
   ↓
6. 返回: ("", "empty")
   ↓
7. 计算匹配率: match_rate = 0 / text_length = 0.0
   ↓
8. match_rate (0.0) < cache_threshold → 进入 else 分支
   ↓
9. 调用 tree.get_smallest_tenant() → 选择树大小最小的 worker
   ↓
10. 调用 tree.insert(text, selected_url) → 插入请求到 Tree
   ↓
11. 返回选中的 worker
```

**结果**:
- ✅ 使用了 cache-aware 逻辑（虽然匹配率是 0）
- ✅ 选择了树大小最小的 worker（最多可用缓存空间）
- ✅ 把请求插入到 Tree 中，供后续请求使用

---

### 2.3 第二次请求（有请求历史）

**场景**: 发送第二个请求 "tell me what is sglang and how to use it"

**流程**:
```
1. 请求到达: "tell me what is sglang and how to use it"
   ↓
2. 调用 select_worker()
   ↓
3. 查找 Tree: self.trees.get(model_id) → 找到 Tree
   ↓
4. 调用 tree.prefix_match("tell me what is sglang and how to use it")
   ↓
5. Tree 中有 "tell me what is sglang" → 匹配！
   ↓
6. 返回: ("tell me what is sglang", "worker1")
   ↓
7. 计算匹配率: 
   match_rate = len("tell me what is sglang") / len("tell me what is sglang and how to use it")
              = 25 / 45
              = 0.56
   ↓
8. match_rate (0.56) > cache_threshold (例如 0.5) → 进入 if 分支
   ↓
9. 路由到 worker1（利用缓存）
   ↓
10. 调用 tree.insert(text, "worker1") → 更新 Tree
   ↓
11. 返回 worker1
```

**结果**:
- ✅ 使用了 cache-aware 逻辑
- ✅ 匹配率 > cache_threshold → 路由到有缓存的 worker
- ✅ 利用了 RadixAttention 缓存，提高性能

---

## 3. 不同阶段的处理逻辑

### 3.1 阶段 1: Tree 不存在（第一次请求）

**代码**: `cache_aware.rs:338`

```rust
} else {
    // No tree for this model
    debug!("Warning: No tree found for model '{}', using random worker selection", model_id);
    // 随机选择
    let mut rng = rand::rng();
    let random_idx = rng.random_range(0..healthy_indices.len());
    Some(healthy_indices[random_idx])
}
```

**行为**:
- ❌ 不使用 cache-aware 逻辑
- ✅ 随机选择一个健康的 worker
- ✅ Tree 会在 `init_workers()` 时初始化

---

### 3.2 阶段 2: Tree 存在但为空（第一次使用该模型）

**代码**: `cache_aware.rs:313`

```rust
} else {
    RouterMetrics::record_cache_miss();
    tree.get_smallest_tenant()  // 选择树大小最小的 worker
}
```

**行为**:
- ✅ 使用 cache-aware 逻辑
- ✅ 匹配率 = 0.0（Tree 为空）
- ✅ 选择树大小最小的 worker（最多可用缓存空间）
- ✅ 插入请求到 Tree

---

### 3.3 阶段 3: Tree 有数据但匹配率低

**代码**: `cache_aware.rs:313`

```rust
} else {
    RouterMetrics::record_cache_miss();
    tree.get_smallest_tenant()  // 选择树大小最小的 worker
}
```

**行为**:
- ✅ 使用 cache-aware 逻辑
- ✅ 匹配率 < cache_threshold（例如 0.3 < 0.5）
- ✅ 选择树大小最小的 worker（最多可用缓存空间）
- ✅ 插入请求到 Tree

---

### 3.4 阶段 4: Tree 有数据且匹配率高

**代码**: `cache_aware.rs:309`

```rust
if match_rate > self.config.cache_threshold {
    RouterMetrics::record_cache_hit();
    matched_worker.to_string()  // 路由到匹配的 worker
}
```

**行为**:
- ✅ 使用 cache-aware 逻辑
- ✅ 匹配率 > cache_threshold（例如 0.6 > 0.5）
- ✅ 路由到匹配的 worker（利用缓存）
- ✅ 插入请求到 Tree

---

## 4. 实际示例

### 示例 1: 第一个请求

**请求**: "tell me what is sglang"

**假设**:
- Tree 已初始化但为空
- cache_threshold = 0.5
- 有 3 个 worker: worker1, worker2, worker3

**流程**:
```
1. prefix_match("tell me what is sglang")
   → 返回: ("", "empty")
   → match_rate = 0 / 25 = 0.0

2. match_rate (0.0) < cache_threshold (0.5)
   → 进入 else 分支

3. tree.get_smallest_tenant()
   → 假设 worker1 的树最小
   → 返回: "worker1"

4. tree.insert("tell me what is sglang", "worker1")
   → 插入到 Tree

5. 返回: worker1
```

**结果**:
- ✅ 路由到 worker1
- ✅ Tree 中记录了 "tell me what is sglang" → worker1

---

### 示例 2: 第二个请求（有前缀匹配）

**请求**: "tell me what is sglang and how to use it"

**假设**:
- Tree 中有 "tell me what is sglang" → worker1
- cache_threshold = 0.5

**流程**:
```
1. prefix_match("tell me what is sglang and how to use it")
   → 匹配到 "tell me what is sglang"
   → 返回: ("tell me what is sglang", "worker1")
   → match_rate = 25 / 45 = 0.56

2. match_rate (0.56) > cache_threshold (0.5)
   → 进入 if 分支

3. 返回: "worker1"

4. tree.insert("tell me what is sglang and how to use it", "worker1")
   → 更新 Tree

5. 返回: worker1
```

**结果**:
- ✅ 路由到 worker1（利用缓存）
- ✅ 匹配率 0.56 > 0.5 → Cache Hit
- ✅ 可以利用 RadixAttention 缓存，提高性能

---

### 示例 3: 第二个请求（无前缀匹配）

**请求**: "what is python"

**假设**:
- Tree 中有 "tell me what is sglang" → worker1
- cache_threshold = 0.5

**流程**:
```
1. prefix_match("what is python")
   → 没有匹配（"what is python" 和 "tell me what is sglang" 前缀不同）
   → 返回: ("", "empty")
   → match_rate = 0 / 13 = 0.0

2. match_rate (0.0) < cache_threshold (0.5)
   → 进入 else 分支

3. tree.get_smallest_tenant()
   → 假设 worker2 的树最小
   → 返回: "worker2"

4. tree.insert("what is python", "worker2")
   → 插入到 Tree

5. 返回: worker2
```

**结果**:
- ✅ 路由到 worker2（树大小最小）
- ✅ 匹配率 0.0 < 0.5 → Cache Miss
- ✅ Tree 中记录了 "what is python" → worker2

---

## 5. 总结

### 5.1 核心答案

**问题**: 如果只发一个请求，是否用不到 Cache-Aware Router？

**答案**: **不是的，每次请求都会使用 Cache-Aware Router**。

**原因**:
1. ✅ Cache-Aware Router 是**每次请求都会调用的路由策略**
2. ✅ 单请求时也会调用，只是匹配率可能是 0
3. ✅ 即使匹配率是 0，也会使用 `get_smallest_tenant()` 选择 worker
4. ✅ 会把请求插入到 Tree 中，供后续请求使用

---

### 5.2 不同场景的处理

| 场景 | Tree 状态 | 匹配率 | 路由策略 | 是否使用 Cache-Aware |
|------|----------|--------|---------|---------------------|
| 第一次请求 | 不存在 | N/A | 随机选择 | ❌ 不使用 |
| Tree 为空 | 存在但为空 | 0.0 | 选择树最小的 worker | ✅ 使用 |
| 无匹配 | 有数据但无匹配 | 0.0 | 选择树最小的 worker | ✅ 使用 |
| 匹配率低 | 有数据但匹配率 < threshold | < threshold | 选择树最小的 worker | ✅ 使用 |
| 匹配率高 | 有数据且匹配率 > threshold | > threshold | 路由到匹配的 worker | ✅ 使用 |

---

### 5.3 关键代码位置

| 场景 | 代码位置 | 关键逻辑 |
|------|---------|---------|
| Tree 不存在 | `cache_aware.rs:338` | 随机选择 worker |
| Tree 为空/无匹配 | `cache_aware.rs:313` | `tree.get_smallest_tenant()` |
| 匹配率高 | `cache_aware.rs:309` | 路由到匹配的 worker |
| 插入 Tree | `cache_aware.rs:322` | `tree.insert(text, url)` |

---

### 5.4 实际效果

**单请求场景**:
- ✅ 仍然会调用 Cache-Aware Router
- ✅ 会选择合适的 worker（随机或树最小的）
- ✅ 会把请求记录到 Tree 中
- ✅ 为后续请求建立缓存基础

**多请求场景**:
- ✅ 利用 Tree 中的历史数据
- ✅ 提高缓存命中率
- ✅ 提高整体性能

---

**结论**: Cache-Aware Router **每次请求都会使用**，不管是一个请求还是多个请求。单请求时虽然匹配率可能是 0，但仍然会使用 cache-aware 逻辑选择 worker，并把请求记录到 Tree 中，为后续请求建立缓存基础。🎯

