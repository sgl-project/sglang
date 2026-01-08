# RadixCache insert 详解与 Router 架构

## 📋 目录

1. [RadixCache.insert() 详解](#1-radixcacheinsert-详解)
2. [Router 是什么？](#2-router-是什么)
3. [Router 架构详解](#3-router-架构详解)
4. [本地 vs 云端部署](#4-本地-vs-云端部署)
5. [传输时间优化](#5-传输时间优化)

---

## 1. RadixCache.insert() 详解

### 1.1 函数签名

**代码位置**: `python/sglang/srt/mem_cache/radix_cache.py:302`

```python
def insert(self, key: RadixKey, value=None, chunked=False):
    """
    插入 KV Cache 到 Radix Tree
    
    Args:
        key: RadixKey，包含 token_ids 和 extra_key
        value: KV Cache 索引（torch.Tensor），如果为 None，使用 token_ids
        chunked: 是否为 chunked prefill
    """
```

---

### 1.2 完整实现（逐行解释）

```python
def insert(self, key: RadixKey, value=None, chunked=False):
    # ========== 步骤 1: 检查是否禁用 ==========
    if self.disable:
        return 0  # 如果禁用缓存，直接返回
    
    # ========== 步骤 2: 转换 Key ==========
    key.token_ids = self.key_convert_fn(key.token_ids)
    # key_convert_fn: 对于 EAGLE，转换为 bigram key
    # 对于普通模式，保持不变
    
    # ========== 步骤 3: 准备 Value ==========
    if value is None:
        # 如果没有提供 value，使用 token_ids 作为 value
        value = torch.tensor(key.token_ids, dtype=torch.int64)
    
    # ========== 步骤 4: EAGLE 特殊处理 ==========
    if self.is_eagle:
        # EAGLE 使用 bigram key，value 长度需要匹配
        value = value[: len(key)]
    
    # ========== 步骤 5: 调用内部插入函数 ==========
    return self._insert_helper(self.root_node, key, value)
```

---

### 1.3 _insert_helper() 核心逻辑

**代码位置**: `python/sglang/srt/mem_cache/radix_cache.py:450`（推测）

**核心流程**:
```python
def _insert_helper(self, node: TreeNode, key: RadixKey, value: torch.Tensor):
    """
    递归插入到 Radix Tree
    
    流程:
    1. 从根节点开始
    2. 查找匹配的子节点
    3. 如果完全匹配，更新节点
    4. 如果部分匹配，分裂节点
    5. 如果没有匹配，创建新节点
    """
    
    curr = node
    curr_idx = 0
    key_token_ids = key.token_ids
    
    while curr_idx < len(key_token_ids):
        # 1. 获取当前字符（token）
        first_token = key_token_ids[curr_idx]
        
        # 2. 查找匹配的子节点
        if first_token in curr.children:
            # 找到匹配的子节点
            child_node = curr.children[first_token]
            child_key = child_node.key.token_ids
            
            # 3. 计算共享前缀长度
            shared_len = self._shared_prefix_length(
                key_token_ids[curr_idx:],
                child_key
            )
            
            if shared_len == len(child_key):
                # 4. 完全匹配：继续到子节点
                curr_idx += shared_len
                curr = child_node
            else:
                # 5. 部分匹配：需要分裂节点
                # 分裂 child_node 为两个节点：
                # - 共享前缀部分（保留）
                # - 不同部分（新建）
                self._split_node(child_node, shared_len)
                curr_idx += shared_len
                break
        else:
            # 6. 没有匹配：创建新节点
            new_node = TreeNode(
                key=RadixKey(token_ids=key_token_ids[curr_idx:], extra_key=key.extra_key),
                value=value[curr_idx:],
                parent=curr
            )
            curr.children[first_token] = new_node
            break
    
    # 7. 更新节点的 value（KV Cache 索引）
    if curr_idx == len(key_token_ids):
        # 完全匹配到现有节点
        curr.value = value
    else:
        # 部分匹配或新建节点
        # value 已经设置好了
    
    # 8. 更新引用计数（防止被驱逐）
    self.inc_lock_ref(curr)
    
    return len(key_token_ids)  # 返回插入的 token 数量
```

---

### 1.4 插入示例

**示例 1: 插入新节点**

```
初始 Tree:
    root
     |
    (empty)

插入: "hello" → [KV Cache indices: 0,1,2,3,4]

结果:
    root
     |
    'h'
     |
  "ello"
     |
  [KV Cache: 0,1,2,3,4]
```

**示例 2: 插入共享前缀**

```
初始 Tree:
    root
     |
    'h'
     |
  "ello"
     |
  [KV Cache: 0,1,2,3,4]

插入: "help" → [KV Cache indices: 0,1,2,5]

步骤:
1. 匹配 'h' → 找到子节点
2. 匹配 "ello" vs "elp" → 共享前缀 "el" (长度 2)
3. 分裂节点:
   - "el" (共享部分)
   - "lo" (原节点剩余部分)
   - "p" (新节点)

结果:
    root
     |
    'h'
     |
    "el"
     ├─ "lo" [KV Cache: 0,1,2,3,4]
     └─ "p" [KV Cache: 0,1,2,5]
```

---

### 1.5 节点分裂（Split Node）

**场景**: 部分匹配时需要分裂节点

**代码逻辑**:
```python
def _split_node(self, node: TreeNode, split_pos: int):
    """
    分裂节点
    
    原节点: "ello" [KV Cache: 0,1,2,3,4]
    分裂位置: 2 (共享前缀 "el")
    
    结果:
    - 父节点: "el" [KV Cache: 0,1]
    - 子节点1: "lo" [KV Cache: 2,3,4] (原节点剩余部分)
    """
    
    # 1. 创建新的父节点（共享前缀）
    new_parent = TreeNode(
        key=RadixKey(token_ids=node.key.token_ids[:split_pos]),
        value=node.value[:split_pos],
        parent=node.parent
    )
    
    # 2. 更新原节点（剩余部分）
    node.key.token_ids = node.key.token_ids[split_pos:]
    node.value = node.value[split_pos:]
    node.parent = new_parent
    
    # 3. 将原节点移动到新父节点下
    new_parent.children[node.key.token_ids[0]] = node
    
    # 4. 更新原父节点的引用
    old_parent = node.parent
    old_parent.children[new_parent.key.token_ids[0]] = new_parent
```

---

## 2. Router 是什么？

### 2.1 核心答案

**问题**: 什么是 Router？是本地 GPU 到 router，不发送给云端？这样省略传输时间？

**答案**: 
- ✅ **Router 是一个独立的负载均衡服务**，用于在多个 SGLang Worker 之间路由请求
- ✅ **可以是本地部署**（同一台机器或同一集群），也可以是分布式部署
- ✅ **本地部署时**，确实可以省略网络传输时间（使用 localhost 或内网通信）

---

### 2.2 Router 的定义

**代码位置**: `sgl-router/README.md:1`

```
SGLang router is a standalone Rust module that enables data parallelism 
across SGLang instances, providing high-performance request routing and 
advanced load balancing.
```

**作用**:
- ✅ **数据并行**: 在多个 SGLang Worker 之间分发请求
- ✅ **负载均衡**: 智能路由请求到最合适的 Worker
- ✅ **缓存感知**: Cache-Aware Router 优化缓存利用率

---

### 2.3 Router vs Worker

```
┌─────────────────────────────────────────────────────────┐
│ Client（客户端）                                         │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP Request
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Router（路由服务）                                       │
│  - 接收客户端请求                                        │
│  - 选择最合适的 Worker                                  │
│  - 转发请求到 Worker                                    │
│  - 返回响应给客户端                                     │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP Request (转发)
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Worker 1（SGLang Runtime）                              │
│  - 运行模型推理                                          │
│  - 管理 GPU 资源                                        │
│  - 处理 KV Cache                                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Worker 2（SGLang Runtime）                              │
│  - 运行模型推理                                          │
│  - 管理 GPU 资源                                        │
│  - 处理 KV Cache                                        │
└─────────────────────────────────────────────────────────┘

... (更多 Workers)
```

---

## 3. Router 架构详解

### 3.1 部署模式

#### a) **Co-launch Mode（同机部署）**

**代码位置**: `sgl-router/py_src/sglang_router/launch_server.py:145`

```python
def main():
    # 1. 启动多个 Worker 进程
    for i, worker_port in enumerate(worker_ports):
        proc = launch_server_process(server_args, worker_port, i)
        server_processes.append(proc)
    
    # 2. 启动 Router（在同一台机器上）
    router_args.worker_urls = [
        f"http://{server_args.host}:{port}" for port in worker_ports
    ]
    launch_router(router_args)
```

**架构**:
```
同一台机器:
    ├─ Router (localhost:30000)
    ├─ Worker 1 (localhost:8000)
    ├─ Worker 2 (localhost:8001)
    └─ Worker 3 (localhost:8002)

通信: localhost（本地回环，几乎零延迟）
```

**优势**:
- ✅ **零网络延迟**: localhost 通信，延迟 < 0.1ms
- ✅ **高带宽**: 本地内存带宽，不受网络限制
- ✅ **简单部署**: 一键启动所有服务

---

#### b) **Separate Launch Mode（分布式部署）**

**代码位置**: `sgl-router/README.md:75`

```bash
# Worker 1 (机器 A)
python -m sglang.srt.server --port 8000

# Worker 2 (机器 B)
python -m sglang.srt.server --port 8000

# Router (机器 C)
python -m sglang_router.launch_router \
    --worker-urls http://machine-a:8000 http://machine-b:8000
```

**架构**:
```
机器 A:
    └─ Worker 1 (machine-a:8000)

机器 B:
    └─ Worker 2 (machine-b:8000)

机器 C:
    └─ Router (machine-c:30000)
        ├─ 连接到 Worker 1 (内网)
        └─ 连接到 Worker 2 (内网)

通信: 内网（通常 1-10ms 延迟）
```

**优势**:
- ✅ **横向扩展**: 可以添加更多 Worker
- ✅ **资源隔离**: Worker 独立部署
- ✅ **灵活配置**: 可以独立扩展 Router 和 Worker

---

### 3.2 Router 的工作流程

**代码位置**: `sgl-router/src/routers/http/router.rs:156`

```rust
pub async fn route_typed_request<T>(&self, ...) -> Response {
    let start = Instant::now();
    let text = typed_req.extract_text_for_routing();
    
    // 1. 选择 Worker
    let worker = match self.select_worker_for_model(model_id, Some(&text)) {
        Some(w) => w,
        None => return error_response(),
    };
    
    // 2. 转发请求到 Worker
    let response = self.client
        .post(worker.url())
        .json(&typed_req)
        .send()
        .await?;
    
    // 3. 返回响应给客户端
    return response;
}
```

**完整流程**:
```
1. 客户端发送请求到 Router
   ↓
2. Router 提取请求文本（用于路由决策）
   ↓
3. Router 调用 select_worker()（Cache-Aware Router）
   ├─ prefix_match() 查找匹配的 Worker
   ├─ 计算匹配率
   └─ 选择最合适的 Worker
   ↓
4. Router 转发请求到选中的 Worker
   ├─ HTTP POST 请求
   └─ 包含完整的请求数据
   ↓
5. Worker 处理请求（GPU 推理）
   ↓
6. Worker 返回响应给 Router
   ↓
7. Router 返回响应给客户端
```

---

## 4. 本地 vs 云端部署

### 4.1 本地部署（Localhost）

**场景**: Router 和 Worker 在同一台机器上

```
┌─────────────────────────────────────────┐
│ 本地机器（localhost）                    │
│                                          │
│  ┌──────────────┐                       │
│  │   Router     │                       │
│  │ :30000       │                       │
│  └──────┬───────┘                       │
│         │ localhost (0.1ms)            │
│         ↓                               │
│  ┌──────────────┐                       │
│  │   Worker 1   │                       │
│  │ :8000        │                       │
│  │ GPU 0        │                       │
│  └──────────────┘                       │
│                                          │
│  ┌──────────────┐                       │
│  │   Worker 2   │                       │
│  │ :8001        │                       │
│  │ GPU 1        │                       │
│  └──────────────┘                       │
└─────────────────────────────────────────┘
```

**通信方式**:
- ✅ **localhost**: 本地回环接口
- ✅ **延迟**: < 0.1ms（几乎零延迟）
- ✅ **带宽**: 不受网络限制（本地内存带宽）

**优势**:
- ✅ **零网络延迟**: 本地通信，延迟可忽略
- ✅ **高带宽**: 不受网络带宽限制
- ✅ **简单部署**: 一键启动

---

### 4.2 云端部署（Distributed）

**场景**: Router 和 Worker 在不同机器上

```
┌─────────────────────────────────────────┐
│ 机器 A（Router）                         │
│  ┌──────────────┐                       │
│  │   Router     │                       │
│  │ :30000       │                       │
│  └──────┬───────┘                       │
└─────────┼───────────────────────────────┘
          │ 内网/外网 (1-100ms)
          ↓
┌─────────┼───────────────────────────────┐
│ 机器 B  │  Worker 1                     │
│         │  :8000                        │
│         │  GPU 0                       │
└─────────┼───────────────────────────────┘
          │
          ↓
┌─────────┼───────────────────────────────┐
│ 机器 C  │  Worker 2                     │
│         │  :8000                        │
│         │  GPU 0                       │
└─────────┼───────────────────────────────┘
```

**通信方式**:
- ✅ **内网**: 同一数据中心（1-10ms 延迟）
- ✅ **外网**: 跨数据中心（10-100ms 延迟）
- ✅ **带宽**: 受网络带宽限制

**优势**:
- ✅ **横向扩展**: 可以添加更多 Worker
- ✅ **资源隔离**: Worker 独立部署
- ✅ **灵活配置**: 可以独立扩展

---

### 4.3 混合部署（Hybrid）

**场景**: Router 和部分 Worker 在同一机器，部分 Worker 在远程

```
┌─────────────────────────────────────────┐
│ 机器 A（Router + Worker 1）              │
│  ┌──────────────┐                       │
│  │   Router     │                       │
│  └──────┬───────┘                       │
│         │ localhost (0.1ms)            │
│         ↓                               │
│  ┌──────────────┐                       │
│  │   Worker 1   │                       │
│  └──────────────┘                       │
└─────────┼───────────────────────────────┘
          │ 内网 (5ms)
          ↓
┌─────────┼───────────────────────────────┐
│ 机器 B  │  Worker 2                     │
│         │  :8000                        │
└─────────┼───────────────────────────────┘
```

**优势**:
- ✅ **本地优先**: 优先使用本地 Worker（零延迟）
- ✅ **远程扩展**: 可以添加远程 Worker
- ✅ **灵活配置**: 根据需求调整部署

---

## 5. 传输时间优化

### 5.1 本地部署的优势

**问题**: 本地 GPU 到 Router，不发送给云端，省略传输时间？

**答案**: **是的！** 本地部署可以大幅减少传输时间。

**对比**:

| 部署方式 | 传输路径 | 延迟 | 带宽 |
|---------|---------|------|------|
| **本地部署** | Router → Worker (localhost) | < 0.1ms | 不受限 |
| **内网部署** | Router → Worker (内网) | 1-10ms | 1-10 Gbps |
| **外网部署** | Router → Worker (外网) | 10-100ms | 100 Mbps - 1 Gbps |

**性能提升**:
- ✅ **延迟减少**: 从 10-100ms 减少到 < 0.1ms（**100-1000x**）
- ✅ **带宽提升**: 从网络带宽提升到本地内存带宽（**10-100x**）

---

### 5.2 实际传输时间

**本地部署（localhost）**:
```
Router → Worker:
  - 传输时间: < 0.1ms
  - 带宽: 不受限（本地内存）
  - 总开销: 可忽略
```

**内网部署（同一数据中心）**:
```
Router → Worker:
  - 传输时间: 1-10ms
  - 带宽: 1-10 Gbps
  - 总开销: 较小（但可感知）
```

**外网部署（跨数据中心）**:
```
Router → Worker:
  - 传输时间: 10-100ms
  - 带宽: 100 Mbps - 1 Gbps
  - 总开销: 较大（影响性能）
```

---

### 5.3 Cache-Aware Router 的额外优势

**即使在内网/外网部署中，Cache-Aware Router 也能优化**:

1. **减少不必要的传输**:
   - ✅ 路由到有缓存的 Worker
   - ✅ 减少重复计算
   - ✅ 提高缓存命中率

2. **负载均衡**:
   - ✅ 避免单个 Worker 过载
   - ✅ 提高整体吞吐量
   - ✅ 减少等待时间

3. **故障容错**:
   - ✅ 自动重试失败的请求
   - ✅ 故障 Worker 自动隔离
   - ✅ 提高系统可用性

---

## 6. 总结

### 6.1 RadixCache.insert() 总结

**功能**:
- ✅ **插入 KV Cache**: 将 KV Cache 索引插入到 Radix Tree
- ✅ **树结构管理**: 自动构建和维护 Radix Tree
- ✅ **节点分裂**: 处理部分匹配的情况

**关键步骤**:
1. 查找匹配的子节点
2. 计算共享前缀长度
3. 完全匹配 → 更新节点
4. 部分匹配 → 分裂节点
5. 无匹配 → 创建新节点

---

### 6.2 Router 架构总结

**Router 是什么**:
- ✅ **独立的负载均衡服务**: 在多个 Worker 之间路由请求
- ✅ **可以是本地部署**: Router 和 Worker 在同一台机器
- ✅ **可以是分布式部署**: Router 和 Worker 在不同机器

**部署方式**:
- ✅ **Co-launch Mode**: Router 和 Worker 同机启动（最简单）
- ✅ **Separate Launch**: Router 和 Worker 独立启动（最灵活）
- ✅ **混合部署**: 部分 Worker 本地，部分远程

---

### 6.3 传输时间优化总结

**本地部署的优势**:
- ✅ **零网络延迟**: localhost 通信，延迟 < 0.1ms
- ✅ **高带宽**: 不受网络带宽限制
- ✅ **简单部署**: 一键启动所有服务

**即使分布式部署，Cache-Aware Router 也能优化**:
- ✅ **减少重复计算**: 路由到有缓存的 Worker
- ✅ **负载均衡**: 避免单个 Worker 过载
- ✅ **故障容错**: 自动重试和隔离

---

**结论**: 
1. **RadixCache.insert()** 负责将 KV Cache 插入到 Radix Tree，支持节点分裂和树结构管理
2. **Router** 是一个独立的负载均衡服务，可以是本地部署（localhost，零延迟）或分布式部署（内网/外网）
3. **本地部署时**，确实可以省略网络传输时间（< 0.1ms），大幅提升性能
4. **即使分布式部署**，Cache-Aware Router 也能通过智能路由优化性能

文档已保存到: `yc_self_learn/md/19_RadixCache_insert详解与Router架构.md`

