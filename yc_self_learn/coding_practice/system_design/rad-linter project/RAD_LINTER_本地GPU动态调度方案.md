# Rad-Linter 本地 GPU 动态任务分配方案
# On-Premise GPU Dynamic Task Scheduling for Rad-Linter

**作者**：Yanda Cheng  
**项目**：Rad-Linter  
**核心难点**：本地化部署 + 异构 GPU 集群 + 动态负载调度  
**与 KYC 的最大区别**：KYC 使用托管 API（Fireworks），Rad-Linter 需要本地 GPU 资源管理和调度

---

## 📋 目录

1. [核心挑战](#核心挑战)
2. [任务分类](#任务分类)
3. [工业级解决方案](#工业级解决方案)
4. [关键设计原则](#关键设计原则)
5. [落地实施方案](#落地实施方案)
6. [监控与告警](#监控与告警)

---

## 核心挑战

### Rad-Linter 的独特难点

**Rad-Linter 真正卡脖子的不是"prompt写得多好"，而是：**
- **On-Prem 本地化**：必须部署在医院本地，不能使用云端 API
- **本地 GPU 动态任务分配**：资源小、业务抖动、又要稳定可审计
- **异构 GPU 集群**：多台服务器、多张 GPU、型号不同、版本不同、空闲算力随时波动

### 为什么这是核心挑战？

与传统 KYC 项目（使用托管 API）不同，Rad-Linter 需要：
- **资源管理**：自己管理 GPU 资源，不能依赖云服务商的自动扩缩容
- **动态调度**：任务请求到达时间不可预测，需要动态分配 GPU 资源
- **延迟敏感**：签字前的 lint 检查必须快速响应（P95 < 8s）
- **异构适配**：不同型号的 GPU 性能差异巨大，需要智能调度

### 典型场景

**场景描述**：
- 多台本地服务器
- 每台服务器有不同型号的 GPU（如：4090×2、A100×1、L40S×2）
- GPU 驱动版本不同、显存不同
- 其他系统也在使用 GPU，空闲算力波动
- **目标**：为了响应速度，如何在异构、动态环境下实现工业级调度？

---

## 任务分类

### 两类 GPU 任务

#### 1. LLM Judge（强依赖 GPU，且 latency 敏感）

**任务特征**：
- **输入**：report_facts + visual_facts
- **输出**：lint_result（结构化 JSON）
- **特点**：
  - 短但频繁
  - 需要低 P95/P99（签字前必须快速响应）
  - 强依赖 GPU 推理

**性能要求**：
- P95 < 8s
- P99 < 15s
- 并发量：中等（10-100 req/s）

#### 2. Evidence Builder（可能依赖 GPU，也可能 CPU）

**任务特征**：
- **图像侧**：检测/分割/测量（可能是 GPU）
- **文本侧**：NER/negation（通常 CPU 足够）
- **特点**：
  - 更重（单任务耗时更长）
  - 可以异步处理
  - 吞吐导向（可以排队批量处理）

**性能要求**：
- 延迟要求：较低（可以异步）
- 并发量：较低（可以排队）

### 核心设计原则

**工业级"动态分配"的第一原则：这两类不要抢同一块 GPU。**

**原因**：
- 分割任务吃满显存 → LLM TTFT 爆炸 → 医生端卡住
- LLM 推理需要稳定的 KV Cache，显存碎片会导致性能下降
- 两种任务的工作负载特性不同，混跑会导致相互干扰

---

## 工业级解决方案

### 方案对比

从"最稳最好运维"到"最强最复杂"：

| 方案 | 适用场景 | 优势 | 代价 | 推荐度 |
|------|---------|------|------|--------|
| **方案 A** | 单团队、快速落地 | 实现简单、落地快 | 需要自建 Router | ⭐⭐⭐⭐⭐ |
| **方案 B** | 多团队共享、长期运行 | 工业规范、可审计 | 运维成本高 | ⭐⭐⭐⭐ |
| **方案 C** | 资源紧张、严格 SLO | 性能最优、SLO 保证 | 工程复杂度最高 | ⭐⭐⭐ |

---

### 方案 A（最推荐）：多副本 + Router 负载均衡

**关键词**：SGLang/vLLM 多 replica + 自适应路由 + 每卡单独并发上限

#### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Requests                          │
│                  (Realtime + Batch)                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Router / Load Balancer                   │
│  • 实时采集 Worker 指标                                      │
│  • 最短完成时间路由 (Shortest-Estimated-Completion-Time)    │
│  • 熔断/降级策略                                             │
└──────┬────────────────┬──────────────────┬─────────────────┘
       │                │                  │
       ▼                ▼                  ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ LLM Worker  │  │ LLM Worker  │  │ LLM Worker  │
│ (GPU 0)     │  │ (GPU 1)     │  │ (GPU 2)     │
│             │  │             │  │             │
│ CUDA_VIS... │  │ CUDA_VIS... │  │ CUDA_VIS... │
│ = 0         │  │ = 1         │  │ = 2         │
│             │  │             │  │             │
│ Max Conc:   │  │ Max Conc:   │  │ Max Conc:   │
│ 8           │  │ 16          │  │ 32          │
│ (4090)      │  │ (A100)      │  │ (L40S)      │
└─────────────┘  └─────────────┘  └─────────────┘
```

#### 实现要点

##### 1. 每张 GPU 起一个 LLM Worker

**绑定方式**：
```bash
# Worker 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 sglang serve --model-path /path/to/model

# Worker 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 sglang serve --model-path /path/to/model

# Worker 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 sglang serve --model-path /model
```

**关键配置**：
- 每个 worker 独立进程/容器
- 每卡单独设置并发上限
- 独立监控端口

##### 2. Worker 容量配置（按 GPU 型号）

**配置表示例**：
```yaml
workers:
  - worker_id: "worker_0"
    gpu_id: 0
    gpu_model: "RTX_4090"
    gpu_memory_gb: 24
    max_concurrency: 8      # 同时处理请求数
    max_batch_tokens: 4096  # 批处理 token 上限
    max_num_seqs: 16        # 批处理序列数上限
    
  - worker_id: "worker_1"
    gpu_id: 1
    gpu_model: "A100"
    gpu_memory_gb: 40
    max_concurrency: 16
    max_batch_tokens: 8192
    max_num_seqs: 32
    
  - worker_id: "worker_2"
    gpu_id: 2
    gpu_model: "L40S"
    gpu_memory_gb: 48
    max_concurrency: 32
    max_batch_tokens: 16384
    max_num_seqs: 64
```

##### 3. Router 实时指标采集

**关键指标**（每 5-10 秒更新）：
```python
class WorkerMetrics:
    inflight_requests: int        # 当前正在处理的请求数
    free_vram_mb: int             # 剩余显存（MB）
    kv_cache_used_mb: int         # KV Cache 占用的显存
    recent_ttft_p95_ms: float     # 最近 1 分钟的 TTFT P95（毫秒）
    recent_latency_p95_ms: float  # 最近 1 分钟的端到端延迟 P95
    oom_count_last_5min: int      # 过去 5 分钟的 OOM 次数
    queue_length: int             # 队列中等待的请求数
    gpu_utilization: float        # GPU 利用率（参考用，不是主要依据）
```

**采集方式**：
- DCGM exporter（推荐，准确）
- nvidia-smi 轮询（简单但不够实时）
- Worker 主动上报（最准但需要修改代码）

##### 4. Router 路由策略

**核心算法**：Shortest-Estimated-Completion-Time (SECT)

**打分公式**：
```python
def score_worker(worker_metrics: WorkerMetrics, request: Request) -> float:
    """
    计算 Worker 的得分，得分越低越好（优先选择）
    """
    # 基础延迟估算
    base_latency = estimate_base_latency(
        worker_metrics.gpu_model,
        request.estimated_tokens
    )
    
    # 队列等待时间
    queue_wait = worker_metrics.queue_length * base_latency
    
    # 显存压力惩罚
    memory_penalty = 0
    if worker_metrics.free_vram_mb < request.estimated_vram:
        memory_penalty = 1000  # 显存不足，惩罚
    
    # OOM 惩罚
    oom_penalty = worker_metrics.oom_count_last_5min * 500
    
    # 当前负载惩罚
    load_penalty = worker_metrics.inflight_requests / worker_metrics.max_concurrency
    
    # 延迟惩罚（如果最近延迟很高）
    latency_penalty = 0
    if worker_metrics.recent_latency_p95_ms > SLA_P95_MS:
        latency_penalty = (worker_metrics.recent_latency_p95_ms - SLA_P95_MS) / 10
    
    total_score = (
        base_latency +
        queue_wait +
        memory_penalty +
        oom_penalty +
        load_penalty * 100 +
        latency_penalty
    )
    
    return total_score

def select_worker(workers: List[WorkerMetrics], request: Request) -> str:
    """
    选择得分最低（最快）的 Worker
    """
    scores = [(score_worker(w, request), w.worker_id) for w in workers]
    scores.sort()  # 按得分排序
    
    # 过滤掉不可用的 Worker
    available = [(s, wid) for s, wid in scores if is_worker_available(wid, request)]
    
    if not available:
        raise NoAvailableWorkerError("所有 Worker 都不可用")
    
    return available[0][1]  # 返回得分最低的 Worker ID
```

##### 5. 两级队列（Realtime vs Batch）

**队列设计**：
```python
class PriorityQueue:
    def __init__(self):
        self.realtime_queue = Queue()  # 签字前 lint（必须快）
        self.batch_queue = Queue()     # 夜间重跑/回溯（可以等）
    
    def enqueue(self, request: Request):
        if request.priority == "realtime":
            self.realtime_queue.put(request)
        else:
            self.batch_queue.put(request)
    
    def dequeue(self):
        # realtime 永远优先
        if not self.realtime_queue.empty():
            return self.realtime_queue.get()
        elif not self.batch_queue.empty():
            return self.batch_queue.get()
        return None
```

**调度策略**：
- **Realtime 永远抢占 Batch**：即使 batch 已经在排队，realtime 请求也会优先处理
- **Batch 限速 + 合并请求**：batch 任务可以合并处理，提高吞吐

#### 你会得到什么

✅ **异构 GPU 自动利用**：强卡多扛、弱卡少扛  
✅ **空闲算力波动适配**：某张卡被别人占用/变慢，路由自然避开  
✅ **响应速度最优**：最贴近"线上推理负载均衡"的工业实践  
✅ **实现简单**：不需要复杂的集群管理

#### 代价

⚠️ **需要参数化管理**：每个 worker 的容量需要配置  
⚠️ **需要轻量监控**：DCGM exporter 或 nvidia-smi 轮询  
⚠️ **需要自己写 Router**：或者使用现成的网关/sidecar

---

### 方案 B（医院/企业最爱）：Kubernetes + GPU Operator

**关键词**：集群级资源治理 + QoS + 可审计

#### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│              Kubernetes Cluster                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Kueue / Volcano Scheduler               │  │
│  │  • 任务排队                                           │  │
│  │  • 优先级/抢占                                        │  │
│  │  • 资源回收                                           │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                      │
│  ┌──────────────────┴───────────────────────────────────┐  │
│  │              GPU Operator                            │  │
│  │  • 节点标签管理                                        │  │
│  │  • GPU 资源发现                                       │  │
│  │  • 设备插件                                           │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                      │
│  ┌──────────────────┴───────────────────────────────────┐  │
│  │              Node Labels                             │  │
│  │  gpu.model=4090 / gpu.mem=24g / gpu.cc=8.9          │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                      │
│  ┌──────────────────┴───────────────────────────────────┐  │
│  │              Pod Scheduling                          │  │
│  │  • PriorityClass: realtime > batch                   │  │
│  │  • ResourceQuota: 防止资源耗尽                       │  │
│  │  • Anti-affinity: LLM 与 CV 隔离                    │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                      │
│         ┌───────────┴───────────┐                         │
│         ▼                       ▼                         │
│  ┌──────────────┐      ┌──────────────┐                  │
│  │ LLM Pool     │      │ CV/Batch Pool│                  │
│  │ (Node Group) │      │ (Node Group) │                  │
│  │              │      │              │                  │
│  │ • 4090×2     │      │ • 4090×1     │                  │
│  │ • A100×1     │      │ • L40S×2     │                  │
│  └──────────────┘      └──────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

#### 实现要点

##### 1. Node 标签管理

**给每个节点打标签**：
```yaml
# Node 1
labels:
  gpu.model: "RTX_4090"
  gpu.memory: "24Gi"
  gpu.compute_capability: "8.9"
  gpu.count: "2"
  pool: "llm-pool"

# Node 2
labels:
  gpu.model: "A100"
  gpu.memory: "40Gi"
  gpu.compute_capability: "8.0"
  gpu.count: "1"
  pool: "llm-pool"

# Node 3
labels:
  gpu.model: "L40S"
  gpu.memory: "48Gi"
  gpu.compute_capability: "8.9"
  gpu.count: "2"
  pool: "cv-pool"
```

##### 2. PriorityClass 定义

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: realtime-priority
value: 1000
description: "Realtime lint requests (pre-approval)"

---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: batch-priority
value: 100
description: "Batch processing tasks"
```

##### 3. Pod Anti-Affinity（避免混跑）

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-judge
spec:
  template:
    spec:
      priorityClassName: realtime-priority
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: pool
                operator: In
                values:
                - llm-pool
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - cv-worker  # 避免与 CV 任务同节点
            topologyKey: kubernetes.io/hostname
      containers:
      - name: sglang
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

##### 4. ResourceQuota（防止资源耗尽）

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: llm-pool-quota
spec:
  hard:
    requests.nvidia.com/gpu: "8"    # 最多申请 8 张 GPU
    limits.nvidia.com/gpu: "8"
```

#### 你会得到什么

✅ **最强的"工业规范感"**：权限、审计、配额、故障隔离都好做  
✅ **多团队共享友好**：你说的"空余算力波动"常来自共享  
✅ **可审计性**：所有资源分配都有审计日志  
✅ **故障隔离**：Pod 故障不影响其他 Pod

#### 代价

⚠️ **运维成本较高**：需要熟悉 K8s 运维  
⚠️ **实时性能感知不够敏感**：K8s 调度更偏资源层，不会根据 TTFT 动态避开某张卡  
⚠️ **需要配合 Router**：K8s 负责"谁能用多少资源"，Router 负责"请求打到哪块卡最快"

**现实做法**：
- **K8s 负责**：资源配额、权限管理、故障隔离
- **Router 负责**：实时路由决策、性能优化

---

### 方案 C（最强但最复杂）：Slurm / Volcano + SLO 驱动调度

**关键词**：HPC 调度思路 + SLO 驱动的在线容量管理

#### 适用场景

✅ **资源很紧张**：GPU 数量少，竞争激烈  
✅ **严格低延迟要求**：SLO 必须保证  
✅ **需要降级策略**：资源不足时可以降级

#### 核心思想

**给每张 GPU 建一个"在线容量模型"**：

```python
def estimate_capacity(
    model_size: int,           # 模型参数量
    ctx_len: int,              # 上下文长度
    kv_cache_size: int,        # KV Cache 大小
    batch_tokens: int,         # 批处理 token 数
    gpu_type: str,             # GPU 型号
    current_util: float,       # 当前利用率
    temp_throttle: bool        # 是否热降频
) -> float:
    """
    估算 GPU 的可用容量（能同时处理多少个请求）
    """
    # 基础容量（基于 GPU 型号）
    base_capacity = GPU_CAPACITY_TABLE[gpu_type]
    
    # 模型大小影响
    model_factor = estimate_model_factor(model_size, ctx_len)
    
    # KV Cache 占用
    kv_factor = kv_cache_size / TOTAL_VRAM[gpu_type]
    
    # 当前负载影响
    util_factor = 1.0 - current_util
    
    # 热降频影响
    throttle_factor = 0.7 if temp_throttle else 1.0
    
    # 综合容量
    capacity = (
        base_capacity *
        model_factor *
        (1.0 - kv_factor) *
        util_factor *
        throttle_factor
    )
    
    return max(0, capacity)

def can_satisfy_slo(
    worker: WorkerMetrics,
    request: Request,
    target_p95_ms: float
) -> bool:
    """
    判断 Worker 是否能满足请求的 SLO 要求
    """
    estimated_latency = estimate_latency(worker, request)
    return estimated_latency <= target_p95_ms
```

#### Admission Control（准入控制）

```python
class AdmissionController:
    def __init__(self):
        self.capacity_model = CapacityModel()
        self.slo_targets = {
            "realtime": 8000,   # P95 < 8s
            "batch": 30000,     # P95 < 30s
        }
    
    def admit(self, request: Request) -> AdmissionDecision:
        """
        决定是否接受请求，以及如何处理
        """
        # 找到所有可用的 Worker
        available_workers = self.find_available_workers(request)
        
        if not available_workers:
            # 没有可用 Worker，需要降级
            return self.degrade_request(request)
        
        # 检查是否能满足 SLO
        capable_workers = [
            w for w in available_workers
            if self.can_satisfy_slo(w, request, self.slo_targets[request.priority])
        ]
        
        if capable_workers:
            # 有能满足 SLO 的 Worker，直接接受
            return AdmissionDecision(
                action="accept",
                target_worker=select_best_worker(capable_workers),
                estimated_latency=self.estimate_latency(capable_workers[0], request)
            )
        else:
            # 不能满足 SLO，降级处理
            return self.degrade_request(request)
    
    def degrade_request(self, request: Request) -> AdmissionDecision:
        """
        降级策略
        """
        # 策略 1：使用更小的模型
        if self.can_use_smaller_model(request):
            return AdmissionDecision(
                action="degrade_model",
                target_worker=self.find_small_model_worker(),
                estimated_latency=self.estimate_with_small_model(request)
            )
        
        # 策略 2：使用规则引擎兜底
        if self.can_use_rules(request):
            return AdmissionDecision(
                action="degrade_rules",
                target_worker="rule_engine",
                estimated_latency=50  # 规则引擎很快
            )
        
        # 策略 3：转人工复核
        return AdmissionDecision(
            action="escalate_human",
            target_worker="human_queue",
            estimated_latency=None
        )
```

#### 你会得到什么

✅ **真正意义上的"响应速度优先"**：SLO 能守住  
✅ **资源抖动时保持系统不崩**：宁可降级也不超时  
✅ **智能降级**：自动选择最佳降级策略

#### 代价

⚠️ **工程复杂度最高**：需要自己维护容量模型和策略  
⚠️ **需要大量测试**：容量模型需要校准  
⚠️ **运维成本高**：需要持续监控和调整

---

## 关键设计原则

### 1. GPU 分池（硬隔离，最关键）

**不要混跑**：LLM 和 CV 任务不要在同一块 GPU 上运行

**分池策略**：
- **Pool A：LLM Serving Pool**（高优先级、低延迟）
  - 只跑 SGLang / vLLM / Triton(LLM)
  - 设置更严格的 admission control
  - 避免排队爆炸

- **Pool B：CV/重计算 Pool**（低优先级、吞吐）
  - 跑分割/测量/embedding 等
  - 允许排队、允许晚点完成
  - 可以批量处理

**动态分配不是"所有任务动态抢 GPU"，而是先把 GPU 角色固定，再在各自池内做动态调度。**

### 2. 队列与优先级（软件层的"动态任务分配"）

**两级队列**：
- **q_realtime**：签字前 lint（必须快）
- **q_batch**：夜间重跑/训练数据生成/大批量回溯

**调度策略**：
- realtime 永远抢占 batch（即使 batch 已经在排队）
- batch 允许"限速 + 合并请求"

### 3. LLM Serving 内部：动态批处理 + KV 管控

**核心三件事**：
- **Dynamic batching**：把短请求凑批，吞吐上去
- **并发上限**：按显存/kv-cache 估算，做 admission control
- **多模型/多 LoRA**：尽量少切换（切换=cache抖动=延迟尖峰）

### 4. 本地 GPU 少但想"切片用"：MIG / MPS

#### MIG（硬切片，强隔离，最稳定）

**适用**：A100/H100/L40S 等支持 MIG 的卡

**做法**：把一张卡切成多个 GPU 实例（例如 2g/4g/...）

**好处**：
- LLM 和 CV 真隔离，互不炸显存
- 每个切片独立管理，互不干扰

**坏处**：
- 切片后单实例算力/显存变小，LLM 可能跑不下大模型
- 需要提前规划切片大小

#### MPS（软并发，更灵活但隔离弱）

**适用**：同类 workload 并发（例如多个轻量 CV）

**不适合**：CV + LLM 混跑（容易抖）

**经验**：能 MIG 就 MIG（医院喜欢稳定），MPS 属于"卡少凑合"

---

## 关键难点与解法对照表

### 难点 A：显存抖动导致 TTFT/P99 爆炸

**解法**：
1. ✅ **LLM 与 CV 分池**（最有效）
2. ✅ **LLM serving 做 admission control**（宁可排队也别 OOM）
3. ✅ **模型常驻**，减少频繁加载/卸载

### 难点 B：Realtime 和 Batch 互相拖死

**解法**：
1. ✅ **两级队列 + 抢占**（realtime 优先）
2. ✅ **Batch 限速 + 合并请求**（尤其夜间）

### 难点 C：多任务/多模型导致 Cache Thrash

**解法**：
1. ✅ **同一 GPU 尽量固定模型**（或固定一组）
2. ✅ **需要多模型就多 replica**，不要同一 replica 上频繁切

### 难点 D：本地 GPU 利用率低（任务太碎）

**解法**：
1. ✅ **动态 batching**（SGLang/vLLM）
2. ✅ **把 lint 请求标准化**成"短 prompt + 结构化输出"，易凑批

### 难点 E：异构 GPU + 波动负载

**关键洞察**：异构 + 波动时，"动态分配"千万别只靠 GPU utilization

**常见踩坑**：
- ❌ 看 GPU% 觉得空，其实 KV cache/显存碎片/热降频才是延迟杀手

**正确的信号**（Router 决策用这 5 个）：
1. ✅ **inflight_requests**：当前正在处理的请求数
2. ✅ **free_vram + kv_cache_used**：剩余显存和 KV Cache 占用
3. ✅ **recent_ttft_p95**：最近 TTFT P95
4. ✅ **recent_latency_p95**：最近端到端延迟 P95
5. ✅ **oom_count_last_5min**：一旦出现就临时降权/熔断

---

## 落地实施方案

### 最小工业级（1-2 周能落地）

**组合拳**：
1. ✅ **LLM serving 多副本**（每卡一个 worker）
2. ✅ **Router 做实时指标路由**（inflight + 显存/kv + 最近延迟）
3. ✅ **每 worker 不同并发上限**（按 GPU 型号/显存写配置表）
4. ✅ **两级队列**：realtime / batch（realtime 永远优先）

### 进阶工业级（要长期跑、多团队共享）

**在上面基础上加**：
5. ✅ **K8s 配额/优先级/隔离**（LLM pool vs batch pool）
6. ✅ **熔断与降级策略**（超时→切小模型/切规则/切人工复核）

### 落地模板（最现实的版本）

**如果你要一个"医院端能跑、团队也维护得起"的组合：**

```
LLM Judge：
├─ SGLang（或 vLLM）常驻服务
├─ Router 动态批处理
└─ 每卡一个 Worker，独立监控

Evidence Builder：
├─ 异步任务队列
└─ 单独 GPU/CPU 池（与 LLM 隔离）

调度：
├─ 两级队列（realtime/batch）
└─ 并发上限（admission control）

隔离：
├─ 有条件就 MIG
└─ 没条件就至少分池 + 限流

监控：
├─ P95/P99
├─ TTFT
├─ Inflight
├─ 显存
├─ OOM 次数
└─ 自动放行率
```

---

## 监控与告警

### 核心指标

#### 性能指标
- **P50/P95/P99 Latency**：端到端延迟
- **TTFT (Time To First Token)**：首 token 延迟
- **Throughput**：吞吐量（req/s）

#### 资源指标
- **GPU Utilization**：GPU 利用率（参考用）
- **VRAM Usage**：显存使用率
- **KV Cache Usage**：KV Cache 占用
- **Inflight Requests**：正在处理的请求数

#### 质量指标
- **OOM Count**：OOM 次数
- **Error Rate**：错误率
- **Automation Rate**：自动放行率
- **False Positive/Negative Rate**：误报/漏检率

### 告警规则

#### 延迟告警
```yaml
- alert: HighLatencyP95
  expr: latency_p95_ms > 8000
  for: 5m
  severity: warning

- alert: HighLatencyP99
  expr: latency_p99_ms > 15000
  for: 5m
  severity: critical
```

#### 资源告警
```yaml
- alert: HighOOMRate
  expr: oom_count_last_5min > 0
  for: 1m
  severity: critical

- alert: HighVRAMUsage
  expr: vram_usage_percent > 90
  for: 5m
  severity: warning
```

#### 质量告警
```yaml
- alert: LowAutomationRate
  expr: automation_rate < 0.80
  for: 10m
  severity: warning

- alert: HighErrorRate
  expr: error_rate > 0.05
  for: 5m
  severity: critical
```

---

## 总结

### 核心要点

1. **GPU 分池是基础**：LLM 和 CV 不要混跑
2. **Router 是关键**：异构 GPU + 波动负载需要智能路由
3. **不要只看 GPU%**：KV Cache、显存碎片、热降频才是延迟杀手
4. **两级队列**：Realtime 优先，Batch 可降级
5. **容量模型**：根据实际情况估算并发上限

### 与 KYC 项目的最大区别

| 维度 | KYC 项目 | Rad-Linter |
|------|---------|-----------|
| **资源管理** | 使用托管 API（Fireworks） | 本地 GPU 资源管理 |
| **调度策略** | API 自动扩缩容 | 需要自己实现动态调度 |
| **延迟优化** | API 端优化 | 本地 GPU 调度优化 |
| **成本控制** | 按 token 付费 | GPU 资源成本 |

### 推荐方案选择

**如果你的场景是**：
- ✅ 单团队、快速落地 → **方案 A**
- ✅ 多团队共享、长期运行 → **方案 B**
- ✅ 资源紧张、严格 SLO → **方案 C**

**但无论选哪个，都要做到**：
- GPU 分池（LLM vs CV）
- 两级队列（Realtime vs Batch）
- Router 智能路由（基于实时指标）
- 容量模型（估算并发上限）

---

## 下一步

**需要具体配置时，请提供**：
1. 每台机器的 GPU 型号和数量（如：4090×2、A100×1、L40S×2）
2. LLM Judge 用的模型规模（7B/14B/...）和典型上下文长度（2k/8k/16k）
3. 请求形态：Realtime vs Batch 占比

**我可以给出**：
- 最稳的分池策略
- 每类 GPU 的并发/批处理上限建议
- Router 的打分与熔断规则（可直接写进代码）