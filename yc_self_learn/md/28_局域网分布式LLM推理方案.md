# 局域网分布式 LLM 推理方案

## 📋 问题背景

**核心问题**：医院的局域网，能不能让局域网的所有机器同时帮忙跑 token，这样加速本地 LLM 的运行速度？

**答案**：**可以，但需要正确的分布式策略** ✅

---

## 1. 分布式 LLM 推理的三种并行方式

### 1.1 数据并行（Data Parallelism, DP）

**概念**：每个机器运行完整的模型，处理不同的请求

```
机器 1: [完整模型] → 处理请求 A
机器 2: [完整模型] → 处理请求 B
机器 3: [完整模型] → 处理请求 C
```

**优点**：
- ✅ 实现简单
- ✅ 适合多请求场景
- ✅ 单机故障不影响其他机器

**缺点**：
- ❌ 每台机器都需要完整模型（内存要求高）
- ❌ 不适合单请求加速

**适用场景**：**多请求并发处理**（医院多用户同时使用）

---

### 1.2 张量并行（Tensor Parallelism, TP）

**概念**：模型的不同层分布在不同机器上

```
机器 1: [Layer 0-5]   → 处理 token 的一部分
机器 2: [Layer 6-11]  → 处理 token 的另一部分
机器 3: [Layer 12-17] → 处理 token 的剩余部分
```

**工作原理**：
```
Token → 机器1 (Layer 0-5) → 通信 → 机器2 (Layer 6-11) → 通信 → 机器3 (Layer 12-17) → 输出
```

**优点**：
- ✅ 可以运行超大模型（单机内存不够）
- ✅ 单请求可以加速（如果通信开销小）

**缺点**：
- ❌ 需要频繁通信（每层都要通信）
- ❌ 网络延迟影响性能
- ❌ 一台机器故障，整个请求失败

**适用场景**：**超大模型 + 低延迟网络**（InfiniBand/RDMA）

---

### 1.3 流水线并行（Pipeline Parallelism, PP）

**概念**：模型的不同部分分布在不同机器上，按顺序处理

```
机器 1: [Layer 0-5]   → 处理 token 1, 2, 3...
机器 2: [Layer 6-11]  → 处理 token 1, 2, 3...（等待机器1完成）
机器 3: [Layer 12-17] → 处理 token 1, 2, 3...（等待机器2完成）
```

**工作原理**：
```
Token 1: 机器1 → 机器2 → 机器3
Token 2: 机器1 → 机器2 → 机器3（流水线）
Token 3: 机器1 → 机器2 → 机器3
```

**优点**：
- ✅ 可以运行超大模型
- ✅ 通信开销相对较小（只在层之间通信）

**缺点**：
- ❌ 流水线气泡（Pipeline Bubble）导致 GPU 利用率低
- ❌ 单请求延迟可能增加

**适用场景**：**超大模型 + 多请求批处理**

---

## 2. 医院局域网场景分析

### 2.1 场景特点

**硬件环境**：
- 多台普通 PC（可能没有 GPU 或只有低端 GPU）
- 千兆以太网（1 Gbps）
- 可能没有专用高速网络（InfiniBand/RDMA）

**使用场景**：
- 多用户同时使用
- 可能需要处理医疗文档、报告等
- 对延迟有一定要求，但不是极致

---

### 2.2 可行性分析

#### 方案 1：数据并行（推荐 ✅）

**实现方式**：
```python
# 伪代码：数据并行部署
# 机器 1（主节点）
router = Router()
router.add_worker("192.168.1.10:8000")  # 机器 1
router.add_worker("192.168.1.11:8000")  # 机器 2
router.add_worker("192.168.1.12:8000")  # 机器 3

# 每个机器运行完整模型
# 机器 1, 2, 3 各自运行：
python -m sglang.launch_server \
    --model-path /path/to/model \
    --port 8000
```

**优点**：
- ✅ **实现简单**：每台机器独立运行，互不干扰
- ✅ **容错性好**：一台机器故障，其他机器继续工作
- ✅ **适合多请求**：多个用户同时使用，负载均衡
- ✅ **网络要求低**：只需要 HTTP 请求转发，不需要频繁通信

---

### 2.3 实际应用案例：已经有人做了！✅

**重要说明**：数据并行**不是新概念**，而是**非常成熟的技术**，已经在多个项目中实现和应用。

#### a) **开源项目实现**

**1. SGLang（你正在用的项目）**
- ✅ **完整实现**：`python/sglang/srt/managers/data_parallel_controller.py`
- ✅ **负载均衡策略**：
  - Round Robin（轮询）
  - Shortest Queue（最短队列）
  - Minimum Tokens（最少 Token）
- ✅ **Router 支持**：Cache-Aware Load Balancing

**代码位置**：
```python
# SGLang 数据并行控制器
class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""
    
    def __init__(self, server_args, port_args):
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        # 支持三种负载均衡策略
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
            LoadBalanceMethod.MINIMUM_TOKENS: self.minimum_tokens_scheduler,
        }
```

**2. vLLM**
- ✅ 支持数据并行（Data Parallelism）
- ✅ 支持多 Worker 负载均衡
- GitHub: https://github.com/vllm-project/vllm

**3. Ray Serve**
- ✅ 专门用于分布式模型服务
- ✅ 自动负载均衡
- ✅ 支持动态扩缩容
- GitHub: https://github.com/ray-project/ray

**4. TensorFlow Serving / TorchServe**
- ✅ 企业级模型服务框架
- ✅ 支持多实例负载均衡
- ✅ 生产环境广泛应用

#### b) **实际应用案例**

**1. 中国电信研究院 + 北京大学**
- ✅ **项目**：企业级 LLM 推理优化方案
- ✅ **成果**：
  - 平均端到端时延降低 40%
  - 短请求首 token 时延下降 75%
  - 解决了多任务混合场景中短请求受长请求干扰的问题
- ✅ **技术**：数据并行 + 智能调度

**2. 华为云盘古大模型**
- ✅ 在多个领域广泛应用
- ✅ 展示了分布式环境中部署大模型的实践经验
- ✅ 支持多节点推理

**3. OpenAI / Anthropic 等商业公司**
- ✅ 内部使用数据并行处理大量并发请求
- ✅ 虽然不公开代码，但技术原理相同

#### c) **为什么感觉"还没开始研究"？**

**可能的原因**：

1. **关注点不同**
   - 学术界更关注**张量并行/流水线并行**（可以运行超大模型）
   - 工业界更关注**数据并行**（提高并发处理能力）
   - 数据并行相对"简单"，研究价值不如 TP/PP

2. **宣传不够**
   - 数据并行是"基础设施"，不像新算法那样吸引眼球
   - 很多实现是"工程实践"，不是"研究论文"

3. **命名混淆**
   - "数据并行"听起来像训练时的概念
   - 推理时的数据并行（负载均衡）可能被忽略

4. **实现分散**
   - 不同项目有不同的实现方式
   - 没有统一的"标准"实现

**实际情况**：
- ✅ **技术已经非常成熟**
- ✅ **多个开源项目都有实现**
- ✅ **生产环境广泛应用**
- ✅ **只是可能不够"显眼"**

---

### 2.4 SGLang 的实际使用

**SGLang 已经支持数据并行**：

#### a) **单机多 Worker（数据并行）**

```bash
# 启动 3 个 Worker（在同一台机器上）
python -m sglang.launch_server \
    --model-path /path/to/model \
    --dp-size 3 \
    --load-balance-method shortest_queue
```

#### b) **多机数据并行（通过 Router）**

```bash
# 机器 1: Router
sgl-router \
    --host 0.0.0.0 \
    --port 30000 \
    --policy cache_aware \
    --cache-threshold 0.5 \
    --balance-abs-threshold 32

# 机器 2, 3, 4: Workers
python -m sglang.launch_server \
    --model-path /path/to/model \
    --port 8000 \
    --host 0.0.0.0
```

**Router 配置**：
```yaml
# router_config.yaml
workers:
  - address: "192.168.1.11:8000"
  - address: "192.168.1.12:8000"
  - address: "192.168.1.13:8000"

policy: cache_aware
cache_threshold: 0.5
balance_abs_threshold: 32
balance_rel_threshold: 1.0001
```

---

### 2.4 异构 GPU 环境的挑战与解决方案 ⚠️

**核心问题**：如果医院局域网中每台机器的 GPU 不一样，会不会比较麻烦？

**答案**：**会有挑战，但有解决方案** ✅

#### a) **异构 GPU 带来的问题**

**场景示例**：
```
机器 1: RTX 3060 (8GB, 较低性能)
机器 2: RTX 4090 (24GB, 高性能)
机器 3: A100 (80GB, 极高性能)
```

**主要挑战**：

1. **计算能力差异**
   - RTX 3060: ~25 TFLOPS
   - RTX 4090: ~83 TFLOPS
   - A100: ~312 TFLOPS
   - **问题**：性能差异 3-12 倍，导致负载不均衡

2. **显存大小差异**
   - RTX 3060: 8GB（可能只能运行 7B 模型量化版）
   - RTX 4090: 24GB（可以运行 7B 模型 FP16）
   - A100: 80GB（可以运行 70B 模型）
   - **问题**：不同机器能运行的模型大小不同

3. **延迟差异**
   - 高性能 GPU：处理快（50ms）
   - 低性能 GPU：处理慢（200ms）
   - **问题**：用户请求延迟不一致

4. **兼容性问题**
   - CUDA 版本可能不同
   - 驱动版本可能不同
   - **问题**：软件兼容性

---

#### b) **解决方案**

**方案 1：基于性能的负载均衡（推荐 ✅）**

**原理**：根据 GPU 性能分配不同数量的请求

```python
# 伪代码：基于性能的负载均衡
class PerformanceBasedRouter:
    def __init__(self):
        # 配置每个 Worker 的性能权重
        self.worker_weights = {
            "192.168.1.11:8000": 1.0,   # RTX 3060 (基准)
            "192.168.1.12:8000": 3.0,   # RTX 4090 (3x 性能)
            "192.168.1.13:8000": 12.0,  # A100 (12x 性能)
        }
    
    def select_worker(self, workers):
        # 根据权重分配请求
        # RTX 3060: 1 个请求
        # RTX 4090: 3 个请求
        # A100: 12 个请求
        return weighted_round_robin(workers, self.worker_weights)
```

**实现方式**：

**1. 手动配置权重**
```yaml
# router_config.yaml
workers:
  - address: "192.168.1.11:8000"
    weight: 1.0      # RTX 3060
    gpu_model: "RTX 3060"
    memory_gb: 8
    
  - address: "192.168.1.12:8000"
    weight: 3.0      # RTX 4090
    gpu_model: "RTX 4090"
    memory_gb: 24
    
  - address: "192.168.1.13:8000"
    weight: 12.0     # A100
    gpu_model: "A100"
    memory_gb: 80

policy: weighted_round_robin
```

**2. 自动性能检测**
```python
# Router 自动检测每个 Worker 的性能
def auto_detect_performance(worker_url):
    # 发送测试请求
    test_request = {"prompt": "test", "max_tokens": 10}
    start_time = time.time()
    response = requests.post(f"{worker_url}/generate", json=test_request)
    latency = time.time() - start_time
    
    # 计算性能分数（延迟越低，性能越高）
    performance_score = 1.0 / latency
    return performance_score
```

---

**方案 2：动态 Batch Size 调整**

**原理**：高性能 GPU 处理更大的 batch，低性能 GPU 处理更小的 batch

```python
# 伪代码：动态 batch size
class AdaptiveBatchRouter:
    def __init__(self):
        self.worker_batch_sizes = {
            "192.168.1.11:8000": 4,   # RTX 3060: 小 batch
            "192.168.1.12:8000": 12,  # RTX 4090: 中 batch
            "192.168.1.13:8000": 32,  # A100: 大 batch
        }
    
    def route_request(self, request):
        # 根据 Worker 的 batch size 容量分配
        worker = select_worker_with_capacity(
            self.worker_batch_sizes
        )
        return worker
```

**SGLang 支持**：
```bash
# 每个 Worker 可以配置不同的 max_batch_size
# 机器 1 (RTX 3060)
python -m sglang.launch_server \
    --model-path /path/to/model \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4

# 机器 2 (RTX 4090)
python -m sglang.launch_server \
    --model-path /path/to/model \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 16

# 机器 3 (A100)
python -m sglang.launch_server \
    --model-path /path/to/model \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 64
```

---

**方案 3：请求优先级路由**

**原理**：简单请求给低性能 GPU，复杂请求给高性能 GPU

```python
# 伪代码：基于请求复杂度的路由
class ComplexityBasedRouter:
    def route_request(self, request):
        complexity = estimate_complexity(request)
        
        if complexity == "simple":
            # 短请求 → RTX 3060
            return "192.168.1.11:8000"
        elif complexity == "medium":
            # 中等请求 → RTX 4090
            return "192.168.1.12:8000"
        else:
            # 复杂请求 → A100
            return "192.168.1.13:8000"
    
    def estimate_complexity(self, request):
        # 根据输入长度、输出长度等估算复杂度
        input_len = len(request["prompt"])
        max_tokens = request.get("max_tokens", 100)
        
        if input_len < 100 and max_tokens < 50:
            return "simple"
        elif input_len < 1000 and max_tokens < 500:
            return "medium"
        else:
            return "complex"
```

---

**方案 4：模型量化适配**

**原理**：不同 GPU 运行不同量化版本的模型

```bash
# 机器 1 (RTX 3060, 8GB): 运行 INT4 量化模型
python -m sglang.launch_server \
    --model-path /path/to/llama-7b-int4 \
    --port 8000

# 机器 2 (RTX 4090, 24GB): 运行 INT8 量化模型
python -m sglang.launch_server \
    --model-path /path/to/llama-7b-int8 \
    --port 8000

# 机器 3 (A100, 80GB): 运行 FP16 模型
python -m sglang.launch_server \
    --model-path /path/to/llama-7b-fp16 \
    --port 8000
```

**优点**：
- ✅ 充分利用每台机器的显存
- ✅ 性能差异相对较小（都是 7B 模型）
- ✅ 统一接口（Router 不需要知道模型版本）

---

**方案 5：延迟感知路由**

**原理**：实时监控每个 Worker 的延迟，动态调整路由

```python
# 伪代码：延迟感知路由
class LatencyAwareRouter:
    def __init__(self):
        self.worker_latencies = {}  # 实时延迟统计
    
    def update_latency(self, worker_url, latency):
        # 更新 Worker 的延迟统计
        if worker_url not in self.worker_latencies:
            self.worker_latencies[worker_url] = []
        self.worker_latencies[worker_url].append(latency)
        # 只保留最近 100 个样本
        self.worker_latencies[worker_url] = \
            self.worker_latencies[worker_url][-100:]
    
    def select_worker(self, workers):
        # 选择平均延迟最低的 Worker
        avg_latencies = {
            w: np.mean(self.worker_latencies.get(w, [1000]))
            for w in workers
        }
        return min(avg_latencies, key=avg_latencies.get)
```

---

#### c) **实际部署建议**

**1. 混合策略（推荐）**

```yaml
# router_config.yaml
workers:
  - address: "192.168.1.11:8000"
    weight: 1.0
    max_batch_size: 4
    model_quantization: "int4"
    
  - address: "192.168.1.12:8000"
    weight: 3.0
    max_batch_size: 16
    model_quantization: "int8"
    
  - address: "192.168.1.13:8000"
    weight: 12.0
    max_batch_size: 64
    model_quantization: "fp16"

policy: weighted_latency_aware
# 结合权重和实时延迟
```

**2. 监控和调整**

```python
# 监控每个 Worker 的性能指标
metrics = {
    "worker_1": {
        "avg_latency": 200,  # ms
        "throughput": 5,     # req/s
        "gpu_utilization": 80,
    },
    "worker_2": {
        "avg_latency": 70,   # ms
        "throughput": 15,    # req/s
        "gpu_utilization": 60,
    },
    "worker_3": {
        "avg_latency": 50,   # ms
        "throughput": 20,    # req/s
        "gpu_utilization": 40,
    },
}

# 根据实际性能动态调整权重
adjust_weights(metrics)
```

---

#### d) **兼容性处理**

**1. CUDA 版本统一**

```bash
# 确保所有机器使用相同的 CUDA 版本
# 机器 1, 2, 3 都安装 CUDA 11.8
# 使用 Docker 容器可以保证环境一致
docker run --gpus all \
    -e CUDA_VERSION=11.8 \
    lmsysorg/sglang:latest
```

**2. 驱动版本检查**

```python
# 启动前检查驱动版本
def check_driver_compatibility():
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv"],
        capture_output=True
    )
    # 确保所有机器驱动版本兼容
    return result.stdout
```

---

#### e) **总结：异构 GPU 环境**

**挑战**：
- ⚠️ 性能差异大（3-12 倍）
- ⚠️ 显存大小不同
- ⚠️ 延迟不一致
- ⚠️ 兼容性问题

**解决方案**：
- ✅ **基于性能的负载均衡**（权重分配）
- ✅ **动态 Batch Size 调整**
- ✅ **请求优先级路由**（简单→低性能，复杂→高性能）
- ✅ **模型量化适配**（不同 GPU 运行不同量化版本）
- ✅ **延迟感知路由**（实时监控，动态调整）

**实际效果**：
- ✅ **可以工作**：数据并行在异构环境下仍然可行
- ✅ **需要配置**：需要根据 GPU 性能调整权重和 batch size
- ✅ **性能优化**：通过智能路由，可以最大化整体吞吐量

**结论**：**异构 GPU 环境确实有挑战，但通过合理的配置和路由策略，完全可以应对** ✅

---

#### c) **负载均衡策略**

SGLang Router 支持多种策略：

1. **Random**：随机分配
2. **Round Robin**：轮询分配
3. **Power of Two**：选择两个 Worker，选负载更低的
4. **Cache-Aware**（默认）：
   - 系统平衡时：使用缓存感知路由（提高缓存命中率）
   - 系统不平衡时：使用最短队列路由（负载均衡）

---

### 2.5 其他开源实现

#### a) **Ray Serve（推荐用于生产环境）**

```python
from ray import serve
from sglang import launch_server

# 部署多个副本
@serve.deployment(num_replicas=3)
class LLMService:
    def __init__(self):
        self.model = load_model()
    
    def __call__(self, request):
        return self.model.generate(request)

# 自动负载均衡
serve.run(LLMService.bind())
```

**优点**：
- ✅ 自动负载均衡
- ✅ 动态扩缩容
- ✅ 健康检查
- ✅ 生产级可靠性

#### b) **vLLM 分布式**

```bash
# 使用 vLLM 的多 Worker 模式
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --tensor-parallel-size 1 \
    --worker-use-ray  # 使用 Ray 做分布式
```

---

### 2.6 总结：数据并行已经非常成熟

**核心答案**：

**问题**：有人做了吗？是不是大语言模型的开发研究人员还没开始研究这个？

**答案**：
- ✅ **已经非常成熟**：多个开源项目都有完整实现
- ✅ **生产环境广泛应用**：中国电信、华为云等都有实际应用
- ✅ **SGLang 已经支持**：可以直接使用
- ⚠️ **可能感觉"还没研究"的原因**：
  - 关注点不同（学术界更关注 TP/PP）
  - 宣传不够（工程实践 vs 研究论文）
  - 命名混淆（数据并行听起来像训练概念）

**实际状态**：
- ✅ **技术成熟度**：⭐⭐⭐⭐⭐（5/5）
- ✅ **开源实现**：SGLang, vLLM, Ray Serve 等
- ✅ **生产应用**：多个企业级应用案例
- ✅ **文档完善**：有详细的部署文档

**结论**：**不是"还没开始研究"，而是"已经非常成熟，只是可能不够显眼"** ✅

**缺点**：
- ❌ 每台机器都需要完整模型（内存要求高）
- ❌ 单请求不能加速（但多请求可以并发处理）

**适用场景**：**医院多用户同时使用** ✅

---

#### 方案 2：张量并行（不推荐 ❌）

**为什么不适合**：
- ❌ **网络延迟高**：千兆以太网延迟 ~1ms，每层都要通信，累积延迟大
- ❌ **通信开销大**：每层都要传输大量数据（激活值）
- ❌ **容错性差**：一台机器故障，整个请求失败
- ❌ **实现复杂**：需要 NCCL 等通信库，配置复杂

**性能估算**：
```
单层通信时间：~10ms（千兆网）
模型层数：24 层
总通信时间：24 × 10ms = 240ms
GPU 计算时间：~50ms
总延迟：290ms（比单机还慢！）
```

**结论**：**不适合普通局域网** ❌

---

#### 方案 3：流水线并行（部分可行 ⚠️）

**适用条件**：
- ✅ 模型太大，单机内存不够
- ✅ 多请求批处理（提高 GPU 利用率）
- ⚠️ 网络延迟要低（< 5ms）

**实现方式**：
```python
# 伪代码：流水线并行
# 机器 1: Layer 0-7
python -m sglang.launch_server \
    --model-path /path/to/model \
    --pipeline-parallel-size 3 \
    --pipeline-parallel-rank 0

# 机器 2: Layer 8-15
python -m sglang.launch_server \
    --model-path /path/to/model \
    --pipeline-parallel-size 3 \
    --pipeline-parallel-rank 1

# 机器 3: Layer 16-23
python -m sglang.launch_server \
    --model-path /path/to/model \
    --pipeline-parallel-size 3 \
    --pipeline-parallel-rank 2
```

**适用场景**：**超大模型 + 多请求批处理**

---

## 3. 实际部署方案

### 3.1 推荐方案：数据并行 + 负载均衡

**架构**：
```
┌─────────────────────────────────────────┐
│  客户端（医院用户）                        │
└─────────────────────────────────────────┘
           ↓ HTTP 请求
┌─────────────────────────────────────────┐
│  Router / 负载均衡器（机器 1）             │
│  - 接收请求                               │
│  - 负载均衡（轮询/最短队列）                │
│  - 转发到 Worker                         │
└─────────────────────────────────────────┘
    ↓              ↓              ↓
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Worker1 │   │ Worker2 │   │ Worker3 │
│ 机器 2   │   │ 机器 3   │   │ 机器 4   │
│ 完整模型 │   │ 完整模型 │   │ 完整模型 │
└─────────┘   └─────────┘   └─────────┘
```

**实现步骤**：

#### Step 1: 在每个机器上部署模型

```bash
# 机器 2, 3, 4 上分别运行
cd /path/to/sglang
python -m sglang.launch_server \
    --model-path /path/to/llama-7b \
    --port 8000 \
    --host 0.0.0.0
```

#### Step 2: 部署 Router（负载均衡）

```python
# router.py（运行在机器 1）
from fastapi import FastAPI
import requests
import random

app = FastAPI()

# Worker 列表
workers = [
    "http://192.168.1.11:8000",  # 机器 2
    "http://192.168.1.12:8000",  # 机器 3
    "http://192.168.1.13:8000",  # 机器 4
]

@app.post("/v1/chat/completions")
async def chat(request: dict):
    # 负载均衡：轮询或最短队列
    worker = select_worker(workers)  # 选择最空闲的 worker
    
    # 转发请求
    response = requests.post(
        f"{worker}/v1/chat/completions",
        json=request
    )
    
    return response.json()

def select_worker(workers):
    # 简单轮询
    return random.choice(workers)
    # 或：选择最短队列的 worker
    # return min(workers, key=get_queue_length)
```

#### Step 3: 客户端使用

```python
# 客户端代码
import requests

response = requests.post(
    "http://192.168.1.10:8000/v1/chat/completions",  # Router 地址
    json={
        "model": "llama-7b",
        "messages": [{"role": "user", "content": "你好"}]
    }
)
```

---

### 3.2 性能优化

#### a) **负载均衡策略**

**1. 轮询（Round Robin）**
```python
current_worker = 0
def select_worker(workers):
    global current_worker
    worker = workers[current_worker]
    current_worker = (current_worker + 1) % len(workers)
    return worker
```

**2. 最短队列（Shortest Queue）**
```python
def select_worker(workers):
    # 查询每个 worker 的队列长度
    queue_lengths = [get_queue_length(w) for w in workers]
    return workers[queue_lengths.index(min(queue_lengths))]
```

**3. 最少 Token（Minimum Tokens）**
```python
def select_worker(workers):
    # 查询每个 worker 的当前 token 数
    token_counts = [get_token_count(w) for w in workers]
    return workers[token_counts.index(min(token_counts))]
```

#### b) **缓存优化**

**Prefix Cache 共享**（如果使用 SGLang）：
```python
# Router 可以做前缀匹配，将相似请求路由到同一 Worker
# 提高缓存命中率
def select_worker(workers, request):
    prefix = extract_prefix(request)
    cached_worker = get_cached_worker(prefix)
    if cached_worker:
        return cached_worker
    return select_worker(workers)
```

---

## 4. 技术资源参考

### 4.1 相关技术帖子

**1. TinyML 低功耗 AI**
- 百度云文章：TinyML 技术在边缘设备上的应用
- 关键词：模型压缩、量化、剪枝

**2. 分布式 LLM 推理**
- SGLang 多节点部署文档：`docs/references/multi_node_deployment/multi_node.md`
- Kubernetes 分布式部署示例：`docker/k8s-sglang-distributed-sts.yaml`

**3. 边缘 AI 芯片**
- YouTube 视频：低功耗，大算力！最适合大模型的 AI 芯片

---

### 4.2 开源项目

**1. SGLang**
- 支持多节点分布式推理
- 支持数据并行、张量并行、流水线并行
- GitHub: https://github.com/sgl-project/sglang

**2. vLLM**
- 高性能 LLM 推理引擎
- 支持分布式推理
- GitHub: https://github.com/vllm-project/vllm

**3. TensorFlow Lite**
- 边缘设备推理框架
- 支持模型量化
- 适合 TinyML 场景

---

## 5. 性能对比

### 5.1 单机 vs 数据并行（3 台机器）

| 场景 | 单机 | 数据并行（3台） | 提升 |
|------|------|----------------|------|
| **单请求延迟** | 100ms | 100ms | 无变化 |
| **3 个并发请求** | 300ms | 100ms | 3x ✅ |
| **10 个并发请求** | 1000ms | ~333ms | 3x ✅ |
| **吞吐量（req/s）** | 10 | 30 | 3x ✅ |

**结论**：数据并行**不能加速单请求**，但可以**提高并发处理能力** ✅

---

### 5.2 网络要求

| 方案 | 网络带宽要求 | 网络延迟要求 | 适用场景 |
|------|------------|------------|---------|
| **数据并行** | 低（HTTP 请求） | 低（< 10ms） | 普通局域网 ✅ |
| **张量并行** | 高（GB/s） | 极低（< 1ms） | InfiniBand |
| **流水线并行** | 中（MB/s） | 低（< 5ms） | 高速局域网 |

---

## 6. 实际部署建议

### 6.1 医院局域网场景

**推荐方案**：**数据并行 + 负载均衡**

**理由**：
1. ✅ **实现简单**：每台机器独立运行，配置简单
2. ✅ **容错性好**：单机故障不影响整体服务
3. ✅ **适合多用户**：多个医生/护士同时使用
4. ✅ **网络要求低**：普通千兆网即可

**部署步骤**：
1. 在每台机器上部署模型（7B 模型约需 14GB 内存）
2. 部署 Router 做负载均衡
3. 客户端连接到 Router

---

### 6.2 性能优化建议

**1. 模型选择**
- 使用量化模型（INT8/INT4）减少内存占用
- 7B 模型 INT8 量化后约 7GB

**2. 硬件要求**
- 每台机器至少 16GB 内存
- 如果有 GPU，可以加速（但 CPU 也可以运行）

**3. 网络优化**
- 使用有线网络（避免 WiFi 延迟）
- 确保网络带宽充足（千兆网足够）

**4. 缓存优化**
- 使用 Prefix Cache（SGLang 支持）
- 相似请求路由到同一 Worker

---

## 7. 总结

### 7.1 核心答案

**问题**：医院的局域网，能不能让局域网的所有机器同时帮忙跑 token，加速本地 LLM？

**答案**：
- ✅ **可以，但需要正确的策略**
- ✅ **推荐：数据并行**（每台机器运行完整模型，处理不同请求）
- ❌ **不推荐：张量并行**（需要高速网络，普通局域网不适合）

### 7.2 关键要点

1. **数据并行**：适合多请求并发，不能加速单请求
2. **张量并行**：需要高速网络（InfiniBand），普通局域网不适合
3. **流水线并行**：适合超大模型，但需要多请求批处理

### 7.3 实际应用

**医院场景**：
- ✅ 多用户同时使用 → 数据并行 ✅
- ✅ 普通千兆网 → 数据并行 ✅
- ✅ 容错要求高 → 数据并行 ✅

**结论**：**数据并行是最适合医院局域网场景的方案** ✅

---

**参考资源**：
- SGLang 多节点部署文档
- vLLM 分布式推理文档
- TinyML 技术文档
- 分布式系统设计最佳实践
