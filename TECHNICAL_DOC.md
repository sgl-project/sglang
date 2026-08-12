# SGLang 技术文档

> 基于 `python/sglang/srt/` 源码分析，版本：main 分支

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 整体架构](#2-整体架构)
- [3. 核心执行链路](#3-核心执行链路)
- [4. 三大核心组件详解](#4-三大核心组件详解)
- [5. 内存管理与 KV Cache](#5-内存管理与-kv-cache)
- [6. 调度策略](#6-调度策略)
- [7. 分布式并行](#7-分布式并行)
- [8. 推测解码 (Speculative Decoding)](#8-推测解码)
- [9. Prefill-Decode 分离 (Disaggregation)](#9-prefill-decode-分离)
- [10. 关键优化技术](#10-关键优化技术)
- [11. 目录结构速查](#11-目录结构速查)

---

## 1. 项目概述

SGLang 是一个高性能 LLM 推理服务框架，核心特性：

- **RadixAttention**：基于 Radix Tree 的前缀缓存，实现 prompt 级 KV Cache 复用
- **零开销调度器**：CPU 调度与 GPU 计算重叠，最大化 GPU 利用率
- **多策略调度**：LPM（最长前缀匹配）、FCFS、DFS-Weight 等
- **分布式并行**：TP/PP/EP/DP/CP 全面支持
- **推测解码**：EAGLE、N-gram、DFLASH 等多种算法
- **Prefill-Decode 分离**：PD 架构支持大规模部署
- **结构化输出**：JSON Schema / Regex / EBNF 约束解码

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        HTTP Server (FastAPI)                     │
│                    http_server.py / openai/ / anthropic/         │
└─────────────────────────────┬───────────────────────────────────┘
                              │ ZMQ IPC
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TokenizerManager (主进程)                    │
│              tokenizer_manager.py                                │
│  • 请求解析 + Tokenize                                          │
│  • 多模态数据预处理                                               │
│  • 流式输出管理                                                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │ ZMQ IPC
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Scheduler (子进程)                           │
│              scheduler.py                                        │
│  • 请求调度 + 批处理管理                                           │
│  • 调度策略选择 (LPM/FCFS/DFS)                                    │
│  • KV Cache 管理 + RadixCache                                    │
│  • Overlap 调度 (CPU/GPU 重叠)                                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TpModelWorker                                │
│              tp_worker.py                                        │
│  • Tensor Parallel Worker                                        │
│  • ScheduleBatch → ModelWorkerBatch → ForwardBatch              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ModelRunner                                  │
│              model_runner.py                                     │
│  • 模型加载 + GPU 内存管理                                        │
│  • CUDA Graph 捕获                                               │
│  • Forward Pass 执行                                             │
│  • Attention Backend (FlashInfer/FA3/FA4/Triton/...)            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DetokenizerManager (子进程)                  │
│              detokenizer_manager.py                              │
│  • Token ID → 文本                                               │
│  • 增量解码                                                       │
│  • 流式输出                                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 进程模型

| 组件 | 进程 | 通信方式 |
|------|------|---------|
| HTTP Server + Engine | 主进程 | - |
| TokenizerManager | 主进程 | ZMQ IPC (PUSH/PULL) |
| Scheduler | 子进程 | ZMQ IPC |
| DetokenizerManager | 子进程 | ZMQ IPC |
| TP Worker | Scheduler 内嵌 | 共享内存 |

---

## 3. 核心执行链路

### 3.1 请求入口链路

```
HTTP Request
  → http_server.py (FastAPI 路由)
    → Engine.generate() / Engine.async_generate()
      → TokenizerManager.generate_request()
        → tokenize + 构建 GenerateReqInput
          → ZMQ send to Scheduler
```

### 3.2 Scheduler 主循环

```python
# scheduler.py::event_loop_normal()
while True:
    # 1. 接收请求
    recv_reqs = self.recv_requests()          # ZMQ PULL
    self.process_input_requests(recv_reqs)

    # 2. 调度下一批
    batch = self.get_next_batch_to_run()
    self.cur_batch = batch

    # 3. 执行 forward
    if batch:
        result = self.run_batch(batch)         # GPU forward
        self.process_batch_result(batch, result)

    # 4. 更新状态
    self.last_batch = batch
```

### 3.3 Overlap 调度（核心优化）

```python
# scheduler.py::event_loop_overlap()
while True:
    recv_reqs = self.recv_requests()
    self.process_input_requests(recv_reqs)

    # 获取下一批
    batch = self.get_next_batch_to_run()

    # 启动当前 batch 的 GPU 计算
    if batch:
        batch_result = self.run_batch(batch)

    # 处理上一批的结果（与当前 GPU 计算重叠）
    if self.last_batch:
        pop_and_process()  # CPU 处理上一批结果

    # 采样（依赖上一批的 grammar 结果）
    self.launch_batch_sample_if_needed(batch_result)
```

### 3.4 数据结构流转

```
ScheduleBatch (CPU) → ModelWorkerBatch → ForwardBatch (GPU)
```

| 数据结构 | 管理者 | 位置 | 说明 |
|----------|--------|------|------|
| `ScheduleBatch` | Scheduler | CPU | 高级调度数据，包含所有请求信息 |
| `ModelWorkerBatch` | TpModelWorker | CPU→GPU | ScheduleBatch 的子集，仅含 forward 相关数据 |
| `ForwardBatch` | ModelRunner | GPU | 低级 tensor 数据，直接用于模型 forward |

---

## 4. 三大核心组件详解

### 4.1 TokenizerManager

**文件**: `tokenizer_manager.py`

**职责**:
- 请求解析和 Tokenize
- 多模态数据预处理（图像/视频/音频）
- 流式输出管理（SSE）
- Session 管理
- LoRA 适配器热加载

**关键流程**:
```python
async def generate_request(self, request: GenerateReqInput, ...):
    # 1. Tokenize 输入
    tokenized = await self.tokenize(request)

    # 2. 多模态处理
    if request.multimodal_inputs:
        mm_data = await self.process_multimodal(request)

    # 3. 发送到 Scheduler
    await self.send_to_scheduler(tokenized)

    # 4. 等待结果并流式返回
    async for output in self.wait_for_result(request_id):
        yield output
```

### 4.2 Scheduler

**文件**: `scheduler.py` (3800+ 行)

**职责**:
- 请求调度和批处理管理
- KV Cache 分配和管理
- RadixCache 前缀缓存
- Overlap 调度
- 分布式协调（TP/PP/EP/DP）

**关键方法**:
```python
class Scheduler:
    def event_loop_normal(self):      # 普通调度循环
    def event_loop_overlap(self):     # Overlap 调度循环
    def get_next_batch_to_run(self):  # 获取下一批
    def run_batch(self, batch):       # 执行 forward
    def process_batch_result(self):   # 处理结果
    def recv_requests(self):          # 接收请求
```

**初始化流程**:
```python
def __init__(self, server_args, port_args, gpu_id, tp_rank, ...):
    self.init_model_config()           # 模型配置
    self.init_ipc_channels(port_args)  # ZMQ 通信
    self.init_tokenizer()              # Tokenizer
    self.init_tp_model_worker()        # TP Worker
    self.init_cache_with_memory_pool() # KV Cache
    self.init_running_status()         # 运行状态
    self.init_schedule_policy()        # 调度策略
    self.init_overlap()                # Overlap 调度
    self.init_disaggregation()         # PD 分离
```

### 4.3 ModelRunner

**文件**: `model_runner.py` (3300+ 行)

**职责**:
- 模型加载和初始化
- GPU 内存管理
- CUDA Graph 捕获
- Forward Pass 执行
- Attention Backend 管理

**关键流程**:
```python
class ModelRunner:
    def initialize(self):
        self.load_model()                    # 加载模型权重
        self.init_memory_pool()              # 初始化 KV Cache
        self.capture_cuda_graph()            # 捕获 CUDA Graph

    def forward_batch_generation(self, forward_batch):
        # 1. 准备输入
        # 2. 执行模型 forward
        # 3. Logits 处理
        # 4. 采样
        # 5. 返回输出
```

---

## 5. 内存管理与 KV Cache

### 5.1 两级内存池

```
ReqToTokenPool (请求→Token 映射)
    ↓
TokenToKVPoolAllocator (Token→KV Cache 索引)
    ↓
KVCache (实际 KV 数据)
```

**文件**: `mem_cache/memory_pool.py`

```python
# 两级内存池
self.req_to_token_pool        # req_id → token_positions
self.token_to_kv_pool_allocator  # token_index → kv_cache_slot
```

### 5.2 RadixCache（核心创新）

**文件**: `mem_cache/radix_cache.py`

基于 Radix Tree 的前缀缓存，实现 prompt 级 KV Cache 复用：

```
Radix Tree 结构:
         [root]
        /      \
   [sys prompt]  [user msg 1]
      |              |
   [tokens 0-100]  [tokens 101-200]
```

**核心操作**:
- `match_prefix()`: 查找最长前缀匹配
- `insert()`: 插入新的 KV Cache
- `evict()`: 缓存淘汰（LRU/LFU/FIFO 等策略）
- `lock_ref()` / `dec_lock_ref()`: 引用计数管理

**支持的缓存实现**:
| 实现 | 说明 |
|------|------|
| `RadixCache` | 默认 Radix Tree 实现 |
| `RadixCacheCpp` | C++ 实验性实现 |
| `ChunkCache` | 分块缓存（无 Radix Tree） |
| `HiRadixCache` | 分层缓存（GPU + CPU） |
| `SWARadixCache` | Sliding Window Attention 缓存 |
| `UnifiedRadixCache` | 统一缓存（支持多种组件） |

### 5.2 Paged Attention

SGLang 实现了 Paged Attention，将 KV Cache 分为固定大小的 page：
- `page_size` 参数控制（默认 16）
- 支持 non-contiguous 存储
- 与 RadixCache 协同工作

---

## 6. 调度策略

**文件**: `managers/schedule_policy.py`

### 6.1 Cache-Aware 策略

| 策略 | 说明 |
|------|------|
| **LPM** (Longest Prefix Match) | 优先调度与缓存前缀最长匹配的请求 |
| **DFS-Weight** | 深度优先搜索加权调度 |

### 6.2 Cache-Agnostic 策略

| 策略 | 说明 |
|------|------|
| **FCFS** (First Come First Serve) | 先来先服务 |
| **LOF** (Longest Output First) | 最长输出优先 |
| **RANDOM** | 随机调度 |
| **ROUTING_KEY** | 按路由键频率优先 |

### 6.3 调度流程

```python
def get_next_batch_to_run(self):
    # 1. 检查 running batch 是否可以继续 decode
    if self.running_batch and self.can_run_decode():
        return self.running_batch  # Continuous Batching

    # 2. 从 waiting queue 调度新请求
    new_batch = self.policy.get_priority_queue(self.waiting_queue)

    # 3. 检查资源（GPU 内存、token 限制）
    if not self.check_resources(new_batch):
        return None

    # 4. 执行 prefix cache 匹配
    if self.tree_cache:
        new_batch = self.match_prefix(new_batch)

    return new_batch
```

### 6.4 Chunked Prefill

将长 prompt 分块处理，避免单次 prefill 过长：
- `chunked_prefill_size` 参数控制
- 支持与 decode batch 混合执行（Mixed Chunk）

---

## 7. 分布式并行

### 7.1 并行策略

| 策略 | 说明 | 文件 |
|------|------|------|
| **Tensor Parallel (TP)** | 张量并行，切分注意力头 | `distributed/` |
| **Pipeline Parallel (PP)** | 流水线并行，切分层 | `managers/scheduler_pp_mixin.py` |
| **Expert Parallel (EP)** | 专家并行，MoE 模型 | `layers/moe/` |
| **Data Parallel (DP)** | 数据并行 | `managers/data_parallel_controller.py` |
| **Context Parallel (CP)** | 上下文并行 | `layers/attention/` |
| **DP Attention** | DP 级别的注意力 | `layers/dp_attention.py` |

### 7.2 通信

**文件**: `distributed/`

- 使用 PyTorch Distributed (NCCL)
- 支持多种通信后端：NCCL、Gloo、CCL
- ZMQ IPC 用于进程间通信

### 7.3 Expert Parallelism (EP)

**文件**: `eplb/`, `elastic_ep/`

- 支持 Expert Load Balancing (EPLB)
- Elastic EP 支持动态专家分配
- Expert Distribution Recorder 记录专家使用情况

---

## 8. 推测解码

**文件**: `speculative/`

### 8.1 支持的算法

| 算法 | 说明 | 文件 |
|------|------|------|
| **EAGLE** | 基于特征的推测解码 | `eagle_worker.py` |
| **EAGLE v2** | 增强版 EAGLE | `eagle_worker_v2.py` |
| **Multi-Layer EAGLE** | 多层 EAGLE | `multi_layer_eagle_worker.py` |
| **N-gram** | N-gram 预测 | `ngram_worker.py` |
| **DFLASH** | 跨 KV Transfer 推测解码 | `dflash_worker.py` |
| **Standalone** | 独立 draft 模型 | `standalone_worker.py` |

### 8.2 EAGLE 流程

```
1. Draft Model 生成 K 个候选 token
2. Target Model 并行验证 K 个 token
3. 接受匹配的 token，拒绝不匹配的
4. 重复直到完成
```

### 8.3 自适应推测解码

**文件**: `adaptive_runtime_state.py`, `adaptive_spec_params.py`

- 根据验证通过率动态调整 draft 长度
- 监控 speculative decoding 的效率

---

## 9. Prefill-Decode 分离

**文件**: `disaggregation/`

### 9.1 架构

```
┌─────────────┐         KV Transfer         ┌─────────────┐
│  Prefill     │ ──────────────────────────→ │  Decode      │
│  Engine      │    (RDMA/NIXL/Mooncake)     │  Engine      │
└─────────────┘                              └─────────────┘
```

### 9.2 Transfer Backend

| Backend | 说明 | 文件 |
|---------|------|------|
| **NIXL** | NVIDIA NIXL | `nixl/` |
| **Mooncake** | Mooncake KV Transfer | `mooncake/` |
| **MoRI** | MoRI Transfer | `mori/` |
| **gRPC** | gRPC Transfer | `encode_grpc_server.py` |

### 9.3 流程

**Prefill 端**:
1. 执行 prefill 计算
2. 将 KV Cache 序列化
3. 通过 Transfer Backend 发送 KV

**Decode 端**:
1. 接收 KV Cache
2. 分配 GPU 内存
3. 开始 decode

---

## 10. 关键优化技术

### 10.1 CUDA Graph

**文件**: `model_executor/cuda_graph_runner.py`

- 捕获固定 shape 的 forward pass
- 减少 kernel launch 开销
- 支持 Breakable CUDA Graph（动态 batch）

### 10.2 Attention Backend

**文件**: `layers/attention/`

支持多种 Attention 实现：
- FlashInfer
- FlashAttention v3/v4
- Triton
- CUTLASS MLA
- TRT-LLM MLA

### 10.3 量化支持

**文件**: `layers/quantization/`

- FP8 (E4M3/E5M2)
- FP4
- INT4 (AWQ/GPTQ)
- GPTQ
- SmoothQuant

### 10.4 结构化输出

**文件**: `constrained/`

- JSON Schema 约束
- Regex 约束
- EBNF 约束
- 压缩有限状态机 (Compressed FSM)

### 10.5 LoRA

**文件**: `lora/`

- 多 LoRA 批处理
- LoRA 热加载/卸载
- LoRA Overlap Loading

---

## 11. 目录结构速查

```
python/sglang/srt/
├── entrypoints/          # HTTP 服务入口
│   ├── engine.py         # Engine 核心（三组件启动）
│   ├── http_server.py    # FastAPI HTTP 服务
│   ├── openai/           # OpenAI API 兼容
│   └── anthropic/        # Anthropic API 兼容
│
├── managers/             # 管理层
│   ├── scheduler.py      # 调度器（3800+ 行核心）
│   ├── tokenizer_manager.py  # Tokenizer 管理
│   ├── detokenizer_manager.py # Detokenizer 管理
│   ├── tp_worker.py      # Tensor Parallel Worker
│   ├── schedule_batch.py # 批处理数据结构
│   ├── schedule_policy.py # 调度策略
│   ├── io_struct.py      # IPC 数据结构
│   └── data_parallel_controller.py # DP 控制器
│
├── model_executor/       # 模型执行层
│   ├── model_runner.py   # 模型运行器（3300+ 行）
│   ├── forward_batch_info.py # Forward Batch 信息
│   ├── cuda_graph_runner.py # CUDA Graph
│   └── breakable_cuda_graph/ # 可中断 CUDA Graph
│
├── mem_cache/            # 内存和缓存管理
│   ├── memory_pool.py    # 两级内存池
│   ├── radix_cache.py    # Radix Tree Cache
│   ├── allocator.py      # Token 分配器
│   ├── hiradix_cache.py  # 分层缓存
│   └── evict_policy.py   # 淘汰策略
│
├── layers/               # 模型层
│   ├── attention/        # Attention 实现
│   ├── moe/              # MoE 层
│   ├── quantization/     # 量化
│   ├── radix_attention.py # RadixAttention
│   └── linear.py         # 线性层
│
├── speculative/          # 推测解码
│   ├── eagle_worker.py   # EAGLE
│   ├── ngram_worker.py   # N-gram
│   └── dflash_worker.py  # DFLASH
│
├── disaggregation/       # PD 分离
│   ├── prefill.py        # Prefill 端
│   ├── decode.py         # Decode 端
│   ├── nixl/             # NIXL Transfer
│   └── mooncake/         # Mooncake Transfer
│
├── distributed/          # 分布式通信
│   ├── parallel_state.py # 并行状态
│   └── device_communicators/ # 设备通信
│
├── sampling/             # 采样
├── constrained/          # 约束解码
├── lora/                 # LoRA
├── configs/              # 模型配置
├── eplb/                 # Expert Parallel Load Balancing
├── elastic_ep/           # Elastic Expert Parallel
└── observability/        # 可观测性
```

---

## 附录：关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunked_prefill_size` | - | 分块 prefill 大小 |
| `max_running_requests` | - | 最大运行请求数 |
| `schedule_policy` | "lpm" | 调度策略 |
| `disable_overlap_schedule` | False | 禁用 overlap 调度 |
| `enable_dp_attention` | False | 启用 DP Attention |
| `page_size` | 16 | Paged Attention page 大小 |
| `disable_radix_cache` | False | 禁用 RadixCache |
| `speculative_algorithm` | None | 推测解码算法 |
| `disaggregation_mode` | None | PD 分离模式 |

---

*文档生成时间：2026-07-10*
*基于 SGLang main 分支源码分析*
