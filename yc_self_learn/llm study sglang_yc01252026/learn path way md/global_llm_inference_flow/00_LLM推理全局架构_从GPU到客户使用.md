# LLM 推理全局架构：从 GPU 到客户使用

## 📚 文档说明

**本文档位于**: `learn path way md/global_llm_inference_flow/`

**目标读者**: 需要建立 LLM 推理「端到端全局图景」的工程师与学习者。

**阅读后你将**:
- 理解从 GPU 算力到用户屏幕的完整数据流与职责分层
- 知道每一层「谁在做什么、输入输出是什么」
- 能把 SGLang / vLLM 等具体实现对号入座到全局架构中

---

## 🎯 一、全局图景（一句话 + 一张图）

**一句话**：用户发来一段话 → 经过网关/路由与调度 → 在 GPU 上做 Token 化、Prefill、逐 Token Decode → 结果再经后处理与网络返回，最终呈现在客户端。

**一张图**：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            用 户 / 客 户 端                                        │
│  (浏览器 / App / API 调用方)                                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ① 接入层 (Entry / API)                                                            │
│  HTTP/gRPC、OpenAI 兼容 API、鉴权、限流、路由到具体实例                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ② 调度层 (Router / Scheduler)                                                     │
│  请求排队、批处理(Batching)、调度到 Worker、优先级、多租户                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ③ 运行时 / 执行层 (Runtime / Execution)                                           │
│  Prefill / Decode 编排、Pipeline Stage、内存与 KV Cache 管理、并行策略 (TP/PP/SP)    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ④ 模型与计算层 (Model + Compute)                                                  │
│  模型权重加载、Embedding、Layer Forward、Attention(KV Cache)、FFN、Sampling          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ⑤ 硬件层 (GPU / Kernel)                                                          │
│  CUDA/Triton Kernel、Tensor Core、HBM、算力与带宽                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**数据流方向**：用户 → ① → ② → ③ → ④ → ⑤（前向），再 ⑤ → ④ → ③ → ② → ① → 用户（返回）。

---

## 📋 二、各层职责与输入输出（从下往上讲，便于理解「谁依赖谁」）

### ⑤ 硬件层：GPU 与 Kernel

**职责**：
- 提供算力（矩阵乘、Attention、激活）与高带宽内存（HBM）
- 执行具体 Kernel（如 FlashAttention、GEMM、LayerNorm）

**输入**：来自上一层的 GPU 张量（例如 Q/K/V、权重块、KV Cache 指针）  
**输出**：算好的张量（logits、hidden states、next token logits）

**典型技术**：CUDA、Triton、FlashInfer、cuBLAS、Tensor Core、NVLink/NVSwitch（多卡）

**详见**：[01_硬件层_GPU与Kernel详解_行业明星与最佳实践.md](./01_硬件层_GPU与Kernel详解_行业明星与最佳实践.md)（细节、最佳公司/案例、行业 super star）

---

### ④ 模型与计算层（Model + Compute）

**职责**：
- 加载并持有模型权重（含量化、分片）
- 定义并执行「一层一层」的前向：Embedding → N × (Attention + FFN) → LM Head → logits
- 管理 KV Cache 的读写接口（实际内存可在运行时层）
- 采样（temperature、top_p、top_k）得到 next token id

**输入**：token ids（或 embeddings）、KV Cache 状态、采样参数  
**输出**：next token id(s)、更新后的 KV Cache、可选 logits

**典型实现**：SGLang/vLLM 的 ModelRunner、单层 Forward、Attention Backend（FlashInfer/Triton 等）

---

### ③ 运行时 / 执行层（Runtime / Execution）

**职责**：
- 把「多个请求」组织成 batch，决定当前 step 做 Prefill 还是 Decode
- 管理 KV Cache 的分配、释放、分页（Paged KV）、前缀复用（RadixAttention 等）
- 编排多 Stage（Prefill vs Decode）、多卡并行（TP/PP/SP）、通信（all-reduce、all-to-all）
- 与调度层约定「接多少请求、何时阻塞」

**输入**：来自调度层的一批请求（含 prompt、采样参数、request_id 等）  
**输出**：生成的 token 流、完成/取消状态、可选的 timing 信息

**典型实现**：SGLang 的 Scheduler、MemCache、Pipeline、Worker 内执行循环

---

### ② 调度层（Router / Scheduler）

**职责**：
- 接收接入层转发的请求，排队、按策略组成 batch
- 决定把 batch 发给哪个 Worker（多实例时做负载均衡与路由）
- 管理优先级、限流、多租户配额，避免单用户拖垮系统

**输入**：来自接入层的请求（可能带 tenant_id、priority、deadline）  
**输出**：发给运行时的「可执行 batch」或等待信号

**典型实现**：SGLang Router、内部 Scheduler 的队列与 batch 构造逻辑

---

### ① 接入层（Entry / API）

**职责**：
- 暴露 HTTP/gRPC/WebSocket 等 API（常见 OpenAI 兼容）
- 鉴权、限流、请求校验、协议转换
- 将请求转发到调度层（或直接到单实例的运行时）

**输入**：客户端 HTTP 请求体（如 `prompt`、`max_tokens`、`stream: true`）  
**输出**：转发给调度层的内部请求结构；对客户端返回 JSON/SSE 流

**典型实现**：FastAPI/Starlette 服务、OpenAI-compatible routes、Gateway 组件

---

### 用户 / 客户端

**职责**：发起请求（同步或流式）、展示结果、处理错误与重试。

**与系统的边界**：以「HTTP 请求 / 响应」或「WebSocket 消息」为界；不关心内部是单卡还是多卡、是否用 FlashInfer。

---

## 🔄 三、单次请求的端到端时序（简化）

1. **用户** 在客户端输入 prompt，点击发送。
2. **接入层** 校验、鉴权，构造内部请求，交给 **调度层**。
3. **调度层** 将请求放入队列，凑成 batch（或立即调度），交给 **运行时**。
4. **运行时** 为请求分配 KV Cache、决定 Prefill/Decode：
   - **Prefill**：整段 prompt 一次前向，填满 KV Cache，得到第一个 token。
   - **Decode**：每次用当前 token + KV Cache 算下一个 token，直到 EOS 或达到 max_tokens。
5. **模型层** 在 **GPU** 上执行 Attention（读/写 KV）、FFN、LM Head、Sampling。
6. 每个（或每批）token 从 **运行时** 经 **调度层**、**接入层** 返回 **客户端**（流式则边算边推）。
7. 请求结束后，**运行时** 释放 KV Cache，**调度层** 从队列移除该请求。

---

## 📐 四、与 SGLang 的对应关系（帮助对号入座）

| 全局层级       | SGLang 中大致对应                          |
|----------------|--------------------------------------------|
| ① 接入层      | HTTP server、OpenAI API 兼容入口           |
| ② 调度层      | Router、Scheduler、请求队列与 batching    |
| ③ 运行时      | Scheduler 与 Worker 内的执行循环、MemCache、KV 分配、RadixAttention |
| ④ 模型与计算  | ModelRunner、Layer、Attention Backend、Sampling |
| ⑤ 硬件/Kernel | FlashInfer、Triton kernel、CUDA、GPU       |

学习 SGLang 时，可随时用「当前在看的是哪一层」来锚定自己在全局中的位置。

---

## ✅ 五、自测：你是否建立了全局图景

- [ ] 能画出从「用户输入」到「用户看到输出」的 5 层结构。
- [ ] 能说出每一层的输入、输出和主要职责。
- [ ] 能解释 Prefill 和 Decode 在「层」中的位置（运行时编排 + 模型/GPU 执行）。
- [ ] 能把 SGLang 的 Router、Scheduler、MemCache、Attention Backend 对号入座到某一层。

---

## 📎 六、本系列后续文档建议

- **01_接入层与调度层详解.md**：API、Router、Batching、多租户。
- **02_运行时与KV_Cache详解.md**：Prefill/Decode、Paged KV、RadixAttention。
- **03_模型层与Attention详解.md**：Forward、FlashInfer、Sampling。
- **04_从GPU到用户的完整数据流.md**：一次请求的详细 trace 与关键指标（TTFT、TPOT、吞吐）。

---

*文档版本：v1 | 建议与 SGLang 学习路径、Case Study 配合使用。*
