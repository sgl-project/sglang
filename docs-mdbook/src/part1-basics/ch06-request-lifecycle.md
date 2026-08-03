# 第 6 章 一次请求的完整生命周期

> 这是全书最重要的骨架章节。建议配合 `managers/tokenizer_manager.py`、`managers/scheduler.py`、`model_executor/model_runner.py` 一起读。

## 6.1 全景图

以一个文本生成请求 `POST /v1/chat/completions` 为例，从进入进程到返回结果，要经过下面 8 个环节：

```text
┌────────────┐  1. FastAPI 入口       ┌──────────────────┐
│  客户端     │ ─────────────────────▶ │ http_server.py   │
└────────────┘                        └────────┬─────────┘
                                    2. 协议适配    │ ChatCompletionRequest
                                     (serving_chat)│ → GenerateReqInput
                                                  ▼
                                    ┌──────────────────┐
                                    │ TokenizerManager │  3. tokenize + 校验
                                    │ (tokenizer_      │     采样参数、多模态
                                    │  manager.py)     │     会话状态
                                    └────────┬─────────┘
                         4. ZMQ 投递          │ BatchTokenizedGenerateReqInput
                                             ▼
                                    ┌──────────────────┐
                                    │ Scheduler 进程    │  5. 调度循环
                                    │ (scheduler.py)    │     入 waiting queue
                                    └────────┬─────────┘     组 batch
                                             │ ScheduleBatch
                                             ▼
                                    ┌──────────────────┐
                                    │ ModelRunner      │  6. GPU 前向
                                    │ (model_runner.py)│     prefill/decode
                                    └────────┬─────────┘
                                             │ 新 token / logprob
                                             ▼
                                    ┌──────────────────┐
                                    │ Detokenizer 进程  │  7. token → 文本
                                    │ (detokenizer_    │
                                    │  manager.py)     │
                                    └────────┬─────────┘
                                             │ 文本增量
                                             ▼
                                    ┌──────────────────┐
                                    │ HTTP 响应/SSE     │  8. 返回客户端
                                    └──────────────────┘
```

## 6.2 环节拆解

### ① FastAPI 入口

`entrypoints/http_server.py` 的 `launch_server()` 创建 FastAPI app、挂路由、启动 uvicorn。每个请求进入对应的 handler，例如 `/v1/chat/completions` 最终调用 `OpenAIServingChat.create_chat_completion()`。

### ② 协议适配

`serving_chat.py` 把 OpenAI 消息列表、tools、sampling 参数转换成 `GenerateReqInput`。注意此阶段还没有 tokenize——`GenerateReqInput` 里装的是文本。

### ③ TokenizerManager

`TokenizerManager.generate_request()`（`managers/tokenizer_manager.py:721`）：

```python
obj.normalize_batch_and_arguments()      # 单/批统一
...
tokenized_obj = await self._tokenize_one_request(obj)
self._send_one_request(tokenized_obj)    # ZMQ → Scheduler
async for response in self._wait_one_response(obj, request):
    yield response                        # 流式：边等边吐
```

这里完成了：分词（tokenize）、采样参数解析、LoRA 解析、多模态预处理、`rid_to_state` 状态注册。`rid_to_state` 以请求 ID 为键，后续所有异步响应都靠它归位。

### ④ ZMQ 投递

TokenizerManager 通过 `send_to_scheduler` 管道（ZeroMQ）把 `BatchTokenizedGenerateReqInput` 发给 Scheduler 进程（`_dispatch_to_scheduler`）。同时注册一个 `asyncio.Queue`，等待调度器返回。

### ⑤ Scheduler 调度循环

`managers/scheduler.py` 的 `event_loop_normal()`（第 1632 行）是无限循环：

```python
recv_reqs = self.request_receiver.recv_requests()   # 收请求
self.process_input_requests(recv_reqs)              # 入 waiting queue
plan = self.get_next_batch_to_run(...)              # 选下一批
batch = plan.batch_to_run
if batch:
    result = self.run_batch(batch)                  # GPU 执行
    self.process_batch_result(batch, result)        # 处理输出
```

关键点：

- 请求先进 **waiting queue**，被选中后才进入 **running batch**。
- 调度器按策略（`schedule_policy.py`）计算优先级，同时受显存预算（`mem_cache` 的 allocator）约束。
- prefill 和 decode 是不同形态的 batch，调度器分别处理（`get_new_batch_prefill` / decode 分支）。

### ⑥ ModelRunner 前向

`model_executor/model_runner.py` 的 `forward()` 接收 `ForwardBatch`，调用模型 `forward`，返回 `ModelRunnerOutput`（包含新 token、logprobs、缓存更新信息）。GPU 执行涉及 CUDA graph / torch.compile 等加速，第 9 章展开。

### ⑦ Detokenizer

`managers/detokenizer_manager.py` 负责把 token 流还原成文本。它支持增量式 detokenize，保证流式场景下不出现拼接乱码，同时计算"已完成 token 数"等元信息。

### ⑧ 响应返回

结果通过 ZMQ 送回 TokenizerManager 的响应队列，HTTP 层按 SSE 或完整 JSON 返回。

## 6.3 三类关键消息对象

生命周期中流转的对象值得单独记忆：

| 对象 | 位置 | 出现阶段 |
| --- | --- | --- |
| `GenerateReqInput` | `managers/io_struct.py` | HTTP → TokenizerManager |
| `BatchTokenizedGenerateReqInput` | `managers/io_struct.py` | TokenizerManager → Scheduler |
| `ScheduleBatch` | `managers/schedule_batch.py` | Scheduler 内部 |
| `ModelRunnerOutput` / `GenerationBatchResult` | `model_executor/forward_batch_info.py`、`managers/io_struct.py` | Scheduler → Detokenizer |
| `BatchStrOutput` | `managers/io_struct.py` | Detokenizer → TokenizerManager |

## 6.4 批请求如何走同一条路

`GenerateReqInput` 支持列表输入（`text: List[str]`）。`normalize_batch_and_arguments()` 会广播参数，`_handle_batch_request()` 并行 tokenize，然后**作为一批**投递给 scheduler。批请求在 GPU 上共享一次前向，吞吐更高——这就是"连续批处理"的输入侧。

## 6.5 观察它的工具

- 启动时加 `--log-level debug`，能看到每个请求的 tokenize、dispatch、step 日志。
- `--enable-cache-report` 让响应里带上命中缓存的 token 数（`cached_tokens`）。
- 打开 `/health` 可确认服务就绪；`/get_model_info` 可拿到上下文长度等配置。

## 6.6 本章小结

- 一条请求的骨架：协议 → tokenize → ZMQ → 调度 → GPU → detokenize → HTTP。
- 两次"翻译"最重要：协议层翻译成 `GenerateReqInput`，TokenizerManager 翻译成 token ids 投给调度器。
- 调度器内部是"收请求 → 组 batch → 执行 → 处理结果"的循环，第 7 章深挖。
- 多进程 + ZMQ 的架构保证了 HTTP 层与 GPU 执行层互不阻塞。
