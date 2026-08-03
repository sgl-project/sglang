# 第 5 章 HTTP 服务层：OpenAI 兼容 API 与协议适配

## 5.1 入口文件

服务端入口是 `python/sglang/srt/entrypoints/http_server.py`，文件头注释直接说明了身份：

> "The entry point of inference server. (SRT = SGLang Runtime). This file implements HTTP APIs for the inference engine via fastapi."

它用 FastAPI + uvicorn 承载所有 HTTP 端点，并挂载了三个协议族：

- **OpenAI 兼容**：`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`、`/v1/rerank`、`/v1/score`、`/v1/classify`、`/v1/tokenize`、`/v1/detokenize`、`/v1/models` 等。
- **Anthropic 兼容**：`/v1/messages`（`entrypoints/anthropic/`）。
- **Ollama 兼容**：`/chat`、`/generate`（`entrypoints/ollama/`）。
- **原生接口**：`/generate`、`/generate_stream`、`/embedding`、`/flush_cache`、`/health`、`/get_model_info` 等。

这些端点统一挂在 FastAPI 上，启动时代码大致是：

```python
app = FastAPI(...)
app.add_middleware(CORSMiddleware, ...)

app.post("/v1/chat/completions", response_model=None)
app.post("/v1/completions", ...)
app.post("/generate", ...)
...
```

## 5.2 协议适配层：从 OpenAI 消息到内部请求

`entrypoints/openai/` 目录下的 `serving_chat.py`、`serving_completions.py`、`serving_embedding.py` 等，职责是把协议对象转换成内部的 `GenerateReqInput`（定义在 `managers/io_struct.py`），再调用 `TokenizerManager.generate_request()`。

链路示意：

```text
POST /v1/chat/completions
  → ChatCompletionRequest (pydantic/msgspec 校验)
  → OpenAIServingChat.create_chat_completion()
      ├─ 处理 tools / tool_call / 结构化输出
      ├─ 组装 messages → prompt
      └─ 构造 GenerateReqInput
          → TokenizerManager.generate_request()  (异步生成器，逐段 yield 响应)
  → 流式场景用 SSE (text/event-stream) 逐块返回
```

`serving_chat.py` 里的 `MessageProcessingResult`、`ResponseParserProtocol` 等类型，体现了它对工具调用（function calling）的完整支持：模型输出会经过 parser（`entrypoints/openai/` 内的 `parse_function_call` 相关实现）拆成 `tool_calls` 字段返回。

## 5.3 内部核心数据结构：GenerateReqInput

`managers/io_struct.py` 的 `GenerateReqInput` 是 HTTP 层与调度层之间的"通用语言"，字段非常丰富，值得通读一遍。几个关键分组：

| 分组 | 字段示例 |
| --- | --- |
| 输入 | `text`、`input_ids`、`input_embeds` |
| 多模态 | `image_data`、`video_data`、`audio_data`、`mm_hashes` |
| 采样 | `sampling_params`（字典，交给 `SamplingParams`） |
| 输出控制 | `stream`、`return_logprob`、`logprob_start_len`、`top_logprobs_num`、`return_hidden_states` |
| 会话 | `session_id`、`session_params` |
| LoRA | `lora_path`、`lora_id` |
| 结构化 | `custom_logit_processor`、`positional_embed_overrides` |
| PD 分离 | `bootstrap_host/port/room/pair_key` |

`GenerateReqInput` 内部有 `normalize_batch_and_arguments()`，把单请求/批请求统一成内部表示——批请求的元素会广播成等长列表，这是后续 batch 处理的前提。

## 5.4 TokenizerManager：HTTP 与 Scheduler 之间的枢纽

`managers/tokenizer_manager.py` 的 `TokenizerManager` 是每个请求真正的第一站：

```python
async def generate_request(self, obj, request=None):
    obj.normalize_batch_and_arguments()
    ...
    tokenized_obj = await self._tokenize_one_request(obj)
    self._send_one_request(tokenized_obj)   # 通过 ZMQ 发给 Scheduler
    async for response in self._wait_one_response(obj, request):
        yield response
```

它维护 `rid_to_state`（每个请求的状态机），处理 tokenize、采样参数校验、多模态预处理、会话管理，以及把 tokenizer 输出（`BatchTokenizedGenerateReqInput`）通过 ZMQ `send_to_scheduler` 管道投递给调度器。

## 5.5 流式返回：SSE 与异步生成器

SGLang 的流式是"服务端真正逐 token 生成、HTTP 层逐块转发"：

1. 调度器每步 decode 产生新 token；
2. `TokenizerManager._wait_one_response` 从响应队列读取增量；
3. detokenizer 把增量 token 还原成文本（避免逐 token 拼接误差）；
4. `http_server.py` 用 `StreamingResponse` / SSE 格式（`entrypoints/openai/sse_utils.py`）推给客户端。

因此即使只发一个请求，日志里也能看到 prefill 一步 + 多个 decode 步的循环。

## 5.6 其他重要端点

- `/flush_cache`：清空 radix cache / memory pool（`mem_cache/flush_cache.py`），模型权重更新或测试隔离时常用。
- `/get_model_info`：返回模型名、上下文长度等（`get_init_info` 来自 scheduler）。
- `/health`：健康检查，供负载均衡/编排系统探活（`managers/schedule_batch.py` 或 `entrypoints` 中实现）。
- `/update_weights_from_disk` 等：在线权重更新（RL 训练场景用）。

## 5.7 本章小结

- HTTP 层 = FastAPI 入口 + 多协议适配 + TokenizerManager。
- 协议对象最终都会变成统一的 `GenerateReqInput`，这是"一套内部表示，多种外部协议"的关键。
- 流式输出是端到端逐 token 的，HTTP 层只是转发者。
- 下一章把 HTTP 层、TokenizerManager、Scheduler、ModelRunner、Detokenizer 串成一条完整的请求生命周期。
