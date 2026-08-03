# 第 5 章 对外 API 入门：OpenAI 兼容接口怎么用

> 本章继续"只讲用法"。目标：不看文档也能用 curl/requests 完成聊天、流式、JSON 输出、工具调用和多模态请求。

## 5.1 服务暴露了什么

启动 `sglang serve` 后，服务端对外提供两类接口：

1. **OpenAI 兼容接口**（`/v1/...`）：和 OpenAI 官方 API 的请求格式一致，SDK 可以直接换 base_url 接入；
2. **原生接口**（`/generate`、`/health`、`/flush_cache` 等）：SGLang 内部使用，调试和运维常用。

常用端点一览：

| 端点 | 用途 |
| --- | --- |
| `POST /v1/chat/completions` | 聊天补全（最常用） |
| `POST /v1/completions` | 纯文本补全 |
| `POST /v1/embeddings` | 向量化 |
| `POST /v1/rerank`、`/v1/score` | 重排、打分（reward 模型） |
| `POST /v1/tokenize`、`/v1/detokenize` | 文本 ↔ token |
| `GET /v1/models` | 查看可用模型 |
| `GET /health` | 健康检查 |
| `POST /flush_cache` | 清空 KV 缓存 |
| `GET /get_model_info` | 模型信息（上下文长度等） |

## 5.2 聊天与流式

普通聊天：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama",
       "messages": [{"role": "user", "content": "你好"}]}'
```

流式：加 `"stream": true`，并用 `curl -N`（关闭缓冲）看逐块返回：

```bash
curl -N http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama",
       "messages": [{"role": "user", "content": "讲个故事"}],
       "stream": true}'
```

返回的是 SSE 格式：每行一个 `data: {...}`，最后是 `data: [DONE]`。用 Python 读流式：

```python
import requests, json

resp = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={"model": "llama",
          "messages": [{"role": "user", "content": "讲个故事"}],
          "stream": True},
    stream=True,
)
for line in resp.iter_lines():
    if line and line.startswith(b"data: ") and line != b"data: [DONE]":
        delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)
```

## 5.3 强制 JSON 输出

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama",
       "messages": [{"role": "user", "content": "输出用户信息"}],
       "response_format": {"type": "json_schema",
                           "json_schema": {
                             "type": "object",
                             "properties": {
                               "name": {"type": "string"},
                               "age": {"type": "integer"}},
                             "required": ["name", "age"]}}}'
```

返回的 `content` 一定是符合 schema 的 JSON。做数据抽取、Agent 输出规范化时非常有用。

## 5.4 工具调用（Function Calling）

告诉模型"你有这些工具可用"，模型会输出"我要调用哪个工具、传什么参数"：

```python
requests.post("http://localhost:30000/v1/chat/completions", json={
    "model": "llama",
    "messages": [{"role": "user", "content": "北京现在几度？"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
})
```

返回里会带 `tool_calls` 字段：工具名 + 参数 JSON。你的代码执行完工具，把结果作为新的 user 消息发回去，对话继续。

## 5.5 多模态请求

图片进、文本出（需要启动的是 VLM 模型）：

```python
requests.post("http://localhost:30000/v1/chat/completions", json={
    "model": "qwen2-vl",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            {"type": "text", "text": "这是什么？"},
        ],
    }],
})
```

## 5.6 运维三件套

```bash
# 服务活着吗（负载均衡、K8s 探活用这个）
curl http://localhost:30000/health

# 模型信息：上下文长度、模型名
curl http://localhost:30000/get_model_info

# 清空 KV 缓存（换数据集测试、更新权重后常用）
curl -X POST http://localhost:30000/flush_cache
```

## 5.7 本章自测

1. `stream: true` 时返回格式和普通请求有什么区别？为什么流式对用户体验重要？
2. `response_format` 的 JSON Schema 和"生成后解析"相比，可靠性好在哪？
3. 工具调用的完整流程有几步？`tool_calls` 字段在什么时候出现？
4. 用 `requests` 给 `/v1/tokenize` 发 `{"text": "你好世界"}`，看返回的 token id 是什么。

> 现在你会启动、会调用、会流式、会结构化输出。下一章，把第 3 章的餐厅图和 HTTP 请求真正对上号——一次请求的完整旅程。
