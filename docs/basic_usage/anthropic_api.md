# Anthropic 兼容 Messages API

SGLang 提供了一个与 Anthropic 兼容的 `POST /v1/messages` 入口，会将 Anthropic Messages 请求转换到 SGLang 内部的 OpenAI 兼容 chat completion 路径上执行。

## Claude Opus 4.6 → 4.7 协议变更

针对 Claude Opus 4.7，Anthropic Messages API 与 SGLang 相关的变更如下：

- 原先的 `thinking: {"type": "enabled", "budget_tokens": ...}` 被替换为自适应思考：`thinking: {"type": "adaptive"}`。
- `thinking.display` 控制是否在响应中返回思考内容：
  - `"summarized"`：当 SGLang 产生 reasoning 内容时，返回思考文本。
  - `"omitted"`：仍然启用思考，但不向客户端暴露 reasoning 文本。
- `output_config.effort` 新增了 Anthropic 风格的 effort 等级：`low`、`medium`、`high`、`xhigh`。
- `output_config.task_budget` 是给模型可见的预算提示，用于长 agent 循环。**它不是硬性输出上限**，硬性生成上限仍由 `max_tokens` 控制。
- Anthropic 4.7 官方 API 不再接受非默认的 `temperature`、`top_p`、`top_k`。SGLang 出于本地推理兼容性考虑，**仍然继续接受这些字段**。

## 支持的字段

除了已有字段（`model`、`messages`、`max_tokens`、`system`、`tools`、`tool_choice`、`temperature`、`top_p`、`top_k`、`stop_sequences`、`stream`）之外，SGLang 还接受：

```json
{
  "thinking": {
    "type": "adaptive",
    "display": "summarized"
  },
  "output_config": {
    "effort": "xhigh",
    "task_budget": {
      "type": "tokens",
      "total": 20000
    }
  },
  "betas": ["task-budgets-2026-03-13"]
}
```

### Thinking 字段映射

SGLang 将 Anthropic 的思考相关字段映射到内部 OpenAI 兼容请求字段如下：

| Anthropic 字段 | SGLang / OpenAI 兼容字段 |
|---|---|
| `thinking.type = "disabled"` | `reasoning_effort = "none"`，并禁用 chat-template 的 `thinking` / `enable_thinking` 标志 |
| `thinking.type = "enabled"` 或 `"adaptive"` | 启用 chat-template 的 `thinking` / `enable_thinking` 标志 |
| `thinking.budget_tokens` | `custom_params.thinking_budget` |
| `thinking.display = "omitted"` | 保留 reasoning 启用，但屏蔽流式输出中的 reasoning 文本 |
| `output_config.effort = "low"` | `reasoning_effort = "low"` |
| `output_config.effort = "medium"` | `reasoning_effort = "medium"` |
| `output_config.effort = "high"` | `reasoning_effort = "high"` |
| `output_config.effort = "xhigh"` | `reasoning_effort = "max"` |
| `output_config.effort = "max"` | `reasoning_effort = "max"` |
| `output_config.task_budget` | `custom_params.task_budget` |

`xhigh` 被映射到 SGLang 的 `max` reasoning 等级，因为 OpenAI 兼容请求并未定义 `xhigh` 这个值。

## 请求示例

```bash
curl http://localhost:30000/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "default",
    "max_tokens": 4096,
    "thinking": {
      "type": "adaptive",
      "display": "summarized"
    },
    "output_config": {
      "effort": "xhigh",
      "task_budget": {"type": "tokens", "total": 20000}
    },
    "messages": [
      {"role": "user", "content": "Solve this step by step: 123 * 456"}
    ]
  }'
```

当 SGLang 启动时配置了兼容的 reasoning parser，**非流式响应**可能在最终的 `text` 块之前包含一个 Anthropic `thinking` 内容块；**流式响应**会将 OpenAI 兼容协议下的 `reasoning_content` 增量映射为 Anthropic 的 `thinking_delta` 事件。

## 已知限制

- SGLang 不会生成 Anthropic 的思考加密签名（cryptographic thinking signature）；返回的 `signature` 为空字符串。
- `task_budget` 会被原样保留在 `custom_params` 中以便下游组件使用，但其本身**不会**作为运行时硬性限制器生效。
- Anthropic 4.7 官方的 tokenizer / 计数行为与具体模型相关；SGLang `/v1/messages/count_tokens` 使用的是当前服务模型自身的 tokenizer。

## 测试方案

Anthropic 兼容层由三组测试覆盖：协议层、非流式 serving 层、流式 serving 层。其中**非流式和流式两组都覆盖了"启用思考 + 工具调用"场景**，因此你在 Claude 4.7 上最关心的两个验证点 —— *流式 vs 非流式响应* 与 *工具调用 + 思考启用* —— 都由 CI 保障。

### 单元测试

运行所有 Anthropic 协议与转换相关测试：

```bash
PYTHONPATH=python python3 -m unittest discover \
  -s test/registered/unit/entrypoints/anthropic \
  -p 'test_*.py'
```

或单独运行某个文件：

```bash
PYTHONPATH=python python3 -m unittest \
  test.registered.unit.entrypoints.anthropic.test_anthropic_protocol \
  test.registered.unit.entrypoints.anthropic.test_anthropic_serving \
  test.registered.unit.entrypoints.anthropic.test_anthropic_streaming
```

#### `test_anthropic_protocol.py` —— 请求 schema

- `thinking`、`output_config`、`task_budget`、`betas` 请求字段解析。
- `thinking.type = "enabled"` 时的 budget 校验。
- `thinking_delta` 协议序列化。

#### `test_anthropic_serving.py` —— 非流式响应

- `thinking.type` 转换为 `chat_template_kwargs` 与 `reasoning_effort`。
- `output_config.effort = "xhigh"` 转换为 SGLang 的 `reasoning_effort = "max"`。
- `thinking.budget_tokens` 转换为 `custom_params.thinking_budget`。
- `output_config.task_budget` 保留到 `custom_params.task_budget`。
- OpenAI / SGLang 的 `reasoning_content` 反向转换为 Anthropic `thinking` 块。
- **启用思考 + 工具调用（非流式）**：携带 `thinking.type = "adaptive"`、`display = "summarized"` 与 `tools` 的请求会被转换为保留工具列表与 `tool_choice = "auto"` 的 chat completion 请求。模拟的 chat 响应同时包含 `reasoning_content` 与一个 `tool_calls` 条目；转换后的 Anthropic 响应中的内容块顺序为 `thinking` → `tool_use`，`stop_reason = "tool_use"`，且工具入参经 JSON 双向转换后保持不变。
- **`display = "omitted"` + 工具调用**：结构与上一项相同，但 `thinking` 块的文本为空，`tool_use` 块依旧完整。

#### `test_anthropic_streaming.py` —— 流式响应

流式测试用一个伪造的 OpenAI SSE 源驱动 `AnthropicServing._generate_anthropic_stream`，并断言其输出的 Anthropic 事件流。覆盖：

- **纯文本流式（无思考、无工具）**：事件顺序为 `message_start` → `content_block_start (text)` → 一个或多个 `content_block_delta (text_delta)` → `content_block_stop` → `message_delta`（`stop_reason = "end_turn"` 与 usage）→ `message_stop`。重组后的 `text_delta` payload 与模型输出一致。
- **启用思考的流式（`adaptive` + `summarized`）**：先开启一个 `thinking` 内容块并发送 `thinking_delta` 事件；当 `content` 增量到达时，关闭思考块并打开新的 `text` 块。
- **`display = "omitted"` 的流式**：仍然发送 `thinking` 块的 `content_block_start`（这样客户端可以渲染一个空占位），但**不发送** `thinking_delta` 事件；后续 `text_delta` 事件正常工作。
- **启用思考 + 工具调用（流式）**：期望事件顺序为 `message_start` → `content_block_start (thinking)` → `thinking_delta`* → `content_block_stop` → `content_block_start (tool_use)` → `input_json_delta`* → `content_block_stop` → `message_delta (stop_reason = "tool_use")` → `message_stop`。重组后的 `partial_json` 能解析回原始工具入参；内容块索引从 thinking 块（`index = 0`）递增到 tool_use 块（`index = 1`）。

### 既有 Anthropic 兼容性测试

运行已有的 server 级 Anthropic 测试：

```bash
PYTHONPATH=python python3 -m unittest \
  openai_server.basic.test_anthropic_server \
  openai_server.function_call.test_anthropic_tool_use
```

它们用于验证已有的 `/v1/messages`、流式、count tokens、视觉内容转换、工具调用等行为继续可用。


启动一个带 reasoning 解析器的、具备推理能力的模型，然后发送上文示例请求，验证以下点：

1. 请求能够被正常接受，且 `thinking.type = "adaptive"` 与 `output_config.effort = "xhigh"` 生效。
2. 当 `display = "summarized"` 时，**非流式**响应在最终 `text` 块之前包含一个 `thinking` 块。
3. **流式**响应输出 `type = "thinking"` 的 `content_block_start`，随后是 `thinking_delta` 事件，最后是 `text` 块。
4. 切换到 `display = "omitted"` 时请求仍然合法，且不会输出流式 `thinking_delta` 事件。
5. 切换到 `thinking.type = "disabled"` 时表现为不思考的行为。
6. 携带 `tools` 且 `thinking.type = "adaptive"` 时，**非流式**与**流式**响应都会包含一个 `thinking` 块和一个 `tool_use` 块，且 `stop_reason` 为 `tool_use`。


- 使用 `temperature`、`top_p`、`top_k` 的请求在 SGLang 中应继续可用，以保证向后兼容。
- 使用 Anthropic `tool_choice` 值的请求应保持现有映射：`auto`、`none`、`any` → OpenAI `required`，以及具名 `tool`。
- `/v1/messages/count_tokens` 应能接受新增字段，并继续正常返回 `input_tokens`。
