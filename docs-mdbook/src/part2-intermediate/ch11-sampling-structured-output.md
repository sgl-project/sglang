# 第 11 章 采样与结构化输出

## 11.1 从 logits 到 token

模型最后一层输出 vocab 大小的 logits 向量，采样器（`layers/sampler.py`）负责把它变成"下一个 token"：

1. **后处理**：temperature 缩放、top-k/top-p/min-p 过滤；
2. **掩码**：应用 grammar/结构化约束（本章 11.4）；
3. **采样**：argmax（贪心）或从概率分布抽样（torch.multinomial 风格）；
4. **停停走走**：EOS 判定、max_tokens 截断、跳过特殊 token。

采样参数类型是 `sampling/sampling_params.py` 的 `SamplingParams`（msgspec Struct，支持 `array_like` 以便批处理），HTTP 层传入的字典在这里被解析和校验。需要返回 logprobs 时，`layers/logprob_processor.py` 与 `logits_processor.py` 负责收集各位置概率。

## 11.2 为什么"结构化输出"是个难题

直接让模型生成 JSON，模型可能：字段名写错、多了逗号、值类型不对、把内容截断。传统做法是"生成后解析 + 重试"，浪费 token 且不可靠。更优做法是**在采样阶段就只允许合法的 token**——这就是 constrained decoding（约束解码）。

SGLang 支持三种结构化约束入口：

| 方式 | 示例 |
| --- | --- |
| 正则 | `regex=r"\d{3}-\d{4}"` |
| JSON Schema | `json_schema={...}`（OpenAI `response_format`） |
| 无约束但校验 | 生成后由 parser 处理（tool calling 也走 parser） |

## 11.3 约束后端：一套接口，多套引擎

`constrained/` 目录把"如何根据当前状态计算合法 token 掩码"抽象成 `BaseGrammarBackend`（`base_grammar_backend.py`）：

- `xgrammar_backend.py`：xgrammar（默认主力，快、支持 JSON schema + regex + EBNF）；
- `outlines_backend.py`：outlines（老牌，支持 FSM）；
- `llguidance_backend.py`：微软 llguidance；
- `reasoner_grammar_backend.py`：推理模型专用（配合 `separate_reasoning`）。

核心对象是 `BaseGrammarObject`：调度器每步采样时调用 `fill_vocab_mask()` 生成掩码并应用到 logits，模型输出的 token 再喂回 `accept_token()` 推进自动机状态；如果某条路径失败还可 `rollback()`。

```python
class XGrammarGrammar(BaseGrammarObject):
    def fill_vocab_mask(self, vocab_mask, idx): ...
    def accept_token(self, token): ...
    def try_jump_forward(self, tokenizer): ...   # 见 11.5
```

## 11.4 GrammarManager：与调度器协作

`constrained/grammar_manager.py` 是调度器侧的协调者：

- 编译/缓存 grammar（`create_grammar_backend`），同一个 schema 只编译一次；
- 维护 `grammar_queue`，等待 grammar 就绪的请求先不入队（`get_ready_grammar_requests` 在 `_get_new_batch_prefill_raw` 开头被调用）；
- 支持 DP/TP 下的同步（`dp_tp_cpu_group`），保证多个副本对同一请求的约束状态一致。

## 11.5 Jump-Forward：让约束解码提速

当 grammar 有确定前缀时（如 JSON 里的 `"name": ` 这类固定文本），不需要逐 token 生成——`try_jump_forward()` 会一次性跳过这段确定文本。`outlines_jump_forward.py` 与 xgrammar 的实现都提供这个能力，README 里 "3x faster JSON decoding" 就来自压缩 FSM + jump forward 的组合。

## 11.6 工具调用（Function Calling）

工具调用本质是"结构化输出 + 协议解析"：

1. 请求带 `tools`，服务端（`entrypoints/openai/serving_chat.py`）把工具描述拼进 prompt；
2. 模型输出可能包含 `tool_call` 片段；
3. `function_call/function_call_parser.py` 与 OpenAI serving 层的 parser 把输出拆成 `tool_calls` 结构返回；
4. 客户端执行工具后，把结果作为新的 user 消息发回，模型继续。

仓库里 `examples/chat_template/tool_chat_template_*.jinja` 是各模型工具调用的 prompt 模板，`examples/usage/function_call/` 有完整示例。

## 11.7 采样/约束相关的性能注意点

- grammar 编译是 CPU 密集的，SGLang 做了缓存与队列化，避免阻塞调度循环；
- 每个受约束请求都要持有自己的自动机状态，批处理时状态数组按 batch 维度维护；
- 对推理模型（Reasoner），约束只应用于"回答"部分，思考部分保持自由——`reasoner_grammar_backend.py` 处理这条边界。

## 11.8 本章小结

- 采样器决定"下一个 token 是谁"，约束后端决定"哪些 token 合法"。
- 结构化输出 = 采样期约束（推荐）+ 生成后解析（兜底）。
- xgrammar 是当前默认，outlines/llguidance 是可插拔备选。
- Jump-forward 与 grammar 缓存是结构化输出场景的性能关键。
- 下一章看多模态：图像/视频/音频如何进入这条流水线。
