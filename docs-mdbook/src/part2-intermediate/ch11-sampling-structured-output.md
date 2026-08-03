# 第 11 章 采样与结构化输出代码走读：从概率到"必须合法的 token"

> 代码来自 `python/sglang/srt/sampling/`、`layers/sampler.py`、`constrained/`。

## 11.1 SamplingParams：采样参数的"落地形态"

HTTP 层收到的 `sampling_params` 字典，最终会变成 `SamplingParams`（`sampling/sampling_params.py:45`，msgspec Struct）：

```python
class SamplingParams(msgspec.Struct, kw_only=True, array_like=True):
    max_new_tokens: Optional[int] = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = TOP_K_ALL
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    ...
```

两个值得注意的设计：

1. `array_like=True`：一个 batch 里每个请求参数不同时，字段可以展开成数组，采样器按行取用——**采样参数也是"批量化"的**；
2. `json_schema` / `regex` / `ebnf` 直接挂在采样参数上：结构化约束是采样的一部分，不是后处理。

## 11.2 Sampler.forward：采样的一步步

`layers/sampler.py` 第 97 行，`Sampler.forward` 接受模型输出的 logits 和批量采样信息：

```python
def forward(self, logits_output, sampling_info, return_logprob, ...):
    logits = logits_output.next_token_logits
    logits = self._preprocess_logits(logits, sampling_info)   # 自定义 processor、NaN 处理

    if sampling_info.is_all_greedy:
        # 整批都是贪心 → 快路径
        batch_next_token_ids = torch.argmax(logits, -1)
        ...
    else:
        # 需要 temperature / top-p / top-k / min-p 的路径
        logits = logits / sampling_info.temperatures
        probs = torch.softmax(logits, dim=-1)
        batch_next_token_ids = self._sample_from_probs(probs, sampling_info)
```

`_sample_from_probs` 内部按 `need_top_p_sampling`、`need_top_k_sampling`、`need_min_p_sampling` 等标志逐项过滤，最后 `torch.multinomial` 采样。

注意"整批贪心快路径"的工程意义：RL/评测场景大量请求是贪心的，省掉整套概率计算能省不少 GPU 时间。

## 11.3 结构化输出：在采样前就把非法 token 屏蔽

第 4 章用过 `json_schema`，现在看服务端怎么保证"输出一定合法"。

`constrained/` 定义接口：`BaseGrammarBackend`（`base_grammar_backend.py`）编译语法，产出 `BaseGrammarObject`——每个受约束请求持有自己的"自动机状态"。核心方法：

```python
class BaseGrammarObject:
    def accept_token(self, token: int): ...      # 消费一个 token，推进状态
    def allocate_vocab_mask(self, vocab_size, batch_size): ...
    def fill_vocab_mask(self, vocab_mask, idx):  # 计算当前合法 token 掩码
    def apply_vocab_mask(self, logits, vocab_mask): ...  # logits 上屏蔽非法 token
```

xgrammar 后端实现（`constrained/xgrammar_backend.py`）：

```python
class XGrammarGrammar(BaseGrammarObject):
    def accept_token(self, token: int):
        ...
    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        ...
    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        # 语法确定的部分（如 JSON 里的固定字段名）可以一次跳过
        ...
```

配合 `layers/sampler.py` 的掩码应用，受约束请求的采样循环变成：

```text
每步：
  1. grammar.fill_vocab_mask(vocab_mask, idx)      # 算合法集
  2. grammar.apply_vocab_mask(logits, vocab_mask)  # 屏蔽非法 logits
  3. 采样出新 token
  4. grammar.accept_token(token)                   # 推进自动机
```

模型"想"输出非法 token？logits 已经是 -inf，采样器根本没机会选它。

## 11.4 GrammarManager：调度器侧的协调者

`constrained/grammar_manager.py` 是调度器与 grammar 之间的桥：

```python
# 调度器每轮 prefill 前：
if self.grammar_manager.has_waiting_grammars():
    ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
    for req in ready_grammar_requests:
        self._add_request_to_queue(req)
```

它的职责：

- **编译缓存**：同一个 schema 只编译一次，`create_grammar_backend` 复用；
- **队列化**：grammar 还没编译好的请求先不入队（避免阻塞调度循环）；
- **多副本同步**：DP/TP 下多个副本对同一请求的自动机状态要保持一致（用 `dp_tp_cpu_group` 同步）。

## 11.5 Jump-Forward：约束解码的提速器

JSON 生成中 `"name": ` 这类**语法确定的片段**不需要逐 token 生成。`try_jump_forward()` 返回"接下来必须是什么"，采样器直接跳过：

```text
普通：{"n"  "a"  "m"  "e"  """  ":"   ...
跳步：{"name":                                    ...（一次跳过 8 个 token）
```

README 里 "3x faster JSON decoding" 主要来自压缩 FSM + jump-forward 的组合。

## 11.6 工具调用：结构化输出的另一面

工具调用 = "生成一个工具调用结构 + 协议解析"。`function_call/function_call_parser.py` 负责把模型输出的工具调用文本拆成结构化结果；OpenAI serving 层再把结果包装成 `tool_calls` 字段。它和 grammar 的区别：

- grammar：**采样期**强制合法（可靠但需要自动机）；
- parser：**生成后**解析（灵活，但可能解析失败要重试）。

SGLang 两种都用：JSON 输出用 grammar，工具调用通常用 parser + 模板约束。

## 11.7 自己动手的实验

1. 对比：同一 prompt 分别用 `regex="\d{11}"` 和无约束各生成 20 次，数非法输出数量——验证"约束=0 非法"。
2. 生成一个复杂 JSON schema，打开 debug 日志，观察 grammar 编译耗时（体会为什么 SGLang 要缓存编译结果）。
3. 用 `/v1/tokenize` 验证 jump-forward 的效果：约束生成一个 `{"a": 1}`，数一数实际 decode 步数远小于 token 数。

## 11.8 本章小结

- SamplingParams 是批量化、带约束的采样参数容器。
- 采样器分"整批贪心快路径"和"通用概率路径"。
- 结构化输出 = 采样期 grammar 掩码 + 状态推进，保证"每一步都合法"。
- GrammarManager 负责编译缓存与调度器协作；jump-forward 跳过确定片段提速。

> 下一章看多模态：图像/视频/音频怎么变成 token 进入这条流水线。
