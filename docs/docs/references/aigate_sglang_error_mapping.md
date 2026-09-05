# AIgate - SGLang 错误处理实现方案

实施分支：AIgate `fix/prefill-http-status-propagation`，基线提交 `a72ca908`；AIgate 实现提交为 `6bc2cbcfdf22f57e55b56af3e60922738011cdbb`。SGLang 依据 `2377fc40398b3d33334de75b90fbe5e86e714259` 核对，本次修改 AIgate 的兼容处理，不修改 SGLang 运行时代码。

`ret_code` 保持只写日志和指标，不放入 API 响应、响应头或 SSE。对外返回 AIgate 封装的错误信息，内部按用户责任记录 `104xx`、服务责任记录 `10503`。

## 1. 目标与基线问题

开发前的基线已保留 Prefill 的 HTTP status，但存在以下问题；本次实现统一处理这些路径：

- `upstreamHTTPRet` 只按 HTTP status 分类，上游 4xx 基本归为用户错误。
- Prefill 会重新封装错误；Worker/Decode 的 OpenAI 非流式错误仍会原样返回。
- 用户错误的日志/指标数字码原为 `101xx`，本次迁移为 `104xx`。
- SSE 错误没有统一分类；Anthropic 转换器可能忽略 `error`，再输出正常结束事件。Prefill 返回 200 时，也没有检查响应体是否包含错误。

**核心问题：SGLang 的 HTTP 400 不一定是用户问题。** [serving_base.py:114] 将 `ValueError` 统一转为 400，模板缺失、grammar backend 未配置、内部字段错误都可能走这条路径。因此需要按“错误原因 + 字段来源”判责。

## 2. 统一分类与 SGLang 错误字典

### 统一解析、分类和返回

Worker、Prefill、Decode，以及非流式和 SSE，都调用同一个错误分类函数：

```text
SGLang 响应
  → 提取 HTTP status、error code/type/message、角色和阶段
  → 命中明确用户错误，且由用户原始请求造成：104xx
  → 服务配置、容量、P/D、运行时、连接或超时故障：10503
  → 无法识别或证据冲突：10503
  → 对外生成 AIgate 错误信息；内部将 ret_code 写入日志和指标
```

解析器兼容顶层 `message/type/code`、嵌套 `error` 和 FastAPI `detail`；保留现有错误体大小、读取超时限制。消息匹配使用限定版本的完整消息或锚定规则，不用 `Contains("invalid")` 这类宽泛判断。

### SGLang 错误清单与 AIgate 判责

以下按本地 SGLang `2377fc40398b3d33334de75b90fbe5e86e714259` 核对，官方文档用于确认参数语义和部署限制（查阅日期：2026-09-05；采样、Server Arguments 和 P/D 文档复核：2026-09-06）。**消息和返回方式以该提交源码为准**；表中的 `{n}`、`{limit}`、`{e}` 表示动态值，`…` 表示省略片段，不应直接作为宽泛匹配规则。

覆盖 Python 推理主链路、Scheduler、P/D，以及相关扩展接口的显式拒绝分支。第三方 tokenizer、模型模板、grammar 编译器和 GPU 运行时会产生动态异常，不能穷举为稳定错误码；这类异常保留统一兜底。

#### 返回码和错误体：先区分传输状态与错误语义

SGLang 在这些路径中没有统一的业务错误码表：`code` 常是 HTTP 数字，`type` 又随入口和捕获方式变化。同一个 400 可以出现 `BadRequestError`、`BadRequest`、`Bad Request`、`"400"` 或 SSE 中的 `BAD_REQUEST`。[base:110]、[base:196]、[http:612]、[chat:1633]

| 入口 / 路径 | 实际返回方式 | AIgate 应如何读取 |
| --- | --- | --- |
| Python OpenAI 请求校验、非流式错误 | 常见顶层 `{"object":"error","message":"…","type":"…","param":null,"code":400}` | 兼容顶层错误；不能只查 `error.message`。[base:196] |
| Python OpenAI `ValueError` / 其他异常 | 通常分别为 400 / 500；其他异常消息为 `Internal server error: {e}` | 异常类型不是责任依据。500 默认服务错误，400 继续判责。[base:114] |
| Python 主应用请求体、字段类型、JSON 校验 | `RequestValidationError` 被改为 **400**；不是 FastAPI 默认的 422 | 利用字段位置与原始请求确认责任；错误文本可能回显请求，不能整体对外返回。[http:612] |
| Scheduler / P/D `FINISH_ABORT` | 非流式：400 转 `ValueError`，500/503 转 `HTTPException`；流式：yield abort，OpenAI 编成 `data: {"error":{…,"code":400/500/503}}` | **HTTP 200 仍可能是失败**；解析完整 SSE 帧，在转码、写出和统计前处理。[tokenizer:1646]、[chat:1633] |
| Native `/generate` | 异常可为 `{"error":{"message":"…"}}`；流式还可能在 `meta_info.finish_reason` 中携带 `type=abort/status_code/message` | 若接入 Native，补充该结构；不能要求总有 `error.code`。[http:920]、[http:2091]、[tokenizer:1646] |
| `/v1/responses` | `{"error":{"message":"…","type":"invalid_request_error","param":…,"code":400}}`，默认错误 status=400 | 它有独立捕获逻辑，甚至部分 `RuntimeError` 也被包装成 400。[responses:198]、[responses:315] |
| 原生 `/v1/messages` | `{"type":"error","error":{"type":"invalid_request_error/api_error/…","message":"…"}}`，不保证有数字 `code` | 同时读取 HTTP status 和事件类型；部分内部转换异常会变成 500 `api_error`，不能套用 OpenAI 的 400 假设。[http:544]、[anthropic:747]、[anthropic:807] |
| 鉴权中间件 | 401 `{"error":"Unauthorized"}`；403 `{"error":"Forbidden"}` | `error` 也可能是字符串；AIgate → SGLang 的内部鉴权失败默认服务责任。[auth:180] |
| 请求解压中间件 | 400 纯文本，例如 `decompress failed` | 允许受限纯文本诊断；按压缩操作的执行方判责。[decompress:46] |
| 断连、超时、进程退出 | 可能没有 HTTP 响应，或已发 200 后流被截断 | 属于传输/流完整性故障，不能凭已有 200、usage 或空 EOF 记成功。 |

下面的 `400/500/503` 表示该错误路径的状态语义；**`abort 400/500/503` 特指可能藏在 HTTP 200 流中的失败**。提前抛出的 400 也可能因已经开始输出而成为 SSE error，AIgate 必须同时支持两种承载方式。

#### 已实现的 AIgate 归类

保留原表的分类粒度，不为每条 SGLang 消息新增观测码。**下列数字已用于实现；`ret_code` 只写日志和指标，API 的 `error.code` 使用右列字符串。** 明确的本地模型未提供、客户端取消另外使用 `10407`、`10408`，见第 3 节。

| 确认的责任 / 原因 | 日志和指标 ret_code | 对外 error.code |
| --- | --- | --- |
| 已确认的用户请求结构、字段类型或参数组合错误 | `10401` | `invalid_request` |
| 用户工具定义、工具选择或历史工具参数不合法 | `10402` | `invalid_tool_definition` |
| 用户采样参数不合法 | `10403` | `invalid_sampling_parameters` |
| 用户输入或请求输出违反公开长度限制 | `10404` | `context_length_exceeded` |
| 用户结构化输出约束不合法 | `10406` | `invalid_output_constraint` |
| 服务配置、容量、协议转换、调度、P/D、运行时、传输故障或无法确定 | `10503` | `engine_error` |

**下表是源码错误字典与目标判责依据，并非所有行都已加入运行时用户白名单。当前启用规则及保守兜底边界见第 3 节。**

**以下所有用户归类都以“字段来自用户原始请求，且违反 AIgate 公开协议/限制”为前提。** 如果是 AIgate 注入、改写或路由后才不合法，改记 `10503 / engine_error`。表中“条件判责”未满足条件时也按此兜底。

#### A. 请求结构、工具与参数组合

| SGLang 状态 | 错误消息 / 源码模板 | 触发原因及 AIgate 判责 | ret_code / 对外 error.code | 源码 |
| --- | --- | --- | --- | --- |
| 400 | `Messages cannot be empty.` | Chat 的 `messages` 为空 | `10401 / invalid_request` | [chat:865] |
| 400 | `Prompt cannot be empty` | Completions 的 `prompt` 为空 | `10401 / invalid_request` | [completions:65] |
| 400 | Pydantic 的缺字段、类型、枚举、JSON 解析错误；文本含字段位置 | 仅当字段位置和原始请求证明用户传参错误时成立；不能只匹配 `validation error` | `10401 / invalid_request`；否则 `10503 / engine_error` | [http:612]、[protocol:699] |
| 400 | `Unsupported Media Type: Only 'application/json' is allowed` | Python 主应用这里返回 400。该分支 AIgate 自己设置上游 Content-Type，因此内部命中通常是构造请求故障 | `10503 / engine_error` | [http:658] |
| 400 | `Tools cannot be empty if tool choice is set to required.` | required 但无有效工具 | `10402 / invalid_tool_definition` | [chat:884] |
| 400 | `Tools cannot be empty if tool choice is set to a specific tool.` | 指定工具但工具列表为空 | `10402 / invalid_tool_definition` | [chat:891] |
| 400 | `Tool '{name}' not found in tools list.` | 选择的工具不在有效工具列表 | `10402 / invalid_tool_definition` | [chat:894] |
| 400 | `Tool names must be unique across request and message tools.` | 同时使用请求级和消息级 tools 时，合并后名字重复；不是所有 tools 请求都走这个检查 | `10402 / invalid_tool_definition` | [chat:901] |
| 400 | `Tool {i} function has invalid 'parameters' schema: {e}` | 工具参数不符合 JSON Schema；源码先规范化部分类型别名再校验 | `10402 / invalid_tool_definition` | [chat:906] |
| 400 | `Tool {i} function 'parameters' schema is too deeply nested or contains a cycle.` | schema 过深或循环 | `10402 / invalid_tool_definition` | [chat:919] |
| 400 | `Assistant tool call function.arguments must be valid JSON.` / `…must be a JSON object.` | 用户提供的历史 assistant tool arguments 非法；若历史由服务拼接则服务责任 | `10402 / invalid_tool_definition`；否则 `10503 / engine_error` | [chat:127] |
| 400 | `Function tools must include a name.` | Responses function tool 缺 name | `10402 / invalid_tool_definition` | [protocol:1566] |
| 400 | `tool_choice 'required' or a named tool cannot be combined with response_format, regex, or ebnf: the tool-call constraint and the output constraint cannot both be honored.` | required/named tool 与输出约束冲突 | `10406 / invalid_output_constraint` | [protocol:1154] |
| 400 | `return_sampling_mask requires return_meta_info=true.` | 返回采样 mask 的依赖字段缺失 | `10401 / invalid_request` | [chat:870] |
| 400 | `return_prompt_token_ids is not supported with streaming.` / `return_token_ids is not supported with streaming on /v1/chat/completions.` / `return_meta_info is not supported with streaming.` | 请求不支持的流式返回字段组合 | `10401 / invalid_request` | [chat:984] |
| 400 | `reasoning_effort must not be a boolean` / `invalid reasoning effort: {value!r}` | reasoning 参数类型或值无法解析 | `10401 / invalid_request` | [protocol:976] |
| 400 | `Inkling reasoning_effort must not be a boolean` / `…must be in [0.0, 0.99]` / `invalid Inkling reasoning_effort: {value!r}` | Inkling 的模型特定参数限制 | `10401 / invalid_request` | [chat:643] |
| 400 | `Harmony does not support reasoning effort none` | gpt-oss/Harmony 的该转换分支不接受 none | `10401 / invalid_request` | [chat:976] |
| 400 | `thinking parts require exactly one of 'thinking' or 'text'` / `thinking content parts are only valid in assistant messages` | thinking 历史消息字段互斥、角色限制 | `10401 / invalid_request` | [protocol:541]、[protocol:716] |

#### B. 采样参数与输出约束

参数范围来自本地 `SamplingParams.verify`，与官方 [Sampling Parameters](https://docs.sglang.io/docs/basic_usage/sampling_params) 对照。OpenAI 请求模型还会提前做字段校验，因此不保证一定返回下面这条底层消息。原生 `max_new_tokens=0` 可被底层接受；不要据 OpenAI 的正数约束误拒所有 Native 请求。

| SGLang 状态 | 错误消息 / 源码模板 | 触发原因及 AIgate 判责 | ret_code / 对外 error.code | 源码 |
| --- | --- | --- | --- | --- |
| 400 | `temperature must be a non-negative finite number, got {v}.` | 负数、NaN、Inf 等；0 合法 | `10403 / invalid_sampling_parameters` | [sampling:151] |
| 400 | `top_p must be in (0, 1], got {v}.` | top_p ≤ 0 或 > 1 | `10403 / invalid_sampling_parameters` | [sampling:156] |
| 400 | `min_p must be in [0, 1], got {v}.` | min_p 越界 | `10403 / invalid_sampling_parameters` | [sampling:158] |
| 400 | `top_k must be -1 (disable) or at least 1, got {v}.` | top_k 非法；-1 在构造时已归一化，不能照抄 verify 内部表达式作网关规则 | `10403 / invalid_sampling_parameters` | [sampling:148]、[sampling:160] |
| 400 | `frequency_penalty must be in [-2, 2], got {v}.` | frequency_penalty 越界 | `10403 / invalid_sampling_parameters` | [sampling:164] |
| 400 | `presence_penalty must be in [-2, 2], got {v}.` | presence_penalty 越界 | `10403 / invalid_sampling_parameters` | [sampling:169] |
| 400 | `repetition_penalty must be in (0, 2] (1.0 = no penalty), got {v}.` | repetition_penalty ≤ 0 或 > 2 | `10403 / invalid_sampling_parameters` | [sampling:173] |
| 400 | `min_new_tokens must be in [0, max_new_tokens], got {v}.` / `…[0, max_new_tokens({max})]…` | 最小输出为负或大于最大输出 | `10403 / invalid_sampling_parameters` | [sampling:178] |
| 400 | `max_new_tokens must be at least 0, got {v}.` / `max_tokens must be positive` | Native 最大输出为负；Completions max_tokens 非正数。字段来源与入口需分别匹配 | `10403 / invalid_sampling_parameters` | [sampling:183]、[protocol:410] |
| 400 | `logit_bias must has keys in [0, {max_id}], got {id}.` | token ID 超出词表；原文包含 `must has`。非数字 key 还可能产生通用 int 转换异常，不能宽泛匹配 | `10403 / invalid_sampling_parameters` | [sampling:193] |
| 400 | `Only one of json_schema, regex, ebnf, or structural_tag can be set.` | 同时指定多个输出约束 | `10406 / invalid_output_constraint` | [sampling:201] |
| 400 | `schema_ is required for json_schema response format request.` | JSON Schema 响应格式缺 schema | `10406 / invalid_output_constraint` | [chat:937]、[completions:178] |
| abort 400 | `Failed to compile {json/regex/ebnf/structural_tag} grammar: {detail}` | **条件判责**：明确是用户 grammar 语法错误才记 10406；编译库异常、不支持的部署 backend、缓存的超时结果均为服务问题 | `10406 / invalid_output_constraint`；否则 `10503 / engine_error` | [grammar:162]、[grammar:277] |
| abort 400 | `invalid stop_regex {pattern!r}: {e}` | Scheduler 对非法 stop_regex 的兜底拒绝；更早 normalize 阶段也可能先报动态异常 | 明确用户正则错误时 `10403 / invalid_sampling_parameters`；否则 `10503 / engine_error` | [batch:1580]、[sampling:212] |

官方 [Structured Outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs) 和 [Tool Parser](https://docs.sglang.io/docs/advanced_features/tool_parser) 可用于确认约束用法与 backend 支持。`Failed to compile` 只是统一包装前缀，不能单独加入用户错误白名单。

#### C. 长度、token 与多模态输入

| SGLang 状态 | 错误消息 / 源码模板 | 触发原因及 AIgate 判责 | ret_code / 对外 error.code | 源码 |
| --- | --- | --- | --- | --- |
| 400 | `max_completion_tokens is too large: {n}.This model supports at most {limit} completion tokens.` | 请求最大输出大于服务配置 context，且未开启自动截断；注意原文句号后无空格 | 违反公开输出限制才 `10404 / context_length_exceeded`；实例限制不一致则 `10503 / engine_error` | [chat:925] |
| 400 | `The input ({n} tokens) is longer than the model's context length ({limit} tokens).` | TokenizerManager 按输入 token + reserved tokens 检查，条件为 `>= context_len`；不开自动截断才抛异常 | 超过公开输入限制才 `10404 / context_length_exceeded`；否则 `10503 / engine_error` | [tokenizer:1160] |
| 400 | `Requested token count exceeds the model's maximum context length of {limit} tokens. You requested a total of {total} tokens: …` | `validate_total_tokens` 开启时，输入 + 最大输出 > context，且不自动截断 | 违反公开总长度限制才 `10404 / context_length_exceeded`；否则 `10503 / engine_error` | [tokenizer:1184] |
| abort 400 | `Input length ({n} tokens) exceeds the maximum allowed length ({limit} tokens). Use a shorter input or enable --allow-auto-truncate.` | Scheduler 按 `max_req_input_len` 拒绝；该值受 context、有效 KV 容量和预留空间限制，**不等于公开 context** | 超过公开输入限制才 `10404 / context_length_exceeded`；仅实例容量不足或无法确认则 `10503 / engine_error` | [length:206]、[scheduler:2654]、[tp_worker.py:532] |
| abort 400 | `Multimodal prompt is too long after expanding multimodal tokens. After expanding …` | 图片/视频等展开后的 token 数达到 Scheduler 上限 | 按公开的多模态长度口径确认超限才 `10404 / context_length_exceeded`；否则 `10503 / engine_error` | [scheduler:2640] |
| abort 400 | `Request {rid} exceeds the maximum number of tokens: {n} > {limit}` | Prefill/Decode 单请求超过实例 token/KV 容量，不是模型公开 context 校验 | 当前固定 `10503 / engine_error`；Decode 的重新预填充长度可能包含已生成 token，不将该数字当作用户输入长度 | [prefill.py:368]、[decode.py:747] |
| abort 400 | `Request {rid} requires too many SWA KV tokens for decode preallocation: {required} > {capacity}` | Decode 滑动窗口 KV 预分配容量不足 | `10503 / engine_error` | [decode:761] |
| 400 | `The input_ids {ids} contains values greater than the vocab size ({size}).` | 原生输入 token 越界；源码条件为 `>= vocab_size`，消息写的是 greater than | 用户直接提交 token 时 `10401 / invalid_request`；内部 tokenizer 生成则 `10503 / engine_error` | [tokenizer:1315] |
| 400 | `token_ids_logprob must be a flat list of integers.` / `token_ids_logprob contains out-of-vocabulary token id {id}; valid range is [0, {size}).` | 指定 logprob token 的结构或范围错误 | `10401 / invalid_request` | [tokenizer:1295] |
| 400 | `Model only supports text input; received unsupported content type '{type}'.` | 对文本模型发送媒体内容 | 公开模型确为纯文本才 `10401 / invalid_request`；把多模态模型错误路由到纯文本实例则 `10503 / engine_error` | [chat:944] |
| 400 | `{Modality} count {n} exceeds limit {limit} per request.` | 超过实例 `limit_mm_data_per_request` | 公开数量限制被违反才 `10401 / invalid_request`；否则 `10503 / engine_error` | [tokenizer:1252] |
| 400，可能多层包装 | `Invalid media URL: {url!r}` | 下载器要求有效 HTTP(S) URL | 用户 URL 语法错误时 `10401 / invalid_request` | [media:1579] |
| 400，可能多层包装 | `Media URL domain is not allowed. Allowed domains: …; input domain: …` | 服务端媒体域名白名单拒绝，也检查重定向目标 | 用户违反公开域名规则才 `10401 / invalid_request`；仅内部配置拒绝则 `10503 / engine_error` | [media:1584] |
| 400，可能多层包装 | `Remote media exceeds the {max_bytes} byte download limit` | Content-Length 或累计下载字节超过服务配置上限 | 超过公开媒体大小限制才 `10401 / invalid_request`；否则 `10503 / engine_error` | [media:1640] |
| 400，可能多层包装 | `Invalid image: …` / `Invalid audio format: …` / `Unsupported video input type: …` | 媒体输入类型不支持；不同 loader 也可能抛其他异常 | 确认原始媒体参数非法才 `10401 / invalid_request`；否则 `10503 / engine_error` | [media:1713]、[media:1900]、[media:1961] |
| 400，可能多层包装 | `Could not decode audio: {e}` / `Could not decode video: {e}`；非法 base64、无法识别图片的库错误 | 媒体内容损坏，或解码器/依赖故障；video 部分 RuntimeError 也会被转成 ValueError | 明确用户数据损坏才 `10401 / invalid_request`；解码部署故障或不明原因 `10503 / engine_error` | [media:1764]、[media:1976] |
| 400 或 500，取决于包装 | `Timed out while downloading media URL: {url}` / `Media URL exceeded {n} redirects: {url}` / HTTP 下载异常 | 媒体下载失败；`requests.RequestException` 会被媒体 loader 包为 ValueError，**网络超时也可能最终变 400** | 默认 `10503 / engine_error`；仅证实用户 URL 违反公开规则时按 `10401 / invalid_request` | [media:1618]、[media:1685]、[mm:817] |
| 400 或 500 | `Error while loading data {data}: {e}` / `An exception occurred while loading {modality} data at index {i}: {e}` | 媒体加载统一包装；同一个前缀同时覆盖用户数据与服务异常 | 必须解析已验证的内层原因；未知 `10503 / engine_error` | [mm:817]、[mm:1179] |
| 400 | `mm_content_hashes has {n} entries for {m} images` / `Conflicting content hashes for image_data[{i}]` | 媒体 hash 数量或两处 hash 冲突 | 公开允许用户提交这些字段且用户填错才 `10401 / invalid_request`；内部生成则 `10503 / engine_error` | [tokenizer:1144] |

长度判责必须有 AIgate 自己维护的模型公开限制和字段来源；不能从错误消息里的实例上限反推对外合同。官方 [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) 区分模型 context 与 `max-total-tokens` 等容量配置，并说明部分队列参数在 P/D 下不生效。媒体错误中的 URL、输入数据、内部允许域名和模型路径仅留受控诊断，不直接作为公开 message。

#### D. 服务配置、模型处理和内部字段：即使 400 也不能直接归用户

| SGLang 状态 | 错误消息 / 源码模板 | 触发原因及 AIgate 判责 | ret_code / 对外 error.code | 源码 |
| --- | --- | --- | --- | --- |
| 400 | `This model has no HF chat template and no custom chat encoder; cannot encode chat messages with …` | 缺少服务端聊天模板/编码器 | `10503 / engine_error` | [chat_encoding:204] |
| 400，动态文本 | Jinja 模板异常、DS32 编码异常 | 模板可因非法历史消息报错，也可因部署模板不兼容报错；不能按异常类型或模板错误前缀归用户 | 默认 `10503 / engine_error`；只有验证到用户违反公开消息规则才 `10401 / invalid_request` | [chat:1423]、[base:120] |
| abort 400 | `Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none` | 实例未启用 grammar backend | `10503 / engine_error` | [grammar:139] |
| abort 400 | `Grammar preprocessing timed out: req.grammar_key=…`；缓存命中时也可为 `Failed to compile … grammar: Grammar preprocessing timed out` | grammar 编译超时，缓存的失败也会影响后续请求 | `10503 / engine_error` | [grammar:162]、[grammar:292] |
| 400 | `SGLANG_INKLING_DEFAULT_REASONING_EFFORT must be numeric` / `…must be in [0.0, 0.99]` | 服务端环境变量配置非法，不是用户 reasoning_effort | `10503 / engine_error` | [chat:674] |
| 400 | `max_thinking_tokens requires the server to be launched with --enable-strict-thinking` | 请求能力依赖的服务开关未启用 | `10503 / engine_error` | [tokenizer:777] |
| 400 | `The server is not configured to return hidden states.` / `The requested return_hidden_states mode exceeds the server maximum …` | hidden states 功能未启用或服务模式不足 | `10503 / engine_error` | [tokenizer:1225] |
| 400 | `The server is not configured to enable custom logit processor.` | 服务开关未启用 | `10503 / engine_error` | [tokenizer:1243] |
| 400 | `input_embeds is provided while disable_radix_cache is False.` | input_embeds 所需的服务配置不满足 | `10503 / engine_error` | [tokenizer:970] |
| 400 | `The engine initialized with skip_tokenizer_init=True cannot accept text prompts.` | 接受文本请求的实例未加载 tokenizer | `10503 / engine_error` | [tokenizer:981] |
| 400 | `stop=… is unavailable when skip_tokenizer_init=True…`，stop_regex/min_new_tokens 的同类提示 | 相关采样功能依赖 tokenizer；不能因为消息含采样字段就记 10403 | `10503 / engine_error` | [sampling:305] |
| 400 | `Multimodal inputs are not supported when --language-model-only is set; the encoder is not loaded. Restart without the flag.` | 请求发到没有 encoder 的实例 | `10503 / engine_error` | [tokenizer:999] |
| 503 | `The encoder did not return multimodal embeddings. The request was not run locally in language-only mode.` | EPD encoder 未返回有效 embedding | `10503 / engine_error` | [tokenizer:179] |
| 400 | `Invalid request: Disaggregated request received without bootstrap room id. …` | P/D 内部必需字段缺失；当前 AIgate 负责注入 | `10503 / engine_error` | [scheduler:2497] |
| 400 | `Invalid X-Data-Parallel-Rank header: must be an integer, got '{value}'` / `routed_dp_rank={rank} out of range [0, {size})` | DP 路由字段非法或实例拓扑不匹配 | 内部路由问题 `10503 / engine_error` | [base:287]、[tokenizer:787] |
| 400 | `invalid {header} header {value!r}: {e}` | `x-override-bootstrap-port/room/routed-dp-rank/…` 的类型转换失败 | 内部覆盖字段问题 `10503 / engine_error` | [headers:9] |
| 400 | `Duplicate request ID detected: {rid}` / `Duplicate request IDs detected within the request: {duplicates}` | 单请求冲突或 batch 内重复；当前 AIgate 注入 rid | `10503 / engine_error` | [tokenizer:3408]、[native:356] |
| 400 | `LoRA adapter '{name}' was requested, but LoRA is not enabled.` | 实例未启用 LoRA | `10503 / engine_error` | [tokenizer:3298] |
| 400 | `Got LoRA adapter that has never been loaded: {name}…` | adapter 未注册；用户选错名字与实例漏加载需分开 | 违反公开 adapter 列表才 `10401 / invalid_request`；已公开模型漏部署则 `10503 / engine_error` | [tokenizer:3342] |
| 400 | `Received request with {n} unique loras requested but max loaded loras is {limit}` | 请求需要的 adapter 数超过实例加载上限 | 默认 `10503 / engine_error`；只有公开请求数量限制被违反才 `10401 / invalid_request` | [tokenizer:3325] |
| 400 | `Failed to implicitly load LoRA adapter {name}: {detail}` | 已注册 adapter 自动重新加载失败 | `10503 / engine_error` | [tokenizer:3357] |
| 500 | `Failed to parse reasoning content` | 对模型生成结果进行 reasoning 解析失败 | `10503 / engine_error` | [chat:1920] |

官方 [LoRA Serving](https://docs.sglang.io/docs/advanced_features/lora) 说明 adapter 的启用、预加载和动态加载机制，可用于区分“用户选择不存在的公开 adapter”和“服务未部署已承诺的 adapter”。模型能力应在 AIgate 入口校验；请求已被接受后因后端缺配置失败，默认按服务问题处理。

#### E. 调度、P/D、资源和运行时故障

| SGLang 状态 | 错误消息 / 源码模板 | 触发原因及 AIgate 判责 | ret_code / 对外 error.code | 源码 |
| --- | --- | --- | --- | --- |
| abort 503 | `The request queue is full.` | Scheduler 等待队列已满；这里是实例过载，**不是用户配额 429** | `10503 / engine_error` | [scheduler:2840] |
| abort 503 | `The request is aborted by a higher priority request.` | 低优先级请求被更高优先级请求抢占 | `10503 / engine_error` | [scheduler:2851] |
| abort 503 | `Using priority is disabled for this server. Please send a new request without a priority.` | 服务关闭优先级调度且配置为拒绝携带 priority 的请求 | 已接受请求的后端能力/配置问题，默认 `10503 / engine_error` | [scheduler:2822] |
| abort 503 | `Request waiting timeout reached.` | 超过 `SGLANG_REQ_WAITING_TIMEOUT` | `10503 / engine_error` | [scheduler:2887] |
| abort 503 | `Request running timeout reached.` | 超过 `SGLANG_REQ_RUNNING_TIMEOUT` | `10503 / engine_error` | [scheduler:1665] |
| abort 500 | `Prefill bootstrap failed for request rank=…` | Prefill bootstrap 建链失败，可能追加底层异常 | `10503 / engine_error` | [prefill:985] |
| abort 500 | `Decode handshake failed for request rank=…` | Decode 与 Prefill 握手失败，可能追加底层异常 | `10503 / engine_error` | [decode:897] |
| abort 500 | `Prefill transfer failed for request rank=…` | Prefill KV 发送失败 | `10503 / engine_error` | [prefill:950] |
| abort 500 | `Decode transfer failed for request rank=…` | Decode KV 接收或 HiCache 恢复失败 | `10503 / engine_error` | [decode:2270] |
| abort 500 | `Metadata unexpectedly not ready after readiness gate (bootstrap_room=0)` | 就绪检查后元数据仍未就绪 | `10503 / engine_error` | [decode:2085] |
| abort 500 | `Metadata corruption detected - bootstrap_room mismatch` | P/D 元数据 room 不匹配，可能发生 buffer 冲突 | `10503 / engine_error` | [decode:2103] |
| abort 503 | `_reclaim_swa_tail_capacity` 返回的容量回收失败诊断 | Decode 无法回收 SWA KV 预分配空间；消息依具体分支变化 | `10503 / engine_error` | [decode:1303] |
| abort 500 | `Grammar accept_token failed for req {rid} with token {id}: {e}` | 生成 token 无法被 grammar matcher 接受；属于运行中的生成故障 | `10503 / engine_error` | [prefill:764] |
| abort 500 | `Retraction host KV pool exhausted. Aborting the request.` | 回退请求时，host KV 池无法保存备份 | `10503 / engine_error` | [batch:2883] |
| abort 500 | `Out of memory even after retracting all other requests in the decode batch. Aborting the last request.` | 已回退其他请求，仍不能满足剩余请求的内存需求 | `10503 / engine_error` | [batch:2900] |
| 500 或无完整响应 | `Internal server error: {e}`；CUDA/OOM、依赖错误、worker 退出等动态异常 | 捕获到的异常可被包装为 500；进程退出/流中异常也可能仅表现为断连，并不保证有 JSON | `10503 / engine_error` | [base:127] |
| 501 | `{handler} does not support streaming requests` / `…non-streaming requests` | 调用未实现的默认 handler；不应把“所有 NotImplementedError”都映射为 501 | `10503 / engine_error` | [base:160] |
| 400 或无可写响应 | `Request is disconnected from the client side (type 1). Abort request …` / `…(type 3). Abort request …`；`Client disconnected` | SGLang 所称 client 是直接调用者，可能就是 AIgate；不能据此判定终端用户取消 | 必须与 AIgate 的用户断连/内部取消原因关联；不明原因按 `10503 / engine_error` | [tokenizer:1711]、[tokenizer:1792]、[responses:639] |

P/D bootstrap、传输阶段及超时配置见官方 [PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)。根因与连带取消应分别记录：某一端报错后 AIgate 取消另一端，不应让后发生的取消覆盖原错误。

#### F. 原生 Anthropic 与 Responses 扩展入口

这部分用于扩展错误字典。仅在 AIgate 实际调用对应上游入口时启用这些规则；经 AIgate 转为 OpenAI Chat 的请求，按转换后入口读取错误，同时保留原始字段来源。

| SGLang 入口 / 状态 | 错误消息 / 源码模板 | 触发原因及 AIgate 判责 | ret_code / 对外 error.code | 源码 |
| --- | --- | --- | --- | --- |
| Anthropic / 400 | `Model is required` / `max_tokens must be positive` | 请求必需字段缺失或 max_tokens 非正数 | `10401 / invalid_request`；max_tokens 归 `10403 / invalid_sampling_parameters` | [anthropic_protocol:386] |
| Anthropic / 400 | `input_schema must be a dictionary` | 工具 schema 不是对象 | `10402 / invalid_tool_definition` | [anthropic_protocol:147] |
| Anthropic / 400 | `thinking.budget_tokens is required when thinking.type is 'enabled'` | enabled 缺预算字段 | `10401 / invalid_request` | [anthropic_protocol:281] |
| Anthropic / 400 | `thinking.budget_tokens must be >= 1024 (got {n})` | thinking 预算小于协议要求 | `10401 / invalid_request` | [anthropic_protocol:290] |
| Anthropic / 400 | `thinking.budget_tokens is not allowed when thinking.type is 'disabled'` / `…'adaptive'`；`thinking.display is not allowed when thinking.type is 'disabled'` | thinking 模式与字段冲突 | `10401 / invalid_request` | [anthropic_protocol:295] |
| Anthropic / 400 | `Anthropic redacted_thinking history is not supported` | 用户提交当前兼容入口不支持的历史块；AIgate 自己转换生成则服务责任 | `10401 / invalid_request`；否则 `10503 / engine_error` | [anthropic:386] |
| Anthropic / 400 | `tool_choice references tool {name!r} but it is not in the forwarded tools list …` | 指定工具不在可转发工具列表，包含被过滤的内置工具 | 用户选择违反公开支持列表才 `10402 / invalid_tool_definition`；否则 `10503 / engine_error` | [anthropic:714] |
| Anthropic / 400 | `tool_choice=… requires at least one custom tool; all supplied tools were server-side Anthropic built-ins …` | required/指定工具只剩 backend 无法执行的内置工具 | 按公开工具能力及转换责任判 `10402 / invalid_tool_definition` 或 `10503 / engine_error` | [anthropic:720] |
| Chat 转换为 400；原生 Anthropic 内部转换可能 500 | `Cannot rewrap thinking history: no reasoning detector is configured for this model` / `Anthropic thinking is not supported for models without a reasoning parser` | 依赖的 reasoning parser 未配置；不能把已包装成通用 500 的错误反向猜成用户原因 | `10503 / engine_error` | [chat:2330]、[chat:2370]、[anthropic:807] |
| Chat 转换为 400；原生 Anthropic 内部转换可能 500 | `Reasoning parser '{name}' is always-on and cannot be disabled via Anthropic thinking` / `Anthropic thinking is not supported for reasoning parser '{name}'` | parser 能力与请求模式冲突 | 默认 `10503 / engine_error`；在 AIgate 入口独立证实违反公开模式约束时可拒为 `10401 / invalid_request` | [chat:2399]、[chat:2421] |
| Responses / 400 | `Model not loaded` | 没有可用 tokenizer manager；400 仍是服务故障 | `10503 / engine_error` | [responses:239] |
| Responses / 400 | `tool_choice="required" requires at least one tool with type="function"; other built-in tool types cannot be forced.` | required 却没有 function tool | `10402 / invalid_tool_definition` | [responses:246] |
| Responses / 400 | `Cannot combine tool calls with constrained decoding (text.format / regex / ebnf / structural_tag / json_schema). Remove one.` | 工具约束与另一种输出约束冲突；与 Chat 的错误文本不同 | `10406 / invalid_output_constraint` | [protocol:1827] |
| Responses / 400 | `logprobs are not supported with gpt-oss models` / `logprobs are not supported in streaming mode` | 请求不支持的 logprobs 模型/模式组合 | 违反公开能力约束才 `10401 / invalid_request`；否则 `10503 / engine_error` | [responses:255] |
| Responses / 400 | `structured output (text.format) is not supported with gpt-oss models` | 当前 Harmony 实现不支持该约束 | 违反公开能力约束才 `10406 / invalid_output_constraint`；否则 `10503 / engine_error` | [responses:265] |
| Responses / 400 | `web_search requires a browser backend. Set EXA_API_KEY on the SGLang server …` | browser 服务或密钥缺失 | `10503 / engine_error` | [responses:272] |
| Responses / 400 | `MCP tool server is not supported in background mode and streaming mode` | 服务所选工具 backend 与请求模式不兼容 | 默认 `10503 / engine_error`；公开禁用组合且用户违反时 `10401 / invalid_request` | [responses:325] |
| Responses / 400 | `Unsupported Responses API input item type: {type!r}` | 不支持的输入项类型 | `10401 / invalid_request` | [responses:1112] |
| Responses / 400 | `Invalid 'response_id': '{id}'. Expected an ID that begins with 'resp'.` | 实际代码检查 `resp_` 前缀，消息写 `resp` | 用户提供非法 ID 才 `10401 / invalid_request` | [responses:1445] |
| Responses / 404 | `Response with id '{id}' not found.` | response_store 找不到 ID；它是实例内存存储，路由错误、重启丢失也可能发生 | 用户提交不存在/失效的公开 ID 才 `10401 / invalid_request`；状态或路由丢失 `10503 / engine_error` | [responses:180]、[responses:1455] |
| Responses / 400 | `Unknown error` / `{e} {e.__cause__}` 等通用包装 | 预处理、工具或生成异常被默认错误方法包为 400 | `10503 / engine_error`；不能只看 `invalid_request_error` | [responses:315]、[responses:557] |

#### G. Embedding、Rerank、Tokenize、Native batch/session 等补充接口

以下不代表该 AIgate 分支已经对外开放这些接口。接入时复用同一分类器，并按公开字段来源启用规则；AIgate 内部生成的 token、batch、session 或 embedding 字段出错仍归服务。

| SGLang 入口 / 状态 | 错误消息 / 源码模板 | 触发原因及 AIgate 判责 | ret_code / 对外 error.code | 源码 |
| --- | --- | --- | --- | --- |
| Embedding / 400 | `encoding_format must be either 'float' or 'base64', got {v!r}` | 不支持的编码格式 | `10401 / invalid_request` | [embedding:44] |
| Embedding、Classify / 400 | `Input cannot be empty` / `Input cannot be empty or whitespace only` / `Input at index {i} cannot be empty or whitespace only` | 输入或 batch 元素为空 | `10401 / invalid_request` | [embedding:52]、[classify:82] |
| Embedding、Classify / 400 | `All items in input list must be strings` / `…must be integers` / `Token ID at index {i} must be non-negative` | batch 类型混合或 token 为负 | `10401 / invalid_request` | [embedding:66]、[classify:103] |
| Embedding / 400 | `This model does not appear to be an embedding model by default. Please add …--is-embedding…` | embedding 请求到生成模式实例 | 路由/启动模式问题 `10503 / engine_error` | [tokenizer:1211] |
| Embedding / 400 | `Requested dimensions must be greater than 0` / `Provided dimensions are greater than max embedding dimension: {n}` | 请求维度非法 | 违反公开维度范围才 `10401 / invalid_request`；否则 `10503 / engine_error` | [tokenizer:1278] |
| Embedding / 400 | `Model '…' does not support matryoshka representation, changing output dimensions will lead to poor results.` / `Model '…' only supports … matryoshka dimensions…` | 模型不支持缩维或所选维度 | 违反公开模型能力才 `10401 / invalid_request`；部署不符 `10503 / engine_error` | [tokenizer:1267] |
| Embedding / 400 | `embed_override_token_id is required when embed_overrides is provided` / `embed_override_token_id requires embed_overrides to be provided` | embedding override 的字段依赖不满足 | 原始用户字段非法才 `10401 / invalid_request`；否则 `10503 / engine_error` | [embedding:156] |
| Rerank / 400 | `Query cannot be empty` / `Query cannot be empty or whitespace only` / `Documents cannot be empty` / `Each document must be a non-empty string` / `Each document cannot be empty or whitespace only` | query、documents 或元素为空 | `10401 / invalid_request` | [rerank:225] |
| Rerank / 400 | `Value error, parameter top_n should be larger than 0.` | top_n 非正数 | `10401 / invalid_request` | [protocol:1416] |
| Rerank / 400 | `Invalid rerank request adaptation…` / `Invalid embedding score for rerank at index …` | 模式/模板适配错误，或生成的评分结果非法；方法内部分异常直接包成 400 | `10503 / engine_error` | [rerank:302]、[rerank:390]、[rerank:577] |
| Tokenize / 400 | `Exactly one of 'prompt' or 'messages' must be provided.` / `Invalid prompt type: …` | 输入互斥或类型错误 | `10401 / invalid_request` | [protocol:1467]、[tokenize:70] |
| Detokenize / 400 | `Invalid input: 'tokens' must be a list of integers.` / `Invalid input: Sublist in 'tokens' must contain only integers. Found: …` / `Invalid tokens type: …` | token 列表形状、类型不合法 | `10401 / invalid_request` | [tokenize:138] |
| Tokenize / 400 | `Chat template tokenization requires a template manager.` / `Failed to render chat messages into token ids.` | 模板管理器或编码配置缺失 | `10503 / engine_error` | [tokenize:89] |
| Detokenize / 400 | `Error decoding tokens: {e}. Input tokens might be invalid for the model.`，type=`DecodeError` | 这里按异常文本含 `decode` 来选 400，不足以证明用户 token 有错 | 默认 `10503 / engine_error`；独立证实原始 token 非法才 `10401 / invalid_request` | [tokenize:177] |
| Tokenize / 500 | `Internal server error during tokenization: {e}` / `Internal server error during detokenization: {e}` | tokenizer/解码运行时异常 | `10503 / engine_error` | [tokenize:78]、[tokenize:185] |
| Score / 400 | `label_token_ids is required for generation (CausalLM) models.` / `items must be provided` / `Token ID {id} is out of vocabulary (vocab size: {n})` | 评分输入缺失或 token 越界 | `10401 / invalid_request` | [score:479] |
| Score / 400 | `embed_override_token_id is required when query_embed_overrides or item_embed_overrides are supplied.` / `item_first is not supported when embeddings are supplied` / `item_embed_overrides length (…) must match items length (…).` | 评分 embedding 输入组合/长度错误 | 用户原始字段非法才 `10401 / invalid_request`；内部构造则 `10503 / engine_error` | [score:488] |
| Native / 400 | `Either text, input_ids or input_embeds should be provided.` / `input_ids cannot be empty.` | 生成输入缺失 | `10401 / invalid_request` | [native:414]、[native:446] |
| Native / 400 | `session_id and session_params cannot both be set` | 两种 session 参数同时指定 | 用户原始字段冲突才 `10401 / invalid_request` | [native:395] |
| Native batch / 400 | `Text should be a list for batch processing.` / `input_ids should be a list of lists for batch processing.` / `input_embeds should be a list for batch processing.` | batch 形状错误 | 用户原始 batch 非法才 `10401 / invalid_request`；内部合批错误 `10503 / engine_error` | [native:542] |
| Native batch / 400 | `The length of image_data should be equal to the batch size.`；rid、lora_path、mm_content_hashes、return_hidden_states 的对应长度错误 | batch 配套字段与样本数不一致 | 用户原始 batch 非法才 `10401 / invalid_request`；网关注入/合批错误 `10503 / engine_error` | [native:582]、[native:617]、[native:682]、[native:736] |
| Native batch / 400 | `The parallel_sample_num should be the same for all samples in sample params.`；parallel_sample_num > 1 与列表参数的组合错误 | 并行采样和 batch 参数不兼容 | 用户原始组合非法才 `10401 / invalid_request`；内部合批错误 `10503 / engine_error` | [native:474]、[native:699] |
| Native batch / 400 | extra_key、cache_salt 必须是合法类型、逐项字符串且数量匹配 | 缓存隔离字段或批次结构错误，具体消息随字段变化 | 用户公开字段非法才 `10401 / invalid_request`；内部缓存键错误 `10503 / engine_error` | [native:768]、[native:787] |
| Native session / abort 400 | `Invalid request: close was requested for session {id}` / `Invalid request: session id {id} does not exist` | session 正在关闭或当前实例找不到 | 用户误用公开 session 才 `10401 / invalid_request`；粘性路由、内部 session 生命周期错误 `10503 / engine_error` | [scheduler:2540] |
| 模型查询 / 404 | `The model '{model}' does not exist`，code=`model_not_found` | `GET /v1/models/{model}` 查询失败；不能据此推断 Chat 一定做同样校验 | 用户查询不存在的公开模型可记 `10401 / invalid_request`；AIgate 模型映射/路由错误 `10503 / engine_error` | [http:1895] |

官方 [Embedding API](https://docs.sglang.io/docs/basic_usage/openai_api_embeddings) 说明生成式 embedding 模型的启动模式要求；[Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache) 说明 session 机制。两者用于确认功能和配置语义，具体错误文本仍以上述源码为准。

#### H. 其他状态与容易误当成错误的返回

| 返回 / 现象 | AIgate 处理原则 | 依据 |
| --- | --- | --- |
| 内部 401 / 403 | 默认 `10503 / engine_error`，不能当作终端用户 API key 错误；AIgate 自己的入口鉴权另行处理 | [auth:180] |
| 未知路由 404、方法不匹配、上游代理 413/422/429 | 不把状态自动映射为用户责任。413 需要公开大小限制证据；422 需要原始用户字段错误证据；429 需要公开用户/租户配额证据。当前 Python 主应用字段校验为 400、Scheduler 队列满为 503 | [http:612]、[scheduler:2840] |
| HTTP 500 / 502 / 503 / 504，代理 HTML、空错误体、无法识别的错误结构 | 默认 `10503 / engine_error`；没有上游固定 message 也必须覆盖。`http status` 与内部责任码分别记录 | [base:127]；AIgate `upstreamHTTPRet` 当前处理位置见第 3 节 |
| `/health`、`/health_generate` 的 503 | 启动中、停止中或健康探测超时，属于实例可用性；不扩充为用户错误 | [http:679] |
| 另行接入 SGLang Model Gateway（Rust） | 其 helper 用字符串 `error.code`，并设置 `X-SMG-Error-Code`；含 400/404/405/424/500/501/502/503 helper。与 Python 数字 code 分开解析，按路由原因判责；当前 AIgate 直连链路不应凭空套用这些码 | [gateway:22] |
| 另行接入 Rust `sglang-server` | 其 OpenAI 错误 helper 可以主动返回 HTTP 200 + SSE error + `[DONE]`；同样不能只看 HTTP status | [rust:90] |
| `finish_reason=stop/length/tool_calls` | 是生成结束原因；`length` 通常表示达到本次生成长度，不能当作输入 context 超限 | [chat:1644] |
| `FINISH_ABORT` 没有 status_code，message 为 `Aborted` | 可以是主动 abort 或 session 清理；当前 Chat 会走正常 chunk 路径。须结合 AIgate 自己的取消来源判断，不能见 abort 就一律报用户错误或引擎错误 | [batch:282]、[chat:1625] |
| `Tool call parsing error: …` | 当前 required-tool 的一个分支只记日志，退回普通文本；不是返回给 AIgate 的错误消息，不应作为响应匹配规则 | [chat:2203] |
| 自动截断的 warning、Anthropic thinking budget/display 的兼容性 warning | 警告不一定中断请求；不能把日志里的超限/不支持文案当作已经收到错误响应 | [tokenizer:1169]、[anthropic_protocol:270] |
| 错误之后仍有 usage / `[DONE]` | 错误状态不可被尾帧覆盖；本例是明确的失败，按后文 SSE 方案处理 | [chat:1633]、[chat:1769] |

本次仅在现有 Chat / Messages 推理出口启用分类，消息模式固定为上述 SGLang 源码版本。分类使用 `状态语义 + 精确消息模式 + 原始字段证据`；角色/阶段用于诊断，不单独决定用户责任。内部 header/rid/bootstrap、部署开关、容量、超时等先命中服务规则；剩余错误仅在用户白名单和证据同时成立时记录 104xx，其余记录 10503。所有原始诊断都不直接成为公开 message，`ret_code` 始终只写日志和指标。

### 返回 AIgate 自己的错误

分类函数一次性给出责任、错误原因、公开 `code/type/message` 和 HTTP status。API 只序列化公开错误字段；日志和指标根据同一个分类结果记录 `ret_code`。

```json
{
  "id": "<sid>",
  "error": {
    "code": "context_length_exceeded",
    "type": "invalid_request_error",
    "message": "The request exceeds the model's published token limit. Please shorten the input or reduce the requested output."
  }
}
```

服务错误对外使用通用服务失败提示，日志和指标记录 `ret_code=10503`；原始上游错误仅作受控诊断，不直接公开。非流式和建流前统一为：用户参数错误返回 HTTP 400；被判为服务责任的上游 4xx 改为 HTTP 503；已明确的上游 5xx 保留其 HTTP 语义。HTTP status 不参与再次推导责任码。

### 补齐 SSE 和 P/D

- **SSE**：提交下游响应前预读完整首个非心跳响应帧；若为 error，直接返回分类后的 HTTP 错误和 JSON。已提交 HTTP 200 后发生错误，则改写为 AIgate SSE error，不添加 `ret_code`；日志和指标记录失败，不能再转换成正常 `end_turn/message_stop`。
- **Prefill**：HTTP 200 只表示响应已建立，还要检查 ACK/body/SSE 的错误语义；失败后取消 Decode。
- **双端失败**：保留两端诊断，用已确认的原始失败决定最终分类；`pd_peer` 取消不能覆盖根因，也不能被记为用户取消。无法确定因果则按服务失败处理，记录 10503。

SGLang 的流式错误编码见 [serving_chat.py:1633]；P/D 内部链路语义参见官方 [PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)。

### 处理 Scheduler 长度拒绝后的 `error → usage → [DONE]`

截图中的输入长度为 `986483`，Scheduler 上限为 `484090`，却收到 HTTP 200 SSE。源码路径如下：

| 环节 | 当前行为 |
| --- | --- |
| `tokenizer_manager.py:1160` | 按模型 context 校验，失败抛 `ValueError`；这部分可以在建流前返回 HTTP 400 |
| `scheduler.py:2654` → `managers/utils.py:206` | 按 `max_req_input_len` 再次校验，失败调用 `set_finish_with_abort`；流式处理返回 abort 对象，不抛异常 |
| `serving_chat.py:1515、1633` | 首项预取拿到正常 yield 的 error 帧，仍创建 HTTP 200 的 StreamingResponse |
| `serving_chat.py:1586、1769` | 先收集 usage，abort 后只跳出生成循环，仍会追加 usage 和 `[DONE]` |
| AIgate 基线 `requestBody`、`copyStreaming` | 非 Anthropic passthrough 的流请求，在未指定时默认设置 `include_usage=true`；原实现先写出响应字节，再做观测解析。本次在 `guardStream` 前置错误检查，保留正常流默认 usage 行为 |

Scheduler 上限受 context、KV 容量和预留空间共同约束，见 [tp_worker.py:532]。因此，截图中的 `484090` 不能直接当成公开模型 context。`set_finish_with_abort` 还会将输入替换成一个占位 token，见 [schedule_batch.py:1844]；截图中 `prompt_tokens=1` 不能代表原始输入长度，错误后的 usage 不能作为成功请求的有效用量。

**本次 AIgate 已实现以下拦截，正常请求仍保留 include_usage：**

1. 在 `decodeResponse` 中、协议转换及下游 Write/Flush 之前，按完整 SSE frame 预读。跳过心跳，使用读取超时和帧大小上限；首帧为 error 时关闭上游并返回 AIgate JSON 错误，不转发后续 usage。正常帧须保留并按序转发。
2. 已开始输出后收到 error，设置失败终态、输出规范化错误并关闭上游；丢弃后续普通帧、usage 和扩展信息，包括同一次 Read 中紧随 error 的帧。OpenAI 可由 AIgate 补一个 `[DONE]`，不能据此记成功。
3. 本例在生成前被拒绝，记录 `usage_valid=false`、`usage_missing_reason=upstream_rejected`，并写入判责后的 `ret_code`。若此前已产生真实输出，保留已确认的部分用量，不能由失败后的 usage 覆盖，也不能将失败请求改记成功。

**后续可选的 SGLang 源头修复（本次未修改）**：在 `_generate_chat_stream` 中，首次正常输出前遇到带 status 的 abort，通过保留原 status 的异常路径返回 HTTP 错误；已开始输出则发送 SSE error。两种情况下都跳过正常 usage/扩展尾帧，并完成请求清理。AIgate 仍需保留上述拦截，以兼容未升级实例。

## 3. AIgate 实现设计

### 模块与调用顺序

| 模块 / 函数 | 已实现行为 |
| --- | --- |
| [upstreamerror/error.go]、[upstreamerror/classify.go] | `Parse → Classify → Result`；解析错误信封、按用户白名单及字段证据判责。`Result.Status/RetCode/Reason` 均标记 `json:"-"`，只公开 `code/type/message` |
| [session/session.go]、[frontend/http.go] | 在 base64 图片替换前保存 `OriginalJSON`，随后保留实际用于转发的 `RawJSON`；原始副本只供判责，不新增原始请求日志 |
| [inference/errors.go]：`withEvidence / upstreamFailure / DispatchError.Record` | 比较原始字段与最终发送 JSON；读取当前模型公开限制；分别记录角色诊断和请求最终分类 |
| [inference/client.go]：`DispatchAggregated / Dispatch / decodeResponse` | Worker/Prefill/Decode 共用分类器；非 2xx、HTTP 200 JSON error、SSE error 都生成 AIgate 错误。非流式错误不再裸透传 |
| [inference/guarded_stream.go]：`frameReader / guardStream / inspectPrefill` | 在转换前按完整 SSE 帧检查；兼容分片、同次 Read 多帧、多行 data、LF/CRLF；首帧前失败返回 JSON，流中失败发送规范化 SSE error 后关闭上游 |
| [inference/stream.go]：`StreamConverter.process` | 转换器增加 error 防线，错误不能转换为正常 Anthropic 结束事件 |
| [observability/usage.go]：`StreamParser / ParseUsageJSON` | error 帧中的 usage 及后续 usage 无效；保留错误前已确认用量；不以占位 token 或正文估算实际用量 |
| [observability/retcode.go]、[observability/otlp_log.go] | 接入 `104xx/10503`；`CloseRequestLog` 保留已分类结果，HTTP 200 不再覆盖失败；请求计数与日志采用同一 `ret_code` |

`Unchanged` 比较 JSON 值，而不是字段顺序或空白。用户未提供但 AIgate 注入的字段、被图片替换/Anthropic 转码/模板默认值改写的相关字段，不能支持用户判责。缺少原始请求或公开限制时保守归服务。

### 当前启用的用户规则与边界

| 已启用规则 | 判责证据 |
| --- | --- |
| 空 messages、工具列表/选择/唯一性/schema、历史工具参数、thinking 历史、return_* 参数组合 | 固定完整消息或锚定模式 + 相关原始字段未改变 + 相应字段存在或组合条件成立 |
| temperature、top_p、min_p、top_k、frequency/presence/repetition penalty；max_tokens 非正数 | 除消息模式外，独立检查用户原始数值范围 |
| min_new_tokens 的 Chat 入口错误 | 对应原始 `min_tokens`；比较原始最大输出及消息中的上限。内部将最大输出缩为 1 导致的冲突不能归用户 |
| logit_bias token ID 越界 | 原始 bias 中存在报错 key、未被改写，且该 key 超过错误给出的词表范围 |
| 输出约束互斥、缺少 schema、工具与输出约束冲突 | 用户实际提供对应约束且未被 AIgate 改写；部署 backend 未开启、动态 grammar 编译异常仍归服务 |
| 模型输入/总长度/输出长度，以及多模态展开后输入长度 | 版本固定消息中的请求 token 数 + 本地公开模型限制 + 原始相关字段未改变 |
| token_ids_logprob、reasoning_effort、原生 Anthropic input_schema / thinking 参数错误 | 固定消息与相关原始字段证据；转换导致相关字段变化时归服务 |
| 结构化 FastAPI `detail[]` | 当前仅接纳 `body.messages/tools` 的缺失/列表类型错误和 `body.stream` 的布尔类型错误；逐条核对原始字段，任一不确定则归服务 |

保守边界：

- **未单独实现的字典行统一走 `10503` 兜底。** 动态 Pydantic 文本、Jinja 异常、`Failed to compile … grammar`、`invalid stop_regex …`、媒体解码/下载故障、媒体数量/域名限制、模型特定能力冲突，当前均不凭消息片段判用户。后续需补公开能力配置或可信结构化证据及回归测试。
- 本分支原有 `requestBody` 会删除 `response_format`；本次不改变该兼容策略。原请求提供了该字段但最终未发送时，相关错误不能归用户。
- P/D 的 `Request … exceeds the maximum number of tokens` 固定归服务；Decode rebootstrap 的计数可能包含已生成 token，不能按输入限制误判。
- HTTP 401/403/404/413/429、未知 4xx、5xx、无错误体、证据冲突均归服务；上游 429 不等于用户配额超限。只对白名单接纳 400/422 语义；HTTP 与错误体数字状态冲突时归服务。
- Native / Responses / Embedding / Rerank / Tokenize 等目录保留为后续接入字典。本次不新增这些 API 路由，也不承诺自动启用其用户规则。Native abort 错误结构已可解析；不带 status 的普通 abort 不自动当作失败。

### 公开模型长度配置

在 Frontend TOML 的 `sglang.public_model_limits` 下按**对外模型名**配置。以下仅是语法示例，不能直接当作生产模型限制：

```toml
[sglang.public_model_limits."demo"]
context_tokens = 65536
max_input_tokens = 60000
max_output_tokens = 8192
```

- 默认值 `0` 表示未知；不从 `context_tokens` 自动推导输入或输出上限，也不从某个实例的错误反推公开上限。
- AIgate 当前不自行 tokenize；从版本固定的错误消息读取请求 token 数，结合自有配置判定。输入计数口径必须与服务 tokenizer、模板和多模态展开一致；预留空间应体现在公开输入限制中。
- `max_input_tokens` 用于确认输入超限，`max_output_tokens` 用于确认请求输出超限，`context_tokens` 用于输入加请求输出的总长度校验。总长度错误还要核对原始最大输出与消息中的输出数字一致。
- 配置必须非负；指定 context 时，输入/输出各自不能大于 context。配置项只决定错误判责，本次没有新增前置 token 校验或静默截断。
- 截图中的输入 `986483`：若该模型公开输入上限已确认是 `900000`，则返回 HTTP 400 / `context_length_exceeded`，日志记 `10404`；若公开上限是 `1000000` 或未配置，则按实例容量或不确定故障返回 HTTP 503 / `engine_error`，日志记 `10503`。实例 `484090` 不能替代这个判断。

### 流式状态与 P/D 失败选择

| 状态 | 对外行为 | 日志和用量 |
| --- | --- | --- |
| 尚未输出有效响应帧，收到 error | 关闭上游；返回分类后的 HTTP 400/5xx 和 JSON；不发送 SSE 200 | 记录失败及 `104xx/10503`；无有效 usage |
| 已输出正常帧，随后 error | 保留已发 HTTP 状态；OpenAI 输出 AIgate error + 一个 `[DONE]`；Anthropic 输出 `event: error` | 记录失败；丢弃同次/后续 Read 中错误之后的 usage，不输出正常 `message_stop/end_turn` |
| 等待首帧时仅收到心跳 | 心跳不提交下游响应，也不延长首帧总等待预算 | 超时按服务故障处理 |
| 帧超长、JSON 损坏、无终止标记且提前 EOF、流读取超时 | 建流前返回 502/504；流中用 AIgate SSE error 结束 | `10503`；已确认的错误不能被后续 EOF/取消覆盖 |
| Prefill 200 ACK/body/SSE 包含错误 | 立即失败并取消 Decode，不等待后续 usage/EOF | `prefill_ack_received=false`；保留 Prefill 原始诊断 |
| 一端失败，另一端收到 AIgate 的取消 | 首个有效失败决定结果 | `cancel_origin=pd_peer` 仅作角色诊断，不替换根因、不计用户取消 |
| 两端各自返回不同分类/原因 | 合并为服务失败，HTTP 503，`error_reason=pd_multiple_failures` | 保存两端诊断；不将独立失败声称为连带取消 |

缓冲限制复用 `sglang.max_response_body_bytes`（非流式体、单个 SSE 帧、Prefill ACK 总量）。非 2xx 的诊断体读取最多等待 `min(request_timeout, 250ms)`；完整响应和流式帧使用 `request_timeout`。正常流在首个完整非心跳帧检查后按需读取；单次上游预读缓冲为 32 KiB，不缓冲整条生成流。

### 日志、指标与兼容迁移

| 场景 | 本次 ret_code | 对外行为 |
| --- | --- | --- |
| 已确认的 SGLang 用户错误 | `10401/10402/10403/10404/10406` | 按第 2 节 AIgate 字符串错误码返回 |
| AIgate 本地请求解析/校验错误 | `10401` | 保留现有解析错误信封与字符串 code |
| AIgate 已确认请求模型未提供：`model_not_registered/model_not_configured` | `10407` | 保留现有模型错误信封 |
| AIgate 确认的客户端取消 | `10408` | 保留现有 499 或流中关闭行为；上游自称 client disconnected 不足以触发此码 |
| AIgate、Router、Directory、引擎服务故障及未知错误 | `10503` | SGLang 错误统一为 `engine_error`；其他本地路由/目录错误的公开字符串 code 保持原契约 |

原有 `101xx` 用户观测码迁移为上述 `104xx`；原有 `102xx/103xx/105xx/106xx` 服务观测码收敛到 `10503`。本次不分配 `10405/10409/10499`。服务细分由 `ret / error_reason / error_phase` 和角色字段保留，不能再依赖数字码区分 Router、网络、容量等故障。

角色诊断记录 `{role}_upstream_http_status/error_status/ret_code/reason/phase` 及有界的 `error_code/error_type/error_message`；请求最终记录 `ret/ret_code/error_reason/error_responsibility`。`__classified_result` 仅用于内部状态传递，不写公开响应和外发日志字段。`aigate_frontend_requests_total` 即使 HTTP 200 也使用失败结果与对应 `ret_code`。告警及跨版本历史查询需兼容新旧码；不把原始错误文本加入指标标签。

## 4. 回归验证与交付边界

新增测试覆盖：

- 分类：顶层/嵌套 error、HTTP 200 JSON、Native abort、Anthropic error、结构化 detail、未知/冲突状态、字段注入/改写、公开长度限制和内部容量区别。
- 流式：首帧失败、流中失败、1/7/32768 字节分片、CRLF/多行 data、`error → usage → [DONE]` 同次 Read、无 EOF 的 Prefill SSE error、帧大小/超时限制，以及正常流的顺序和背压。
- P/D：错误立即取消仍在等响应头的另一端；双方独立失败保留诊断；内部取消不覆盖根因。
- 对外与观测：真实 Client + Frontend + Prometheus 路径，验证首帧 HTTP 400/503、流中 HTTP 200 但失败指标、usage 无效、响应无原始引擎诊断及 `ret_code`；正常流默认 usage 保留。

常规验证命令：

```bash
go test ./internal/upstreamerror ./internal/config ./internal/protocols
go test -race ./...
git diff --check
```

当前环境验证说明（2026-09-06）：独立分类、配置、协议包可直接测试。目标分支依赖的私有 `git.iflytek.com/AIaaS/otlp-sdk/v3 v3.4.2` 无法下载，完整普通构建尚未验证。临时 Go overlay 只隔离 OTLP SDK 初始化/发送入口，保留实际请求日志归类与 Prometheus 实现，全仓库 race 测试通过；该验证不代表 OTLP SDK 集成通过。需要真实 etcd 的集成测试因未设置 `TEST_ETCD_ENDPOINTS` 跳过。获得私有依赖后应在无 overlay 环境重跑完整测试；本次没有修改 `go.mod/go.sum`，没有将替代 SDK 写入仓库。

[serving_base.py:114]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:114
[serving_chat.py:865]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:865
[sampling_params.py:151]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:151
[prefill.py:368]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/prefill.py:368
[decode.py:747]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/decode.py:747
[serving_chat.py:1633]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1633
[tp_worker.py:532]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tp_worker.py:532
[schedule_batch.py:1844]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/schedule_batch.py:1844

[base:110]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:110
[base:196]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:196
[http:612]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/http_server.py:612
[chat:1633]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1633
[base:114]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:114
[tokenizer:1646]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1646
[http:920]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/http_server.py:920
[http:2091]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/http_server.py:2091
[responses:198]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:198
[responses:315]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:315
[http:544]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/http_server.py:544
[anthropic:747]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/serving.py:747
[anthropic:807]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/serving.py:807
[auth:180]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/auth.py:180
[decompress:46]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/http_request_decompression.py:46
[chat:865]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:865
[completions:65]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_completions.py:65
[protocol:699]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:699
[http:658]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/http_server.py:658
[chat:884]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:884
[chat:891]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:891
[chat:894]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:894
[chat:901]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:901
[chat:906]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:906
[chat:919]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:919
[chat:127]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:127
[protocol:1566]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:1566
[protocol:1154]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:1154
[protocol:1827]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:1827
[chat:870]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:870
[chat:984]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:984
[protocol:976]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:976
[chat:643]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:643
[chat:976]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:976
[protocol:541]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:541
[protocol:716]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:716
[sampling:151]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:151
[sampling:156]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:156
[sampling:158]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:158
[sampling:148]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:148
[sampling:160]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:160
[sampling:164]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:164
[sampling:169]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:169
[sampling:173]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:173
[sampling:178]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:178
[sampling:183]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:183
[protocol:410]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:410
[sampling:193]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:193
[sampling:201]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:201
[chat:937]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:937
[completions:178]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_completions.py:178
[grammar:162]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/constrained/grammar_manager.py:162
[grammar:277]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/constrained/grammar_manager.py:277
[batch:1580]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/schedule_batch.py:1580
[sampling:212]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:212
[chat:925]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:925
[tokenizer:1160]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1160
[tokenizer:1184]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1184
[length:206]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/utils.py:206
[scheduler:2654]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/scheduler.py:2654
[scheduler:2640]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/scheduler.py:2640
[decode:761]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/decode.py:761
[tokenizer:1315]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1315
[tokenizer:1295]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1295
[chat:944]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:944
[tokenizer:1252]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1252
[media:1579]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1579
[media:1584]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1584
[media:1640]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1640
[media:1713]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1713
[media:1900]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1900
[media:1961]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1961
[media:1764]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1764
[media:1976]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1976
[media:1618]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1618
[media:1685]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/utils/common.py:1685
[mm:817]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/multimodal/processors/base_processor.py:817
[mm:1179]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/multimodal/processors/base_processor.py:1179
[tokenizer:1144]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1144
[chat_encoding:204]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/chat_encoding.py:204
[chat:1423]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1423
[base:120]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:120
[grammar:139]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/constrained/grammar_manager.py:139
[grammar:292]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/constrained/grammar_manager.py:292
[chat:674]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:674
[tokenizer:777]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:777
[tokenizer:1225]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1225
[tokenizer:1243]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1243
[tokenizer:970]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:970
[tokenizer:981]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:981
[sampling:305]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/sampling/sampling_params.py:305
[tokenizer:999]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:999
[tokenizer:179]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:179
[scheduler:2497]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/scheduler.py:2497
[base:287]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:287
[tokenizer:787]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:787
[headers:9]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/request_headers.py:9
[tokenizer:3408]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:3408
[native:356]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:356
[tokenizer:3298]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:3298
[tokenizer:3342]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:3342
[tokenizer:3325]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:3325
[tokenizer:3357]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:3357
[chat:1920]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1920
[scheduler:2840]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/scheduler.py:2840
[scheduler:2851]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/scheduler.py:2851
[scheduler:2822]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/scheduler.py:2822
[scheduler:2887]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/scheduler.py:2887
[scheduler:1665]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/scheduler.py:1665
[prefill:985]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/prefill.py:985
[decode:897]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/decode.py:897
[prefill:950]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/prefill.py:950
[decode:2270]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/decode.py:2270
[decode:2085]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/decode.py:2085
[decode:2103]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/decode.py:2103
[decode:1303]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/decode.py:1303
[prefill:764]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/disaggregation/prefill.py:764
[batch:2883]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/schedule_batch.py:2883
[batch:2900]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/schedule_batch.py:2900
[base:127]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:127
[base:160]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_base.py:160
[tokenizer:1711]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1711
[tokenizer:1792]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1792
[responses:639]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:639
[anthropic_protocol:386]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/protocol.py:386
[anthropic_protocol:147]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/protocol.py:147
[anthropic_protocol:281]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/protocol.py:281
[anthropic_protocol:290]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/protocol.py:290
[anthropic_protocol:295]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/protocol.py:295
[anthropic:386]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/serving.py:386
[anthropic:714]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/serving.py:714
[anthropic:720]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/serving.py:720
[chat:2330]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:2330
[chat:2370]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:2370
[chat:2399]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:2399
[chat:2421]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:2421
[responses:239]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:239
[responses:246]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:246
[responses:255]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:255
[responses:265]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:265
[responses:272]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:272
[responses:325]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:325
[responses:1112]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:1112
[responses:1445]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:1445
[responses:180]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:180
[responses:1455]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:1455
[responses:557]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py:557
[embedding:44]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_embedding.py:44
[embedding:52]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_embedding.py:52
[classify:82]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_classify.py:82
[embedding:66]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_embedding.py:66
[classify:103]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_classify.py:103
[tokenizer:1211]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1211
[tokenizer:1278]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1278
[tokenizer:1267]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1267
[embedding:156]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_embedding.py:156
[rerank:225]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_rerank.py:225
[protocol:1416]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:1416
[rerank:302]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_rerank.py:302
[rerank:390]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_rerank.py:390
[rerank:577]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_rerank.py:577
[protocol:1467]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/protocol.py:1467
[tokenize:70]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_tokenize.py:70
[tokenize:138]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_tokenize.py:138
[tokenize:89]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_tokenize.py:89
[tokenize:177]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_tokenize.py:177
[tokenize:78]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_tokenize.py:78
[tokenize:185]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_tokenize.py:185
[score:479]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager_score_mixin.py:479
[score:488]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager_score_mixin.py:488
[native:414]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:414
[native:446]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:446
[native:395]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:395
[native:542]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:542
[native:582]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:582
[native:617]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:617
[native:682]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:682
[native:736]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:736
[native:474]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:474
[native:699]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:699
[native:768]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:768
[native:787]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/io_struct.py:787
[scheduler:2540]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/scheduler.py:2540
[http:1895]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/http_server.py:1895
[http:679]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/http_server.py:679
[gateway:22]: /home/kaizhang36/sglang_spark/sglang/sgl-model-gateway/src/routers/error.rs:22
[rust:90]: /home/kaizhang36/sglang_spark/sglang/rust/sglang-server/src/api_server/openai.rs:90
[chat:1644]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1644
[batch:282]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/schedule_batch.py:282
[chat:1625]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1625
[chat:2203]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:2203
[tokenizer:1169]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/managers/tokenizer_manager.py:1169
[anthropic_protocol:270]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/anthropic/protocol.py:270
[chat:1769]: /home/kaizhang36/sglang_spark/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1769

[upstreamerror/error.go]: /home/kaizhang36/go/src/dev/AIgate/internal/upstreamerror/error.go
[upstreamerror/classify.go]: /home/kaizhang36/go/src/dev/AIgate/internal/upstreamerror/classify.go
[session/session.go]: /home/kaizhang36/go/src/dev/AIgate/internal/session/session.go
[frontend/http.go]: /home/kaizhang36/go/src/dev/AIgate/internal/frontend/http.go
[inference/errors.go]: /home/kaizhang36/go/src/dev/AIgate/internal/inference/errors.go
[inference/client.go]: /home/kaizhang36/go/src/dev/AIgate/internal/inference/client.go
[inference/guarded_stream.go]: /home/kaizhang36/go/src/dev/AIgate/internal/inference/guarded_stream.go
[inference/stream.go]: /home/kaizhang36/go/src/dev/AIgate/internal/inference/stream.go
[observability/usage.go]: /home/kaizhang36/go/src/dev/AIgate/internal/observability/usage.go
[observability/retcode.go]: /home/kaizhang36/go/src/dev/AIgate/internal/observability/retcode.go
[observability/otlp_log.go]: /home/kaizhang36/go/src/dev/AIgate/internal/observability/otlp_log.go
