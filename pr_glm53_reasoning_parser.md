# Auto-detect reasoning / tool-call parsers by default; fix GLM-5.3 reasoning split

## Motivation

- `--reasoning-parser` and `--tool-call-parser` defaulted to `None`, so the parsers found by chat-template auto-detection were only applied when users passed `auto`. Without the flags, reasoning and raw tool-call markup stay in `content`.
- GLM-5.3 and GLM-5.3-Flash templates always open `<think>` in the generation prompt and have no `enable_thinking` toggle, so even with `glm45` enabled the parser never enters reasoning mode.

## Changes

- `arg_groups/fields/serving.py`: default `reasoning_parser` and `tool_call_parser` to `"auto"`.
- `parser/template_detection.py`: add the `glm53_always_think` reasoning-mode rule, so GLM-5.3 templates force reasoning.
- `parser/template_detection.py`, `entrypoints/engine.py`: when auto-detection finds no parser, log at info level without claiming `auto` was specified.

## Results

### GLM-5.3: low-effort arithmetic

GLM-5.3's prompt always ends in `<|assistant|><think>`. Here the model output `17*23=391</think>391`, with no opening `<think>` of its own.

A) No parser, the old default (reconstructed; the earlier live run returned the same shape):

```json
{
  "message": {
    "role": "assistant",
    "content": "17*23=391</think>391",
    "reasoning_content": null,
    "tool_calls": null
  },
  "finish_reason": "stop"
}
```

B) Parsers on (`auto` or `glm45`), without the `glm53_always_think` rule (reconstructed): identical to A. The parser never sees an opening `<think>`, so it treats everything as normal text.

C) Fixed (real, current server):

```json
{
  "message": {
    "role": "assistant",
    "content": "391",
    "reasoning_content": "17*23=391",
    "tool_calls": null
  },
  "finish_reason": "stop"
}
```

At default (max) effort it's the same pattern, with a longer reasoning part before `</think>`: `17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391`.

### GLM-5.3: tool call

Model output: `</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>`

A) No parser (reconstructed): the tool call comes back as text, and no `tool_calls` are parsed.

```json
{
  "message": {
    "role": "assistant",
    "content": "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
    "reasoning_content": null,
    "tool_calls": null
  },
  "finish_reason": "stop"
}
```

B) Parsers on, without the rule (reconstructed): the tool call parses, but a stray `</think>` leaks into `content`.

```json
{
  "message": {
    "role": "assistant",
    "content": "</think>",
    "reasoning_content": null,
    "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]
  },
  "finish_reason": "tool_calls"
}
```

C) Fixed (real):

```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "reasoning_content": null,
    "tool_calls": [
      {
        "id": "call_ecdc9584f55847fa9c6bb13d",
        "index": 0,
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
      }
    ]
  },
  "finish_reason": "tool_calls"
}
```

### GLM-5.2: GPQA question `gpqa-141` from our eval, thinking on

No parser, what our eval actually received (real, from the saved answer; `num_reasoning_tokens` 0):

```json
{
  "message": {
    "role": "assistant",
    "content": "We need answer MC. Need analyze. Density matrix rho=1/2(|0><0|+|1><1|) = I/2. Bloch vector r=(0,0,0) maximally mixed state center of Bloch sphere. Options A. Need final format Answer: A. Need maybe explain. Ensure last line exactly 'Answer: A'.</think>The density matrix is\n\n\\[\n\\rho=\\frac{1}{2}\\left(|0\\rangle\\langle 0|+|1\\rangle\\langle 1|\\right)=\\frac{I}{2}.\n\\]\n\nThis is the maximally mixed state, corresponding to the center of the Bloch sphere. Its Bloch vector is\n\n\\[\n\\mathbf{r}=(0,0,0).\n\\]\n\nAnswer: A",
    "reasoning_content": null
  },
  "finish_reason": "stop"
}
```

With a parser (reconstructed): GLM-5.2 needs no new rule, because its `enable_thinking` toggle defaults on and SGLang forces reasoning mode from that.

```json
{
  "message": {
    "role": "assistant",
    "content": "The density matrix is\n\n\\[\n\\rho=\\frac{1}{2}\\left(|0\\rangle\\langle 0|+|1\\rangle\\langle 1|\\right)=\\frac{I}{2}.\n\\]\n\nThis is the maximally mixed state, corresponding to the center of the Bloch sphere. Its Bloch vector is\n\n\\[\n\\mathbf{r}=(0,0,0).\n\\]\n\nAnswer: A",
    "reasoning_content": "We need answer MC. Need analyze. Density matrix rho=1/2(|0><0|+|1><1|) = I/2. Bloch vector r=(0,0,0) maximally mixed state center of Bloch sphere. Options A. Need final format Answer: A. Need maybe explain. Ensure last line exactly 'Answer: A'."
  },
  "finish_reason": "stop"
}
```

## Tests

- Unit tests covering parsers, template detection and server args pass.
- Live check: `sglang serve --model-path zai-org/GLM-5.3 --tp 8 ...` with no parser flags auto-detects `glm45` / `glm47`. Non-streaming, streaming and tool-call requests return reasoning in `reasoning_content` and parsed `tool_calls`.
