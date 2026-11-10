# Reasoning effort

All three language models were trained to read a reasoning-effort hint in the
prompt. The server writes that hint into the prompt and does nothing else. It
never cuts the thinking short. `max_tokens` is the only hard limit, and
Anthropic `thinking.budget_tokens` is accepted but not enforced.

Set the effort with `reasoning_effort` (OpenAI `/v1/chat/completions`) or
`output_config.effort` (Anthropic `/v1/messages`, which is what Claude Code's
effort setting sends).

| Requested | GLM-5.3-Flash | Qwen3.8-Flash-Next | DeepSeek-V4.1-Flash |
|---|---|---|---|
| nothing | Max | xhigh | 75 (high) |
| `low` | Low | low | 50 |
| `medium` | High | medium (no hint) | 75 |
| `high` | High | xhigh | 75 |
| `xhigh` | Max | xhigh | 75 |
| `max` | Max | xhigh | 100 |
| float in [0, 0.99] | Max | rejected (HTTP 400) | value × 100 |
| `minimal` | Low | rejected (HTTP 400) | 75 |
| `none` | Max | thinking off | thinking off |

What reaches the model:

- **GLM-5.3-Flash:** `Reasoning Effort: Low`, `High` or `Max`. Anything the
  template does not know renders as Max. The model always thinks; Anthropic
  `thinking: {"type": "disabled"}` is rejected.
- **Qwen3.8-Flash-Next:** a sentence in the system prompt. `low` asks for brief
  thinking, `xhigh` for careful thinking, `medium` adds no sentence. Turn thinking
  off with `reasoning_effort: "none"`, `"chat_template_kwargs": {"enable_thinking": false}`
  or Anthropic `thinking: {"type": "disabled"}`.
- **DeepSeek-V4.1-Flash:** `Reasoning Effort: N (range 1-100, ...)` at the start
  of the conversation, only in thinking mode. OpenAI requests think only when
  they carry a `reasoning_effort` or `"chat_template_kwargs": {"thinking": true}`;
  Anthropic requests think when `thinking` is enabled or adaptive. An exact level
  goes in `"chat_template_kwargs": {"thinking": true, "reasoning_effort": 40}`.
  Unknown strings fall back to 75 with a warning in the server log.
  `SGLANG_DSV41_REASONING_EFFORT` changes the default.
