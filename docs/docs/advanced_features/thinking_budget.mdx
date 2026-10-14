---
title: "Thinking Budget"
metatags:
    description: "SGLang thinking budget: control how many tokens a reasoning model spends thinking, from soft chat-template hints to hard per-request caps with strict thinking."
---

Reasoning models can spend an unbounded number of tokens inside the thinking phase. SGLang gives you three ways to control that, from soft hints to hard per-request caps:

1. **Chat-template hints** (`enable_thinking`, `thinking_budget`, `reasoning_effort`): the model's own chat template decides what to do with them. No token guarantee.
2. **Strict thinking** (`--enable-strict-thinking`): a grammar-level token filter that enforces an exact per-request budget. Recommended when you need a guarantee.
3. **Custom logit processor** (`--enable-custom-logit-processor`): a logit processor that forces the thinking phase to end at the budget. Use when the grammar backend cannot be used.

The examples below use [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f), but every mechanism except the built-in logit processors is model-agnostic: strict thinking derives its token ids from the configured reasoning parser, so it works for any model whose parser is set.

## Chat-template hints

The OpenAI chat completions API passes `chat_template_kwargs` through to the model's chat template:

```python Chat-template kwargs
response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[{"role": "user", "content": "What is 1+3?"}],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True, "thinking_budget": 512}
    },
)
```

Two things to know:

- `enable_thinking` toggles the thinking mode for hybrid models such as Qwen3. It is template-dependent.
- `thinking_budget` only does something if the model's own chat template defines that variable (the official Qwen3 template does). If the template does not define it, the kwarg is silently ignored.

The `reasoning_effort` request field is forwarded into the chat template the same way.

These are hints to the model, not token guarantees. The model can overshoot the budget.

## Hard budget with strict thinking

Launch the server with `--enable-strict-thinking` and a reasoning parser:

```bash Launch with strict thinking
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --reasoning-parser qwen3 \
    --enable-strict-thinking
```

Strict thinking requires a grammar backend that supports token filtering. `xgrammar` is the default grammar backend, so no `--grammar-backend` flag is needed. Startup fails if the configured backend cannot filter tokens.

Enable `--reasoning-parser` so responses separate `reasoning_content` from the final answer (see [Reasoning Parser](/docs/advanced_features/separate_reasoning)). Strict thinking derives the thinking-end token ids from the parser's `think_end_token` through the tokenizer, so it is not tied to any hardcoded ids.

### Behavior

- When the budget is spent during thinking, the vocab mask allows only the `</think>` token sequence, so the cap is exact. The model then produces the final answer.
- With strict thinking enabled and no budget set, behavior is unchanged except that model-specific excluded tokens (for Qwen3, `<tool_call>`, `</tool_call>`, `<|im_end|>`, `<|endoftext|>`) are blocked during the thinking phase.
- To apply a server-wide cap to every request, set the environment variable `SGLANG_MAX_THINK_TOKENS` (default `-1`, no cap).

### Native `/generate` API

Set `max_thinking_tokens` per request, and always pair it with `require_reasoning: true`:

```python /generate with a thinking budget
import requests

response = requests.post(
    f"http://localhost:{port}/generate",
    json={
        "text": prompt,
        "require_reasoning": True,
        "max_thinking_tokens": 512,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 2048,
        },
    },
)
```

<Warning>
Without `require_reasoning: true`, the grammar object starts in the generation state and `max_thinking_tokens` is silently ignored. Sending `max_thinking_tokens` to a server launched without `--enable-strict-thinking` returns a 400 error.
</Warning>

### OpenAI chat completions API

There is no `max_thinking_tokens` field on chat completions. Pass the budget through `custom_params` instead; the runtime picks up the `thinking_budget` key the same way. The chat path sets `require_reasoning` for you based on the thinking mode:

```python Chat completions with a thinking budget
response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[{"role": "user", "content": "What is 1+3?"}],
    extra_body={"custom_params": {"thinking_budget": 512}},
)

print(response.choices[0].message.reasoning_content)
print(response.choices[0].message.content)
```

## Hard budget with a custom logit processor

When the grammar backend cannot be used, a custom logit processor can enforce the budget instead. Launch the server with `--enable-custom-logit-processor`:

```bash Launch with custom logit processors
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --reasoning-parser qwen3 \
    --enable-custom-logit-processor
```

Pass a serialized processor class and the budget on each request. This works on both `/generate` and chat completions:

```python Logit processor with a thinking budget
from sglang.srt.sampling.custom_logit_processor import (
    Qwen3ThinkingBudgetLogitProcessor,
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[{"role": "user", "content": "What is 1+3?"}],
    extra_body={
        "custom_logit_processor": Qwen3ThinkingBudgetLogitProcessor.to_str(),
        "custom_params": {"thinking_budget": 512},
    },
)
```

The processor counts the tokens generated after the `<think>` start token. Once the budget is reached, it first forces a newline token, then forces `</think>`. The model may emit its own `</think>` right after the forced one; this is harmless.

### Built-in processors

SGLang ships processors in `python/sglang/srt/sampling/custom_logit_processor.py`, each with hardcoded thinking start / end / newline token ids:

| Processor | Model family |
| --- | --- |
| `Qwen3ThinkingBudgetLogitProcessor` | Qwen3 |
| `Glm4MoeThinkingBudgetLogitProcessor` | GLM-4.5 / GLM-4.6 |
| `DeepSeekR1ThinkingBudgetLogitProcessor` | DeepSeek-R1 |
| `InklingThinkingBudgetLogitProcessor` | Inkling |

These only work for models whose tokenizer produces those exact ids. For any other model, subclass `ThinkingBudgetLogitProcessor` with the model's own ids:

```python Custom thinking-budget processor
from sglang.srt.sampling.custom_logit_processor import (
    ThinkingBudgetLogitProcessor,
)

# Find the ids with your model's tokenizer, e.g.:
# tokenizer.encode("</think>", add_special_tokens=False)
class MyModelThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    THINKING_START_TOKEN_ID = 12345
    THINKING_END_TOKEN_ID = 12346
    NEW_LINE_TOKEN_ID = 198
```

## Verify the budget applies

Budget knobs can be silently ignored, so confirm the cap with a deterministic comparison:

1. Run a prompt with `temperature=0` and no budget. Record the thinking length.
2. Run the same prompt with the budget set, again at `temperature=0`.

A working cap ends the thinking exactly at the budget, usually mid-sentence. An ignored knob reproduces the baseline output token for token. Common silent-failure modes:

- Missing `require_reasoning: true` on the native `/generate` API.
- `chat_template_kwargs.thinking_budget` on a model whose chat template does not define the variable.
- A built-in logit processor whose hardcoded token ids do not match the model's tokenizer.
