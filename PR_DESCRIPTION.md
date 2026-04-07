## PR Motivation

For reasoning-capable models like Qwen3, GLM45, Nemotron-3, and InternS1, the `enable_thinking` parameter in `chat_template_kwargs` controls whether the model should output reasoning content. However, the `/v1/responses` endpoint was not properly checking this parameter when determining whether to parse reasoning content.

This caused inconsistent behavior where:
- When `enable_thinking=false` was set, the model would still output reasoning content
- The reasoning parser would incorrectly process non-reasoning outputs as reasoning content
- User expectations about disabling thinking were not respected

This PR aligns the `/v1/responses` endpoint behavior with the `/v1/chat/completions` endpoint by properly checking the `enable_thinking` parameter before enabling reasoning parsing.

## PR Modifications

### 1. Enhanced `_make_response_output_items()` Method (`python/sglang/srt/entrypoints/openai/serving_responses.py`)

Added logic to check `enable_thinking` parameter for reasoning-capable models:

```python
# For models like qwen3/glm45/nemotron_3/interns1, check enable_thinking
# to determine if reasoning should be parsed, mirroring serving_chat.py logic
enable_reasoning = True
if self.reasoning_parser in ["qwen3", "glm45", "nemotron_3", "interns1"]:
    enable_reasoning = (
        not request.chat_template_kwargs
        or request.chat_template_kwargs.get("enable_thinking") is not False
    )
```

**Key Changes:**
- For Qwen3, GLM45, Nemotron-3, and InternS1 models, check the `enable_thinking` parameter
- If `chat_template_kwargs` is not provided or `enable_thinking` is not explicitly set to `False`, enable reasoning (default behavior)
- If `enable_thinking=false` is explicitly set, disable reasoning parsing
- Other models without reasoning parsers are unaffected

### 2. Alignment with serving_chat.py

This change mirrors the existing logic in `serving_chat.py` to ensure consistent behavior across all OpenAI-compatible endpoints:
- `/v1/chat/completions` - Already checks `enable_thinking`
- `/v1/responses` - Now also checks `enable_thinking` (with this PR)

## Behavior Changes

| Model | enable_thinking | Before | After |
|-------|----------------|--------|-------|
| qwen3/glm45/nemotron_3/interns1 | Not set or true | Reasoning parsed | Reasoning parsed (unchanged) |
| qwen3/glm45/nemotron_3/interns1 | false | Reasoning parsed (incorrect) | Reasoning disabled (fixed) |
| Other models | Any | Original behavior | Original behavior (unchanged) |

## Supported Models

This fix applies to the following reasoning-capable models:
- **Qwen3** (`qwen3`)
- **GLM45** (`glm45`)
- **Nemotron-3** (`nemotron_3`)
- **InternS1** (`interns1`)

## Backward Compatibility

- **Default behavior unchanged**: When `enable_thinking` is not specified, reasoning is enabled by default
- **Explicit disable now works**: When `enable_thinking=false` is set, reasoning is properly disabled
- **Non-reasoning models unaffected**: Models without reasoning parsers continue to work as before

## Testing

- [x] Verified reasoning is parsed when `enable_thinking` is not set
- [x] Verified reasoning is parsed when `enable_thinking=true` is set
- [x] Verified reasoning is disabled when `enable_thinking=false` is set
- [x] Verified non-reasoning models are not affected by this change
- [x] Verified behavior matches `/v1/chat/completions` endpoint
