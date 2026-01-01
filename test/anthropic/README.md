# Anthropic API Tests

Integration tests for SGLang's Anthropic Messages API compatibility.

## Setup

1. Start SGLang server:
```bash
python -m sglang.launch_server \
    --model-path <model> \
    --port 8000 \
    --tool-call-parser <parser> \  # required for tool calling tests
    --api-key <key>                # optional, for auth tests
```

2. Set environment:
```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="your-api-key"  # must match --api-key if set
```

## Run Tests

```bash
# From python/ directory
cd python/
uv sync --extra dev

# Run all tests
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=your-key \
uv run pytest ../test/anthropic/tests/ -v

# Run specific test file
uv run pytest ../test/anthropic/tests/test_basic_chat.py -v
```

## Test Files

| File | Description |
|------|-------------|
| `test_basic_chat.py` | Simple messages, system prompts, multi-turn, sampling params |
| `test_streaming.py` | SSE streaming, event sequence, text_stream helper |
| `test_tool_calling.py` | Tool calls, tool results, multi-tool, streaming tools |
| `test_tool_choice.py` | tool_choice modes: auto, any, specific tool |
| `test_authentication.py` | x-api-key and Bearer token auth, 401 rejection |
| `test_error_handling.py` | Validation errors, malformed requests |
| `test_edge_cases.py` | Unicode, emoji, special chars, long conversations |
| `test_token_counting.py` | `/v1/messages/count_tokens` endpoint |

## Test Naming Convention

Tests follow the pattern: `test_<endpoint>__<expected_behavior>`

Examples:
- `test_messages_create__returns_valid_response_for_simple_message`
- `test_messages_stream__emits_proper_event_sequence`
- `test_count_tokens__longer_content_returns_more_tokens`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_BASE_URL` | `http://localhost:8000` | SGLang server URL |
| `ANTHROPIC_API_KEY` | `dummy-key` | API key (must match server's `--api-key`) |
| `TEST_MODEL_NAME` | `default` | Model name for requests |
