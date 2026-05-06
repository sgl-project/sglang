import json
from dataclasses import dataclass
from typing import List, Optional

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    popen_launch_server,
)


@dataclass
class ToolCallTestParams:
    test_basic: bool = True
    test_auto: bool = True
    test_streaming: bool = True
    test_required: bool = True
    test_none: bool = True
    test_specific: bool = True
    test_strict: bool = True
    test_multiturn: bool = True
    test_thinking: bool = False  # model-specific, e.g. DeepSeek
    test_reasoning_usage: bool = False  # verify usage.reasoning_tokens > 0
    test_parallel: bool = True
    test_streaming_parallel: bool = True


@dataclass
class ToolCallTestResult:
    model: str
    passed: bool
    num_passed: int
    num_total: int
    failures: List[str]
    variant: Optional[str] = None


# ---- tool definitions ----

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Compute the sum of two integers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First integer"},
                "b": {"type": "integer", "description": "Second integer"},
            },
            "required": ["a", "b"],
        },
    },
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}

ADD_TOOL_STRICT = {
    "type": "function",
    "function": {**ADD_TOOL["function"], "strict": True},
}

WEATHER_TOOL_STRICT = {
    "type": "function",
    "function": {**WEATHER_TOOL["function"], "strict": True},
}


def _call(
    client,
    model,
    content,
    tools=None,
    tool_choice="required",
    temperature=0.1,
    **kwargs,
):
    """Single-turn tool call request. Defaults to ADD_TOOL_STRICT + required."""
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        tools=tools or [ADD_TOOL_STRICT],
        tool_choice=tool_choice,
        temperature=temperature,
        **kwargs,
    )


# ---- test cases ----


def _test_basic_format(client, model):
    """Format + field placement: tool_calls present, content empty, valid JSON args."""
    response = _call(client, model, "Compute 3 + 5")
    msg = response.choices[0].message
    assert msg.tool_calls and len(msg.tool_calls) > 0
    assert not msg.content, f"content should be empty, got: {msg.content}"
    tc = msg.tool_calls[0]
    assert tc.function.name == "add", f"expected 'add', got '{tc.function.name}'"
    assert isinstance(json.loads(tc.function.arguments), dict)
    assert response.choices[0].finish_reason == "tool_calls"


def _test_auto(client, model):
    """tool_choice=auto should populate tool_calls, not content (#17942)."""
    response = _call(client, model, "Compute 3 + 5", tool_choice="auto")
    msg = response.choices[0].message
    assert msg.tool_calls and len(msg.tool_calls) > 0
    assert not msg.content, f"content should be empty, got: {msg.content}"
    assert response.choices[0].finish_reason == "tool_calls"


def _test_streaming(client, model):
    """Streaming chunks should concatenate to valid JSON."""
    response = _call(client, model, "Compute 5 + 7", stream=True)
    chunks = list(response)
    assert len(chunks) > 0
    arg_fragments = []
    name = None
    for chunk in chunks:
        if chunk.choices[0].delta.tool_calls:
            tc = chunk.choices[0].delta.tool_calls[0]
            name = tc.function.name or name
            if tc.function.arguments:
                arg_fragments.append(tc.function.arguments)
    assert name == "add", f"expected 'add', got '{name}'"
    args = json.loads("".join(arg_fragments))
    assert "a" in args and "b" in args
    assert chunks[-1].choices[0].finish_reason == "tool_calls"


def _test_required(client, model):
    """tool_choice='required' must return a tool call even for unrelated queries."""
    response = _call(
        client,
        model,
        "What is the capital of France?",
        tools=[ADD_TOOL, WEATHER_TOOL],
    )
    assert response.choices[0].message.tool_calls


def _test_none(client, model):
    """tool_choice='none' must not return any tool call."""
    response = _call(client, model, "What is 1+1?", tool_choice="none")
    assert response.choices[0].message.tool_calls is None
    assert response.choices[0].finish_reason == "stop"


def _test_specific(client, model):
    """Specifying a function name should return that function."""
    response = _call(
        client,
        model,
        "What is the capital of France?",
        tools=[ADD_TOOL, WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    tc = response.choices[0].message.tool_calls
    assert tc and tc[0].function.name == "get_weather"


def _test_strict(client, model):
    """strict: true should enforce schema on arguments."""
    response = _call(client, model, "Compute 5 - 7")
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    assert "a" in args and "b" in args


def _test_multiturn(client, model):
    """Pass tool result back, model should reply based on it."""
    # turn 1: get tool call
    messages = [{"role": "user", "content": "What is 3 + 5?"}]
    r1 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[ADD_TOOL_STRICT],
        tool_choice="required",
        temperature=0.1,
    )
    tc = r1.choices[0].message.tool_calls[0]
    # turn 2: pass result back
    messages.append(r1.choices[0].message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": "8",
            "name": tc.function.name,
        }
    )
    r2 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[ADD_TOOL],
        temperature=0.1,
    )
    assert "8" in (r2.choices[0].message.content or "")


def _test_thinking(client, model):
    """After tool result with thinking enabled, output should be in content not reasoning_content."""
    thinking_body = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    # turn 1
    messages = [{"role": "user", "content": "What is 3 + 5?"}]
    r1 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[ADD_TOOL_STRICT],
        tool_choice="required",
        temperature=0.1,
        extra_body=thinking_body,
    )
    tc = r1.choices[0].message.tool_calls[0]
    # turn 2
    messages.append(r1.choices[0].message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": "8",
            "name": tc.function.name,
        }
    )
    r2 = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[ADD_TOOL],
        temperature=0.1,
        extra_body=thinking_body,
    )
    content = r2.choices[0].message.content or ""
    assert "8" in content, f"expected '8' in content, got: {content}"


def _test_reasoning_usage(client, model):
    """With thinking enabled, usage.reasoning_tokens should be reported as > 0."""
    thinking_body = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 3 + 5?"}],
        tools=[ADD_TOOL_STRICT],
        tool_choice="required",
        temperature=0.1,
        extra_body=thinking_body,
    )
    usage = response.usage
    assert usage is not None, "usage should not be None"
    assert (
        usage.reasoning_tokens and usage.reasoning_tokens > 0
    ), f"expected reasoning_tokens > 0, got {usage.reasoning_tokens}"
    if usage.completion_tokens_details:
        detail_reasoning = usage.completion_tokens_details.get("reasoning_tokens", 0)
        assert (
            detail_reasoning > 0
        ), f"expected completion_tokens_details.reasoning_tokens > 0, got {detail_reasoning}"


def _test_parallel(client, model):
    """Single request should return multiple tool calls."""
    response = _call(
        client,
        model,
        "Please call both functions: use add to compute 3+5, and use get_weather to check the weather in Tokyo.",
        tools=[ADD_TOOL_STRICT, WEATHER_TOOL_STRICT],
        tool_choice="auto",
        temperature=0,
    )
    tc = response.choices[0].message.tool_calls
    assert tc and len(tc) >= 2, f"expected >= 2 tool calls, got {len(tc) if tc else 0}"


def _test_streaming_parallel(client, model):
    """Streaming with tool_choice=auto should return multiple tool calls."""
    response = _call(
        client,
        model,
        "What is 3+5 and what is the weather in Tokyo?",
        tools=[ADD_TOOL, WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
    )
    # collect tool calls from streaming chunks
    tool_calls = {}
    for chunk in response:
        if not chunk.choices[0].delta.tool_calls:
            continue
        for tc in chunk.choices[0].delta.tool_calls:
            idx = tc.index
            if idx not in tool_calls:
                tool_calls[idx] = {"name": "", "arguments": ""}
            if tc.function.name:
                tool_calls[idx]["name"] = tc.function.name
            if tc.function.arguments:
                tool_calls[idx]["arguments"] += tc.function.arguments
    assert len(tool_calls) >= 2, f"expected >= 2 tool calls, got {len(tool_calls)}"
    for idx, tc in tool_calls.items():
        assert tc["name"], f"tool call {idx} missing function name"
        args = json.loads(tc["arguments"])
        assert isinstance(args, dict), f"tool call {idx} arguments not a dict"


_TESTS = [
    ("basic_format", _test_basic_format, "test_basic"),
    # ("auto", _test_auto, "test_auto"),
    ("streaming", _test_streaming, "test_streaming"),
    ("required", _test_required, "test_required"),
    ("none", _test_none, "test_none"),
    ("specific", _test_specific, "test_specific"),
    ("strict", _test_strict, "test_strict"),
    ("multiturn", _test_multiturn, "test_multiturn"),
    ("thinking", _test_thinking, "test_thinking"),
    # ("reasoning_usage", _test_reasoning_usage, "test_reasoning_usage"),
    ("parallel", _test_parallel, "test_parallel"),
    # ("streaming_parallel", _test_streaming_parallel, "test_streaming_parallel"),
]


# ---- runner ----


def run_tool_call_test(
    model: ModelLaunchSettings,
    params: ToolCallTestParams,
    base_url: Optional[str] = None,
) -> ToolCallTestResult:
    """Launch server, run enabled test cases, return results."""
    base_url = base_url or DEFAULT_URL_FOR_TEST

    print(f"\n{'=' * 60}")
    print(f"Running TOOL CALL test for {model.model_path}")
    if model.variant:
        print(f"  Variant: {model.variant}")
    print(f"{'=' * 60}\n")

    process = None
    try:
        process = popen_launch_server(
            model.model_path,
            base_url,
            other_args=model.extra_args,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=model.env,
        )
        client = openai.Client(api_key="sk-test", base_url=base_url + "/v1")

        passed_list = []
        failed_list = []

        for name, fn, flag in _TESTS:
            if not getattr(params, flag):
                continue
            try:
                fn(client, model.model_path)
                passed_list.append(name)
                print(f"  PASS: {name}")
            except Exception as e:
                failed_list.append(f"{name}: {e}")
                print(f"  FAIL: {name}: {e}")

        total = len(passed_list) + len(failed_list)
        print(f"\n  Result: {len(passed_list)}/{total} passed")

        return ToolCallTestResult(
            model=model.model_path,
            passed=len(failed_list) == 0,
            num_passed=len(passed_list),
            num_total=total,
            failures=failed_list,
            variant=model.variant,
        )

    except Exception as e:
        print(f"  Server launch failed: {e}")
        return ToolCallTestResult(
            model=model.model_path,
            passed=False,
            num_passed=0,
            num_total=0,
            failures=[f"Server launch failed: {e}"],
            variant=model.variant,
        )

    finally:
        if process:
            kill_process_tree(process.pid)
