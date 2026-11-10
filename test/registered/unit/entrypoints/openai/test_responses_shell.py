"""Beam shell is a client-executed tool, never an inference-host executor."""

import asyncio
import json
import sys
from unittest.mock import Mock

import pytest
from utils import engine_chunk, event_payloads, make_serving

from sglang.srt.entrypoints.openai.protocol import (
    MessageProcessingResult,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.responses_tools.beam import BeamToolAdapter
from sglang.srt.parser.beam import BeamReasoningDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

SHELL = {"type": "shell", "environment": {"type": "local"}}


@pytest.fixture
def shell_serving(monkeypatch):
    serving = make_serving(model_type="beam")
    serving.enable_response_store = True
    serving.reasoning_parser = serving.tool_call_parser = "beam"
    serving._reasoning_detector = BeamReasoningDetector()
    serving.template_manager.chat_template_name = None
    tokenizer = serving.tokenizer_manager.tokenizer
    tokenizer.response_template = None
    tokenizer.init_kwargs = {}
    tokenizer.decode.return_value = "<|start|>assistant"
    requests, rendered = [], []

    def render(request, tools, is_multimodal):
        requests.append(request)
        rendered.append(tools)
        return MessageProcessingResult(
            prompt="<|start|>assistant",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )

    monkeypatch.setattr(serving, "_apply_jinja_template", render)
    return serving, requests, rendered


def registry(tools=None):
    return BeamToolAdapter(tools if tools is not None else [SHELL])


def test_shell_uses_bash_command_schema():
    (tool,) = registry().flat_tools
    assert tool["name"] == "bash"
    assert tool["parameters"]["required"] == ["command"]
    assert list(tool["parameters"]["properties"]) == ["command"]
    assert tool["parameters"]["properties"]["command"]["type"] == "string"


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "shell"}],
        [{"type": "shell", "environment": {"type": "container_auto"}}],
        [{"type": "shell", "environment": {"type": "local", "skills": []}}],
        [SHELL, {"type": "function", "name": "bash"}],
    ],
)
def test_shell_rejects_unsupported_options_and_collisions(tools):
    with pytest.raises(ValueError):
        registry(tools)


def test_shell_is_not_enabled_on_other_models():
    serving = make_serving()
    serving.tokenizer_manager.generate_request = generate = Mock()

    response = asyncio.run(
        serving.create_responses(ResponsesRequest(input="Run", tools=[SHELL]))
    )

    assert response.status_code == 400
    assert json.loads(response.body)["error"]["param"] == "tools"
    generate.assert_not_called()


@pytest.mark.parametrize(
    "action",
    [
        {"commands": []},
        {"commands": [123]},
        {"commands": ["pwd", "ls"]},
        {"commands": ["pwd"], "timeout_ms": 1},
        {"commands": ["pwd"], "max_output_length": 10},
        {"commands": ["pwd"], "working_directory": "/tmp"},
    ],
)
def test_unrepresentable_shell_replay_is_rejected(action):
    with pytest.raises(ValueError):
        request = ResponsesRequest(
            input=[{"type": "shell_call", "call_id": "call_x", "action": action}]
        )
        registry().input_item(request.input[0])


def test_shell_replay_accepts_sdk_input_without_output_id_or_status():
    replay = registry().input_item(
        {"type": "shell_call", "call_id": "call_x", "action": {"commands": ["pwd"]}}
    )
    assert replay["call_id"] == "call_x"
    assert replay["id"] is None and replay["status"] == "completed"
    assert json.loads(replay["arguments"]) == {"command": "pwd"}


def test_shell_result_rejects_missing_outcome():
    with pytest.raises(ValueError):
        registry().input_item(
            {
                "type": "shell_call_output",
                "call_id": "call_x",
                "output": [{"stdout": "ok", "stderr": ""}],
            }
        )


# Replay serializes each outcome the same way in every mode.
@pytest.mark.parametrize(
    "mode,outcome,choice",
    [
        ("full", {"type": "exit", "exit_code": 7}, "required"),
        ("stream", {"type": "timeout"}, {"type": "shell"}),
        ("background", {"type": "exit", "exit_code": 7}, "auto"),
    ],
)
def test_beam_shell_call_and_client_result_continuation(
    shell_serving, mode, outcome, choice
):
    stream = mode == "stream"
    serving, seen, rendered = shell_serving
    command = "printf '%s\\n' 'space preserving' | cat; echo \"$HOME\""
    raw = (
        "<|channel|>analysis<|message|>Run it.<|end|>"
        "<|start|>assistant to=bash<|channel|>analysis <|constrain|>xml<|message|>"
        f'<rfl:parameter name="command">\n{command}\n</rfl:parameter><|call|>'
    )
    engines = []

    async def generate(request, *args):
        engines.append(request)
        for index in range(1, len(raw) + 1) if stream else [len(raw)]:
            yield engine_chunk(raw[:index], index, finish=index == len(raw))

    serving.tokenizer_manager.generate_request = generate

    async def run():
        result = await serving.create_responses(
            ResponsesRequest(
                model="x",
                input="Run command",
                tools=[SHELL, {"type": "function", "name": "other"}],
                tool_choice=choice,
                stream=stream,
                background=mode == "background",
                store=True,
            )
        )
        if mode == "background":
            await serving.background_tasks[result.id]
            result = await serving.retrieve_responses(result.id)
        if stream:
            events = event_payloads([event async for event in result])
            assert [e["sequence_number"] for e in events] == list(range(len(events)))
            assert not any(
                e["type"].startswith("response.function_call_arguments") for e in events
            )
            assert all(
                e["response"]["tool_choice"] == choice
                for e in events
                if "response" in e
            )
            result = events[-1]["response"]
            assert [
                e["item"] for e in events if e["type"] == "response.output_item.done"
            ] == result["output"]
            done = next(
                e["item"]
                for e in events
                if e["type"] == "response.output_item.done"
                and e["item"]["type"] == "shell_call"
            )
        else:
            assert result.output[-1].type == "shell_call"
            result = result.model_dump()
            done = result["output"][-1]
        assert result["tool_choice"] == choice
        if choice != "auto":
            from xgrammar import Grammar
            from xgrammar.testing import _is_grammar_accept_string

            grammar = Grammar.from_structural_tag(
                engines[0].sampling_params["structural_tag"]
            )
            assert _is_grammar_accept_string(grammar, raw)
            assert not _is_grammar_accept_string(
                grammar, raw.replace("to=bash", "to=undeclared")
            )
        call = result["output"][-1]
        assert result["usage"]["input_tokens"] == 5
        assert result["usage"]["output_tokens"] == len(raw)
        stored = await serving.retrieve_responses(result["id"])
        assert stored.output[-1].type == "shell_call"
        assert call == done
        assert call["type"] == "shell_call" and call["action"]["commands"] == [command]
        assert call["action"]["timeout_ms"] == 600000
        # The template declares the trained bash for a plain bash in `tools=`.
        assert "builtin_tools" not in seen[0].chat_template_kwargs
        assert [t["function"]["name"] for t in rendered[0]] == (
            ["bash"] if isinstance(choice, dict) else ["bash", "other"]
        )
        output = {
            "type": "shell_call_output",
            "call_id": call["call_id"],
            "max_output_length": 4096,
            "output": [
                {"stdout": "partial output", "stderr": "failed", "outcome": outcome}
            ],
        }
        for continuation in (
            {"previous_response_id": result["id"], "input": [output]},
            {"input": [call, output]},
        ):
            for tools in ([SHELL], []):
                response = await serving.create_responses(
                    ResponsesRequest(
                        model="x",
                        store=False,
                        tools=tools,
                        **continuation,
                    )
                )
                assert response.status == "completed"
                messages = seen[-1].messages
                replay = next(m for m in messages if getattr(m, "tool_calls", None))
                assert replay.tool_calls[0].function.name == "bash"
                assert json.loads(replay.tool_calls[0].function.arguments) == {
                    "command": command
                }
                assert json.loads(messages[-1].content) == output
                if not tools:
                    assert not seen[-1].tools

    asyncio.run(run())


def test_namespaced_bash_remains_a_client_function():
    item = {
        "type": "function_call",
        "name": "bash",
        "namespace": "office",
        "id": "fc_x",
        "call_id": "call_x",
        "arguments": "{}",
    }
    adapter = registry()
    assert adapter.output_item(item) == item
    assert (
        adapter.event({"type": "response.output_item.added", "item": item})["item"]
        == item
    )
    assert adapter.builtin_streams == {}


@pytest.mark.parametrize("status", ["incomplete", "in_progress"])
def test_partial_shell_calls_never_expose_commands(status):
    r = registry()
    call = r.output_item(
        {
            "type": "function_call",
            "name": "bash",
            "id": "fc_x",
            "call_id": "call_x",
            "arguments": '{"command":"echo partial"}',
            "status": status,
        }
    )
    assert call["status"] == status
    assert call["action"]["commands"] == []


@pytest.mark.parametrize("arguments", ['{"command":["pwd"]}', "not json"])
def test_invalid_shell_terminal_event_matches_failed_response(arguments):
    r = registry()
    event = r.event(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "name": "bash",
                        "id": "fc_x",
                        "call_id": "call_x",
                        "arguments": arguments,
                        "status": "completed",
                    }
                ],
            },
        }
    )
    assert event["type"] == "response.failed"
    assert event["response"]["status"] == "failed"
    assert event["response"]["error"]["code"] == "server_error"
    assert event["response"]["output"][0]["action"]["commands"] == []


@pytest.mark.parametrize("stream", [False, True])
def test_concurrent_shell_and_function_requests_keep_separate_translation(
    shell_serving, stream
):
    serving, _, _ = shell_serving
    raw = (
        "<|channel|>analysis<|message|>Run it.<|end|>"
        "<|start|>assistant to=bash<|channel|>analysis <|constrain|>xml<|message|>"
        '<rfl:parameter name="command">\npwd\n</rfl:parameter><|call|>'
    )

    async def run():
        ready = asyncio.Event()
        entered = 0

        async def generate(adapted, *args):
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            yield engine_chunk(raw, finish=True)

        serving.tokenizer_manager.generate_request = generate

        async def complete(tool):
            response = await serving.create_responses(
                ResponsesRequest(
                    model="x",
                    input="Run",
                    tools=[tool],
                    stream=stream,
                    request_id=f"resp_{tool['type']}",
                )
            )
            if stream:
                events = event_payloads([event async for event in response])
                return events[-1]["response"], events
            return response.model_dump(), []

        shell, function = await asyncio.wait_for(
            asyncio.gather(
                complete(SHELL),
                complete(
                    {
                        "type": "function",
                        "name": "bash",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                        },
                    }
                ),
            ),
            timeout=5,
        )
        assert shell[0]["output"][-1]["type"] == "shell_call"
        assert function[0]["output"][-1]["type"] == "function_call"
        if stream:
            assert not any(
                e["type"].startswith("response.function_call_arguments")
                for e in shell[1]
            )
            assert any(
                e["type"].startswith("response.function_call_arguments")
                for e in function[1]
            )

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_truncated_shell_history_can_be_continued(shell_serving, stream):
    """An empty cutoff call must not make the server reject its own history."""
    serving, seen, _ = shell_serving
    raw = (
        "<|channel|>analysis<|message|>Run.<|end|>"
        "<|start|>assistant to=bash<|channel|>analysis <|constrain|>xml<|message|>"
        '<rfl:parameter name="command">\npw'
    )

    async def generate(*args):
        chunk = engine_chunk(raw, finish=True)
        chunk["meta_info"]["finish_reason"] = {"type": "length"}
        yield chunk

    serving.tokenizer_manager.generate_request = generate

    async def run():
        response = await serving.create_responses(
            ResponsesRequest(model="x", input="Run", tools=[SHELL], stream=stream)
        )
        if stream:
            data = event_payloads([event async for event in response])[-1]["response"]
        else:
            data = response.model_dump()
        call = data["output"][-1]
        assert call["status"] == "incomplete" and call["action"]["commands"] == []
        for replay in (
            {"previous_response_id": data["id"], "input": "Try again"},
            {"input": [*data["output"], {"role": "user", "content": "Try again"}]},
        ):
            result = await serving.create_responses(
                ResponsesRequest(model="x", tools=[SHELL], **replay)
            )
            assert result.status == "incomplete"
            assert seen[-1].messages[-1].content == "Try again"
            assert not any(getattr(m, "tool_calls", None) for m in seen[-1].messages)

    asyncio.run(run())


@pytest.mark.parametrize("choice", ["auto", "none"])
def test_native_shell_declares_bash_with_the_client_tools(shell_serving, choice):
    serving, seen, rendered = shell_serving
    client = {
        "type": "function",
        "name": "run_command",
        "description": "Client function",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    }

    def call(name):
        return f'<|start|>assistant to={name}<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">\npwd\n</rfl:parameter><|call|>'

    async def generate(*args):
        raw = "<|channel|>analysis<|message|>Run.<|end|>"
        if choice == "none":
            raw += "<|start|>assistant<|channel|>final<|message|>No tools.<|return|>"
        else:
            raw += call("bash") + call("run_command")
        yield engine_chunk(raw, finish=True)

    serving.tokenizer_manager.generate_request = generate
    response = asyncio.run(
        serving.create_responses(
            ResponsesRequest(
                model="x", input="Run", tools=[SHELL, client], tool_choice=choice
            )
        )
    )
    data = response.model_dump()
    calls = [i for i in data["output"] if i["type"] != "reasoning"]
    assert "builtin_tools" not in seen[0].chat_template_kwargs
    if choice == "none":
        assert [i["type"] for i in calls] == ["message"]
        assert rendered[0] is None
    else:
        assert [i["type"] for i in calls] == ["shell_call", "function_call"]
        assert calls[-1]["name"] == "run_command"
        assert [t["function"]["name"] for t in rendered[0]] == ["bash", "run_command"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
