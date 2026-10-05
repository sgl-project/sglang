"""Beam shell is a client-executed tool, never an inference-host executor."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

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
CANONICAL_BASH = json.loads(
    (Path(__file__).parent / "fixtures/beam_bash_tool.json").read_text()
)
# Exact declaration produced by Olympus's structured-bash training definition.
CANONICAL_BASH_DECLARATION = (
    "# Tools\n\n"
    "// Execute a bash command. The output contains the command's stdout and stderr.\n"
    "type bash = (_: {\n"
    "// The bash command to execute.\n"
    "command: string,\n"
    "}) => any;"
)


def registry(tools=None):
    return BeamToolAdapter(tools if tools is not None else [SHELL])


def test_shell_schema_matches_canonical_training_definition():
    assert registry().flat_tools == [CANONICAL_BASH]


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "shell"}],
        [{"type": "shell", "environment": {"type": "container_auto"}}],
        [{"type": "shell", "environment": {"type": "local", "skills": []}}],
        [{"type": "local_shell"}],
        [SHELL, {"type": "function", "name": "bash"}],
        [{"type": "function", "name": "bash"}, SHELL],
        [SHELL, SHELL],
    ],
)
def test_shell_rejects_unsupported_options_and_collisions(tools):
    with pytest.raises(ValueError):
        registry(tools)


def test_shell_is_not_enabled_on_other_models():
    from sglang.srt.entrypoints.openai.responses_tools.adapter import get_tool_adapter

    with pytest.raises(ValueError, match="not supported"):
        get_tool_adapter("llama")([SHELL])


@pytest.mark.parametrize(
    "action",
    [
        {"commands": []},
        {"commands": ["pwd", "ls"]},
        {"commands": ["pwd"], "timeout_ms": 1},
        {"commands": ["pwd"], "max_output_length": 10},
        {"commands": ["pwd"], "working_directory": "/tmp"},
    ],
)
def test_unrepresentable_shell_replay_is_rejected(action):
    with pytest.raises(ValueError):
        registry().input_item(
            {"type": "shell_call", "call_id": "call_x", "action": action}
        )


@pytest.mark.parametrize("mode", ["full", "stream", "background"])
@pytest.mark.parametrize(
    "outcome", [{"type": "exit", "exit_code": 7}, {"type": "timeout"}]
)
def test_beam_shell_call_and_client_result_continuation(mode, outcome):
    stream = mode == "stream"
    serving = make_serving(model_type="beam")
    serving.enable_response_store = True
    serving.reasoning_parser = serving.tool_call_parser = "beam"
    serving._reasoning_detector = BeamReasoningDetector()
    tokenizer = serving.tokenizer_manager.tokenizer
    tokenizer.response_template = None
    tokenizer.init_kwargs = {}
    tokenizer.decode.return_value = "<|start|>assistant"
    command = "printf '%s\\n' 'space preserving' | cat; echo \"$HOME\""
    raw = (
        "<|channel|>analysis<|message|>Run it.<|end|>"
        "<|start|>assistant to=refl.bash<|channel|>analysis <|constrain|>xml<|message|>"
        f'<rfl:parameter name="command">{command}</rfl:parameter><|call|>'
    )
    seen = []

    async def generate(*args):
        for index in range(1, len(raw) + 1) if stream else [len(raw)]:
            yield engine_chunk(raw[:index], index, finish=index == len(raw))

    def render(chat_request, is_multimodal):
        seen.append(chat_request)
        return MessageProcessingResult(
            prompt="<|start|>assistant",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )

    serving.tokenizer_manager.generate_request = generate

    async def run():
        result = await serving.create_responses(
            ResponsesRequest(
                model="x",
                input="Run command",
                tools=[SHELL, {"type": "function", "name": "other"}],
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
            result = events[-1]["response"]
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
        call = result["output"][-1]
        assert result["usage"]["input_tokens"] == 5
        assert result["usage"]["output_tokens"] == len(raw)
        stored = await serving.retrieve_responses(result["id"])
        assert stored.output[-1].type == "shell_call"
        assert call == done
        assert call["type"] == "shell_call" and call["action"]["commands"] == [command]
        assert call["action"]["timeout_ms"] == 600000
        assert [
            t["function"]["name"] for t in seen[0].chat_template_kwargs["builtin_tools"]
        ] == ["bash"]
        assert [
            t["function"]["name"] for t in seen[0].chat_template_kwargs["tools"]
        ] == ["other"]
        assert seen[0].tools[0].function.parameters == CANONICAL_BASH["parameters"]
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
            response = await serving.create_responses(
                ResponsesRequest(model="x", store=False, **continuation)
            )
            assert response.status == "completed"
            messages = seen[-1].messages
            replay = next(m for m in messages if getattr(m, "tool_calls", None))
            assert replay.tool_calls[0].function.name == "refl.bash"
            assert json.loads(replay.tool_calls[0].function.arguments) == {
                "command": command
            }
            assert json.loads(messages[-1].content) == output

    with patch.object(serving, "_apply_conversation_template", side_effect=render):
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


def test_invalid_bash_arguments_fail_instead_of_returning_executable_command():
    r = registry()
    response = r.response(
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": "bash",
                    "call_id": "call_x",
                    "arguments": '{"command":["pwd"]}',
                    "status": "completed",
                }
            ],
        }
    )
    assert response["status"] == "failed"
    assert response["output"][0]["action"]["commands"] == []


@pytest.mark.parametrize("status", ["incomplete", "in_progress"])
def test_partial_shell_calls_never_expose_commands(status):
    r = registry()
    call = r.output_item(
        {
            "type": "function_call",
            "name": "bash",
            "call_id": "call_x",
            "arguments": '{"command":"echo partial"}',
            "status": status,
        }
    )
    assert call["status"] == status
    assert call["action"]["commands"] == []


def test_invalid_shell_terminal_event_matches_failed_response():
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
                        "call_id": "call_x",
                        "arguments": "not json",
                        "status": "completed",
                    }
                ],
            },
        }
    )
    assert event["type"] == "response.failed"
    assert event["response"]["status"] == "failed"
    assert event["response"]["error"]["code"] == "server_error"


@pytest.mark.parametrize("stream", [False, True])
def test_concurrent_shell_and_function_requests_keep_separate_translation(stream):
    serving = make_serving(model_type="beam")
    serving.reasoning_parser = serving.tool_call_parser = "beam"
    serving._reasoning_detector = BeamReasoningDetector()
    tokenizer = serving.tokenizer_manager.tokenizer
    tokenizer.response_template = None
    tokenizer.init_kwargs = {}
    tokenizer.decode.return_value = "<|start|>assistant"
    raw = (
        "<|channel|>analysis<|message|>Run it.<|end|>"
        "<|start|>assistant to=refl.bash<|channel|>analysis <|constrain|>xml<|message|>"
        '<rfl:parameter name="command">pwd</rfl:parameter><|call|>'
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
            text = (
                raw
                if "shell" in adapted.rid
                else raw.replace("to=refl.bash", "to=bash")
            )
            yield engine_chunk(text, finish=True)

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

    with patch.object(
        serving,
        "_apply_conversation_template",
        return_value=MessageProcessingResult(
            prompt="<|start|>assistant",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        ),
    ):
        asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["auto", "required", {"type": "shell"}])
@pytest.mark.parametrize("parser", ["beam", "beam_checkpoint", "response_template"])
def test_native_shell_real_template_and_replay(stream, choice, parser):
    from utils import install_beam_chat_template

    serving = make_serving(model_type="beam")
    serving.enable_response_store = True
    serving.reasoning_parser = serving.tool_call_parser = "beam"
    serving._reasoning_detector = BeamReasoningDetector()
    prompts = install_beam_chat_template(serving, parser)
    command = "printf '%s' 'space preserving'"
    raw = (
        "<|channel|>analysis<|message|>Run it.<|end|>"
        "<|start|>assistant to=refl.bash<|channel|>analysis <|constrain|>xml<|message|>"
        f'<rfl:parameter name="command">{command}</rfl:parameter><|call|>'
    )
    engines = []

    async def generate(request, *args):
        engines.append(request)
        for index in range(1, len(raw) + 1) if stream else [len(raw)]:
            yield engine_chunk(raw[:index], index, finish=index == len(raw))

    serving.tokenizer_manager.generate_request = generate

    async def run():
        request = ResponsesRequest(
            model="x", input="Run", tools=[SHELL], tool_choice=choice, stream=stream
        )
        response = await serving.create_responses(request)
        if stream:
            events = event_payloads([event async for event in response])
            assert all(
                event["response"]["tool_choice"] == choice
                for event in events
                if "response" in event
            )
            data = events[-1]["response"]
            done = [
                e["item"] for e in events if e["type"] == "response.output_item.done"
            ]
            assert done == data["output"]
            assert not any(
                e["type"].startswith("response.function_call_arguments") for e in events
            )
        else:
            data = response.model_dump()
        assert data["tool_choice"] == choice
        shell = data["output"][-1]
        assert shell["type"] == "shell_call"
        assert shell["action"]["commands"] == [command]
        system = prompts[0].split("<|end|>", 1)[0]
        declaration = system[system.index("# Tools") :].split(
            "\n\n# Valid channels:", 1
        )[0]
        assert declaration == CANONICAL_BASH_DECLARATION
        assert "type refl.bash" not in prompts[0]
        assert prompts[0].count("type bash =") == 1
        if choice != "auto":
            from xgrammar import Grammar
            from xgrammar.testing import _is_grammar_accept_string

            grammar = Grammar.from_structural_tag(
                engines[0].sampling_params["structural_tag"]
            )
            assert _is_grammar_accept_string(grammar, raw)
            assert not _is_grammar_accept_string(
                grammar, raw.replace("to=refl.bash", "to=refl.other")
            )
        output = {
            "type": "shell_call_output",
            "call_id": shell["call_id"],
            "output": [
                {
                    "stdout": "ok",
                    "stderr": "",
                    "outcome": {"type": "exit", "exit_code": 0},
                }
            ],
        }
        for replay in (
            {"input": [shell, output]},
            {"input": [output], "previous_response_id": data["id"]},
        ):
            # History keeps its native recipient even when new calls are disabled.
            for tools, next_choice in (
                ([SHELL], "auto"),
                ([], "auto"),
                ([SHELL], "none"),
            ):
                result = await serving.create_responses(
                    ResponsesRequest(
                        model="x", tools=tools, tool_choice=next_choice, **replay
                    )
                )
                assert result.status == "completed"
                assert "assistant to=refl.bash" in prompts[-1]
                assert "<|start|>refl.bash to=assistant" in prompts[-1]
                assert "refl.refl.bash" not in prompts[-1]
                if not tools or next_choice == "none":
                    assert "# Tools" not in prompts[-1]

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_truncated_shell_history_can_be_continued(stream):
    """An empty cutoff call must not make the server reject its own history."""
    from utils import install_beam_chat_template

    serving = make_serving(model_type="beam")
    serving.enable_response_store = True
    serving.reasoning_parser = serving.tool_call_parser = "beam"
    serving._reasoning_detector = BeamReasoningDetector()
    prompts = install_beam_chat_template(serving)
    raw = (
        "<|channel|>analysis<|message|>Run.<|end|>"
        "<|start|>assistant to=refl.bash<|channel|>analysis <|constrain|>xml<|message|>"
        '<rfl:parameter name="command">pw'
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
            assert "Try again" in prompts[-1]
            assert "assistant to=refl.bash" not in prompts[-1]

    asyncio.run(run())


@pytest.mark.parametrize(
    "choice", ["auto", "none", {"type": "function", "name": "run_command"}]
)
def test_native_shell_and_client_function_use_separate_recipients(choice):
    from utils import install_beam_chat_template

    serving = make_serving(model_type="beam")
    serving.reasoning_parser = serving.tool_call_parser = "beam"
    serving._reasoning_detector = BeamReasoningDetector()
    prompts = install_beam_chat_template(serving)
    client = {
        "type": "function",
        "name": "run_command",
        "description": "Client function",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    }

    def call(name):
        return f'<|start|>assistant to={name}<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">pwd</rfl:parameter><|call|>'

    async def generate(*args):
        raw = "<|channel|>analysis<|message|>Run.<|end|>"
        if choice == "none":
            raw += "<|start|>assistant<|channel|>final<|message|>No tools.<|return|>"
        else:
            if choice == "auto":
                raw += call("refl.bash")
            raw += call("run_command")
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
    system, rest = prompts[0].split("<|end|>", 1)
    if choice == "none":
        assert [i["type"] for i in calls] == ["message"]
        assert "# Tools" not in prompts[0]
    else:
        assert (
            calls[-1]["type"] == "function_call" and calls[-1]["name"] == "run_command"
        )
        assert "Client function" in rest and "Client function" not in system
        if choice == "auto":
            assert calls[0]["type"] == "shell_call"
            assert "type bash =" in system
        else:
            assert "# Tools" not in system
            assert len(calls) == 1


@pytest.mark.parametrize("kwargs", [{"builtin_tools": ["bash"]}, {"tools": []}])
def test_native_shell_rejects_conflicting_template_tool_definitions(kwargs):
    from utils import install_beam_chat_template

    serving = make_serving(model_type="beam")
    serving.tool_call_parser = "beam"
    install_beam_chat_template(serving)
    response = asyncio.run(
        serving.create_responses(
            ResponsesRequest(
                model="x",
                input="Run",
                tools=[SHELL, {"type": "function", "name": "other"}],
                chat_template_kwargs=kwargs,
            )
        )
    )
    assert response.status_code == 400
    assert "managed by Responses built-in tools" in response.body.decode()
    serving.tokenizer_manager.generate_request.assert_not_called()
