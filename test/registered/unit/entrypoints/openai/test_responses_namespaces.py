"""Namespace request/output contracts, using existing CPU serving fixtures."""

import asyncio
import json
import sys
from unittest.mock import patch

import pytest
from utils import engine_chunk, event_payloads, make_serving

from sglang.srt.entrypoints.openai.protocol import (
    MessageProcessingResult,
    ResponsesRequest,
    ResponseTool,
)
from sglang.srt.entrypoints.openai.responses_adapters import (
    flatten_response_input,
    flatten_response_tools,
    response_tool_choice,
    response_tool_name,
)
from sglang.srt.parser.beam import BeamReasoningDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def namespace(kind="function", name="weather", namespace="office"):
    return {
        "type": "namespace",
        "name": namespace,
        "description": "Office tools",
        "tools": [
            {
                "type": kind,
                "name": name,
                "description": "Forecast",
                **(
                    {
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        }
                    }
                    if kind == "function"
                    else {}
                ),
            }
        ],
    }


def flatten(tools):
    return flatten_response_tools([ResponseTool.model_validate(t) for t in tools])


@pytest.mark.parametrize(
    "tools",
    [
        [namespace(), namespace()],
        [namespace(), {"type": "function", "name": "office__weather"}],
        [namespace(name="c", namespace="a__b"), namespace(name="b__c", namespace="a")],
        [
            {
                **namespace(),
                "tools": [{"type": "namespace", "name": "nested", "tools": []}],
            }
        ],
    ],
)
def test_ambiguous_or_unsupported_namespace_rejected(tools):
    with pytest.raises(ValueError):
        flatten(tools)


def test_names_and_choices_are_reversible_without_hashing():
    tools = [
        namespace(name="n" * 128, namespace="s" * 128),
        namespace(namespace="home"),
    ]
    flat, names = flatten(tools)
    assert flat[0]["name"] == "s" * 128 + "__" + "n" * 128
    assert response_tool_choice(
        {"type": "function", "namespace": "home", "name": "weather"}, flat, names
    ) == {"type": "function", "name": "home__weather"}
    assert flat[1]["description"] == "Namespace home: Office tools\n\nForecast"
    call = {
        "type": "function_call",
        "namespace": "s" * 128,
        "name": "n" * 128,
        "call_id": "call_x",
        "arguments": "{}",
    }
    internal = flatten_response_input(call)
    assert {**internal, **response_tool_name(internal["name"], names)} == call
    assert flatten(list(reversed(tools)))[1] == names
    with pytest.raises(ValueError, match="not declared"):
        response_tool_choice({"type": "function", "name": "weather"}, flat, names)


@pytest.mark.parametrize(
    "tools,choice",
    [
        (
            [{"type": "function", "name": "office__weather"}],
            {"type": "function", "namespace": "office", "name": "weather"},
        ),
        ([namespace()], {"type": "function", "name": "office__weather"}),
    ],
)
def test_named_choice_matches_public_namespace(tools, choice):
    flat, names = flatten(tools)
    with pytest.raises(ValueError, match="not declared"):
        response_tool_choice(choice, flat, names)
    response = asyncio.run(
        make_serving().create_responses(
            ResponsesRequest(input="hi", tools=tools, tool_choice=choice)
        )
    )
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["param"] == "tool_choice"


@pytest.mark.parametrize("name", [None, "", 1])
@pytest.mark.parametrize("kind", ["function_call", "custom_tool_call"])
def test_replayed_call_requires_name(kind, name):
    call = {"type": kind, "call_id": "call_x"}
    if name is not None:
        call["name"] = name
    response = asyncio.run(
        make_serving().create_responses(ResponsesRequest(model="x", input=[call]))
    )
    assert response.status_code == 400
    assert "require a name" in json.loads(response.body)["error"]["message"]


@pytest.mark.parametrize("use_harmony", [False, True])
def test_tool_translation_keeps_request_and_private_state_separate(use_harmony):
    from sglang.srt.entrypoints.openai.responses_tools.adapter import (
        ResponseToolAdapter,
    )

    request = ResponsesRequest(input="hi", tools=[namespace("custom")])
    public = request.model_dump()
    translated, _ = ResponseToolAdapter.from_request(
        request, use_harmony=use_harmony, tool_call_parser=None
    )
    assert request.model_dump() == public
    assert translated.tools[0].name == "office__weather"
    assert translated.tools[0].type == ("function" if use_harmony else "custom")
    assert translated._tool_state.public_tools == request.tools
    assert translated._tool_state.custom_names == {"office__weather"}
    assert translated.model_dump().keys() == public.keys()
    assert "_tool_state" not in ResponsesRequest.model_json_schema()["properties"]
    assert request._tool_state.names == {}


def test_namespace_mapping_preserves_extra_tool_options():
    tool = namespace()
    tool["tools"][0]["vendor_option"] = "kept"
    flat, _ = flatten([tool])
    assert flat[0]["vendor_option"] == "kept"


@pytest.mark.parametrize("choice", ["auto", "named"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind", ["function", "custom"])
@pytest.mark.parametrize("nsname", ["office", "n" * 128])
def test_namespace_beam_generation_and_continuation(stream, kind, nsname, choice):
    serving = make_serving()
    serving.enable_response_store = True
    serving.reasoning_parser = serving.tool_call_parser = "beam"
    serving._reasoning_detector = BeamReasoningDetector()
    tokenizer = serving.tokenizer_manager.tokenizer
    tokenizer.response_template = None
    tokenizer.init_kwargs = {}
    tokenizer.decode.return_value = "<|start|>assistant"
    captured = []
    field, value = ("city", "Paris") if kind == "function" else ("input", "hello")
    raw = (
        "<|channel|>analysis<|message|>Check.<|end|>"
        f"<|start|>assistant to={nsname}__weather<|channel|>analysis <|constrain|>xml<|message|>"
        f'<rfl:parameter name="{field}">{value}</rfl:parameter><|call|>'
    )

    async def generate(adapted, raw_request):
        for index in range(1, len(raw) + 1) if stream else [len(raw)]:
            yield engine_chunk(raw[:index], index, finish=index == len(raw))

    def render(chat_request, is_multimodal):
        captured.append(chat_request)
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
        tool_choice = (
            {"type": kind, "name": "weather", "namespace": nsname}
            if choice == "named"
            else "auto"
        )
        request = ResponsesRequest(
            model="x",
            stream=stream,
            store=True,
            input="Check weather",
            tools=[namespace(kind, namespace=nsname)],
            tool_choice=tool_choice,
        )
        response = await serving.create_responses(request)
        if stream:
            events = event_payloads([e async for e in response])
            assert all(
                e["response"]["tool_choice"] == tool_choice
                for e in events
                if "response" in e
            )
            response = next(
                e["response"] for e in events if e["type"] == "response.completed"
            )
            calls = [
                e["item"]
                for e in events
                if e["type"]
                in ("response.output_item.added", "response.output_item.done")
                and e["item"]["type"] in ("function_call", "custom_tool_call")
            ]
            assert all(
                c["name"] == "weather" and c["namespace"] == nsname for c in calls
            )
        else:
            assert response.output[-1].name == "weather"
            assert response.output[-1].namespace == nsname
            response = response.model_dump()
        call = response["output"][-1]
        assert response["tool_choice"] == tool_choice
        assert call["name"] == "weather" and call["namespace"] == nsname
        assert call["type"] == (
            "function_call" if kind == "function" else "custom_tool_call"
        )
        assert response["tools"][0]["type"] == "namespace"
        output = {
            "type": f"{call['type']}_output",
            "call_id": call["call_id"],
            "output": "Sunny",
        }
        # Stored response objects expose public names; replay works without tools.
        stored = await serving.retrieve_responses(response["id"])
        assert stored.tool_choice == tool_choice
        assert stored.output[-1].namespace == nsname
        assert stored.output[-1].name == "weather"
        for continuation in (
            {"previous_response_id": response["id"], "input": [output]},
            {"input": [call, output]},
        ):
            result = await serving.create_responses(
                ResponsesRequest(model="x", store=False, **continuation)
            )
            assert result.status == "completed"
            messages = captured[-1].messages
            replay = next(m for m in messages if getattr(m, "tool_calls", None))
            assert replay.tool_calls[0].function.name == f"{nsname}__weather"
            assert messages[-1].content == "Sunny"

    with patch.object(serving, "_apply_conversation_template", side_effect=render):
        asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "kind,cutoff,described",
    [
        ("function", False, True),
        ("custom", False, True),
        ("custom", False, False),  # a custom tool's description is optional
        ("custom", True, True),  # generation stops inside the payload
    ],
)
def test_harmony_namespace_handler(stream, kind, cutoff, described):
    from openai_harmony import Conversation, Message, Role

    from sglang.srt.entrypoints.harmony_utils import get_encoding

    serving = make_serving()
    serving.use_harmony = True
    serving.default_chat_template_kwargs = {}
    value = 'quote " newline\n 😀'
    args = json.dumps({"input": value} if kind == "custom" else {"city": value})
    tokens = get_encoding().render_conversation(
        Conversation.from_messages(
            [
                Message.from_role_and_content(Role.ASSISTANT, args)
                .with_channel("commentary")
                .with_recipient("functions.office__weather")
            ]
        )
    )

    prefix = get_encoding().render_conversation_for_completion(
        Conversation.from_messages([]), Role.ASSISTANT
    )
    tokens = tokens[len(prefix) :]
    if cutoff:
        tokens = next(
            tokens[:end]
            for end in range(len(tokens))
            if get_encoding().decode(tokens[:end]).endswith('{"input": "quote')
        )
        value = "quote"
    tool = namespace(kind)
    if not described:
        del tool["description"], tool["tools"][0]["description"]

    async def generate(*args):
        for index in range(1, len(tokens) + 1) if stream else [len(tokens)]:
            chunk = engine_chunk("", index, finish=index == len(tokens))
            if cutoff and index == len(tokens):
                chunk["meta_info"]["finish_reason"] = {"type": "length"}
            chunk["output_ids"] = tokens[:index]
            yield chunk

    serving.tokenizer_manager.generate_request = generate

    async def run():
        response = await serving.create_responses(
            ResponsesRequest(
                model="x",
                input="Call tool",
                stream=stream,
                store=False,
                tools=[tool],
            )
        )
        if stream:
            events = event_payloads([e async for e in response])
            response = events[-1]["response"]
            if kind == "custom":
                assert (
                    "".join(
                        e["delta"]
                        for e in events
                        if e["type"] == "response.custom_tool_call_input.delta"
                    )
                    == value
                )
                assert not any(
                    e["type"].startswith("response.function_call_arguments")
                    for e in events
                )
        else:
            assert response.output[-1].name == "weather"
            assert response.output[-1].namespace == "office"
            response = response.model_dump()
        call = response["output"][0]
        assert call["namespace"] == "office" and call["name"] == "weather"
        if kind == "custom":
            assert call["input"] == value
        else:
            assert json.loads(call["arguments"]) == {"city": value}

    asyncio.run(run())


def test_named_namespace_choice_reaches_ordinary_serving_path():
    serving = make_serving()
    serving.tool_call_parser = None
    serving.reasoning_parser = None
    seen = []

    def render(chat_request, is_multimodal):
        seen.append(chat_request)
        return MessageProcessingResult(
            prompt="prompt",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )

    async def generate(*args):
        yield engine_chunk(
            '[{"name":"office__weather","parameters":{"city":"Paris"}}]', finish=True
        )

    serving.tokenizer_manager.generate_request = generate
    with patch.object(serving, "_apply_conversation_template", side_effect=render):
        response = asyncio.run(
            serving.create_responses(
                ResponsesRequest(
                    model="x",
                    store=False,
                    input="Weather?",
                    tools=[namespace()],
                    tool_choice={
                        "type": "function",
                        "name": "weather",
                        "namespace": "office",
                    },
                )
            )
        )
    assert seen[0].tool_choice.function.name == "office__weather"
    call = response.model_dump()["output"][0]
    assert call["name"] == "weather" and call["namespace"] == "office"
    assert json.loads(call["arguments"]) == {"city": "Paris"}


def test_namespace_reuses_qwen_parser_and_unknown_name_filter():
    from sglang.srt.function_call.function_call_parser import FunctionCallParser

    serving = make_serving()
    flat, _ = flatten([namespace()])
    request = ResponsesRequest(model="x", input="hi", tools=flat)
    tools = serving._response_tools_to_chat_tools(request)
    parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
    _, calls = parser.parse_non_stream(
        '<tool_call>\n{"name":"office__weather","arguments":{"city":"Paris"}}\n</tool_call>'
    )
    assert calls[0].name == "office__weather"
    parser = FunctionCallParser(tools=tools, tool_call_parser="qwen25")
    _, calls = parser.parse_non_stream(
        '<tool_call>\n{"name":"undeclared","arguments":{}}\n</tool_call>'
    )
    assert calls == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
