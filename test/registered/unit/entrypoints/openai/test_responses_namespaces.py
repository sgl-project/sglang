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
from sglang.srt.entrypoints.openai.responses_tools.beam import flatten_response_tools
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
    response = asyncio.run(
        make_serving(model_type="beam").create_responses(
            ResponsesRequest(input="hi", tools=tools, tool_choice=choice)
        )
    )
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["param"] == "tool_choice"


@pytest.mark.parametrize("name", [None, "", 1])
def test_replayed_call_requires_name(name):
    call = {"type": "function_call", "call_id": "call_x"}
    if name is not None:
        call["name"] = name
    response = asyncio.run(
        make_serving(model_type="beam").create_responses(
            ResponsesRequest(model="x", input=[call])
        )
    )
    assert response.status_code == 400
    assert "require a name" in json.loads(response.body)["error"]["message"]


def test_namespace_mapping_preserves_extra_tool_options():
    tool = namespace()
    tool["tools"][0]["vendor_option"] = "kept"
    flat, _ = flatten([tool])
    assert flat[0]["vendor_option"] == "kept"
    assert flat[0]["description"] == "Namespace office: Office tools\n\nForecast"


@pytest.mark.parametrize(
    "stream,kind,choice",
    [
        (False, "function", "auto"),
        (False, "custom", "named"),
        (True, "function", "named"),
        (True, "custom", "auto"),
    ],
)
def test_namespace_beam_generation_and_continuation(stream, kind, choice):
    nsname = "n" * 128
    serving = make_serving(model_type="beam")
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
