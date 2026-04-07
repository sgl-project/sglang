"""Small-model local smoke tests for the Responses API.

These tests are intentionally narrow. They exist to keep a single-GPU local
development loop healthy before adding heavier semantic or compatibility suites.
"""

from __future__ import annotations

import asyncio
import json
import os

import openai
import pytest

GET_WEATHER_FUNCTION = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather in a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "A city name such as Berlin.",
            }
        },
        "required": ["location"],
    },
}

CALCULATE_FUNCTION = {
    "type": "function",
    "name": "calculate",
    "description": "Perform a mathematical calculation.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate.",
            }
        },
        "required": ["expression"],
    },
}

CALCULATE_CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
    },
}

_TEXT_SMOKE_MODEL = os.environ.get("SGLANG_TEXT_SMOKE_MODEL", "llama-1b")
_TOOL_SMOKE_MODEL = os.environ.get("SGLANG_TOOL_SMOKE_MODEL", "qwen-7b")


def _gateway_ws_url(base_url: str) -> str:
    """Convert the router base URL into a websocket endpoint URL."""
    if base_url.startswith("https://"):
        return f"wss://{base_url.removeprefix('https://')}/v1/responses"
    return f"ws://{base_url.removeprefix('http://')}/v1/responses"


def _ws_request(
    model: str,
    *,
    input,
    store: bool,
    previous_response_id: str | None = None,
    generate: bool | None = None,
    temperature: float = 0,
    max_output_tokens: int = 16,
    tools: list[dict] | None = None,
    tool_choice=None,
) -> dict:
    request = {
        "type": "response.create",
        "model": model,
        "input": input,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "store": store,
    }
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    if generate is not None:
        request["generate"] = generate
    if tools is not None:
        request["tools"] = tools
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    return request


def _ws_error_code(event: dict) -> str | None:
    error = event.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        if isinstance(code, str):
            return code
    code = event.get("code")
    return code if isinstance(code, str) else None


def _tool_output_chain_turn_input(turn_index: int) -> list[dict]:
    return [
        {
            "type": "function_call_output",
            "call_id": f"call_ws_smoke_{turn_index}",
            "output": json.dumps(
                {
                    "step": turn_index,
                    "status": "ok",
                    "summary": f"tool result {turn_index}",
                }
            ),
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Continue from this tool result and reply with hello.",
                }
            ],
        },
    ]


async def _send_ws_request_and_collect(
    websocket,
    request: dict,
    *,
    fail_on_error: bool = True,
) -> list[dict]:
    await websocket.send(json.dumps(request))

    events: list[dict] = []
    while True:
        payload = await asyncio.wait_for(websocket.recv(), timeout=90)
        event = json.loads(payload)
        events.append(event)

        if event.get("type") == "error":
            if fail_on_error:
                raise AssertionError(f"Unexpected websocket error: {event}")
            break

        if event.get("type") == "response.completed":
            break

    return events


async def _collect_ws_events(ws_url: str, model: str) -> list[dict]:
    """Send one `response.create` request and collect events until terminal state."""
    import websockets

    async with websockets.connect(
        ws_url, open_timeout=30, close_timeout=5
    ) as websocket:
        return await _send_ws_request_and_collect(
            websocket,
            _ws_request(
                model,
                input="Reply with the single word: hello",
                store=False,
            ),
        )


def _response_output_text(completed_event: dict) -> str:
    """Best-effort extraction of assistant text from a completed response event."""
    for item in completed_event.get("response", {}).get("output", []):
        if item.get("type") != "message":
            continue
        for content_part in item.get("content", []):
            text = content_part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _response_function_calls(output) -> list:
    return [item for item in output if item.type == "function_call"]


def _completed_response_function_calls(completed_event: dict) -> list[dict]:
    return [
        item
        for item in completed_event.get("response", {}).get("output", [])
        if item.get("type") == "function_call"
    ]


def _collect_http_event_types(client, model: str) -> list[str]:
    """Collect the logical event sequence from the HTTP streaming Responses path."""
    response = client.responses.create(
        model=model,
        input="Reply with the single word: hello",
        stream=True,
        temperature=0,
        max_output_tokens=16,
        store=False,
    )
    return [event.type for event in response]


@pytest.mark.e2e
@pytest.mark.model(_TEXT_SMOKE_MODEL)
@pytest.mark.gateway(extra_args=["--history-backend", "memory"])
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestResponsesLocalSmoke:
    """Minimal local Responses checks on a small cached model."""

    def test_basic_response_creation(self, setup_backend):
        """Basic non-streaming response creation should succeed locally."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model,
            input="Reply with the single word: hello",
            temperature=0,
            max_output_tokens=16,
        )

        assert resp.id is not None
        assert resp.error is None
        assert resp.status == "completed"
        assert resp.usage is not None
        assert len(resp.output_text) > 0

    def test_streaming_response(self, setup_backend):
        """Streaming should emit a normal response event flow."""
        _, model, client, gateway = setup_backend

        resp = client.responses.create(
            model=model,
            input="Count from 1 to 3.",
            stream=True,
            temperature=0,
            max_output_tokens=32,
        )

        events = list(resp)
        assert len(events) > 0

        created = [event for event in events if event.type == "response.created"]
        completed = [event for event in events if event.type == "response.completed"]

        assert created, "Expected at least one response.created event"
        assert completed, "Expected exactly one terminal completed event"
        assert len(completed) == 1

    def test_websocket_response_create(self, setup_backend):
        """WebSocket Responses should complete end-to-end on the local smoke model."""
        _, model, _, gateway = setup_backend

        events = asyncio.run(
            _collect_ws_events(_gateway_ws_url(gateway.base_url), model)
        )

        event_types = [event["type"] for event in events]
        completed = events[-1]

        assert "response.created" in event_types
        assert completed["type"] == "response.completed"
        assert completed["response"]["status"] == "completed"
        assert len(completed["response"]["output"]) > 0
        assert _response_output_text(completed).strip()

    def test_http_ws_event_type_parity(self, setup_backend):
        """HTTP SSE and WS should expose the same logical event sequence locally."""
        _, model, client, gateway = setup_backend

        http_event_types = _collect_http_event_types(client, model)
        ws_event_types = [
            event["type"]
            for event in asyncio.run(
                _collect_ws_events(_gateway_ws_url(gateway.base_url), model)
            )
        ]

        assert ws_event_types == http_event_types

    def test_websocket_store_false_continuation_is_connection_local(
        self, setup_backend
    ):
        """`store=false` should continue on the same socket and fail after reconnect."""
        _, model, _, gateway = setup_backend

        async def run() -> tuple[list[dict], list[dict], list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                first_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="First websocket turn. Reply with hello.",
                        store=False,
                    ),
                )
                response_id = first_events[-1]["response"]["id"]
                second_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Second websocket turn. Reply with hello again.",
                        previous_response_id=response_id,
                        store=False,
                    ),
                )

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                reconnect_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Reconnect websocket turn.",
                        previous_response_id=response_id,
                        store=False,
                    ),
                    fail_on_error=False,
                )

            return first_events, second_events, reconnect_events

        first_events, second_events, reconnect_events = asyncio.run(run())

        assert first_events[-1]["type"] == "response.completed"
        assert second_events[-1]["type"] == "response.completed"
        assert reconnect_events[-1]["type"] == "error"
        assert _ws_error_code(reconnect_events[-1]) == "previous_response_not_found"

    def test_websocket_only_latest_store_false_response_is_cached_per_connection(
        self, setup_backend
    ):
        """WS keeps only the most recent `store=false` response in connection-local cache."""
        _, model, _, gateway = setup_backend

        async def run() -> tuple[list[dict], list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                first_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="First store-false websocket turn.",
                        store=False,
                    ),
                )
                first_response_id = first_events[-1]["response"]["id"]

                second_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Second store-false websocket turn.",
                        previous_response_id=first_response_id,
                        store=False,
                    ),
                )
                second_response_id = second_events[-1]["response"]["id"]

                stale_retry_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Retry the stale first response id.",
                        previous_response_id=first_response_id,
                        store=False,
                    ),
                    fail_on_error=False,
                )

                latest_retry_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Retry the latest response id.",
                        previous_response_id=second_response_id,
                        store=False,
                    ),
                )

            return stale_retry_events, latest_retry_events

        stale_retry_events, latest_retry_events = asyncio.run(run())

        assert stale_retry_events[-1]["type"] == "error"
        assert _ws_error_code(stale_retry_events[-1]) == "previous_response_not_found"
        assert latest_retry_events[-1]["type"] == "response.completed"

    def test_websocket_invalid_previous_response_id_fails(self, setup_backend):
        """Fresh websocket connections should reject unknown previous_response_id values."""
        _, model, _, gateway = setup_backend

        async def run() -> list[dict]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                return await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="This should fail.",
                        previous_response_id="resp_missing_ws",
                        store=False,
                    ),
                    fail_on_error=False,
                )

        events = asyncio.run(run())

        assert events[-1]["type"] == "error"
        assert _ws_error_code(events[-1]) == "previous_response_not_found"

    def test_websocket_store_true_continuation_survives_reconnect_and_is_retrievable(
        self, setup_backend
    ):
        """Stored WS responses should survive reconnect and be retrievable over HTTP."""
        _, model, client, gateway = setup_backend

        async def run() -> tuple[dict, list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                first_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Persist this websocket turn.",
                        store=True,
                    ),
                )
                first_completed = first_events[-1]
                response_id = first_completed["response"]["id"]

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                reconnect_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Continue after reconnect.",
                        previous_response_id=response_id,
                        store=False,
                    ),
                )

            return first_completed, reconnect_events

        first_completed, reconnect_events = asyncio.run(run())
        stored_response_id = first_completed["response"]["id"]

        assert reconnect_events[-1]["type"] == "response.completed"

        retrieved = client.responses.retrieve(response_id=stored_response_id)
        assert retrieved.id == stored_response_id
        assert retrieved.status == "completed"
        assert retrieved.store is True

    def test_websocket_store_false_response_is_not_retrievable_over_http(
        self, setup_backend
    ):
        """WS `store=false` responses should not become retrievable over HTTP APIs."""
        _, model, client, gateway = setup_backend

        async def run() -> str:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Do not persist this websocket response.",
                        store=False,
                    ),
                )
            return events[-1]["response"]["id"]

        response_id = asyncio.run(run())

        with pytest.raises(openai.NotFoundError):
            client.responses.retrieve(response_id=response_id)

    def test_websocket_function_call_output_continuation_survives_reconnect_when_stored(
        self, setup_backend
    ):
        """Stored WS chains should accept incremental `function_call_output` items."""
        _, model, client, gateway = setup_backend

        async def run() -> tuple[str, dict]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                seed_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Seed a tool-output websocket chain.",
                        store=True,
                    ),
                )
                seed_response_id = seed_events[-1]["response"]["id"]

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                follow_up_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input=_tool_output_chain_turn_input(1),
                        previous_response_id=seed_response_id,
                        store=True,
                    ),
                )

            return seed_response_id, follow_up_events[-1]["response"]

        seed_response_id, continued_response = asyncio.run(run())
        continued_response_id = continued_response["id"]

        seed_response = client.responses.retrieve(response_id=seed_response_id)
        assert seed_response.id == seed_response_id
        assert seed_response.store is True

        retrieved = client.responses.retrieve(response_id=continued_response_id)
        assert retrieved.id == continued_response_id
        assert retrieved.status == "completed"
        assert retrieved.store is True
        assert len(retrieved.output) > 0

    def test_websocket_generate_false_returns_chainable_response_id(
        self, setup_backend
    ):
        """`generate=false` should return a response id that the same socket can chain from."""
        _, model, _, gateway = setup_backend

        async def run() -> tuple[list[dict], list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                warmup_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Warm up this websocket request state.",
                        store=False,
                        generate=False,
                    ),
                )
                warmup_response_id = warmup_events[-1]["response"]["id"]
                follow_up_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Now answer with the single word: warmed",
                        previous_response_id=warmup_response_id,
                        store=False,
                    ),
                )
            return warmup_events, follow_up_events

        warmup_events, follow_up_events = asyncio.run(run())

        assert warmup_events[-1]["type"] == "response.completed"
        assert warmup_events[-1]["response"]["output"] == []
        assert follow_up_events[-1]["type"] == "response.completed"
        assert _response_output_text(follow_up_events[-1]).strip()

    def test_websocket_generate_false_store_false_is_not_retrievable_or_reconnectable(
        self, setup_backend
    ):
        """Warmup responses with `store=false` stay socket-local and non-durable."""
        _, model, client, gateway = setup_backend

        async def run() -> tuple[str, list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                warmup_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Warm up without durable storage.",
                        store=False,
                        generate=False,
                    ),
                )
                warmup_response_id = warmup_events[-1]["response"]["id"]

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                reconnect_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Continue from non-durable warmup.",
                        previous_response_id=warmup_response_id,
                        store=False,
                    ),
                    fail_on_error=False,
                )

            return warmup_response_id, reconnect_events

        warmup_response_id, reconnect_events = asyncio.run(run())

        assert reconnect_events[-1]["type"] == "error"
        assert _ws_error_code(reconnect_events[-1]) == "previous_response_not_found"

        with pytest.raises(openai.NotFoundError):
            client.responses.retrieve(response_id=warmup_response_id)

    def test_websocket_generate_false_store_true_survives_reconnect_and_is_retrievable(
        self, setup_backend
    ):
        """Warmup responses with `store=true` should persist like normal WS responses."""
        _, model, client, gateway = setup_backend

        async def run() -> tuple[str, list[dict]]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                warmup_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Warm up with durable storage.",
                        store=True,
                        generate=False,
                    ),
                )
                warmup_response_id = warmup_events[-1]["response"]["id"]

            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                reconnect_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Continue from stored warmup.",
                        previous_response_id=warmup_response_id,
                        store=False,
                    ),
                )

            return warmup_response_id, reconnect_events

        warmup_response_id, reconnect_events = asyncio.run(run())

        retrieved = client.responses.retrieve(response_id=warmup_response_id)
        assert retrieved.id == warmup_response_id
        assert retrieved.status == "completed"
        assert retrieved.store is True
        assert retrieved.output == []

        assert reconnect_events[-1]["type"] == "response.completed"
        assert _response_output_text(reconnect_events[-1]).strip()

    def test_http_invalid_previous_response_id_is_not_found(self, setup_backend):
        """HTTP Responses should reject unknown previous_response_id values."""
        _, model, client, _ = setup_backend

        with pytest.raises(openai.NotFoundError):
            client.responses.create(
                model=model,
                input="This should fail.",
                previous_response_id="resp_missing_http",
                max_output_tokens=16,
            )


@pytest.mark.e2e
@pytest.mark.model(_TOOL_SMOKE_MODEL)
@pytest.mark.gateway(
    extra_args=["--tool-call-parser", "qwen", "--history-backend", "memory"]
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestResponsesLocalToolSmoke:
    """Function-tool Responses smoke on qwen-7b."""

    def test_chat_streaming_function_call_first_turn(self, setup_backend):
        """Local chat streaming should surface qwen tool-call deltas before Responses adapts them."""
        _, model, client, _ = setup_backend

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Calculate 42 * 17. Use the tool.",
                }
            ],
            tools=[CALCULATE_CHAT_TOOL],
            tool_choice="required",
            stream=True,
            stream_options={"include_usage": True},
            temperature=0,
            max_tokens=128,
        )

        tool_call_chunks = []
        finish_reason = None

        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta.tool_calls:
                tool_call_chunks.extend(choice.delta.tool_calls)
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

        assert tool_call_chunks, "expected chat streaming to emit tool-call chunks"
        assert finish_reason == "tool_calls"

    def test_http_responses_function_call_first_turn(self, setup_backend):
        """The router should emit a real function call via qwen-7b."""
        _, model, client, _ = setup_backend

        resp = client.responses.create(
            model=model,
            input="Calculate 15 * 23. Use the tool.",
            tools=[CALCULATE_FUNCTION],
            tool_choice="required",
            stream=False,
            temperature=0,
            max_output_tokens=128,
        )

        assert resp.error is None
        assert resp.status == "in_progress"

        function_calls = _response_function_calls(resp.output)
        assert function_calls, "expected at least one function call in the first turn"
        assert function_calls[0].name == "calculate"
        assert function_calls[0].call_id
        assert function_calls[0].arguments

        arguments = json.loads(function_calls[0].arguments)
        assert "expression" in arguments
        assert "15" in arguments["expression"]
        assert "23" in arguments["expression"]

    def test_http_streaming_function_call_first_turn(self, setup_backend):
        """Streaming Responses should expose function-call events and final output."""
        _, model, client, _ = setup_backend

        events = list(
            client.responses.create(
                model=model,
                input="Calculate 42 * 17. Use the tool.",
                tools=[CALCULATE_FUNCTION],
                tool_choice="required",
                stream=True,
                temperature=0,
                max_output_tokens=128,
            )
        )

        event_types = [event.type for event in events]
        assert "response.created" in event_types
        assert "response.function_call_arguments.delta" in event_types
        assert "response.function_call_arguments.done" in event_types

        completed_events = [
            event for event in events if event.type == "response.completed"
        ]
        assert len(completed_events) == 1

        completed = completed_events[0]
        assert completed.response.status == "in_progress"

        function_calls = _response_function_calls(completed.response.output)
        assert (
            function_calls
        ), "expected a function call in the completed response payload"
        assert function_calls[0].name == "calculate"
        assert function_calls[0].call_id
        assert function_calls[0].arguments

    def test_http_function_call_two_turn_continuation(self, setup_backend):
        """HTTP Responses should continue from function_call_output-only turns."""
        _, model, client, _ = setup_backend

        first = client.responses.create(
            model=model,
            input="Calculate 15 * 23. Use the tool.",
            tools=[CALCULATE_FUNCTION],
            tool_choice="required",
            stream=False,
            temperature=0,
            max_output_tokens=128,
        )

        function_calls = _response_function_calls(first.output)
        assert (
            function_calls
        ), "expected a completed function call in the first HTTP turn"
        assert first.status == "in_progress"
        assert function_calls[0].name == "calculate"
        assert function_calls[0].arguments

        second = client.responses.create(
            model=model,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": function_calls[0].call_id,
                    "output": json.dumps({"result": 345}),
                }
            ],
            previous_response_id=first.id,
            tools=[CALCULATE_FUNCTION],
            tool_choice="auto",
            temperature=0,
            max_output_tokens=128,
        )

        assert second.status == "completed"
        assert "345" in second.output_text

    def test_websocket_function_call_two_turn_continuation(self, setup_backend):
        """WS should support a first-turn function call and a second-turn tool result."""
        _, model, _, gateway = setup_backend

        async def run() -> tuple[dict, dict]:
            import websockets

            ws_url = _gateway_ws_url(gateway.base_url)
            async with websockets.connect(
                ws_url, open_timeout=30, close_timeout=5
            ) as websocket:
                first_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input="Calculate 15 * 23. Use the tool.",
                        store=False,
                        tools=[CALCULATE_FUNCTION],
                        tool_choice="required",
                        max_output_tokens=128,
                    ),
                )
                first_completed = first_events[-1]
                function_call = _completed_response_function_calls(first_completed)[0]

                second_events = await _send_ws_request_and_collect(
                    websocket,
                    _ws_request(
                        model,
                        input=[
                            {
                                "type": "function_call_output",
                                "call_id": function_call["call_id"],
                                "output": json.dumps({"result": 345}),
                            }
                        ],
                        previous_response_id=first_completed["response"]["id"],
                        store=False,
                        tools=[CALCULATE_FUNCTION],
                        tool_choice="auto",
                        max_output_tokens=128,
                    ),
                )

            return first_completed, second_events[-1]

        first_completed, second_completed = asyncio.run(run())

        function_calls = _completed_response_function_calls(first_completed)
        assert function_calls, "expected a completed function call in the first WS turn"
        assert first_completed["response"]["status"] == "in_progress"
        assert function_calls[0]["name"] == "calculate"
        assert function_calls[0]["arguments"]

        assert second_completed["type"] == "response.completed"
        assert second_completed["response"]["status"] == "completed"
        assert "345" in _response_output_text(second_completed)
