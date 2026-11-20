# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Codex /v1/responses requests against MiniCPM5: which model
types translate tools, namespace translation, multi-turn replay, streaming SSE
events, and tool_search items.

Exercises the public ``minicpm5`` tool-call parser.
"""

import asyncio
import json
import unittest
from unittest.mock import Mock

from utils import (
    engine_chunk,
    event_payloads,
    make_serving,
)

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.responses_tools.beam import BeamToolAdapter
from sglang.srt.entrypoints.openai.responses_tools.minicpm import MiniCPMToolAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


CODEX_MULTI_AGENT_NAMESPACE = {
    "type": "namespace",
    "name": "multi_agent_v1",
    "description": "Multi-agent tools",
    "tools": [
        {
            "type": "function",
            "name": "spawn_agent",
            "description": "Spawn a subagent",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
        {
            "type": "function",
            "name": "wait_agent",
            "description": "Wait for subagents to complete",
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {"type": "array", "items": {"type": "string"}},
                    "timeout_ms": {"type": "integer"},
                },
                "required": ["targets"],
            },
        },
    ],
}


def _make_minicpm_serving():
    serving = make_serving(model_type="minicpm5_moe")
    serving.tool_call_parser = "minicpm5"
    serving.reasoning_parser = "qwen3"
    return serving


class CodexNamespaceAdapterTestCase(CustomTestCase):
    def test_adapter_selected_by_model_type(self):
        expected = {
            "minicpm5_moe": MiniCPMToolAdapter,
            "minicpm5_latent_moe_swa_msa": MiniCPMToolAdapter,
            "beam": BeamToolAdapter,
            "llama": None,
            "minicpm3": None,
        }
        for model_type, adapter_cls in expected.items():
            with self.subTest(model_type=model_type):
                self.assertIs(
                    make_serving(model_type=model_type)._tool_adapter_cls, adapter_cls
                )

    def test_create_responses_translates_tools_only_for_adapted_models(self):
        seen = {}

        async def stop(request, prev_response, tokenizer, *, tool_adapter=None):
            seen["tools"] = [tool.name for tool in request.tools]
            seen["adapter"] = tool_adapter
            raise ValueError("stop before generation")

        for model_type, tool_names, adapter_cls in (
            (
                "minicpm5_moe",
                ["multi_agent_v1__spawn_agent", "multi_agent_v1__wait_agent"],
                MiniCPMToolAdapter,
            ),
            ("llama", ["multi_agent_v1"], type(None)),
        ):
            with self.subTest(model_type=model_type):
                serving = make_serving(model_type=model_type)
                serving._make_request = stop
                request = ResponsesRequest(
                    model="x",
                    input="Ask a subagent to calculate 1+1.",
                    tools=[CODEX_MULTI_AGENT_NAMESPACE],
                    store=False,
                )
                asyncio.run(serving.create_responses(request))
                self.assertEqual(seen["tools"], tool_names)
                self.assertIsInstance(seen["adapter"], adapter_cls)

    def test_flatten_and_public_echo_preserved(self):
        request = ResponsesRequest(
            model="x",
            input="Ask a subagent to calculate 1+1.",
            tools=[CODEX_MULTI_AGENT_NAMESPACE],
            store=False,
        )
        translated, adapter = MiniCPMToolAdapter.from_request(
            request, tool_call_parser="minicpm5"
        )
        flat_names = [t.name for t in translated.tools]
        self.assertEqual(
            flat_names,
            ["multi_agent_v1__spawn_agent", "multi_agent_v1__wait_agent"],
        )
        self.assertEqual(
            translated._tool_state.names["multi_agent_v1__wait_agent"],
            {"name": "wait_agent", "namespace": "multi_agent_v1"},
        )
        # Public tools echoed in response must remain the original namespace shape
        self.assertEqual(translated._tool_state.public_tools[0].type, "namespace")
        self.assertEqual(translated._tool_state.public_tools[0].name, "multi_agent_v1")

    def test_non_stream_wait_agent_roundtrip(self):
        serving = _make_minicpm_serving()
        raw_request = ResponsesRequest(
            model="x",
            input=[
                {"role": "user", "content": "Ask a subagent to calculate 1+1."},
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_spawn",
                    "name": "spawn_agent",
                    "namespace": "multi_agent_v1",
                    "arguments": '{"message": "Please calculate 1+1 and report the result."}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_spawn",
                    "output": '{"agent_id":"agent_1"}',
                },
            ],
            tools=[CODEX_MULTI_AGENT_NAMESPACE],
            store=False,
        )
        translated, adapter = serving._tool_adapter_cls.from_request(
            raw_request, tool_call_parser=serving.tool_call_parser
        )

        # 1. Verify input replay rewrites historical function_call name to joined alias
        messages = serving._construct_input_messages(translated, tool_adapter=adapter)
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(
            assistant_msgs[0]["tool_calls"][0]["function"]["name"],
            "multi_agent_v1__spawn_agent",
        )

        # Verify the XML output restores the bare name and namespace.
        model_xml = (
            "<think>\nNeed to wait for the subagent.\n</think>\n\n"
            '<function name="multi_agent_v1__wait_agent">'
            "<param name=\"targets\">['agent_1']</param>"
            '<param name="timeout_ms">10000</param>'
            "</function>"
        )
        context = SimpleContext()
        context.append_output(engine_chunk(model_xml, finish=True))

        async def empty():
            if False:
                yield

        response = asyncio.run(
            serving.responses_full_generator(
                translated,
                {},
                empty(),
                context,
                "x",
                Mock(),
                RequestResponseMetadata(request_id=translated.request_id),
                require_reasoning=True,
                tool_adapter=adapter,
            )
        )
        dumped = response.model_dump()
        fn_calls = [it for it in dumped["output"] if it["type"] == "function_call"]
        self.assertEqual(len(fn_calls), 1)
        self.assertEqual(fn_calls[0]["name"], "wait_agent")
        self.assertEqual(fn_calls[0]["namespace"], "multi_agent_v1")
        self.assertEqual(
            json.loads(fn_calls[0]["arguments"]),
            {
                "targets": ["agent_1"],
                "timeout_ms": 10000,
            },
        )
        self.assertEqual(dumped["tools"][0]["type"], "namespace")
        self.assertEqual(dumped["tools"][0]["name"], "multi_agent_v1")

    def test_streaming_namespace_events(self):
        serving = _make_minicpm_serving()
        raw_request = ResponsesRequest(
            model="x",
            input="Ask a subagent to calculate 1+1.",
            tools=[CODEX_MULTI_AGENT_NAMESPACE],
            stream=True,
            store=False,
        )
        translated, adapter = serving._tool_adapter_cls.from_request(
            raw_request, tool_call_parser=serving.tool_call_parser
        )

        expected_msg = "Calculate 1+1 (ignore </function> literal)"
        msg_body = f"<![CDATA[{expected_msg}]]>"
        chunks = [
            engine_chunk("<think>Spawning subagent.</think>\n\n", 4),
            engine_chunk(
                '<think>Spawning subagent.</think>\n\n<function name="multi_agent_v1__spawn_agent">',
                8,
            ),
            engine_chunk(
                '<think>Spawning subagent.</think>\n\n<function name="multi_agent_v1__spawn_agent">'
                f'<param name="message">{msg_body}</param></function>',
                15,
                finish=True,
            ),
        ]

        async def gen():
            for ch in chunks:
                yield ch

        async def collect():
            out = []
            async for ev in serving.responses_stream_generator_non_harmony(
                translated,
                sampling_params={},
                result_generator=gen(),
                model_name="x",
                tokenizer=Mock(),
                request_metadata=RequestResponseMetadata(
                    request_id=translated.request_id
                ),
                require_reasoning=True,
                tool_adapter=adapter,
            ):
                out.append(ev)
            return out

        events = asyncio.run(collect())
        payloads = event_payloads(events)

        added_fn = [
            p["item"]
            for p in payloads
            if p["type"] == "response.output_item.added"
            and p["item"]["type"] == "function_call"
        ]
        self.assertEqual(len(added_fn), 1)
        self.assertEqual(added_fn[0]["name"], "spawn_agent")
        self.assertEqual(added_fn[0]["namespace"], "multi_agent_v1")

        args_done = [
            p for p in payloads if p["type"] == "response.function_call_arguments.done"
        ]
        self.assertEqual(len(args_done), 1)
        self.assertEqual(args_done[0]["name"], "spawn_agent")
        self.assertEqual(
            json.loads(args_done[0]["arguments"]), {"message": expected_msg}
        )

        done_fn = [
            p["item"]
            for p in payloads
            if p["type"] == "response.output_item.done"
            and p["item"]["type"] == "function_call"
        ]
        self.assertEqual(len(done_fn), 1)
        self.assertEqual(done_fn[0]["name"], "spawn_agent")
        self.assertEqual(done_fn[0]["namespace"], "multi_agent_v1")

        completed = [p for p in payloads if p["type"] == "response.completed"]
        self.assertEqual(len(completed), 1)
        final_fn = [
            it
            for it in completed[0]["response"]["output"]
            if it["type"] == "function_call"
        ]
        self.assertEqual(final_fn[0]["name"], "spawn_agent")
        self.assertEqual(final_fn[0]["namespace"], "multi_agent_v1")

    def test_commentary_and_tool_call_replay_merged_into_single_assistant_turn(self):
        serving = _make_minicpm_serving()
        raw_request = ResponsesRequest(
            model="x",
            input=[
                {"role": "user", "content": "Ask a subagent to calculate 1+1."},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "plan"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "\n\n"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "spawn_agent",
                    "namespace": "multi_agent_v1",
                    "arguments": '{"message": "1+1"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"agent_id": "a1"}',
                },
            ],
            tools=[CODEX_MULTI_AGENT_NAMESPACE],
            store=False,
        )
        translated, adapter = serving._tool_adapter_cls.from_request(
            raw_request, tool_call_parser="minicpm5"
        )
        messages = serving._construct_input_messages(translated, tool_adapter=adapter)
        roles = [m["role"] for m in messages]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertEqual(messages[1]["reasoning_content"], "plan")
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["name"],
            "multi_agent_v1__spawn_agent",
        )

    def test_tool_search_call_and_lazy_loaded_namespace_supported(self):
        serving = _make_minicpm_serving()
        raw_request = ResponsesRequest(
            model="x",
            input=[
                {"role": "user", "content": "Ask a subagent to calculate 1+1."},
                {
                    "type": "tool_search_call",
                    "call_id": "ts_1",
                    "arguments": {"query": "subagent"},
                },
                {
                    "type": "tool_search_output",
                    "call_id": "ts_1",
                    "status": "completed",
                    "tools": [CODEX_MULTI_AGENT_NAMESPACE],
                },
                # A later search that loads the same namespace again.
                {
                    "type": "tool_search_output",
                    "call_id": "ts_2",
                    "status": "completed",
                    "tools": [CODEX_MULTI_AGENT_NAMESPACE],
                },
            ],
            tools=[{"type": "tool_search"}],
            store=False,
        )
        translated, adapter = serving._tool_adapter_cls.from_request(
            raw_request, tool_call_parser="minicpm5"
        )
        flat_names = [t.name for t in translated.tools if t.name]
        self.assertEqual(
            flat_names, ["multi_agent_v1__spawn_agent", "multi_agent_v1__wait_agent"]
        )
        # The response echoes the tools the client declared, not the loaded ones.
        self.assertEqual(
            [t.type for t in translated._tool_state.public_tools], ["tool_search"]
        )

        messages = serving._construct_input_messages(translated, tool_adapter=adapter)
        self.assertEqual(
            messages, [{"role": "user", "content": "Ask a subagent to calculate 1+1."}]
        )


if __name__ == "__main__":
    unittest.main()
