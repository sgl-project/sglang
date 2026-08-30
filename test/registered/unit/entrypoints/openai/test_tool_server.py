"""Unit tests for sglang.srt.entrypoints.openai.tool_server."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.openai.tool_server import (
    MCPToolServer,
    post_process_tools_description,
    trim_schema,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


class TestTrimSchema(unittest.TestCase):
    def test_removes_title(self):
        result = trim_schema({"title": "X", "type": "string"})
        self.assertNotIn("title", result)
        self.assertEqual(result["type"], "string")

    def test_keeps_non_none_default(self):
        result = trim_schema({"default": "abc", "type": "string"})
        self.assertEqual(result["default"], "abc")

    def test_drops_none_default(self):
        result = trim_schema({"default": None, "type": "string"})
        self.assertNotIn("default", result)

    def test_collapses_anyof_into_type_list(self):
        result = trim_schema({"anyOf": [{"type": "string"}, {"type": "integer"}]})
        self.assertNotIn("anyOf", result)
        self.assertEqual(set(result["type"]), {"string", "integer"})

    def test_anyof_filters_out_null_type(self):
        result = trim_schema({"anyOf": [{"type": "string"}, {"type": "null"}]})
        self.assertEqual(result["type"], ["string"])
        self.assertNotIn("anyOf", result)

    def test_recurses_into_properties(self):
        result = trim_schema(
            {
                "type": "object",
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "age": {"default": None, "type": "integer"},
                },
            }
        )
        self.assertNotIn("title", result["properties"]["name"])
        self.assertNotIn("default", result["properties"]["age"])


def _fake_tool(input_schema, **annotation_kwargs):
    return SimpleNamespace(
        name="t",
        description="d",
        inputSchema=input_schema,
        annotations=SimpleNamespace(**annotation_kwargs),
    )


class TestPostProcessToolsDescription(unittest.TestCase):
    def test_trims_each_tools_input_schema(self):
        tools = [
            _fake_tool({"title": "A", "type": "string"}),
            _fake_tool({"default": None, "type": "string"}),
        ]
        result = post_process_tools_description(SimpleNamespace(tools=tools))
        self.assertNotIn("title", result.tools[0].inputSchema)
        self.assertNotIn("default", result.tools[1].inputSchema)

    def test_excludes_tools_with_include_in_prompt_false(self):
        keep = _fake_tool({}, include_in_prompt=True)
        drop = _fake_tool({}, include_in_prompt=False)
        result = post_process_tools_description(SimpleNamespace(tools=[keep, drop]))
        self.assertEqual([t for t in result.tools], [keep])

    def test_keeps_tool_when_include_in_prompt_attribute_missing(self):
        tools = [_fake_tool({})]
        result = post_process_tools_description(SimpleNamespace(tools=tools))
        self.assertEqual(len(result.tools), 1)


class TestMCPToolServerLookup(unittest.TestCase):
    def test_lookup_round_trip_against_harmony_tool_descriptions(self):
        server = MCPToolServer()
        self.assertFalse(server.has_tool("missing"))
        self.assertIsNone(server.get_tool_description("missing"))

        sentinel = object()
        server.harmony_tool_descriptions = {"foo": sentinel}
        self.assertTrue(server.has_tool("foo"))
        self.assertFalse(server.has_tool("bar"))
        self.assertIs(server.get_tool_description("foo"), sentinel)
        self.assertIsNone(server.get_tool_description("bar"))


def _mcp_tool(name, description, input_schema, **annotation_kwargs):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema,
        annotations=SimpleNamespace(**annotation_kwargs),
    )


def _mcp_response(server_name, instructions, tools):
    initialize_response = SimpleNamespace(
        serverInfo=SimpleNamespace(name=server_name),
        instructions=instructions,
    )
    list_tools_response = SimpleNamespace(tools=tools)
    return initialize_response, list_tools_response


class TestMCPToolServerAddToolServer(unittest.TestCase):
    def test_processes_mcp_servers_into_harmony_with_dedup(self):
        # Two URLs sharing harmony namespace "browser": urls dict keeps the
        # first, harmony_tool_descriptions silently overwrites (asymmetric
        # by design), and a warning fires for the second.
        responses = {
            "http://host-a:8001/sse": _mcp_response(
                "browser",
                "first",
                [
                    _mcp_tool(
                        "search",
                        "search the web",
                        {
                            "title": "SearchInput",
                            "type": "object",
                            "properties": {
                                "q": {"title": "Q", "type": "string"},
                            },
                        },
                    ),
                    _mcp_tool(
                        "internal",
                        "hidden",
                        {"type": "object"},
                        include_in_prompt=False,
                    ),
                ],
            ),
            "http://host-b:8002/sse": _mcp_response(
                "browser",
                "second",
                [_mcp_tool("open", "open url", {"type": "object"})],
            ),
        }

        async def fake_list_server_and_tools(url):
            return responses[url]

        server = MCPToolServer()
        with (
            patch(
                "sglang.srt.entrypoints.openai.tool_server.list_server_and_tools",
                side_effect=fake_list_server_and_tools,
            ),
            self.assertLogs(
                "sglang.srt.entrypoints.openai.tool_server", level="WARNING"
            ) as captured_logs,
        ):
            asyncio.run(server.add_tool_server("host-a:8001,host-b:8002"))

        self.assertEqual(server.urls, {"browser": "http://host-a:8001/sse"})
        self.assertTrue(
            any(
                "already exists" in record and "host-b:8002" in record
                for record in captured_logs.output
            ),
            captured_logs.output,
        )

        ns = server.harmony_tool_descriptions["browser"]
        self.assertEqual(ns.description, "second")
        self.assertEqual([t.name for t in ns.tools], ["open"])

        # Second pass with only host-a so trim/filter assertions aren't
        # clobbered by the dup overwrite above.
        server2 = MCPToolServer()
        with patch(
            "sglang.srt.entrypoints.openai.tool_server.list_server_and_tools",
            side_effect=fake_list_server_and_tools,
        ):
            asyncio.run(server2.add_tool_server("host-a:8001"))

        ns_a = server2.harmony_tool_descriptions["browser"]
        self.assertEqual([t.name for t in ns_a.tools], ["search"])
        search = ns_a.tools[0]
        self.assertNotIn("title", search.parameters)
        self.assertNotIn("title", search.parameters["properties"]["q"])
        self.assertEqual(ns_a.name, "browser")
        self.assertEqual(ns_a.description, "first")


if __name__ == "__main__":
    unittest.main()
