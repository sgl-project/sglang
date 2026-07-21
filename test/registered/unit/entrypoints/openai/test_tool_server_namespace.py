"""Regression tests for MCP tool-server namespace routing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.openai.tool_server import MCPToolServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _server_response(namespace: str, instructions: str, tool_name: str):
    initialize_response = SimpleNamespace(
        serverInfo=SimpleNamespace(name=namespace),
        instructions=instructions,
    )
    list_tools_response = SimpleNamespace(
        tools=[
            SimpleNamespace(
                name=tool_name,
                description=f"{tool_name} description",
                inputSchema={"type": "object", "properties": {}},
                annotations=None,
            )
        ]
    )
    return initialize_response, list_tools_response


class TestMCPToolServerNamespaceRouting(CustomTestCase):
    def test_initializes_empty_namespace_registries(self):
        server = MCPToolServer()

        self.assertEqual(server.harmony_tool_descriptions, {})
        self.assertEqual(server.urls, {})

    def test_duplicate_namespace_keeps_description_and_url_from_same_server(self):
        responses = {
            "http://first:8001/sse": _server_response(
                "browser", "first server", "search"
            ),
            "http://second:8002/sse": _server_response(
                "browser", "second server", "open"
            ),
        }

        async def fake_list_server_and_tools(url):
            return responses[url]

        def fake_namespace_config(*, name, description, tools):
            return SimpleNamespace(name=name, description=description, tools=tools)

        def fake_tool_description(*, name, description, parameters):
            return SimpleNamespace(
                name=name, description=description, parameters=parameters
            )

        server = MCPToolServer()
        with (
            patch(
                "sglang.srt.entrypoints.openai.tool_server.list_server_and_tools",
                side_effect=fake_list_server_and_tools,
            ),
            patch(
                "sglang.srt.entrypoints.openai.tool_server.ToolNamespaceConfig",
                side_effect=fake_namespace_config,
            ),
            patch(
                "sglang.srt.entrypoints.openai.tool_server.ToolDescription.new",
                side_effect=fake_tool_description,
            ),
            self.assertLogs(
                "sglang.srt.entrypoints.openai.tool_server", level="WARNING"
            ) as captured_logs,
        ):
            asyncio.run(server.add_tool_server("first:8001,second:8002"))

        self.assertEqual(server.urls, {"browser": "http://first:8001/sse"})
        namespace = server.get_tool_description("browser")
        self.assertEqual(namespace.description, "first server")
        self.assertEqual([tool.name for tool in namespace.tools], ["search"])
        self.assertTrue(
            any(
                "Ignoring duplicate tool server" in line
                for line in captured_logs.output
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
