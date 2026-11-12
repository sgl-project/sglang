import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.entrypoints.openai.tool_server import MCPToolServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMCPToolServer(CustomTestCase):
    def test_optional_tool_description(self):
        for description in (None, "", "Search the documentation."):
            with self.subTest(description=description):
                schema = {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                }
                tools = [
                    SimpleNamespace(
                        name="search",
                        description=description,
                        inputSchema=schema,
                        annotations=None,
                    ),
                    SimpleNamespace(
                        name="ping",
                        description="Check connectivity.",
                        inputSchema={"type": "object"},
                        annotations=None,
                    ),
                ]
                discovery = AsyncMock(
                    return_value=(
                        SimpleNamespace(
                            serverInfo=SimpleNamespace(name="example"),
                            instructions=None,
                        ),
                        SimpleNamespace(tools=tools),
                    )
                )
                server = MCPToolServer()
                with patch(
                    "sglang.srt.entrypoints.openai.tool_server.list_server_and_tools",
                    discovery,
                ):
                    asyncio.run(server.add_tool_server("localhost:8000"))

                registered = server.get_tool_description("example").tools
                self.assertEqual(len(registered), 2)
                self.assertEqual(registered[0].name, "search")
                self.assertEqual(
                    registered[0].description,
                    "" if description is None else description,
                )
                self.assertEqual(registered[0].parameters, schema)
                self.assertEqual(registered[1].name, "ping")
                self.assertEqual(registered[1].description, "Check connectivity.")


if __name__ == "__main__":
    unittest.main()
