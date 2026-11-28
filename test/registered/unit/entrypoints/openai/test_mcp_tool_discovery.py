import asyncio
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

from sglang.srt.entrypoints.openai.tool_server import list_server_and_tools
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _discover(pages):
    initialized = object()
    session = SimpleNamespace(
        initialize=AsyncMock(return_value=initialized),
        list_tools=AsyncMock(side_effect=pages),
    )

    @asynccontextmanager
    async def streams(**kwargs):
        yield (object(), object())

    @asynccontextmanager
    async def connect(*args):
        yield session

    with (
        patch("sglang.srt.entrypoints.openai.tool_server.sse_client", streams),
        patch("sglang.srt.entrypoints.openai.tool_server.ClientSession", connect),
    ):
        result = asyncio.run(list_server_and_tools("http://localhost:8000/sse"))
    return initialized, session, result


class TestMCPToolDiscovery(CustomTestCase):
    def test_all_tool_pages_are_discovered(self):
        pages = [
            SimpleNamespace(tools=["first"], nextCursor="page-2"),
            SimpleNamespace(tools=["second"], nextCursor="page-3"),
            SimpleNamespace(tools=["third"], nextCursor=None),
        ]
        initialized, session, (info, result) = _discover(pages)
        self.assertIs(info, initialized)
        self.assertEqual(result.tools, ["first", "second", "third"])
        self.assertIsNone(result.nextCursor)
        self.assertEqual(
            session.list_tools.call_args_list,
            [call(), call(cursor="page-2"), call(cursor="page-3")],
        )

    def test_empty_cursor_is_followed(self):
        _, session, (_, result) = _discover(
            [
                SimpleNamespace(tools=[], nextCursor=""),
                SimpleNamespace(tools=["last"], nextCursor=None),
            ]
        )
        self.assertEqual(result.tools, ["last"])
        self.assertEqual(session.list_tools.call_args_list, [call(), call(cursor="")])

    def test_repeated_cursor_fails_instead_of_looping(self):
        with self.assertRaisesRegex(ValueError, "repeated pagination cursor"):
            _discover(
                [
                    SimpleNamespace(tools=["first"], nextCursor="same"),
                    SimpleNamespace(tools=["second"], nextCursor="same"),
                ]
            )

    def test_single_page_needs_no_extra_request(self):
        _, session, (_, result) = _discover(
            [SimpleNamespace(tools=["only"], nextCursor=None)]
        )
        self.assertEqual(result.tools, ["only"])
        session.list_tools.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
