import asyncio
import os
import unittest
from unittest.mock import patch

from utils import make_serving

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
from sglang.srt.entrypoints.openai.tool_server import NativeToolServer
from sglang.srt.entrypoints.search.exa_client import (
    EXA_INTEGRATION_HEADER,
    EXA_INTEGRATION_NAME,
    ExaClient,
    ExaSearchConfig,
)
from sglang.srt.entrypoints.tool import HarmonyBrowserTool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class ExaClientTestCase(unittest.TestCase):
    def test_headers_include_sglang_integration_tag(self):
        client = ExaClient("test-key")

        headers = client._headers()

        self.assertEqual(headers["x-api-key"], "test-key")
        self.assertEqual(headers[EXA_INTEGRATION_HEADER], EXA_INTEGRATION_NAME)
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_default_search_payload_uses_server_side_defaults(self):
        client = ExaClient("test-key")

        payload = client._search_payload("SGLang native web search")

        self.assertEqual(payload["numResults"], 10)
        self.assertEqual(payload["type"], "auto")
        self.assertEqual(payload["contents"], {"highlights": True})

    def test_contents_payload_requests_text_and_highlights(self):
        client = ExaClient("test-key")

        payload = client._contents_payload(["https://example.com"])

        self.assertEqual(payload["urls"], ["https://example.com"])
        self.assertTrue(payload["text"])
        self.assertTrue(payload["highlights"])

    def test_config_can_be_set_from_server_environment(self):
        env = {
            "SGLANG_EXA_NUM_RESULTS": "7",
            "SGLANG_EXA_SEARCH_TYPE": "fast",
            "SGLANG_EXA_INCLUDE_HIGHLIGHTS": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            config = ExaSearchConfig.from_env()

        self.assertEqual(config.num_results, 7)
        self.assertEqual(config.search_type, "fast")
        self.assertFalse(config.include_highlights)

    def test_post_sends_integration_header_without_network(self):
        captured = {}

        class FakeResponse:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def text(self):
                return '{"ok": true}'

            async def json(self):
                return {"ok": True}

        class FakeSession:
            def __init__(self, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def post(self, url, json, headers):
                captured["url"] = url
                captured["json"] = json
                captured["headers"] = headers
                return FakeResponse()

        client = ExaClient("test-key")
        with patch(
            "sglang.srt.entrypoints.search.exa_client.aiohttp.ClientSession",
            FakeSession,
        ):
            result = asyncio.run(client._post("/search", {"query": "sglang"}))

        self.assertEqual(result, {"ok": True})
        self.assertEqual(captured["url"], "https://api.exa.ai/search")
        self.assertEqual(captured["json"], {"query": "sglang"})
        self.assertEqual(captured["headers"][EXA_INTEGRATION_HEADER], "sglang")


class ResponsesNativeWebSearchTestCase(unittest.TestCase):
    def test_harmony_web_search_requires_configured_backend(self):
        serving = make_serving()
        serving.use_harmony = True
        serving.supports_browsing = False
        request = ResponsesRequest(
            model="x",
            input="search the web",
            tools=[{"type": "web_search"}],
            store=False,
        )

        result = asyncio.run(serving.create_responses(request, raw_request=None))

        self.assertEqual(getattr(result, "status_code", None), 400)
        self.assertIn("EXA_API_KEY", result.body.decode())
        self.assertIn("https://dashboard.exa.ai/api-keys", result.body.decode())


class NativeWebSearchIntegrationTestCase(unittest.TestCase):
    def test_native_tool_server_hits_exa_client_with_server_api_key(self):
        captured_calls = []

        class FakeResponse:
            status = 200

            def __init__(self, url, payload):
                self.url = url
                self.payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def text(self):
                return "{}"

            async def json(self):
                if self.url.endswith("/search"):
                    return {
                        "requestId": "mock_req_search",
                        "results": [
                            {
                                "title": "Mock SGLang Result",
                                "url": "https://example.com/sglang",
                                "highlights": ["SGLang native web search via Exa."],
                            }
                        ],
                    }
                return {
                    "requestId": "mock_req_contents",
                    "results": [
                        {
                            "title": "Mock SGLang Result",
                            "url": self.payload["urls"][0],
                            "text": "Opened content returned through Exa contents.",
                        }
                    ],
                }

        class FakeSession:
            def __init__(self, timeout):
                self.timeout = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            def post(self, url, json, headers):
                captured_calls.append({"url": url, "json": json, "headers": headers})
                return FakeResponse(url, json)

        async def run_tool_flow():
            native_tool_server = NativeToolServer()
            self.assertTrue(native_tool_server.has_tool("browser"))
            async with native_tool_server.get_tool_session("browser") as browser_tool:
                context = FakeContext()
                search_result = await browser_tool._dispatch_browser_call(
                    context,
                    "browser.search",
                    {"query": "test native web search"},
                )
                open_result = await browser_tool._dispatch_browser_call(
                    context, "browser.open", {"cursor": 1}
                )
            return search_result, open_result

        with (
            patch.dict(os.environ, {"EXA_API_KEY": "mock-sglang-key"}, clear=False),
            patch(
                "sglang.srt.entrypoints.search.exa_client.aiohttp.ClientSession",
                FakeSession,
            ),
        ):
            search_result, open_result = asyncio.run(run_tool_flow())

        self.assertIn("Mock SGLang Result", search_result)
        self.assertIn("SGLang native web search via Exa.", search_result)
        self.assertIn("Opened content returned through Exa contents.", open_result)
        self.assertEqual(len(captured_calls), 2)

        search_call, contents_call = captured_calls
        self.assertEqual(search_call["url"], "https://api.exa.ai/search")
        self.assertEqual(search_call["headers"]["x-api-key"], "mock-sglang-key")
        self.assertEqual(search_call["headers"][EXA_INTEGRATION_HEADER], "sglang")
        self.assertEqual(search_call["json"]["numResults"], 10)
        self.assertEqual(search_call["json"]["type"], "auto")
        self.assertEqual(search_call["json"]["contents"], {"highlights": True})

        self.assertEqual(contents_call["url"], "https://api.exa.ai/contents")
        self.assertEqual(contents_call["headers"]["x-api-key"], "mock-sglang-key")
        self.assertEqual(contents_call["headers"][EXA_INTEGRATION_HEADER], "sglang")
        self.assertEqual(contents_call["json"]["urls"], ["https://example.com/sglang"])
        self.assertTrue(contents_call["json"]["text"])
        self.assertTrue(contents_call["json"]["highlights"])


class FakeExaClient:
    def __init__(self):
        self.search_queries = []
        self.content_urls = []

    async def search(self, query):
        self.search_queries.append(query)
        return {
            "requestId": "req_123",
            "results": [
                {
                    "title": "SGLang",
                    "url": "https://example.com/sglang",
                    "highlights": ["Native web search powered by Exa."],
                }
            ],
        }

    async def contents(self, urls):
        self.content_urls.append(urls)
        return {
            "results": [
                {
                    "title": "SGLang",
                    "url": urls[0],
                    "text": "SGLang native web search uses Exa for retrieval.",
                    "highlights": ["Exa for retrieval."],
                }
            ]
        }


class FakeContext:
    pass


class HarmonyBrowserToolTestCase(unittest.TestCase):
    def test_search_and_open_use_exa_client_with_request_scoped_state(self):
        client = FakeExaClient()
        tool = HarmonyBrowserTool(client=client)
        context = FakeContext()

        search_result = asyncio.run(
            tool._dispatch_browser_call(
                context, "browser.search", {"query": "SGLang web search"}
            )
        )
        open_result = asyncio.run(
            tool._dispatch_browser_call(context, "browser.open", {"cursor": 1})
        )
        find_result = asyncio.run(
            tool._dispatch_browser_call(
                context,
                "browser.find",
                {"url": "https://example.com/direct", "pattern": "retrieval"},
            )
        )

        self.assertEqual(client.search_queries, ["SGLang web search"])
        self.assertEqual(
            client.content_urls,
            [["https://example.com/sglang"], ["https://example.com/direct"]],
        )
        self.assertNotIn("req_123", search_result)
        self.assertIn("[1] SGLang", search_result)
        self.assertIn("Snippet: Native web search powered by Exa.", search_result)
        self.assertIn("Opened page: SGLang", open_result)
        self.assertIn("SGLang native web search uses Exa", open_result)
        self.assertIn("SGLang native web search uses Exa", find_result)


if __name__ == "__main__":
    unittest.main()
