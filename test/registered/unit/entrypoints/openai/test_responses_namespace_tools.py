import asyncio
import unittest

from utils import (
    StreamFixture,
    engine_chunk,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import (
    MessageProcessingResult,
    ResponsesRequest,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

QUERY = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
BROWSER = {
    "type": "namespace",
    "name": "browser",
    "description": "Browse the web",
    "defer_loading": True,
    "tools": [
        {
            "type": "function",
            "name": "search",
            "description": "Search the web",
            "parameters": QUERY,
            "defer_loading": False,
        }
    ],
}
LOOKUP = {
    "type": "function",
    "name": "lookup",
    "description": "Look up a record.",
    "parameters": QUERY,
    "defer_loading": True,
}


def _request(**kwargs) -> ResponsesRequest:
    payload = {
        "model": "x",
        "input": "search",
        "tools": [BROWSER, LOOKUP],
        "store": False,
    }
    payload.update(kwargs)
    return ResponsesRequest(**payload)


class NamespaceToolChatToolsTestCase(CustomTestCase):
    def test_function_tool_keeps_defer_loading(self):
        """``defer_loading`` on a Responses function tool used to be dropped while
        parsing the request, so the chat template rendered a deferred tool in full."""
        tools = OpenAIServingResponses._response_tools_to_chat_tools(_request())
        by_name = {tool.function.name: tool.function for tool in tools}
        self.assertTrue(by_name["lookup"].defer_loading)

    def test_namespace_members_are_registered_under_qualified_names(self):
        """The tool-call parser only accepts registered names, and the model calls
        namespace members as ``<namespace>.<function>``."""
        tools = OpenAIServingResponses._response_tools_to_chat_tools(_request())
        self.assertEqual(
            [tool.function.name for tool in tools], ["browser.search", "lookup"]
        )


class NamespaceToolTemplateTestCase(CustomTestCase):
    def _template_tools(self, request):
        serving = make_serving()
        seen = {}

        def fake_process(chat_request, is_multimodal):
            seen["template_tools"] = chat_request._template_tools
            return MessageProcessingResult(
                prompt="prompt",
                prompt_ids=[1, 2, 3],
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=[],
            )

        serving._process_messages = fake_process
        asyncio.run(
            serving._make_request(request, None, serving.tokenizer_manager.tokenizer)
        )
        return seen["template_tools"]

    def test_namespace_reaches_the_template_nested(self):
        """Namespace metadata and namespace-level deferral exist only on the
        namespace entry, so flattening it before the template loses both."""
        tools = self._template_tools(_request())
        self.assertEqual(tools[0]["type"], "namespace")
        self.assertEqual(tools[0]["name"], "browser")
        self.assertTrue(tools[0]["defer_loading"])
        self.assertEqual(tools[0]["tools"][0]["name"], "search")
        self.assertFalse(tools[0]["tools"][0]["defer_loading"])
        self.assertEqual(tools[1]["function"]["name"], "lookup")
        self.assertTrue(tools[1]["function"]["defer_loading"])

    def test_requests_without_namespaces_keep_the_flattened_tools(self):
        self.assertIsNone(self._template_tools(_request(tools=[LOOKUP])))


class NamespaceToolCallOutputTestCase(CustomTestCase):
    RAW = (
        '[{"name": "browser.search", "parameters": {"query": "news"}},'
        ' {"name": "lookup", "parameters": {"query": "C123"}}]'
    )

    def _serving(self):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = None
        return serving

    def _assert_split(self, items):
        search, lookup = items
        self.assertEqual(search["type"], "function_call")
        self.assertEqual((search["namespace"], search["name"]), ("browser", "search"))
        self.assertEqual(lookup["name"], "lookup")
        self.assertNotIn("namespace", lookup)

    def test_full_response_reports_namespace_and_function_name(self):
        """Responses clients route a namespace member's call by ``namespace`` plus
        the bare ``name``, not the flattened call name the model emits."""
        items = self._serving()._make_response_output_items(
            _request(tool_choice="required"),
            self.RAW,
            tokenizer=None,
            require_reasoning=False,
        )
        self._assert_split([item.model_dump(exclude_none=True) for item in items])

    def test_stream_reports_namespace_and_function_name(self):
        raw = '[{"name": "browser.search", "parameters": {"query": "news"}}]'
        request = _request(tool_choice="required", stream=True)
        events = StreamFixture(self._serving(), request).run(
            [engine_chunk(raw[:30]), engine_chunk(raw, 2, finish=True)]
        )
        pairs = list(zip(event_types(events), event_payloads(events)))
        items = [p["item"] for t, p in pairs if t.startswith("response.output_item.")]
        items += find_completed_event(events)["response"]["output"]
        self.assertEqual(len(items), 3)
        for item in items:
            self.assertEqual((item["namespace"], item["name"]), ("browser", "search"))


class NamespaceToolCallReplayTestCase(CustomTestCase):
    def test_replayed_call_is_rejoined_for_the_template(self):
        """The template and the tool-call parser know the call as
        ``<namespace>.<function>``; a replayed item carries the two separately."""
        request = _request(
            input=[
                {"role": "user", "content": "search"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "namespace": "browser",
                    "name": "search",
                    "arguments": '{"query": "news"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "lookup",
                    "arguments": '{"query": "C123"}',
                },
            ]
        )
        messages = [
            OpenAIServingResponses._normalize_response_message_for_chat(item)
            for item in request.input[1:]
        ]
        self.assertEqual(
            [m["tool_calls"][0]["function"]["name"] for m in messages],
            ["browser.search", "lookup"],
        )


if __name__ == "__main__":
    unittest.main()
