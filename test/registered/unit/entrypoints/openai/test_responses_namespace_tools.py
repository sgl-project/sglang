import asyncio
import json
import unittest

from transformers.utils.chat_template_utils import _compile_jinja_template
from utils import (
    StreamFixture,
    engine_chunk,
    event_payloads,
    event_types,
    find_completed_event,
    make_serving,
)

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
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


def _muse_serving():
    serving = make_serving()
    serving.tool_call_parser = "muse"
    return serving


class NamespaceToolChatToolsTestCase(CustomTestCase):
    def test_function_tool_keeps_defer_loading(self):
        """``defer_loading`` on a Responses function tool used to be dropped while
        parsing the request, so the chat template rendered a deferred tool in full."""
        tools = _muse_serving()._response_tools_to_chat_tools(_request())
        by_name = {tool.function.name: tool.function for tool in tools}
        self.assertTrue(by_name["lookup"].defer_loading)

    def test_namespace_members_are_registered_under_qualified_names(self):
        """The tool-call parser only accepts registered names, and the model calls
        namespace members as ``<namespace>.<function>``."""
        tools = _muse_serving()._response_tools_to_chat_tools(_request())
        self.assertEqual(
            [tool.function.name for tool in tools], ["browser.search", "lookup"]
        )


class NamespaceToolTemplateTestCase(CustomTestCase):
    def _template_tools(self, request, tool_call_parser="muse"):
        serving = make_serving()
        serving.tool_call_parser = tool_call_parser
        serving.template_manager.chat_template_name = None
        serving.template_manager.jinja_template_content_format = "string"
        serving._prompt_text_round_trip_is_lossy = False
        tokenizer = serving.tokenizer_manager.tokenizer
        template = _compile_jinja_template("{{ tools | tojson }}")
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: (
            template.render(messages=messages, **kwargs)
        )
        tokenizer.encode.side_effect = lambda text, **kwargs: list(text.encode("utf-8"))
        _, _, _, processed = asyncio.run(
            serving._make_request(request, None, tokenizer)
        )
        return json.loads(bytes(processed.prompt_ids).decode("utf-8"))

    def test_namespace_reaches_the_template_nested(self):
        """Muse distinguishes inherited, explicitly eager and deferred members;
        flattening namespaces or filling absent flags changes its prompt."""
        members = [
            {"name": "inherited", "parameters": QUERY},
            {"name": "eager", "parameters": QUERY, "defer_loading": False},
            {"name": "deferred", "parameters": QUERY, "defer_loading": True},
        ]
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                supplied = (
                    [{"type": "function", "function": member} for member in members]
                    if wrapped
                    else members
                )
                tools = self._template_tools(
                    _request(tools=[{**BROWSER, "tools": supplied}, LOOKUP])
                )
                self.assertEqual(tools[0]["type"], "namespace")
                self.assertEqual(tools[0]["name"], "browser")
                self.assertEqual(tools[0]["description"], "Browse the web")
                self.assertTrue(tools[0]["defer_loading"])
                functions = [m.get("function", m) for m in tools[0]["tools"]]
                self.assertEqual(
                    [m["name"] for m in functions], ["inherited", "eager", "deferred"]
                )
                self.assertNotIn("defer_loading", functions[0])
                self.assertFalse(functions[1]["defer_loading"])
                self.assertTrue(functions[2]["defer_loading"])
                self.assertEqual(tools[1]["function"]["name"], "lookup")
                self.assertTrue(tools[1]["function"]["defer_loading"])

    def test_requests_without_namespaces_keep_the_flattened_tools(self):
        (tool,) = self._template_tools(_request(tools=[LOOKUP]))
        self.assertEqual(tool["function"]["name"], "lookup")
        self.assertTrue(tool["function"]["defer_loading"])

    def test_non_muse_keeps_existing_function_and_custom_tool_schemas(self):
        """Muse-only metadata must not change another model's rendered tools."""
        custom = {
            "type": "custom",
            "name": "execute",
            "description": "Execute a command.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": 'start: "pwd"',
            },
        }
        function = {k: v for k, v in LOOKUP.items() if k != "defer_loading"}
        for parser in (None, "qwen25", "llama3"):
            with self.subTest(parser=parser):
                expected = self._template_tools(
                    _request(tools=[function, custom]), parser
                )
                self.assertEqual(
                    [tool["function"]["name"] for tool in expected],
                    ["lookup", "execute"],
                )
                self.assertEqual(
                    expected[1]["function"]["parameters"]["required"], ["input"]
                )
                for namespaces in ([], [BROWSER]):
                    with self.subTest(namespaces=bool(namespaces)):
                        actual = self._template_tools(
                            _request(
                                tools=[
                                    *namespaces,
                                    LOOKUP,
                                    {**custom, "defer_loading": True},
                                ]
                            ),
                            parser,
                        )
                        self.assertEqual(actual, expected)

    def test_namespace_does_not_hide_custom_tool_schema(self):
        """Adding a namespace must not remove a custom tool from the prompt."""
        for tool_format in (
            {"type": "text"},
            {"type": "grammar", "syntax": "lark", "definition": 'start: "pwd"'},
        ):
            with self.subTest(tool_format=tool_format):
                custom = {
                    "type": "custom",
                    "name": "execute",
                    "description": "Execute a command.",
                    "format": tool_format,
                    "defer_loading": True,
                }
                tools = self._template_tools(_request(tools=[BROWSER, custom]))
                self.assertEqual(len(tools), 2)
                function = tools[1]["function"]
                self.assertEqual(function["name"], "execute")
                self.assertEqual(function["parameters"]["required"], ["input"])
                self.assertEqual(
                    function["parameters"]["properties"]["input"]["type"], "string"
                )
                self.assertTrue(function["defer_loading"])
                self.assertIn("Execute a command.", function["description"])
                if tool_format["type"] == "grammar":
                    self.assertIn('start: "pwd"', function["description"])


class NamespaceToolCallOutputTestCase(CustomTestCase):
    RAW = (
        '[{"name": "browser.search", "parameters": {"query": "news"}},'
        ' {"name": "lookup", "parameters": {"query": "C123"}}]'
    )
    QWEN_RAW = (
        '<tool_call>\n{"name": "browser.search", "arguments": {"query": "news"}}\n</tool_call>'
        '\n<tool_call>\n{"name": "lookup", "arguments": {"query": "C123"}}\n</tool_call>'
    )

    def _serving(self, parser=None):
        serving = make_serving()
        serving.reasoning_parser = None
        serving.tool_call_parser = parser
        return serving

    def _assert_unsplit(self, items):
        search, lookup = items
        self.assertEqual(search["type"], "function_call")
        self.assertEqual(search["name"], "browser.search")
        self.assertNotIn("namespace", search)
        self.assertEqual(json.loads(search["arguments"]), {"query": "news"})
        self.assertEqual(lookup["name"], "lookup")
        self.assertNotIn("namespace", lookup)

    def test_non_muse_full_response_keeps_dotted_function_names(self):
        """A namespace must not rename another model's ordinary function calls."""
        for parser in (None, "qwen25"):
            with self.subTest(parser=parser):
                items = self._serving(parser)._make_response_output_items(
                    _request(
                        tools=[BROWSER, {**LOOKUP, "name": "browser.search"}, LOOKUP],
                        tool_choice="required",
                    ),
                    self.RAW if parser is None else self.QWEN_RAW,
                    tokenizer=None,
                    require_reasoning=False,
                )
                self._assert_unsplit(
                    [item.model_dump(exclude_none=True) for item in items]
                )

    def test_non_muse_stream_keeps_dotted_function_names(self):
        request = _request(
            tools=[BROWSER, {**LOOKUP, "name": "browser.search"}, LOOKUP],
            tool_choice="required",
            stream=True,
        )
        for parser in (None, "qwen25"):
            with self.subTest(parser=parser):
                raw = (
                    '[{"name": "browser.search", "parameters": {"query": "news"}}]'
                    if parser is None
                    else self.QWEN_RAW.split("\n<tool_call>")[0]
                )
                events = StreamFixture(self._serving(parser), request).run(
                    [engine_chunk(raw[:30]), engine_chunk(raw, 2, finish=True)]
                )
                pairs = list(zip(event_types(events), event_payloads(events)))
                items = [
                    p["item"] for t, p in pairs if t.startswith("response.output_item.")
                ]
                items += find_completed_event(events)["response"]["output"]
                self.assertEqual(len(items), 3)
                for item in items:
                    self.assertEqual(item["name"], "browser.search")
                    self.assertNotIn("namespace", item)

    def test_muse_calls_accept_flat_and_wrapped_namespace_members(self):
        """Both member shapes accepted by Muse's template must parse and split.

        Template contract: meta-models/Muse-Spark-Open-816B-A42B-NVFP4,
        revision 4239e4db19376904f2e82c167e78f70abb5455fe.
        """
        raw = (
            " to=browser.search<|message|><atem:function_calls>"
            '<atem:invoke name="browser.search">'
            '<atem:parameter name="query">news</atem:parameter>'
            "</atem:invoke></atem:function_calls><|eot|>"
        )
        function = {"name": "search", "parameters": QUERY, "defer_loading": True}
        for member in (function, {"type": "function", "function": function}):
            with self.subTest(member=member):
                request = _request(tools=[{**BROWSER, "tools": [member]}])
                serving = self._serving("muse")
                tools = serving._response_tools_to_chat_tools(request)
                self.assertEqual(
                    [tool.function.name for tool in tools], ["browser.search"]
                )
                self.assertTrue(tools[0].function.defer_loading)
                items = serving._make_response_output_items(
                    request, raw, tokenizer=None, require_reasoning=False
                )
                self.assertEqual(len(items), 1)
                full = items[0].model_dump(exclude_none=True)
                events = StreamFixture(serving, request).run(
                    [engine_chunk(raw[:70]), engine_chunk(raw, 2, finish=True)]
                )
                pairs = list(zip(event_types(events), event_payloads(events)))
                streamed = [
                    p["item"] for t, p in pairs if t.startswith("response.output_item.")
                ]
                streamed += find_completed_event(events)["response"]["output"]
                self.assertEqual(len(streamed), 3)
                for item in (full, *streamed):
                    self.assertEqual(item["type"], "function_call")
                    self.assertEqual(
                        (item["namespace"], item["name"]), ("browser", "search")
                    )
                    if item["status"] == "completed":
                        self.assertEqual(
                            json.loads(item["arguments"]), {"query": "news"}
                        )


class NamespaceToolCallReplayTestCase(CustomTestCase):
    def test_replayed_call_is_rejoined_only_for_muse(self):
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
        for parser in ("muse", None, "qwen25"):
            with self.subTest(parser=parser):
                serving = make_serving()
                serving.tool_call_parser = parser
                messages = serving._construct_input_messages(request)
                self.assertEqual(
                    [
                        call["function"]["name"]
                        for message in messages
                        for call in message.get("tool_calls", [])
                    ],
                    ["browser.search" if parser == "muse" else "search", "lookup"],
                )


if __name__ == "__main__":
    unittest.main()
