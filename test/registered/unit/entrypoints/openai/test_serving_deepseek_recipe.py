import asyncio
import json
import re
import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def _tokenizer_file():
    """The V4.1 tokenizer; only outside CI does an unreachable Hub skip the test."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
    except Exception as error:  # noqa: BLE001
        if is_in_ci():
            raise
        raise unittest.SkipTest(f"the tokenizer is not available: {error}")


def _serving(
    test, tokenizer, *, is_multimodal=False, handler=None, engine=None, **server_args
):
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    test.addCleanup(reset_context)
    args = ServerArgs(
        model_path="dummy",
        reasoning_parser="deepseek-v41",
        tool_call_parser="deepseekv41",
        **server_args,
    )
    publish(args, role="tokenizer")
    manager = SimpleNamespace(
        model_path="dummy",
        served_model_name="dummy",
        server_args=args,
        tokenizer=tokenizer,
        model_config=SimpleNamespace(
            is_multimodal=is_multimodal,
            context_len=131072,
            hf_config=SimpleNamespace(
                model_type="deepseek_v41",
                architectures=["DeepseekV41ForCausalLM"],
                to_dict=lambda: {},
            ),
            get_default_sampling_params=lambda: {},
        ),
        config_value=lambda name: getattr(args, name),
        lora_registry=None,
        generate_request=engine,
        create_abort_task=lambda request: None,
    )
    template_manager = SimpleNamespace(
        chat_template_name=None,
        jinja_template_content_format=None,
        reasoning_config=None,
        force_reasoning=False,
        python_chat_template=None,
        jinja_template_may_reorder_tool_results=False,
    )
    return (handler or OpenAIServingChat)(manager, template_manager)


class TestAdapterRegistry(CustomTestCase):
    def _context(self, spec, tokenizer=None, default_kwargs=None):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        return registry.ChatEncodingAdapterContext(
            chat_encoding_spec=spec,
            tokenizer=tokenizer,
            reasoning_parser=None,
            tool_call_parser=None,
            chat_hooks=registry.ChatOutputHooks(
                thinking_requested=lambda request: False,
                tool_call_parsing_active=lambda request: False,
                history_tool_calls_count=lambda request: 0,
                tool_call_id=lambda item, index: "",
                effective_tools=lambda request: [],
                split_assistant_prefix=lambda messages, request: (messages, None),
                append_assistant_prefix=lambda prompt_ids, prefix: prompt_ids,
            ),
            hf_config=SimpleNamespace(
                model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"]
            ),
            model_path="",
            revision=None,
            default_chat_template_kwargs=default_kwargs,
        )

    def _fast_tokenizer(self):
        from tokenizers import Tokenizer

        return SimpleNamespace(backend_tokenizer=Tokenizer.from_file(_tokenizer_file()))

    def test_bad_default_effort_fails_startup(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        tokenizer = self._fast_tokenizer()
        for spec, effort in (("dsv41", "extreme"), ("dsv4", 60)):
            with self.subTest(spec=spec, effort=effort), self.assertRaises(ValueError):
                registry.build_chat_encoding_adapter(
                    self._context(
                        spec,
                        tokenizer=tokenizer,
                        default_kwargs={"reasoning_effort": effort},
                    )
                )

    def test_no_tokenizer_builds_no_adapter(self):
        """--skip-tokenizer-init servers start, with no chat prompts to encode."""
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        for spec in ("dsv4", "dsv41"):
            with self.subTest(spec=spec):
                self.assertIsNone(
                    registry.build_chat_encoding_adapter(self._context(spec))
                )

    def test_tool_dumps_render_as_the_client_sent_them(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        adapter = registry.build_chat_encoding_adapter(
            self._context("dsv41", tokenizer=self._fast_tokenizer())
        )
        # The handler's tool dumps carry None for unset fields.
        tool = {
            "type": "function",
            "function": {
                "name": "f",
                "description": "d",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        dumped = {
            "type": "function",
            "function": {**tool["function"], "strict": None},
            "defer_loading": None,
        }
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(
            adapter.encode_messages(
                messages=[],
                request=request,
                tools=[dumped],
                thinking=False,
                image_data=[],
            ),
            adapter.encode_messages(
                messages=[],
                request=request,
                tools=[tool],
                thinking=False,
                image_data=[],
            ),
        )


class TestHandlerRequestSemantics(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from transformers import PreTrainedTokenizerFast

        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_file=_tokenizer_file())

    def _token(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def _prompt(self, chat, **fields):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}], **fields
        )
        return chat._process_messages(request, False)

    def test_thinking_defaults_off(self):
        chat = _serving(self, self.tokenizer)
        default = self._prompt(chat)
        self.assertEqual(default.prompt_ids[-1], self._token("</think>"))
        self.assertFalse(default.require_reasoning)
        thinking = self._prompt(chat, chat_template_kwargs={"thinking": True})
        self.assertEqual(thinking.prompt_ids[-1], self._token("<think>"))
        self.assertTrue(thinking.require_reasoning)

    def test_request_effort_wins_over_the_server_default(self):
        thinking = {"chat_template_kwargs": {"thinking": True}}
        plain = _serving(self, self.tokenizer)
        low = self._prompt(plain, reasoning_effort="low", **thinking).prompt_ids
        self.assertNotEqual(
            low, self._prompt(plain, reasoning_effort="max", **thinking).prompt_ids
        )
        chat = _serving(
            self,
            self.tokenizer,
            default_chat_template_kwargs={"reasoning_effort": "max"},
        )
        self.assertEqual(
            self._prompt(chat, reasoning_effort="low", **thinking).prompt_ids, low
        )

    def test_trailing_assistant_follows_sglang_rules(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        chat = _serving(self, self.tokenizer)

        def prompt(messages, **fields):
            request = ChatCompletionRequest(messages=messages, **fields)
            return chat._process_messages(request, False).prompt_ids

        answer = {"role": "assistant", "content": "Hel"}
        # Without continue_final_message the trailing turn becomes a user turn.
        self.assertEqual(
            prompt([{"role": "user", "content": "hi"}, answer]),
            prompt(
                [{"role": "user", "content": "hi"}, {"role": "user", "content": "Hel"}]
            ),
        )
        # A null content is "", and the trailing turn's tool calls are discarded.
        hi, empty = (
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        )
        call = {"id": "a", "type": "function", "function": {"name": "f"}}
        calling = {"role": "assistant", "content": None, "tool_calls": [call]}
        for fields in ({}, {"continue_final_message": True}):
            self.assertEqual(
                prompt([hi, calling], **fields), prompt([hi, empty], **fields)
            )
        call["function"]["arguments"] = "{}"
        result = {"role": "tool", "tool_call_id": "a"}
        self.assertEqual(
            prompt([hi, calling, {**result, "content": None}]),
            prompt([hi, calling, {**result, "content": ""}]),
        )
        # A task renders its own tail; the continued prefix follows it.
        self.assertEqual(
            prompt(
                [{"role": "user", "content": "hi"}, answer],
                task="title",
                continue_final_message=True,
            ),
            prompt([{"role": "user", "content": "hi"}], task="title")
            + self.tokenizer.encode("Hel", add_special_tokens=False),
        )

    def test_tool_result_images_follow_prompt_order(self):
        """Out-of-order tool results render in call order; their images must follow."""
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        chat = _serving(self, self.tokenizer, is_multimodal=True)
        names = ("a", "b")
        urls = {name: f"https://example.com/{name}.png" for name in names}
        calls = [
            {
                "id": name,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
            for name in names
        ]
        results = [
            {
                "role": "tool",
                "tool_call_id": name,
                "content": [
                    {"type": "text", "text": f"result {name}"},
                    {"type": "image_url", "image_url": {"url": urls[name]}},
                ],
            }
            for name in reversed(names)
        ]
        request = ChatCompletionRequest(
            messages=[
                {"role": "user", "content": "both?"},
                {"role": "assistant", "content": "", "tool_calls": calls},
                *results,
            ],
            tools=[
                {
                    "type": "function",
                    "function": {"name": name, "parameters": {"type": "object"}},
                }
                for name in names
            ],
        )
        processed = chat._process_messages(request, True)
        prompt = self.tokenizer.decode(processed.prompt_ids)
        self.assertLess(prompt.index("result a"), prompt.index("result b"))
        self.assertEqual(
            [image.url for image in processed.image_data], [urls["a"], urls["b"]]
        )

    def test_request_reaches_the_engine_with_sglang_semantics(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        chat = _serving(self, self.tokenizer)
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            chat_template_kwargs={"thinking": True, "reasoning_effort": 30},
        )
        adapted, _ = chat._convert_to_internal_request(request)
        self.assertEqual(adapted.sampling_params["temperature"], 0.0)
        self.assertIsNone(adapted.sampling_params["max_new_tokens"])
        self.assertIn(
            "Reasoning Effort: 30 (range 1-100",
            self.tokenizer.decode(adapted.input_ids),
        )
        # A numeric string budget is the number it spells.
        for effort in ("30", "0.3"):
            request.chat_template_kwargs["reasoning_effort"] = effort
            adapted, _ = chat._convert_to_internal_request(request)
            text = self.tokenizer.decode(adapted.input_ids)
            self.assertIn("Reasoning Effort: 30 (range 1-100", text)
        titled = self._prompt(chat, task="title").prompt_ids
        self.assertEqual(titled[-1], self._token("<｜title｜>"))

    def _parse(self, chat, text, *, matched=None, **fields):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        eos = self._token("<｜end▁of▁sentence｜>")
        request = ChatCompletionRequest(
            messages=fields.pop("messages", [{"role": "user", "content": "hi"}]),
            **fields,
        )
        output_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if matched is None:
            output_ids.append(eos)
        reasoning, content, tool_calls, _ = chat._encoding_output().parse_chat_response(
            request=request,
            output_ids=output_ids,
            finish_reason={"type": "stop", "matched": matched or eos},
        )
        return reasoning, content, tool_calls

    def test_output_parsing_starts_where_the_prompt_ends(self):
        chat = _serving(self, self.tokenizer)
        thinking = {"chat_template_kwargs": {"thinking": True}}
        answer = "Let me think.</think>Hello"
        self.assertEqual(
            self._parse(chat, answer, **thinking)[:2], ("Let me think.", "Hello")
        )
        # Thinking is off by default and a task replaces the thinking header,
        # so the output is all answer.
        self.assertIsNone(self._parse(chat, answer)[0])
        self.assertEqual(
            self._parse(chat, "A title", task="title", **thinking)[:2],
            (None, "A title"),
        )
        continued = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        self.assertEqual(
            self._parse(
                chat,
                " world",
                messages=continued,
                continue_final_message=True,
                **thinking,
            )[:2],
            (None, " world"),
        )

    def test_streamed_tool_call(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        chat = _serving(self, self.tokenizer)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "weather?"}], tools=tools
        )
        output_ids = self.tokenizer.encode(
            "Sure.\n\n<｜DSML｜ calls>\n"
            '<｜DSML｜ invoke name="get_weather">\n'
            '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
            add_special_tokens=False,
        ) + [self._token("<｜end▁of▁sentence｜>")]
        finish = {"type": "stop", "matched": output_ids[-1]}
        parsers, has_tool_calls, calls = {}, {}, []
        for end in range(1, len(output_ids) + 1):
            last = end == len(output_ids)
            for chunk in chat._encoding_output().stream_chat_chunks(
                content={
                    "output_ids": output_ids[:end],
                    "meta_info": {
                        "id": "chat",
                        "completion_tokens": end,
                        "finish_reason": finish if last else None,
                    },
                },
                index=0,
                request=request,
                created=0,
                parser_dict=parsers,
                has_tool_calls=has_tool_calls,
                choice_logprobs=None,
                finish_reason_type="stop" if last else None,
                usage=None,
            ):
                delta = json.loads(chunk.removeprefix("data: "))["choices"][0]["delta"]
                calls += delta.get("tool_calls") or []
        self.assertTrue(has_tool_calls[0])
        # SGLang names the call; recipe streams its arguments.
        self.assertRegex(calls[0]["id"], r"^call_[0-9a-f]{24}$")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads("".join(call["function"]["arguments"] for call in calls)),
            {"city": "Paris"},
        )

    def test_native_endpoints_parse_through_recipe(self):
        from sglang.srt.entrypoints.openai.chat_encoding_adapters import (
            native_text_parser,
        )
        from sglang.srt.entrypoints.openai.protocol import Tool

        chat = _serving(self, self.tokenizer)
        adapter = native_text_parser(
            "deepseek-v41",
            chat_encoding_spec=chat.chat_encoding_spec,
            adapter=chat.encoding_adapter,
        )
        # V4 output never opens with <think>; the text parsers left it unsplit.
        self.assertEqual(adapter.separate_reasoning_text("a</think>b"), ("a", "b"))
        call = (
            '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n'
            '<｜DSML｜ parameter name="x" string="true">1</｜DSML｜ parameter>\n'
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
        )
        tool = Tool(type="function", function={"name": "f", "parameters": {}})
        text, (item,) = adapter.parse_function_call_text("Sure." + call, [tool])
        self.assertEqual(
            (text, item.name, item.parameters), ("Sure.", "f", '{"x": "1"}')
        )
        with self.assertRaises(ValueError):
            native_text_parser("deepseek-v41", chat_encoding_spec=None, adapter=None)


SEARCH_TOOL = {"type": "tool_search", "execution": "client"}
WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
SEARCH_DOCS_TOOL = {
    "name": "search_docs",
    "description": "Search",
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}
TICKET_TOOL = {
    "name": "create_ticket",
    "description": "Ticket",
    "input_schema": {
        "type": "object",
        "properties": {"order": {"type": "string"}},
        "required": ["order"],
    },
    "strict": True,
    "defer_loading": True,
}
CRM_NAMESPACE = {
    "type": "namespace",
    "name": "crm",
    "description": "CRM",
    "tools": [
        {
            "type": "function",
            "name": "create_ticket",
            "description": "Ticket",
            "parameters": {
                "type": "object",
                "properties": {"order": {"type": "string"}},
            },
        }
    ],
}
APPLY_PATCH_TOOL = {"type": "custom", "name": "apply_patch"}
PATCH = "*** Begin Patch\n*** Delete File: a.txt\n*** End Patch"
ASK_TICKET = {"role": "user", "content": "Open a ticket for A-1"}
ADD_TICKET = {
    "role": "system",
    "content": [
        {
            "type": "tool_addition",
            "tool": {"type": "tool_reference", "name": "create_ticket"},
        }
    ],
}
DEFINE_TICKET = {
    "role": "system",
    "content": [
        {
            "type": "tool_addition",
            "tool": {
                "type": "tool_definition",
                "definition": {
                    key: value
                    for key, value in TICKET_TOOL.items()
                    if key != "defer_loading"
                },
            },
        }
    ],
}


def _search_history(found_tools):
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "type": "tool_search_call",
            "call_id": "search",
            "execution": "client",
            "arguments": {"query": "weather"},
        },
        {
            "type": "tool_search_output",
            "call_id": "search",
            "execution": "client",
            "tools": found_tools,
        },
    ]


def _dsml_call(name, **arguments):
    parameters = "".join(
        f'<｜DSML｜ parameter name="{key}" string="true">{value}</｜DSML｜ parameter>\n'
        for key, value in arguments.items()
    )
    return (
        f'<｜DSML｜ calls>\n<｜DSML｜ invoke name="{name}">\n{parameters}'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )


def _schema_names(prompt):
    """The tools whose schemas the prompt shows, initially or as updates."""
    return re.findall(r'^\{"name": "([^"]+)"', prompt, flags=re.MULTILINE)


async def _sse_payloads(stream):
    return [
        json.loads(line.removeprefix("data: "))
        async for chunk in stream
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ]


class _Engine:
    """Answers each request with its next scripted output and keeps its prompt.
    `chunk_tokens` splits the output into chunks of that many tokens; `finish_type`
    is the reason the last one carries."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.outputs = []
        self.prompts = []
        self.requests = []
        self.chunk_tokens = None
        self.finish_type = "stop"

    def __call__(self, adapted_request, raw_request):
        self.requests.append(adapted_request)
        self.prompts.append(self.tokenizer.decode(adapted_request.input_ids))
        output_ids = self.tokenizer.encode(
            self.outputs.pop(0), add_special_tokens=False
        )
        if self.finish_type == "stop":
            eos = self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
            output_ids.append(eos)
            finish_reason = {"type": "stop", "matched": eos}
        else:
            finish_reason = {"type": self.finish_type, "length": len(output_ids)}
        step = self.chunk_tokens or len(output_ids)
        ends = [*range(step, len(output_ids), step), len(output_ids)]

        async def generate():
            # The engine's default streaming: each chunk holds the output so far.
            for end in ends:
                yield {
                    "text": "",
                    "output_ids": output_ids[:end],
                    "meta_info": {
                        "id": "rid",
                        "prompt_tokens": len(adapted_request.input_ids),
                        "completion_tokens": end,
                        "cached_tokens": 0,
                        "weight_version": "default",
                        "finish_reason": (
                            finish_reason if end == len(output_ids) else None
                        ),
                    },
                }

        return generate()


class TestNativeToolLoading(CustomTestCase):
    """Responses and Messages requests that load tools mid-conversation render once,
    through recipe's conversion of their own wire form."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from transformers import PreTrainedTokenizerFast

        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_file=_tokenizer_file())

    def _responses(self, **model):
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        engine = _Engine(self.tokenizer)
        serving = _serving(
            self,
            self.tokenizer,
            handler=OpenAIServingResponses,
            engine=engine,
            enable_response_store=True,
            **model,
        )
        return serving, engine

    def _messages(self, **model):
        from sglang.srt.entrypoints.anthropic.serving import AnthropicServing

        engine = _Engine(self.tokenizer)
        chat = _serving(self, self.tokenizer, engine=engine, **model)
        return AnthropicServing(chat), engine

    def _respond(self, serving, engine, output, **fields):
        """The response to a Responses request the engine answers with `output`;
        a streamed one as its events."""
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        engine.outputs.append(output)
        request = ResponsesRequest(model="dummy", **fields)
        response = asyncio.run(serving.create_responses(request))
        return asyncio.run(_sse_payloads(response)) if request.stream else response

    def _messages_request(self, messages, **fields):
        from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

        return AnthropicMessagesRequest(
            **{
                "model": "dummy",
                "max_tokens": 64,
                "tools": [SEARCH_DOCS_TOOL, TICKET_TOOL],
                "messages": messages,
                **fields,
            }
        )

    def test_responses_tool_search_discovery_to_call(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        serving, engine = self._responses()

        def respond(output, **fields):
            engine.outputs.append(output)
            request = ResponsesRequest(model="dummy", tools=[SEARCH_TOOL], **fields)
            return request, asyncio.run(serving.create_responses(request))

        # The search call streams as a typed item whose arguments arrive whole.
        _, stream = respond(
            _dsml_call("tool_search", query="weather"),
            input="Weather in Paris?",
            stream=True,
        )
        payloads = asyncio.run(_sse_payloads(stream))
        items = [
            payload["item"]
            for payload in payloads
            if payload["type"].startswith("response.output_item.")
        ]
        self.assertEqual(
            [(item["type"], item["status"], item["arguments"]) for item in items],
            [
                ("tool_search_call", "in_progress", {}),
                ("tool_search_call", "completed", {"query": "weather"}),
            ],
        )
        self.assertNotIn(
            "response.function_call_arguments.delta",
            [payload["type"] for payload in payloads],
        )
        searched = payloads[-1]["response"]
        self.assertEqual(searched["output"], [items[-1]])
        self.assertEqual(_schema_names(engine.prompts[-1]), ["tool_search"])

        # The returned schema renders where the search returned it and is callable.
        request, called = respond(
            _dsml_call("get_weather", city="Paris"),
            previous_response_id=searched["id"],
            input=[
                {
                    "type": "tool_search_output",
                    "call_id": items[-1]["call_id"],
                    "execution": "client",
                    "status": "completed",
                    "tools": [WEATHER_TOOL],
                }
            ],
            tool_choice={"type": "function", "name": "get_weather"},
        )
        prompt = engine.prompts[-1]
        self.assertGreater(
            prompt.index("### Updated Tool Schemas"), prompt.index("</tool_result>")
        )
        self.assertEqual(_schema_names(prompt), ["tool_search", "get_weather"])
        self.assertEqual(
            [
                tool.function.name
                for tool in serving._response_tools_to_chat_tools(request)
            ],
            _schema_names(prompt),
        )
        (call,) = called.output
        self.assertEqual(
            (call.type, call.name, json.loads(call.arguments)),
            ("function_call", "get_weather", {"city": "Paris"}),
        )

        _, answered = respond(
            "Sunny.",
            previous_response_id=called.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "sunny",
                }
            ],
        )
        self.assertIn('<｜DSML｜ invoke name="get_weather">', engine.prompts[-1])
        self.assertEqual(answered.output[-1].content[0].text, "Sunny.")

    def test_responses_tool_choice_follows_loaded_tools(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        serving, engine = self._responses()
        for tools, choice, param in (
            ([SEARCH_TOOL], {"type": "function", "name": "get_weather"}, "tool_choice"),
            ([{"type": "tool_search"}], "auto", "tools"),
        ):
            with self.subTest(param=param):
                rejected = asyncio.run(
                    serving.create_responses(
                        ResponsesRequest(
                            model="dummy", tools=tools, input="hi", tool_choice=choice
                        )
                    )
                )
                self.assertEqual(json.loads(rejected.body)["error"]["param"], param)

        # `none` hides even the tools the history loaded.
        request = ResponsesRequest(
            model="dummy",
            tools=[SEARCH_TOOL],
            tool_choice="none",
            input=_search_history([WEATHER_TOOL]),
        )
        engine.outputs.append("Sunny.")
        answered = asyncio.run(serving.create_responses(request))
        self.assertEqual(answered.output[-1].content[0].text, "Sunny.")
        self.assertIn("<tool_result>", engine.prompts[-1])
        self.assertEqual(_schema_names(engine.prompts[-1]), [])
        self.assertEqual(serving._response_tools_to_chat_tools(request), [])

    def test_responses_search_output_loads_only_renderable_tools(self):
        """A malformed search result, or one the encoding cannot render, is a client
        error before generation, not a server error."""
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        serving, engine = self._responses()
        for name, found_tools in (
            ("not a list", 42),
            ("custom", [{"type": "custom", "name": "shell"}]),
            ("custom member", [{**CRM_NAMESPACE, "tools": [APPLY_PATCH_TOOL]}]),
        ):
            with self.subTest(name):
                rejected = asyncio.run(
                    serving.create_responses(
                        ResponsesRequest(
                            model="dummy",
                            tools=[SEARCH_TOOL],
                            input=_search_history(found_tools),
                        )
                    )
                )
                self.assertEqual(rejected.status_code, 400)
                self.assertEqual(json.loads(rejected.body)["error"]["param"], "tools")
        self.assertEqual(engine.prompts, [])

    def test_responses_namespace_member_keeps_its_identity(self):
        """A namespace member is called by the encoding's flat name and reported
        by its namespace and own name, whether declared or discovered."""
        serving, engine = self._responses()
        flat = "crm::create_ticket"
        call = _dsml_call(flat, order="A-1")
        identity = {
            "type": "function_call",
            "namespace": "crm",
            "name": "create_ticket",
        }
        discovered = {
            "tools": [SEARCH_TOOL],
            "input": _search_history([CRM_NAMESPACE]),
        }
        for name, fields, schemas in (
            ("declared", {"tools": [CRM_NAMESPACE], "input": "hi"}, [flat]),
            ("discovered", discovered, ["tool_search", flat]),
        ):
            with self.subTest(name):
                response = self._respond(
                    serving,
                    engine,
                    call,
                    tool_choice={**identity, "type": "function"},
                    **fields,
                )
                self.assertEqual(_schema_names(engine.prompts[-1]), schemas)
                # The named choice constrains the output to the flat name.
                self.assertIn(
                    flat, json.dumps(engine.requests[-1].sampling_params, default=str)
                )
                (item,) = response.output
                self.assertEqual(item.model_dump(include=set(identity)), identity)
                self.assertEqual(json.loads(item.arguments), {"order": "A-1"})

                # The stored call replays under the name the model called.
                self._respond(
                    serving,
                    engine,
                    "Done.",
                    previous_response_id=response.id,
                    input=[
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": "ok",
                        }
                    ],
                    tools=fields["tools"],
                )
                self.assertIn(f'invoke name="{flat}"', engine.prompts[-1])

                events = self._respond(serving, engine, call, stream=True, **fields)
                items = [
                    event["item"]
                    for event in events
                    if event["type"].startswith("response.output_item.")
                ]
                self.assertEqual(
                    [{key: item[key] for key in identity} for item in items],
                    [identity, identity],
                )

        rejected = self._respond(
            serving,
            engine,
            call,
            tool_choice={
                "type": "function",
                "namespace": "ops",
                "name": "create_ticket",
            },
            **discovered,
        )
        self.assertEqual(json.loads(rejected.body)["error"]["param"], "tool_choice")

    def test_responses_replays_a_namespace_call_under_the_called_name(self):
        """A member discovered again gets a suffixed call name beside the first
        one. Both report the same namespace and name, so a call replayed from
        those would render as the other definition's call."""
        rediscovered = {
            **CRM_NAMESPACE,
            "tools": [
                {
                    **CRM_NAMESPACE["tools"][0],
                    "parameters": {
                        "type": "object",
                        "properties": {"ticket_id": {"type": "string"}},
                    },
                }
            ],
        }
        fields = {
            "tools": [CRM_NAMESPACE, SEARCH_TOOL],
            "input": _search_history([rediscovered]),
        }
        called = "crm::create_ticket_1"
        for stream in (False, True):
            with self.subTest(stream=stream):
                serving, engine = self._responses()
                response = self._respond(
                    serving,
                    engine,
                    _dsml_call(called, ticket_id="B-2"),
                    stream=stream,
                    **fields,
                )
                response = response[-1]["response"] if stream else response.model_dump()
                self.assertEqual(
                    _schema_names(engine.prompts[-1]),
                    ["crm::create_ticket", "tool_search", called],
                )
                (item,) = response["output"]
                self.assertEqual(
                    (item["namespace"], item["name"]), ("crm", "create_ticket")
                )

                self._respond(
                    serving,
                    engine,
                    "Done.",
                    previous_response_id=response["id"],
                    input=[
                        {
                            "type": "function_call_output",
                            "call_id": item["call_id"],
                            "output": "ok",
                        }
                    ],
                    tools=fields["tools"],
                )
                self.assertEqual(
                    re.findall(r'invoke name="([^"]+)"', engine.prompts[-1])[-2:],
                    ["tool_search", called],
                )
                self.assertIn(
                    '<｜DSML｜ parameter name="ticket_id" string="true">B-2',
                    engine.prompts[-1],
                )

    def test_responses_named_choice_takes_the_latest_namespace_definition(self):
        """Naming a member discovered again constrains the output to its latest
        definition, in whatever order recipe lists the members' call names."""
        from unittest.mock import patch

        from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe

        rediscovered = {
            **CRM_NAMESPACE,
            "tools": [
                {
                    **CRM_NAMESPACE["tools"][0],
                    "parameters": {
                        "type": "object",
                        "properties": {"ticket_id": {"type": "string"}},
                    },
                }
            ],
        }
        latest = "crm::create_ticket_1"
        namespace_tool_names = DeepSeekRecipe.namespace_tool_names
        for latest_first in (False, True):

            def ordered(recipe, namespaces, latest_first=latest_first):
                names = namespace_tool_names(recipe, namespaces)
                return dict(sorted(names.items(), reverse=latest_first))

            for stream in (False, True):
                with (
                    self.subTest(latest_first=latest_first, stream=stream),
                    patch.object(DeepSeekRecipe, "namespace_tool_names", ordered),
                ):
                    serving, engine = self._responses()
                    self._respond(
                        serving,
                        engine,
                        _dsml_call(latest, ticket_id="B-2"),
                        stream=stream,
                        tools=[CRM_NAMESPACE, SEARCH_TOOL],
                        input=_search_history([rediscovered]),
                        tool_choice={
                            "type": "function",
                            "namespace": "crm",
                            "name": "create_ticket",
                        },
                    )
                    tag = json.loads(
                        engine.requests[-1].sampling_params["structural_tag"]
                    )
                    forced = tag["format"]["elements"][1]
                    self.assertIn(f'invoke name="{latest}"', forced["begin"])
                    self.assertEqual(
                        list(forced["content"]["json_schema"]["properties"]),
                        ["ticket_id"],
                    )

    def test_responses_discovered_apply_patch_is_a_custom_tool(self):
        serving, engine = self._responses()
        fields = {
            "tools": [SEARCH_TOOL],
            "input": _search_history([APPLY_PATCH_TOOL]),
        }
        call = _dsml_call("apply_patch", input=PATCH)
        response = self._respond(serving, engine, call, **fields)
        self.assertEqual(
            _schema_names(engine.prompts[-1]), ["tool_search", "apply_patch"]
        )
        (item,) = response.output
        self.assertEqual(
            (item.type, item.name, item.input),
            ("custom_tool_call", "apply_patch", PATCH),
        )

        events = self._respond(serving, engine, call, stream=True, **fields)
        types = [event["type"] for event in events]
        self.assertNotIn("response.function_call_arguments.delta", types)
        self.assertEqual(
            "".join(
                event["delta"]
                for event in events
                if event["type"] == "response.custom_tool_call_input.delta"
            ),
            PATCH,
        )
        (done,) = (
            event["item"]
            for event in events
            if event["type"] == "response.output_item.done"
        )
        self.assertEqual(
            (done["type"], done["name"], done["input"]),
            ("custom_tool_call", "apply_patch", PATCH),
        )

    def test_images_reach_the_engine_in_prompt_order(self):
        """Images of a request rendered from its own wire form are sent with the
        prompt ids, one per placeholder and in placeholder order."""
        urls = [f"https://example.com/{index}.png" for index in range(4)]
        placeholder = "<｜deepseek_image｜>"

        def image(url):
            return {"type": "input_image", "image_url": url}

        search_call, search_output = _search_history([WEATHER_TOOL])[1:]
        responses_fields = {
            "tools": [SEARCH_TOOL],
            "input": [
                {"role": "system", "content": [image(urls[0])]},
                {
                    "role": "user",
                    "content": [
                        image(urls[1]),
                        {"type": "input_text", "text": "Weather here?"},
                        image(urls[2]),
                    ],
                },
                search_call,
                search_output,
                {"role": "user", "content": [image(urls[3])]},
            ],
        }
        serving, engine = self._responses(is_multimodal=True)
        self._respond(serving, engine, "Sunny.", **responses_fields)
        sent = engine.requests[-1]
        self.assertEqual([data.url for data in sent.image_data], urls)
        prompt = engine.prompts[-1]
        self.assertEqual(prompt.count(placeholder), len(urls))
        self.assertLess(
            prompt.index("### Updated Tool Schemas"), prompt.rindex(placeholder)
        )

        def block(url):
            return {"type": "image", "source": {"type": "url", "url": url}}

        anthropic, engine = self._messages(is_multimodal=True)
        engine.outputs.append("A ticket.")
        messages_request = self._messages_request(
            [
                {"role": "user", "content": [block(urls[0]), block(urls[1])]},
                ADD_TICKET,
                {"role": "user", "content": [block(urls[2])]},
            ]
        )
        asyncio.run(anthropic.handle_messages(messages_request, None))
        sent = engine.requests[-1]
        self.assertEqual([data.url for data in sent.image_data], urls[:3])
        self.assertEqual(engine.prompts[-1].count(placeholder), 3)

        # A text-only checkpoint refuses the images instead of dropping them.
        serving, engine = self._responses()
        rejected = self._respond(serving, engine, "Sunny.", **responses_fields)
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("image input is not supported", rejected.body.decode())
        self.assertEqual(engine.prompts, [])

    def test_auto_reports_only_calls_to_loaded_tools(self):
        """Without a grammar, a call to a deferred tool the prompt never showed is
        not reported; calls to the tools it shows are."""
        loaded, unloaded = (
            _dsml_call("search_docs", q="ticket"),
            _dsml_call("create_ticket", order="A-1"),
        )
        anthropic, engine = self._messages()

        def tool_uses(output):
            engine.outputs.append(output)
            request = self._messages_request([ASK_TICKET])
            response = json.loads(
                asyncio.run(anthropic.handle_messages(request, None)).body
            )
            self.assertEqual(_schema_names(engine.prompts[-1]), ["search_docs"])
            return response["stop_reason"], [
                block["name"]
                for block in response["content"]
                if block["type"] == "tool_use"
            ]

        self.assertEqual(tool_uses(unloaded), ("end_turn", []))
        self.assertEqual(tool_uses(loaded), ("tool_use", ["search_docs"]))
        both = unloaded.replace("</｜DSML｜ calls>", "") + loaded.partition("\n")[2]
        self.assertEqual(tool_uses(both), ("tool_use", ["search_docs"]))

        serving, engine = self._responses()
        fields = {"tools": [SEARCH_TOOL, WEATHER_TOOL], "input": "Open a ticket"}
        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self._respond(
                    serving,
                    engine,
                    both.replace("search_docs", "get_weather"),
                    stream=stream,
                    **fields,
                )
                output = (
                    response[-1]["response"]["output"]
                    if stream
                    else [item.model_dump() for item in response.output]
                )
                self.assertEqual(
                    [(item["type"], item["name"]) for item in output],
                    [("function_call", "get_weather")],
                )

    def _none_answers(self, anthropic, serving, engine, output, **formats):
        """What a natively rendered `tool_choice: none` request returns for
        `output`, per API: its text, how it finished, and the whole response."""
        answers = {}
        stream = engine.chunk_tokens is not None
        engine.outputs.append(output)
        response = asyncio.run(
            anthropic.handle_messages(
                self._messages_request(
                    [ASK_TICKET, ADD_TICKET],
                    tool_choice={"type": "none"},
                    stream=stream,
                    **formats.get("messages", {}),
                ),
                None,
            )
        )
        if stream:
            events = asyncio.run(_sse_payloads(response.body_iterator))
            blocks = [
                event["content_block"]
                for event in events
                if event["type"] == "content_block_start"
            ]
            text = "".join(
                event["delta"].get("text", "")
                for event in events
                if event["type"] == "content_block_delta"
            )
            (finish,) = (
                event["delta"]["stop_reason"]
                for event in events
                if event["type"] == "message_delta"
            )
        else:
            events = json.loads(response.body)
            blocks = events["content"]
            text = "".join(block.get("text", "") for block in blocks)
            finish = events["stop_reason"]
        self.assertEqual({block["type"] for block in blocks} - {"text"}, set())
        answers["messages"] = (text, finish, events)

        response = self._respond(
            serving,
            engine,
            output,
            stream=stream,
            tools=[SEARCH_TOOL],
            tool_choice="none",
            input=_search_history([WEATHER_TOOL]),
            **formats.get("responses", {}),
        )
        final = response[-1]["response"] if stream else response.model_dump()
        self.assertEqual(
            {item["type"] for item in final["output"]} - {"message"}, set()
        )
        text = "".join(
            part["text"] for item in final["output"] for part in item["content"]
        )
        if stream:
            self.assertEqual(
                "".join(
                    event["delta"]
                    for event in response
                    if event["type"] == "response.output_text.delta"
                ),
                text,
            )
        answers["responses"] = (text, final["status"], response if stream else final)
        return answers

    def test_none_ends_the_answer_at_a_call_the_model_writes_anyway(self):
        """Under `tool_choice: none` the answer ends where the model starts a call
        block, as any answer does: the text before it is returned, the call and
        what follows are not, and no call is reported. A JSON answer is read as
        JSON, so a call marker spelled in one of its strings stays."""
        from sglang.srt.environ import envs

        anthropic, engine = self._messages()
        serving, _ = self._responses()
        # One engine serves both handlers, so both see the same chunking.
        serving.tokenizer_manager.generate_request = engine
        call = _dsml_call("create_ticket", order="A-1")
        other = _dsml_call("get_weather", city="Paris")
        unclosed = call[: call.index("</｜DSML｜ invoke>")]
        texts = (
            (call, ""),
            (_dsml_call("tool_search", query="weather"), ""),
            ("Sure." + call, "Sure."),
            ("Grüße ☃ 你好" + unclosed, "Grüße ☃ 你好"),
            ("BEFORE" + call + "BETWEEN" + other + "AFTER", "BEFORE"),
            ("BEFORE" + call + "AFTER", "BEFORE"),
            # The marker spelled in prose is the marker: the tokenizer has one
            # token for it either way.
            ("Write <｜DSML｜ calls> to call.", "Write "),
            ("No ticket needed.", "No ticket needed."),
        )
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        spelled = '{"answer":"<｜DSML｜ calls>"}'
        json_formats = [
            {
                "messages": {
                    "output_config": {
                        "format": {"type": "json_schema", "schema": schema}
                    }
                },
                "responses": {"text": {"format": responses_format}},
            }
            for responses_format in (
                {
                    "type": "json_schema",
                    "name": "out",
                    "strict": True,
                    "schema": schema,
                },
                {"type": "json_object"},
            )
        ]

        def answers(output, **formats):
            engine.chunk_tokens = chunk_tokens
            return self._none_answers(anthropic, serving, engine, output, **formats)

        for forward_unknown in (False, True):
            for chunk_tokens in (None, 10**6, 1, 2, 7):
                with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(forward_unknown):
                    for output, text in texts:
                        with self.subTest(
                            output=output, chunk=chunk_tokens, forward=forward_unknown
                        ):
                            for api, (answer, finish, response) in answers(
                                output
                            ).items():
                                self.assertEqual((api, answer), (api, text))
                                self.assertIn(finish, ("end_turn", "completed"))
                                if "DSML" not in text:
                                    self.assertNotIn(
                                        "DSML", json.dumps(response, ensure_ascii=False)
                                    )
                    for formats in json_formats:
                        with self.subTest(
                            json=formats["responses"]["text"]["format"]["type"],
                            chunk=chunk_tokens,
                            forward=forward_unknown,
                        ):
                            for api, (answer, finish, _) in answers(
                                spelled, **formats
                            ).items():
                                self.assertEqual((api, answer), (api, spelled))
                                self.assertIn(finish, ("end_turn", "completed"))

        # A dropped call does not turn a cut-off answer into a finished one.
        engine.finish_type = "length"
        for chunk_tokens in (None, 2):
            with self.subTest("length", chunk=chunk_tokens):
                cut = answers("Sure." + unclosed)
                self.assertEqual(cut["messages"][:2], ("Sure.", "max_tokens"))
                self.assertEqual(cut["responses"][:2], ("Sure.", "incomplete"))
        engine.finish_type = "abort"
        for chunk_tokens in (None, 2):
            with self.subTest("abort", chunk=chunk_tokens):
                aborted = answers("Sure." + unclosed)
                self.assertEqual(aborted["messages"][0], "Sure.")
                self.assertEqual(aborted["responses"][:2], ("Sure.", "failed"))

    def test_responses_deferred_tool_loads_from_a_search_result(self):
        """A declared tool that defers loading is hidden and not callable until a
        search result names it, by reference or with its definition."""
        serving, engine = self._responses()
        deferred = {**WEATHER_TOOL, "defer_loading": True}
        named = {"type": "function", "name": "get_weather"}
        call = _dsml_call("get_weather", city="Paris")

        def output(response):
            return [
                (item.type, getattr(item, "name", None)) for item in response.output
            ]

        for tools, schemas in (
            ([deferred], []),
            ([SEARCH_TOOL, deferred], ["tool_search"]),
        ):
            with self.subTest("unloaded", tools=len(tools)):
                response = self._respond(serving, engine, call, tools=tools, input="hi")
                self.assertEqual(_schema_names(engine.prompts[-1]), schemas)
                self.assertNotIn(
                    "function_call", [kind for kind, _ in output(response)]
                )
                rejected = self._respond(
                    serving, engine, call, tools=tools, input="hi", tool_choice=named
                )
                self.assertEqual(
                    json.loads(rejected.body)["error"]["param"], "tool_choice"
                )
                engine.outputs.clear()

        for name, found in (("reference", named), ("definition", deferred)):
            with self.subTest("loaded", found=name):
                response = self._respond(
                    serving,
                    engine,
                    call,
                    tools=[SEARCH_TOOL, deferred],
                    input=_search_history([found]),
                    tool_choice=named,
                )
                prompt = engine.prompts[-1]
                self.assertEqual(_schema_names(prompt), ["tool_search", "get_weather"])
                loaded = prompt.index("### Updated Tool Schemas")
                self.assertGreater(loaded, prompt.index("</tool_result>"))
                self.assertIn('"required": ["city"]', prompt[loaded:])
                tag = json.loads(engine.requests[-1].sampling_params["structural_tag"])
                forced = tag["format"]["elements"][1]
                self.assertIn('invoke name="get_weather"', forced["begin"])
                self.assertEqual(
                    list(forced["content"]["json_schema"]["properties"]), ["city"]
                )
                self.assertEqual(output(response), [("function_call", "get_weather")])

        # A namespace member defers the same way and keeps its identity.
        member = CRM_NAMESPACE["tools"][0]
        response = self._respond(
            serving,
            engine,
            _dsml_call("crm::create_ticket", order="A-1"),
            tools=[
                SEARCH_TOOL,
                {**CRM_NAMESPACE, "tools": [{**member, "defer_loading": True}]},
            ],
            input=_search_history(
                [
                    {
                        **CRM_NAMESPACE,
                        "tools": [{"type": "function", "name": "create_ticket"}],
                    }
                ]
            ),
        )
        prompt = engine.prompts[-1]
        self.assertEqual(_schema_names(prompt), ["tool_search", "crm::create_ticket"])
        self.assertIn('"order"', prompt[prompt.index("### Updated Tool Schemas") :])
        (item,) = response.output
        self.assertEqual((item.namespace, item.name), ("crm", "create_ticket"))

    def test_responses_deferred_tool_loads_as_the_type_it_declares(self):
        """A search result that returns a deferred custom tool's name as a function
        is rejected, so its call is never reported as a custom call; the custom
        tool itself loads and is forced only by a custom choice of a custom tool."""
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        serving, engine = self._responses()
        deferred = {**APPLY_PATCH_TOOL, "defer_loading": True}
        function = {
            "type": "function",
            "name": "apply_patch",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}},
            },
        }
        custom_choice = {"type": "custom", "name": "apply_patch"}
        for stream in (False, True):
            for choice in ("auto", custom_choice):
                with self.subTest("conflict", stream=stream, choice=choice):
                    request = ResponsesRequest(
                        model="dummy",
                        stream=stream,
                        tools=[SEARCH_TOOL, deferred],
                        input=_search_history([function]),
                        tool_choice=choice,
                    )
                    rejected = asyncio.run(serving.create_responses(request))
                    self.assertEqual(rejected.status_code, 400)
                    error = json.loads(rejected.body)["error"]
                    self.assertEqual(error["param"], "tools")
                    self.assertIn("deferred custom tool", error["message"])
        self.assertEqual(engine.prompts, [])

        for stream in (False, True):
            with self.subTest("loaded", stream=stream):
                response = self._respond(
                    serving,
                    engine,
                    _dsml_call("apply_patch", input=PATCH),
                    stream=stream,
                    tools=[SEARCH_TOOL, deferred],
                    input=_search_history([APPLY_PATCH_TOOL]),
                    tool_choice=custom_choice,
                )
                output = (
                    response[-1]["response"]["output"]
                    if stream
                    else [item.model_dump() for item in response.output]
                )
                self.assertEqual(
                    [(item["type"], item["input"]) for item in output],
                    [("custom_tool_call", PATCH)],
                )

        # A custom choice does not force a function tool.
        rejected = self._respond(
            serving,
            engine,
            "",
            tools=[SEARCH_TOOL, deferred, WEATHER_TOOL],
            input="hi",
            tool_choice={"type": "custom", "name": "get_weather"},
        )
        self.assertEqual(json.loads(rejected.body)["error"]["param"], "tool_choice")

    def test_messages_response_format_renders_with_a_deferred_load(self):
        """`output_config.format` shows its schema in the prompt and constrains the
        answer, beside a deferred tool that loads where it is added."""
        anthropic, engine = self._messages()
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }

        def respond(output, **fields):
            engine.outputs.append(output)
            request = self._messages_request(
                [ASK_TICKET, ADD_TICKET],
                output_config={"format": {"type": "json_schema", "schema": schema}},
                **fields,
            )
            response = asyncio.run(anthropic.handle_messages(request, None))
            return response.status_code, json.loads(response.body)

        for choice, schemas in (
            ({}, ["search_docs", "create_ticket"]),
            ({"tool_choice": {"type": "none"}}, []),
        ):
            with self.subTest(choice=choice):
                status, response = respond('{"answer": "ok"}', **choice)
                self.assertEqual(status, 200)
                prompt = engine.prompts[-1]
                self.assertEqual(_schema_names(prompt), schemas)
                shown = prompt.index("## Response Format:")
                self.assertLess(shown, prompt.index(ASK_TICKET["content"]))
                self.assertIn(
                    json.dumps(schema), prompt[shown : prompt.index("<｜User｜>")]
                )
                self.assertEqual(
                    json.loads(engine.requests[-1].sampling_params["json_schema"]),
                    schema,
                )
                self.assertEqual(
                    response["content"], [{"type": "text", "text": '{"answer": "ok"}'}]
                )

        # A forced call and an answer schema cannot both constrain the output.
        count = len(engine.prompts)
        status, response = respond(
            "", tool_choice={"type": "tool", "name": "create_ticket"}
        )
        self.assertEqual(status, 400)
        self.assertIn("cannot be combined", response["error"]["message"])
        self.assertEqual(len(engine.prompts), count)

    def test_trailing_assistant_keeps_the_loaded_tools(self):
        """A final assistant message is re-sent as a user turn, as on requests that
        load no tools, or continued under continue_final_message; either way the
        prompt holds one open assistant turn and the tools the history loaded."""
        from unittest.mock import patch

        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        prefill = {"role": "assistant", "content": "The answer is"}
        loaded = '"name": "create_ticket"'
        prepare = OpenAIServingChat.prepare_native_prompt

        def continued(chat, request, **fields):
            request.continue_final_message = True
            return prepare(chat, request, **fields)

        anthropic, engine = self._messages()
        request = self._messages_request(
            [ASK_TICKET, ADD_TICKET, prefill],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        def respond(output):
            engine.outputs.append(output)
            response = asyncio.run(anthropic.handle_messages(request, None))
            prompt = engine.prompts[-1]
            self.assertEqual(_schema_names(prompt), ["search_docs", "create_ticket"])
            return prompt.partition(loaded)[2], json.loads(response.body)["content"]

        tail, content = respond("Hm.</think> 42.")
        self.assertTrue(tail.endswith("<｜User｜>The answer is<｜Assistant｜><think>"))
        self.assertEqual([block["type"] for block in content], ["thinking", "text"])
        # The continued turn stays open, and the output resumes its answer.
        with patch.object(OpenAIServingChat, "prepare_native_prompt", continued):
            tail, content = respond(" 42.")
        self.assertTrue(tail.endswith("<｜Assistant｜><think></think>The answer is"))
        self.assertEqual(content, [{"type": "text", "text": " 42."}])

        serving, engine = self._responses()
        self._respond(
            serving,
            engine,
            "42.",
            tools=[SEARCH_TOOL],
            input=[*_search_history([WEATHER_TOOL]), prefill],
        )
        prompt = engine.prompts[-1]
        self.assertEqual(_schema_names(prompt), ["tool_search", "get_weather"])
        self.assertTrue(
            prompt.endswith("<｜User｜>The answer is<｜Assistant｜></think>")
        )

    def test_messages_deferred_tool_loads_where_referenced(self):
        anthropic, engine = self._messages()
        referenced = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "found",
                        "name": "search_docs",
                        "input": {"q": "ticket"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "found",
                        "content": [
                            {"type": "tool_reference", "tool_name": "create_ticket"}
                        ],
                    }
                ],
            },
        ]
        for name, loading, anchor in (
            ("tool_reference", referenced, "</tool_result>"),
            ("tool_addition", [ADD_TICKET], ASK_TICKET["content"]),
        ):
            with self.subTest(name):
                request = self._messages_request(
                    [ASK_TICKET, *loading],
                    tool_choice={"type": "tool", "name": "create_ticket"},
                )
                chat_request = anthropic._convert_to_chat_completion_request(request)
                prompt = self.tokenizer.decode(chat_request.input_ids)
                self.assertGreater(
                    prompt.index("### Updated Tool Schemas"), prompt.index(anchor)
                )
                # Calls are constrained to the tools shown, `strict` included.
                self.assertEqual(
                    [
                        (tool.function.name, bool(tool.function.strict))
                        for tool in anthropic.openai_serving_chat._effective_tools(
                            chat_request
                        )
                    ],
                    [("search_docs", False), ("create_ticket", True)],
                )
                self.assertEqual(
                    _schema_names(prompt), ["search_docs", "create_ticket"]
                )

                engine.outputs.append(_dsml_call("create_ticket", order="A-1"))
                response = json.loads(
                    asyncio.run(anthropic.handle_messages(request, None)).body
                )
                self.assertEqual(engine.prompts[-1], prompt)
                self.assertEqual(response["stop_reason"], "tool_use")
                self.assertEqual(
                    [(block["name"], block["input"]) for block in response["content"]],
                    [("create_ticket", {"order": "A-1"})],
                )

    def test_messages_tool_choice_follows_loaded_tools(self):
        anthropic, _ = self._messages()
        for fields, error in (
            ({"tool_choice": {"type": "tool", "name": "create_ticket"}}, "deferred"),
            (
                {"tools": [{**SEARCH_DOCS_TOOL, "defer_loading": True}, TICKET_TOOL]},
                "cannot be deferred",
            ),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                anthropic._convert_to_chat_completion_request(
                    self._messages_request([ASK_TICKET], **fields)
                )

        # `none` hides even the tools the history loaded.
        chat_request = anthropic._convert_to_chat_completion_request(
            self._messages_request(
                [ASK_TICKET, ADD_TICKET], tool_choice={"type": "none"}
            )
        )
        self.assertEqual(
            _schema_names(self.tokenizer.decode(chat_request.input_ids)), []
        )
        self.assertEqual(
            anthropic.openai_serving_chat._effective_tools(chat_request), []
        )

    def test_messages_inline_tool_definition_loads_where_defined(self):
        """A custom tool a `tool_addition` block defines by value renders as the
        same tool declared deferred and added by reference."""
        anthropic, engine = self._messages()
        named = {"tool_choice": {"type": "tool", "name": "create_ticket"}}
        again = [{"role": "assistant", "content": "Done"}, ASK_TICKET]

        def prompt(messages, **fields):
            chat_request = anthropic._convert_to_chat_completion_request(
                self._messages_request(messages, **fields)
            )
            return self.tokenizer.decode(chat_request.input_ids), chat_request

        by_reference, _ = prompt([ASK_TICKET, ADD_TICKET], **named)
        request = self._messages_request(
            [ASK_TICKET, DEFINE_TICKET], tools=[SEARCH_DOCS_TOOL], **named
        )
        chat_request = anthropic._convert_to_chat_completion_request(request)
        self.assertEqual(self.tokenizer.decode(chat_request.input_ids), by_reference)
        self.assertGreater(
            by_reference.index('{"name": "create_ticket"'),
            by_reference.index(ASK_TICKET["content"]),
        )
        self.assertEqual(
            [
                (tool.function.name, bool(tool.function.strict))
                for tool in anthropic.openai_serving_chat._effective_tools(chat_request)
            ],
            [("search_docs", False), ("create_ticket", True)],
        )
        engine.outputs.append(_dsml_call("create_ticket", order="A-1"))
        response = json.loads(
            asyncio.run(anthropic.handle_messages(request, None)).body
        )
        self.assertEqual(
            [(block["name"], block["input"]) for block in response["content"]],
            [("create_ticket", {"order": "A-1"})],
        )

        # A definition of a tool already offered unchanged adds nothing; one equal
        # to a deferred declaration loads that declaration.
        once, _ = prompt([ASK_TICKET, ADD_TICKET, *again])
        self.assertEqual(
            prompt(
                [ASK_TICKET, DEFINE_TICKET, *again, DEFINE_TICKET],
                tools=[SEARCH_DOCS_TOOL],
            )[0],
            once,
        )
        self.assertEqual(prompt([ASK_TICKET, DEFINE_TICKET], **named)[0], by_reference)
        shown = [
            SEARCH_DOCS_TOOL,
            DEFINE_TICKET["content"][0]["tool"]["definition"],
            {**TICKET_TOOL, "name": "other"},
        ]
        unchanged, _ = prompt([ASK_TICKET, DEFINE_TICKET], tools=shown)
        self.assertEqual(unchanged, prompt([ASK_TICKET], tools=shown)[0])
        self.assertNotIn("### Updated Tool Schemas", unchanged)

        def addition(tool):
            return {
                "role": "system",
                "content": [{"type": "tool_addition", "tool": tool}],
            }

        changed = addition(
            {
                "type": "tool_definition",
                "definition": {**SEARCH_DOCS_TOOL, "name": "create_ticket"},
            }
        )
        mcp_toolset = {"type": "mcp_toolset", "mcp_server_name": "crm"}
        removal = {
            "type": "tool_removal",
            "tool": {"type": "tool_reference", "name": "search_docs"},
        }
        generations = len(engine.requests)
        for name, messages, tools, error in (
            ("redefined", [DEFINE_TICKET, changed], [SEARCH_DOCS_TOOL], "different"),
            ("declared differently", [changed], None, "different"),
            (
                "server tool definition",
                [addition({"type": "tool_definition", "definition": mcp_toolset})],
                None,
                "custom tools only",
            ),
            (
                "mcp tool",
                [
                    addition(
                        {
                            "type": "mcp_tool_reference",
                            "server_name": "crm",
                            "name": "create_ticket",
                        }
                    )
                ],
                None,
                "mcp_tool_reference is not supported",
            ),
            (
                "mcp toolset",
                [addition({"type": "mcp_toolset_reference", "server_name": "crm"})],
                None,
                "mcp_toolset_reference is not supported",
            ),
            (
                "removal",
                [{"role": "system", "content": [removal]}],
                None,
                "tool_removal blocks are not supported",
            ),
            ("no tool shown initially", [DEFINE_TICKET], [], "cannot be deferred"),
        ):
            with self.subTest(name):
                fields = {} if tools is None else {"tools": tools}
                response = asyncio.run(
                    anthropic.handle_messages(
                        self._messages_request([ASK_TICKET, *messages], **fields), None
                    )
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn(error, json.loads(response.body)["error"]["message"])
        self.assertEqual(len(engine.requests), generations)

    def test_messages_inline_tool_definition_takes_effect_only_where_it_stands(self):
        """A reference cannot load a tool an inline definition introduces later, and
        an inline definition outside a system message is rejected even when it
        repeats a tool already shown."""
        anthropic, engine = self._messages()
        said = {"role": "assistant", "content": "Done"}
        found = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t", "name": "search_docs", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": [
                            {"type": "tool_reference", "tool_name": "create_ticket"}
                        ],
                    }
                ],
            },
        ]
        definition = DEFINE_TICKET["content"][0]["tool"]["definition"]
        cases = [
            (
                "addition before the definition",
                [ASK_TICKET, ADD_TICKET, said, ASK_TICKET, DEFINE_TICKET],
                [SEARCH_DOCS_TOOL],
                "precedes the tool_definition",
            ),
            (
                "tool result reference before the definition",
                [ASK_TICKET, *found, DEFINE_TICKET],
                [SEARCH_DOCS_TOOL],
                "precedes the tool_definition",
            ),
        ] + [
            (
                f"definition in a {role} message",
                [ASK_TICKET, {"role": role, "content": DEFINE_TICKET["content"]}],
                [SEARCH_DOCS_TOOL, definition],
                "only be in system messages",
            )
            for role in ("user", "assistant")
        ]
        for name, messages, tools, error in cases:
            for stream in (False, True):
                with self.subTest(name, stream=stream):
                    response = asyncio.run(
                        anthropic.handle_messages(
                            self._messages_request(
                                messages, tools=tools, stream=stream
                            ),
                            None,
                        )
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(error, json.loads(response.body)["error"]["message"])
        self.assertEqual(engine.requests, [])

        def prompt(messages, **fields):
            chat_request = anthropic._convert_to_chat_completion_request(
                self._messages_request(messages, **fields)
            )
            return self.tokenizer.decode(chat_request.input_ids)

        # A reference after the definition loads the tool, like a deferred declaration.
        later = [said, ASK_TICKET, *found]
        self.assertEqual(
            prompt([ASK_TICKET, DEFINE_TICKET, *later], tools=[SEARCH_DOCS_TOOL]),
            prompt([ASK_TICKET, ADD_TICKET, *later]),
        )

    def test_messages_clear_at_drops_a_reminder_once_a_user_message_follows(self):
        anthropic, engine = self._messages()
        reminder = {
            "role": "system",
            "clear_at": "next_user_message",
            "content": [{"type": "text", "text": "REMINDER"}],
        }
        said = {"role": "assistant", "content": "Sure."}
        called = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t", "name": "search_docs", "input": {}}
            ],
        }
        result = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}],
        }
        for native in (True, False):
            for name, messages, rendered in (
                ("last message", [ASK_TICKET, reminder], True),
                ("assistant only", [ASK_TICKET, reminder, said], True),
                ("user follows", [ASK_TICKET, reminder, said, ASK_TICKET], False),
                ("tool result follows", [ASK_TICKET, reminder, called, result], False),
                (
                    "never",
                    [ASK_TICKET, {**reminder, "clear_at": "never"}, called, result],
                    True,
                ),
            ):
                with self.subTest(name, native=native):
                    engine.outputs.append("ok")
                    request = self._messages_request(
                        messages,
                        tools=[SEARCH_DOCS_TOOL, *([TICKET_TOOL] if native else [])],
                    )
                    response = asyncio.run(anthropic.handle_messages(request, None))
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual("REMINDER" in engine.prompts[-1], rendered)

        generations = len(engine.requests)
        text = reminder["content"][0]
        for name, messages in (
            ("user role", [{**ASK_TICKET, "clear_at": "never"}]),
            (
                "tool change",
                [ASK_TICKET, {**reminder, "content": ADD_TICKET["content"]}],
            ),
            (
                "output_config",
                [ASK_TICKET, {**reminder, "output_config": {"effort": "low"}}],
            ),
            (
                "cache_control",
                [
                    ASK_TICKET,
                    {
                        **reminder,
                        "content": [{**text, "cache_control": {"type": "ephemeral"}}],
                    },
                ],
            ),
            ("user directly after", [ASK_TICKET, reminder, ASK_TICKET]),
            ("after an assistant", [ASK_TICKET, said, reminder]),
        ):
            with self.subTest(name):
                response = asyncio.run(
                    anthropic.handle_messages(self._messages_request(messages), None)
                )
                self.assertEqual(response.status_code, 400)
                self.assertIn("clear_at", json.loads(response.body)["error"]["message"])
        self.assertEqual(len(engine.requests), generations)

    def test_prompt_is_recipes_rendering_in_sglang_settings(self):
        """The prompt is recipe's own encoding of the request, with SGLang's
        thinking and effort decisions spelled in its wire form."""
        import deepseek_recipe as recipe

        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        encoding = recipe.DeepseekV41Encoding().with_tokenizer(
            recipe.Tokenizer.from_file(_tokenizer_file())
        )

        def reference(request_type, body):
            request = request_type({"model": "DeepSeek-V4.1-Flash", **body})
            return encoding.encode(
                request.convert(recipe.ConversionOptions()).conversation
            )

        anthropic, _ = self._messages()
        body = {
            "tools": [SEARCH_DOCS_TOOL, TICKET_TOOL],
            "messages": [ASK_TICKET, ADD_TICKET],
        }
        # Thinking is off by default; an Anthropic `xhigh` effort is SGLang's `max`.
        for fields, settings in (
            ({}, {"thinking": {"type": "disabled"}}),
            (
                {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "xhigh"},
                },
                {"output_config": {"effort": "max"}},
            ),
        ):
            with self.subTest(fields=fields):
                chat_request = anthropic._convert_to_chat_completion_request(
                    self._messages_request(body["messages"], **fields)
                )
                self.assertEqual(
                    chat_request.input_ids,
                    reference(recipe.MessagesRequest, {**body, **settings}),
                )

        serving, _ = self._responses()
        body = {"tools": [SEARCH_TOOL], "input": "Weather in Paris?"}
        request = ResponsesRequest(model="dummy", reasoning={"effort": "low"}, **body)
        _, _, (prompt_ids,), _ = asyncio.run(
            serving._make_request(request, None, self.tokenizer)
        )
        self.assertEqual(
            prompt_ids,
            reference(
                recipe.ResponsesRequest, {**body, "reasoning": {"effort": "low"}}
            ),
        )


if __name__ == "__main__":
    unittest.main()
