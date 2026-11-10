import copy
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


def _serving(test, tokenizer, *, is_multimodal=False, responses=False, **server_args):
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
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
    )
    template_manager = SimpleNamespace(
        chat_template_name=None,
        jinja_template_content_format=None,
        reasoning_config=None,
        force_reasoning=False,
        python_chat_template=None,
        jinja_template_may_reorder_tool_results=False,
    )
    serving_class = OpenAIServingResponses if responses else OpenAIServingChat
    return serving_class(manager, template_manager)


def _function_tool(name, description, parameter):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {parameter: {"type": "string"}},
                "required": [parameter],
            },
        },
    }


_SEARCH_DOCS = _function_tool("search_docs", "Search the help center.", "query")
_CREATE_TICKET = _function_tool("create_ticket", "Open a support ticket.", "summary")
# A tool a system message declares after an answered turn.
_DECLARING_MESSAGES = [
    {"role": "user", "content": "My order is late."},
    {"role": "assistant", "content": "Let me check."},
    {"role": "system", "content": "", "tools": [_CREATE_TICKET]},
    {"role": "user", "content": "Please open a ticket."},
]
_IMAGE = "<｜deepseek_image｜>"


def _recipe_options(thinking):
    import deepseek_recipe as recipe

    special_tokens = recipe.SpecialTokens(image=_IMAGE)
    options = recipe.ConversionOptions().with_special_tokens(special_tokens)
    return options.with_default_thinking_mode(thinking)


def _grammar_tools(constraint):
    """The tools a DSML tool-call grammar lets the model invoke."""
    if constraint is None:
        return set()
    return _invoked(constraint[1].model_dump_json())


def _invoked(grammar):
    return set(re.findall(r'invoke name=\\+"([^"\\]+)', grammar or ""))


def _dsml(*calls):
    """Model output that invokes each (name, arguments), string values only."""
    invokes = "\n".join(
        f'<｜DSML｜ invoke name="{name}">\n'
        + "\n".join(
            f'<｜DSML｜ parameter name="{key}" string="true">{value}</｜DSML｜ parameter>'
            for key, value in arguments.items()
        )
        + "\n</｜DSML｜ invoke>"
        for name, arguments in calls
    )
    return f"\n\n<｜DSML｜ calls>\n{invokes}\n</｜DSML｜ calls>"


def _engine(serving, output_ids, finish):
    """The engine answers `output_ids`: as one result, or streamed token by token.
    Returns the requests it receives."""
    from unittest.mock import Mock

    received = []

    async def generate(request, raw_request=None):
        received.append(request)
        for end in range(1 if request.stream else len(output_ids), len(output_ids) + 1):
            yield {
                "text": "",
                "output_ids": output_ids[:end],
                "meta_info": {
                    "id": "rid",
                    "prompt_tokens": 5,
                    "completion_tokens": end,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "weight_version": "default",
                    "finish_reason": finish if end == len(output_ids) else None,
                },
            }

    serving.tokenizer_manager.generate_request = generate
    serving.tokenizer_manager.create_abort_task = Mock()
    return received


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

    def test_dsv4_rejects_message_level_tools(self):
        """V4 does not render tools a message declares, so the request must fail
        rather than reach the model without them."""
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        adapter = registry.build_chat_encoding_adapter(
            self._context("dsv4", tokenizer=self._fast_tokenizer())
        )
        request = ChatCompletionRequest(messages=_DECLARING_MESSAGES)
        with self.assertRaises(ValueError):
            adapter.encode_messages(
                messages=[], request=request, tools=None, thinking=False, image_data=[]
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


class TestDynamicTools(CustomTestCase):
    """Tools that load mid-conversation render where deepseek-recipe 0.2.0 puts them:
    the oracles are recipe's own Conversation, Responses and Messages conversions."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import deepseek_recipe as recipe
        from transformers import PreTrainedTokenizerFast

        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_file=_tokenizer_file())
        cls.encoding = (
            recipe.DeepseekV41Encoding()
            .with_special_tokens(recipe.SpecialTokens(image=_IMAGE))
            .with_tokenizer(recipe.Tokenizer.from_file(_tokenizer_file()))
        )

    def _chat(self, chat, **fields):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        return chat._process_messages(ChatCompletionRequest(**fields), False)

    def test_chat_message_tools_render_where_declared(self):
        """A tool a system message declares renders there as recipe's updated tool
        schemas, not in the initial tool list, so the prompt before it holds."""
        import deepseek_recipe as recipe

        chat = _serving(self, self.tokenizer)

        def definition(tool, **fields):
            function = tool["function"]
            return recipe.ToolDefinition(
                function["name"],
                function["description"],
                function["parameters"],
                **fields,
            )

        search = definition(_SEARCH_DOCS)
        ticket = definition(_CREATE_TICKET, defer_loading=True)
        # Message-only tools activate tool mode, so recipe adds its tools preamble.
        cases = {
            "empty content": ("", [_SEARCH_DOCS], [search, ticket]),
            "content": ("Tickets are open.", [_SEARCH_DOCS], [search, ticket]),
            "message only": ("", None, [ticket]),
        }
        for case, (content, request_tools, definitions) in cases.items():
            with self.subTest(case):
                messages = copy.deepcopy(_DECLARING_MESSAGES)
                messages[2]["content"] = content
                expected = recipe.Conversation(
                    [
                        recipe.UserMessage("My order is late."),
                        recipe.AssistantMessage("Let me check."),
                        recipe.SystemMessage(content, tools=[ticket]),
                        recipe.UserMessage("Please open a ticket."),
                    ],
                    thinking_mode=False,
                    tools=definitions,
                )
                self.assertEqual(
                    self._chat(chat, messages=messages, tools=request_tools).prompt_ids,
                    self.encoding.encode(expected),
                )
        # With tools already active, the prompt up to the declaration is unchanged.
        prompt = self._chat(
            chat,
            model="DeepSeek-V4.1-Flash",
            thinking={"type": "disabled"},
            messages=_DECLARING_MESSAGES,
            tools=[_SEARCH_DOCS],
        ).prompt_ids
        without = self._chat(
            chat,
            messages=_DECLARING_MESSAGES[:2] + _DECLARING_MESSAGES[3:],
            tools=[_SEARCH_DOCS],
        ).prompt_ids
        system = self.tokenizer.convert_tokens_to_ids("<｜System｜>")
        systems = [i for i, token in enumerate(prompt) if token == system]
        self.assertEqual(len(systems), 2)  # the tools preamble, then the declaration
        self.assertEqual(prompt[: systems[1]], without[: systems[1]])

    def test_chat_tool_choice_matches_message_tools(self):
        """Each tool choice shows the model the tools its grammar allows."""
        chat = _serving(self, self.tokenizer)
        named = {"type": "function", "function": {"name": "create_ticket"}}
        cases = (
            ("required", {"search_docs", "create_ticket"}),
            (named, {"create_ticket"}),
            ("none", set()),
        )
        for choice, shown in cases:
            with self.subTest(choice=choice):
                processed = self._chat(
                    chat,
                    messages=_DECLARING_MESSAGES,
                    tools=[_SEARCH_DOCS],
                    tool_choice=choice,
                )
                prompt = self.tokenizer.decode(processed.prompt_ids)
                for name in ("search_docs", "create_ticket"):
                    self.assertEqual(f'"name": "{name}"' in prompt, name in shown)
                self.assertEqual(_grammar_tools(processed.tool_call_constraint), shown)

    _TOOL_SEARCH = {
        "type": "tool_search",
        "execution": "client",
        "description": "Find tools.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
    _LOOKUP = {
        "type": "function",
        "name": "lookup",
        "description": "Look up a key.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
    }

    def _create(self, serving, **fields):
        """A /v1/responses call: the response, an error, or the stream's events."""
        import asyncio

        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        async def run():
            request = ResponsesRequest(model="dummy", **fields)
            result = await serving.create_responses(request)
            if not fields.get("stream"):
                return result
            return [json.loads(e.split("data: ", 1)[1]) async for e in result if e]

        return asyncio.run(run())

    def _native_responses(self, items, tools, *, thinking=False, **fields):
        import deepseek_recipe as recipe

        body = {"model": "DeepSeek-V4.1-Flash", "input": items, "tools": tools}
        request = recipe.ResponsesRequest(body | fields)
        conversation = request.convert(_recipe_options(thinking)).conversation
        return self.encoding.encode(conversation)

    def _output(self, *calls, finish="stop"):
        """Output ids of the calls and their finish; a stop ends on the EOS."""
        eos = self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
        output_ids = self.tokenizer.encode(_dsml(*calls), add_special_tokens=False)
        if finish == "stop":
            return output_ids + [eos], {"type": "stop", "matched": eos}
        return output_ids, {"type": finish}

    def test_responses_tool_search_round_trip(self):
        """A client tool_search call is reported as one, streamed or not, and the
        tools its output loads render where recipe's Responses adapter puts them,
        replayed by previous_response_id as when the client echoes the history."""
        serving = _serving(
            self, self.tokenizer, responses=True, enable_response_store=True
        )
        tools = [self._TOOL_SEARCH]
        question = {"role": "user", "content": "Weather in Paris?"}
        received = _engine(serving, *self._output(("tool_search", {"query": "x"})))
        first = self._create(serving, input=[question], tools=tools)
        self.assertEqual(
            received[-1].input_ids, self._native_responses([question], tools)
        )
        effort = {"effort": "high"}
        self._create(serving, input=[question], tools=tools, reasoning=effort)
        self.assertEqual(
            received[-1].input_ids,
            self._native_responses([question], tools, thinking=True, reasoning=effort),
        )
        expected = {
            "type": "tool_search_call",
            "execution": "client",
            "arguments": {"query": "x"},
            "status": "completed",
        }

        def fields(item):
            return {key: item.get(key) for key in expected}

        self.assertEqual(
            [fields(item.model_dump()) for item in first.output], [expected]
        )
        events = self._create(serving, input=[question], tools=tools, stream=True)
        self.assertFalse(
            [e for e in events if e["type"].startswith("response.function_call")]
        )
        added, done = (
            [e["item"] for e in events if e["type"] == f"response.output_item.{kind}"]
            for kind in ("added", "done")
        )
        self.assertEqual(
            [fields(item) for item in added],
            [{**expected, "arguments": {}, "status": "in_progress"}],
        )
        self.assertEqual([fields(item) for item in done], [expected])
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["output"], done)

        # The client runs the search and returns what it loaded.
        loaded = {
            "type": "tool_search_output",
            "call_id": first.output[0].call_id,
            "execution": "client",
            "tools": [self._LOOKUP],
        }
        self._create(
            serving, input=[loaded], tools=tools, previous_response_id=first.id
        )
        history = [question, first.output[0].model_dump(), loaded]
        self._create(serving, input=history, tools=tools)
        self.assertEqual(received[-2].input_ids, received[-1].input_ids)
        self.assertEqual(received[-1].input_ids, self._native_responses(history, tools))

    def test_responses_tool_choice_admission(self):
        """Required and named choices admit the tools the request has loaded, the
        search tool and discovered tools included, and their grammar forces them."""
        serving = _serving(
            self, self.tokenizer, responses=True, enable_response_store=True
        )
        tools = [self._TOOL_SEARCH, {**self._LOOKUP, "defer_loading": True}]
        received = _engine(serving, *self._output(("tool_search", {"query": "x"})))
        searched = self._create(serving, input="Find lookup.", tools=tools)
        loaded = {
            "type": "tool_search_output",
            "call_id": searched.output[0].call_id,
            "execution": "client",
            "tools": [self._LOOKUP],
        }
        after_search = {"input": [loaded], "previous_response_id": searched.id}
        cases = (
            ("required", {}, {"tool_search"}),
            ({"type": "tool_search"}, {}, {"tool_search"}),
            ({"type": "function", "name": "tool_search"}, {}, {"tool_search"}),
            ({"type": "function", "name": "lookup"}, after_search, {"lookup"}),
            ("required", after_search, {"tool_search", "lookup"}),
        )
        for choice, fields, grammar in cases:
            with self.subTest(choice=choice, loaded=bool(fields)):
                response = self._create(
                    serving,
                    tools=tools,
                    tool_choice=choice,
                    **{"input": "Go.", **fields},
                )
                self.assertEqual(response.status, "completed")
                self.assertEqual(response.tool_choice, choice)
                sampling = received[-1].sampling_params
                self.assertEqual(_invoked(sampling.get("structural_tag")), grammar)
        # A deferred tool no search has returned cannot be forced.
        for choice, deferred_only in (
            ({"type": "function", "name": "lookup"}, False),
            ("required", True),
        ):
            with self.subTest(choice=choice, deferred_only=deferred_only):
                response = self._create(
                    serving,
                    input="Go.",
                    tools=tools[1:] if deferred_only else tools,
                    tool_choice=choice,
                )
                self.assertEqual(response.status_code, 400)
        # Choosing the search tool needs a client one: no fallback to auto, and no
        # function that only shares its name. A namespace member is chosen in its
        # namespace, never as the top-level tool that shares its name.
        named_search = {**self._LOOKUP, "name": "tool_search"}
        store = {
            "type": "namespace",
            "name": "store",
            "description": "Key-value store.",
            "tools": [{**self._LOOKUP, "defer_loading": True}],
        }
        colliding = [self._TOOL_SEARCH, self._LOOKUP, store]
        in_store = {"type": "function", "name": "lookup", "namespace": "store"}
        rejected = (
            ({"type": "tool_search"}, [], "tool_search"),
            ({"type": "tool_search"}, [self._LOOKUP], "tool_search"),
            ({"type": "tool_search"}, [named_search], "tool_search"),
            (in_store, colliding, "'store'"),
        )
        for choice, declared, named in rejected:
            with self.subTest(choice=choice, declared=declared):
                engine_requests = len(received)
                response = self._create(
                    serving, input="Go.", tools=declared, tool_choice=choice
                )
                self.assertEqual(getattr(response, "status_code", None), 400)
                error = json.loads(response.body)["error"]
                self.assertIn(named, error["message"])
                self.assertEqual(len(received), engine_requests)
        # A function named tool_search can still be forced, and so can the
        # top-level lookup, and the namespace's lookup once a search loads it.
        store_loaded = [
            {"role": "user", "content": "Find lookup."},
            {
                "type": "tool_search_call",
                "call_id": "s1",
                "execution": "client",
                "arguments": {"query": "lookup"},
            },
            {
                "type": "tool_search_output",
                "call_id": "s1",
                "execution": "client",
                "tools": [{**store, "tools": [self._LOOKUP]}],
            },
        ]
        named_function = {"type": "function", "name": "tool_search"}
        deferring = [named_search, {**self._LOOKUP, "defer_loading": True}]
        allowed = (
            (named_function, [named_search], "Go.", "tool_search"),
            (named_function, deferring, "Go.", "tool_search"),
            ({"type": "function", "name": "lookup"}, colliding, "Go.", "lookup"),
            (in_store, colliding, store_loaded, "store::lookup"),
        )
        for choice, declared, items, callee in allowed:
            with self.subTest(choice=choice, declared=declared):
                response = self._create(
                    serving, input=items, tools=declared, tool_choice=choice
                )
                self.assertEqual(response.status, "completed")
                self.assertEqual(response.tool_choice, choice)
                sampling = received[-1].sampling_params
                self.assertEqual(_invoked(sampling.get("structural_tag")), {callee})
                # The fake model calls tool_search: a search only if one is declared.
                searches = self._TOOL_SEARCH in declared
                self.assertEqual(
                    response.output[0].type,
                    "tool_search_call" if searches else "function_call",
                )

    def test_responses_search_item_finalization(self):
        """As recipe reports it, the output limit leaves the search call it cuts off
        incomplete with no arguments, even when its markup closed; a call that a
        later one closed keeps its arguments."""
        serving = _serving(self, self.tokenizer, responses=True)
        search = ("tool_search", {"query": "x"})
        complete = {"status": "completed", "arguments": {"query": "x"}}
        cut = {"status": "incomplete", "arguments": {}}
        # A function call keeps SGLang's function-call convention.
        function = {"status": "completed", "arguments": '{"key": "k"}'}
        lookup = ("lookup", {"key": "k"})
        cases = (
            ("stop", (search,), "stop", "completed", [complete]),
            ("length", (search,), "length", "incomplete", [cut]),
            ("partial", (search,), "partial", "incomplete", [cut]),
            ("second cut", (search, search), "length", "incomplete", [complete, cut]),
            (
                "call cut",
                (search, lookup),
                "length",
                "incomplete",
                [complete, function],
            ),
        )
        for case, calls, finish, status, items in cases:
            if finish == "partial":
                # Cut inside the call: its parameter never closes.
                text = _dsml(*calls).split("</｜DSML｜ parameter>")[0]
                output_ids = self.tokenizer.encode(text, add_special_tokens=False)
                _engine(serving, output_ids, {"type": "length"})
            else:
                _engine(serving, *self._output(*calls, finish=finish))
            request = {"input": "Search.", "tools": [self._TOOL_SEARCH, self._LOOKUP]}
            with self.subTest(case):
                response = self._create(serving, **request)
                self.assertEqual(response.status, status)
                outputs = [item.model_dump() for item in response.output]
                self.assertEqual(
                    [{k: item[k] for k in ("status", "arguments")} for item in outputs],
                    items,
                )
                events = self._create(serving, stream=True, **request)
                done = [
                    {k: e["item"][k] for k in ("status", "arguments")}
                    for e in events
                    if e["type"] == "response.output_item.done"
                ]
                self.assertEqual(done, items)
                self.assertEqual(events[-1]["type"], f"response.{status}")
                self.assertEqual(
                    [
                        {k: item[k] for k in ("status", "arguments")}
                        for item in events[-1]["response"]["output"]
                    ],
                    items,
                )

    def test_responses_strict_tools_keep_their_grammar(self):
        """A strict function beside the client search tool, or returned by it, keeps
        its argument grammar; recipe would refuse the strict and non-strict mix."""
        import xgrammar as xgr
        from xgrammar.testing import _is_grammar_accept_string

        serving = _serving(self, self.tokenizer, responses=True)
        received = _engine(serving, *self._output())
        schema = {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": ["limit"],
        }
        strict = {**self._LOOKUP, "parameters": schema, "strict": True}
        history = [
            {"role": "user", "content": "Find lookup."},
            {
                "type": "tool_search_call",
                "call_id": "s1",
                "execution": "client",
                "arguments": {"query": "lookup"},
            },
            {
                "type": "tool_search_output",
                "call_id": "s1",
                "execution": "client",
                "tools": [strict],
            },
        ]
        cases = (
            ("declared", "Go.", [self._TOOL_SEARCH, strict], "required"),
            ("discovered", history, [self._TOOL_SEARCH], "required"),
        )
        call = _dsml(("lookup", {"limit": 3})).replace(
            'string="true"', 'string="false"'
        )
        for case, items, tools, choice in cases:
            with self.subTest(case):
                response = self._create(
                    serving, input=items, tools=tools, tool_choice=choice
                )
                self.assertEqual(response.status, "completed")
                grammar = xgr.Grammar.from_structural_tag(
                    received[-1].sampling_params["structural_tag"]
                )
                self.assertTrue(_is_grammar_accept_string(grammar, call))
                for wrong in (call.replace(">3<", '>"3"<'), call.replace(">3<", "><")):
                    self.assertFalse(_is_grammar_accept_string(grammar, wrong))

    def test_responses_deferred_tools_load_from_search(self):
        """A deferred request tool stays out of the prompt until a search output
        returns its definition, which then renders where recipe puts it."""
        serving = _serving(
            self, self.tokenizer, responses=True, enable_response_store=True
        )
        received = _engine(serving, *self._output())
        deferred = {**self._LOOKUP, "defer_loading": True}
        visible = {**self._LOOKUP, "name": "list_keys"}
        namespace = {
            "type": "namespace",
            "name": "store",
            "description": "Key-value store.",
            "tools": [visible, deferred],
        }

        def prompt(**fields):
            response = self._create(serving, **fields)
            return response, received[-1].input_ids

        cases = (
            ("with search", [self._TOOL_SEARCH, deferred], [self._TOOL_SEARCH]),
            ("without search", [deferred], []),
            (
                "namespace member",
                [self._TOOL_SEARCH, namespace],
                [self._TOOL_SEARCH, {**namespace, "tools": [visible]}],
            ),
        )
        for case, tools, shown in cases:
            with self.subTest(case):
                response, deferring = prompt(input="Go.", tools=tools)
                self.assertEqual(deferring, prompt(input="Go.", tools=shown)[1])
                # The response reports the request's tools as declared.
                self.assertEqual(
                    [tool.model_dump(exclude_unset=True) for tool in response.tools],
                    tools,
                )
        search = {
            "type": "tool_search_call",
            "call_id": "s1",
            "execution": "client",
            "arguments": {"query": "lookup"},
        }
        loaded = {
            "type": "tool_search_output",
            "call_id": "s1",
            "execution": "client",
            "tools": [self._LOOKUP],
        }
        history = [{"role": "user", "content": "Find lookup."}, search, loaded]
        _, discovered = prompt(input=history, tools=[self._TOOL_SEARCH, deferred])
        self.assertEqual(
            discovered, self._native_responses(history, [self._TOOL_SEARCH])
        )

    def test_messages_load_deferred_tools_where_referenced(self):
        """Deferred tools stay out of the initial list and render where a tool
        result references them or a system message adds them, as recipe's Messages
        adapter places them; a named choice keeps only that tool, prompt and grammar."""
        import deepseek_recipe as recipe

        from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
        from sglang.srt.entrypoints.anthropic.serving import AnthropicServing

        chat = _serving(self, self.tokenizer)
        anthropic = AnthropicServing(chat)

        def tool(name, description, **fields):
            schema = {"type": "object", "properties": {"arg": {"type": "string"}}}
            return {
                "name": name,
                "description": description,
                "input_schema": schema,
                **fields,
            }

        tools = [
            tool("search", "Search for tools."),
            tool("Bash", "Run a shell command.", defer_loading=True),
            tool("Read", "Read a file.", defer_loading=True),
        ]
        referenced = [
            {"role": "user", "content": "List the files."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "Found:"},
                            {"type": "tool_reference", "tool_name": "Bash"},
                            {"type": "text", "text": "Use it."},
                        ],
                    },
                    {"type": "text", "text": "Go on."},
                ],
            },
        ]
        added = referenced + [
            {"role": "assistant", "content": "Done."},
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Reading is allowed."},
                    {
                        "type": "tool_addition",
                        "tool": {"type": "tool_reference", "name": "Read"},
                    },
                ],
            },
            {"role": "user", "content": "Read the README."},
        ]

        def native(body, choice=None):
            request = recipe.MessagesRequest(body | {"model": "DeepSeek-V4.1-Flash"})
            conversation = request.convert(_recipe_options(False)).conversation
            # SGLang's grammar, not a prompt prefix, forces a named call.
            return self.encoding.encode(
                recipe.Conversation(
                    conversation.messages,
                    thinking_mode=False,
                    tools=conversation.tools,
                    tool_choice=choice or conversation.tool_choice,
                )
            )

        named = {"type": "tool", "name": "Bash"}
        loaded = {"search", "Bash", "Read"}
        before = referenced[:1]
        cases = (
            ("tool_reference", referenced, None, set()),
            ("tool_addition", added, None, set()),
            ("none", added, {"type": "none"}, set()),
            ("named", added, named, {"Bash"}),
            # `any` forces the loaded tools only: deferred ones until they load.
            ("any", added, {"type": "any"}, loaded),
            ("any before loading", before, {"type": "any"}, {"search"}),
        )
        for case, messages, choice, grammar in cases:
            body = {"model": "dummy", "max_tokens": 64, "tools": tools}
            body["messages"] = messages
            if choice is not None:
                body["tool_choice"] = choice
            with self.subTest(case):
                request = anthropic._convert_to_chat_completion_request(
                    AnthropicMessagesRequest.model_validate(body)
                )
                processed = chat._process_messages(request, False)
                self.assertEqual(
                    processed.prompt_ids,
                    native(body, choice="auto" if choice is named else None),
                )
                self.assertEqual(
                    _grammar_tools(processed.tool_call_constraint), grammar
                )
        # A deferred tool nothing has loaded cannot be forced.
        body = {"model": "dummy", "max_tokens": 64, "tools": tools}
        body |= {
            "messages": referenced,
            "tool_choice": {"type": "tool", "name": "Read"},
        }
        request = anthropic._convert_to_chat_completion_request(
            AnthropicMessagesRequest.model_validate(body)
        )
        error = chat._validate_request(request)
        self.assertIsNotNone(error)
        self.assertIn("'Read'", error)

    def test_messages_handler_calls_a_loaded_tool(self):
        """/v1/messages answers with a call to the tool a result loaded, streamed or
        not, and rejects a reference to an undeclared tool."""
        import asyncio

        from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
        from sglang.srt.entrypoints.anthropic.serving import AnthropicServing

        chat = _serving(self, self.tokenizer)
        anthropic = AnthropicServing(chat)
        schema = {"type": "object", "properties": {"cmd": {"type": "string"}}}
        tools = [
            {"name": "search", "input_schema": {"type": "object"}},
            {"name": "Bash", "input_schema": schema, "defer_loading": True},
        ]
        result = [
            {"type": "text", "text": "Found:"},
            {"type": "tool_reference", "tool_name": "Bash"},
        ]
        messages = [
            {"role": "user", "content": "List the files."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": result}
                ],
            },
        ]
        eos = self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
        output_ids = self.tokenizer.encode(
            _dsml(("Bash", {"cmd": "ls"})), add_special_tokens=False
        )
        _engine(chat, output_ids + [eos], {"type": "stop", "matched": eos})
        raw_request = SimpleNamespace(
            headers={}, is_disconnected=lambda: asyncio.sleep(0, result=False)
        )

        def handle(body):
            async def run():
                request = AnthropicMessagesRequest.model_validate(
                    {"model": "dummy", "max_tokens": 64, "tools": tools, **body}
                )
                response = await anthropic.handle_messages(request, raw_request)
                if not request.stream or response.status_code != 200:
                    return response.status_code, json.loads(response.body)
                text = "".join(
                    [
                        chunk if isinstance(chunk, str) else chunk.decode()
                        async for chunk in response.body_iterator
                    ]
                )
                lines = text.splitlines()
                return 200, [json.loads(l[6:]) for l in lines if l.startswith("data: ")]

            return asyncio.run(run())

        status, response = handle({"messages": messages})
        self.assertEqual(status, 200)
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(
            [(b["type"], b["name"], b["input"]) for b in response["content"]],
            [("tool_use", "Bash", {"cmd": "ls"})],
        )
        status, events = handle({"messages": messages, "stream": True})
        starts = [
            e["content_block"] for e in events if e["type"] == "content_block_start"
        ]
        self.assertEqual(
            [(b["type"], b["name"]) for b in starts], [("tool_use", "Bash")]
        )
        arguments = "".join(
            e["delta"]["partial_json"]
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "input_json_delta"
        )
        self.assertEqual(json.loads(arguments), {"cmd": "ls"})
        self.assertEqual(
            [e["delta"]["stop_reason"] for e in events if e["type"] == "message_delta"],
            ["tool_use"],
        )
        unknown = copy.deepcopy(messages)
        unknown[2]["content"][0]["content"][1]["tool_name"] = "Read"
        for body in (
            {"messages": unknown},
            {"messages": messages[:1], "tool_choice": {"type": "tool", "name": "Bash"}},
        ):
            with self.subTest(body=body):
                status, error = handle(body)
                self.assertEqual(status, 400)
                self.assertEqual(error["error"]["type"], "invalid_request_error")

    def test_messages_strict_tool_survives_deferral(self):
        """A deferred tool must not drop another tool's strict argument grammar."""
        import xgrammar as xgr
        from xgrammar.testing import _is_grammar_accept_string

        from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
        from sglang.srt.entrypoints.anthropic.serving import AnthropicServing

        chat = _serving(self, self.tokenizer)
        anthropic = AnthropicServing(chat)
        schema = {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": ["limit"],
        }
        strict = {"name": "search", "input_schema": schema, "strict": True}
        deferred = {"name": "Bash", "input_schema": schema, "defer_loading": True}
        call = _dsml(("search", {"limit": 3})).replace(
            'string="true"', 'string="false"'
        )
        for tools in ([strict], [strict, deferred]):
            with self.subTest(deferred=len(tools) > 1):
                body = {"model": "dummy", "max_tokens": 64, "tools": tools}
                body["messages"] = [{"role": "user", "content": "Search."}]
                request = anthropic._convert_to_chat_completion_request(
                    AnthropicMessagesRequest.model_validate(body)
                )
                constraint = chat._process_messages(request, False).tool_call_constraint
                self.assertIsNotNone(constraint)
                grammar = xgr.Grammar.from_structural_tag(
                    constraint[1].model_dump_json()
                )
                self.assertTrue(_is_grammar_accept_string(grammar, call))
                for wrong in (
                    call.replace(">3<", '>"three"<'),
                    call.replace(">3<", "><"),
                ):
                    self.assertFalse(_is_grammar_accept_string(grammar, wrong))

    def test_chat_rejects_request_level_deferral(self):
        """Chat has no reference that would load a deferred request tool, so it is
        refused rather than hidden yet callable, or shown despite the deferral."""
        chat = _serving(self, self.tokenizer)
        with self.assertRaisesRegex(ValueError, "defer_loading"):
            self._chat(
                chat,
                messages=[{"role": "user", "content": "Search."}],
                tools=[_SEARCH_DOCS, {**_CREATE_TICKET, "defer_loading": True}],
            )


if __name__ == "__main__":
    unittest.main()
