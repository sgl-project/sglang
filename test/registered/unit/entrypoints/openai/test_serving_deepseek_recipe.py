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
            vocab_size=len(tokenizer),
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

    def test_message_tools_render_where_declared(self):
        """Tools a system or developer message declares reach the prompt at that
        message, as the parser expects, unless tool_choice hides them."""
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        chat = _serving(self, self.tokenizer)
        search, ticket = (
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool",
                    "parameters": {"type": "object"},
                },
            }
            for name in ("search_docs", "create_ticket")
        )
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        declaration = {"role": "developer", "content": None, "tools": [ticket]}
        ask = "<｜User｜>file it"

        def prompt(messages, tools=(search,), tool_choice="auto"):
            request = ChatCompletionRequest(
                messages=[*history, *messages, {"role": "user", "content": "file it"}],
                tools=list(tools) or None,
                tool_choice=tool_choice,
                chat_template_kwargs={"thinking": True, "reasoning_effort": 30},
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "ticket", "schema": {"type": "object"}},
                },
            )
            return self.tokenizer.decode(
                chat._process_messages(request, False).prompt_ids
            )

        declared = prompt([]).replace(
            ask,
            "<｜System｜>\n\n### Updated Tool Schemas\n\n"
            + json.dumps(ticket["function"])
            + ask,
        )
        self.assertEqual(prompt([declaration]), declared)
        # SGLang accepts a tool without its type.
        untyped = {**declaration, "tools": [{"function": ticket["function"]}]}
        self.assertEqual(prompt([untyped]), declared)
        # Without request tools the opening block still teaches the call format.
        self.assertEqual(
            prompt([declaration], tools=()),
            declared.replace(json.dumps(search["function"]), ""),
        )
        for choice in (
            "none",
            {"type": "function", "function": {"name": "search_docs"}},
        ):
            with self.subTest(tool_choice=choice):
                self.assertEqual(
                    prompt([declaration], tool_choice=choice),
                    prompt([], tool_choice=choice),
                )

    def test_message_tools_keep_image_placeholders(self):
        """A declaring message keeps its images and its text that spells the image
        token where the same message without tools renders them."""
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
        from sglang.srt.parser.deepseek_recipe_renderer import IMAGE_TOKEN

        chat = _serving(self, self.tokenizer, is_multimodal=True)
        image_id = self._token(IMAGE_TOKEN)
        user = "<｜User｜>"
        tool = {"function": {"name": "f", "parameters": {"type": "object"}}}

        def image(name):
            url = f"https://example.com/{name}.png"
            return {"type": "image_url", "image_url": {"url": url}}

        def text(value):
            return {"type": "text", "text": value}

        def render(content, tools):
            request = ChatCompletionRequest(
                messages=[
                    {"role": "system", "content": content, "tools": tools},
                    {"role": "user", "content": [image("u"), text("hi")]},
                ],
            )
            processed = chat._process_messages(request, True)
            between_images, ids = [], []
            for token_id in [*processed.prompt_ids, image_id]:
                if token_id == image_id:
                    between_images.append(self.tokenizer.decode(ids))
                    ids = []
                else:
                    ids.append(token_id)
            return between_images, [item.url for item in processed.image_data]

        literal = f"literal {IMAGE_TOKEN}"
        for name, content, images in (
            ("after", [text("see"), image("a"), text(literal)], "a"),
            (
                "interleaved",
                [image("a"), text("one"), image("b"), text(f"{literal} {literal}")],
                "ab",
            ),
            ("no image", literal, ""),
        ):
            with self.subTest(name):
                declared, declared_images = render(content, [tool])
                plain, plain_images = render(content, None)
                self.assertEqual(
                    declared_images,
                    [image(name)["image_url"]["url"] for name in images + "u"],
                )
                self.assertEqual(declared_images, plain_images)
                self.assertEqual(len(declared), len(plain))
                for declared_text, plain_text in zip(declared, plain):
                    system, _, rest = declared_text.partition(user)
                    plain_system, _, plain_rest = plain_text.partition(user)
                    self.assertTrue(system.startswith(plain_system), declared_text)
                    self.assertEqual(rest, plain_rest)
                self.assertIn("### Updated Tool Schemas", declared[len(images)])
        # Recipe's message constructor cannot place text that spells the token
        # ahead of an image, so the request fails instead of moving the image.
        for name, content in (
            ("before", [text(literal), image("a")]),
            ("between", [image("a"), text(literal), image("b")]),
        ):
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "before one of its images"):
                    render(content, [tool])

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


def _searched_again(found_tools):
    """A further search call and its result, after `_search_history`."""
    return [{**item, "call_id": "again"} for item in _search_history(found_tools)[1:]]


def _dsml_call(name, **arguments):
    parameters = "".join(
        f'<｜DSML｜ parameter name="{key}" string="true">{value}</｜DSML｜ parameter>\n'
        for key, value in arguments.items()
    )
    return (
        f'<｜DSML｜ calls>\n<｜DSML｜ invoke name="{name}">\n{parameters}'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )


def _schemas(prompt):
    """The schema lines the prompt shows, initially or as updates."""
    return re.findall(r'^\{"name": .*$', prompt, flags=re.MULTILINE)


def _schema_names(prompt):
    return re.findall(r'^\{"name": "([^"]+)"', prompt, flags=re.MULTILINE)


async def _sse_payloads(stream):
    return [
        json.loads(line.removeprefix("data: "))
        async for chunk in stream
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ]


def _counting_conversions():
    """Patches recipe conversion to record each converted body."""
    from unittest.mock import patch

    from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe

    bodies = []
    converted = DeepSeekRecipe._converted

    def counted(recipe, request_type, body, options):
        bodies.append(body)
        return converted(recipe, request_type, body, options)

    return bodies, patch.object(DeepSeekRecipe, "_converted", counted)


class _Engine:
    """Answers each request with its next scripted output and keeps its prompt;
    `chunk_tokens` splits the output into chunks of that many tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.outputs = []
        self.prompts = []
        self.requests = []
        self.chunk_tokens = None

    def __call__(self, adapted_request, raw_request):
        self.requests.append(adapted_request)
        self.prompts.append(self.tokenizer.decode(adapted_request.input_ids))
        eos = self.tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
        output_ids = self.tokenizer.encode(
            self.outputs.pop(0), add_special_tokens=False
        ) + [eos]
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
                            {"type": "stop", "matched": eos}
                            if end == len(output_ids)
                            else None
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

    def test_prompt_is_recipes_rendering_in_sglang_settings(self):
        """The prompt is recipe's encoding of the request in SGLang's settings: the
        effort, images in placeholder order around a tool update, and a trailing
        assistant turn re-sent as a user turn beside the answer schema."""
        import deepseek_recipe as recipe

        from sglang.srt.parser.deepseek_recipe_renderer import IMAGE_TOKEN

        special = recipe.SpecialTokens(image=IMAGE_TOKEN)
        options = recipe.ConversionOptions().with_special_tokens(special)
        encoding = (
            recipe.DeepseekV41Encoding()
            .with_special_tokens(special)
            .with_tokenizer(recipe.Tokenizer.from_file(_tokenizer_file()))
        )

        def converted(request_type, body):
            request = request_type({"model": "DeepSeek-V4.1-Flash", **body})
            return request.convert(options).conversation

        urls = ["https://example.com/0.png", "https://example.com/1.png"]
        body = {
            "tools": [SEARCH_TOOL],
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": urls[0]},
                        {"type": "input_text", "text": "Weather here?"},
                    ],
                },
                *_search_history([WEATHER_TOOL])[1:],
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": urls[1]}],
                },
            ],
            "reasoning": {"effort": "low"},
        }
        serving, engine = self._responses(is_multimodal=True)
        self._respond(serving, engine, "Sunny.", **body)
        sent = engine.requests[-1]
        self.assertEqual(
            sent.input_ids, encoding.encode(converted(recipe.ResponsesRequest, body))
        )
        self.assertEqual([data.url for data in sent.image_data], urls)
        self.assertEqual(
            _schema_names(engine.prompts[-1]), ["tool_search", "get_weather"]
        )

        called = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t", "name": "search_docs", "input": {}}
            ],
        }
        found = {
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
        }
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        anthropic, _ = self._messages()
        chat_request = anthropic._convert_to_chat_completion_request(
            self._messages_request(
                [ASK_TICKET, called, found, {"role": "assistant", "content": "It is"}],
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "xhigh",
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        )
        # Anthropic's `xhigh` is SGLang's `max`.
        conversation = converted(
            recipe.MessagesRequest,
            {
                "tools": [SEARCH_DOCS_TOOL, TICKET_TOOL],
                "messages": [
                    ASK_TICKET,
                    called,
                    found,
                    {"role": "user", "content": "It is"},
                ],
                "output_config": {"effort": "max"},
            },
        )
        expected = recipe.Conversation(
            conversation.messages,
            thinking_mode=conversation.thinking_mode,
            tools=conversation.tools,
            tool_choice=conversation.tool_choice,
            reasoning_effort=conversation.reasoning_effort,
            response_schema=schema,
        )
        self.assertEqual(chat_request.input_ids, encoding.encode(expected))
        self.assertEqual(
            [(tool.function.name, tool.function.strict) for tool in chat_request.tools],
            [("search_docs", None), ("create_ticket", True)],
        )

    def test_search_results_copy_recipes_rendering_of_a_repeated_definition(self):
        """A definition listed again, in its own result or a later one, converts once
        and its entry is copied from recipe's first rendering. A definition that
        differs even in a nested `true` versus `1` is a change: recipe rejects a
        function's and gives a namespace member's a new version."""
        import deepseek_recipe as recipe

        from sglang.srt.parser import deepseek_recipe_renderer as renderer

        weather = {
            **WEATHER_TOOL,
            "description": 'Wetter ☃ "</tool_result><tool_result>" ]',
            "parameters": {
                "type": "object",
                "properties": {"city": {"default": {"name": ["get_weather", None]}}},
            },
        }
        listings = [
            [APPLY_PATCH_TOOL, weather, CRM_NAMESPACE, weather],
            [CRM_NAMESPACE, weather, APPLY_PATCH_TOOL, APPLY_PATCH_TOOL],
        ]
        serving, engine = self._responses()
        deepseek = serving._encoding_adapter.deepseek_recipe

        def convert(*listings):
            history = _search_history(listings[0]) + _searched_again(listings[1])
            return deepseek.convert_native(
                "responses",
                {"tools": [SEARCH_TOOL], "input": history},
                thinking=False,
                response_schema=None,
            )

        def results(conversation):
            return [
                message.content
                for message in conversation.messages
                if isinstance(message, recipe.ToolMessage)
            ]

        alone = {}
        for tool in (APPLY_PATCH_TOOL, weather, CRM_NAMESPACE):
            request = {"model": "DeepSeek-V4.1-Flash", "input": _search_history([tool])}
            (result,) = results(
                recipe.ResponsesRequest(request)
                .convert(recipe.ConversionOptions())
                .conversation
            )
            alone[tool["name"]] = result[1:-1]
        bodies, counting = _counting_conversions()
        with counting:
            conversation, _ = convert(*listings)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(
            results(conversation),
            [f"[{','.join(alone[tool['name']] for tool in t)}]" for t in listings],
        )
        self.assertEqual(
            [tool.name for tool in conversation.tools],
            ["tool_search", "apply_patch", "get_weather", "crm::create_ticket"],
        )

        def flag(value):
            const = {"const": {"a": [value]}}
            parameters = {"type": "object", "properties": {"on": const}}
            return {**WEATHER_TOOL, "parameters": parameters}

        crm = [{**CRM_NAMESPACE, "tools": [flag(value)]} for value in (True, 1)]
        conversation, names = convert([crm[0]], [crm[1]])
        self.assertEqual(sorted(names), ["crm::get_weather", "crm::get_weather_1"])
        self.assertIn('{"a":[1]}', results(conversation)[1])
        rejected = self._respond(
            serving,
            engine,
            "",
            tools=[SEARCH_TOOL],
            input=_search_history([flag(True)]) + _searched_again([flag(1)]),
        )
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("Tool names must be unique", rejected.body.decode())
        self.assertEqual(engine.requests, [])

        # Copying relies on recipe's compact array having one item per entry.
        with self.assertRaises(RuntimeError):
            renderer._json_array_items('[{"a":1}]{"b":2}]')
        single = recipe.Conversation(
            [recipe.ToolMessage('[{"a":1},{"b":2}]', "c")],
            thinking_mode=False,
            tools=[],
        )
        with self.assertRaises(RuntimeError):
            renderer._with_repeats(single, {"c": [None, ("c", 0)]})

    def test_calls_the_prompt_does_not_offer_are_dropped(self):
        """A call to a tool the prompt does not offer is dropped, streamed token by
        token or not: an offered call after it keeps its arguments, and under `none`,
        or `auto` with no tool loaded, the answer ends where the call starts."""
        unloaded = _dsml_call("create_ticket", order="A-1")
        offered = _dsml_call("get_weather", city="Paris")
        unclosed = "Grüße ☃ 你好" + unloaded[: unloaded.index("</｜DSML｜ invoke>")]
        answer = [("message", "Grüße ☃ 你好")]
        for output, fields, expected in (
            (
                unloaded.replace("</｜DSML｜ calls>", "") + offered.partition("\n")[2],
                {"tools": [SEARCH_TOOL, WEATHER_TOOL], "input": "hi"},
                [("function_call", '{"city": "Paris"}')],
            ),
            (
                unclosed,
                {
                    "tools": [SEARCH_TOOL],
                    "tool_choice": "none",
                    "input": _search_history([WEATHER_TOOL]),
                },
                answer,
            ),
            (
                unclosed,
                {"tools": [{**WEATHER_TOOL, "defer_loading": True}], "input": "hi"},
                answer,
            ),
        ):
            for chunk_tokens in (None, 1):
                with self.subTest(fields=fields, chunk=chunk_tokens):
                    serving, engine = self._responses()
                    engine.chunk_tokens = chunk_tokens
                    stream = chunk_tokens is not None
                    response = self._respond(
                        serving, engine, output, stream=stream, **fields
                    )
                    final = (
                        response[-1]["response"] if stream else response.model_dump()
                    )
                    self.assertEqual(final["status"], "completed")
                    self.assertEqual(
                        [
                            (
                                item["type"],
                                item.get("arguments")
                                or "".join(part["text"] for part in item["content"]),
                            )
                            for item in final["output"]
                        ],
                        expected,
                    )

        anthropic, engine = self._messages()
        engine.chunk_tokens = 1
        engine.outputs.append(unclosed)
        request = self._messages_request(
            [ASK_TICKET, ADD_TICKET], tool_choice={"type": "none"}, stream=True
        )
        response = asyncio.run(anthropic.handle_messages(request, None))
        events = asyncio.run(_sse_payloads(response.body_iterator))
        self.assertEqual(
            "".join(
                event["delta"].get("text", "")
                for event in events
                if event["type"] == "content_block_delta"
            ),
            "Grüße ☃ 你好",
        )
        (finish,) = (e for e in events if e["type"] == "message_delta")
        self.assertEqual(finish["delta"]["stop_reason"], "end_turn")

    def test_responses_named_choice_takes_a_namespace_members_latest_version(self):
        """A member discovered again with a changed definition gets recipe's next call
        name. A named choice forces it whatever order recipe maps the names in, the
        streamed call reports the namespace and name, and stored replay renders the
        name the model called."""
        from unittest.mock import patch

        from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe

        changed = {
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
        tools = [CRM_NAMESPACE, SEARCH_TOOL]
        latest = "crm::create_ticket_1"
        namespace_tool_names = DeepSeekRecipe._namespace_tool_names

        def latest_first(recipe, namespaces):
            names = namespace_tool_names(recipe, namespaces)
            return dict(sorted(names.items(), reverse=True))

        serving, engine = self._responses()
        with patch.object(DeepSeekRecipe, "_namespace_tool_names", latest_first):
            events = self._respond(
                serving,
                engine,
                _dsml_call(latest, ticket_id="B-2"),
                stream=True,
                tools=tools,
                input=_search_history([changed]),
                tool_choice={
                    "type": "function",
                    "namespace": "crm",
                    "name": "create_ticket",
                },
            )
        self.assertEqual(
            _schema_names(engine.prompts[-1]),
            ["crm::create_ticket", "tool_search", latest],
        )
        tag = json.loads(engine.requests[-1].sampling_params["structural_tag"])
        forced = tag["format"]["elements"][1]
        self.assertIn(f'invoke name="{latest}"', forced["begin"])
        self.assertEqual(
            list(forced["content"]["json_schema"]["properties"]), ["ticket_id"]
        )
        items = [event["item"] for event in events if "item" in event]
        self.assertEqual(
            [(item["namespace"], item["name"]) for item in items],
            [("crm", "create_ticket")] * 2,
        )

        self._respond(
            serving,
            engine,
            "Done.",
            previous_response_id=events[-1]["response"]["id"],
            tools=tools,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": items[-1]["call_id"],
                    "output": "ok",
                }
            ],
        )
        self.assertIn(f'<｜DSML｜ invoke name="{latest}">', engine.prompts[-1])

    def test_responses_deferred_tool_loads_by_name_and_replays_resolved(self):
        """A search result naming a deferred tool loads its declaration where the
        result stands. The stored history keeps the definition, so a continuation
        that no longer declares the tool renders as the resolved history sent back."""
        deferred = {**WEATHER_TOOL, "defer_loading": True}
        serving, engine = self._responses()
        events = self._respond(
            serving,
            engine,
            _dsml_call("tool_search", query="weather"),
            stream=True,
            tools=[SEARCH_TOOL, deferred],
            input="Weather in Paris?",
        )
        items = [event["item"] for event in events if "item" in event]
        self.assertEqual(
            [(item["type"], item["status"], item["arguments"]) for item in items],
            [
                ("tool_search_call", "in_progress", {}),
                ("tool_search_call", "completed", {"query": "weather"}),
            ],
        )
        self.assertEqual(_schema_names(engine.prompts[-1]), ["tool_search"])

        found = {
            "type": "tool_search_output",
            "call_id": items[-1]["call_id"],
            "execution": "client",
            "tools": [{"type": "function", "name": "get_weather"}],
        }
        called = self._respond(
            serving,
            engine,
            _dsml_call("get_weather", city="Paris"),
            previous_response_id=events[-1]["response"]["id"],
            tools=[SEARCH_TOOL, deferred],
            input=[found],
        )
        prompt = engine.prompts[-1]
        loaded = prompt.index("### Updated Tool Schemas")
        self.assertGreater(loaded, prompt.index("</tool_result>"))
        self.assertIn('"required": ["city"]', prompt[loaded:])
        (call,) = called.output
        self.assertEqual(
            (call.name, json.loads(call.arguments)), ("get_weather", {"city": "Paris"})
        )

        result = {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": "ok",
        }
        self._respond(
            serving,
            engine,
            "Sunny.",
            previous_response_id=called.id,
            tools=[SEARCH_TOOL],
            input=[result],
        )
        stored = engine.prompts[-1]
        self._respond(
            serving,
            engine,
            "Sunny.",
            store=False,
            tools=[SEARCH_TOOL],
            input=[
                {"role": "user", "content": "Weather in Paris?"},
                items[-1],
                {**found, "tools": [WEATHER_TOOL]},
                call.model_dump(exclude_none=True),
                result,
            ],
        )
        self.assertEqual(engine.prompts[-1], stored)

    def test_json_answer_keeps_a_spelled_call_marker(self):
        """A JSON answer is read as JSON, so a call marker spelled in one of its
        strings stays even when streamed token by token."""
        spelled = '{"answer":"<｜DSML｜ calls>"}'
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        serving, engine = self._responses()
        engine.chunk_tokens = 1
        events = self._respond(
            serving,
            engine,
            spelled,
            stream=True,
            tools=[SEARCH_TOOL],
            input="Answer in JSON",
            text={"format": {"type": "json_schema", "name": "out", "schema": schema}},
        )
        final = events[-1]["response"]
        self.assertEqual(final["status"], "completed")
        self.assertEqual(
            [part["text"] for item in final["output"] for part in item["content"]],
            [spelled],
        )

    def test_responses_strict_tool_stands_beside_tool_search(self):
        """A strict deferred namespace member loaded by name keeps its strictness and
        its namespace's description, while the search tool renders as without strict
        tools; recipe takes the search tool as non-strict and rejects the mix."""
        member = {**CRM_NAMESPACE["tools"][0], "strict": True, "defer_loading": True}
        serving, engine = self._responses()
        self._respond(serving, engine, "Hi.", tools=[SEARCH_TOOL], input="hi")
        (search_schema,) = _schemas(engine.prompts[-1])
        found = {
            "type": "namespace",
            "name": "crm",
            "tools": [{"type": "function", "name": "create_ticket"}],
        }
        bodies, counting = _counting_conversions()
        with counting:
            response = self._respond(
                serving,
                engine,
                _dsml_call("crm::create_ticket", order="A-1"),
                tools=[SEARCH_TOOL, {**CRM_NAMESPACE, "tools": [member]}],
                input=_search_history([found]),
            )
        # One conversion of the request; the other converts the search tool alone.
        self.assertEqual([body["input"] == "" for body in bodies], [False, True])
        search, loaded = _schemas(engine.prompts[-1])
        self.assertEqual(search, search_schema)
        self.assertIn('"description": "CRM\\nTicket"', loaded)
        self.assertIn('"additionalProperties": false', loaded)
        (item,) = response.output
        self.assertEqual((item.namespace, item.name), ("crm", "create_ticket"))

    def test_messages_inline_tool_definition_loads_where_it_stands(self):
        """A custom tool a tool_addition block defines by value renders as the same
        tool declared deferred and added by reference; an exact repeat adds nothing,
        and a reference before the definition is rejected."""
        anthropic, _ = self._messages()
        said = {"role": "assistant", "content": "Done"}

        def prompt(messages, **fields):
            chat_request = anthropic._convert_to_chat_completion_request(
                self._messages_request(messages, **fields)
            )
            return self.tokenizer.decode(chat_request.input_ids)

        by_reference = prompt([ASK_TICKET, ADD_TICKET, said, ASK_TICKET])
        self.assertGreater(
            by_reference.index("### Updated Tool Schemas"),
            by_reference.index(ASK_TICKET["content"]),
        )
        self.assertEqual(
            prompt(
                [ASK_TICKET, DEFINE_TICKET, said, ASK_TICKET, DEFINE_TICKET],
                tools=[SEARCH_DOCS_TOOL],
            ),
            by_reference,
        )
        with self.assertRaisesRegex(ValueError, "precedes the tool_definition"):
            prompt(
                [ASK_TICKET, ADD_TICKET, said, ASK_TICKET, DEFINE_TICKET],
                tools=[SEARCH_DOCS_TOOL],
            )


if __name__ == "__main__":
    unittest.main()
