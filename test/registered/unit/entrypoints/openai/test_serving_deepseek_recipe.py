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

    def test_logprob_bytes_are_the_token_bytes(self):
        """The vocabulary is byte-level even though its first ids are special tokens,
        also behind fastokens' backend, whose decoder carries no type, and when the
        ByteLevel decoder sits in a Sequence (LFM2)."""
        from tokenizers import decoders
        from transformers import PreTrainedTokenizerFast

        for backend in ("huggingface", "fastokens", "sequence decoder"):
            with self.subTest(backend):
                tokenizer = self.tokenizer
                if backend == "sequence decoder":
                    tokenizer = PreTrainedTokenizerFast(
                        tokenizer_file=_tokenizer_file()
                    )
                    tokenizer.backend_tokenizer.decoder = decoders.Sequence(
                        [decoders.ByteLevel()]
                    )
                if backend == "fastokens":
                    try:
                        from sglang.srt.hf_transformers.tokenizer import (
                            _shim_fastokens_backend,
                        )

                        tokenizer = _shim_fastokens_backend(
                            PreTrainedTokenizerFast(tokenizer_file=_tokenizer_file())
                        )
                    except ImportError:
                        self.skipTest("fastokens is not installed")
                chat = _serving(self, tokenizer)
                [logprob] = chat._build_token_logprobs_from_raw(
                    [(-0.1, self._token("ð"), "�")], None
                )
                self.assertEqual(logprob.bytes, [0xF0])

    def test_only_byte_preserving_decoder_sequences_use_byte_table(self):
        from tokenizers import Tokenizer, decoders, models
        from transformers import PreTrainedTokenizerFast

        from sglang.srt.entrypoints.openai import utils

        def make_tokenizer(decoder):
            backend = Tokenizer(models.BPE(vocab={"ð": 0, "é": 1}, merges=[]))
            backend.decoder = decoder
            return PreTrainedTokenizerFast(tokenizer_object=backend)

        utils._BYTE_LEVEL_VERDICTS.clear()
        nested = make_tokenizer(
            decoders.Sequence([decoders.Sequence([decoders.ByteLevel()])])
        )
        self.assertEqual(utils.token_id_to_bytes(nested, 0), [0xF0])

        transforming = make_tokenizer(
            decoders.Sequence([decoders.Replace("é", "Ã©"), decoders.ByteLevel()])
        )
        self.assertIsNone(utils.token_id_to_bytes(transforming, 1))

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
STRING = {"type": "string"}
WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather",
    "parameters": {
        "type": "object",
        "properties": {"city": STRING, "unit": STRING},
        "required": ["city"],
    },
}
DEFERRED_WEATHER = {**WEATHER_TOOL, "defer_loading": True}
NAMED_WEATHER = {"type": "function", "name": "get_weather"}
APPLY_PATCH_TOOL = {"type": "custom", "name": "apply_patch"}
SHELL_TOOL = {"type": "custom", "name": "shell_exec", "description": "Run a command"}
SEARCH_DOCS_TOOL = {
    "name": "search_docs",
    "description": "Search",
    "input_schema": {"type": "object", "properties": {"q": STRING}},
}
TICKET_TOOL = {
    "name": "create_ticket",
    "description": "Ticket",
    "input_schema": {
        "type": "object",
        "properties": {"order": STRING},
        "required": ["order"],
    },
    "strict": True,
    "defer_loading": True,
}
ASK_TICKET = {"role": "user", "content": "Open a ticket for A-1"}
SAID = {"role": "assistant", "content": "Done"}
SEARCHED_DOCS = {
    "role": "assistant",
    "content": [{"type": "tool_use", "id": "t", "name": "search_docs", "input": {}}],
}


def _function(name, **properties):
    parameters = {"type": "object", "properties": properties}
    return {"type": "function", "name": name, "parameters": parameters}


def _crm(*members, **fields):
    return {"type": "namespace", "name": "crm", "tools": list(members), **fields}


TICKET_MEMBER = {**_function("create_ticket", order=STRING), "description": "Ticket"}
CRM_NAMESPACE = _crm(TICKET_MEMBER, description="CRM")


def _addition(tool):
    return {"role": "system", "content": [{"type": "tool_addition", "tool": tool}]}


ADD_TICKET = _addition({"type": "tool_reference", "name": "create_ticket"})
DEFINE_TICKET = _addition(
    {
        "type": "tool_definition",
        "definition": {k: v for k, v in TICKET_TOOL.items() if k != "defer_loading"},
    }
)


def _tool_result(content):
    block = {"type": "tool_result", "tool_use_id": "t", "content": content}
    return {"role": "user", "content": [block]}


def _search_history(found_tools, call_id="search"):
    """A question, a client search call and the result that found `found_tools`."""
    search = {"call_id": call_id, "execution": "client"}
    return [
        {"role": "user", "content": "Weather in Paris?"},
        {"type": "tool_search_call", "arguments": {"query": "weather"}, **search},
        {"type": "tool_search_output", "tools": found_tools, **search},
    ]


def _result(call_id):
    return {"type": "function_call_output", "call_id": call_id, "output": "ok"}


def _invoke(name, **arguments):
    parameters = "".join(
        f'<｜DSML｜ parameter name="{key}" string="true">{value}</｜DSML｜ parameter>\n'
        for key, value in arguments.items()
    )
    return f'<｜DSML｜ invoke name="{name}">\n{parameters}</｜DSML｜ invoke>\n'


def _dsml_calls(*invokes):
    return "<｜DSML｜ calls>\n" + "".join(invokes) + "</｜DSML｜ calls>"


def _dsml_call(name, **arguments):
    return _dsml_calls(_invoke(name, **arguments))


def _schemas(prompt):
    """The schema lines the prompt shows, initially or as updates."""
    return re.findall(r'^\{"name": .*$', prompt, flags=re.MULTILINE)


def _schema_names(prompt):
    return " ".join(re.findall(r'^\{"name": "([^"]+)"', prompt, flags=re.MULTILINE))


async def _sse_payloads(stream):
    return [
        json.loads(line.removeprefix("data: "))
        async for chunk in stream
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ]


def _recipe_v41():
    """recipe's own conversion options and V4.1 encoding, with the served image
    token."""
    import deepseek_recipe as recipe

    from sglang.srt.parser.deepseek_recipe_renderer import IMAGE_TOKEN

    special = recipe.SpecialTokens(image=IMAGE_TOKEN)
    options = recipe.ConversionOptions().with_special_tokens(special)
    encoding = (
        recipe.DeepseekV41Encoding()
        .with_special_tokens(special)
        .with_tokenizer(recipe.Tokenizer.from_file(_tokenizer_file()))
    )
    return options, encoding


class _Engine:
    """Answers each request with its next scripted output and keeps its prompt;
    `chunk_tokens` splits the output into chunks of that many tokens, and
    `before_output` is awaited once, ahead of the next output."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.outputs = []
        self.prompts = []
        self.requests = []
        self.chunk_tokens = None
        self.before_output = None

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
            before, self.before_output = self.before_output, None
            if before is not None:
                await before()
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
        """`respond(output, **fields)`, the response to a Responses request the
        engine answers with `output`, a streamed one as its events; then the engine
        and the handler."""
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
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

        def respond(output, **fields):
            engine.outputs.append(output)
            request = ResponsesRequest(model="dummy", **fields)
            response = asyncio.run(serving.create_responses(request))
            return asyncio.run(_sse_payloads(response)) if request.stream else response

        return respond, engine, serving

    def _queued(self, serving, **fields):
        """A background Responses request, stored as received and never run."""
        from unittest.mock import AsyncMock, patch

        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        request = ResponsesRequest(model="dummy", background=True, **fields)
        with patch.object(serving, "_run_background_request", AsyncMock()):
            queued = asyncio.run(serving.create_responses(request))
        self.assertEqual(queued.status, "queued")
        return queued.id

    def _messages(self, **model):
        from sglang.srt.entrypoints.anthropic.serving import AnthropicServing

        engine = _Engine(self.tokenizer)
        chat = _serving(self, self.tokenizer, engine=engine, **model)
        return AnthropicServing(chat), engine

    def _messages_request(self, messages, **fields):
        from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

        fields = {"tools": [SEARCH_DOCS_TOOL, TICKET_TOOL], **fields}
        return AnthropicMessagesRequest(
            model="dummy", max_tokens=64, messages=messages, **fields
        )

    def _assert_rejected(self, response, param, error):
        self.assertEqual(response.status_code, 400)
        rejected = json.loads(response.body)["error"]
        self.assertEqual(rejected["param"], param)
        self.assertIn(error, rejected["message"])

    def test_prompt_is_recipes_rendering_in_sglang_settings(self):
        """The prompt is recipe's encoding of the request in SGLang's settings: the
        effort, images in placeholder order around a tool update, and a trailing
        assistant turn re-sent as a user turn beside the answer schema."""
        import deepseek_recipe as recipe

        options, encoding = _recipe_v41()

        def converted(request_type, body):
            request = request_type({"model": "DeepSeek-V4.1-Flash", **body})
            return request.convert(options).conversation

        def image(url):
            return {"type": "input_image", "image_url": url}

        urls = ["https://example.com/0.png", "https://example.com/1.png"]
        asked = [image(urls[0]), {"type": "input_text", "text": "Weather here?"}]
        body = {
            "tools": [SEARCH_TOOL],
            "input": [
                {"role": "user", "content": asked},
                *_search_history([WEATHER_TOOL])[1:],
                {"role": "user", "content": [image(urls[1])]},
            ],
            "reasoning": {"effort": "low"},
        }
        respond, engine, _ = self._responses(is_multimodal=True)
        respond("Sunny.", **body)
        sent = engine.requests[-1]
        self.assertEqual(
            sent.input_ids, encoding.encode(converted(recipe.ResponsesRequest, body))
        )
        self.assertEqual([data.url for data in sent.image_data], urls)
        self.assertEqual(_schema_names(engine.prompts[-1]), "tool_search get_weather")

        found = _tool_result([{"type": "tool_reference", "tool_name": "create_ticket"}])
        history = [ASK_TICKET, SEARCHED_DOCS, found]
        schema = {"type": "object", "properties": {"answer": STRING}}
        anthropic, _ = self._messages()
        chat_request = anthropic._convert_to_chat_completion_request(
            self._messages_request(
                [*history, {"role": "assistant", "content": "It is"}],
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
                "messages": [*history, {"role": "user", "content": "It is"}],
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

    def test_responses_search_results_reuse_exact_definitions(self):
        """Search results that repeat a definition exactly, in one result, across
        results, after its declaration or in parallel results in either order,
        render as recipe renders the request, and reported call names follow the
        members recipe registers: `true` for `1` is a new version, `0.0` for
        `-0.0` is not, and a member named like a next version keeps its name."""
        import deepseek_recipe as recipe

        def lookup(default, name="lookup"):
            return _function(name, q={"type": "string", "default": default})

        sibling = lookup(1, name="lookup_1")
        ticket = {**WEATHER_TOOL, "name": "create_ticket"}
        first = _crm(lookup(1), sibling, sibling, lookup(-0.0, "rate"))
        again = _crm(sibling, lookup(True), lookup(0.0, "rate"))
        body = {
            "tools": [SEARCH_TOOL, WEATHER_TOOL, _crm(lookup(1))],
            "input": _search_history([WEATHER_TOOL, first, ticket])
            + _search_history([ticket, again], "again")[1:],
        }
        respond, engine, _ = self._responses()
        response = respond(
            _dsml_calls(
                _invoke("crm::lookup_1", q="A-1"), _invoke("crm::lookup_2", q="B-2")
            ),
            **body,
        )
        options, encoding = _recipe_v41()

        def recipes_prompt(body):
            request = recipe.ResponsesRequest({"model": "DeepSeek-V4.1-Flash", **body})
            converted = request.convert(options.with_default_thinking_mode(False))
            return encoding.encode(converted.conversation)

        self.assertEqual(engine.requests[-1].input_ids, recipes_prompt(body))
        self.assertEqual(
            _schema_names(engine.prompts[-1]),
            "tool_search get_weather crm::lookup crm::lookup_1 crm::rate "
            "create_ticket crm::lookup_2",
        )
        self.assertEqual(
            [(item.namespace, item.name) for item in response.output],
            [("crm", "lookup_1"), ("crm", "lookup")],
        )

        ask, search, found = _search_history([WEATHER_TOOL])
        other = {"call_id": "again"}
        for results in ([found, {**found, **other}], [{**found, **other}, found]):
            parallel = {
                "tools": [SEARCH_TOOL],
                "input": [ask, search, {**search, **other}, *results],
            }
            respond("Sunny.", **parallel)
            self.assertEqual(engine.requests[-1].input_ids, recipes_prompt(parallel))
            self.assertEqual(
                _schema_names(engine.prompts[-1]), "tool_search get_weather"
            )

    def test_responses_named_choice_takes_a_namespace_members_latest_version(self):
        """A member discovered again with a changed definition (v2) gets recipe's next
        call name, and relisting the declared definition (v1) reuses its first
        registration. A named choice forces the last registered distinct definition,
        v2, whatever order recipe maps the names in; the streamed call reports the
        namespace and name, and stored replay renders the name the model called."""
        from unittest.mock import patch

        from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe

        changed = _crm(_function("create_ticket", ticket_id=STRING))
        tools = [CRM_NAMESPACE, SEARCH_TOOL]
        latest = "crm::create_ticket_1"
        convert_native = DeepSeekRecipe.convert_native

        def latest_first(recipe, *args, **kwargs):
            conversation, names = convert_native(recipe, *args, **kwargs)
            return conversation, dict(sorted(names.items(), reverse=True))

        respond, engine, _ = self._responses()
        choice = {"type": "function", "namespace": "crm", "name": "create_ticket"}
        with patch.object(DeepSeekRecipe, "convert_native", latest_first):
            events = respond(
                _dsml_call(latest, ticket_id="B-2"),
                stream=True,
                tools=tools,
                input=_search_history([changed])
                + _search_history([CRM_NAMESPACE], "again")[1:],
                tool_choice=choice,
            )
        self.assertEqual(
            _schema_names(engine.prompts[-1]),
            f"crm::create_ticket tool_search {latest}",
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
        respond(
            "Done.",
            previous_response_id=events[-1]["response"]["id"],
            tools=tools,
            input=[_result(items[-1]["call_id"])],
        )
        self.assertIn(f'<｜DSML｜ invoke name="{latest}">', engine.prompts[-1])

    def test_responses_stored_namespace_call_keeps_what_it_called(self):
        """A continuation that declares other tools can move the name recipe gave a
        stored call. A name that now stands for another definition (`true` for `1`,
        `1.0` for `1`, but not `0.0` for `-0.0`) or another member, or a definition
        that lost its name, is a client error before generation, also through a
        background response that has not run. A member really named like a numbered
        version replays, as a call to a dropped tool once the request drops it."""

        def x(const):
            return _function("x", a={"const": const})

        def versions(old, new):
            return (
                _search_history([_crm(x(new))])
                + _search_history([_crm(x(old))], "again")[1:]
            )

        def stored_call(declared, history, name="crm::x_1"):
            respond, engine, serving = self._responses()
            response = respond(
                _dsml_call(name, q="1"), tools=[*declared, SEARCH_TOOL], input=history
            )
            (item,) = response.output
            return respond, engine, serving, response.id, item

        def continued(respond, stored, item, tools):
            return respond(
                "Done.",
                previous_response_id=stored,
                tools=[*tools, SEARCH_TOOL],
                input=[_result(item.call_id)],
            )

        def assert_rejected(response, engine, error):
            self._assert_rejected(response, "input", f"as 'crm::x_1', which {error}")
            self.assertEqual(len(engine.requests), 1)

        numbered = _function("x_1", c=STRING)
        ask, found = _search_history([])[:1], _search_history([_crm(x(2))])
        gives = "this request's tools and search results give"
        lost = "this request's tools and search results no longer load"
        for name, declared, history, member, later, error in (
            ("true for 1", [x(1)], versions(1, True), "x", [], f"{gives} a different"),
            ("1.0 for 1", [x(1)], versions(1, 1.0), "x", [], f"{gives} a different"),
            ("another member", [x(1)], found, "x", [numbered], f"{gives} to crm.x_1"),
            ("lost its name", [x(1)], found, "x", [], lost),
            ("numbered member", [x(1), numbered], ask, "x_1", [x(1), numbered], None),
            ("numbered member dropped", [x(1), numbered], ask, "x_1", [x(1)], None),
        ):
            with self.subTest(name):
                respond, engine, _, stored, item = stored_call(
                    [_crm(*declared)], history
                )
                self.assertEqual(item.name, member)
                response = continued(
                    respond, stored, item, [_crm(*later)] * bool(later)
                )
                if error is not None:
                    assert_rejected(response, engine, error)
                    continue
                self.assertEqual(response.status, "completed")
                self.assertIn('<｜DSML｜ invoke name="crm::x_1">', engine.prompts[-1])

        respond, engine, serving, stored, item = stored_call(
            [_crm(x(1))], versions(1, True)
        )
        queued = self._queued(
            serving,
            previous_response_id=stored,
            tools=[_crm(x(1)), SEARCH_TOOL],
            input=[_result(item.call_id)],
        )
        response = respond(
            "Done.", previous_response_id=queued, tools=[SEARCH_TOOL], input="And?"
        )
        assert_rejected(response, engine, f"{gives} a different definition")

        # Dropping the declaration leaves the name to the search result's exact
        # repeat of it, which spells the zero with the other sign.
        respond, engine, _, stored, item = stored_call(
            [_crm(x(-0.0))], _search_history([_crm(x(0.0))]), name="crm::x"
        )
        self.assertEqual(continued(respond, stored, item, []).status, "completed")
        self.assertIn('"const": 0.0', engine.prompts[-1])

    def test_responses_stored_calls_are_those_of_the_stored_history(self):
        """A stored response replays the history its prompt read. A background
        response it continued, completing with a namespace call while it ran, adds
        neither the call nor its record; a response stored again under a reused id
        keeps no record of the calls it replaced."""
        from unittest.mock import patch

        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest

        changed = _crm({**TICKET_MEMBER, "description": "Other"}, description="CRM")
        called = _dsml_call("crm::create_ticket", order="A-1")
        fields = {"tools": [CRM_NAMESPACE], "input": "Open a ticket"}

        respond, engine, serving = self._responses()
        run, held = serving._run_background_request, []

        def hold(*args, **kwargs):
            held.append((args, kwargs))
            return asyncio.sleep(0)

        async def parent_completes():
            args, kwargs = held.pop()
            engine.outputs.append(called)
            await run(*args, **kwargs)

        parent = ResponsesRequest(model="dummy", background=True, **fields)
        with patch.object(serving, "_run_background_request", hold):
            queued = asyncio.run(serving.create_responses(parent))
        engine.before_output = parent_completes
        child = respond(
            "One moment.",
            previous_response_id=queued.id,
            tools=[CRM_NAMESPACE],
            input="Never mind",
        )
        (parent_call,) = serving.response_store[queued.id].output
        self.assertEqual(
            (parent_call.namespace, parent_call.status), ("crm", "completed")
        )

        respond_again, engine_again, _ = self._responses()
        respond_again(called, request_id="resp_reused", **fields)
        respond_again("Hello.", request_id="resp_reused", input="Hi")
        for name, respond, engine, stored in (
            ("background parent completed", respond, engine, child.id),
            ("request id reused", respond_again, engine_again, "resp_reused"),
        ):
            with self.subTest(name):
                continued = respond(
                    "Done.", previous_response_id=stored, tools=[changed], input="Go on"
                )
                self.assertEqual(continued.status, "completed")
                self.assertNotIn('invoke name="crm::create_ticket"', engine.prompts[-1])

    def test_responses_deferred_tools_load_by_name_and_replay_as_loaded(self):
        """A search result naming a deferred tool loads its declaration where the
        result stands, and the stored result keeps that definition: a continuation
        that drops the declaration or redeclares it strict, or that continues a
        background response not yet run, renders the resolved history. A strict
        deferred namespace member loads the same way in one conversion, beside a
        search tool that renders as without strict tools."""
        from unittest.mock import patch

        from sglang.srt.parser.deepseek_recipe_renderer import DeepSeekRecipe

        tools = [SEARCH_TOOL, DEFERRED_WEATHER]
        respond, engine, serving = self._responses()
        events = respond(
            _dsml_call("tool_search", query="weather"),
            stream=True,
            tools=tools,
            input="Weather in Paris?",
        )
        search = [event["item"] for event in events if "item" in event]
        self.assertEqual(
            [(item["type"], item["status"], item["arguments"]) for item in search],
            [
                ("tool_search_call", "in_progress", {}),
                ("tool_search_call", "completed", {"query": "weather"}),
            ],
        )
        (search_schema,) = _schemas(engine.prompts[-1])

        found = {
            **_search_history([NAMED_WEATHER])[2],
            "call_id": search[-1]["call_id"],
        }
        called = respond(
            _dsml_call("get_weather", city="Paris"),
            previous_response_id=events[-1]["response"]["id"],
            tools=tools,
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

        result = _result(call.call_id)
        for later in ([], [{**DEFERRED_WEATHER, "strict": True}]):
            respond(
                "Sunny.",
                previous_response_id=called.id,
                tools=[SEARCH_TOOL, *later],
                input=[result],
            )
        sent_back = [
            {"role": "user", "content": "Weather in Paris?"},
            search[-1],
            {**found, "tools": [WEATHER_TOOL]},
            call.model_dump(exclude_none=True),
            result,
        ]
        respond("Sunny.", store=False, tools=[SEARCH_TOOL], input=sent_back)
        self.assertEqual(engine.prompts[2:], [engine.prompts[-1]] * 3)

        queued = self._queued(
            serving, tools=tools, input=_search_history([NAMED_WEATHER])
        )
        respond(
            "Sunny.", previous_response_id=queued, tools=[SEARCH_TOOL], input="And?"
        )
        self.assertEqual(len(engine.prompts), 6)
        self.assertIn('"required": ["city"]', engine.prompts[-1])

        member = {**TICKET_MEMBER, "strict": True, "defer_loading": True}
        named = _crm({"type": "function", "name": "create_ticket"})
        with patch.object(
            DeepSeekRecipe,
            "_converted",
            autospec=True,
            side_effect=DeepSeekRecipe._converted,
        ) as conversions:
            response = respond(
                _dsml_call("crm::create_ticket", order="A-1"),
                tools=[SEARCH_TOOL, _crm(member, description="CRM")],
                input=_search_history([named]),
            )
        self.assertEqual(conversions.call_count, 1)
        search_shown, loaded = _schemas(engine.prompts[-1])
        self.assertEqual(search_shown, search_schema)
        self.assertIn('"description": "CRM\\nTicket"', loaded)
        self.assertIn('"additionalProperties": false', loaded)
        (item,) = response.output
        self.assertEqual((item.namespace, item.name), ("crm", "create_ticket"))

    def test_responses_custom_tools_keep_their_kind(self):
        """A declared custom tool renders as the single-string function shim and
        stays custom beside a deferred namespace of its name: a custom choice names
        it, and its streamed custom call replays the same stored or sent back. A
        named choice of the other kind, declared or found, is rejected by SGLang."""
        ask = {"role": "user", "content": "List the files"}
        shadow = _crm(TICKET_MEMBER, name="shell_exec", defer_loading=True)
        fields = {"tools": [SEARCH_TOOL, SHELL_TOOL, shadow]}
        respond, engine, _ = self._responses()
        events = respond(
            _dsml_call("shell_exec", input="ls -l"),
            stream=True,
            input=[ask],
            tool_choice={"type": "custom", "name": "shell_exec"},
            **fields,
        )
        self.assertEqual(_schema_names(engine.prompts[-1]), "tool_search shell_exec")
        final = events[-1]["response"]
        (item,) = final["output"]
        self.assertEqual(
            (item["type"], item["name"], item["input"]),
            ("custom_tool_call", "shell_exec", "ls -l"),
        )
        result = {**_result(item["call_id"]), "type": "custom_tool_call_output"}
        respond("Done.", previous_response_id=final["id"], input=[result], **fields)
        stored = engine.prompts[-1]
        self.assertIn('<｜DSML｜ parameter name="input" string="true">ls -l<', stored)
        respond("Done.", store=False, input=[ask, item, result], **fields)
        self.assertEqual(engine.prompts[2:], [stored])

        for fields in (
            {"tools": [SEARCH_TOOL, WEATHER_TOOL, APPLY_PATCH_TOOL], "input": "hi"},
            {
                "tools": [SEARCH_TOOL],
                "input": _search_history([WEATHER_TOOL, APPLY_PATCH_TOOL]),
            },
        ):
            for kind, name in (("function", "apply_patch"), ("custom", "get_weather")):
                with self.subTest(fields=fields, kind=kind):
                    rejected = respond(
                        "", tool_choice={"type": kind, "name": name}, **fields
                    )
                    self._assert_rejected(
                        rejected, "tool_choice", f"names {name!r} as a {kind} tool"
                    )
        self.assertEqual(len(engine.prompts), 3)

    def test_responses_malformed_tools_are_client_errors(self):
        """A tool declaration or search result that is malformed, that the encoding
        cannot render, or that gives one name two kinds or two deferred definitions
        is a client error before generation, not a server error or a silent pick."""
        function = {**WEATHER_TOOL, "name": "apply_patch"}
        nameless = {"type": "custom", "defer_loading": True}
        unnamed = {**WEATHER_TOOL, "name": {"get": "weather"}}
        twice = [
            {**DEFERRED_WEATHER, "parameters": {"properties": {"q": {"const": const}}}}
            for const in (True, 1)
        ]
        respond, engine, _ = self._responses()
        for tools, found, error in (
            ([], 42, "must be a list"),
            ([nameless], [], "custom tools must have a name"),
            ([DEFERRED_WEATHER], [unnamed], "each with a name"),
            ([], [_crm(APPLY_PATCH_TOOL)], "namespace tools must list function tools"),
            ([], [SHELL_TOOL], "only the 'apply_patch' custom tool"),
            ([APPLY_PATCH_TOOL], [function], "'apply_patch' as a function tool"),
            ([], [function, APPLY_PATCH_TOOL], "'apply_patch' as a custom tool"),
            (twice, [NAMED_WEATHER], "defers 'get_weather' more than once"),
        ):
            with self.subTest(error):
                rejected = respond(
                    "", tools=[SEARCH_TOOL, *tools], input=_search_history(found)
                )
                self._assert_rejected(rejected, "tools", error)
        self.assertEqual(engine.requests, [])

    def test_output_keeps_only_what_the_prompt_allows(self):
        """A call to a tool the prompt does not offer is dropped, streamed or not: an
        offered call after it keeps its arguments, and under `none`, or `auto` with
        no tool loaded, the answer ends where the call starts. A JSON answer keeps a
        call marker spelled in one of its strings."""
        unloaded = _invoke("create_ticket", order="A-1")
        unclosed = "Grüße ☃ 你好<｜DSML｜ calls>\n" + unloaded.partition("</")[0]
        answer = [("message", "Grüße ☃ 你好")]
        spelled = '{"answer":"<｜DSML｜ calls>"}'
        schema = {"type": "object", "properties": {"answer": STRING}}
        searched = {"tools": [SEARCH_TOOL], "input": _search_history([WEATHER_TOOL])}
        loaded = {"tools": [SEARCH_TOOL, WEATHER_TOOL], "input": "hi"}
        both = _dsml_calls(unloaded, _invoke("get_weather", city="Paris"))
        called = [("function_call", '{"city": "Paris"}')]
        json_answer = {"type": "json_schema", "name": "out", "schema": schema}
        for output, chunk_tokens, fields, expected in (
            (both, None, loaded, called),
            (both, 1, loaded, called),
            (unclosed, 1, {**searched, "tool_choice": "none"}, answer),
            (unclosed, None, {"tools": [DEFERRED_WEATHER], "input": "hi"}, answer),
            (
                spelled,
                1,
                {**loaded, "text": {"format": json_answer}},
                [("message", spelled)],
            ),
        ):
            with self.subTest(fields=fields, chunk=chunk_tokens):
                respond, engine, _ = self._responses()
                engine.chunk_tokens = chunk_tokens
                stream = chunk_tokens is not None
                response = respond(output, stream=stream, **fields)
                final = response[-1]["response"] if stream else response.model_dump()
                self.assertEqual(final["status"], "completed")
                shown = [
                    (item["type"], item.get("arguments") or item["content"][0]["text"])
                    for item in final["output"]
                ]
                self.assertEqual(shown, expected)

        anthropic, engine = self._messages()
        engine.chunk_tokens, engine.outputs = 1, [unclosed]
        request = self._messages_request(
            [ASK_TICKET, ADD_TICKET], tool_choice={"type": "none"}, stream=True
        )
        response = asyncio.run(anthropic.handle_messages(request, None))
        events = asyncio.run(_sse_payloads(response.body_iterator))
        deltas = [event["delta"] for event in events if "delta" in event]
        self.assertEqual("".join(d.get("text", "") for d in deltas), "Grüße ☃ 你好")
        self.assertEqual(
            [d["stop_reason"] for d in deltas if "stop_reason" in d], ["end_turn"]
        )

    def test_messages_blocks_take_effect_where_they_stand(self):
        """A tool a tool_addition block defines by value renders as the same tool
        declared deferred and added by reference; an exact repeat adds nothing, and
        a reference before the definition is a client error. A `clear_at:
        "next_user_message"` system message renders until a user message follows,
        and is rejected where it could not be dropped cleanly. A search result
        block, given directly or in a tool result, reads as its Chat path text, and
        count_tokens counts the prompt Messages sends."""
        from sglang.srt.entrypoints.anthropic import protocol

        anthropic, engine = self._messages()

        def prompt(messages, **fields):
            chat_request = anthropic._convert_to_chat_completion_request(
                self._messages_request(messages, **fields)
            )
            return self.tokenizer.decode(chat_request.input_ids)

        again = [SAID, ASK_TICKET]
        by_reference = prompt([ASK_TICKET, ADD_TICKET, *again])
        self.assertGreater(
            by_reference.index("### Updated Tool Schemas"),
            by_reference.index(ASK_TICKET["content"]),
        )
        self.assertEqual(
            prompt(
                [ASK_TICKET, DEFINE_TICKET, *again, DEFINE_TICKET],
                tools=[SEARCH_DOCS_TOOL],
            ),
            by_reference,
        )
        self.assertEqual(
            prompt([ASK_TICKET, ADD_TICKET, *again, DEFINE_TICKET]), by_reference
        )
        early = self._messages_request(
            [ASK_TICKET, ADD_TICKET, *again, DEFINE_TICKET], tools=[SEARCH_DOCS_TOOL]
        )
        rejected = asyncio.run(anthropic.handle_messages(early, None))
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("precedes the tool_definition", rejected.body.decode())

        text = {"type": "text", "text": "REMINDER"}
        note = {"role": "system", "clear_at": "next_user_message", "content": [text]}
        answered = [SEARCHED_DOCS, _tool_result("ok")]
        for messages, rendered in (
            ([ASK_TICKET, note, SAID], True),
            ([ASK_TICKET, note, *answered], False),
            ([ASK_TICKET, {**note, "clear_at": "never"}, *answered], True),
        ):
            with self.subTest(messages=messages):
                self.assertEqual("REMINDER" in prompt(messages), rendered)
        cached = [{**text, "cache_control": {"type": "ephemeral"}}]
        for messages, error in (
            ([{**ASK_TICKET, "clear_at": "never"}], "only permitted on role 'system'"),
            ([ASK_TICKET, {**note, **ADD_TICKET}], "supports text blocks only"),
            ([ASK_TICKET, {**note, "output_config": {}}], "output_config is not"),
            ([ASK_TICKET, {**note, "content": cached}], "cache_control is not"),
            ([ASK_TICKET, note, ASK_TICKET], "must follow a user message"),
            ([ASK_TICKET, SAID, note], "must follow a user message"),
        ):
            with self.subTest(messages=messages):
                with self.assertRaisesRegex(ValueError, error):
                    prompt(messages)

        found = {
            "type": "search_result",
            "source": "https://example.com/refunds",
            "title": "Refunds",
            "content": [{"type": "text", "text": "Refunds take 5 days."}],
        }
        asked = {"role": "user", "content": [found, {"type": "text", "text": "Why?"}]}
        engine.outputs.append("5 days.")
        request = self._messages_request([asked, SEARCHED_DOCS, _tool_result([found])])
        response = asyncio.run(anthropic.handle_messages(request, None))
        self.assertEqual(json.loads(response.body)["content"][0]["text"], "5 days.")
        shown = "Title: Refunds\nSource: https://example.com/refunds\nContent: Refunds"
        self.assertEqual(engine.prompts[-1].count(shown), 2)

        count = protocol.AnthropicCountTokensRequest(
            model="dummy", messages=request.messages, tools=request.tools
        )
        counted = asyncio.run(anthropic.handle_count_tokens(count, None))
        self.assertEqual(counted.status_code, 200)
        self.assertEqual(
            json.loads(counted.body)["input_tokens"], len(engine.requests[-1].input_ids)
        )


if __name__ == "__main__":
    unittest.main()
