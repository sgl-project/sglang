import asyncio
import json
import subprocess
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils import engine_chunk, input_processor, make_serving, sync_serving

from sglang.srt.entrypoints.anthropic.protocol import AnthropicCountTokensRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.chat_input.processor import ChatInputProcessor
from sglang.srt.entrypoints.chat_input.types import (
    ChatInput,
    ChatModelConfig,
    RenderedPrompt,
    TextPrompt,
    TokenPrompt,
)
from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.chat_input_adapter import (
    from_chat_request,
    from_responses_request,
    from_tokenize_request,
    with_prepared_options,
)
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ResponsesRequest,
    TokenizeRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
from sglang.srt.parser.template_detection import ReasoningToggleConfig
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_processor(serving):
    return input_processor(serving)


class ChatInputProcessorTest(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def make_serving(self, spec=None, multimodal=False, template=None):
        serving = make_serving(is_multimodal=multimodal)
        serving.chat_encoding_spec = spec
        serving._dsv4_reasoning_effort_profile = "official" if spec == "dsv4" else None
        serving.template_manager.chat_template_name = template
        serving.template_manager.jinja_template_content_format = "openai"
        serving.default_chat_template_kwargs = {}
        tokenizer = serving.tokenizer_manager.tokenizer
        tokenizer.apply_chat_template.return_value = (
            [4, -1, 6] if spec == "kimi_k3" else "rendered prompt"
        )
        tokenizer.decode.return_value = "decoded prompt"
        return serving

    def assert_preparation_matches(self, serving, request):
        original = request.model_dump()
        tokenizer = serving.tokenizer_manager.tokenizer
        old_request = request.model_copy(deep=True)
        tokenizer.reset_mock()
        old, old_request = sync_serving(serving)._convert_to_internal_request(
            old_request
        )
        old_calls = tokenizer.mock_calls[:]
        tokenizer.reset_mock()
        chat_input = from_chat_request(request)
        new = make_processor(serving).prepare(chat_input)
        self.assertEqual(request.model_dump(), original)
        self.assertEqual(chat_input, from_chat_request(request))
        self.assertEqual(tokenizer.mock_calls, old_calls)
        self.assertEqual(
            new.prompt.to_generate_kwargs(),
            {key: getattr(old, key) for key in new.prompt.to_generate_kwargs()},
        )
        for field in (
            "image_data",
            "audio_data",
            "video_data",
            "modalities",
            "require_reasoning",
        ):
            self.assertEqual(getattr(new, field), getattr(old, field), field)
        self.assertEqual(new.stop, old.sampling_params["stop"])
        self.assertEqual(
            new.skip_special_tokens, old.sampling_params["skip_special_tokens"]
        )
        self.assertEqual(new.chat_template_kwargs, old_request.chat_template_kwargs)
        self.assertEqual(new.reasoning_effort, old_request.reasoning_effort)
        return new

    def test_existing_renderers_preserve_payloads_and_tokenizer_calls(self):
        messages = [{"role": "user", "content": "hi"}]
        variants = (
            {},
            {"reasoning_effort": "high"},
            {"input_ids": [11, 12], "stop": ["STOP"]},
            {
                "messages": messages + [{"role": "assistant", "content": "prefix"}],
                "continue_final_message": True,
            },
        )
        paths = (
            (None, False, None),
            (None, True, None),
            (None, False, "chatml"),
            (None, True, "chatml"),
            ("dsv32", False, None),
            ("dsv4", False, None),
            ("inkling", True, None),
            ("kimi_k3", True, None),
        )
        for spec, multimodal, template in paths:
            for variant in variants:
                with self.subTest(
                    spec=spec, multimodal=multimodal, template=template, variant=variant
                ):
                    serving = self.make_serving(spec, multimodal, template)
                    data = {"model": "x", "messages": messages, **variant}
                    self.assert_preparation_matches(
                        serving, ChatCompletionRequest(**data)
                    )

    def test_tool_constraints_and_media_order_match(self):
        for spec in (None, "kimi_k3"):
            for choice in (
                "auto",
                "required",
                {"type": "function", "function": {"name": "lookup"}},
            ):
                with self.subTest(spec=spec, choice=choice):
                    serving = self.make_serving(spec, multimodal=True)
                    if spec == "kimi_k3":
                        serving.tool_call_parser = "kimi_k3"
                    request = ChatCompletionRequest(
                        model="x",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "compare"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "https://example.com/one.png"
                                        },
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "https://example.com/two.png"
                                        },
                                    },
                                ],
                            }
                        ],
                        tools=[
                            {
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                        ],
                        tool_choice=choice,
                    )
                    self.assert_preparation_matches(serving, request)

    def test_responses_adapter_matches_existing_message_preparation(self):
        for with_tools in (False, True):
            with self.subTest(with_tools=with_tools):
                serving = self.make_serving(multimodal=True)
                serving.default_chat_template_kwargs = {"thinking": False}
                request = ResponsesRequest(
                    model="x",
                    input="hi",
                    reasoning={"effort": "high"},
                    store=False,
                    tools=[
                        {
                            "type": "function",
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        }
                    ]
                    if with_tools
                    else [],
                    tool_choice={"type": "function", "name": "lookup"}
                    if with_tools
                    else "auto",
                )
                original = request.model_dump()
                messages = serving._construct_input_messages(request)
                chat_input = from_responses_request(request, messages)
                prepared = make_processor(serving).prepare(chat_input)
                self.assertEqual(request.model_dump(), original)
                baseline_request = request.model_copy(deep=True)
                _, old = asyncio.run(
                    serving._make_request(
                        baseline_request, None, serving.tokenizer_manager.tokenizer
                    )
                )
                self.assertEqual(
                    prepared.prompt.to_generate_kwargs(),
                    old.prompt.to_generate_kwargs(),
                )
                self.assertEqual(
                    prepared.tool_call_constraint, old.tool_call_constraint
                )
                self.assertEqual(
                    prepared.chat_template_kwargs, old.chat_template_kwargs
                )

    def test_tokenize_reuses_precomputed_ids_without_selecting_generation_input(self):
        for template in (None, "chatml"):
            with self.subTest(template=template):
                serving = self.make_serving(multimodal=True, template=template)
                request = TokenizeRequest(messages=[{"role": "user", "content": "hi"}])
                prepared = make_processor(serving).prepare(
                    from_tokenize_request(request)
                )
                self.assertIsInstance(prepared.prompt, TextPrompt)
                tokenizer = serving.tokenizer_manager.tokenizer
                tokenizer.reset_mock()
                self.assertEqual(prepared.prompt.tokenize(tokenizer), [1, 2, 3])
                if template is None:
                    tokenizer.encode.assert_not_called()
                else:
                    tokenizer.encode.assert_called_once_with(
                        prepared.prompt.text, add_special_tokens=False
                    )
                self.assertEqual(
                    prepared.prompt.to_generate_kwargs(), {"text": prepared.prompt.text}
                )

    def test_tokenize_adapter_matches_legacy_protocol_normalization(self):
        messages = [{"role": "user", "content": "hi"}]
        variants = (
            {},
            {"reasoning_effort": "none"},
            {"reasoning": {"effort": "low"}},
            {"input_ids": [21, 22]},
            {"response_format": {"type": "json_object"}},
            {"tools": [{"type": "function", "function": {"name": "lookup"}}]},
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Use tools",
                        "tools": [{"type": "function", "function": {"name": "lookup"}}],
                    },
                    *messages,
                ]
            },
        )
        for spec in (None, "kimi_k3", "inkling"):
            for variant in variants:
                with self.subTest(spec=spec, variant=variant):
                    serving = self.make_serving(spec, multimodal=True)
                    endpoint = object.__new__(OpenAIServingTokenize)
                    endpoint.input_processor = make_processor(serving)
                    endpoint.tokenizer_manager = serving.tokenizer_manager
                    request = TokenizeRequest(**{"messages": messages, **variant})
                    original = request.model_dump()
                    tokenizer = serving.tokenizer_manager.tokenizer
                    tokenizer.reset_mock()
                    expected = (
                        make_processor(serving)
                        .prepare(
                            from_chat_request(request.to_chat_completion_request())
                        )
                        .prompt.tokenize(tokenizer)
                    )
                    old_calls = tokenizer.mock_calls[:]
                    tokenizer.reset_mock()
                    actual = endpoint._tokenize_chat_request(request)
                    self.assertEqual(actual, expected)
                    self.assertEqual(tokenizer.mock_calls, old_calls)
                    self.assertEqual(request.model_dump(), original)

    def test_reasoning_options_match_parser_requirements(self):
        for parser in ("k2_horizon", "hunyuan", "mistral", "inkling", "kimi_k3"):
            for effort in (None, "none", "high"):
                with self.subTest(parser=parser, effort=effort):
                    serving = self.make_serving()
                    serving.reasoning_parser = parser
                    serving.template_manager.reasoning_config = ReasoningToggleConfig(
                        toggle_param="thinking", default_enabled=True
                    )
                    if parser == "hunyuan":
                        serving.template_manager.reasoning_config = (
                            ReasoningToggleConfig(special_case="hunyuan_effort")
                        )
                    self.assert_preparation_matches(
                        serving,
                        ChatCompletionRequest(
                            model="x",
                            messages=[{"role": "user", "content": "hi"}],
                            reasoning_effort=effort,
                        ),
                    )

    def test_resolved_options_are_request_local_on_success_and_failure(self):
        serving = self.make_serving()
        serving.reasoning_parser = "deepseek-r1"
        serving.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        defaults = {"thinking": True, "nested": {"value": 1}}
        serving.default_chat_template_kwargs = defaults
        processor = make_processor(serving)
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"thinking": False},
        )
        before = request.model_dump()
        chat_input = from_chat_request(request)
        first = processor.prepare(chat_input)
        self.assertFalse(first.require_reasoning)
        self.assertEqual(request.model_dump(), before)
        self.assertEqual(chat_input.chat_template_kwargs, {"thinking": False})
        first.chat_template_kwargs["nested"]["value"] = 99
        self.assertEqual(defaults["nested"]["value"], 1)
        second = processor.prepare(ChatInput(messages=request.messages))
        self.assertTrue(second.require_reasoning)
        self.assertEqual(second.chat_template_kwargs["nested"]["value"], 1)
        effective = with_prepared_options(request, second)
        self.assertTrue(effective.chat_template_kwargs["thinking"])
        self.assertEqual(request.model_dump(), before)
        processor.renderer.render = Mock(side_effect=ValueError("render failed"))
        with self.assertRaisesRegex(ValueError, "render failed"):
            processor.prepare(chat_input)
        self.assertEqual(chat_input.chat_template_kwargs, {"thinking": False})
        self.assertEqual(defaults, {"thinking": True, "nested": {"value": 1}})

    def test_new_renderer_requires_no_serving_model_registration(self):
        config = ChatModelConfig(
            tokenizer=Mock(),
            template_manager=SimpleNamespace(reasoning_config=None),
            is_multimodal=True,
            chat_encoding_spec="unregistered",
        )
        renderer = Mock()
        renderer.render.return_value = RenderedPrompt(
            prompt=TokenPrompt([4, -1, 6]), image_data=["image"]
        )
        processor = ChatInputProcessor(config, renderer=renderer)
        result = processor.prepare(
            ChatInput(messages=[{"role": "user", "content": "hi"}])
        )
        self.assertEqual(result.prompt.to_generate_kwargs(), {"input_ids": [4, -1, 6]})
        self.assertEqual(result.image_data, ["image"])
        renderer.render.assert_called_once()

    def test_text_only_media_rejection_does_not_render_or_mutate_input(self):
        serving = self.make_serving()
        processor = make_processor(serving)
        processor.renderer.render = Mock()
        request = ChatInput(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "image.png"}}
                    ],
                }
            ]
        )
        original = request.model_dump()
        with self.assertRaisesRegex(ValueError, "only supports text"):
            processor.prepare(request)
        processor.renderer.render.assert_not_called()

        self.assertEqual(request.model_dump(), original)

    def test_stop_policy_preserves_legacy_payloads(self):
        for spec, template in ((None, None), (None, "chatml"), ("kimi_k3", None)):
            for stop in (None, "USER_STOP", ["USER_STOP"]):
                for ignore_eos in (False, True):
                    for input_ids in (None, [31, 32]):
                        with self.subTest(
                            spec=spec,
                            template=template,
                            stop=stop,
                            ignore_eos=ignore_eos,
                            input_ids=input_ids,
                        ):
                            serving = self.make_serving(spec, True, template)
                            if spec == "kimi_k3":
                                serving.tool_call_parser = "kimi_k3"
                            self.assert_preparation_matches(
                                serving,
                                ChatCompletionRequest(
                                    messages=[{"role": "user", "content": "hi"}],
                                    tools=[
                                        {
                                            "type": "function",
                                            "function": {"name": "lookup"},
                                        }
                                    ],
                                    stop=stop,
                                    ignore_eos=ignore_eos,
                                    input_ids=input_ids,
                                ),
                            )

    def test_template_stops_are_not_mutated_by_request_policy(self):
        config = ChatModelConfig(
            tokenizer=Mock(),
            template_manager=SimpleNamespace(reasoning_config=None),
            is_multimodal=True,
        )
        rendered = RenderedPrompt(
            prompt=TokenPrompt([4, -1, 6]), template_stop=["TEMPLATE_STOP"]
        )
        renderer = Mock()
        renderer.render.return_value = rendered
        processor = ChatInputProcessor(config, renderer)
        request = ChatInput(
            messages=[{"role": "user", "content": "hi"}], stop=["USER_STOP"]
        )
        first = processor.prepare(request)
        self.assertEqual(first.stop, ["TEMPLATE_STOP", "USER_STOP"])
        second = processor.prepare(request.model_copy(update={"ignore_eos": True}))
        self.assertEqual(second.stop, ["USER_STOP"])
        self.assertEqual(rendered.template_stop, ["TEMPLATE_STOP"])
        self.assertEqual(request.stop, ["USER_STOP"])

    def test_adapters_do_not_delegate_to_chat_endpoint_validation(self):
        methods = (
            "set_tool_choice_default",
            "normalize_reasoning_inputs",
            "set_json_schema",
            "validate_reasoning_effort_type",
        )
        with ExitStack() as patches:
            for method in methods:
                patches.enter_context(
                    patch.object(
                        ChatCompletionRequest,
                        method,
                        side_effect=AssertionError("Chat endpoint validation invoked"),
                    )
                )
            messages = [{"role": "user", "content": "hi"}]
            tokenize_input = from_tokenize_request(
                TokenizeRequest(messages=messages, reasoning_effort="none")
            )
            responses_input = from_responses_request(
                ResponsesRequest(input="hi", reasoning={"effort": "none"}), messages
            )
        self.assertEqual(
            tokenize_input.chat_template_kwargs, responses_input.chat_template_kwargs
        )
        self.assertFalse(responses_input.chat_template_kwargs["thinking"])

    def test_input_contract_imports_without_openai_protocol(self):
        code = """
import importlib.abc
import sys

class BlockOpenAIProtocol(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sglang.srt.entrypoints.openai.protocol':
            raise AssertionError('Shared input contract depends on the API protocol')

sys.meta_path.insert(0, BlockOpenAIProtocol())
from sglang.srt.entrypoints.chat_input.types import ChatInput

chat_input = ChatInput(messages=[{'role': 'user', 'content': 'hi'}])
assert chat_input.messages[0].content == 'hi'
"""
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_endpoints_share_the_custom_renderer_and_keep_token_input(self):
        fixture = self.make_serving(multimodal=True)
        manager = fixture.tokenizer_manager
        renderer = Mock()
        renderer.render.return_value = RenderedPrompt(
            prompt=TokenPrompt([4, -1, 6]),
            image_data=["image"],
            template_stop=["END"],
        )
        processor = ChatInputProcessor(make_processor(fixture).config, renderer)
        endpoints = [
            cls(manager, fixture.template_manager, input_processor=processor)
            for cls in (
                OpenAIServingChat,
                OpenAIServingResponses,
                OpenAIServingTokenize,
            )
        ]
        chat, responses, tokenize = endpoints
        self.assertTrue(
            all(endpoint.input_processor is processor for endpoint in endpoints)
        )
        captured = []

        async def generate(request, raw_request):
            captured.append(request)
            yield engine_chunk("done", finish=True)

        manager.generate_request = generate
        manager.tokenizer.reset_mock()
        messages = [{"role": "user", "content": "hi"}]
        chat_request = ChatCompletionRequest(messages=messages)
        original = chat_request.model_dump()
        adapted, _ = chat._convert_to_internal_request(chat_request)
        response = asyncio.run(
            responses.create_responses(
                ResponsesRequest(model="x", input="hi", store=False)
            )
        )
        tokens = tokenize._tokenize_chat_request(TokenizeRequest(messages=messages))
        self.assertFalse(hasattr(response, "body"), getattr(response, "body", None))
        self.assertEqual(response.status, "completed")
        self.assertEqual(tokens, [4, -1, 6])
        for request in (adapted, captured[0]):
            self.assertEqual(request.input_ids, tokens)
            self.assertIsNone(request.text)
            self.assertEqual(request.image_data, ["image"])
            self.assertEqual(request.sampling_params["stop"], ["END"])
        manager.tokenizer.encode.assert_not_called()
        self.assertEqual(chat_request.model_dump(), original)
        count_response = asyncio.run(
            AnthropicServing(chat).handle_count_tokens(
                AnthropicCountTokensRequest(model="x", messages=messages), None
            )
        )
        self.assertEqual(count_response.status_code, 200)
        self.assertEqual(json.loads(count_response.body)["input_tokens"], len(tokens))

    def test_harmony_keeps_native_rendering_for_full_and_streaming_responses(self):
        serving = self.make_serving()
        serving.use_harmony = True
        serving.input_processor.prepare = Mock(
            side_effect=AssertionError("Unexpected chat rendering")
        )
        serving._construct_input_messages_with_harmony = Mock(
            return_value=["native message"]
        )
        captured = []

        async def generate(request, raw_request):
            captured.append(request)
            yield engine_chunk("done", finish=True)

        async def finish(request, params, result_generator, *args, **kwargs):
            async for _ in result_generator:
                pass
            return "completed"

        async def stream(request, params, result_generator, *args, **kwargs):
            async for _ in result_generator:
                yield "completed"

        serving.tokenizer_manager.generate_request = generate
        serving.responses_full_generator = finish
        serving.responses_stream_generator = stream
        module = "sglang.srt.entrypoints.openai.serving_responses"
        with (
            patch(module + ".render_for_completion", return_value=[77, 88]) as render,
            patch(
                module + ".HarmonyContext", side_effect=lambda *args: SimpleContext()
            ),
            patch(
                module + ".StreamingHarmonyContext",
                side_effect=lambda *args: SimpleContext(),
            ),
        ):
            for streaming in (False, True):
                request = ResponsesRequest(
                    model="x", input="hi", store=False, stream=streaming
                )

                async def run():
                    response = await serving.create_responses(request)
                    return (
                        [chunk async for chunk in response] if streaming else response
                    )

                self.assertEqual(
                    asyncio.run(run()), ["completed"] if streaming else "completed"
                )
            self.assertEqual(render.call_count, 2)
            render.assert_called_with(["native message"])
        self.assertEqual(
            [request.input_ids for request in captured], [[77, 88], [77, 88]]
        )
        serving.input_processor.prepare.assert_not_called()

    def test_tokenize_keeps_validation_without_constructing_a_chat_handler(self):
        fixture = self.make_serving()
        processor = make_processor(fixture)
        tokenize = OpenAIServingTokenize(
            fixture.tokenizer_manager,
            fixture.template_manager,
            input_processor=processor,
        )
        processor.renderer.render = Mock()
        requests = (
            TokenizeRequest(messages=[]),
            TokenizeRequest(
                messages=[{"role": "user", "content": "hi"}], tool_choice="required"
            ),
            TokenizeRequest(
                messages=[{"role": "user", "content": "hi"}], return_sampling_mask=True
            ),
        )
        for request in requests:
            with (
                self.subTest(request=request.model_dump()),
                self.assertRaises(ValueError),
            ):
                tokenize._tokenize_chat_request(request)
        processor.renderer.render.assert_not_called()

        request = TokenizeRequest(
            messages=[{"role": "user", "content": "hi"}],
            input_ids=[31, 32],
            return_sampling_mask="false",
            max_tokens="5",
        )
        self.assertEqual(tokenize._tokenize_chat_request(request), [31, 32])


if __name__ == "__main__":
    unittest.main()
