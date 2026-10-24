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


def _serving(test, tokenizer, *, is_multimodal=False, **server_args):
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
    )
    template_manager = SimpleNamespace(
        chat_template_name=None,
        jinja_template_content_format=None,
        reasoning_config=None,
        force_reasoning=False,
        python_chat_template=None,
        jinja_template_may_reorder_tool_results=False,
    )
    return OpenAIServingChat(manager, template_manager)


class TestAdapterRegistry(CustomTestCase):
    def _context(self, spec, tokenizer=None, default_kwargs=None):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        return registry.ChatEncodingAdapterContext(
            chat_encoding_spec=spec,
            tokenizer=tokenizer,
            reasoning_parser=None,
            tool_call_parser=None,
            chat_hooks=registry.ChatOutputHooks(
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
        titled = self._prompt(chat, task="title").prompt_ids
        self.assertEqual(titled[-1], self._token("<｜title｜>"))


if __name__ == "__main__":
    unittest.main()
