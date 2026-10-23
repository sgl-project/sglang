"""DeepSeek-V4 / V4.1 chat encoding through deepseek-recipe: the adapter
registration and the prompt path under SGLang's request semantics."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def _user(content="x"):
    return {"role": "user", "content": content}


def _tokenizer_file():
    """The V4.1 tokenizer from the Hub cache or the Hub; skip when neither answers."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
    except Exception as error:  # noqa: BLE001
        raise unittest.SkipTest(f"the tokenizer is not available: {error}")


def _serving(test, tokenizer, **server_args):
    """A chat handler configured like a V4.1 server."""
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
            is_multimodal=False,
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
    """deepseek-recipe is the builder of the dsv4 / dsv41 specs."""

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

    def test_recipe_backend_needs_the_fast_tokenizer(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            registry.build_chat_encoding_adapter(
                self._context("dsv41", tokenizer=SimpleNamespace())
            )

    def test_tool_dumps_render_as_the_client_sent_them(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        adapter = registry.build_chat_encoding_adapter(
            self._context("dsv41", tokenizer=self._fast_tokenizer())
        )
        self.assertFalse(adapter.strict_tool_call_arguments)
        self.assertIsNone(adapter.output)
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
        request = ChatCompletionRequest(messages=[_user("hi")])
        self.assertEqual(
            adapter.encode_messages(
                messages=[], request=request, tools=[dumped], thinking=False
            ),
            adapter.encode_messages(
                messages=[], request=request, tools=[tool], thinking=False
            ),
        )


class TestHandlerRequestSemantics(CustomTestCase):
    """Thinking, effort and sampling follow SGLang's rules; recipe only renders."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from transformers import PreTrainedTokenizerFast

        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_file=_tokenizer_file())

    def _token(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def _prompt(self, chat, **fields):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        request = ChatCompletionRequest(messages=[_user("hi")], **fields)
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

    def test_request_reaches_the_engine_with_sglang_semantics(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        chat = _serving(self, self.tokenizer)
        request = ChatCompletionRequest(
            messages=[_user("hi")],
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
