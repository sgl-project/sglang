"""The deepseek-recipe chat encoding backend: the adapter registration, the
prompt path and the handler's thinking decision."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"


def _user(content="x"):
    return {"role": "user", "content": content}


def _tokenizer_file(test):
    """The V4.1 tokenizer from the Hub cache or the Hub; skip when neither answers."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
    except Exception as error:  # noqa: BLE001
        test.skipTest(f"the tokenizer is not available: {error}")


def _serving(test, tokenizer):
    """A chat handler on the recipe backend, configured like a V4.1 server."""
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    test.addCleanup(reset_context)
    args = ServerArgs(
        model_path="dummy",
        chat_encoding_backend="recipe",
        reasoning_parser="deepseek-v41",
        tool_call_parser="deepseekv41",
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
    """The recipe backend is the `recipe` builder of the dsv4 / dsv41 specs."""

    def _context(self, spec, backend, tokenizer=None, default_kwargs=None):
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
            backend=backend,
        )

    def test_builtin_backend_keeps_the_inline_deepseek_encoders(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        self.assertTrue(registry.has_chat_encoding_adapter("dsv41"))
        self.assertTrue(registry.has_chat_encoding_adapter("dsv4"))
        self.assertIsNone(
            registry.build_chat_encoding_adapter(self._context("dsv41", "builtin"))
        )

    def test_recipe_backend_is_deepseek_only(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        with self.assertRaises(ValueError):
            registry.build_chat_encoding_adapter(self._context("kimi_k3", "recipe"))

    def test_default_effort_must_be_a_recipe_effort(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        with self.assertRaisesRegex(ValueError, "reasoning_effort=60"):
            registry.build_chat_encoding_adapter(
                self._context(
                    "dsv41",
                    "recipe",
                    tokenizer=SimpleNamespace(backend_tokenizer=None),
                    default_kwargs={"reasoning_effort": 60},
                )
            )

    def test_recipe_backend_needs_the_fast_tokenizer(self):
        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry

        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            registry.build_chat_encoding_adapter(
                self._context("dsv41", "recipe", tokenizer=SimpleNamespace())
            )

    def test_recipe_adapter_encodes_and_overlays_sampling(self):
        from tokenizers import Tokenizer

        from sglang.srt.entrypoints.openai import chat_encoding_adapters as registry
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        tokenizer = Tokenizer.from_file(_tokenizer_file(self))
        adapter = registry.build_chat_encoding_adapter(
            self._context(
                "dsv41",
                "recipe",
                tokenizer=SimpleNamespace(backend_tokenizer=tokenizer),
            )
        )
        self.assertFalse(adapter.strict_tool_call_arguments)
        self.assertIsNone(adapter.output)
        # The client's model name is SGLang's business; recipe sees its own.
        request = ChatCompletionRequest(
            model="anything",
            messages=[{"role": "user", "content": ""}],
            thinking={"type": "disabled"},
        )
        self.assertEqual(
            adapter.encode_messages(messages=[], request=request, tools=None),
            [
                tokenizer.token_to_id(token)
                for token in (
                    "<｜begin▁of▁sentence｜>",
                    "<｜User｜>",
                    "<｜Assistant｜>",
                    "</think>",
                )
            ],
        )
        params = {"max_new_tokens": None}
        adapter.apply_sampling(
            request=request,
            sampling_params=params,
            prompt_tokens=4,
            context_length=100_000,
        )
        self.assertEqual(params["max_new_tokens"], 8192)
        # The handler's tool dumps carry None for unset fields; recipe must see the
        # tool as the client sent it.
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
        with_tools = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "disabled"},
        )
        self.assertEqual(
            adapter.encode_messages(messages=[], request=with_tools, tools=[dumped]),
            adapter.encode_messages(messages=[], request=with_tools, tools=[tool]),
        )


class TestHandlerThinking(CustomTestCase):
    """The handler takes its thinking decision from recipe, whose default is on,
    so the engine counts reasoning tokens and the parsers split the output."""

    def test_thinking_follows_recipe(self):
        from transformers import PreTrainedTokenizerFast

        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=_tokenizer_file(self))
        chat = _serving(self, tokenizer)
        think_start = tokenizer.convert_tokens_to_ids("<think>")
        think_end = tokenizer.convert_tokens_to_ids("</think>")
        output = "I think.</think>Hello"

        def respond(**fields):
            request = ChatCompletionRequest(messages=[_user("hi")], **fields)
            processed = chat._process_messages(request, False)
            ret = [
                {
                    "text": output,
                    "output_ids": tokenizer.encode(output, add_special_tokens=False)
                    + [1],
                    "index": 0,
                    "meta_info": {
                        "id": "chatcmpl-1",
                        "prompt_tokens": len(processed.prompt_ids),
                        "completion_tokens": 7,
                        "cached_tokens": 0,
                        "weight_version": "v",
                        "finish_reason": {"type": "stop", "matched": 1},
                        "output_token_logprobs": [],
                        "output_top_logprobs": None,
                    },
                }
            ]
            message = chat._build_chat_response(request, ret, 0).choices[0].message
            return (
                processed.prompt_ids[-1],
                processed.require_reasoning,
                message.reasoning_content,
                message.content,
            )

        # The API's default and its `thinking` field, which SGLang's rule never read.
        for fields in (
            {},
            {"thinking": {"type": "enabled"}},
            {"reasoning_effort": "high"},
        ):
            self.assertEqual(
                respond(**fields), (think_start, True, "I think.", "Hello"), fields
            )
        for fields in (
            {"thinking": {"type": "disabled"}},
            {"reasoning_effort": "none"},
        ):
            self.assertEqual(respond(**fields)[:2], (think_end, False), fields)


if __name__ == "__main__":
    unittest.main()
