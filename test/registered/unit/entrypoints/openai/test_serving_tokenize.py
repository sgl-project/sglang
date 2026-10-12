"""Unit tests for chat formatting options on /v1/tokenize.

A real word-level tokenizer and Jinja chat template run through the real TemplateManager,
OpenAIServingChat rendering and OpenAIServingTokenize handler; only the tokenizer manager is
a stand-in. Expected IDs come from literal token strings, not from the code under test.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import unittest
from unittest.mock import Mock

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai.protocol import TokenizeRequest, TokenizeResponse
from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORDS = ["[UNK]", "<bos>", "<user>", "<assistant>", "</s>", "call", "lookup", "Paris"]
WORDS += ["hi", "hello", "style", "formal", "casual"]
CHAT_TEMPLATE = (
    "{% if style is defined %}style {{ style }} {% endif %}"
    "{% for m in messages %}<{{ m.role }}> "
    "{% for c in m.tool_calls or [] %}"
    "call {{ c.function.name }} {{ c.function.arguments.city }} {% endfor %}"
    "{{ m.content }} </s> {% endfor %}"
    "{% if add_generation_prompt %}<assistant> {% endif %}"
)
HISTORY = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def _tokenizer():
    vocab = {word: i for i, word in enumerate(WORDS)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", vocab["<bos>"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", unk_token="[UNK]"
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def _ids(text):
    return [WORDS.index(word) for word in text.split()]


def _tokenize(serving, **body):
    request = TokenizeRequest.model_validate(body)
    response = asyncio.run(
        serving._handle_non_streaming_request(request, request, None)
    )
    if isinstance(response, TokenizeResponse):
        return 200, response.tokens
    return response.status_code, json.loads(response.body)["message"]


class TestTokenizeFormattingOptions(CustomTestCase):
    def _serving(self, default_kwargs=None, chat_template=None, multimodal=False):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", default_chat_template_kwargs=default_kwargs),
            role="tokenizer",
        )
        config = {
            "tool_call_parser": None,
            "reasoning_parser": None,
            "grammar_backend": "xgrammar",
            "enable_cache_report": False,
            "incremental_streaming_output": False,
        }
        tm = Mock()
        tm.tokenizer, tm.processor, tm.served_model_name = _tokenizer(), None, "test"
        tm.model_config = Mock(is_multimodal=multimodal, context_len=4096)
        tm.model_config.get_default_sampling_params.return_value = {}
        tm.model_config.hf_config = Mock(architectures=["LlamaForCausalLM"])
        tm.model_config.hf_config.to_dict.return_value = {}
        tm.config_value.side_effect = config.__getitem__
        tm.server_args = Mock(**config)
        templates = TemplateManager()
        templates.initialize_templates(
            tm, model_path="test-model", chat_template=chat_template
        )
        return OpenAIServingTokenize(tm, templates)

    def test_omitted_options_render_as_chat(self):
        # The final assistant turn becomes a user turn, the generation prompt
        # follows, and add_special_tokens does not apply to messages.
        self.assertEqual(
            _tokenize(self._serving(), messages=HISTORY, add_special_tokens=True),
            (200, _ids("<user> hi </s> <user> hello </s> <assistant>")),
        )

    def test_supplied_options_render_messages_as_given(self):
        expected = {
            (False, None): "<user> hi </s> <assistant> hello </s>",
            (True, None): "<user> hi </s> <assistant> hello </s> <assistant>",
            (False, True): "<bos> <user> hi </s> <assistant> hello </s>",
            (False, False): "<user> hi </s> <assistant> hello </s>",
        }
        # The multimodal path caches renders; the two generation-prompt cases
        # share every other cache-key input.
        for multimodal in (False, True):
            serving = self._serving(multimodal=multimodal)
            for (prompt, special), text in expected.items():
                body = {"messages": HISTORY, "add_generation_prompt": prompt}
                if special is not None:
                    body["add_special_tokens"] = special
                with self.subTest(
                    multimodal=multimodal, prompt=prompt, special=special
                ):
                    self.assertEqual(_tokenize(serving, **body), (200, _ids(text)))

    def test_tool_call_arguments_are_decoded_or_rejected(self):
        serving = self._serving()

        def history(arguments):
            call = {"name": "lookup", "arguments": arguments}
            return [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "type": "function", "function": call}],
                },
            ]

        self.assertEqual(
            _tokenize(
                serving,
                messages=history('{"city": "Paris"}'),
                add_generation_prompt=False,
            ),
            (200, _ids("<user> hi </s> <assistant> call lookup Paris </s>")),
        )
        status, message = _tokenize(
            serving, messages=history('"Paris"'), add_generation_prompt=False
        )
        self.assertEqual(status, 400)
        self.assertIn("JSON object", message)

    def test_request_template_kwargs_override_server_default(self):
        serving = self._serving(default_kwargs={"style": "formal"})
        messages = HISTORY[:1]
        self.assertEqual(
            _tokenize(serving, messages=messages, add_generation_prompt=False),
            (200, _ids("style formal <user> hi </s>")),
        )
        self.assertEqual(
            _tokenize(
                serving,
                messages=messages,
                add_generation_prompt=False,
                chat_template_kwargs={"style": "casual"},
            ),
            (200, _ids("style casual <user> hi </s>")),
        )

    def test_options_that_cannot_apply_are_rejected(self):
        # A conversation template cannot take the options; a continued final
        # turn needs the generation prompt that opens it.
        chatml = self._serving(chat_template="chatml")
        self.assertEqual(_tokenize(chatml, messages=HISTORY)[0], 200)
        status, message = _tokenize(
            chatml, messages=HISTORY, add_generation_prompt=False
        )
        self.assertEqual(status, 400)
        self.assertIn("Jinja chat template", message)
        status, message = _tokenize(
            self._serving(),
            messages=HISTORY,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        self.assertEqual(status, 400)
        self.assertIn("continue_final_message", message)


if __name__ == "__main__":
    unittest.main()
