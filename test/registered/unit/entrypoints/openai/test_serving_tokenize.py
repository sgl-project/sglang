from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    TokenizeRequest,
    TokenizeResponse,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.entrypoints.openai.serving_tokenize import (
    OpenAIServingDetokenize,
    OpenAIServingTokenize,
)
from sglang.srt.parser.template_manager import TemplateManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORDS = ["[UNK]", "<bos>", "<user>", "<assistant>", "</s>", "call", "lookup", "Paris"]
WORDS += ["hi", "hello"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<{{ m.role }}> "
    "{% for c in m.tool_calls or [] %}"
    "call {{ c.function.name }} {{ c.function.arguments.city }} {% endfor %}"
    "{{ m.content }} </s> {% endfor %}"
    "{% if add_generation_prompt %}<assistant> {% endif %}"
)
CALL = {
    "id": "c1",
    "type": "function",
    "function": {"name": "lookup", "arguments": '{"city": "Paris"}'},
}
HISTORY = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello", "tool_calls": [CALL]},
]


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
    response = asyncio.run(serving.handle_request(request, None))
    if isinstance(response, TokenizeResponse):
        return 200, response.tokens
    return response.status_code, json.loads(response.body)["message"]


class TestTokenizeFormattingOptions(CustomTestCase):
    def _serving(
        self, chat_template=None, multimodal=False, jinja_template=CHAT_TEMPLATE
    ):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        tm = Mock()
        tm.tokenizer, tm.processor = _tokenizer(), None
        tm.tokenizer.chat_template = jinja_template
        tm.model_config = Mock(is_multimodal=multimodal, context_len=4096)
        tm.model_config.get_default_sampling_params.return_value = {}
        tm.model_config.hf_config = Mock(architectures=["LlamaForCausalLM"])
        tm.config_value.return_value = None
        templates = TemplateManager()
        templates.initialize_templates(
            tm, model_path="test-model", chat_template=chat_template
        )
        tm.request_logger.log_requests = False

        def tokenize(adapted, _raw_request):
            if adapted.input_ids is not None:
                return adapted.input_ids
            return tm.tokenizer(adapted.text)["input_ids"]

        tm.tokenize_request = AsyncMock(side_effect=tokenize)
        return OpenAIServingTokenize(
            tm, OpenAIServingChat(tm, templates), OpenAIServingCompletion(tm, templates)
        )

    def test_formatting_options(self):
        turn = "<user> hi </s> <assistant> call lookup Paris hello </s>"
        expected = {
            (None, True): "<user> hi </s> <user> hello </s> <assistant>",
            (False, None): turn,
            (True, None): f"{turn} <assistant>",
            (False, True): f"<bos> {turn}",
            (False, False): turn,
        }
        # Multimodal renders are cached; the prompt flag must separate their keys.
        serving = self._serving(multimodal=True)
        for (prompt, special), text in expected.items():
            body = {"messages": HISTORY}
            if prompt is not None:
                body["add_generation_prompt"] = prompt
            if special is not None:
                body["add_special_tokens"] = special
            with self.subTest(prompt=prompt, special=special):
                self.assertEqual(_tokenize(serving, **body), (200, _ids(text)))

    def test_options_that_cannot_apply_are_rejected(self):
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

    def test_dedicated_and_generic_routes_use_generation_conversion(self):
        serving = self._serving()
        for endpoint, request, expected in (
            (
                serving.chat_serving,
                ChatCompletionRequest(messages=HISTORY),
                _ids("<user> hi </s> <user> hello </s> <assistant>"),
            ),
            (
                serving.completion_serving,
                CompletionRequest(prompt=["hi", "hello"], stream=True),
                [_ids("<bos> hi"), _ids("<bos> hello")],
            ),
        ):
            with (
                self.subTest(request=type(request).__name__),
                patch.object(
                    endpoint,
                    "_convert_to_internal_request",
                    wraps=endpoint._convert_to_internal_request,
                ) as convert,
            ):
                generic = TokenizeRequest.model_validate(
                    request.model_dump(exclude_unset=True)
                )
                for body in (request, generic):
                    response = asyncio.run(serving.handle_request(body, None))
                    self.assertIsInstance(response, TokenizeResponse)
                    self.assertEqual(response.tokens, expected)
                    self.assertEqual(
                        response.count,
                        [2, 2]
                        if isinstance(request, CompletionRequest)
                        else len(expected),
                    )
                self.assertEqual(convert.call_count, 2)
                self.assertIsInstance(convert.call_args.args[0], type(request))

    def test_prompt_special_tokens_and_pre_tokenized_ids(self):
        serving = self._serving()
        for prompt, expected in (
            ("hi", _ids("hi")),
            (["hi", "hello"], [_ids("hi"), _ids("hello")]),
            ([3, 4], [3, 4]),
            ([[3], [4, 5]], [[3], [4, 5]]),
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    _tokenize(serving, prompt=prompt, add_special_tokens=False),
                    (200, expected),
                )

    def test_epd_chat_passes_media_to_tokenizer_manager(self):
        template = (
            "{% for m in messages %}{% for part in m.content %}"
            "{% if part.type == 'image' %}<image> {% else %}{{ part.text }}{% endif %}"
            "{% endfor %}{% endfor %}"
        )
        serving = self._serving(multimodal=True, jinja_template=template)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image.png"}},
                    {"type": "text", "text": "hi"},
                ],
            }
        ]
        tokens = [7, 7, 7, 8]
        tokenize = serving.tokenizer_manager.tokenize_request = AsyncMock(
            return_value=tokens
        )
        with get_context().override_server_args(language_only=True):
            for request in (
                ChatCompletionRequest(messages=messages),
                TokenizeRequest(messages=messages),
            ):
                response = asyncio.run(serving.handle_request(request, None))
                self.assertEqual(response.tokens, tokens)
                self.assertEqual(response.count, len(tokens))
                adapted, _ = tokenize.await_args.args
                self.assertEqual(adapted.image_data[0].url, "image.png")

    def test_invalid_requests_do_not_reach_tokenizer_manager(self):
        serving = self._serving()
        for request in (
            ChatCompletionRequest(messages=[]),
            CompletionRequest(prompt=[]),
        ):
            response = asyncio.run(serving.handle_request(request, None))
            self.assertEqual(response.status_code, 400)
        serving.tokenizer_manager.tokenize_request.assert_not_awaited()
        for body in ({}, {"prompt": "hi", "messages": HISTORY}):
            with self.assertRaisesRegex(ValueError, "Exactly one"):
                TokenizeRequest.model_validate(body)


class TestDetokenize(CustomTestCase):
    def test_out_of_vocab_ids_are_rejected(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        tm = Mock(tokenizer=_tokenizer(), model_config=Mock(vocab_size=len(WORDS)))
        serving = OpenAIServingDetokenize(tm)

        def detokenize(tokens):
            request = DetokenizeRequest(tokens=tokens)
            return asyncio.run(
                serving._handle_non_streaming_request(request, request, None)
            )

        self.assertEqual(detokenize(_ids("hi hello")).text, "hi hello")
        for tokens in ([len(WORDS)], [[8], [2**40]]):
            with self.subTest(tokens=tokens):
                self.assertEqual(detokenize(tokens).status_code, 400)


if __name__ == "__main__":
    unittest.main()
