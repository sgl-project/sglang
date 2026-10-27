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
    response = asyncio.run(
        serving._handle_non_streaming_request(request, request, None)
    )
    if isinstance(response, TokenizeResponse):
        return 200, response.tokens
    return response.status_code, json.loads(response.body)["message"]


class TestTokenizeFormattingOptions(CustomTestCase):
    def _serving(self, chat_template=None, multimodal=False):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        tm = Mock()
        tm.tokenizer, tm.processor = _tokenizer(), None
        tm.model_config = Mock(is_multimodal=multimodal, context_len=4096)
        tm.model_config.get_default_sampling_params.return_value = {}
        tm.model_config.hf_config = Mock(architectures=["LlamaForCausalLM"])
        tm.config_value.return_value = None
        templates = TemplateManager()
        templates.initialize_templates(
            tm, model_path="test-model", chat_template=chat_template
        )
        return OpenAIServingTokenize(tm, templates)

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


if __name__ == "__main__":
    unittest.main()
