"""Chat-template encoders must hand the engine a list of token ids.

``apply_chat_template(tokenize=True)`` returns a ``BatchEncoding`` on
transformers 5 unless ``return_dict=False`` is passed; used as the prompt,
that dict is its two keys rather than the tokens.
"""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.ollama.protocol import OllamaChatRequest, OllamaMessage
from sglang.srt.entrypoints.ollama.serving import OllamaServing
from sglang.srt.entrypoints.openai.chat_encoding import encode_simple_chat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)
MESSAGES = [{"role": "user", "content": "hello world"}]
# "user: hello world\nassistant:" under the vocabulary built below.
MESSAGES_IDS = [2, 4, 5, 6, 3, 4]


def make_chat_tokenizer() -> PreTrainedTokenizerFast:
    """A tiny word-level tokenizer with a chat template; nothing to download."""
    words = ["user", "assistant", ":", "hello", "world"]
    vocab = {"[UNK]": 0, "[PAD]": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    )
    hf_tokenizer.chat_template = CHAT_TEMPLATE
    return hf_tokenizer


class _FakeTokenizerManager:
    """Records the engine request and answers it with one canned chunk."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.served_model_name = "test-model"
        self.requests = []

    async def generate_request(self, gen_request, raw_request=None):
        self.requests.append(gen_request)
        yield {
            "text": "ok",
            "meta_info": {
                "prompt_tokens": len(MESSAGES_IDS),
                "completion_tokens": 1,
                "finish_reason": "stop",
            },
        }


class TestChatTemplateTokenIds(CustomTestCase):
    def setUp(self):
        self.tokenizer = make_chat_tokenizer()
        rendered = self.tokenizer.apply_chat_template(
            MESSAGES, tokenize=False, add_generation_prompt=True
        )
        assert self.tokenizer.encode(rendered, add_special_tokens=False) == MESSAGES_IDS

    def test_encode_simple_chat_returns_the_template_token_ids(self):
        ids = encode_simple_chat(tokenizer=self.tokenizer, spec=None, messages=MESSAGES)
        self.assertEqual(ids, MESSAGES_IDS)

    def test_ollama_chat_sends_the_template_token_ids(self):
        manager = _FakeTokenizerManager(self.tokenizer)
        request = OllamaChatRequest(
            model="test-model",
            messages=[OllamaMessage(role="user", content="hello world")],
            stream=False,
        )

        response = asyncio.run(
            OllamaServing(manager).handle_chat(request, raw_request=None)
        )

        self.assertEqual(response.message.content, "ok")
        (gen_request,) = manager.requests
        self.assertEqual(gen_request.input_ids, MESSAGES_IDS)


if __name__ == "__main__":
    unittest.main()
