"""Voxtral audio prompts splice the audio tokens into the tokenizer's id list."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.srt.multimodal.processors.voxtral import (
    AUDIO_TOKEN_ID,
    BEGIN_AUDIO_TOKEN_ID,
    INST_TOKEN_ID,
    VoxtralMultimodalProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BOS_ID, EOS_ID, INST_END_ID = 1, 2, 4
TRANSCRIBE_ID, THIS_ID = 5, 6
MISTRAL_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}[INST]{{ message['content'] }}[/INST]{% endfor %}"
)


def make_mistral_like_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level stand-in for the Mistral tokenizer with its control-token ids."""
    vocab = {
        "[UNK]": 0,
        "<s>": BOS_ID,
        "</s>": EOS_ID,
        "[INST]": INST_TOKEN_ID,
        "[/INST]": INST_END_ID,
        "transcribe": TRANSCRIBE_ID,
        "this": THIS_ID,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=["[INST]", "[/INST]"],
    )
    hf_tokenizer.chat_template = MISTRAL_TEMPLATE
    return hf_tokenizer


class TestVoxtralAudioPrompt(CustomTestCase):
    def test_audio_tokens_follow_the_last_inst_token(self):
        tokenizer = make_mistral_like_tokenizer()
        text_ids = [BOS_ID, INST_TOKEN_ID, TRANSCRIBE_ID, THIS_ID, INST_END_ID]
        self.assertEqual(
            tokenizer.encode(
                "<s>[INST]transcribe this[/INST]", add_special_tokens=False
            ),
            text_ids,
        )
        processor = object.__new__(VoxtralMultimodalProcessor)

        input_ids = processor._build_input_ids_with_audio(
            tokenizer, "<s>[INST]transcribe this[/INST]", audio_token_counts=[2]
        )

        self.assertEqual(
            input_ids,
            text_ids[:2]
            + [BEGIN_AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID]
            + text_ids[2:],
        )


if __name__ == "__main__":
    unittest.main()
