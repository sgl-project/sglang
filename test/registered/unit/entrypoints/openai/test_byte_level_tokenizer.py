import unittest

from sglang.srt.entrypoints.openai.utils import (
    _BYTE_LEVEL_TOKENIZERS,
    _is_byte_level_tokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Tokenizer:
    def __init__(self, pieces, added_ids=()):
        self.pieces = pieces
        self.added_tokens_decoder = {i: pieces[i] for i in added_ids}

    def get_vocab(self):
        return {piece: i for i, piece in enumerate(self.pieces)}

    def convert_ids_to_tokens(self, token_id):
        return self.pieces[token_id]


class TestIsByteLevelTokenizer(CustomTestCase):
    def setUp(self):
        _BYTE_LEVEL_TOKENIZERS.clear()

    def test_added_tokens_are_not_sampled(self):
        # DeepSeek's byte-level vocabulary has added tokens at both ends.
        pieces = [
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜▁pad▁｜>",
            "!",
            "Ġthe",
            "èĢĮæĪĲçļĦ",
            "<｜place▁holder▁no▁0｜>",
            "<｜place▁holder▁no▁1｜>",
        ]
        tokenizer = _Tokenizer(pieces, added_ids=(0, 1, 2, 6, 7))
        self.assertTrue(_is_byte_level_tokenizer(tokenizer))

    def test_sentencepiece_vocabulary_is_not_byte_level(self):
        pieces = ["<unk>", "<s>", "</s>", "▁the", "▁quick", "é", "▁brown", "▁fox"]
        tokenizer = _Tokenizer(pieces, added_ids=(0, 1, 2))
        self.assertFalse(_is_byte_level_tokenizer(tokenizer))


if __name__ == "__main__":
    unittest.main()
