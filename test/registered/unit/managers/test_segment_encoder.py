"""Piecewise chat-prompt encoding must give the ids of one full encode.

The equivalence holds only because a fast tokenizer cuts verbatim added tokens
out before it normalizes or pre-tokenizes; tokenizers that break that premise
must be turned away instead of silently producing different ids.
"""

import unittest

from tokenizers import (
    AddedToken,
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

from sglang.srt.managers import segment_encoder
from sglang.srt.managers.segment_encoder import SegmentCachedEncoder
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# "<x>" is a prefix of "<x>y"; listed first so only longest-match ordering
# picks the longer token.
ADDED = ["<x>", "<x>y", "<|user|>", "<|assistant|>", "<tool_call>"]
CORPUS = [
    "def foo(bar):\n    return bar + 1\n",
    "The quick brown fox jumps over the lazy dog. 12345 67890",
    "cr\u00e8me \u00fcber na\u00efve \ufb01ne \u4f60\u597d",
]
TEXTS = [
    "<|user|>\nhello<|assistant|><tool_call>{}",
    "<x>y<x>yy<x> <x>",
    "e<x>\u0301e\u0301 \ufb01<|user|>\u0301",
    "  <|user|>  \n\n<|assistant|<|assistant|>12345678 ",
    "<|user|><|user|><|user|>",
    "",
]


def _tokenizer(*, pre_tokenizer=None, lstrip=False, add_bos=False):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizer or pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    tok.train_from_iterator(
        CORPUS,
        trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    tok.add_special_tokens(
        [AddedToken(t, normalized=False, lstrip=lstrip) for t in ADDED]
    )
    if add_bos:
        bos = tok.token_to_id("<x>")
        tok.post_processor = processors.TemplateProcessing(
            single="<x> $A", special_tokens=[("<x>", bos)]
        )
    return PreTrainedTokenizerFast(tokenizer_object=tok)


class TestSegmentCachedEncoder(unittest.TestCase):
    def test_matches_full_encode_on_cold_warm_and_evicted_cache(self):
        tok = _tokenizer()
        encoder = SegmentCachedEncoder.build(tok)
        self.assertIsNotNone(encoder)
        old_limit = segment_encoder._MAX_CACHED_TOKENS
        segment_encoder._MAX_CACHED_TOKENS = 8
        try:
            for _ in range(2):
                for text in TEXTS:
                    with self.subTest(text=text):
                        self.assertEqual(encoder.encode(text), tok.encode(text))
            held = [len(ids) for ids in encoder._cache.values()]
            self.assertEqual(encoder._cached_tokens, sum(held))
            self.assertTrue(sum(held) <= 8 or len(held) == 1)
        finally:
            segment_encoder._MAX_CACHED_TOKENS = old_limit

    def test_padded_call_on_shared_tokenizer_does_not_leak_into_encode(self):
        """A processor's padding=True call must not pad later prompt pieces."""
        tok = _tokenizer()
        tok.pad_token = "<x>"
        encoder = SegmentCachedEncoder.build(tok)
        tok(["hi", "a much longer second text"], padding=True, truncation=True)
        for text in TEXTS:
            with self.subTest(text=text):
                self.assertEqual(encoder.encode(text), tok.encode(text))

    def test_tokenizers_that_break_the_split_are_rejected(self):
        cases = {
            "metaspace": _tokenizer(
                pre_tokenizer=pre_tokenizers.Metaspace(prepend_scheme="first")
            ),
            "lstrip": _tokenizer(lstrip=True),
            "bos": _tokenizer(add_bos=True),
        }
        for name, tok in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(SegmentCachedEncoder.build(tok))


if __name__ == "__main__":
    unittest.main()
