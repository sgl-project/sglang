"""Image requests tokenize the rendered chat template, so a Mistral tokenizer must map the
control tokens it spells out back to ids instead of splitting them into characters.
"""

import unittest

from sglang.srt.utils.hf_transformers.mistral_utils import (
    patch_mistral_common_tokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

CONTROL_IDS = {
    "<s>": 1,
    "</s>": 2,
    "[INST]": 3,
    "[/INST]": 4,
    "[MODEL_SETTINGS]": 6,
    "[IMG]": 10,
    "[IMG_END]": 13,
    "[THINK]": 34,
    "[/THINK]": 35,
}


def _chars(text):
    return [1000 + ord(c) for c in text]


class _MistralCommonStub:
    """Stands in for MistralCommonBackend; the class name gates the patch.

    Plain text becomes one id per character and control tokens have no text form,
    as in the real tokenizer, whose special tokens also include unnamed filler slots.
    """

    chat_template = "x"
    all_special_tokens = [*CONTROL_IDS, "<SPECIAL_9>"]

    def _text_to_ids(self, text, add_special_tokens):
        return _chars(text)

    def convert_tokens_to_ids(self, value):
        return CONTROL_IDS[value]

    def apply_chat_template(self, messages, **kwargs):
        return []

    def add_special_tokens(self, *args, **kwargs):
        return 0

    def decode(self, *args, **kwargs):
        return ""

    def batch_decode(self, *args, **kwargs):
        return []


class TestControlTokenMarkers(CustomTestCase):
    def setUp(self):
        self.encode = patch_mistral_common_tokenizer(_MistralCommonStub())._text_to_ids

    def test_every_named_control_token_maps_to_its_id(self):
        text = "<s>[INST][IMG][IMG][IMG_END]hi[/INST][MODEL_SETTINGS]a</s>"
        expected = [1, 3, 10, 10, 13, *_chars("hi"), 4, 6, *_chars("a"), 2]
        self.assertEqual(self.encode(text, False), expected)

    def test_reasoning_markers_map_to_their_ids(self):
        text = "[THINK]a[/THINK]"
        self.assertEqual(self.encode(text, False), [34, *_chars("a"), 35])

    def test_names_that_are_not_control_tokens_stay_text(self):
        for text in ("<SPECIAL_9>", "[ARGS]"):
            self.assertEqual(self.encode(text, False), _chars(text))


if __name__ == "__main__":
    unittest.main()
