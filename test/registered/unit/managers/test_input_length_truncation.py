"""Unit test for auto-truncation in validate_input_length.

An over-long input that ends with the separator (e.g. a BERT [SEP]) must keep it
as its last token, as Hugging Face truncation=True does; encoder embeddings depend
on it. Inputs that do not end with it, and generation models (no separator to
preserve), keep the plain head slice."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.utils import validate_input_length
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CLS, SEP = 101, 102
MAX_LEN = 8


def _truncate(input_ids, preserve_last_token_id):
    req = SimpleNamespace(origin_input_ids=list(input_ids))
    error = validate_input_length(req, MAX_LEN, True, preserve_last_token_id)
    assert error is None
    return req.origin_input_ids


class TestInputLengthTruncation(unittest.TestCase):
    def test_trailing_separator_is_kept(self):
        input_ids = [CLS, *range(1, 11), SEP]
        self.assertEqual(_truncate(input_ids, SEP), [CLS, 1, 2, 3, 4, 5, 6, SEP])

    def test_input_without_trailing_separator_is_head_sliced(self):
        input_ids = [CLS, *range(1, 12)]
        self.assertEqual(_truncate(input_ids, SEP), input_ids[:MAX_LEN])

    def test_no_preserve_token_is_head_sliced(self):
        input_ids = [CLS, *range(1, 11), SEP]
        self.assertEqual(_truncate(input_ids, None), input_ids[:MAX_LEN])


if __name__ == "__main__":
    unittest.main()
